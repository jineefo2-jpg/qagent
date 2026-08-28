"""影子文件原子替换（架构文档 §10.2）。

写者永远写 staging 文件；校验通过后 CHECKPOINT → 硬链接旧文件为零拷贝快照 → os.replace 到正式路径。
一个机制同时解决：DuckDB 跨进程"一写多读"限制、备份、秒级回滚。

三条必须遵守的细节：
  1. promote 前必须对 staging 做 CHECKPOINT 并关闭连接，否则 WAL 残留、替换后的库读不到最后一批写入
  2. staging 与正式路径必须在同一文件系统（os.replace 才是原子的、os.link 才可用）
  3. 读者进程靠 inode 变化自动重连（query._conn 每次 stat 路径）；写者进程替换前不得持有正式路径的连接
"""
from __future__ import annotations
import datetime as _dt
import os
import pathlib
import shutil

import duckdb


def _stamp_snapshot(path: str) -> str:
    """把即将上线的库的数据指纹写进它自己的 _meta.snapshot_id，并返回。
    备份用这个指纹命名 —— 否则 docs/oos-runs.md 里记的 data_snapshot_id 无法对应到磁盘上的哪份 .bak。
    用自己的写连接算（不经 query 的只读单例：同进程混用读写连接 DuckDB 会拒绝）。"""
    from .query import _compute_snapshot
    c = duckdb.connect(path)
    try:
        sid = _compute_snapshot(c)
        c.execute("INSERT INTO _meta (key, value) VALUES ('snapshot_id', ?) "
                  "ON CONFLICT (key) DO UPDATE SET value = excluded.value", [sid])
        # ★ 顺手清掉全量回补的续跑标记：`full_end` 的语义是「这个 staging 里有一次
        #   尚未发布的全量回补」，而 promote 正是「发布」——它的生命到此为止。
        #   不清的后果不是小事：promote 用 os.replace 把 staging **变成**正式库，
        #   标记就永久留在 market 里；之后每次 daily 把 market 拷回 staging，
        #   run_daily 的守卫都会看到它并拒绝 = 全量成功之后每日增量永远跑不起来
        #   （2026-08-28 实测）。清在 replace 之前，成为正式库的那个文件从来没带过它。
        #   注意不能改在 run_full 收尾清：一次 run_full 正常返回 ≠ 整个回补完成
        #   （跨夜续跑），那样会把 test_run_full_refuses_drifting_end_on_resume 守的
        #   「第二晚 --end 漂移」保护一起弄没。
        c.execute("DELETE FROM _meta WHERE key = 'full_end'")
    finally:
        c.close()
    return sid


def _snapshot_of(path: str) -> str:
    """库自己记的指纹；promote 之前建立的老库没有这个键 → 现算（数据都在，没有算不出来的道理）。"""
    from .query import _compute_snapshot
    c = duckdb.connect(path, read_only=True)
    try:
        r = c.execute("SELECT value FROM _meta WHERE key='snapshot_id'").fetchone()
        return r[0] if r else _compute_snapshot(c)
    finally:
        c.close()


def _checkpoint(path: str) -> None:
    c = duckdb.connect(path)
    try:
        c.execute("CHECKPOINT")
    finally:
        c.close()
    wal = pathlib.Path(path + ".wal")
    if wal.exists():                                   # CHECKPOINT + close 后不该还有 WAL
        raise RuntimeError(f"CHECKPOINT 后仍残留 WAL: {wal}")


def _prune_backups(market_path: str, keep: int) -> None:
    p = pathlib.Path(market_path)
    baks = sorted(p.parent.glob(p.name + ".bak.*"), key=lambda x: (x.stat().st_mtime, x.name))
    for old in (baks[:-keep] if keep > 0 else baks):
        old.unlink()


def promote(staging_path: str, market_path: str, *, keep: int = 3) -> str:
    """staging → market 原子替换。返回快照备份路径（首次 promote 无旧文件时返回 ""）。
    旧 market 用 os.link 做零拷贝快照 `<market>.bak.<UTC时间戳>`，只留最近 keep 份。"""
    staging = pathlib.Path(staging_path)
    market = pathlib.Path(market_path)
    if not staging.exists():
        raise FileNotFoundError(f"staging 不存在: {staging_path}")
    market.parent.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != market.parent.stat().st_dev:
        raise RuntimeError("staging 与 market 必须在同一文件系统（os.replace / os.link 的前提）")

    _stamp_snapshot(str(staging))          # 指纹写进库自身，promote 之后仍可离线查
    _checkpoint(str(staging))

    backup = ""
    if market.exists():
        # 用旧库自己的 snapshot_id 命名，回滚时能直接对上 oos-runs.md 记的 data_snapshot_id
        backup = str(market) + f".bak.{_snapshot_of(str(market))}"
        if pathlib.Path(backup).exists():   # 同一份数据重复 promote → 快照名相同，无需再存一份
            backup = ""
        wal = pathlib.Path(str(market) + ".wal")
        if wal.exists():                                # 正式库有 WAL = 有人原地写过正式路径，违反"只写 staging"
            raise RuntimeError(f"正式库残留 WAL: {wal} —— 有进程原地写了 market，先排查再 promote")
        if backup:
            os.link(str(market), backup)               # 零拷贝快照：同一 inode 多一个名字
    os.replace(str(staging), str(market))              # 原子：读者要么看到旧 inode，要么看到新 inode
    _prune_backups(str(market), keep)
    return backup


def rollback(market_path: str, backup_path: str | None = None) -> str:
    """回滚到某份快照（默认最新一份）。当前正式文件先做一份快照再被替换。"""
    market = pathlib.Path(market_path)
    if backup_path is None:
        baks = sorted(market.parent.glob(market.name + ".bak.*"), key=lambda x: (x.stat().st_mtime, x.name))
        if not baks:
            raise FileNotFoundError("没有可回滚的快照")
        backup_path = str(baks[-1])
    if not pathlib.Path(backup_path).exists():
        raise FileNotFoundError(f"快照不存在: {backup_path}")
    # ★ 用 copy 而非 hardlink：promote 会 CHECKPOINT（以读写方式打开），
    #   hardlink 是同一个 inode，会就地改动你正想保住的那份快照。
    tmp = str(market) + ".rollback.tmp"
    shutil.copyfile(backup_path, tmp)
    return promote(tmp, str(market))
