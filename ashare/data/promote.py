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

import duckdb


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

    _checkpoint(str(staging))

    backup = ""
    if market.exists():
        stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        backup = str(market) + f".bak.{stamp}"
        os.link(str(market), backup)                   # 零拷贝快照：同一 inode 多一个名字
        wal = pathlib.Path(str(market) + ".wal")        # 旧库的 WAL（正常不该有）不跟着走
        if wal.exists():
            wal.unlink()
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
    tmp = str(market) + ".rollback.tmp"
    os.link(backup_path, tmp)                          # 快照本身保留，用新名字挂同一 inode
    return promote(tmp, str(market))
