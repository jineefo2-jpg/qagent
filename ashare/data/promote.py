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
import sys as _sys
import os
import pathlib
import shutil

import duckdb



# 每张表的「可见性日期」列：某行写进来之后，它会从这一天起影响因子。
# 参考表（calendar/stock_basic/stock_status/industry_member）没有 _ingested_at，
# 覆盖不到（见 schema.sql 的 snapshot_log 注释与那条 ponytail 天花板）。
_NOTHING_AFFECTED = _dt.date(9999, 12, 31)   # 哨兵：本次发布不影响任何已有日期
_VISIBILITY_COL = {"daily_bar": "trade_date", "daily_basic": "trade_date",
                   "money_flow": "trade_date", "index_daily": "trade_date",
                   "financial_pit": "ann_date", "macro_indicator": "publish_date"}


# 参考表：没有 _ingested_at（每次 ingest 整表 INSERT OR REPLACE），只能与上一版比内容。
# 值 = 用来定位「这条记录从哪天起影响因子」的日期列。
_REF_TABLES = {"calendar": "trade_date", "stock_basic": "list_date",
               "stock_status": "start_date", "industry_member": "in_date"}


def _ref_tables_min_changed(c, old_market: str):
    """参考表相对上一版 market 的历史变更起点；没有上一版或比不了 → None（不贡献）。

    ★ 为什么非比不可：股票池（stock_basic/stock_status）与行业归属（industry_member）
      一变，当天整个横截面的 processed_value 都变（去极值/中性化/zscore 全是横截面统计量）。
      而这四张表每天被整表重写，`_ingested_at` 对它们恒等于今天，识别不出「历史行真的变了」。
    ★ 日常变更（新上市、新 ST 段、新行业段、日历往未来延）产生的都是**新日期**的行，
      diff 出来的最小日期落在未来，不影响历史因子 —— 正是我们要的。
      申万成分回溯调整 / list_date 勘误则会被正确压到被改的那一天。
    """
    # ATTACH 不接受参数占位符（DuckDB 限制），路径只能拼进语句 —— 它来自本模块调用方
    # （market 路径），不是外部输入；仍按 SQL 字面量转义单引号。
    c.execute(f"ATTACH '{old_market.replace(chr(39), chr(39) * 2)}' AS _prev (READ_ONLY)")
    try:
        md = None
        # ★ 附加库的名字在 table_CATALOG，不是 table_schema（后者是库内的 'main'）。
        #   写错的后果是这个集合恒空、四张表全被跳过，整道保护静默失效。
        have = {x[0] for x in c.execute("SELECT table_name FROM information_schema.tables "
                                        "WHERE table_catalog='_prev'").fetchall()}
        for t, col in _REF_TABLES.items():
            if t not in have:
                continue
            # 逐列比，且**排除 _ingested_at**：它每次 ingest 都刷新，比进去等于每行都"变了"。
            cols = [r[0] for r in c.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name=? AND column_name <> '_ingested_at' "
                "ORDER BY ordinal_position", [t]).fetchall()]
            if not cols:
                continue
            cl = ", ".join(cols)          # 列名来自 information_schema，非外部输入
            # ★ 每个 EXCEPT 都要括号：EXCEPT 与 UNION ALL 同优先级左结合，
            #   不加括号会被解析成 ((a EXCEPT b) UNION ALL b) EXCEPT a —— 结果恒空。
            row = c.execute(
                f"SELECT min({col}) FROM ("
                f"  (SELECT {cl} FROM {t} EXCEPT SELECT {cl} FROM _prev.{t})"
                f"  UNION ALL"
                f"  (SELECT {cl} FROM _prev.{t} EXCEPT SELECT {cl} FROM {t}))").fetchone()
            if row and row[0] is not None:
                md = row[0] if md is None else min(md, row[0])
        return md
    finally:
        c.execute("DETACH _prev")


def _log_snapshot(c, sid: str, old_market: str | None = None) -> None:
    """把本次发布记进 snapshot_log：`min_affected_date` = 本轮 ingest 写入的行里，
    最早的那个【可见性日期】。因子有效性判据靠它把「尾部追加」和「历史被改」分开
    （见 schema.sql 的表注释）。拿不到 ingest 起始时间戳（老库、或直接 promote 一个
    手工造的库）→ 写 NULL，读侧保守当成「全部失效」。"""
    r = c.execute("SELECT value FROM _meta WHERE key='ingest_started_at'").fetchone()
    started = r[0] if r else None
    # ★ 「这次没影响任何数据」与「判定不了」必须分开：前者写哨兵 9999-12-31（读侧照常
    #   放行历史因子 —— 比如只改了 schema 的一次维护性发布），后者写 NULL（保守失效）。
    #   合成一个 NULL 的话，一次无害的维护发布会把 3300 万行历史因子全判死。
    md = _NOTHING_AFFECTED if started else None
    if started:
        have = {x[0] for x in c.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        for t, col in _VISIBILITY_COL.items():
            if t not in have:
                continue
            row = c.execute(f"SELECT min({col}) FROM {t} WHERE _ingested_at >= ?", [started]).fetchone()
            if row and row[0] is not None:
                md = min(md, row[0]) if md is not None else row[0]
    # ★ md 已是 None（没有 ingest 起始戳 = 判定不了带时间戳的那批表）时不再细化：
    #   参考表只说得清四张小表，拿它去补一个"看起来精确"的日期会【低估】影响范围。
    #   判定不了就一路 NULL 到底，读侧保守全失效。
    if md is not None and old_market:       # 参考表：与上一版比内容（它们没有 _ingested_at）
        try:
            ref = _ref_tables_min_changed(c, old_market)
        except Exception:                   # noqa: BLE001 — 上一版不可读/结构不同：判定不了
            # ★ 失败方向必须是保守：比不了 = 不知道参考表有没有被改 = 整个 min_affected
            #   置 NULL（读侧全失效）。从前这里吞成「没变」，一个 ATTACH 语法错误就能
            #   静默关掉整道保护（2026-08-28 自查发现）。
            md = None
        else:
            if ref is not None:
                md = ref if md is None else min(md, ref)
    # ★ 不变量：数据指纹变了（sid ≠ 台账最后一行）却算出"什么都没影响"，说明筛选器
    #   一行都没匹配上（时钟/时区偏移、只改了参考表、手工造的 staging）。这时候宣布
    #   哨兵会让**所有历史快照对所有日期永久有效** —— 一个从不被检查的自相矛盾。
    #   落回 NULL（保守全失效）并出声。
    last = c.execute("SELECT snapshot_id FROM snapshot_log ORDER BY promoted_at DESC LIMIT 1").fetchone()
    if md == _NOTHING_AFFECTED and last and last[0] != sid:
        print(f"[promote] ⚠ 数据指纹已变（{last[0]} → {sid}）却测不出受影响日期："
              f"min_affected_date 记 NULL（保守：该快照之前的因子全部失效）", file=_sys.stderr)
        md = None
    # ON CONFLICT DO NOTHING：台账是 append-only。指纹相同 ⇒ 数据相同 ⇒ 原来那行本来就对，
    # 没有重算的理由；而改写 promoted_at 会让它跳到排序末尾，缩短它前面每一行的"之后"列表
    # （判据整篇是按"它之后的每一次发布"讲的）。
    c.execute("INSERT INTO snapshot_log (snapshot_id, promoted_at, min_affected_date) "
              "VALUES (?, now(), ?) ON CONFLICT (snapshot_id) DO NOTHING", [sid, md])


def _stamp_snapshot(path: str, old_market: str | None = None) -> str:
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
        _log_snapshot(c, sid, old_market)
        # 与 full_end 同处理：本轮 ingest 到此为止。不清的话这个戳会随 os.replace 进正式库、
        # 再被 copyfile 带回下一个 staging —— 于是「没有 ingest 的手工 promote」会拿上一轮的
        # 戳算出一个像模像样的 min_affected，而不是保守的 NULL。
        c.execute("DELETE FROM _meta WHERE key = 'ingest_started_at'")
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

    _stamp_snapshot(str(staging), str(market) if market.exists() else None)          # 指纹写进库自身，promote 之后仍可离线查
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
