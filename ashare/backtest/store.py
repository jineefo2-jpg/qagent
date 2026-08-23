"""回测产物落盘 + 样本外台账（D7）。

★ 落盘分两半，两半都是承重件（架构 §4.3 裁决 ⑤，2026-08-21）
  · **标量与两个 D7 指纹**进 `derived.duckdb` 的 `backtest_run` 表，经
    `ashare/data/derived_store.py`（L1：只有 `ashare/data/**` 能 `import duckdb`）。
    本模块只做转调，与 `factors/store.py` 同一个形状：L1 那层收发 DataFrame 与基础
    类型，L3 这层负责把 `BacktestResult` 序列化成一行、再把一行装回来。
    schema 里躺着一张没有写入方的表，比没有这张表更糟 —— 「按 run_id 查历史运行」
    看起来可用，实际永远查不到。
  · **DataFrame 与自由形态的 dict**（equity / positions / trades / blocked /
    ic / layers / attribution / config / metrics / warnings）进同名的 sidecar 文件。
  `load` 从两边各取一半拼回来：**任何一半漏掉一个字段，往返用例立刻红**，
  而不是安静地拿默认值补上（`ic` / `layers` / `attribution` / `warnings` 都有默认值，
  这正是「往返丢字段」最容易溜过去的地方）。

★ 为什么 sidecar 是 pickle 而不是 parquet
  架构 §4.3 写的是「derived.duckdb + parquet」，但本机没有 pyarrow / fastparquet，
  而 `BacktestResult` 里有 MultiIndex 的 `positions`、`date` 类型的索引与一个自由
  形态的 `metrics` dict。CSV 往返会把日期变字符串、把 MultiIndex 拍平 ——
  「save/load 往返一致」这条验收就成了空话。
  ponytail: pickle，装上 pyarrow 之后换 parquet（只读本机自己生成的文件，
  不要 load 外来的 .pkl —— pickle 会执行代码）。

★ 台账为什么由代码写
  U6 裁决：人工往 markdown 里补一行必然漏记，而漏记就是 D7 失效 —— 「样本外只跑一次」
  这道闸只有在**每一次**样本外运行都留痕时才成立。所以 `run_backtest` 收尾自动调
  `append_oos_run`。两类运行**不记**，各有理由：
    · `end <= 2019-12-31` 压根没碰样本外；
    · `shuffle_seed is not None` 是闸 3 的置换对照（200 次），它们是零假设的样本、
      不是策略的样本外运行，逐条记会把真正那几行埋掉。
  同指纹第二次出现【照记不误】，另外在备注里写明「重复」—— 台账的价值正在于让污染
  看得见，因为难看就不记等于把闸拆了。
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import pathlib
import re

import pandas as pd

from ..data import derived_store
from .types import BacktestResult

# 相对路径，与 `_db.DEFAULT_*_PATH` 同惯例（跟随 CWD，测试里 monkeypatch 掉）
RUNS_DIR = pathlib.Path("data/backtest_runs")
OOS_LOG_PATH = pathlib.Path("docs/oos-runs.md")

# 样本内 2010-01-01 – 2019-12-31（§8 闸 1）。end 超过它就触到了样本外。
OOS_CUTOFF = _dt.date(2019, 12, 31)

# 闸 1–5 都在 `guards.py`，本期未实现 —— 台账里必须写明「这一行没有任何闸背书」。
_UNRUN_GATES = "闸1 样本外/闸2 walk-forward/闸3 shuffle/闸4 成本敏感/闸5 参数高原"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PICKLE_FORMAT = 2

# 这几个字段住在 `backtest_run` 表里，不再进 sidecar —— 一个字段只有一个家。
# 两处都存的话，改了一处就有两个互相矛盾的真相，而 `load` 只会读到其中一个。
_ROW_FIELDS = ("param_hash", "data_snapshot_id", "engine_version", "started_at", "elapsed_sec")


def make_run_id(result: BacktestResult) -> str:
    """`{param_hash}_{data_snapshot_id}_{started_at:%Y%m%dT%H%M%S}`。

    两个指纹都在名字里是有意的（D7 缺一不可）：光看文件名就能判断「这次和上次
    是不是同一个实验、同一批数据」，不必先把结果读出来。
    """
    return f"{result.param_hash}_{result.data_snapshot_id}_{result.started_at:%Y%m%dT%H%M%S}"


def _path(run_id: str) -> pathlib.Path:
    # run_id 会经 REST / Agent 工具层传进来（`load` 是查询接口），拼进路径前必须验：
    # `../../etc/passwd` 在这里不是理论风险，是一次目录穿越。
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"run_id {run_id!r} 只能是字母/数字/点/下划线/连字符 —— "
                         f"它要拼进文件路径，含分隔符就是一次目录穿越")
    return pathlib.Path(RUNS_DIR) / f"{run_id}.pkl"


def _json(obj) -> str:
    """`config_json` / `metrics_json` 用的宽松序列化（这两列是给人查的，不进任何计算）。

    ponytail: `default=str` + 默认的 `allow_nan=True`。metrics 里 NaN 是常态
    （σ=0 时的 Sharpe），`allow_nan=False` 会让 save 直接炸；代价是这两列是
    Python 方言的 JSON（`NaN` 字面量），严格解析器读不了。真要发出去再换。
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


def save(result: BacktestResult, run_id: str | None = None) -> pathlib.Path:
    """落盘：标量进 `backtest_run` 表，帧与自由 dict 进 sidecar。返回 sidecar 路径。

    `run_id=None` 用 `make_run_id`。同一个 run_id 重存是整行替换（幂等）。
    """
    rid = run_id or make_run_id(result)
    p = _path(rid)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 存字段 dict 而不是对象本身：dataclass 加字段时旧文件仍读得回来（缺的走默认值），
    # 而 pickle 一个对象是把类的当时布局也钉进去。
    # 用 `_dc.fields` 全量减去 `_ROW_FIELDS`，不手写白名单：以后新增字段默认进 sidecar
    # （安全方向 —— 漏进表里只是查不到，漏进任何一边都会让往返丢字段）。
    payload = {f.name: getattr(result, f.name)
               for f in _dc.fields(result) if f.name not in _ROW_FIELDS}
    payload["_format"] = _PICKLE_FORMAT
    pd.to_pickle(payload, str(p))
    derived_store.write_backtest_run({
        "run_id": rid,
        "param_hash": result.param_hash,
        "data_snapshot_id": result.data_snapshot_id,
        "engine_version": result.engine_version,
        "started_at": result.started_at,
        "elapsed_sec": float(result.elapsed_sec),
        "config_json": _json(_dc.asdict(result.config)),
        "metrics_json": _json(result.metrics),
        # 与 `append_oos_run` 同一个判据、同一个常量：两处写着不同的「样本外」定义，
        # 台账与库表就会各说各话。
        "is_oos": bool(result.config.end > OOS_CUTOFF),
    })
    return p


def load(run_id: str) -> BacktestResult:
    """按 `run_id` 读回（表里一行 + sidecar 一份）。缺任一半都 `FileNotFoundError`。

    不返回空结果，也不拿默认值补齐缺的那一半：`ic` / `layers` / `attribution` /
    `warnings` 都有默认值，「往返丢字段」会静默通过。
    """
    p = _path(run_id)                       # 先验 run_id（目录穿越），再碰库
    row = derived_store.read_backtest_run(run_id)
    if row is None or not p.exists():
        raise FileNotFoundError(
            f"没有 run_id={run_id!r} 的回测产物（backtest_run 行={row is not None}，"
            f"sidecar={p.exists()}：{p}）")
    payload = dict(pd.read_pickle(str(p)))
    payload.pop("_format", None)
    return BacktestResult(**payload, **{k: row[k] for k in _ROW_FIELDS})


def append_oos_run(result: BacktestResult) -> "tuple[bool, list[str]]":
    """触及样本外时往 `docs/oos-runs.md` 追加一行。返回 `(是否追加, warnings)`。

    warnings 通道不是装饰（global-constraints ★）：本函数唯一的降级是「这次不记」，
    而不记正是 D7 失效的样子，必须让调用方（引擎）把理由汇进 `BacktestResult.warnings`。
    """
    cfg = result.config
    if cfg.end <= OOS_CUTOFF:
        return False, [f"回测终点 {cfg.end} 未超过样本内边界 {OOS_CUTOFF}，不记样本外台账"]
    if cfg.shuffle_seed is not None:
        return False, [f"shuffle_seed={cfg.shuffle_seed} 是闸 3 的置换对照（零假设样本），"
                       f"不记样本外台账 —— 200 次逐条记会把真正的样本外运行埋掉"]

    p = pathlib.Path(OOS_LOG_PATH)
    text = p.read_text(encoding="utf-8") if p.exists() else _EMPTY_LEDGER
    warns: list = []
    note = f"未跑：{_UNRUN_GATES}"
    # 「同一 (param_hash, data_snapshot_id) 只该有一次」（derived_schema.sql）。
    # 第二次照记，但两处都要喊出来：台账的用处就是让污染看得见。
    if result.param_hash in text and result.data_snapshot_id in text:
        warns.append(f"⚠ D7 重复指纹：param_hash={result.param_hash} + "
                     f"data_snapshot_id={result.data_snapshot_id} 已在样本外台账里出现过 —— "
                     f"再跑一次等于把样本外污染成样本内")
        note = "⚠ 重复指纹（D7 污染） · " + note

    sharpe = result.metrics.get("sharpe")
    row = " | ".join([
        _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "+".join(n for n, _ in cfg.factors)[:48] or "—",
        result.param_hash,
        result.data_snapshot_id,
        result.engine_version,
        f"{cfg.start}~{cfg.end}",
        "—",                                        # 样本内 Sharpe 由闸 1 成对写入
        "—" if sharpe is None or pd.isna(sharpe) else f"{float(sharpe):.3f}",
        "未跑",
        note,
    ])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip("\n") + "\n| " + row + " |\n", encoding="utf-8")
    return True, warns


_EMPTY_LEDGER = """# 样本外运行台账（D7）

> 每次运行由 `run_backtest` **自动追加**一行（不靠人工）。

| 运行时间 (UTC) | strategy_version | param_hash | data_snapshot_id | engine_version | 区间 | Sharpe(IS) | Sharpe(OOS) | 五闸 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
"""
