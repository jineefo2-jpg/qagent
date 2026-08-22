"""回测产物落盘 + 样本外台账（D7）。

★ 为什么不写 `backtest_run` 表（2026-08-22，Task 13 落地时的实际约束）
  架构 §4.3 与 `derived_schema.sql` 都为回测运行留了 `backtest_run` 表，写它要走
  `ashare/data/derived_store.py`（L1：只有 `ashare/data/**` 能 `import duckdb`）——
  而那个模块**目前没有** `write_backtest_run` / `read_backtest_run`。本任务不许改
  第四个文件，所以这一版落盘走文件，`backtest_run` 表暂时是空的。
  缺口记在这里而不是悄悄绕过：D7 的台账仍然完整（`docs/oos-runs.md` 是本模块自动写的，
  两个指纹一个不少），少的是「按 run_id 在库里查一次历史运行」这件事。
  补法是给 `derived_store` 加一对 DataFrame 进出的函数，本模块再转调 —— 与
  `factors/store.py` 同一个形状。

★ 为什么是 pickle 而不是 parquet
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
import pathlib
import re

import pandas as pd

from .types import BacktestResult

# 相对路径，与 `_db.DEFAULT_*_PATH` 同惯例（跟随 CWD，测试里 monkeypatch 掉）
RUNS_DIR = pathlib.Path("data/backtest_runs")
OOS_LOG_PATH = pathlib.Path("docs/oos-runs.md")

# 样本内 2010-01-01 – 2019-12-31（§8 闸 1）。end 超过它就触到了样本外。
OOS_CUTOFF = _dt.date(2019, 12, 31)

# 闸 1–5 都在 `guards.py`，本期未实现 —— 台账里必须写明「这一行没有任何闸背书」。
_UNRUN_GATES = "闸1 样本外/闸2 walk-forward/闸3 shuffle/闸4 成本敏感/闸5 参数高原"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PICKLE_FORMAT = 1


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


def save(result: BacktestResult, run_id: str | None = None) -> pathlib.Path:
    """把整个 `BacktestResult` 落盘，返回文件路径。`run_id=None` 用 `make_run_id`。"""
    rid = run_id or make_run_id(result)
    p = _path(rid)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 存字段 dict 而不是对象本身：dataclass 加字段时旧文件仍读得回来（缺的走默认值），
    # 而 pickle 一个对象是把类的当时布局也钉进去。
    payload = {f.name: getattr(result, f.name) for f in _dc.fields(result)}
    payload["_format"] = _PICKLE_FORMAT
    pd.to_pickle(payload, str(p))
    return p


def load(run_id: str) -> BacktestResult:
    """按 `run_id` 读回。文件不在就 `FileNotFoundError` —— 不返回空结果。"""
    p = _path(run_id)
    if not p.exists():
        raise FileNotFoundError(f"没有 run_id={run_id!r} 的回测产物：{p}")
    payload = dict(pd.read_pickle(str(p)))
    payload.pop("_format", None)
    return BacktestResult(**payload)


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
