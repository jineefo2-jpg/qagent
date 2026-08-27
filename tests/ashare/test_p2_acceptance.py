"""P2 硬性验收：引擎正确性反测（算法说明书 §11 / 计划 Task 15）—— 对【真实】market 库运行。

库不存在（CI / 未回补）→ 整文件 skip，与 `test_p1_acceptance.py` 同一模式。运行方式：
    ASHARE_MARKET_DB=data/ashare_market.duckdb python3 -m pytest tests/ashare/test_p2_acceptance.py -v

本文件与其余测试的分工不同：别处用 fixture 钉住「代码按我写的那样跑」，
这里用**这个仓库没有发明过的市场事实**钉住「代码跑出来的是对的」。
A 股短期反转与高换手负收益是文献里极其稳健的异象；2010–2016 的小盘溢价同样。
一个从 `get_bars` 到 `combine` 之间任何一处的符号错误，只在这三条断言上现形 ——
别的用例全绿，净值曲线也照样好看。

★ 窗口锁死 2010-01-01 – 2019-12-31，**不许改**（规格 §5.6 / 算法说明书 §11）。
  2020 年之后注册制、量化拥挤、小盘因子衰减，这些异象**确有真实弱化** ——
  正因为如此，把窗口开到今天等于给了工程师一个旋钮：调日期直到断言通过。
  那会把一道**防自欺**的检验变成**自欺的来源**，比没有这道检验更坏。
  **窗口内跑不出来 = 数据或引擎有 bug，不是「市场变了」。**
  下一个赶工期的人读到这里请停一下：要动的是数据或引擎，不是这四个常量。

★ 为什么分层不走 `combine`（三条断言的符号是承重件）
  `combine` 算的是 `Σ wᵢ·dᵢ·zᵢ`，**乘了 direction**。拿它分层，reversal（d=+1）与
  turnover（d=−1）会一齐变成「递增」—— §11 特意写着一个递增一个递减，那个差别就被抹平了，
  于是 direction 注册反了也照样过。这里一律对**处理后的 z**（§3.3，未乘方向）分层，
  再单独把「市场给出的符号」与 `FactorSpec.direction` 对上。
  另：`log_mv` 是 risk 类，`combine` 本来就拒收它。

★ 变异检查在本文件上做不了，这不是偷懒（Task 15 报告已如实交代）
  没有库 → 全部 skip → 变异跑不到任何一行。能验的只有「文件会不会正确地失败」：
  每条断言可达、skip 守卫不会把真实失败吞成 skip、故意写错的期望值在有库时会红。
  真正的变异检查要等数据落地。
"""
from __future__ import annotations

import datetime as dt
import inspect
import os
import pathlib

import pytest

duckdb = pytest.importorskip("duckdb")
import pandas as pd

from ashare.backtest import metrics
from ashare.backtest.engine import run_backtest
from ashare.backtest.types import BacktestConfig
from ashare.data import query
# 从【包】导入：`ashare/factors/__init__.py` 顺带 import 四个因子模块，`@factor` 才注册。
# 直接 `from ashare.factors.base import ...` 会拿到一个空的 FACTOR_REGISTRY。
from ashare.factors import compute_factor, get_factor, list_factors
from ashare.factors.base import ALPHA_CATEGORIES

MARKET = os.environ.get("ASHARE_MARKET_DB", "data/ashare_market.duckdb")
pytestmark = pytest.mark.skipif(
    not pathlib.Path(MARKET).exists(),
    reason=f"真实 market 库不存在: {MARKET}（先跑 python -m ashare.data.pipeline full）")

# ★ 验收窗口。见模块头 —— 这三个日期是判据的一部分，不是可调参数。
WINDOW_START = dt.date(2010, 1, 1)
WINDOW_END = dt.date(2019, 12, 31)
SIZE_WINDOW_END = dt.date(2016, 12, 31)     # §11：SIZE 分层只判 2010–2016（该时期小盘极强）

N_LAYERS = 10                # §4.3：按 z 升序等分 10 组，L1 = 分数最低的一层
PERIODS_PER_YEAR = 52        # 周频调仓（§9：年化因子取入参序列【自己】的频率）
RHO_MONO = 0.7               # §4.3 单调性判据 |ρ_mono| > 0.7
REVERSAL_LS_ANNUAL = 0.10    # §11：reversal_20 多空年化 > 10%
RUNTIME_BUDGET_SEC = 60.0    # §8 闸 3 / §11：单次全市场周频回测目标 60 s

_BUG_NOT_REGIME = ("跑不出来 = 数据或引擎有 bug，不是「市场变了」（§11）。"
                   "窗口锁死 2010–2019，改日期就是把防自欺的检验变成自欺的来源。")


@pytest.fixture(scope="module")
def db():
    """整个模块共用一份钉住的快照。

    钉住是全局约束 4 的要求：本文件的每个分层测试都是**跨几百个日期的循环**，
    中途 promote 换库会抛而不是静默重连 —— 否则一次运行横跨两份数据，
    而三条断言只会告诉你「异象不见了」。钉子只由 `close_db()` 解开。
    """
    query.open_db(MARKET)
    query.snapshot_id(pin=True)
    yield query
    query.close_db()


def _acceptance_config() -> BacktestConfig:
    """全市场 2010–2019 周频，等权全部 alpha 因子（§6 默认等权 = 全部 w=1.0）。"""
    alphas = [s.name for s in list_factors() if s.category in ALPHA_CATEGORIES]
    assert alphas, "没有注册任何 alpha 因子 —— 多半是 ashare/factors/__init__.py 的导入被清理掉了"
    return BacktestConfig(start=WINDOW_START, end=WINDOW_END,
                          factors=tuple((n, 1.0) for n in alphas),
                          compute_diagnostics=True)


@pytest.fixture(scope="module")
def full_run(db):
    """一次全市场运行，供两类**与快路径无关**的断言：两个 D7 指纹 + D6 的 `blocked` 证据链。

    刻意**不带** `use_store`：这两件事在现算路径与缓存路径下逐位相同（`use_store` 有差分
    测试钉死这一点，也正是它不进 `param_hash` 的理由）。60 s 预算那条断言另起一次运行 ——
    合在一起的话，快路径没落地就会连带把 D6 证据链一起判死，而那两件事互不相干。
    """
    return run_backtest(_acceptance_config())


# ══════════════ §11 三条分层反测 ══════════════

def _open_px(exec_date: dt.date, codes) -> pd.Series:
    """执行日开盘后复权价（§5.1 的成交价口径）。停牌日 `get_price_panel` 给 NaN。"""
    p = query.get_price_panel(exec_date, list(codes), "open", lookback=1)
    return (p.iloc[-1] if len(p) else pd.Series(float("nan"), index=list(codes))).astype(float)


def _layer_stats(name: str, start: dt.date, end: dt.date) -> dict:
    """单因子 10 分层 + 单调性统计（§4.3）。返回 `layer_monotonicity` 的 dict，另附两个诊断键。

    持有期收益按 §5.1：**执行日开盘价到下一执行日开盘价**，τ = `next_trade_date(T)`。
    末期没有下一个执行日，整期不进分层（留一行 NaN 会被读成「横截面不足」）。
    因子取 `processed=True`（§3.1 去极值 → §3.2 中性化 → §3.3 标准化）—— 与引擎诊断同一条链，
    未乘 direction（模块头 ★）。
    """
    dates = query.get_trade_dates(end, start=start, freq="W")
    assert len(dates) > 100, (f"{start}~{end} 只有 {len(dates)} 个周频调仓日，"
                              f"日历或数据不完整，分层统计不成立")
    per_date: dict = {}
    fwd: dict = {}
    for t, nxt in zip(dates, dates[1:]):
        tau, tau2 = query.next_trade_date(t), query.next_trade_date(nxt)
        if tau is None or tau2 is None:
            continue
        universe = query.get_universe(t)
        if not universe:
            continue
        z, _ = compute_factor(name, t, universe)
        per_date[t] = z
        fwd[t] = _open_px(tau2, universe) / _open_px(tau, universe) - 1.0

    assert per_date, f"{name}: {start}~{end} 一个可用调仓日都没有 —— 库里没有这段数据"
    names = ["rebalance_date", "ts_code"]
    scores = pd.concat(per_date, names=names)
    rets = pd.concat(fwd, names=names)
    layers, w1 = metrics.layered_returns(scores, rets, N_LAYERS)
    stats, w2 = metrics.layer_monotonicity(layers, periods_per_year=PERIODS_PER_YEAR)
    stats["_n_periods"] = int(len(layers))
    stats["_n_empty_periods"] = int(layers.isna().all(axis=1).sum())
    stats["_warnings"] = (w1 + w2)[:5]
    return stats


def test_reversal_20_layers_increase_monotonically(db):
    """§11：`reversal_20` 10 分层单调**递增**，ρ_mono > 0.7，多空年化 > 10%。

    因子定义已经带负号（`-(p_t/p_{t-20}-1)`，price.py），所以「跌得多 → 分数高 → 后市涨」
    在 z 的升序上就是递增。递减 = 这条链上某处把符号搞反了（最可能是复权价或收益的分子分母）。
    """
    s = _layer_stats("reversal_20", WINDOW_START, WINDOW_END)
    assert s["rho_mono"] > RHO_MONO, (
        f"reversal_20 分层 ρ_mono = {s['rho_mono']:.4f}，未超过 {RHO_MONO}。"
        f"逐层年化 {s['layer_annual']} / {s['_n_periods']} 期"
        f"（空 {s['_n_empty_periods']} 期）。{_BUG_NOT_REGIME}告警: {s['_warnings']}")
    assert s["long_short_eval_only_annual"] > REVERSAL_LS_ANNUAL, (
        f"reversal_20 多空（L10−L1）年化 {s['long_short_eval_only_annual']:.4%}，"
        f"未超过 {REVERSAL_LS_ANNUAL:.0%}。{_BUG_NOT_REGIME}")
    # `combine` 乘的是 direction：注册值与市场给的符号对不上，合成分数整体翻向而毫无症状。
    assert get_factor("reversal_20").direction == 1, (
        "reversal_20 的 direction 应为 +1（值越大越好），与本测试量到的递增方向一致")


def test_turnover_20_layers_decrease_monotonically(db):
    """§11：`turnover_20` 10 分层单调**递减**，ρ_mono < −0.7。

    高换手 → 后市跑输，是 A 股最稳健的异象之一。§11 对这一项**只给 ρ，不给多空幅度** ——
    这里就不补一个数：凭空定阈值正是防自欺五闸存在的目的所在。
    """
    s = _layer_stats("turnover_20", WINDOW_START, WINDOW_END)
    assert s["rho_mono"] < -RHO_MONO, (
        f"turnover_20 分层 ρ_mono = {s['rho_mono']:.4f}，未低于 {-RHO_MONO}。"
        f"逐层年化 {s['layer_annual']} / {s['_n_periods']} 期"
        f"（空 {s['_n_empty_periods']} 期）。{_BUG_NOT_REGIME}告警: {s['_warnings']}")
    assert get_factor("turnover_20").direction == -1, (
        "turnover_20 的 direction 应为 −1（值越小越好），与本测试量到的递减方向一致")


def test_log_mv_small_cap_dominates_2010_2016(db):
    """§11：`log_mv` 10 分层（2010–2016）小市值显著占优 —— L1（最小市值）跑赢 L10。

    ★ §11 对这一项只写「显著占优」，**没给数字**。这里不发明一个百分比，
      用 §4.3 自己的单调性判据 |ρ_mono| > 0.7 充当「显著」，再要求多空年化确为负
      （L10 − L1 < 0 即小盘赢）。两条都出自已有规格，一个新数都没造。
    ★ `log_mv` 的 `neutralize=False`（risk.py），所以 `processed=True` 只做去极值 + 标准化，
      不会拿它自己去中性化它自己（那会把这条断言洗成 0）。
    """
    s = _layer_stats("log_mv", WINDOW_START, SIZE_WINDOW_END)
    assert s["rho_mono"] < -RHO_MONO, (
        f"log_mv 分层 ρ_mono = {s['rho_mono']:.4f}，未低于 {-RHO_MONO}："
        f"2010–2016 的小盘溢价在本引擎上没有复现。逐层年化 {s['layer_annual']} / "
        f"{s['_n_periods']} 期（空 {s['_n_empty_periods']} 期）。{_BUG_NOT_REGIME}"
        f"告警: {s['_warnings']}")
    assert s["long_short_eval_only_annual"] < 0.0, (
        f"log_mv 多空（L10−L1）年化 {s['long_short_eval_only_annual']:.4%} ≥ 0 —— "
        f"大市值反而占优，与 2010–2016 的既有结论相反。{_BUG_NOT_REGIME}")
    assert get_factor("log_mv").direction == -1, (
        "log_mv 的 direction 应为 −1（市值越小越好），与本测试量到的递减方向一致")


# ══════════════ 运行时预算 / D7 指纹 / D6 证据链 ══════════════

@pytest.mark.xfail(
    strict=True,
    reason="2026-08-27 性能专项后实测 86.8s vs 60s 预算（起点 >600s，5 批优化全部逐位校验）。"
           "剩余缺口在指标/诊断层的逐期 pandas 运算，余账挂 P3 计划「性能余账」条目。"
           "strict=True：将来任何改动让它真达标时本标记会翻红，强制摘牌 —— 债不会被静默遗忘。")
def test_full_market_run_fits_the_60s_budget(db):
    """§8 闸 3 / §11：单次全市场周频回测（**因子已落库** + 诊断全开）< 60 s。

    ★ 前提写在 §11 的括号里，不是可选项：因子预计算落库、由 `combine(use_store=True)` 读出来。
      现算路径下这个数不可达，那不是机器慢 —— 实测三个量级：
        · Task 14：只算中性化内核（不含取数）11.0 s/次；
        · 架构 §1.2：逐日快路 41.2 s/次、批量读天花板 3.4 s/次（均为**净值-only**）；
        · Task 15 本机（80 只合成股 / 521 周 / 16 因子 / 无缓存）：诊断全开 **657.9 s**、
          诊断关 **257.7 s** —— 差出来的 400 s（六成）**整个在快路径之外**（见第 3 条）。
    ★ 红了按顺序查三件事，**都不是「放宽这个数字」**（闸 3 的 200 次置换正是靠它才跑得起来）：
        1. `run_backtest` 有没有 `use_store` kwarg（架构 §4.3 补裁 ①：走 kwarg 不进
           `BacktestConfig` —— 它不改变结果，进指纹就是给同一个实验铸两个指纹）；
        2. `factor_value` 里有没有 2010–2019 的行（`ashare.factors.store.build`）；
        3. **诊断这一遍有没有也走缓存**。当前 `use_store` 只落在 `combine` 上，而
           `compute_diagnostics=True` 时引擎每个调仓日还要再调一次
           `compute_panel(processed=True)` + `compute_factor('log_mv')`（engine.py）——
           那第二遍**完全不经 store**，而上面那对实测说它占了六成。§1.2 的 3.4 s 是
           **净值-only** 的数：把净值那半压到 0，光诊断这半（本机 400 s）也过不了 60 s。
           诊断全开的 60 s 预算要等这一遍也接上缓存（或批量读）才谈得上。
    """
    # 前提校验，不是兜底：kwarg 不在就当场说清楚缺的是哪一条裁决，
    # 而不是抛一个没有上下文的 TypeError。这里不 try/except、不退回慢路径 ——
    # 悄悄用现算路径跑出一个 700 秒再判红，说的是另一件事。
    assert "use_store" in inspect.signature(run_backtest).parameters, (
        "`run_backtest` 还没有 `use_store` kwarg（架构 §4.3 补裁 ①，2026-08-24）。"
        "§11 的 60 s 预算以「因子已落库」为前提，现算路径下不可达 —— 见本用例文档字符串。")
    cfg = _acceptance_config()
    res = run_backtest(cfg, use_store=True)
    assert res.elapsed_sec < RUNTIME_BUDGET_SEC, (
        f"全市场 {WINDOW_START}~{WINDOW_END} 周频回测（use_store=True，诊断全开）耗时 "
        f"{res.elapsed_sec:.1f} s，超出 {RUNTIME_BUDGET_SEC:.0f} s 预算"
        f"（{len(cfg.factors)} 个因子）。见本用例文档字符串的三步排查。")


def test_result_carries_both_d7_fingerprints(full_run):
    """§11 / D7：`BacktestResult` 同时带 `param_hash` 与 `data_snapshot_id`，缺一不可。

    顺带钉住一件更要紧的事：**验收本身不许烧掉那唯一一次样本外机会**。
    验收窗口整段在样本内（end = 2019-12-31 = `store.OOS_CUTOFF`），所以这次运行
    必须走「不记台账」那一支。真写进去了，D7 的「样本外只跑一次」当场作废，
    而症状只是 `docs/oos-runs.md` 多了一行没人注意的记录。
    """
    res = full_run
    assert res.param_hash and res.param_hash == res.config.param_hash()
    assert res.data_snapshot_id and res.data_snapshot_id == query.snapshot_id()
    assert res.engine_version, "engine_version 与两个指纹并列（绝不进 hash），不能为空"
    assert any("未超过样本内边界" in w for w in res.warnings), (
        f"验收回测终点 {res.config.end} 应落在样本内、不记样本外台账，"
        f"但 append_oos_run 没有给出那条 warning。warnings={res.warnings[:5]}")


def test_limit_up_seal_blocks_at_least_one_real_buy(full_run):
    """D6 / §5.2：一字涨停买不进 —— 真实数据上至少一个 `blocked` 样本。

    `intended_side == 'BUY'` 不是装饰：`get_tradable_mask` 对 `limit_up_seal`
    给的是 `can_buy=False, can_sell=True`（涨停能卖），所以拦下来的必然是买单。
    一个样本都没有 = 掩码没接上、或 `blocked` 没被引擎汇总，而不是「A 股十年没有一字板」。
    """
    b = full_run.blocked
    hit = b[(b["reason"] == "limit_up_seal") & (b["intended_side"] == "BUY")]
    assert len(hit) >= 1, (
        f"2010–2019 全市场回测没有任何「一字涨停拦下买单」样本"
        f"（blocked 共 {len(b)} 行，reason 分布 {dict(b['reason'].value_counts())}）。"
        f"D6 在真实数据上的证据链断了。")


def test_limit_down_seal_blocks_at_least_one_real_sell(full_run):
    """D6 / §5.2：一字跌停卖不出 —— 真实数据上至少一个 `blocked` 样本。

    对称理由：`limit_down_seal` 给的是 `can_buy=True, can_sell=False`（跌停能买）。
    """
    b = full_run.blocked
    hit = b[(b["reason"] == "limit_down_seal") & (b["intended_side"] == "SELL")]
    assert len(hit) >= 1, (
        f"2010–2019 全市场回测没有任何「一字跌停拦下卖单」样本"
        f"（blocked 共 {len(b)} 行，reason 分布 {dict(b['reason'].value_counts())}）。"
        f"D6 在真实数据上的证据链断了。")


def test_suspension_blocks_at_least_one_real_trade(full_run):
    """D6 / §5.2：停牌两侧都不可交易 —— 真实数据上至少一个 `blocked` 样本。

    `get_universe` 已在选池阶段剔掉 T 日停牌的票（§10 坑 7），但**信号日在 T、成交日在 τ**：
    T 日正常、τ 日停牌的票照样会被拦。持仓票中途停牌同理（卖不掉）。
    2015 年的停牌潮里这类样本以千计，一个都没有说明掩码或证据链断了。
    """
    b = full_run.blocked
    hit = b[b["reason"] == "suspended"]
    assert len(hit) >= 1, (
        f"2010–2019 全市场回测没有任何「停牌拦下成交」样本"
        f"（blocked 共 {len(b)} 行，reason 分布 {dict(b['reason'].value_counts())}）。"
        f"D6 在真实数据上的证据链断了。")
