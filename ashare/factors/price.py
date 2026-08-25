"""量价因子（算法说明书 §2.1）—— 六个，只吃后复权收盘价 / 成交额 / 换手率。

本文件产出的是【原始因子值】。去极值、行业+市值中性化、zscore 全部在
`pipeline.process` 里做（Task 7 的 compute_factor 负责串起来），这里一个都不做 ——
提前 zscore 会让后面的中性化在已经标准化过的量上再回归一次，两次线性变换的复合
看起来毫无异常，但残差的量纲已经不是原来那个了。

★ 三件让因子"看着对、其实错"的事，本文件各有一处应对：

  1 停牌日在 `get_price_panel` 里是 NaN 且【不 ffill】（D9 的占位行在输出层被掩掉）。
    不填就是一天停牌毁掉整个窗口 —— p_t 或 p_{t-20} 只要有一个是 NaN，比值就是 NaN。
    所以每个因子内部先 ffill；但 ffill 填出来的不是新信息，因此再加一道
    覆盖率闸门：窗口内【原始】非空天数不足 60% 的股票直接置 NaN。
    拿 20 天里 3 天真实价算出来的反转是噪声，"不知道"比"猜一个"诚实。
    ffill 出来的价格能算【区间】收益（两端点是真实价即可），但算不了【逐日】收益：
    停牌日 r=0、复牌日 r 是整段累计 —— 逐日口径的因子（Amihud / max_ret）走
    `_daily_returns` 把这两天一起剔掉。turnover 走另一条路：0 直接当没有数据。

  2 收益口径。波动率与 Amihud 用【对数】收益，反转 / 动量 / max_ret 用【简单】收益 ——
    这是各自文献的口径。混用不会报错，只会静默改数（zigzag 路径上两者能差 20%）。

  3 momentum_120_20 的区间是 [t-120, t-20]，不是"120 日收益减 20 日收益"。
    跳过最近一个月是这个因子的全部意义：不跳就跟 20 日反转在同一段区间上正面相撞。
"""
from __future__ import annotations
import datetime as _dt
import math
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ashare.data import query
from .base import factor

DateLike = Union[str, _dt.date]

_MIN_COVERAGE = 0.60        # 窗口内原始非空天数占比下限，低于此值该股该日置 NaN
_BUFFER = 10                # 窗口之外多取的天数：窗口左端停牌时，ffill 的前值在窗口外


def _closes(as_of_date: DateLike, universe: Sequence[str], span: int, skip: int = 0
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """取窗口 `[t-span, t-skip]` 的后复权收盘价，返回 `(ffill 后的窗口面板, 真实成交掩码, 有效掩码)`。

    `real` 是窗口内【ffill 之前】的 `notna()`，两个用途：
      1 覆盖率闸门按它算 —— ffill 后再数一律是 100%，那道闸门就没了。
      2 逐日收益类因子（Amihud / max_ret）用它剔掉 ffill 造出来的假收益：
        停牌日的 `r` 是恒零，复牌日的 `r` 是【整段停牌期的累计收益】。
        `r.where(real & real.shift(1))` 一次盖掉这两种。

    历史长度连一个窗口都不够时返回全 NaN 面板（而不是抛）：回测头几天必然如此，
    这些股票本来就该是 NaN。
    """
    codes = list(universe)
    need = span - skip + 1
    raw = query.get_price_panel(as_of_date, codes, "close", span + _BUFFER)
    if len(raw) < span + 1:
        blank = pd.DataFrame(np.nan, index=pd.RangeIndex(need), columns=codes)
        return blank, blank.notna(), pd.Series(False, index=codes)
    lo, hi = len(raw) - span - 1, len(raw) - skip
    real = raw.iloc[lo:hi].notna()
    ok = real.sum() >= math.ceil(_MIN_COVERAGE * need)
    return raw.ffill().iloc[lo:hi], real, ok


def _daily_returns(px: pd.DataFrame, real: pd.DataFrame, log: bool) -> pd.DataFrame:
    """逐日收益，且【只保留两端都真实成交的那一天】。

    ffill 把停牌日填成前值，于是停牌日 `r=0`、复牌日 `r` 是跨越整段停牌的累计收益。
    这两个数都不是"某一天的收益"：前者会被 max_ret 当成"最好的一天"（全窗口下跌时
    0 就是最大值），后者在 Amihud 里拿 k+1 天的分子除 1 天的成交额（k 天停牌约 √k 倍虚高，
    direction=+1 → 系统性超配爱停牌的票，且它落在分布【体内】，MAD 去极值截不掉）。
    """
    # 简单收益不用 pct_change：pandas 2 的 fill_method 默认值在变
    r = (np.log(px).diff() if log else px.div(px.shift(1)) - 1.0).iloc[1:]
    return r.where((real & real.shift(1, fill_value=False)).iloc[1:])


@factor(name="reversal_20", direction=1, category="price", lookback_days=20 + _BUFFER, window=20)
def reversal_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`-(p_t / p_{t-window} - 1)`。负号在定义里：过去跌得多的分数高，所以 direction=+1。"""
    px, _, ok = _closes(as_of_date, universe, window)
    return (-(px.iloc[-1] / px.iloc[0] - 1)).where(ok).rename("reversal_20")


@factor(name="momentum_120_20", direction=1, category="price", lookback_days=120 + _BUFFER,
        window=120, skip=20)
def momentum_120_20(as_of_date: DateLike, universe: Sequence[str], *,
                    window: int = 120, skip: int = 20) -> pd.Series:
    """`p_{t-skip} / p_{t-window} - 1`，区间 `[t-120, t-20]`。

    最近 `skip` 天完全不进公式 —— 连覆盖率闸门也只看 `[t-120, t-20]`：
    整月停牌不该让一个本来就不看那一段的因子作废。
    """
    px, _, ok = _closes(as_of_date, universe, window, skip=skip)
    return (px.iloc[-1] / px.iloc[0] - 1).where(ok).rename("momentum_120_20")


@factor(name="volatility_60", direction=-1, category="price", lookback_days=60 + _BUFFER, window=60)
def volatility_60(as_of_date: DateLike, universe: Sequence[str], *, window: int = 60) -> pd.Series:
    """`window` 个对数收益的【样本】标准差（分母 window-1，规格 §2.1 写的就是 1/59）。

    ★ 这里【不】用 `_daily_returns` 的掩码，与 Amihud / max_ret 相反：k 天停牌留下的
      「k 个 0 + 1 个累计收益」平方和恰是 E=(k+1)σ²，分母不变 —— 估计量仍无偏，只是方差变大。
      掩掉反而是另一个同样无偏、但样本更少的估计量。停牌下的噪声该怎么治（收紧
      min_coverage 还是别的）留给 Task 12 用真实数据量 IC 代价后再定，本期只把现值钉住。
    """
    px, _, ok = _closes(as_of_date, universe, window)
    r = np.log(px).diff().iloc[1:]
    return r.std(ddof=1).where(ok).rename("volatility_60")


@factor(name="turnover_20", direction=-1, category="price", lookback_days=20, window=20)
def turnover_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`window` 日平均自由流通换手率（`turnover_rate_f`）。A 股高换手的负向收益极显著。

    ★ 恰好为 0 的换手率一律当【没有数据】，而不是"换手很低"：
      低换手在这个因子里是【好分数】（direction=-1），补 0 就是给停牌股发买入信号。
      源无行 / NULL 的那半 `mean` 的 skipna 本来就管；真正咬人的是 D9 那半 ——
      这家源会给出 `vol=0` 的停牌"行"（`ingest.py` 只为 `daily_bar` 归一了这件事，
      `ingest_daily_basic` 是把 vendor 帧原样 upsert，`validate.py` 也没有换手率检查）。
      源里写 0 时 `notna()` 是 20/20，覆盖率闸门根本不会响，均值却按 20 天摊薄
      （20 天里 8 天停牌，真实换手 5.0 会算成 3.0）。
      闸门本身也必须建在过滤【之后】，否则闸放行、均值失真，两个错凑成一个"正常"的低分。
      判据放在因子里而不是 ingest：换手率恰为 0 在任何来源下都不是有意义的取值，
      域约束跟着因子走就不依赖 vendor 行为。
    """
    codes = list(universe)
    df = query.get_daily_basic(as_of_date, codes, fields=("turnover_rate_f",), lookback=window)
    if df.empty:
        return pd.Series(np.nan, index=codes, name="turnover_20", dtype=float)
    panel = df["turnover_rate_f"].unstack("ts_code").reindex(columns=codes)
    panel = panel.where(panel > 0)
    ok = panel.notna().sum() >= math.ceil(_MIN_COVERAGE * window)
    return panel.mean().where(ok).rename("turnover_20")


@factor(name="amihud_20", direction=1, category="price", lookback_days=20 + _BUFFER, window=20)
def amihud_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`1e9 × mean(|r_s| / amount_s)`，r 取对数收益。值大 = 单位成交额推动的价格幅度大。

    ★ 常数 1e9 与 amount 的单位（Tushare 是千元）都不影响结果：后面要过 zscore，
      任何正的常数乘子都被标准化抹掉。只有拿 ILLIQ 的绝对水平对比文献时才需要在意单位。

    ★ `amount <= 0`（停牌日成交额为 0）必须先剔除：`|r|/0 = inf`，而 MAD 去极值对
      inf 无能为力（median 还在，但 clip 上界之外的 inf 会被截到上界，成为最"非流动"
      的一只），一只停牌股就能绑架整个横截面的排序。
    ★ 但只剔 `amount<=0` 不够：那道闸挡掉的是停牌【当日】，漏掉的是【复牌日】——
      分子是整段停牌的累计收益，分母只有复牌那一天的成交额，两边不是同一个区间。
      所以分子走 `_daily_returns`（两端都真实成交才算），复牌日随停牌日一起剔除。
    ★ 用 `mean` 而不是"恒除以 20"：分子少了几项分母还是 20，会把停牌股系统性地
      算成流动性很好。剔掉多少天由 60% 覆盖率闸门兜底。
    """
    codes = list(universe)
    px, real, ok = _closes(as_of_date, universe, window)
    bars = query.get_bars(as_of_date, codes, lookback=window + _BUFFER, fields=("amount",))
    if bars.empty:
        return pd.Series(np.nan, index=codes, name="amihud_20", dtype=float)
    r = _daily_returns(px, real, log=True)
    amt = bars["amount"].unstack("ts_code").reindex(index=r.index, columns=codes)
    return (1e9 * (r.abs() / amt.where(amt > 0)).mean()).where(ok).rename("amihud_20")


@factor(name="max_ret_20", direction=-1, category="price", lookback_days=20 + _BUFFER, window=20)
def max_ret_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """窗口内单日最大【简单】涨幅（彩票效应：博彩偏好推高的股票后续跑输，direction=-1）。

    ffill 造出来的 0 收益必须剔（`_daily_returns`）：窗口内真实收益全为负时，
    那个 0 会赢下 `max` —— 天天跌 2% 的票本该报 −0.0200，有一天停牌就报 0.0000。
    direction=-1，所以它是【惩罚】而非奖励，但报的是一天根本没有交易的"最好一天"。
    """
    px, real, ok = _closes(as_of_date, universe, window)
    r = _daily_returns(px, real, log=False)
    return r.max().where(ok).rename("max_ret_20")
