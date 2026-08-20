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
            ) -> tuple[pd.DataFrame, pd.Series]:
    """取窗口 `[t-span, t-skip]` 的后复权收盘价，返回 `(ffill 后的窗口面板, 有效掩码)`。

    掩码按窗口内【ffill 之前】的非空天数算 —— ffill 后再数一律是 100%，那道闸门就没了。
    历史长度连一个窗口都不够时返回全 NaN 面板（而不是抛）：回测头几天必然如此，
    这些股票本来就该是 NaN。
    """
    codes = list(universe)
    need = span - skip + 1
    raw = query.get_price_panel(as_of_date, codes, "close", span + _BUFFER)
    if len(raw) < span + 1:
        return (pd.DataFrame(np.nan, index=pd.RangeIndex(need), columns=codes),
                pd.Series(False, index=codes))
    lo, hi = len(raw) - span - 1, len(raw) - skip
    ok = raw.iloc[lo:hi].notna().sum() >= math.ceil(_MIN_COVERAGE * need)
    return raw.ffill().iloc[lo:hi], ok


@factor(name="reversal_20", direction=1, category="price", lookback_days=30, window=20)
def reversal_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`-(p_t / p_{t-window} - 1)`。负号在定义里：过去跌得多的分数高，所以 direction=+1。"""
    px, ok = _closes(as_of_date, universe, window)
    return (-(px.iloc[-1] / px.iloc[0] - 1)).where(ok).rename("reversal_20")


@factor(name="momentum_120_20", direction=1, category="price", lookback_days=130,
        window=120, skip=20)
def momentum_120_20(as_of_date: DateLike, universe: Sequence[str], *,
                    window: int = 120, skip: int = 20) -> pd.Series:
    """`p_{t-skip} / p_{t-window} - 1`，区间 `[t-120, t-20]`。

    最近 `skip` 天完全不进公式 —— 连覆盖率闸门也只看 `[t-120, t-20]`：
    整月停牌不该让一个本来就不看那一段的因子作废。
    """
    px, ok = _closes(as_of_date, universe, window, skip=skip)
    return (px.iloc[-1] / px.iloc[0] - 1).where(ok).rename("momentum_120_20")


@factor(name="volatility_60", direction=-1, category="price", lookback_days=70, window=60)
def volatility_60(as_of_date: DateLike, universe: Sequence[str], *, window: int = 60) -> pd.Series:
    """`window` 个对数收益的【样本】标准差（分母 window-1，规格 §2.1 写的就是 1/59）。"""
    px, ok = _closes(as_of_date, universe, window)
    r = np.log(px).diff().iloc[1:]
    return r.std(ddof=1).where(ok).rename("volatility_60")


@factor(name="turnover_20", direction=-1, category="price", lookback_days=30, window=20)
def turnover_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`window` 日平均自由流通换手率（`turnover_rate_f`）。A 股高换手的负向收益极显著。

    停牌日在 `daily_basic` 里没有行 / 为 NULL，`mean` 跳过即可 —— 不能补 0：
    补 0 会把停牌股算成"低换手"，而低换手在这个因子里是【好分数】。
    """
    codes = list(universe)
    df = query.get_daily_basic(as_of_date, codes, fields=("turnover_rate_f",), lookback=window)
    if df.empty:
        return pd.Series(np.nan, index=codes, name="turnover_20", dtype=float)
    panel = df["turnover_rate_f"].unstack("ts_code").reindex(columns=codes)
    ok = panel.notna().sum() >= math.ceil(_MIN_COVERAGE * window)
    return panel.mean().where(ok).rename("turnover_20")


@factor(name="amihud_20", direction=1, category="price", lookback_days=30, window=20)
def amihud_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """`1e9 × mean(|r_s| / amount_s)`，r 取对数收益。值大 = 单位成交额推动的价格幅度大。

    ★ `amount <= 0`（停牌日成交额为 0）必须先剔除：`|r|/0 = inf`，而 MAD 去极值对
      inf 无能为力（median 还在，但 clip 上界之外的 inf 会被截到上界，成为最"非流动"
      的一只），一只停牌股就能绑架整个横截面的排序。
    ★ 用 `mean` 而不是"恒除以 20"：分子少了几项分母还是 20，会把停牌股系统性地
      算成流动性很好。剔掉多少天由 60% 覆盖率闸门兜底。
    """
    codes = list(universe)
    px, ok = _closes(as_of_date, universe, window)
    bars = query.get_bars(as_of_date, codes, lookback=window + _BUFFER, fields=("amount",))
    if bars.empty:
        return pd.Series(np.nan, index=codes, name="amihud_20", dtype=float)
    r = np.log(px).diff().iloc[1:]
    amt = bars["amount"].unstack("ts_code").reindex(index=r.index, columns=codes)
    return (1e9 * (r.abs() / amt.where(amt > 0)).mean()).where(ok).rename("amihud_20")


@factor(name="max_ret_20", direction=-1, category="price", lookback_days=30, window=20)
def max_ret_20(as_of_date: DateLike, universe: Sequence[str], *, window: int = 20) -> pd.Series:
    """窗口内单日最大【简单】涨幅（彩票效应：博彩偏好推高的股票后续跑输，direction=-1）。"""
    px, ok = _closes(as_of_date, universe, window)
    r = px.div(px.shift(1)).iloc[1:] - 1.0      # 不用 pct_change：pandas 2 的 fill_method 默认值在变
    return r.max().where(ok).rename("max_ret_20")
