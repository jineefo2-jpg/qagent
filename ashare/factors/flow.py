"""资金因子（算法说明书 §2 / 设计规格 §4.4）—— 只有一个：北向持股比例的 20 日变化。

和 `price.py` / `fundamental.py` 一样，本文件产出的是【原始因子值】：去极值 / 中性化 /
zscore 全在 `pipeline.process` 里做。

★ 本文件真正危险的东西不是公式，是「没有数据」这件事怎么表达。

  沪深港通的个股持股明细 **2016-12-05** 才开始披露：在此之前的六年，以及在此之后
  所有不是通道标的的股票，这个因子【没有值】。没有值必须是 NaN，绝不能是 0 ——
  0 在本因子里是一个合法取值（"持股比例没变"），所以填 0 过得了任何形状检查、
  任何 dtype 检查、任何"非空率"统计。而 `combine` 的规则是「覆盖率不足或
  available_from 未到的因子从当日分母中剔除并重新归一」（架构 B5）：填 0 会让它
  留在分母里，分子却恒等于同一个常数 —— 2010–2016 的合成分数被静默降权六年，
  净值曲线照样画得出来，读起来像"那几年策略比较钝"。

  声明在 `FactorSpec.available_from`，短路闸门在 `compute_factor`（Task 7）。
  本文件这一侧的责任只有一条：拿到全 NaN 的输入时【原样传出 NaN】。
  这里任何一处 `fillna(0)` 都是上面那个 bug。

★ 停牌 / 数据洞：`get_money_flow` 按交易日历补齐，缺的日子是一行 NaN。窗口内先 ffill
  —— 停牌期间持股比例本来就不变，前值就是真值；但 ffill 填出来的不是新信息，
  所以再加一道与 `price.py` 同源的 60% 覆盖率闸门。
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

NORTHBOUND_START = _dt.date(2016, 12, 5)    # 沪深港通个股持股明细的披露起始日
_MIN_COVERAGE = 0.60                        # 窗口内原始非空天数占比下限（同 price.py）
_BUFFER = 10                                # 窗口之外多取的天数：左端点缺失时 ffill 的前值在窗口外


@factor(name="north_hold_chg_20", direction=1, category="flow", lookback_days=31,
        available_from=NORTHBOUND_START, window=20)
def north_hold_chg_20(as_of_date: DateLike, universe: Sequence[str], *,
                      window: int = 20) -> pd.Series:
    """`hk_hold_ratio_t − hk_hold_ratio_{t−window}`（百分点之【差】）。direction=+1：北向增持是正信号。

    ★ 取差不取比：持股比例的基数极小，0.01% → 0.02% 是"翻倍"却毫无经济含义。
      用比值会把最不被北向关注的那批股票系统性地排在最前面。

    ★ 只有【一道】早退闸门。原来还有一道 `if raw.empty` 写在它前面，删掉了：
      `get_money_flow` 把结果 reindex 到 `MultiIndex.from_product([codes, days])`
      （`query.py:855`），行数只由日历和 lookback 决定、与库里有没有数据无关 ——
      于是 `raw.empty` ⟺ 池为空或日历为空 ⟹ `len(panel) == 0`，被长度闸门完全覆盖。
      两道并排时删掉任意一道都没有测试变红，那正是"让读者以为这里有两层保护"的假象。
      剩下这道不能删：`len(panel) == 0` 时 `px.iloc[-1]` 直接 IndexError。
    """
    codes = list(universe)
    span = window + 1                       # 两个端点都要，21 天
    nan = pd.Series(np.nan, index=codes, name="north_hold_chg_20", dtype=float)
    raw = query.get_money_flow(as_of_date, codes, fields=("hk_hold_ratio",),
                               lookback=span + _BUFFER)
    panel = raw["hk_hold_ratio"].unstack("ts_code").reindex(columns=codes).astype(float)
    if len(panel) < span:                   # 回测头几天取不满一个窗口；空池 / 空日历也走这里
        return nan
    ok = panel.iloc[-span:].notna().sum() >= math.ceil(_MIN_COVERAGE * span)
    px = panel.ffill().iloc[-span:]
    return (px.iloc[-1] - px.iloc[0]).where(ok).rename("north_hold_chg_20")
