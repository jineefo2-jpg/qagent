"""风险因子（算法说明书 §2.3）—— 三个，全部 `neutralize=False`，**不作 alpha**。

    SIZE = ln(MV)      IND ∈ {0,1}^K（申万一级）      β⁽²⁵⁰⁾ = Cov(rᵢ, r_mkt) / Var(r_mkt)

★ `neutralize=False` 不是风格偏好，是必需：这三个【就是】`pipeline.neutralize` 的回归元。
  把 log_mv 拿去对 log_mv 回归取残差，残差恒等于 0（数值上是浮点噪声），zscore 再把它
  除以一个同样是噪声的标准差 —— 得到一列放大了 1e16 倍的舍入误差。不抛、不告警，
  和一个真因子长得一模一样。

★ 这三个【不进 combine】。`FactorSpec` 里没有 is_alpha 字段，能区分它们的只有
  `category == 'risk'` 与 `neutralize is False` 两个标记（Task 7 的 combine 应据此拒绝，
  见 task-6-report 的规格问题）。`industry` 额外多一层结构性保险：它返回 category dtype，
  `winsorize_mad` 的 `median()` 在 category 上直接 TypeError —— 误用当场炸，
  好过产出一列"看起来是分数"的东西。

★ direction 对风险因子没有语义（industry 连序都没有），但装饰器要求 ±1。三个统一写 −1：
  万一被谁误用作 alpha，小市值 / 低 beta 至少与文献方向一致，不会变成系统性反向下注。
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

MARKET_INDEX = "000985.CSI"     # 中证全指：覆盖全市场，比沪深 300 更适合做小盘股的市场基准
_MIN_COVERAGE = 0.60            # 250 × 0.60 = 150，与 price.py 同一口径


@factor(name="log_mv", direction=-1, category="risk", lookback_days=1, neutralize=False)
def log_mv(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`ln(总市值)`。单位（Tushare 的 total_mv 是万元）不影响任何下游：取对数后单位只是
    一个平移项，被中性化回归的截距整个吸收。

    `total_mv <= 0` 或当日无行（停牌 / 新股）→ NaN，**不是 −inf**：一个 −inf 进
    `neutralize` 的设计矩阵，`lstsq` 整列返回 nan，那一天【全池】的因子作废 ——
    而报出来的样子是"这天没有信号"。
    """
    codes = list(universe)
    mv = query.get_daily_basic(as_of_date, codes, fields=("total_mv",))["total_mv"] \
              .reindex(codes).astype(float)
    return np.log(mv.where(mv > 0)).rename("log_mv")


@factor(name="industry", direction=-1, category="risk", lookback_days=1, neutralize=False)
def industry(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """as_of 时点的申万一级行业（PIT；成分 < 5 家的小行业已由 query 并入 `__OTHER__`，
    否则中性化的行业哑变量会奇异）。

    ★ 返回 **category** dtype 而不是字符串：见模块头 —— 这是"不作 alpha"的结构性保险。
    ★ 池内某股在 `stock_basic` 里没有行 → NaN 且【保留 index】：neutralize 按 universe
      做横截面回归，少一个标签会静默错位。
    ★ 本因子不查 `_meta.industry_source`：降级的行业标签（今天的值回填到上市日）拿去做
      **中性化**才是前视，那道阻断项在 `pipeline.neutralize` 里 —— 它是唯一会把行业
      放进回归的地方。在这里再拦一次会连"取标签看一眼"都做不到。
    """
    codes = list(universe)
    return query.get_industry(as_of_date, codes).reindex(codes) \
                .astype("category").rename("industry")


@factor(name="beta_250", direction=-1, category="risk", lookback_days=260,
        neutralize=False, window=250)
def beta_250(as_of_date: DateLike, universe: Sequence[str], *, window: int = 250) -> pd.Series:
    """对中证全指的 `window` 日 beta：`Cov(rᵢ, r_mkt) / Var(r_mkt)`，日【对数】收益。

    ★ 停牌日**不 ffill** —— 这是本因子与 `price.py` 的关键分歧，理由是 D9：
      停牌占位行（`vol=0`、OHLC = 前收）在 query 层已被掩成 NaN，ffill 会把它还原成
      一个**假的 0 收益**。而"大盘在动、这只票没动"从来不是一个观测 —— 那天根本没有
      观测。把假零当真会把 beta 系统性地拉向 0，停牌越多的股票 beta 越低，中性化于是
      少扣了它们的市场暴露，残差里留下的正是最不流动那批股票的市场敞口。
      不 ffill 的代价是复牌当日的收益也一起作废（它的前收是 NaN）—— 那同样是**对**的：
      拿一段跨停牌期的累计收益去配单日的市场收益是另一种错配。

    ★ 分母的 `Var(r_mkt)` 在【每只股票自己的有效子集】上算。用全窗口方差配一个只剩
      160 天的协方差，就是两个不同样本的比值 —— 停牌多的股票 beta 会系统性地偏。

    ★ 有效观测（个股与指数当日都有收益）少于 `ceil(0.60 × window)`（250 → 150 日）→ NaN。
    ★ Var(r_mkt) = 0（指数窗口内一动不动）→ NaN，不是 ±inf。这里【不需要】分母闸门：
      方差为 0 意味着 dm 全零，于是协方差必然也是 0，`0/0` 在 IEEE 754 里就是 NaN
      （只有 `x/0, x≠0` 才是 inf，而那走不到）。写一道 `where(var > 0)` 是死代码 ——
      变异检查逮到它删掉之后没有任何测试变红。结果本身由
      `test_beta_250_flat_market_is_nan_not_infinite` 钉住。
    """
    codes = list(universe)
    nan = pd.Series(np.nan, index=codes, name="beta_250", dtype=float)
    span = window + 1                       # window 个日收益需要 window+1 根收盘价
    px = query.get_price_panel(as_of_date, codes, "close", span)
    idx = query.get_index_bars(as_of_date, MARKET_INDEX, lookback=span, fields=("close",))
    if px.empty or idx.empty:
        return nan
    rm = np.log(idx["close"].astype(float)).diff().dropna()
    r = np.log(px.astype(float)).diff().reindex(rm.index)       # ★ 按【标签】对齐，不按位置
    if r.empty:
        return nan
    ok = r.notna()
    mkt = ok.mul(rm, axis=0).where(ok)                          # 指数收益广播到每一列，无效日 NaN
    dm, dr = mkt.sub(mkt.mean()), r.sub(r.mean())               # 每列各自在自己的有效子集上去均值
    beta = (dr * dm).sum() / (dm ** 2).sum()                    # ddof 在比值里约掉，不必写
    return beta.where(ok.sum() >= math.ceil(_MIN_COVERAGE * window)).rename("beta_250")
