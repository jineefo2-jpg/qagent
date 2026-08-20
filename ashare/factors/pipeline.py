"""因子处理链（算法说明书 §3）—— 顺序固定，不可调换：

    1 MAD 去极值  →  2 行业 + 市值 OLS 中性化  →  3 zscore  →  4 fillna(0)

★ 为什么 MAD 不是 3σ：A 股横截面尾部极厚，均值与标准差【本身】已被极值污染，
  3σ 的上界会被拉到天上，等于没截。median ± n·1.4826·MAD 只依赖分位数，不被少数极值绑架。

★ 为什么 OLS 不是 WLS（本项目的裁决，与 Barra 的做法不同，理由是问题不同）：
  Barra 用 sqrt(MV) 加权是为了在【风险模型】里高效估计因子收益（异方差校正）。
  我们做的是 alpha 中性化，目的是"残差不再是规模的代理"，而衡量它的度量取决于组合怎么加权。
  本项目的组合是【等权 top-N】，所以相关的度量是【无权】正交 —— 恰是 OLS 的恒等式（X'e = 0）。
  WLS 的残差只在 sqrt(MV) 内积下正交，无权相关系数一般不为零，等权组合会因此带上规模倾斜，
  让人误把小盘暴露当成 alpha。
  另一面：OLS"回归线被小盘主导、大盘残差偏移"确实存在，但大盘股在等权组合里只占少数名字，
  影响最小；WLS 把偏差转移到数量占绝大多数的小盘股上，等权组合里影响最大。
  ★ 会推翻这个裁决的证据：Task 12 的风格归因在真实数据上显示残差仍有显著规模暴露。
    那时正确的补救是加非线性规模项（size²或秩变换），而不是换成 WLS —— WLS 同样不解决非线性。

★ 为什么 fillna(0) 只能在最后：中性化后 0 = 行业内平均水平，是诚实的中性先验；
  提前填会把这些 0 算进 zscore 的均值和标准差，把【所有】股票的分数一起拉偏。

★ 为什么秩亏 / 样本不足时返回原值而不是 NaN：静默返回 NaN 会让一整天的因子凭空消失，
  且没人看得见。返回原值 + warning，让降级的那一天出现在 BacktestResult.warnings 里。
"""
from __future__ import annotations
import datetime as _dt
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ashare.data import query
from .base import FactorSpec
from .risk import industry as _industry_factor, log_mv as _log_mv_factor

DateLike = Union[str, _dt.date]

MIN_OBS = 30            # 有效样本下限（架构 B7）：再少的横截面回归不出可信的行业/市值暴露
_BY_TERMS = ("log_mv", "industry")


def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """`clip(x, m − n·1.4826·MAD, m + n·1.4826·MAD)`，其中 m = median、MAD = median(|x − m|)。

    1.4826 使 MAD 在正态下成为 σ 的一致估计。NaN 原样穿过（末端才允许 fillna）。
    MAD = 0（过半数取值相同，稀疏因子常见）→ 上下界 collapse 到中位数，照着截会把整个
    因子拍平成一个常数，所以原样返回。
    """
    m = s.median()
    mad = (s - m).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return s.copy()
    d = n * 1.4826 * mad
    return s.clip(m - d, m + d)


def zscore(s: pd.Series) -> pd.Series:
    """`(x − mean) / std`。常数横截面 → 0/0 = NaN（不是 inf），由末端 fillna(0) 收成中性。"""
    return (s - s.mean()) / s.std()


def neutralize(s: pd.Series, as_of_date: DateLike, universe: Sequence[str], *,
               by: tuple[str, ...] = _BY_TERMS) -> tuple[pd.Series, list[str]]:
    """横截面 OLS 取残差：`x = α + β·log_mv + Σγₖ·Dₖ + ε`。

    残差与回归元【无权】正交（X'e = 0）—— 这正是等权 top-N 组合关心的度量。
    行业哑变量去掉一列（否则与截距完全共线，规格 §10.8）。
    市值缺失的股票算不出 log_mv，无论 `by` 取什么都置 NaN（不猜市值）。
    回归元取自已注册的 `risk.log_mv` / `risk.industry` —— 与 §9 风格归因是同一个定义。
    返回 `(残差, warnings)`；有效样本 < MIN_OBS 或设计矩阵秩亏 → 返回【原 Series】+ warning。
    """
    bad = [t for t in by if t not in _BY_TERMS]
    if bad:
        # 静默跳过拼错的项 = 这一天的因子根本没中性化，而输出看起来完全正常
        raise ValueError(f"neutralize 的 by 只支持 {_BY_TERMS}，收到未知项 {bad}")

    codes = list(universe)
    # ★ 回归元走【已注册的风险因子】，不在这里另算一遍：
    #   §9 的风格归因用 risk.log_mv / risk.industry 度量残差的规模暴露，而那正是检验
    #   §3.2「OLS 而非 WLS」裁决的唯一手段。若中性化减掉的量与归因度量的量是两份
    #   各自演化的实现，归因可以报告「规模暴露已清零」而账本实际带着倾斜 ——
    #   裁决就永远无法被证伪。两者必须是同一个定义。
    log_mv = _log_mv_factor(as_of_date, codes).reindex(s.index).astype(float)
    valid = s.notna() & log_mv.notna()

    parts = [pd.Series(1.0, index=s.index, name="const")]
    if "log_mv" in by:
        parts.append(log_mv.rename("log_mv"))
    if "industry" in by:
        src = query.industry_source()
        if src != "sw":
            # 降级的行业标签是【今天的值回填到上市日】：拿它做中性化 = 前视污染。
            # 建库时的 --allow-static-industry 只承认建库，不承认这一步，所以抛而不是 warning。
            raise RuntimeError(
                f"industry_source={src!r} 不是 'sw'：行业标签是今天的值回填到上市日，"
                f"用它做行业中性化会把未来的行业分类带进历史横截面（前视）。"
                f"请恢复申万成分数据后重建，或改用 by=('log_mv',)。")
        ind = _industry_factor(as_of_date, codes).reindex(s.index)
        valid &= ind.notna()
        # ★ 用【有效样本】构造哑变量，不能用全体：
        #   某个行业若全员无效（因子值 NaN，或市值缺失/为 0），在全体上建出的那一列在
        #   有效子集上恒为 0 → 设计矩阵秩亏 → 整个横截面退回未中性化的原值。
        #   drop_first 的基准行业若全员无效更糟：剩余哑变量在每个有效行上和为 1，
        #   与截距完全共线，同样秩亏。
        #   这条路是走得到的：get_industry 把缺失和成分 < min_members 的行业都并进 __OTHER__，
        #   而 __OTHER__ 收的恰是次新股之类最容易因子值缺失的名字。
        #   where(valid) 把无效行置 NaN，get_dummies 默认不为 NaN 建列 → 空类别自动消失。
        #   ★ .astype(object) 不可省：risk.industry 返回 category dtype，而 get_dummies
        #     对 categorical 会为【零观测的类别也建列】—— where(valid) 只把值置成 NaN，
        #     类别表还在，于是上面这层保护被无声地废掉，又退回秩亏。转成 object 后
        #     get_dummies 只按实际出现的值建列。
        parts.append(pd.get_dummies(ind.where(valid).astype(object),
                                    prefix="ind", drop_first=True, dtype=float))

    idx = s.index[valid]
    if len(idx) < MIN_OBS:
        return s.copy(), [f"{as_of_date} 中性化跳过：有效样本 {len(idx)} < {MIN_OBS}，返回未中性化的原值"]

    X = pd.concat(parts, axis=1).loc[idx].to_numpy(dtype=float)
    y = s.loc[idx].to_numpy(dtype=float)
    beta, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        # lstsq 秩亏时不抛，它给一个最小范数解 —— 那份"残差"没做过真正的中性化，静默且不可察
        return s.copy(), [f"{as_of_date} 中性化跳过：设计矩阵秩亏（rank {rank} < {X.shape[1]} 列），返回原值"]

    out = pd.Series(np.nan, index=s.index, dtype=float, name=s.name)
    out.loc[idx] = y - X @ beta
    return out, []


def process(s: pd.Series, as_of_date: DateLike, universe: Sequence[str], *,
            spec: FactorSpec) -> tuple[pd.Series, list[str]]:
    """1 winsorize_mad → 2（`spec.neutralize` 时）neutralize → 3 zscore → 4 fillna(0)。顺序不可调换。"""
    out = winsorize_mad(s)
    warnings: list[str] = []
    if spec.neutralize:
        out, warnings = neutralize(out, as_of_date, universe)
    return zscore(out).fillna(0.0), warnings
