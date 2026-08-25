"""基本面因子（算法说明书 §1.2 / §2.2）—— 八个，全部建立在 PIT 财报之上。

和 `price.py` 一样，本文件产出的是【原始因子值】：去极值 / 中性化 / zscore 全在
`pipeline.process` 里做。所以任何正的常数乘子（Tushare 的财报是【元】、`total_mv` 是
【万元】，估值三兄弟因此整体带一个 1e4 因子；`grossprofit_margin` 是百分数）都无害 ——
横截面标准化会把它抹掉。真正致命的是下面这四件事，本文件各有一处应对：

  1 A 股财报是【累计口径】：Q1 报 Q1、半年报报 Q1+Q2、三季报报 Q1+Q2+Q3、年报报全年。
    单季值 = 相邻累计值之差，且【跨年必须重置】—— Q1 的单季值【就是】Q1 累计值，
    不是「Q1 累计 − 上年年报累计」。写错的话 Q1 得到一个巨额负数（§10.1），而
    Q2–Q4 只是"数字变了"（0.50 → 0.53）—— 后者才是能活到生产的那种错。
    ★ TTM 拼接【不在这里做】：`query.get_financial_ttm` 已经实现了流量科目的累计拼接
      与存量科目的期初期末平均。抄到因子层就是四个因子各写一遍同一段易错逻辑。
      只有 np_yoy / sue 需要多期【单季】值，query 不提供，才在 `_single_quarter` 里差分。

  2 分子取 PIT 财报（`ann_date <= as_of`）、分母取 as_of 【当日】市值，两个时点必须一致。
    错配是双向的前视：旧财报配新市值，或新财报配旧市值，都不会抛。

  3 分母【为负或接近 0】一律 NaN，永不 ±inf（§2.2 / §10.2）。A 股有成片的亏损股与
    资不抵债股：净资产 −50 除出来是个【有限的】负 BP，会被排到"最贵"的一端；扭亏样本的
    `25 / (−5) − 1 = −6` 更是把业绩大幅改善读成大幅恶化。而 inf 连 MAD 去极值都拦不住
    （clip 到上界后成为最极端的一只），过完 zscore 就独占整个组合。
    ★ "接近 0"与"为负"同等重要，而且更隐蔽：30/0.01 = 3000 是【有限】的，`den > 0`
      放行、MAD 一样 clip 到上界 —— 落点与 ±inf 分毫不差。所以除了 `_ratio` 这条
      ≤ 0 的公共闸门，roe_ttm / np_yoy / sue 各自还有一条"接近 0"判据（§2.2 的裁决表，
      两条是无阈值的原理性判据，只有 np_yoy 那条 0.01 是判断值）。

  4 期数不足一律 NaN，不外推、不前向填充。sue 的 σ 尤其危险：样本越少 σ 越小、
    SUE 越大 —— 放宽期数闸门等于让数据最少的股票系统性地排在最前面。

`lookback_days` 全部为 1：`financial_pit` 不在 `query._PRELOADABLE` 里（PIT 查询自带
`ann_date <= as_of` 过滤，没有窗口可预热），而估值类因子只读 as_of 当日的 `daily_basic`。
"""
from __future__ import annotations
import datetime as _dt
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ashare.data import query
from .base import factor

DateLike = Union[str, _dt.date]

_Q_LAST_DAY = {3: 31, 6: 30, 9: 30, 12: 31}     # 季末月 → 当月最后一天
_NP_YOY_PERIODS = 6                             # 单季 e* 与 e*−4 各需两期累计值 → e*..e*−5
_SUE_PERIODS = 13                               # 12 期单季（e*..e*−11）→ 累计 e*..e*−12
_SUE_LAGS = 8                                   # §2.2 的差分集合 {k}_{k=0}^{7}
_SUE_SIGMA_EPS = 1e-9                           # σ 相对 mean|Δ| 的【浮点判零】阈，不是调参
# ★ 本文件唯一的【判断值】（其余两条近零判据都是无阈值的原理性判据）：分母至少要有分子
#   的 1%，等价于给隐含增速封顶 100 倍。故意定得宽松 —— 只杀基数效应，不碰真实高增长。
#   §2.2 把它的复核交给 §9 风格归因：若右尾仍由基数效应主导，正确的补救是改用 SUE 那种
#   以历史波动标准化的形式，而不是把 100 倍挪成 50 倍（挪阈值治不了比值在零点的不稳定）。
_NP_YOY_MIN_BASE = 0.01


# ══════════════ 取数与除法的两条公共通道 ══════════════
def _pit(as_of_date: DateLike, codes: list[str], field: str) -> pd.Series:
    """最新一期 PIT 披露的【期末】值，index=universe 顺序，无披露 → NaN。

    `include_restated` 用默认的 False：重述行（`update_flag=1`）的 ann_date 不保证是
    重述当日，拿来回测就是前视（D3）。这里显式不传，靠 query 的默认值兜底。
    """
    return query.get_financial(as_of_date, codes, [field])[field].reindex(codes).astype(float)


def _mv(as_of_date: DateLike, codes: list[str]) -> pd.Series:
    """as_of 当日总市值。停牌 / 无行 → NaN；≤ 0 也当缺失（做分母）。"""
    s = query.get_daily_basic(as_of_date, codes, fields=("total_mv",))["total_mv"] \
             .reindex(codes).astype(float)
    return s.where(s > 0)


def _ratio(num: pd.Series, den: pd.Series, name: str) -> pd.Series:
    """`num / den`，但分母 ≤ 0 或缺失一律 NaN —— 这条通道拦的是 ±inf 与符号翻转。

    ★ 它【不】负责 §2.2 的另一半"分母接近 0"：那一半没有通用判据（怎样算接近 0
      取决于分子的量级 / 端点符号 / 是不是浮点残差），三个因子各自在调用处解决。
    """
    return (num / den.where(den > 0)).rename(name)


# ══════════════ 累计口径 → 单季（§1.2，跨年重置）══════════════
def _q_offset(end: _dt.date, k: int) -> _dt.date:
    """`end` 往前推 k 个季末（k=4 即去年同期）。"""
    m = end.month - 3 * k
    year, month = end.year + (m - 1) // 12, (m - 1) % 12 + 1
    return _dt.date(year, month, _Q_LAST_DAY[month])


def _single_quarter(as_of_date: DateLike, codes: list[str], field: str, n_periods: int
                    ) -> tuple[pd.DataFrame, pd.Series]:
    """PIT 累计值差分成单季值。返回 `(宽表 index=ts_code / columns=end_date, 每股的 e*)`。

    ★ 跨年重置就在这个 if 里：3 月末是年内首期，累计【即】单季，不做任何减法。
    ★ 上一季末不在结果里（未披露 / 超出 n_periods）→ 该期不产出单季值，不用更早的
      期次凑数 —— 用 e*−2 冒充 e*−1 得到的是"两个季度的和"，量级直接翻倍。
    """
    e_star = pd.Series(index=pd.Index(codes, name="ts_code"), dtype=object)
    df = query.get_financial(as_of_date, codes, [field], n_periods=n_periods)
    if df.empty:
        return pd.DataFrame(index=e_star.index), e_star
    e_star = df.groupby(level="ts_code")["end_date"].max().reindex(codes)
    cum = df[field].astype(float).unstack("end_date").reindex(codes)
    single = pd.DataFrame(np.nan, index=cum.index, columns=cum.columns)
    for end in cum.columns:
        if end.month not in _Q_LAST_DAY:            # 非季末（脏数据）→ 不产出
            continue
        if end.month == 3:
            single[end] = cum[end]                  # ★ 年内首期：累计即单季
        else:
            prev = _q_offset(end, 1)
            if prev in cum.columns:
                single[end] = cum[end] - cum[prev]
    return single, e_star


def _lag(single: pd.DataFrame, e_star: pd.Series, k: int) -> pd.Series:
    """每股在【自己的】e* 往前 k 个季度那一期的单季值；该期缺席 → NaN。

    偏移按季末日历算，不是"倒数第 k 行" —— 中间漏报一期时，按行数取会把 e*−5 当成
    e*−4，于是同比拿去年三季度的数当去年同期，静默错季。

    ★ e* 本身不是季末（脏数据）→ 该股 NaN，不能让 `_q_offset` 抛 KeyError：
      `financial_pit.end_date` 是裸 DATE 没有 CHECK，一行 end_date=10-31 就能成为某股的
      e*，抛出去的是【整次调用】—— 同一横截面里数据完好的股票也一起拿不到值。
      `_single_quarter` 的循环里早就有同样的 `continue`，这里补齐另一半。
    """
    col = {c: i for i, c in enumerate(single.columns)}
    mat = single.to_numpy(dtype=float)
    out = []
    for row, end in enumerate(e_star):
        ok = isinstance(end, _dt.date) and end.month in _Q_LAST_DAY
        j = col.get(_q_offset(end, k)) if ok else None
        out.append(mat[row, j] if j is not None else np.nan)
    return pd.Series(out, index=single.index, dtype=float)


# ══════════════ 1–3 估值：分子 PIT 财报 / 分母 as_of 当日市值 ══════════════
@factor(name="ep_ttm", direction=1, category="fundamental", lookback_days=1)
def ep_ttm(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`TTM(归母净利) / 总市值`。亏损股的分子为负是【有意义】的（EP 可以为负），
    所以只闸分母不闸分子。"""
    codes = list(universe)
    return _ratio(query.get_financial_ttm(as_of_date, codes, "n_income_attr_p"),
                  _mv(as_of_date, codes), "ep_ttm")


@factor(name="bp", direction=1, category="fundamental", lookback_days=1)
def bp(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`E_{e*} / 总市值`。★ 分子是【期末】净资产，不是 roe_ttm 那个期初期末平均 ——
    §2.2 里 BP 的下标是 e*、ROE 的分母才写了 ½(·+·)。两者差一成，都"看着合理"。"""
    codes = list(universe)
    return _ratio(_pit(as_of_date, codes, "total_hldr_eqy_exc_min_int"),
                  _mv(as_of_date, codes), "bp")


@factor(name="sp_ttm", direction=1, category="fundamental", lookback_days=1)
def sp_ttm(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`TTM(营业收入) / 总市值`。用 `revenue`（营业收入）而非 `total_revenue`
    （营业总收入，含利息 / 手续费收入）—— 后者会让银行券商的 SP 与制造业不可比。"""
    codes = list(universe)
    return _ratio(query.get_financial_ttm(as_of_date, codes, "revenue"),
                  _mv(as_of_date, codes), "sp_ttm")


# ══════════════ 4–6 盈利能力 / 质量 ══════════════
@factor(name="roe_ttm", direction=1, category="fundamental", lookback_days=1)
def roe_ttm(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`TTM(归母净利) / ½(E_{e*} + E_{e*−4})`，且要求【两个端点】净资产都 > 0。

    分母的期初期末平均【不用自己算】：`get_financial_ttm` 对存量科目返回的就是
    (最新 + 上年同期)/2，与 §2.2 的定义逐字一致。

    ★ 只查平均值 > 0 不够（§2.2 裁决）：E_{e*}=−999.98 配 E_{e*−4}=1000 的平均值是
      0.01，闸门放行后 ROE = 10000 —— 一个【有限】的数，MAD 去极值把它 clip 到上界，
      落点与 ±inf 相同。一大负一大正相抵出来的近零分母描述的是一家正滑向资不抵债的
      公司，根本不是可用的分母。查两个端点的符号是【原理性】判据，不需要任何阈值。
      E_{e*} 走 `_pit`（同一套 PIT 可见集，e* 与 TTM 那边一致），上年同期由
      `2·平均 − E_{e*}` 反解 —— 只为看符号，不必再查一次库。
    """
    codes = list(universe)
    eq_end = _pit(as_of_date, codes, "total_hldr_eqy_exc_min_int")
    eq_avg = query.get_financial_ttm(as_of_date, codes, "total_hldr_eqy_exc_min_int")
    both_positive = (eq_end > 0) & (2.0 * eq_avg - eq_end > 0)
    return _ratio(query.get_financial_ttm(as_of_date, codes, "n_income_attr_p"),
                  eq_avg.where(both_positive), "roe_ttm")


@factor(name="gross_margin", direction=1, category="fundamental", lookback_days=1)
def gross_margin(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """最新一期 PIT 毛利率（`grossprofit_margin`，Tushare fina_indicator，百分数）。

    ★ 不做 TTM：这是【比率】科目，`get_financial_ttm` 对它直接抛 UnknownFieldError，
      而且比率的 TTM 不是四期相加 —— 要分子分母各自 TTM 再相除，而 `financial_pit`
      里没有营业成本列，拼不出来。累计口径的比率本身已是年初至今的加权平均。
    """
    codes = list(universe)
    return _pit(as_of_date, codes, "grossprofit_margin").rename("gross_margin")


@factor(name="accrual", direction=-1, category="fundamental", lookback_days=1)
def accrual(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`(TTM(净利) − TTM(经营现金流)) / TA_{e*}`。★ direction=−1：净利远高于经营现金流
    说明利润含金量低，后续跑输 —— 符号写反就是系统性买进最差的一批。

    两处口径按 §2.2 的下标字面执行，两处都容易随手写反：
      · 分子用 `n_income`（全部净利），不是归母净利 —— 分母是【全部】资产，主体要一致；
      · 分母是【期末】总资产 TA_{i,e*}，不是 get_financial_ttm 的期初期末平均。
    """
    codes = list(universe)
    ni = query.get_financial_ttm(as_of_date, codes, "n_income")
    cfo = query.get_financial_ttm(as_of_date, codes, "n_cashflow_act")
    return _ratio(ni - cfo, _pit(as_of_date, codes, "total_assets"), "accrual")


# ══════════════ 7–8 成长 / 盈余惯性（唯二需要多期单季值的因子）══════════════
@factor(name="np_yoy", direction=1, category="fundamental", lookback_days=1)
def np_yoy(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`NI^Q_{e*} / NI^Q_{e*−4} − 1` —— 单季同比，不是累计同比。

    分母 ≤ 0 → NaN：扭亏样本（去年同期亏损）算出来的比值符号是反的，
    `25 / (−5) − 1 = −6` 会把一个业绩大幅改善的样本打成最差分（§10.2）。

    ★ 分母【接近 0】同样置 NaN，判据是相对下限 `base > 0.01 · |分子|`（§2.2）：
      0.01 的基数配 30 的单季利润给出 2999，这是个【有限】值，`base > 0` 放行、
      MAD 去极值 clip 到上界 —— 与 ±inf 落到同一个位置，而 direction=+1 让它成为
      等权 top-N 的头号买入。A 股有大量微利/保壳公司把利润做到近零，它们次年的
      "增速"是上一年盈余管理的产物，不是经营改善。
      用相对下限而不是绝对下限：全市场净利跨 4 个数量级，对茅台不算钱的门槛
      对小盘股是全部身家。`base > 0.01·|num|` 一条式子同时管住了负分母
      （右端非负，负 base 必然落选）与近零分母。
    """
    codes = list(universe)
    single, e_star = _single_quarter(as_of_date, codes, "n_income_attr_p", _NP_YOY_PERIODS)
    num, base = _lag(single, e_star, 0), _lag(single, e_star, 4)
    return (num / base.where(base > _NP_YOY_MIN_BASE * num.abs()) - 1.0).rename("np_yoy")


@factor(name="sue", direction=1, category="fundamental", lookback_days=1)
def sue(as_of_date: DateLike, universe: Sequence[str]) -> pd.Series:
    """`(NI^Q_{e*} − NI^Q_{e*−4}) / σ({NI^Q_{e*−k} − NI^Q_{e*−k−4}}_{k=0..7})`。

    ★ 分母是同比【差分】的标准差，不是同比【增速】的标准差（规格表格的措辞与 §2.2 的
      公式在这里打架，以公式为准）：分子是有量纲的差分，分母换成无量纲的增速，
      SUE 就不再是"以自身波动为单位的盈余惊喜"，而是一个量纲混杂的数。
    ★ 8 个差分缺一不可（= 12 期单季齐全）。放宽成"有几期算几期"会让 σ 变小、SUE 变大，
      数据最少的次新股因此系统性地排在最前面。
    ★ σ ≈ 0（盈利完全线性增长）→ NaN，不是 ±inf。闸门挂在【差分自身的量级】上
      （σ > 1e-9 · mean|Δ|），不是 σ > 0：8 个差分完全相同时 σ 数学上就是 0，但在
      1e9 的量级上浮点残差会给出 3.6e-7，绝对闸门放行后 SUE 算出 1e15 —— 一个有限的、
      过得了 MAD 去极值的天文数字。相对判据纯粹在回答"这个 σ 实际上是不是零"，
      是浮点判零，不是可调的参数（§2.2）。
    """
    codes = list(universe)
    single, e_star = _single_quarter(as_of_date, codes, "n_income_attr_p", _SUE_PERIODS)
    diff = pd.concat([_lag(single, e_star, k) - _lag(single, e_star, k + 4)
                      for k in range(_SUE_LAGS)], axis=1)
    sd = diff.std(axis=1, ddof=1).where(diff.notna().all(axis=1))
    return (diff.iloc[:, 0] / sd.where(sd > _SUE_SIGMA_EPS * diff.abs().mean(axis=1))).rename("sue")
