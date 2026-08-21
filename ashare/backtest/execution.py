"""成交模拟（铁律 D6 / 算法说明书 §5.1–5.3 + §5.5 退市清仓价）—— 回测最容易骗自己的那一处。

D6 的原文：「T 日收盘算信号，T+1 集合竞价（09:15–09:25）按开盘价成交；涨跌停/停牌不可交易。」
这一行拆开是四条独立的约束，本文件逐条落地，四条各配一条能反测的用例：

★ 1【两个日期，不是一个】
  信号来自 ≤T 收盘的数据，成交发生在 τ=T+1 的开盘。任何用 τ 的数据去【决定】买什么、
  或用 T 的价格去【成交】的路径都是前视。所以 `signal_date` 与 `exec_date` 是两个入参，
  谁也不从谁推导：`exec_date` 只用来取盘口，`signal_date` **只用来守卫**（断言 T < τ），
  在本文件里没有第二个用途 —— 拿它去取任何数据，本身就是这条约束要防的那个 bug。
  架构 §4.3 的签名里没有 `signal_date`，那个签名让"同一天算完就成交"变成一个静默可写
  的合法调用。补上它，引擎必须把两个日期都说出口，写错就当场炸。

★ 2【不可交易是有方向的】
  涨停买不进但卖得出，跌停卖不出但买得进，停牌两侧都不行。`get_tradable_mask` 已经把
  方向编码在 `can_buy` / `can_sell` 两列里（不是一个 bool），本文件按【意图方向】查对应
  的那一列。抹平成"不可交易 = 不动"会让一字涨停当天的减仓凭空消失 —— 而涨停日减仓
  恰恰是反转类因子最常发生的动作。

★ 3【拦住 = 不成交，不是换个价格成交】
  被拦的票停在 `w_prev` 上：不清零（清零 = 凭空以某个价格卖光了）、不重试。于是组合与
  目标之间留下一个真实的缺口，它必须出现在返回值里 —— `blocked` 就是 D6 的证据链。
  每期都恰好达成目标权重的回测，描述的是一个不存在的市场。

★ 4【成交价 = 执行日开盘价，后复权】
  A 股开盘价由 09:15–09:25 集合竞价撮合产生，全部成交按同一个开盘价成交，所以"开盘价"
  是这个模型里唯一可实现的价格；均价与收盘价都需要当天走完才知道（又一次前视）。
  价格一律取 `mask.open_hfq`（后复权，D8）—— 本文件不碰原始价，也不知道复权因子。

★ 5【买不进的额度留现金，绝不摊给别人】（§5.3，2026-08-20 裁决）
  可交易部分按 `min(1, (π − L)/Σ_{j∉F} w^tgt_j)` 放缩 —— **只向下缩，绝不向上放大**。
  初稿的比例再分配会把买不进的额度摊到幸存者头上，顶破 §7.2 刚建立的 `max_single`/
  `max_industry`（实测：一只锁住 0.20 的票把两只可交易的推到 0.26/0.52，而上限是 0.05）。
  判据是可见性不对称：现金拖累直接进净值曲线，回测自己会疼；超额集中度不被任何东西
  计价，只在样本可能根本不包含的尾部兑现。买不进就是买不进 —— 把钱假装投到别处，
  是发明一笔没发生的交易。

★ 6【缩量会把"小幅加仓"翻成"减仓"】（§5.2 的循环，只剩这一支）
  只缩不放之后，可交易的票拿的就是目标权重，`Δw = w^tgt − w^drift` 与 F 无关 ——
  §5.2 那个「掩码依赖方向、方向依赖再分配、再分配依赖掩码」的循环在最常见的分支上
  直接消失（那一支一轮定型）。**残留在缩量这一支**：L 逼近 π 时缩量把一笔小买变成卖，
  而那只票若一字跌停就卖不掉。所以 F 仍须迭代到不动点：F 单调增长，最坏情形全员入 F
  （此时无任何成交、无从违规），故必然收敛。与 `portfolio._violators` 的循环同一个套路。

不做的事（都是有意的，见 `simulate` 文档字符串的"本模型的天花板"）：
  · 不建模【部分成交】—— 全有或全无。流动性只经 §5.4 的冲击成本以【价格】计价，不以
    【成交量】计价。top-N 落在微盘股上时，现实中的集合竞价深度吃不下，回测吃得下。
  · 不做整手取整（A 股买入须 100 股整数倍）。
  · 不算费用 —— 那是 `cost.charge(trade_rows, cost_cfg)` 的事，故本文件不收 `CostConfig`
    入参：一个收下却不用的费用参数会让读者以为成交价里已经含费。
"""
from __future__ import annotations

import datetime as _dt
from typing import Union

import numpy as np
import pandas as pd

from ..data.query import get_tradable_mask

DateLike = Union[str, _dt.date]

TRADE_COLS = ["exec_date", "ts_code", "side", "shares", "price_hfq", "amount"]
BLOCKED_COLS = ["exec_date", "ts_code", "intended_side", "intended_weight", "reason"]

# 退市清仓折价（§5.5 / 架构 B8）：**最后一个有效后复权收盘价 × 0.5**。
# 退市整理期连续跌停、几乎无流动性，按收盘价成交是系统性乐观偏差。掩码给出的
# `close_hfq` 在 delisted 路由上恰好就是"最后一根非停牌 K 线的后复权收盘"，它位于跌停
# 序列的末端 —— 比 B8 初稿的"整理期首日收盘"更低，与 B8 自己的"宁可低估"同向。
#
# 价格只是这条路由的一半，另一半是【目标一律置 0】（§5.5「强制」的定义，2026-08-20 补）：
# 退市股不走普通 Δw 路径。走了的话，`build_targets` 的换手上限会把清仓裁成部分成交，
# 于是一份卖不掉的资产被钉在固定价上永远留在账上 —— 下一期目标与 `prev_w` 相等、Δw=0，
# 零成交零告警，而"退市清仓"那行 warning 照发不误，声称了一件没发生的事。
_DELIST_HAIRCUT = 0.5

# 权重容差。权重量级 1e-2，1e-12 只吞浮点噪声：没有它，再分配的舍入残差会造出
# 一堆 1e-16 权重的"成交"，每一笔都要交佣金。
_W_EPS = 1e-12


def _empty() -> tuple:
    return (pd.DataFrame(columns=TRADE_COLS),
            pd.Series(dtype=float, name="shares").rename_axis("ts_code"),
            pd.DataFrame(columns=BLOCKED_COLS),
            [])


def _allocate(tgt: pd.Series, prev_w: pd.Series, locked: pd.Series, pi: float) -> pd.Series:
    """§5.3：不可交易的锁在 w_prev，可交易的取目标权重原值、**只向下缩绝不向上放大**。

    放缩系数 `min(1, (π − L)/Σ_{j∉F} w^tgt_j)`：额度够就照目标发，不够才按相对比例同比缩。
    差额 `π − L − Σ_{j∉F} w_j` **留现金**（模块头 ★5：多出来的额度摊给幸存者会顶破
    `build_targets` 刚建立的 max_single / max_industry）。
    `L > π` 时额度为 0 —— 可交易部分清零，但**不强行卖出锁定仓位**（卖不出就是卖不出）。
    分母为 0（可交易的目标全为 0）即"把能卖的都卖掉"，结果自然是 0，不是 0/0。
    不归一化到 1：组合允许持现金，Σw ≤ π ≤ 1。
    """
    w = prev_w.where(locked, 0.0)
    free = tgt.where(~locked, 0.0)
    denom = float(free.sum())
    budget = max(pi - float(prev_w[locked].sum()), 0.0)
    return w if denom <= 0 else w + free * min(1.0, budget / denom)


def simulate(exec_date: DateLike,
             targets: pd.Series,
             prev_holdings: pd.Series,
             equity: float,
             *,
             signal_date: DateLike) -> tuple:
    """把目标权重在执行日撮合成成交，返回 `(trades, holdings, blocked, warnings)`。

    Args:
        exec_date: 执行日 τ = T+1。成交价与可交易性都只读这一天。
        targets: `portfolio.build_targets` 的目标权重，index = 目标持仓 ∪ 上期持仓，
            清仓的票显式写 0.0。**总仓位 π 取 `targets.sum()`** —— build_targets 在账本
            填不满时会留现金，π 因此不一定等于宏观层给的那个数，以传进来的账本为准。
            语义是**两值的**：缺席 == 0.0 == 卖光。目标账本按定义是完整的，"没意见"
            不是它的一个状态；`build_targets` 保留每一只 `prv != 0`（portfolio.py:226），
            持仓不会意外缺席。**退市股是唯一例外：目标被强制改写为 0**（§5.5）。
        prev_holdings: 上期持仓【股数】（不是权重），index=ts_code。首期传空 Series。
        equity: 执行日开盘时点的组合权益（现金 + 持仓市值），用于权重 ↔ 股数换算。
        signal_date: 信号日 T。**只用于守卫 T < τ，不用于取任何数据**（见模块头 ★1）。

    Returns:
        - `trades`: 列 = `TRADE_COLS`。`shares` / `amount` 是幅值，方向在 `side` 上。
          费用列由 `cost.charge` 追加。
        - `holdings`: 成交后的持仓股数，已卖光的票不再出现。被拦下的票**逐位等于**
          `prev_holdings` 里的值（不经价格往返，浮点上也不差）。
        - `blocked`: 列 = `BLOCKED_COLS`，D6 的证据链。`intended_side` / `intended_weight`
          描述的是**那笔没做成的交易**（方向 + |Δw| 幅值），不是目标仓位。
          ⚠ **`Σ intended_weight` 不是组合与目标之间的缺口**，两个方向都能偏：缩量支里
          被拦的是【缩过之后】那笔，实测 Σ|intent|=0.55 而真实缺口 0.80（低估）；
          `L > π` 时可交易部分被夹到 0，被拦的那笔反而比缺口大，实测 0.70 vs 0.42（高估）。
          极端情形下一只 `w^tgt == w_prev`（本不想动）的票也会因缩量进 `blocked`。
          ⚠ 交接 Task 12：要缺口规模请用 `targets` 与成交后 `holdings` 现算，别求和这一列。
        - `warnings`: 退市折价清仓 / 锁定权重超过目标仓位。汇进 `BacktestResult.warnings`。

    Raises:
        ValueError: `signal_date >= exec_date`（D6 被违反）；`equity <= 0`；
            `targets` / `prev_holdings` 含 NaN 或 inf；
            或**持仓在执行日取不到价格**。最后一条是刻意炸而不是兜底：D9 保证每只在市
            股票每个交易日都有行（停牌写占位行），`get_tradable_mask` 又把已知退市单独
            路由成 `reason='delisted'` 并附最后一根有效 K 线的收盘价。两者都不成立的持仓
            是【算不出权重】的持仓 —— 标 0 是抹掉净值、标上次价格是编造一段不存在的收益、
            留 NaN 是让整条曲线染上 NaN。三条都比"这一格数据坏了"更糟。

    本模型的天花板（`ponytail:` 记账，都不是疏忽）：
      · **不建模部分成交**：每只票全有或全无。流动性只在 §5.4 的冲击成本里以【价格】
        计价，不以【成交量】计价，也没有成交率折损。代价是集中在微盘股的 top-N 组合
        会拿到现实中拿不到的成交 —— §11 的 SIZE 分层反测（2010–2016 小盘极强）正好
        落在这个盲区上，那一项的结果要按"成交率 100%"打折读。
        要升级：按 `mask.amount` 给单只票设当日可成交额上限，未成交的余量并进 blocked。
      · **不做整手取整**（买入须 100 股整数倍）。100 万本金 / 50 只 = 2 万一只，向下取整
        到整手的权重误差可达个位数百分比。要升级：取整后把释放出来的额度回流再分配。
    """
    exec_d = pd.Timestamp(exec_date).date()
    sig_d = pd.Timestamp(signal_date).date()
    if sig_d >= exec_d:
        raise ValueError(
            f"signal_date={sig_d} 必须严格早于 exec_date={exec_d}（D6：T 日收盘算信号、"
            f"T+1 开盘成交）。相等 = 用收盘后才算得出的信号在当天成交")
    equity = float(equity)
    if not equity > 0:
        raise ValueError(f"equity={equity} 必须为正：权益非正时权重无定义")

    tgt_in = pd.Series(targets, dtype=float)
    prev_in = pd.Series(prev_holdings, dtype=float)
    # 输入里的 NaN/inf 必须当场炸：下面的 fillna(0.0) 只该补【新出现的 label】。
    # 目标 NaN 被填成 0 就是"卖光"，持仓 NaN 被填成 0 就是"空仓"（还顺手绕过无价守卫，
    # 再从一个不存在的零基上把目标买满）—— 两条都不抛、不告警，只在净值上留一道疤。
    for nm, ser in (("targets", tgt_in), ("prev_holdings", prev_in)):
        bad = list(ser.index[~np.isfinite(ser.to_numpy())])
        if bad:
            raise ValueError(f"{nm} 含 NaN/inf：{bad}。缺席才读作 0，写出来的 NaN 不是 0")
    codes = tgt_in.index.union(prev_in.index)
    if len(codes) == 0:
        return _empty()

    tgt = tgt_in.reindex(codes).fillna(0.0)
    shares_prev = prev_in.reindex(codes).fillna(0.0)
    pi = float(tgt.sum())

    mask = get_tradable_mask(exec_d, list(codes)).reindex(codes)
    reason = mask["reason"].astype(str)
    delisted = reason == "delisted"
    # §5.5【强制】清仓：退市股的目标一律 0，不接受上游给的残值（上游的换手上限会把清仓
    # 裁成部分成交，剩下的那半永远卖不掉）。改写的只是【这一只的目标】—— π 仍取调用方的
    # 账本：强卖换回来的是真现金，把 π 跟着调小等于让一次退市静默地给整个组合降仓，
    # 而 π 是宏观层的决策。腾出来的额度受 `_allocate` 的 min(1,·) 封顶，谁也过不了自己的
    # 目标（★5 的上限不会被顶破），填不满的照旧留现金。
    tgt = tgt.where(~delisted, 0.0)
    # 成交价：执行日开盘（后复权）。退市走 B8 折价 —— 它同时是【估值】价：
    # 一只只能半价卖掉的票，账上就值那么多，用开盘价标市值会先虚增权益再在成交时莫名亏掉。
    price = mask["open_hfq"].astype(float).where(
        ~delisted, mask["close_hfq"].astype(float) * _DELIST_HAIRCUT)
    # 价 0 与 NaN 同等对待：np.isfinite(0.0) 是 True，只挡 NaN 的守卫会放 0 过去，
    # 于是 d*equity/price = inf 一路走进 holdings 与权益，且不抛。负价同理（数据坏了）。
    priced = pd.Series(np.isfinite(price.to_numpy()) & (price.to_numpy() > 0), index=codes)

    unpriceable = (shares_prev != 0) & ~priced
    if unpriceable.any():
        bad = list(codes[unpriceable])
        raise ValueError(
            f"exec_date={exec_d} 持仓 {bad} 在执行日取不到价格（reason="
            f"{list(reason[unpriceable])}）—— 无法计算组合权重。D9 保证在市股票每个交易日"
            f"都有行、退市另有 'delisted' 路由，两者皆不成立说明数据有洞，不能拿一个编出来"
            f"的价格继续跑")

    # 取不到价 ⇒ 一律不可交易（此时必然未持仓，上面的守卫已挡住持仓的情形）。
    # 这不是给 no_quote 兜底（那一行掩码本来就两侧 False），而是给「掩码说可交易、价格却是
    # NaN」兜底 —— daily_bar.adj_factor 可空（validate.check_adj_factor_jumps 显式跳过 NULL），
    # 一行缺复权因子就会给出 can_buy=True + open_hfq=NaN。放过去的话成交股数是 NaN，
    # 从此持仓、权益、净值一路 NaN，而且不抛。
    can_buy = mask["can_buy"].astype(bool) & priced
    can_sell = mask["can_sell"].astype(bool) & priced
    reason = reason.where(priced | (reason != ""), "no_price")   # 证据链里不留空理由
    prev_w = (shares_prev * price / equity).where(priced, 0.0)

    # ── F 迭代到不动点（模块头 ★5）──
    locked = pd.Series(False, index=codes)
    intent = pd.Series(np.nan, index=codes)
    w = prev_w
    for _ in range(len(codes) + 1):
        w = _allocate(tgt, prev_w, locked, pi)
        d = w - prev_w
        new = (((d > _W_EPS) & ~can_buy) | ((d < -_W_EPS) & ~can_sell)) & ~locked
        if not new.any():
            break
        intent[new] = d[new]        # 记下"被拦住的那一笔"，而不是最终状态（最终状态是 0）
        locked |= new

    # ── 成交 ──
    d = w - prev_w                  # 锁定的票 w 逐位等于 prev_w，故 d 恰好是 0.0
    traded = d.abs() > _W_EPS
    d_shares = (d * equity / price).where(traded, 0.0)
    # 留下谁按【最终权重】判，不按股数是否恰好为 0 —— `w` 就是成交后的权重（逐位
    # `shares·price/equity == prev_w + d == w`，股数往返有舍入，`w` 没有）。两个阈值必须
    # 同一个量、同一个 eps：卖光后剩的 −6e-11 股在 `holdings != 0` 下留得下来，而它对应的
    # |Δw| ~ 1e-16 又永远达不到成交阈值 —— 从此每一只持过的票都赖在 codes 里，其中任何
    # 一只日后丢了行情，`unpriceable` 守卫（判 shares_prev != 0）就会把整轮回测炸掉。
    holdings = (shares_prev + d_shares)[w.abs() > _W_EPS].rename("shares").rename_axis("ts_code")

    trades = pd.DataFrame({
        "exec_date": exec_d,
        "ts_code": codes[traded],
        "side": np.where(d[traded] > 0, "BUY", "SELL"),
        "shares": d_shares[traded].abs().to_numpy(),
        "price_hfq": price[traded].to_numpy(),
        "amount": (d[traded].abs() * equity).to_numpy(),
    }, columns=TRADE_COLS)

    hit = locked & (intent.abs() > _W_EPS)
    blocked = pd.DataFrame({
        "exec_date": exec_d,
        "ts_code": codes[hit],
        "intended_side": np.where(intent[hit] > 0, "BUY", "SELL"),
        "intended_weight": intent[hit].abs().to_numpy(),
        "reason": reason[hit].to_numpy(),
    }, columns=BLOCKED_COLS)

    warns: list = []
    sold_off = delisted & traded
    if sold_off.any():
        warns.append(f"{exec_d} 退市清仓 {int(sold_off.sum())} 只（B8：按 close_hfq×"
                     f"{_DELIST_HAIRCUT} 入账，非收盘价）：{list(codes[sold_off])}")
    lock_w = float(prev_w[locked].sum())
    if lock_w > pi + _W_EPS:
        warns.append(f"{exec_d} 锁定权重 {lock_w:.4f} > 目标仓位 {pi:.4f}："
                     f"可交易部分全部清零，本期组合由不可交易的持仓决定，与目标无关")
    return trades.reset_index(drop=True), holdings, blocked.reset_index(drop=True), warns
