"""交易成本（算法说明书 §5.4）—— 纯函数，一笔成交进、五列费用出。

    佣金 2.5bp（双边） + 印花税 5bp（**仅卖出**） + 过户费 0.1bp（双边）
    + 冲击 min(0.5 × 委托额/ADV20 × 当日振幅, 30bp)

一个完整往返 ≈ 0.3%。周频年化换手 10–20 倍 → 成本拖累 3%–6%，这是净值曲线上
最大的一块确定性损耗，算错任何一项都会直接改变「这条策略能不能上」的结论。

★【谁来给 ADV20】（Task 11 → Task 12 的悬空交接，在此定案）
  §5.4 的冲击项需要 ADV20 与当日振幅，两者都**不在** `execution.TRADE_COLS` 里，
  掩码也只给单日 `amount`。`simulate` 的作者拒绝为此多发一次投机性查询，是对的。
  裁决：**`charge` 保持纯函数，`adv20` / `range` 由引擎（Task 13）附列**，
  少列直接抛 —— 一个"没有 ADV20 就当冲击为 0"的兜底会让整条曲线少掉全部冲击成本，
  而它看起来完全正常。引擎侧的口径：
    · `adv20` = 信号日 T 往前 20 个交易日 `amount` 的均值，
      **剔除停牌日**（D9 的占位行 `vol=0`/`amount=0`，算进去会系统性低估流动性→高估冲击）；
    · `range` = 信号日 T 往前 20 个交易日 `(high − low)/pre_close` 的**均值**，同样剔停牌
      （占位行 `high==low==pre_close` 振幅为 0，算进去会把均值压低 → 低估成本；
      经 `query.get_bars` 取时这些行的价格是 NaN，`dropna()` 即可）。
      ★ 2026-08-22 修正：原口径是**执行日当天**的振幅 —— 在 τ 开盘成交的那一刻，
        当天的 high/low 还没走出来，那是前视。它只抬高成本、不改信号，所以不会直接画出
        假净值；但拿净额收益调参时这条路径可被利用（波动大的日子成本算得更准 =
        泄露了一点未来）。20 日均值是同一个量的可得估计，且更平滑：靠近 30bp 封顶的那批
        交易形态不变（均值单调平滑、恒 ≥ 0），换口径不动 `charge` 一行代码。
  为什么不在本文件里取数：computation 模块一旦 import query，`charge` 就再也不能
  被单测、被批量向量化、被闸 4 用不同费率重放同一批成交。

★【ADV20 / range 缺失或为 0 一律按封顶收费，不按 0 冲击】
  缺失的方向不是中性的：按 0 收 = "流动性未知 ⇒ 免费成交"，恰好错在把净值画好看那一侧；
  按上限收只会让结果更差，而且一定看得见（同 `get_tradable_mask` 的
  `limit_unknown → 两侧不可交易`：宁可保守，不可假设）。抛异常在这里是错的 ——
  一只上市不足 20 天、或长期停牌的票会炸掉整个 15 年的运行。
  `range == 0` 走的是**同一条**路：20 日振幅全为 0 意味着这 20 天每天 high==low，
  即连续一字板（或整段停牌被剔干净后无样本）—— 那是"根本成交不了"，不是"零波动
  所以零冲击"。旧实现的 `rng >= 0` 把它读成后者，恰好又倒向把净值画好看那一侧。

★【过户费双边，与 §5.4 的 c^b 表格不一致，是表格旧了】
  §5.4 把 c^b 写成 0.00025（只有佣金）、c^s 写成 0.00025+0.0005+0.00001，
  即过户费只在卖出侧 —— 那是 2022-04-29 之前"沪市、按面值"的旧规则
  （`CostConfig.transfer_bps` 的注释还写着"沪市"）。现行规则是**沪深双市、双向、
  成交金额的 0.001%**，正好等于 0.1bp。取现行规则：差 1bp，方向是把成本算得更足。
  brief 的验收断言（买入"佣金 250 元、无印花税"）不涉及过户费，两者不冲突。

★【先封顶、再乘 multiplier】
  闸 4（成本敏感性）把 `multiplier` 设成 2.0。若先乘后封，被封顶的那些笔在 2.0 下
  纹丝不动 —— 而被封顶的恰恰是最不流动、最该压力测试的那批交易，闸 4 对它们等于没跑。
  multiplier 乘在**每一个分量**上而不只是 total：明细列与 total 对不上账，
  下游按分量做的任何拆解（比如"印花税占多少"）都会静默错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import CostConfig

# 追加到 trade_rows 上的五列。顺序即报告顺序，`total_cost` 恒在最后。
COST_COLS = ["commission", "stamp_duty", "transfer_fee", "impact", "total_cost"]

# 引擎必须附上的两列（见模块头「谁来给 ADV20」）。
IMPACT_COLS = ("adv20", "range")

_BPS = 1e-4
_SIDES = ("BUY", "SELL")


def charge(trade_rows: pd.DataFrame, cost_cfg: CostConfig) -> "tuple[pd.DataFrame, list[str]]":
    """给每笔成交计费，返回 `(附了 COST_COLS 的副本, warnings)`。

    Args:
        trade_rows: `execution.simulate` 的 `trades`，**外加引擎附的 `adv20` / `range`**。
            必需列：`side`（'BUY'|'SELL'）、`amount`（成交金额，正数）、`adv20`、
            `range`（**信号日往前 20 日的平均振幅**，不是执行日当天的，见模块头）。
            其余列原样带过。
        cost_cfg: 费率。`multiplier` 是闸 4 的成本敏感倍数，乘在每一个分量上。

    Returns:
        `(DataFrame, warnings)` —— 与 `pipeline.process` / `build_targets` / `simulate`
        同惯例（global-constraints：返回类型必须留 warning 通道）。本函数唯一的降级是
        「ADV20/振幅缺失或为 0 → 按 30bp 封顶收费」，那一条必须汇进 `BacktestResult.warnings`。

    Raises:
        ValueError: 缺必需列、`side` 不是 BUY/SELL、`amount` 非正或非有限、`cost_cfg` 不合法。
            四者都是**契约被破坏**，不是数据缺口：静默跳过任何一条都会让
            成本系统性偏低，而净值曲线看起来毫无异常。
    """
    m = float(cost_cfg.multiplier)
    _rates = (cost_cfg.commission_bps, cost_cfg.stamp_duty_bps, cost_cfg.transfer_bps,
              cost_cfg.impact_coef, cost_cfg.impact_cap_bps)
    if not (m > 0 and min(_rates) >= 0):
        raise ValueError(
            f"CostConfig 不合法（multiplier={m!r}, 费率={_rates}）：`CostConfig` 是 frozen "
            f"dataclass 但不做任何校验，这道闸只能在这里补。multiplier=0 产出一张全零费用表 —— "
            f"一次「零成本」的回测在报告上只表现为「策略特别好」；负费率则变成交易返现，"
            f"同样不会抛、不会告警、只会把净值画得更漂亮")
    df = pd.DataFrame(trade_rows).copy()
    missing = [c for c in ("side", "amount", *IMPACT_COLS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"trade_rows 缺列 {missing}：`charge` 是纯函数，冲击成本要的 "
            f"{list(IMPACT_COLS)} 由引擎附上（adv20 = 信号日往前 20 个交易日的成交额均值；"
            f"range = 同一窗口 (high−low)/pre_close 的均值，**不是执行日当天的振幅**；"
            f"两者都剔停牌占位行）。缺列时按 0 计冲击会让整条净值曲线"
            f"少掉全部冲击成本，且看起来完全正常")

    if df.empty:                       # 无交易日也要带着列出去，下游 groupby 不该 KeyError
        for c in COST_COLS:
            df[c] = pd.Series(dtype=float)
        return df, []

    side = df["side"].astype(str)
    bad_side = sorted(set(side[~side.isin(_SIDES)]))
    if bad_side:
        raise ValueError(f"side 只能是 {list(_SIDES)}，收到 {bad_side[:5]}："
                         f"拼错的方向会静默跳过印花税，卖出成本凭空少 5bp")

    amount = df["amount"].astype(float).to_numpy()
    if not np.all(np.isfinite(amount) & (amount > 0)):
        raise ValueError("amount 必须是有限正数（成交金额是幅值，方向在 side 上）："
                         f"{list(df.index[~(np.isfinite(amount) & (amount > 0))][:5])}")

    is_sell = side.eq("SELL").to_numpy()
    adv20 = df["adv20"].astype(float).to_numpy()
    rng = df["range"].astype(float).to_numpy()

    # ── 冲击：min(coef × 委托额/ADV20 × 振幅, cap)，缺料按 cap（模块头 ★2）──
    #    `rng > 0` 而非 `>= 0`：振幅恰为 0 是「20 天连续一字板 / 无有效样本」，是不可成交，
    #    不是零波动零冲击。它和 NaN 走同一条封顶路径。
    cap = cost_cfg.impact_cap_bps * _BPS
    usable = np.isfinite(adv20) & (adv20 > 0) & np.isfinite(rng) & (rng > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = cost_cfg.impact_coef * (amount / np.where(usable, adv20, 1.0)) * np.where(usable, rng, 0.0)
    impact_rate = np.where(usable, np.minimum(rate, cap), cap)

    df["commission"] = amount * (cost_cfg.commission_bps * _BPS) * m
    df["stamp_duty"] = np.where(is_sell, amount * (cost_cfg.stamp_duty_bps * _BPS), 0.0) * m
    df["transfer_fee"] = amount * (cost_cfg.transfer_bps * _BPS) * m
    df["impact"] = amount * impact_rate * m
    # ★ skipna=False：默认的 sum 会把 NaN 分量当 0 加，于是「某一项算不出来」在总额上
    #   表现为「那一项恰好免费」。分量全由本函数产出，出现 NaN 一定是上面某处出了事，
    #   让它传染到 total_cost 才看得见。
    df["total_cost"] = df[COST_COLS[:-1]].sum(axis=1, skipna=False)

    warns: list = []
    if not usable.all():
        n = int((~usable).sum())
        warns.append(
            f"{n} 笔成交缺 ADV20 或 20 日平均振幅（或其为 0），冲击成本按封顶 "
            f"{cost_cfg.impact_cap_bps:.0f}bp 计（不按 0：流动性未知不等于免费成交）："
            f"{list(df.index[~usable][:5])}")
    return df, warns
