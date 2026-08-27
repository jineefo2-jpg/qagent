# P3 · 策略层与信号闭环 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通最后一公里 —— 每个周频调仓日收盘后产出一份**可人工执行的调仓清单**
（§6.3 JSON：目标仓位、逐单限价带、因子归因、双指纹），配上让清单三个月后仍然可信的
**持仓回写闭环**，以及全系统唯一一次的**样本外判定**（U6）。

**Architecture:** 宏观择时层（5 指标滚动分位 → 总仓位 π）+ 横截面选股层（复用 P2 的
`build_targets`，零新逻辑）+ 输出契约（`ashare/strategy/plan.py`，读因子库与掩码，
写库只经 CLI）。信号与持仓落 `ashare_derived.duckdb`（market 库会被 promote 原子替换，
用户持仓绝不能跟着换掉）。LLM 侧只新增一个只读工具 `get_signal_list`（架构 §6.2 预留位）。

**Tech Stack:** 与 P2 相同（Python 3.9.6 / DuckDB / pandas / numpy），零新增依赖。

## Global Constraints

- 铁律 D1–D9 与 P2 全部约束继续适用。P3 新增三条硬约束：
  - **`ashare/strategy/**` 纳入分层静态检查**：只准 import `ashare.data.query`、
    `ashare.factors`（读侧）、`ashare.backtest.{store,metrics,types,portfolio}`；
    禁 DML 字符串（写库只发生在 `python -m ashare.strategy.plan` 的 CLI `main` 里，
    与「信号写库由 CLI 触发，不在策略模块内」的架构裁决一致）。
  - **宏观层禁 ML**（规格 N5 原文），且**禁全样本分位数** —— 分位必须是滚动 5 年窗（前视）。
  - **信号 JSON 必须同时带 `param_hash` 与 `data_snapshot_id`**（架构 §4.4 第 2 条），
    生成期间 `snapshot_id(pin=True)` 钉住。
- A 股不接下单通道（CLAUDE.md 铁条）：本期一切产物到「清单 + 人工确认回写」为止，
  不出现任何指向 `brokers/` 的 import。
- 提交信息格式与分支惯例同 P2：`feat/ashare-p3-strategy-signals`，基线 = 当前 `main`。

## 前置状态（P1/P2 已交付，不要重做）

- 数据底座实运行：`data/ashare_market.duckdb`（2010 → 2026-08-22），每日增量定时器
  已上线（launchd 工作日 21:30，`scripts/ashare_daily_update.sh`）。
- 宏观层 5 指标的**原料全部在库**：`index_daily.pe_ttm`（000985.CSI）、`cn10y`、
  `m1_yoy/m2_yoy`、`tsf_stock_yoy`、`money_flow.hk_hold_ratio`、`daily_basic.circ_mv`、
  指数 close（MA200）。
- 选股层即 P2 的 `portfolio.build_targets`（top-N 等权 + 三约束 + 换手裁剪），本期只调用。
- 因子库 `factor_value`（2010–2019 周频）构建中；`store.read_current` 是唯一取数口。
- Agent 只读工具层与注册机制已就位（`ashare/agent_tools.py` + M1–M4 防线）。

## 文件结构

```
ashare/strategy/
├── __init__.py          # build_rebalance_plan 公开签名（架构 §4.4 占位兑现）
├── macro.py             # Task 1：5 指标 + score + π
└── plan.py              # Task 3/4：清单生成 + CLI（唯一写库点）
ashare/data/derived_schema.sql   # Task 0：v2 → v3 迁移（signal_plan / position_ledger）
scripts/ashare_daily_update.sh   # Task 6：追加因子/信号环节
tests/ashare/test_strategy_*.py
```

## 未决项裁决（开工前定死，不再讨论）

| # | 裁决 | 理由 |
|---|---|---|
| V1 | ~~落 derived 库~~ **修正（2026-08-27 Task 0 实施时）：新立第三库 `data/ashare_ledger.duckdb`**（带 market 同款 schema_version 守卫） | 初稿只排除了 market（promote 整体替换），漏看了 derived 的既有契约「缓存，可随时 rm 重算」（derived_schema.sql 头注释）—— 信号/持仓是**不可重算的用户资产**，放缓存库等于埋数据丢失雷。P1 欠账（规格 §6.4）随本库补上 |
| V2 | 分位 → 0/0.5/1 的切点定 **30% / 70%** | 规格只说「按分位映射」没给数。30/70 让中档（0.5）覆盖 40% 历史状态 —— 择时层的仓位波动被天然抑制，与「统计功效最低的模块拿最小的权力」同向 |
| V3 | `north_flow_60` 口径 = **Σ₆₀ Δ(hk_hold_ratio × circ_mv) / circ_mv**（全市场加总） | 库里存的是持股**比率**不是流量；用 60 日持股市值变化近似净流入。规格公式的字面直译 |
| V4 | 限价带用**原始价**（交易所真实价格），ATR20 同口径 | 用户按清单去券商 App 下单，后复权价没法输入。这是 D8「后复权唯一真值」的**既定豁免通道**（与涨跌停价同类：交易执行层用原始价），出口走 `get_tradable_mask` 的扩展列，不开新的 `adjust="none"` 后门 |
| V5 | π 的变化**不豁免换手预算** | 择时降仓本质是卖出，卖出就是换手就有成本；豁免它 = 择时层的成本被系统性低估 |
| V6 | nightly **只 build 当日/最近调仓日**的因子；历史因子矩阵在 promote 后按需一次性重建 | 每日 promote 换 snapshot 会让全历史 factor_value 判失效；每晚重建 521 个日期要数小时，不可持续。回测前显式 `build(overwrite=True)` 一次即可 |
| V7 | 样本外判定（策略 vs 固定 80% 仓位 + 五闸的闸 1）**合并为同一次运行** | D7 只给一次机会；拆两次跑 = 第二次已被第一次的结果污染 |
| V8 | 对账单 CSV 只支持「通用列名映射」不做券商方言适配 | 各券商导出格式无穷尽；映射配置一次写在导入界面里，比维护 N 个解析器便宜 |

## Task 0: derived 库 schema v3 —— signal_plan / position_ledger

**Files:** Modify `ashare/data/derived_schema.sql`、`ashare/data/_derived.py`；Create `tests/ashare/test_derived_v3.py`

- `signal_plan(as_of_date, execute_on, plan_json, param_hash, data_snapshot_id, strategy_version, created_at)`，主键 `(as_of_date, param_hash)`
- `position_ledger(as_of_date, ts_code, shares, avg_cost, source, created_at)`，主键 `(as_of_date, ts_code)`；`source ∈ {'reconcile_csv','manual_confirm','signal_assumed'}`
- `order_confirm(as_of_date, ts_code, state, filled_shares, note)`，`state ∈ {'filled','partial','skipped'}` —— 三态确认的持久层

**验收断言**：旧库自动迁移且原 `factor_value`/`backtest_run` 数据无损；schema_version 闸拒绝旧代码开新库。

## Task 1: 宏观择时层 `strategy/macro.py`

**Interfaces**
```python
def macro_indicators(as_of_date) -> pd.DataFrame   # index=month_end, cols=5 指标原值
def macro_score(as_of_date) -> dict                # {"scores": {指标: 0|0.5|1}, "score": float, "position": float}
```

**决策**
- 全指标 PIT：宏观经 `get_macro`（publish_date 过滤），指数/北向按交易日直取。
- 滚动 5 年分位窗不足 5 年（2010–2014 初段）→ 该指标当期记 0.5（中性），不缩窗 —— 缩窗分位在样本早期剧烈抖动。
- `position = 0.2 + 0.8 × score`；**禁 ML、禁全样本分位**（Global Constraints）。

**验收断言**
- 2015-06（顶部区）score 显著高于 2018-12（底部区）？**不许这么写** —— 那是拿已知行情反推指标。
  正确的反测：每个指标单独喂构造数据，验证分位、方向、切点、PIT（publish_date 晚于 as_of 的月份不可见）。
- 窗不足 5 年时返回 0.5 且 detail 注明 `window_short=True`。

## Task 2: 择时接入回测 `BacktestConfig.macro_timing`

**Files:** Modify `ashare/backtest/types.py`、`engine.py`（≤400 行红线内）

- `macro_timing: bool = False`（进 `param_hash` —— 它改变结果）。开启时每个调仓日
  `target_position = macro_score(T)["position"]`，关闭时维持现状。
- π 变化消耗换手预算（V5），`build_targets` 无需改动（`target_position` 本就是入参）。

**验收断言**：`macro_timing=False` 的 `param_hash` 与 P2 现状完全一致（老实验指纹不漂移）；
开启后一个构造的「score 恒 0」场景仓位恒 20%。

## Task 3: 清单生成 `strategy/plan.py::build_rebalance_plan`

**Interfaces**：`build_rebalance_plan(as_of_date, config) -> dict`，返回 = 规格 §6.3 全字段 + `data_snapshot_id`。

**决策**
- 因子分数走 `store.read_current`，miss → 现算（与引擎 use_store 同语义）；
  `factor_contrib` 取该股 |z×w| 前 3 的因子。
- `limit_price_range = 原始前收 ± 0.5 × ATR20(原始价)`（V4），执行时点文案写死
  「T+1 09:15–09:25 集合竞价挂限价单」（D6）。
- `current_weight` 读 `position_ledger` 最新期；无记录 → 全体标注
  `"position_calibrated": false` 且顶部警示语（规格 §6.4：不得静默按虚拟持仓推演）。

**验收断言**：JSON 可序列化、双指纹在场且与 config/库一致；`orders` 权重满足三约束；
剔除「次日预期一字涨停」标的（复用 `get_tradable_mask(execute_on)`）；
未校准场景警示语在场。

## Task 4: 信号 CLI 与落库

**Files:** `python -m ashare.strategy.plan --as-of 2026-08-28 [--config ...]`

- CLI 是**唯一写库点**（signal_plan 表 + `out/signals/{date}.json` 导出）；模块层零 DML。
- 重复生成同 `(as_of, param_hash)` → 幂等覆盖并警示（参数变了则是新行，D7 台账连续）。

## Task 5: 持仓回写闭环（server 侧）

**Files:** Modify `server.py`（`/api/portfolio/reconcile`、`/api/signals/*`、三态确认）；`static/` 最小看板

- 三个只读 GET（最新清单 / 历史清单 / 台账）+ 两个写 POST（CSV 对账导入、单笔三态确认）。
  全部在 owner-lock 中间件之后，写路径复用 `require_user`。
- CSV 列映射在前端配置（V8），导入落 `position_ledger(source='reconcile_csv')`。
- 前端：信号看板（清单表格 + 每笔三态勾选 + 未校准横幅）。**不做**任何「一键执行」。

## Task 6: 定时链条扩展

**Files:** Modify `scripts/ashare_daily_update.sh`

pipeline daily 成功后追加两步（各自失败独立通知、不互相阻塞）：
1. 当日为周频调仓日 → `store.build(alphas, [当日])`（V6：只 build 这一天）；
2. 接着 `python -m ashare.strategy.plan --as-of 当日` 生成下周清单。

**验收断言**：非调仓日两步都跳过（日志留痕）；build 失败时 plan 不跑（现算兜底会把 21:30 的窗口拖爆）。

## Task 7: Agent 工具 `get_signal_list` + 前端入口

- `ashare/agent_tools.py` 注册第 3 个只读工具：读 `signal_plan` 最新行的精简版（<3KB，
  含未校准警示原文）。描述明确「只读，清单需人工执行」。M1–M4 测试延伸覆盖。
- **工具描述里不出现"建议买入"式措辞** —— 它只是转述清单（T3 延伸）。

## Task 8: 择时价值判定（样本内）+ 关停机制

- 2010–2019：`macro_timing=True` vs 恒 0.8 仓位，两次运行对比（Sharpe/Calmar/MDD）。
  这**不是**关停判定（那属样本外），只为发现实现级 bug（如 π 序列常数、成本爆炸）。
- 实现关停开关：`strategy_config.macro_disabled=True` 时 π≡0.8，看板仍显示 5 项分位（规格 §6.1）。

> **Task 8 实测记录（2026-08-27，样本内 2010–2019，含成本，净值-only）**：
> 恒定 0.8 仓位：年化 10.23% / Sharpe 0.489 / Calmar 0.273 / MDD 37.4%。
> 宏观择时 π∈[0.2,1.0]：年化 7.60% / Sharpe 0.484 / Calmar 0.260 / **MDD 29.2%**。
> 择时把最大回撤压掉 8 个百分点（它的本职），但代价是年化 −2.6pp，Sharpe/Calmar
> 双双**未超过**恒定仓位 —— 按 §6.1 关停规则的口径，样本内预演指向「关停」。
> 结构性因素：2010–2014 五个指标分位窗全部不足（滚动 5 年窗的诚实代价），
> 择时层前半程恒为中性 0.6 仓。**不做任何样本内调参救它 —— 那是自欺的入口；
> 正式判定按 V7 在样本外与闸 1 同跑一次。** 宏观逐期开销实测全窗 +4s（缓存生效）。
> 关停机制的实现裁决：不新增 macro_disabled 开关 —— `macro_timing=False +
> position_cap=0.8` 的组合就是关停态（少一个与既有语义重复的旋钮）；
> 看板的 5 项分位展示走 macro_indicators，与开关无关。

## Task 9: 样本外首跑（U6 —— 全系统唯一一次，烧前清单）

**前置全部满足才许跑，任何一项欠费即停：**
1. 五闸对终版参数全绿（闸 2/5 在 P2 是「实现但未作为门槛」—— 此处必须真跑真过）；
2. 因子库覆盖 2020 → 最新（一次性 `build(overwrite=True)`，V6 的「按需重建」时刻）;
3. `docs/oos-runs.md` 现状为空已核对；
4. **用户明示批准**（这一步烧掉的机会不可再生）。

**执行**：一次运行同时产出（V7）：闸 1 判定（`append_oos_run(sharpe_is=...)` 成对写入）、
宏观关停判定（vs 固定 0.8）。结果无论好坏原样入台账 —— 台账只写事实，不写辩解。

---

## 自查

**规格覆盖**：§6.1 宏观层 → Task 1/2/8；§6.2 选股层 → 复用 P2（零任务）；
§6.3 输出契约 → Task 3/4；§6.4 回写闭环 → Task 0/5；架构 §4.4 → Task 3；
架构 §6.2 预留工具位 → Task 7；U6 → Task 9；D7 → V7 + Task 9。

**性能余账（2026-08-27 专项后遗留，随 P3 工程项处理）**：60s 闸实测 86.8s（起点 >600s；
use_store 接线 / 预加载区段索引 / 双备忘录 / 整窗批量读 / 宽面板五批优化全部逐位校验通过）。
剩余缺口集中在指标/诊断层的逐期 pandas 运算（IC 逐期 spearman、Brinson 逐期回归），
被大量裁决测试钉着，动刀须单独立项。净值-only 全窗 67.7s → 闸 3 的 200 次 ≈ 3.8 小时，
五闸已实际可跑 —— 预算存在的目的已达成。验收测试以 xfail(strict) 入档，达标自动翻红逼摘牌。

**已知缺口（有意不做）**：LP 组合优化（U1 维持：先证明等权不够用 —— 样本外之后才有资格谈）；
IC 加权（U2 维持）；券商 API 自动回写（N2：不接下单通道，对账走 CSV/人工）；
宏观层 ML（N5 禁令）。

**类型一致性**：`build_rebalance_plan` 只消费 P2 既有类型（`BacktestConfig`/`PortfolioConstraints`）；
`macro_score` 返回 dict（进 JSON）；三张新表只被 `_derived.py` 与 CLI/API 触达。
