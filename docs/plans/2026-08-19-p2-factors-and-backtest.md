# P2 · 因子库 + 回测引擎 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成一个可复现、防前视/幸存者偏差的**周频横截面回测引擎**与 16 个因子的因子库，且用已知 A 股异象反测引擎正确性。

**Architecture:** 因子是纯函数 `f(as_of_date, universe) -> Series`，只经 `ashare.data.query` 取数（P1 已建，L1–L6 静态守卫）。处理链固定为 MAD 去极值 → 行业+市值 OLS 中性化 → zscore → 填 0（为什么 OLS 不是 WLS：见算法说明书 §3.2）。回测按周调仓：T 日收盘算信号、T+1 开盘价成交、涨跌停/停牌不可交易。因子值预落库到 `data/ashare_derived.duckdb`，否则五闸（200 次 shuffle × 参数网格）跑不动。

**Tech Stack:** Python 3.9.6、DuckDB、pandas、numpy、statsmodels（仅 Newey-West）、pytest

## Global Constraints

- Python **3.9.6**。`ashare/**` 每个文件首行 `from __future__ import annotations`。
- 铁律 D1–D9 见 `CLAUDE.md`「A 股数据与回测铁律」，全部适用。P2 新增三条硬约束：
  - **因子只能经 `ashare.data.query` 的公开函数取数**。L5 禁 `adjust="none"`（后复权是唯一真值），
    L6 禁从 `ashare.data.*` 导入下划线私有名（`_PRELOAD` 里是含 `limit_up` 与未掩码停牌行的原始数据）。
  - **行业中性化前必须查 `_meta.industry_source`**：不是 `'sw'` 说明行业是「今天的值回填到上市日」，
    做行业中性化就是前视污染，必须拒绝（P1 的 `--allow-static-industry` 只放行建库，不放行中性化）。
  - **回测入口必须 `query.snapshot_id(pin=True)`**：钉住后 promote 换库会抛而不是静默重连，
    否则一次运行横跨两个数据库却只记录一个 `data_snapshot_id`（D7 失效）。
- 因子函数签名固定 `f(as_of_date, universe, *, <keyword-only>)`，由 `check_ashare_layering.py` L3 强制。
- 新增依赖：`statsmodels>=0.14`（只用 `cov_type='HAC'` 算 Newey-West 标准误），走仓库既有的
  `try: import X except ImportError` 可选依赖模式。
- 提交信息格式：`feat(ashare): ...` / `test(ashare): ...` / `fix(ashare): ...`
- 分支：`feat/ashare-p2-factors-backtest`，基线 = 当前 `main`。

## 前置状态（P1 已交付，不要重做）

`ashare/data/query.py` 的公开取数函数（首参一律 `as_of_date`，唯一豁免 `get_tradable_mask(exec_date, ...)`）：

| 函数 | 返回 |
|---|---|
| `get_universe(as_of_date, *, min_list_days=250, exclude_st=True, exclude_suspended=True, liquidity_drop_pct=0.20, markets=None)` | `list[str]` 升序 |
| `get_trade_dates(as_of_date, *, start=None, freq='D'\|'W'\|'M')` | `list[date]`；**`'W'` 是 weekly_dates 唯一实现点** |
| `get_price_panel(as_of_date, ts_codes, field='close', lookback=250, adjust='hfq')` | 宽表 index=trade_date, columns=ts_code，停牌为 NaN 不 ffill |
| `get_bars(as_of_date, ts_codes, *, lookback=None, start=None, fields=(...), adjust='hfq')` | 长表 MultiIndex (ts_code, trade_date)，恒带 `is_suspended`，**不返回 limit_up/limit_down** |
| `get_daily_basic(as_of_date, ts_codes, fields=(...), lookback=1)` | lookback=1 → index=ts_code |
| `get_financial(as_of_date, ts_codes, fields, *, n_periods=1, include_restated=False, report_type='1')` | PIT；附 `ann_date/end_date/report_type/lag_days` |
| `get_financial_ttm(as_of_date, ts_codes, field)` | Series；流量科目走累计口径拼接，存量科目期初期末均值，缺期 NaN |
| `get_industry(as_of_date, ts_codes=None, level='l1', *, min_members=5)` | Series；成分 < 5 的行业并入 `__OTHER__` |
| `get_macro(as_of_date, indicators, lookback_periods=60)` | index=period；附 `<ind>__publish_date` |
| `get_money_flow(as_of_date, ts_codes, fields=('hk_hold_ratio',), lookback=20)` | MultiIndex；2016-12 前 NaN 不填 0 |
| `get_index_bars(as_of_date, index_code, lookback=250, fields=('close','pe_ttm'))` | index=trade_date |
| `get_tradable_mask(exec_date, ts_codes)` | index=ts_code；`can_buy/can_sell/reason/open_hfq/close_hfq/amount/amplitude` |
| `snapshot_id(*, pin=False)` / `preload(start, end, tables)` / `clear_preload()` | — |

---

## 文件结构

```
ashare/
├── factors/
│   ├── __init__.py
│   ├── base.py          @factor 装饰器 + FactorSpec + FACTOR_REGISTRY + compute_factor/panel/combine
│   ├── pipeline.py      winsorize_mad / neutralize(OLS) / zscore / process —— 顺序固定不可调换
│   ├── store.py         factor_value 预落库（唯一写 derived 库的因子模块）
│   ├── price.py         6 个量价因子
│   ├── fundamental.py   8 个基本面因子
│   ├── flow.py          1 个资金因子
│   └── risk.py          3 个风险因子（中性化用，不作 alpha）
├── backtest/
│   ├── __init__.py
│   ├── types.py         CostConfig / PortfolioConstraints / BacktestConfig / BacktestResult
│   ├── portfolio.py     build_targets —— top_n / 权重 / 三条约束 / 换手裁剪
│   ├── execution.py     simulate —— 掩码 + 开盘价 + 不可交易时的权重再分配
│   ├── cost.py          charge —— 佣金/印花税/过户费/冲击成本
│   ├── metrics.py       compute —— 净值/相对/因子/交易/归因
│   ├── engine.py        run_backtest（≤ 400 行，只做编排）
│   ├── store.py         BacktestResult 持久化 + docs/oos-runs.md 自动追加
│   └── guards.py        防自欺五闸
└── data/_derived.py     derived 库连接与 schema（factor_value / backtest_run）

tests/ashare/
├── test_factor_base.py      注册表 / param_hash / available_from / min_coverage 重归一
├── test_factor_pipeline.py  MAD / OLS 中性化 / zscore / 秩亏降级 / 行业来源拒绝
├── test_factors_price.py    6 个量价因子的数值断言（构造已知序列）
├── test_factors_fundamental.py  8 个基本面因子（PIT + 累计口径）
├── test_factor_store.py     落库/读取/快照失效
├── test_portfolio.py        约束满足 + 换手裁剪
├── test_execution.py        D6 三情形 + 权重再分配 + 退市清仓
├── test_cost_metrics.py     成本公式 + 指标定义 + Newey-West
├── test_engine.py           时序语义（T 收盘算/T+1 开盘成交）+ 快照钉住
├── test_guards.py           五闸（shuffle 只在同日横截面内打乱）
└── test_p2_acceptance.py    §5.6 已知异象反测（真库，无库 skip）
```

---

## 未决项裁决（开工前定死，不再讨论）

| # | 裁决 | 理由 |
|---|---|---|
| U1 | 组合构建先做**等权 top-N + 行业上限裁剪**，不上 LP | 780 次求解 × 数千变量 + 一个求解器依赖。先证明等权不够用（规格 §6.2） |
| U2 | 因子合成默认**等权**，不做 IC 加权 | `Σ_IC⁻¹ 求逆放大噪声，M=16/T=780 样本外劣于等权（算法说明书 §6） |
| U3 | 宏观择时层**本期不做**（属 P3），`BacktestConfig.macro_timing` 默认 `False`，仓位恒为 `position_cap` | P2 的任务是验证引擎与因子，不是验证择时 |
| U4 | 因子值落库到 `data/ashare_derived.duckdb`，与 market 库分文件 | market 库由 promote 原子替换，derived 跟着换会丢因子；且 derived 只有 P2 写 |
| U5 | 五闸的闸 2（walk-forward）与闸 5（参数高原）**本期实现但不作为交付门槛** | 规格 §12.4 允许压缩；未跑的闸必须写进 `docs/oos-runs.md`，否则自欺 |
| U6 | 样本外区间**本期一次都不跑** | D7：样本外只跑一次。P2 只验证引擎正确性（2010–2019 样本内），跑样本外是 P3 的事 |

---

### Task 1: derived 库 schema + 连接

**Files:** Create `ashare/data/_derived.py`, `ashare/data/derived_schema.sql`; Create `tests/ashare/test_derived_schema.py`
（`.gitignore` 无需改动：P1 的 `/data/` 与 `data/ashare_derived.duckdb*` 已覆盖，加个断言钉住即可）

**Interfaces**
- Produces：`_derived.connect_write(path=None)` / `connect_read(path=None)` / `init_schema(conn)` / `DEFAULT_DERIVED_PATH`

**决策**
- 两张表：
  ```sql
  factor_value(factor_name, param_hash, trade_date, ts_code,
               raw_value DOUBLE, processed_value DOUBLE, snapshot_id VARCHAR,
               PRIMARY KEY (factor_name, param_hash, trade_date, ts_code))
  backtest_run(run_id VARCHAR PRIMARY KEY, param_hash, data_snapshot_id, engine_version,
               started_at TIMESTAMP, elapsed_sec DOUBLE, config_json VARCHAR,
               metrics_json VARCHAR, is_oos BOOLEAN)
  ```
- **`snapshot_id` 是列不是主键的一部分**：重算即覆盖，`snapshot_id` 列记录"这行是哪批数据算出来的"。
  理由：`factor_value` 是**缓存**，真相来源是 market 库。缓存要的是有效性键不是世代历史；
  含进主键会让表随每次 promote 成倍膨胀（16 因子 × 780 周 × 5000 股 已经 6200 万行）。
  D7 的可复现性由「恢复 `.bak.<snapshot_id>` 备份 + 重算因子」保证，不靠缓存留档。
  **代价（Task 8 必须承接）**：`read()` 必须校验行上的 `snapshot_id` 等于当前 `query.snapshot_id()`，
  不等就当未命中 —— 否则会静默把另一批数据算出的因子值喂进回测。
- 与 market 库分文件：market 由 promote 原子替换，derived 不能跟着被换掉。

**验收断言**
- 建表幂等；`factor_value` 主键不含 snapshot_id（同因子同参同日同股在不同快照下**覆盖**，靠 `snapshot_id` 列记录来源）
- 只读连接上 DML 抛 `duckdb.InvalidInputException`
- `data/ashare_derived.duckdb` 被 `.gitignore` 覆盖（断言 `git check-ignore` 命中，防止有人日后放开 `/data/`）

---

### Task 2: @factor 装饰器 + FactorSpec + 注册表

**Files:** Create `ashare/factors/__init__.py`, `ashare/factors/base.py`; Create `tests/ashare/test_factor_base.py`

**Interfaces**
- Produces：
  ```python
  @dataclass(frozen=True)
  class FactorSpec:
      name: str; fn: Callable; direction: int; category: str; lookback_days: int
      neutralize: bool = True; available_from: date | None = None
      min_coverage: float = 0.60; default_params: Mapping[str, Any] = {}
      def param_hash(self, **override) -> str    # sha256(name + canonical_json(default|override))[:12]
  FACTOR_REGISTRY: dict[str, FactorSpec]
  def factor(*, name, direction, category, lookback_days, neutralize=True,
             available_from=None, min_coverage=0.60, **default_params) -> Callable
  def get_factor(name) -> FactorSpec
  def list_factors(category=None) -> list[FactorSpec]
  ```

**决策**
- 用装饰器不用抽象基类（架构 §4.2）：D2 要的签名字面量就是 `f(as_of_date, universe)`，类方法多出 `self` 会让静态检查要特判；16 个因子没有一个需要继承共享状态。
- 重名注册直接 `raise`，不允许静默覆盖。
- `param_hash` 用 canonical JSON（`sort_keys=True, separators=(',',':')`，date 转 isoformat），保证同参数不同书写顺序哈希相同。

**验收断言**
- `@factor(name='x', ...)` 后 `get_factor('x').fn is 原函数`；重名 raise
- `spec.param_hash()` 与 `spec.param_hash(window=20)`（等于默认值）相同；`param_hash(window=10)` 不同
- `param_hash` 对 dict 键序不敏感
- `list_factors('price')` 只返回 price 类
- 装饰器不改变函数本身的可调用性（`fn(as_of, universe)` 仍能直接调）

---

### Task 3: 处理链 —— MAD 去极值 / OLS 中性化 / zscore

**Files:** Create `ashare/factors/pipeline.py`; Create `tests/ashare/test_factor_pipeline.py`

**Interfaces**
- Consumes：`query.get_daily_basic`（`total_mv`）、`query.get_industry`
- Produces：
  ```python
  def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series
  def neutralize(s, as_of_date, universe, *, by=("log_mv", "industry")) -> tuple[pd.Series, list[str]]
  def zscore(s: pd.Series) -> pd.Series
  def process(s, as_of_date, universe, *, spec: FactorSpec) -> tuple[pd.Series, list[str]]
  ```
  （中性化与 process 返回 `(series, warnings)`：秩亏 / 样本不足 / 行业来源不可用都要能上浮到 `BacktestResult.warnings`）

**决策（算法说明书 §3，顺序不可调换）**
1. MAD：`m ± 3 × 1.4826 × MAD` 截断。**不用 3σ** —— A 股横截面尾部太厚，均值与标准差本身已被极值污染。
2. 中性化：对 `[log_mv, 行业 dummy]` 做**横截面 OLS**，取残差。
   - **必须 OLS 不能 WLS**（2026-08-20 裁决，推翻初稿；理由见算法说明书 §3.2）：组合是等权 top-N，
     相关的度量是【无权】正交 = OLS 的恒等式 `X'e = 0`；WLS 只在 sqrt(MV) 内积下正交，
     等权组合会带上规模倾斜。Barra 的 WLS 解的是"风险模型里估因子收益"，是另一个问题。
   - 行业 dummy 去掉一列避免与截距完全共线。
   - **行业中性化前查 `_meta.industry_source`**：不是 `'sw'` 抛 `RuntimeError`（行业是今天的值回填到上市日 → 前视）。
   - 有效样本 < 30 或设计矩阵秩亏 → **返回原 Series 并记 warning**，不静默返回 NaN。
3. zscore：`(x - mean) / std`。
4. `fillna(0)` —— **在 zscore 之后**，中性化后 0 = 行业内平均水平。

**验收断言**
- 构造含 1 个极端值的序列，MAD 截断后该值等于上界；同数据下 3σ 截不干净（对比断言，钉住"为什么不用 3σ"）
- 中性化后残差与 `log_mv` 的【无权】相关系数 ≈ 0（|r| < 1e-10）；各行业残差【无权】均值 ≈ 0
- **OLS ≠ WLS 的钉子（双向）**：因子值必须是 log 市值的【非线性】函数 —— 严格线性时两种估计
  残差都恒为 0，用例退化成永远通过。断言 ①OLS 无权相关 < 1e-10 而 WLS 反事实 > 0.05；
  ②同时钉住代价：OLS 的大盘组确有偏移，别假装没有
- `industry_source='tushare_static'` → `neutralize(by=(...,'industry'))` 抛 `RuntimeError`；`by=('log_mv',)` 仍可用
- 样本 < 30 → 返回原值 + warning 非空
- `process` 的四步顺序：注入一个中间断言（zscore 前不得有 fillna(0)）—— 用带 NaN 的输入，断言 NaN 在中性化阶段仍是 NaN

---

### Task 4: 量价因子（6 个）

**Files:** Create `ashare/factors/price.py`; Create `tests/ashare/test_factors_price.py`

**Interfaces**
- Consumes：`query.get_price_panel`、`query.get_bars`、`query.get_daily_basic`
- Produces（全部 `f(as_of_date, universe, *, window=...) -> pd.Series`，index=ts_code）：

| 因子 | 公式（算法说明书 §2.1） | direction | lookback_days |
|---|---|---|---|
| `reversal_20` | `-(p_t / p_{t-20} - 1)` | +1 | 30 |
| `momentum_120_20` | `p_{t-20} / p_{t-120} - 1`（区间是 [t-120, t-20]，**不是两段收益相减**） | +1 | 130 |
| `volatility_60` | 60 日对数收益标准差 | −1 | 70 |
| `turnover_20` | 20 日平均换手率（`turnover_rate_f`） | −1 | 30 |
| `amihud_20` | `1e9/20 × Σ |r_s| / amount_s` | +1 | 30 |
| `max_ret_20` | 20 日内单日最大涨幅 | −1 | 30 |

**决策**
- 一律用后复权价（`get_price_panel` 默认 `adjust='hfq'`）；停牌日为 NaN。
- **收益率计算前必须 ffill 价格面板**（停牌日 NaN 会让 `pct_change` 产生 NaN 传染），但 ffill 只在因子内部做、
  且 `min_periods` 保证窗口内非空天数 ≥ 窗口的 60%，否则该股该日置 NaN。
- 对数收益 `ln(p_t/p_{t-1})` 用于波动率/Amihud；简单收益用于反转/动量/max_ret（与文献口径一致）。

**验收断言**
- 每个因子用**构造的已知序列**做数值断言（如线性上涨 1%/日 20 天 → `reversal_20 = -(1.01^20 - 1)`）
- `momentum_120_20` 的区间断言：只有 [t-120, t-20] 的价格影响结果，t-19..t 的价格改变不改变因子值
- 停牌股：窗口内非空天数不足 60% → NaN，不是用少量样本硬算
- 所有因子函数前两个位置参数是 `(as_of_date, universe)`（L3 静态检查覆盖，此处再断言一次签名）
- 返回 Series 的 index ⊆ universe

---

### Task 5: 基本面因子（8 个）

**Files:** Create `ashare/factors/fundamental.py`; Create `tests/ashare/test_factors_fundamental.py`

**Interfaces**
- Consumes：`query.get_financial`、`query.get_financial_ttm`、`query.get_daily_basic`
- Produces：

| 因子 | 公式 | direction |
|---|---|---|
| `ep_ttm` | `TTM(归母净利) / 总市值` | +1 |
| `bp` | `归母净资产 / 总市值` | +1 |
| `sp_ttm` | `TTM(营收) / 总市值` | +1 |
| `roe_ttm` | `TTM(归母净利) / 期初期末平均净资产` | +1 |
| `gross_margin` | 毛利率 | +1 |
| `accrual` | `(TTM(净利) − TTM(经营现金流)) / 总资产` | −1 |
| `np_yoy` | `NI^Q_e / NI^Q_{e-4} − 1` | +1 |
| `sue` | `(NI^Q_e − NI^Q_{e-4}) / σ(过去 8 期同比【差额】)` | +1 |  ← 2026-08-20 修正：本表初稿写「增速」错，以算法说明书 §2.2 为准

**决策（算法说明书 §1.2 / §2.2）**
- **一律走 `get_financial_ttm`**，不在因子层自己拼 TTM —— A 股财报是累计口径，拼接逻辑放因子层会被 4 个因子各抄一遍。
- **分子取 PIT 财报、分母取 as_of 当日市值**，两者时点必须一致。
- `np_yoy` 分母 ≤ 0 → NaN（A 股大量扭亏样本会产生 ±∞）。
- `sue` 需至少 12 期【单季】数据 → 累计口径要取 **13** 期（Q1 除外，单季 = 相邻累计之差）。写死 12 会在一年里九个月对所有股票返回 NaN，只在年报季看起来正常。
- 单季值 = 累计值差分，**跨年重置**（Q1 单季 = Q1 累计）—— 这层在 query 的 TTM 里已实现，因子层用
  `get_financial(n_periods=N)` 拿多期累计值时必须自己差分，同样跨年重置。

**验收断言**
- ~~用真实茅台的 PIT 断言（`as_of='2021-04-01'` → 用 2020 年报）~~
  **2026-08-20 撤销：这条断言的事实是错的。** 茅台 FY2020 年报 4 月下旬才披露，
  2021-04-01 时 `e*` 还停在 2020 三季报。改为不写死日期的 PIT 不变量断言：
  `ann_date <= as_of` 恒成立，且年报披露后 `e*` 前移一期 —— 这样换数据快照也不会假失败。
- 跨年重置：构造 Q1/H1/Q3/FY 累计值，断言 `np_yoy` 在 Q1 后用的是单季而非累计
- 分母为负 → NaN（不是 ±∞）
- `ep_ttm` 与 `1/pe_ttm`（`get_daily_basic`）在同一日的相关性 > 0.95（交叉校验两条取数路径）
- 期数不足 → NaN，不外推

---

### Task 6: 资金 + 风险因子

**Files:** Create `ashare/factors/flow.py`, `ashare/factors/risk.py`; Modify `ashare/factors/__init__.py`; Modify `tests/ashare/test_factors_price.py`（追加）

**★ 本任务收口：`__init__.py` 必须 import 四个因子模块**（price/fundamental/flow/risk）。
装饰器只在模块被导入时才注册；不导入的话 `import ashare.factors` 后 `FACTOR_REGISTRY` 是空的，
而**空注册表是静默失败**——回测会一路跑到"没有因子可用"才报，或者更糟：合成分数全 NaN、
`build_targets` 返回空、净值一条直线，看起来像"策略没信号"而不是"代码没装配"。
加一条断言：`len(FACTOR_REGISTRY) == 18` 且四个 category 各自非空。

**Interfaces**
- Produces：
  - flow：`north_hold_chg_20`（北向持股比例 20 日变化，direction=+1，`available_from=date(2016,12,5)`）
  - risk（`neutralize=False`，**不作 alpha**）：`log_mv`（ln 总市值）、`industry`（申万一级，返回 category Series）、`beta_250`（对中证全指 250 日 beta）

**决策**
- `north_hold_chg_20` 的 `available_from=2016-12-05`：早于该日 `compute_factor` 直接返回全 NaN Series，
  **不填 0** —— 填 0 会让 2010–2016 的合成分数被静默降权（分母算了它、分子恒 0）。
- `industry` 不是数值因子，只供 `pipeline.neutralize` 用；不进 `combine`。
- `beta_250` 用 `get_index_bars('000985.CSI')`；窗口内非空 < 150 日 → NaN。
- 规格 A1 已砍 `margin_chg_20`（服务于一个规格自标"待检验"的因子），**本期不做**。

**验收断言**
- `compute_factor('north_hold_chg_20', '2015-06-12', u)` 返回全 NaN 且不抛
- `log_mv` / `beta_250` 的 `spec.neutralize is False`
- `beta_250`：构造与指数完全同步的股票 → beta ≈ 1；2 倍波动 → beta ≈ 2
- **注册表装配断言**：`import ashare.factors` 后 `len(FACTOR_REGISTRY) == 18`；
  `list_factors('price')/('fundamental')/('flow')/('risk')` 各为 6/8/1/3 个

---

> **★ 2026-08-20 追加（Task 4 评审转来）：`compute_factor` 是校验 universe 的唯一入口。**
> 现在 6 个量价因子对「universe 含重复代码」的行为不一致：2 个在 `unstack` 处抛
> `ValueError`，4 个静默返回带重复索引的 Series，而 `pipeline.process` 的横截面回归
> 会把重复项当两只股票加权两次。`get_universe` 返回 sorted unique 所以现在不可达，
> 但守卫要放在这里 —— **一处覆盖 18 个因子**，而不是在每个因子里各写一遍。
> 同时校验：非空、无 NaN 代码、类型为 str。

> **★ 2026-08-20 追加（Task 6 转来）：`combine` 必须用【白名单】排除风险因子，不是黑名单。**
> `FactorSpec` 没有 `is_alpha` 字段，而 `category` 已经携带了完全相同的信息 ——
> 不加第二个字段（两个真相来源会打架）。但判据要**失败关闭**：
> `combine` 只接受 `category ∈ {'price','fundamental','flow'}`，其余一律拒绝并报错。
> 黑名单（`category != 'risk'`）是失败**开放**的：将来新增一个类别、或者类别名打错一个字母，
> 都会静默变成 alpha。
> **为什么这条不能漏**：`industry` 自带保护（category dtype 让 `winsorize_mad` 的
> `median()` 抛 TypeError），但 `log_mv` 和 `beta_250` 会一路顺利穿过 `pipeline.process`
> 产出看起来完全正常的分数。而拿 `log_mv` 当 alpha 就是一个纯粹的规模押注 ——
> 在 A 股历史回测里它会**非常好看**（小盘溢价），这正是它最危险的地方。
>
> 另：Task 6 的装配断言（`len(FACTOR_REGISTRY) == 18`）是在因子函数层验的，
> 因为 `compute_factor` 当时还不存在。Task 7 要**两条路径都断言** ——
> `compute_factor` 的短路会遮住 `store.build` 与直接调 `spec.fn` 的调用方走的那条路。

### Task 7: compute_factor / compute_panel / combine

**Files:** Modify `ashare/factors/base.py`; Modify `tests/ashare/test_factor_base.py`

**Interfaces**
- Produces：
  ```python
  def compute_factor(name, as_of_date, universe, *, processed=True, **param_override) -> pd.Series
  def compute_panel(names, as_of_date, universe, *, processed=True) -> pd.DataFrame
  def combine(weights: Mapping[str, float], as_of_date, universe) -> pd.Series
  ```

**决策**
- `compute_factor`：`available_from` 之前直接返回全 NaN（不调 fn，省一次取数）；`processed=True` 走 `pipeline.process`。
- `combine`：`Σ wᵢ × directionᵢ × processedᵢ`。
  **★ 覆盖率不足（非空占比 < `min_coverage`）或 `available_from` 未到的因子，从当日分母中【剔除】并按剩余因子重新归一**，
  而不是当 0 参与 —— 当 0 参与等于静默降权，会让 2017 年前的合成分数悄悄变味（架构 B5）。
- 全部因子都不可用 → 返回全 NaN Series 并记 warning，不抛（回测该日跳过调仓）。

**验收断言**
- 2015 年（北向不可得）合成分数 == 只用其余因子等权的分数（**不是** 15/16 缩放）
- 某因子覆盖率 50% < min_coverage 0.6 → 从分母剔除
- `direction=-1` 的因子在合成里符号翻转
- `compute_panel` 的列顺序 == 传入的 names 顺序
- 权重非等值时按权重加权（`{a:2, b:1}` → a 的贡献是 b 的两倍）

---

### Task 8: 因子落库 store

> **★ 2026-08-20 裁决：本任务的文件改为 `ashare/data/derived_store.py`。**
> 原定的 `ashare/factors/store.py` 要 `import duckdb`，撞分层闸 L1（只有 `ashare/data/**` 可以）。
> 不放宽 L1 —— 闸是故意粗粒度的，一旦 factors 层能 import duckdb，它同样能连 market.duckdb
> 读未掩码的原始行。派生库读写（因子值 + 回测运行记录）统一归这一个公开模块，
> 只收发 DataFrame 与基础类型，绝不反向 import `ashare.backtest` / `ashare.factors`。
> 不再包一层转发：单实现的抽象没有存在意义。详见架构 §4.3 的裁决框。

> **★ 2026-08-21 修正落点：拆成两个文件。我先前那条裁决自相矛盾。**
> 我说过「派生库读写统一归 `ashare/data/derived_store.py`，**绝不反向 import
> `ashare.factors`**」，但 `build` 必须调 `compute_panel` —— 两句话不能同时成立。
> 分辨清楚：要避免的是**数据层 import 高层类型**（如 `BacktestResult`）；
> 而 `build` 是「算 → 写」的编排，本身是真逻辑，不是转发。
>
> · `ashare/data/derived_store.py`（L1）—— `write_factor_values(df)` /
>   `read_factor_values(...)` / `coverage_report(...)`。独家持有 duckdb，
>   只收发 DataFrame 与基础类型，**不 import `ashare.factors` / `ashare.backtest`**。
> · `ashare/factors/store.py`（L3）—— `build(names, dates, ...)`：调 `compute_panel`
>   算，调 `derived_store.write_factor_values` 写。**不 import duckdb**，所以 L1 通过。
>
> 我先前否掉「包一层转发」是对的 —— 但这不是转发。

> **★ 2026-08-21 追加（Task 6 评审转来）：`build` 不能无脑遍历 `FACTOR_REGISTRY`。**
> `factor_value.raw_value` 是 `DOUBLE`（`derived_schema.sql:19`），而注册表里的
> `industry` 返回的是 **category dtype 的字符串** —— 遍历全表落库会在它这里抛，
> 或者更糟：静默写成 NULL，于是「行业因子存在且全是空值」。
> `build` 只接受 `ALPHA_CATEGORIES`（Task 7 落的白名单）里的因子，
> 其余明确拒绝并说明理由；`log_mv` / `beta_250` 是数值可以存，但它们不是 alpha，
> 存不存由调用方显式指定，不靠遍历撞上。

**Files:** Create `ashare/data/derived_store.py` 与 `ashare/factors/store.py`; Create `tests/ashare/test_factor_store.py`

**Interfaces**
- Consumes：`_derived.connect_write/read`、`compute_factor`、`query.snapshot_id`
- Produces：
  ```python
  def build(names, dates, *, overwrite=False, progress=None) -> dict   # {name: rows_written}
  def read(names, date, universe, processed=True) -> pd.DataFrame       # index=ts_code, columns=names
  def coverage_report(names) -> pd.DataFrame                            # 每因子的日期区间与覆盖率
  ```

**决策**
- **`read` 必须按 `snapshot_id` 校验命中**（Task 1 把 snapshot_id 定为列而非主键的直接代价）：
  只返回 `snapshot_id == query.snapshot_id()` 的行，不等的行**当未命中**。
  不校验就会静默把另一批数据算出的因子值喂进回测 —— 这是 P2 里最容易造出"好看的假净值"的一条路（架构 B4）。
- `build` 幂等：同 `(factor_name, param_hash, trade_date, ts_code)` 覆盖写；`overwrite=False` 时跳过
  已有且 `snapshot_id` 相同的 (因子, 日期)。写入用 `_derived.UPSERT_FACTOR_VALUE`（覆盖语义与
  `snapshot_id` 更新绑在一起；发 `DO NOTHING` 会留下陈旧值配陈旧快照，read 会当成当前快照放行）。
- **PK 命中 ≠ 缓存有效**：`param_hash` 只区分 `default_params`，`neutralize` / `direction` /
  因子函数体都不在哈希里（函数体本来也没法哈希）。所以把 `neutralize=True→False` 改一下，
  同一个 PK 下存的就是另一种语义的 `processed_value`。`build` 把 PK 命中当成"存在某一代"，
  **不能**当成"这一代与当前 spec 一致"；判定有效性靠 `snapshot_id` + `overwrite`。
- `read` 命中不到就**返回空**，不静默现算 —— 现算与落库的口径分歧是最难查的一类 bug；由调用方决定要不要 `build`。
- store 是**唯一写 derived 库**的因子模块。

**验收断言**
- `build` 两次 → 行数不翻倍；第二次返回 0（跳过）
- `snapshot_id` 变化后再 `build` → 该日该因子的行被覆盖且 `snapshot_id` 列更新
- `read` 未命中返回空 DataFrame（带正确列名），不现算
- **`snapshot_id` 不匹配 = 未命中**：写入一批因子值，改动 market 库让 `snapshot_id` 变化，
  断言 `read` 返回空而不是返回陈旧值（这条不通过 = 回测会拿错数据的因子）
- `coverage_report` 给出每因子的 `first_date/last_date/n_dates/mean_coverage`

---

### Task 9: 回测数据结构

**Files:** Create `ashare/backtest/__init__.py`, `ashare/backtest/types.py`; Create `tests/ashare/test_backtest_types.py`

**Interfaces**
- Produces：`CostConfig` / `PortfolioConstraints` / `BacktestConfig`（含 `param_hash()`）/ `BacktestResult`（含 `summary()`）
  —— 字段完全按架构 §4.3，此处不重复。

**决策**
- `BacktestConfig.param_hash()` **只包含影响结果的字段**：`compute_diagnostics` 不进 hash（只影响算不算诊断），
  `shuffle_seed` **进** hash（它改变结果）。
- `factors` 用有序 tuple `((name, weight), ...)` 以便 hash 稳定。
- `macro_timing` 默认 **False**（U3：宏观层属 P3）。
- `BacktestResult.summary()` 必须 < 3 KB（供 REST / Agent 工具返回）。

**验收断言**
- 同参数不同书写顺序 → 同 `param_hash`
- 改 `compute_diagnostics` → hash 不变；改 `shuffle_seed` → hash 变
- 改 `cost.multiplier`（闸 4）→ hash 变
- `summary()` 序列化后 < 3072 字节

---

### Task 10: 组合构建 portfolio.build_targets

**Files:** Create `ashare/backtest/portfolio.py`; Create `tests/ashare/test_portfolio.py`

**Interfaces**
- Produces：`build_targets(scores: pd.Series, target_position: float, prev_weights: pd.Series, industry: pd.Series, constraints: PortfolioConstraints) -> pd.Series`

**决策（算法说明书 §7.2 —— 计划初稿误引「§6.2」，那是设计规格的编号；U1：先等权不上 LP）**
```
1. 按 scores 降序取前 top_n
2. 初始权重 = target_position / top_n
3. 行业上限：Σw > max_industry 的行业，按 scores 删该行业末位，用池中下一名替补
   **★ 2026-08-21 撤销「最多 10 轮，超出则放宽 N」**：评审实测不动点循环在 20 只票的账本上会走到 **15 轮** —— 那是正常运转不是异常。写死 10 轮在 15 只票的样例上会返回 0.075 的仓位对着 0.05 的上限（超 50%），而 59 条用例全绿。循环上界必须是 `len(idx)+1` 并配 `for...else: raise` 自证。「放宽 N」本来也治不了行业上限（π ≤ K·max_ind 与 N 无关）
4. 单股上限 max_single 截断，截下来的额度按剩余股票的 scores 比例再分配
5. 换手上限：若 ||w − w_prev||₁ > max_turnover，按 |Δw| 降序只执行前若干笔直到累计达上限，其余保持 w_prev
```

**验收断言**
- 约束全部满足：`Σw == target_position`（1e-9 容差）、`max(w) ≤ max_single`、每行业 `Σw ≤ max_industry`、`||w−w_prev||₁ ≤ max_turnover`
- 行业上限触发时，被删的是该行业 scores 最低的股票
- 换手裁剪触发时，执行的是 `|Δw|` 最大的那几笔（优先做最重要的调整）
- `prev_weights` 为空（首期）→ 换手约束不阻碍建仓
- scores 全 NaN / 覆盖率不足 → **返回 `(None, warnings)`**（该日不调仓），不抛
  **2026-08-21 最终口径，取代此前两版。** 演化过程本身值得记：
  ① 初稿「返回空 Series」—— 空读作**清仓**，与同句的「不调仓」自相矛盾，且中断常与
     极端行情同期，回测里会伪装成「策略在暴跌前防御性离场」的假净值。
  ② 改「返回 `prev_weights` 原样」—— 修掉了清仓，但引入更隐蔽的一条：
     `build_targets` 交出的是 **T 日收盘**度量的权重（τ 开盘价对它是未来，用了就是 D6 前视），
     而 `simulate` 会在 τ 开盘重算漂移 → `Δw = 整夜跳空` →
     **在一个明确说了「今天不调仓」的日子里，把每只票的隔夜跳空当成交易做掉，还真扣成本。**
  ③ 定为 `None`：它与「空＝清仓」「prev＝这是你的目标」都不同，读不歪。
  **`None` 不等于「引擎什么都不做」**：`simulate(targets=None)` 读作
  「按 τ 开盘的现有持仓持平，只执行强制退出」—— 否则中断日撞上退市股就又回到了
  §5.5 那个「幽灵资产」的洞。零自主交易，强平照做。
  **2026-08-20 修正：本条初稿写「返回空 Series」，与同句的「不调仓」自相矛盾，且是亏钱的那一侧。**
  空 Series 读作「清仓」，`prev_weights` 读作「维持」—— 全 NaN 是数据中断，不是清仓信号。
  数据中断时清仓在回测里会伪装成「策略在暴跌前防御性离场」（中断常与极端行情同期），
  是一条看起来很聪明的假净值。

---

### Task 11: 成交模拟 execution.simulate（D6 落地处）

**Files:** Create `ashare/backtest/execution.py`; Create `tests/ashare/test_execution.py`

**Interfaces**
- Consumes：`query.get_tradable_mask(exec_date, ts_codes)`
- Produces：`simulate(exec_date, targets: pd.Series, prev_holdings: pd.Series, equity: float, cost: CostConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]`
  返回 `(trades, holdings, blocked)`

**决策（规格 D6 / 算法说明书 §5.2–5.3）**
- 成交价 = `mask.open_hfq`（T+1 开盘价）。**禁止用 T 日收盘价**。
- 不可交易集合 `F`：`can_buy=False` 且需买入、`can_sell=False` 且需卖出、`reason='suspended'|'limit_unknown'|'no_quote'`。
- **权重再分配**：锁定权重 `L = Σ_{i∈F} w_prev,i`；可交易部分按目标权重相对比例分配剩余额度 `(target_position − L)`；
  若 `L > target_position` 则可交易部分清零，**不强行卖出锁定仓位**。
- **不归一化到 1** —— 组合允许持现金，`Σw = target_position ≤ 1`。
- 退市（`reason='delisted'`）→ 按 `mask.close_hfq × 0.5` 清仓（规格 B8：退市整理期连续跌停、几乎无流动性，
  按收盘价成交是系统性乐观偏差），并记 warning。
- `blocked` 是 D6 的证据链：`exec_date, ts_code, intended_side, intended_weight, reason`。

**验收断言**
- 一字涨停：`can_buy=False` → 目标里的买入被拦，`blocked` 有该行，权重保持 `prev`
- 一字跌停：`can_sell=False` → 卖出被拦
- 停牌 / `limit_unknown`：两侧都拦
- 权重再分配：3 只股票 1 只锁定，断言可交易的 2 只按相对比例吃掉剩余额度，且 `Σw == target_position`
- `L > target_position` → 可交易部分为 0，锁定仓位不动
- 退市清仓价 == `close_hfq × 0.5` 且 warning 非空
- **成交价断言**：`trades.price_hfq == mask.open_hfq`，且不等于当日 `close_hfq`（钉住"不是收盘价成交"）

---

> **★ 2026-08-20 追加（Task 4 评审转来）：用真实数据重定 `volatility_60` 的 `min_coverage`。**
> 评审做了 2000 路径蒙特卡洛：停牌日经 ffill 后，vol 估计量**无偏但方差爆炸** ——
> 在共享的 0.60 闸边界上，p5/p95 = 0.757/1.385，即一只股票的 60 日波动率在真值的
> −24% 到 +39% 之间掷硬币。噪声会通过 errors-in-variables 把 IC 衰减向零。
> **本期不动阈值**：只用合成蒙特卡洛就调参数，正是本项目的五道闸要防的无根据调参。
> 这里要做的是：量出收紧 `min_coverage` 到 0.70 / 0.80 各自的 IC 与覆盖率代价，
> 再决定。（评审也确认了不该改估计量本身：掩码掉补出来的收益会留下跨多日的那一个，更糟。）

> **★ 2026-08-21 追加（Task 6 评审转来）：风格归因用 groupby 会踩 category dtype 的同一个坑。**
> `pipeline.neutralize` 已经踩过一次：`pd.get_dummies` 对 categorical **会为零观测的
> 类别建列**，把「哑变量只用有效样本构造」的保护无声废掉。评审在 pandas 2.3.3 上确认
> 还有两个同族陷阱没设防：`v.groupby(ind).mean()` 会为零观测类别吐出一行
> （`__OTHER__ → NaN`），`ind.value_counts()` 会报 `__OTHER__ → 0`。
> **§9 的风格归因恰恰就是一次按行业 groupby** —— 一个凭空多出来的 `__OTHER__` 行
> 会让归因表看起来多了一个行业暴露。先 `.astype(object)` 或
> `.cat.remove_unused_categories()`，并写一条零观测行业的用例钉住。

### Task 12: 成本 cost.charge + 指标 metrics.compute

**Files:** Create `ashare/backtest/cost.py`, `ashare/backtest/metrics.py`; Create `tests/ashare/test_cost_metrics.py`; Modify `requirements.txt`（+statsmodels）

**Interfaces**
- Produces：
  ```python
  def charge(trade_rows: pd.DataFrame, cost_cfg: CostConfig) -> pd.DataFrame   # +commission/stamp_duty/transfer_fee/impact/total_cost
  def compute(equity, trades, positions, benchmark_series, *, full: bool) -> dict
  def ic_series(factor_panel, forward_returns) -> pd.DataFrame                  # IC / RankIC per date
  def icir(ic: pd.Series) -> dict                                              # mean / std / icir / t_newey_west / p
  def layered_returns(scores, forward_returns, n_layers=10) -> pd.DataFrame
  ```

**决策（算法说明书 §5.4 / §4 / §9）**
- 成本：佣金双边 2.5bp；印花税**仅卖出** 5bp；过户费 0.1bp；
  冲击 `min(0.5 × 委托额/ADV20 × 滞后 20 日平均振幅, 30bp)`（**不是当日振幅**，见 §5.4 裁决）。一个往返 ≈ 0.3%。
- **换手计算必须扣持仓自然漂移**：`Δw = w_t − w_{t-1} × (1 + r_{t-1})`。不扣会系统性高估换手 15%–30%。
- **IC 主用 RankIC**（Pearson 对涨停连板极敏感）。
- **ICIR 的 t 检验必须用 Newey-West 调整标准误**（滞后阶数 `floor(4(T/100)^(2/9))`）——
  IC 序列有自相关，朴素标准误把 t 值高估 30%–50%，是把噪声因子判成有效因子的头号原因。
- 分层单调性用组序号与组年化收益的 Spearman 秩相关 `ρ_mono`。
- 多空组合仅为因子评估口径，**不是策略**（A 股融券成本与券源不支持系统性做空）。

**验收断言**
- 买入 100 万：佣金 250 + 过户费 10、无印花税；卖出 100 万：佣金 250 + 印花税 **500** + 过户费 10
  **2026-08-21 修正：本条初稿的印花税写成 5000，错了 10 倍**（5000/1e6 = 50bp，而同一份
  brief、`CostConfig.stamp_duty_bps=5.0` 和 §5.4 的 $c^s$ 都写着 5bp）。实现者是靠
  §5.4 自己的「往返 ≈ 0.3%」交叉验出来的：50bp 时光显性成本单边就 0.55%，自相矛盾。
- 冲击成本封顶 30bp
- 换手漂移：持仓不动但股价涨 10% → **满仓时**换手为 0（不是 10%）
  **2026-08-21 修正**：这条只在 $\pi=1$ 成立。留现金时正确答案**非零** ——
  现金不涨，股票涨会把权重顶上去，$\pi(1.1)/(1+0.1\pi)=\pi$ 仅当 $\pi=1$。两种情形都要钉。
- **Newey-West 钉子**：构造强自相关的 IC 序列，断言 NW 的 t 值显著小于朴素 t 值
- Sharpe / Calmar / 最大回撤 / IR 用手算值断言
- 完全单调的分层 → `ρ_mono == 1.0`
- **风格归因必须报告残差的规模暴露**：中性化选了 OLS（算法说明书 §3.2），
  真实数据上若显示残差仍有显著规模暴露，就是推翻那个裁决的证据 —— 补救是加非线性规模项，
  不是换回 WLS。归因不报这一项，这个裁决就永远无法被真实数据检验

---

> **★ 2026-08-20 裁决：`engine_version` 绝不进 `param_hash`。**
>
> Task 9 的实现者指出 §4.3 把 `engine_version` 放在 Result 而非 Config 上，
> 于是「引擎语义改了、参数没改」会得到同一个指纹配不同的数字。这是真的，
> 但**把它塞进 hash 会把 D7 弄坏，而不是修好**：每次引擎升版都会铸出一个新 hash，
> 于是静默地又发一次样本外机会 —— 那正是 D7 要挡的污染。
>
> `(param_hash, data_snapshot_id)` 是**实验的身份**；`engine_version` 是**跑它的代码的身份**。
> 后者记在 `docs/oos-runs.md` 的同一行里做溯源，永远不进指纹。
> 引擎改了还想再跑样本外，就是第二次样本外 —— 该被闸挡下来，由人显式承认。

> **★ 2026-08-20 追加（Task 10 转来，三条都会静默出错）：**
>
> **① `prev_weights` 每期必须从持仓重算，不能把上期 `build_targets` 的返回值喂回去。**
> 换手是**实际要交易的量**，而账本在两次调仓之间随价格漂移。
> 每期算 `prev_w = 上期持仓 × T 日收盘后复权价 / 净值`。
> 把上期目标喂回去是最自然、也是错的那个写法 —— 它假装账本一直停在目标上。
>
> **② `scores` 必须以【完整股票池】为索引、缺失处填 NaN。**
> `build_targets` 用覆盖率闸区分「数据中断」与「正常稀疏」，
> 调用方若先 `dropna()` 再传进来，覆盖率恒为 100%，闸就瞎了。
>
> **③ `build_targets` 返回 `(Series, list[str])`，不是 `Series`。**
> 沿用 `pipeline.process` 已有的约定。调用处写成 `targets, warns = ...`，
> warnings 必须汇进 `BacktestResult.warnings` ——
> **静默放宽换手、静默留现金、静默冻结账本，正是这三件事必须被看见。**

> **★ 2026-08-20 追加（Task 11 评审转来）：数据中断那一期要【跳过 simulate】，不是传冻结权重。**
>
> `build_targets` 在中断日返回上期权重，是**在 T 日收盘度量的**（它只能知道这个，
> τ 开盘价对它是未来，用了就是 D6 前视）。但 `simulate` 会在 τ 开盘重算漂移，
> 于是 `Δw =（T 收盘权重）−（τ 开盘权重）= 整夜跳空`——
> **在一个明确说了「今天不调仓」的日子里，把每一只票的隔夜跳空当成交易做掉，还真扣成本。**
>
> 落法：中断路径 `build_targets` 返回 `(None, warnings)`，引擎见 `None` 直接跳过该期
> `simulate`。`None` 与「空 Series＝清仓」「prev＝这是你的目标」都不同，读不歪。
>
> 附带把两个度量口径写清楚，它们**故意不同**：
> · `build_targets` 拿 **T 日收盘**权重 —— 换手上限因此差一个隔夜跳空，是已知近似不是 bug；
> · `simulate` 拿 **shares**（不是权重），自己在 τ 开盘重算漂移，算出真实的 Δw 与成本。
>   接口收 shares 正是为了这个，Task 13 不得改传权重。
> · 两处的退市折价必须一致，否则净值先虚高再神秘掉下来。

> **★ 2026-08-21 追加（Task 7 转来，三条）：**
>
> **① 空股票池由引擎显式跳过，不靠 catch。** `compute_factor` 对空 universe **抛异常**
> （问一个空池子要因子是编程错误），但空池在真实回测里是会出现的（历史最早期、数据缺口、
> 日历误判）。引擎必须在调用前检查并跳过该日 + 记 warning ——
> 否则一天的空池会杀死整个 15 年的运行。低层失败大声、引擎显式处理预期情况，各司其职。
>
> **② `lookback_days` 的唯一消费者就是这里。** 它至今全 repo 无人读取。
> 落法：`query.preload(start − max(spec.lookback_days for 用到的因子), end)`，
> **在回测入口调一次**，不是每个因子每天调。
> 注意**多声明是安全的、少声明不安全**（多加载一点而已），所以 `beta_250` 声明 260
> 实取 251 不用改。Task 4 留下的那条「声明值 == 实际传给数据层的 lookback」测试
> 只对量价因子成立，正是为了不把这个安全方向锁死。
>
> **③ `build_targets` 的覆盖率闸经 `combine` 喂进来时是【二值】的。**
> `process` 末尾 `fillna(0)`，所以到达 `build_targets` 时每只票都有数 ——
> 覆盖率只可能是 100%（还有因子存活）或 0%（全部因子都被覆盖率闸剔掉）。
> 那道 50% 的闸**探测不到部分降级**，它只是全面中断的最后一道防线。
> 部分降级的信号在别处：`combine` 逐因子的剔除 warning，加上 §9 诊断里
> 「每期实际用了几个因子」这一项 —— Task 12 必须把它算出来，否则
> 「12 个因子失效、拿剩下 2 个把整个账本调了一遍」这件事只能靠翻 9000 条 warning 发现。

> **★ 2026-08-21 追加（Task 10 修复转来）：`positions` 要记【意图】权重，不只记交出的。**
> 换手预算绑定时被裁掉的调仓现在会告警并带上 L1 差额，但
> `BacktestResult.positions.target_weight` 存的仍是**交出的**权重 ——
> 意图中的账本哪里都没留。后果是净值曲线**无法归因**：分不清「跑输是因为信号不行」
> 还是「因为换手约束让信号表达不出来」，而对一个受换手约束的策略，这两者是完全不同的结论。
> 落法：`build_targets` 额外交出裁剪前的目标；引擎写进 `positions.intended_weight`。
> §9 归因用这一列算约束拖累（意图账本的反事实收益 − 实际收益）——
> **不在 P2 建那套归因，只把列留出来**，否则 Task 12 拿不到原料。

> **★ 2026-08-21 追加（Task 12 评审转来，四条引擎侧的落点）：**
>
> **① `metrics.compute(..., initial_capital)`** —— 见架构 §4.3 的 `equity` 口径裁决。
> **② `metrics.compute(..., n_factors_configured)`** —— 「每期用了几个因子」的分母必须是
> **配置的**因子数，不能用观测到的最大值。评审实测：`factors_used = [2,2,2,2]` 时
> 观测最大值就是 2，于是「低于半数的期数 = 0」，**一句告警都没有** ——
> 而「12 个因子失效、拿剩下 2 个把账本调了一遍」正是这个诊断要抓的东西，
> 且**均匀降级才是更常见的那种**（一个因子因为缺列而挂掉，是每期都掉，不是掉一期）。
> **③ 引擎给 `trades` 附 `adv20` 与 `range` 两列。** `range` 取**滞后 20 日平均振幅**
> （§5.4 裁决，不是执行日当天）。`query.get_bars(signal_date, codes, lookback=20,
> fields=("high","low","pre_close"))` 直接支持，并顺带给出 `is_suspended` ——
> D9 占位行的 `high==low==pre_close` 振幅为 0，会把均值拖低从而**低估**成本，必须剔掉，
> 与 ADV20 已有的「剔停牌占位行」规则同源。
> **④ `derived_store.read_factor_values` 在冷库上会对每个因子各发一条告警**，
> 所以 `build` 之前的预热读会一次吐 N 条。不要无条件汇进 `BacktestResult.warnings`。

### Task 13: 引擎 engine.run_backtest

> **★ 调仓频率不设 config 字段，但也不许当函数参数（2026-08-20 裁决）。**
> 周频是用户定的产品决策（周频调仓、持 1–4 周），不是调参旋钮，为单一取值加字段是投机。
> 但 `run_backtest()` **禁止**接受 `freq`/`rebalance_days` 之类的参数 ——
> 那会让日频与周频两次运行共用一个指纹。若将来真要可调，它必须进 `BacktestConfig`，
> 让 `param_hash` 覆盖到。

**Files:** Create `ashare/backtest/engine.py`, `ashare/backtest/store.py`; Create `tests/ashare/test_engine.py`

**Interfaces**
- Consumes：`factors.store.read` / `factors.combine`、`portfolio.build_targets`、`execution.simulate`、`cost.charge`、`metrics.compute`、`query.get_trade_dates(freq='W')` / `get_universe` / `snapshot_id(pin=True)` / `preload`
- Produces：`run_backtest(config, *, on_progress=None) -> BacktestResult`；`store.save(result, run_id)` / `load(run_id)` / `append_oos_run(result)`

**决策（规格 §5.3 / D7）**
- `engine.py` **≤ 400 行且只做编排**，实体逻辑在 portfolio/execution/cost/metrics（架构 A5：硬凑"核心 400 行"
  却把逻辑挪走是文字游戏；这里的口径是可检查的）。
- 主循环：
  ```
  snapshot = query.snapshot_id(pin=True)          # ★ 钉住，中途 promote 会抛而不是静默换库
  query.preload(start, end, ("daily_bar", "daily_basic"))
  for t in query.get_trade_dates(end, start=start, freq='W'):
      universe = query.get_universe(t)
      scores   = factors.combine(weights, t, universe)      # 优先 store.read
      exec_date = query.next_trade_date(t)
      if exec_date is None: break                            # 日历末端
      targets  = portfolio.build_targets(scores, position, prev_w, industry, constraints)
      trades, holdings, blocked = execution.simulate(exec_date, targets, prev_holdings, equity, cost)
      equity  *= (1 + Σ w·r − cost)
  assert query.snapshot_id() == snapshot                     # 结束再核一次
  ```
- `run_id = f"{param_hash}_{data_snapshot_id}_{started_at:%Y%m%dT%H%M%S}"`。
- **`append_oos_run` 自动写 `docs/oos-runs.md`**（U6 决策：人工记录必然漏记，漏记即 D7 失效）；
  只在 `config.end > 2019-12-31`（即触及样本外）时追加，并把未跑的闸列进备注。

**验收断言**
- **时序语义钉子**：构造两只股票，其中一只在 T+1 开盘暴涨。断言组合收益用的是 T+1 开盘价成交，
  且把该股 T 日收盘价改掉**不影响**成交价（证明不是收盘价成交）
- 回测途中 `os.replace` 换库 → 抛 `QueryError`（快照钉住）
- 首尾 `snapshot_id` 一致才产出 result
- `engine.py` 行数 ≤ 400（源码级断言）
- 日历末端 `next_trade_date` 返回 None → 正常结束不抛
- `run_id` 含 param_hash 与 snapshot_id；`save`/`load` 往返一致

---

> **★ 2026-08-21 追加（Task 13 转来）：闸 3 的 200 次置换要先决定「因子算几遍」。**
> 计划里 `combine` 那句「优先 store.read」**没有实现** —— `factors/store.py` 没有 `read`，
> `combine` 也不碰派生库。Task 13 拒绝在 engine 里加这条快路，理由成立：
> 白名单 + 覆盖率闸 + 重新归一都是**策略**，挪进编排层就把 money path 分了叉，违反 A5。
> 正确的落点是 `factors.base.combine(..., use_store=True)`。
>
> **但要不要做，取决于闸 3 的形状**：置换只打乱**同一天横截面内**的分数，因子值本身不变。
> 若闸 3 按「算一次因子、置换 200 次分数」实现，store 快路对它就不重要；
> 若每次置换都重跑整条链，那是 200 × 15 因子 × 780 天。**先定闸 3 怎么跑，再回答要不要做。**

### Task 14: 防自欺五闸 guards

**Files:** Create `ashare/backtest/guards.py`; Create `tests/ashare/test_guards.py`

**Interfaces**
- Produces：
  ```python
  @dataclass class GateResult: name: str; passed: bool; detail: dict; note: str
  def gate1_out_of_sample(cfg) -> GateResult          # 样本外 Sharpe ≥ 样本内 × 0.6
  def gate2_walk_forward(cfg, train_years=5, test_years=1) -> GateResult
  def gate3_shuffle(cfg, n=200, seed=0) -> GateResult
  def gate4_cost_stress(cfg, multiplier=2.0) -> GateResult
  def gate5_param_plateau(cfg, grid) -> GateResult
  def run_all_gates(cfg, *, gates=None) -> dict[str, GateResult]
  ```

**决策（算法说明书 §8）**
- **闸 3 内部强制 `compute_diagnostics=False`**：否则 200 次 × 60s = 3.3 小时，这个闸就没人跑（架构 A3）。
- **闸 3 的置换必须在同一天的横截面内**，绝不可跨时间打乱 —— 跨时间置换会破坏市场整体涨跌的时序结构，
  对照组分布失真、检验结论无效。`p = (1 + #{b: SR_b ≥ SR_real}) / (n+1)`，通过标准 `p < 0.05`。
- 闸 5 `PeakRatio = SR(θ*) / mean(SR(邻域))`，通过标准 `< 1.3`（比值越高越说明最优点是尖峰）。
- `run_all_gates` 返回全部结果，**不因某闸失败提前退出** —— 操作员要一次看到全貌。
- **未跑的闸必须出现在 `GateResult(passed=False, note='未运行')`**，不能悄悄不返回（U5）。

**验收断言**
- 闸 3：用一个**真随机分数**的假回测函数，断言 p 值分布接近均匀（不是恒 < 0.05）；
  用一个与未来收益完全相关的分数，断言 p < 0.05
- 闸 3 的置换确实只在同日内：注入一个记录被置换索引的 spy，断言每次置换的 `trade_date` 集合不变
- 闸 3 强制关诊断：断言传入的 config 的 `compute_diagnostics is False`
- 闸 4：成本 multiplier 确实是 2.0 传下去
- 闸 5：给一个尖峰形状的 SR(θ) → `PeakRatio > 1.3` 判不通过；平缓高原 → 通过
- `run_all_gates(gates=['gate1'])` → 其余闸返回 `passed=False, note='未运行'`

---

### Task 15: 引擎正确性反测（§5.6，P2 硬性验收）

**Files:** Create `tests/ashare/test_p2_acceptance.py`; Modify `docs/oos-runs.md`（说明本期未跑样本外）

**决策**
- **验收窗口锁定 2010-01-01 – 2019-12-31**（规格 §5.6）。2020 年后注册制、量化拥挤、小盘因子衰减，
  这些异象确有真实弱化；把窗口开到今天等于要求工程师"调数据直到跑出结果"——把防自欺的检验变成自欺的来源。
- 库不存在（未回补）→ 整文件 skip，与 P1 验收同一模式。

**验收断言（跑不出来 = 数据或引擎有 bug，不是"市场变了"）**
- `reversal_20` 10 分层：单调**递增**，`ρ_mono > 0.7`，多空年化 > 10%
- `turnover_20` 10 分层：单调**递减**，`ρ_mono < −0.7`
- `log_mv` 10 分层（2010–2016 子区间）：小市值显著占优
- 全市场 2010–2019 周频回测（因子已落库 + `compute_diagnostics=True`）单次运行 **< 60 秒**
- `BacktestResult` 同时带 `param_hash` 与 `data_snapshot_id`
- 涨停买不进 / 跌停卖不出 / 停牌不可交易在真实数据上各能找到 ≥ 1 个 `blocked` 样本

---

## 自查

**规格覆盖**：规格 §5.1 因子清单 → Task 4/5/6；§5.2 处理链 → Task 3；§5.3 引擎语义 → Task 11/13；
§5.4 评估指标 → Task 12；§5.5 五闸 → Task 14；§5.6 反测 → Task 15；架构 §4.2 → Task 2/3/7/8；
§4.3 → Task 9–13；算法说明书 §1–§9 逐节有对应任务。

**已知缺口（有意不做，非遗漏）**：宏观择时层（P3，U3）；LP 组合优化（U1，先证明等权不够用）；
IC 加权合成（U2）；`margin_chg_20`（规格 A1 已砍）；样本外运行（U6，D7 只跑一次，属 P3）。

**类型一致性**：`FactorSpec`（Task 2）被 Task 3 的 `process(spec=)`、Task 7 的 `compute_factor`、
Task 8 的 `build` 复用；`CostConfig`（Task 9）被 Task 11 `simulate(cost=)`、Task 12 `charge(cost_cfg=)` 复用；
`neutralize` / `process` 统一返回 `(series, warnings)`；`GateResult` 只在 Task 14 定义。
