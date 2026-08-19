# A 股全市场量化策略平台 · 设计规格

- 日期：2026-08-18
- 状态：已确认（待评审）
- 上游：本文件是所有下游文档（架构文档 / BRD / 算法说明书）与代码的**唯一契约**

---

## 0. 一句话定义

每周五收盘后自动运行，输出**下周一开盘执行的调仓清单**（买什么、卖什么、多少仓位、什么价位区间挂单），
并由 LLM 生成解释与个股深度报告。**数学模型出信号，LLM 只做解释，物理隔离。**

---

## 1. 目标 / 非目标

### 目标

| 编号 | 目标 |
|---|---|
| G1 | 全 A 股（含已退市）2010-01-01 至今的 **PIT 数据底座** |
| G2 | 可复现、防前视/幸存者偏差的**周频横截面回测引擎** |
| G3 | 宏观择时（总仓位）× 横截面选股（个股权重）的**两层策略** |
| G4 | **每周**产出含时点与价位区间的买卖清单；**每日**产出风险预警（不产出每日调仓清单） |
| G5 | 研报/财报/公告知识库，chunk 带 `publish_date`，支持时点隔离检索 |
| G6 | LLM 生成**个股全量级分析报告**与信号归因（只读） |

### 非目标（明确不做）

| 编号 | 非目标 | 原因 |
|---|---|---|
| N1 | 预测个股绝对价格/点位 | 不可实现。只做横截面相对排序 + 风险预算 |
| N2 | A 股自动下单 | 券商不开放个人合规 API；程序化交易需报备。终点是信号 + 人工执行 |
| N3 | 让 LLM 参与买卖决策 | LLM 输出不可回测、不可复现。见 §2 铁律 D1 |
| N4 | 日内 / 分钟级策略 | 周频调仓已定，分钟数据非必需 |
| N5 | 宏观层用机器学习 | 月频 15 年仅 180 个样本点，任何 ML 都是过拟合 |
| N7 | 每日调仓清单 | 策略为周频，每日出调仓清单是噪音，且会训练用户忽略通知。每日只出**风险预警**（持仓个股停牌 / 跌破止损 / 财报暴雷 / ST 风险） |
| N6 | 动现有 `brokers/` 交易通道 | 本期 A 股不接下单，risk_gate / intent_store 保持现状 |

---

## 2. 设计铁律（违反即判定为缺陷，不接受"先跑通再说"）

### D1 · LLM 与决策层物理隔离
- LLM 层（P5）对信号表、因子表、策略参数**只有读权限**。
- 不存在任何代码路径能让 LLM 输出写回 `signal` / `factor` / `strategy_param`。
- 违反表现：任何 `stock_report` / `explain_signal` 类函数出现写操作。

### D2 · 一切查询强制携带 `as_of_date`
- `ashare/data/query.py` 是**唯一**数据出口，所有公开函数签名第一个参数为 `as_of_date`。
- 因子函数签名固定为 `f(as_of_date, universe) -> pd.Series`，让前视偏差在签名层面就难以写出。
- 禁止在因子/策略代码中直接连 DuckDB 或读 Parquet。

### D3 · 财报走 PIT，键是公告日不是报告期
- `financial_pit` 主键含 `ann_date`；查询恒为 `WHERE ann_date <= :as_of_date`，同 `end_date` 取 `ann_date` 最大者。
- 重述数据（`update_flag=1`）另行入库，**不覆盖**原始披露值。

### D4 · 宏观数据同样是 PIT
- 社融次月 15 日左右公布、PMI 月末、CPI 次月 9 日左右。`macro_indicator` 必须有 `publish_date`。
- 查询恒为 `WHERE publish_date <= :as_of_date`。
- 违反表现：8 月 1 日的回测里用到了 7 月社融。

### D5 · 退市股必须入库
- `stock_basic.delist_date` 保留；股票池按 `as_of_date` 动态生成。
- 违反表现：股票池由"今天还在市的股票"倒推 → 幸存者偏差。

### D6 · 回测成交语义固定
- T 日收盘后算信号 → **T+1 开盘价**成交。禁止用 T 日收盘价成交。
- 「开盘价成交」在现实中**只能通过 09:15–09:25 集合竞价挂限价单实现**（09:25 统一撮合，全部按开盘价成交）。
  09:30 之后下单拿到的是连续竞价价格，与回测假设系统性偏离。执行时点文案必须写「集合竞价阶段」，不得写「开盘后」。
- T+1 一字涨停（`open == limit_up and high == low`）→ 买不进；一字跌停 → 卖不出；停牌 → 不可交易。以上三种情况保留原仓位。
- `limit_up`/`limit_down` 数据缺失时按规则兜底计算（主板 10%、创业板 2020-08-24 起 20%、
  科创板开板起 20%、ST/*ST 5%、北交所 30%、退市整理期 10%、新股上市首日无涨跌幅限制）。
  **规则也算不出的当日，一律按「不可交易」处理**，不得假设可交易 —— 前者只损失一次机会，
  后者会在回测里凭空生成现实中不存在的成交。

### D7 · 样本外只跑一次
- 2010-01-01 ~ 2019-12-31 为训练/调参区间；2020-01-01 至今为样本外。
- 样本外每次运行必须记录到 `docs/oos-runs.md`（日期、`param_hash`、`data_snapshot_id`、结果）。
  改任何参数后再跑样本外，等于把样本外污染成样本内，必须在文档中标注。
- **`param_hash` 单独不足以保证可复现**：财报重述（新增 `update_flag=1` 行）与复权因子的回溯修正，
  会让同一套参数在不同日期跑出不同结果。因此必须并列记录 `data_snapshot_id`（数据版本号）。
  两者任一变化，结果都不可与历史记录直接比较。

### D8 · 后复权价是唯一真值
- 回测内部只用后复权价；展示层用原始价 + `adj_factor` 还原。
- 禁止前复权（前复权会随新数据变动，导致历史回测结果不可复现）。

### D9 · 日线必须按交易日历补齐，停牌日写占位行
- Tushare `daily` 接口在停牌日**不返回行**。若直接入库，`daily_bar` 会出现日期空洞。
- 后果：任何 `rolling(N)` 拿到的是「最近 N 条记录」而非「最近 N 个交易日」。停牌 5 天的股票，
  其 `reversal_20` 实际覆盖 25 个交易日 —— **横截面回看窗口长度不一致，因子被静默污染，零报错**。
- 规则：入库时对每只股票在 `[list_date, min(delist_date, today)]` 区间与 `calendar` 左连接，
  缺失交易日插占位行：`is_suspended=TRUE`，OHLC 全取前一交易日收盘价，`vol=0`，`amount=0`，
  `adj_factor` 沿用前值。
- 校验：`count(daily_bar WHERE ts_code=X)` 必须等于该股在市区间内的交易日数，**误差为 0**（非 0.1%）。

---

## 3. 系统分层

```
┌─ P1 数据底座 ── DuckDB 单文件 + Parquet ─────────────────────────┐
│  calendar │ stock_basic(含 delist_date) │ stock_status(ST 历史)  │
│  daily_bar(后复权+涨跌停+停牌) │ daily_basic(估值/市值)           │
│  financial_pit(PK 含 ann_date) ★ │ macro_indicator(含 publish_date) ★│
│  money_flow(北向/两融/大单) │ index_daily                        │
└──────────────────────────────────────────────────────────────────┘
        ↓  ashare/data/query.py —— 唯一出口，强制 as_of_date
┌─ P2 因子库 + 回测引擎 ───────────────────────────────────────────┐
│  Factor: f(as_of_date, universe) -> Series                       │
│  处理链: 原始值 → MAD 去极值 → 行业+市值中性化 → zscore           │
│  引擎: <400 行；T 收盘算 / T+1 开盘成交 / 涨跌停停牌不可交易      │
│  防自欺五闸: 样本外一次 / walk-forward / shuffle 对照             │
│              / 成本翻倍 / 参数高原                                │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌─ P3 策略层（两层解耦）──────────────────────────────────────────┐
│  宏观仓位层  → 总仓位 20%~100%（5 指标分位投票，禁 ML）           │
│  横截面选股  → 个股目标权重（单股≤5% / 行业≤20% / 周换手≤30%）    │
│  = 调仓清单（含 T+1 开盘执行时点 + ATR 挂单区间）                  │
└──────────────────────────────────────────────────────────────────┘
        ↓ 只读                                    ↑ 只读
┌─ P5 LLM 投研层 ──────────────┐   ┌─ P4 知识库 ────────────────────┐
│ 个股全量级报告                │←──│ 研报/财报/公告 chunk           │
│ 信号归因（因子贡献分解）      │   │ metadata.publish_date          │
│ 组合周报                      │   │ 检索按 as_of_date 过滤 ★       │
│ ── 无写权限（D1）──           │   └────────────────────────────────┘
└──────────────────────────────┘
        ↓
┌─ P6 前端 ── 复用 static/ ───────────────────────────────────────┐
│  每日信号看板 │ 回测报告页 │ 个股详情页 │ 宏观仪表盘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. P1 · 数据底座

### 4.1 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 存储 | DuckDB 单文件 + Parquet 冷备 | 全 A 15 年日线 ≈ 1500 万行，Parquet < 1GB。零运维，横截面直接写 SQL |
| 主数据源 | Tushare Pro | 财报带 `ann_date`（可自建 PIT）、含退市股、复权因子、宏观库、北向资金 |
| 校验源 | BaoStock（免费） | 日线与复权因子交叉校验 |
| 补充源 | akshare | 公告文本、另类数据 |

数据源做成可插拔 adapter（`ashare/data/sources/`），接口统一，便于未来换源。

### 4.2 表结构

```sql
-- 交易日历
calendar(trade_date DATE PRIMARY KEY, is_open BOOLEAN, pre_trade_date DATE);

-- 股票基础信息（含已退市）
stock_basic(
  ts_code VARCHAR PRIMARY KEY,          -- 600519.SH
  symbol VARCHAR, name VARCHAR,
  sw_l1 VARCHAR, sw_l2 VARCHAR, sw_l3 VARCHAR,   -- 申万三级行业
  market VARCHAR,                        -- 主板/创业板/科创板/北交所
  list_date DATE,
  delist_date DATE,                      -- NULL = 在市 ★ D5
  is_hs VARCHAR,                         -- 沪深港通标的
  _ingested_at TIMESTAMP
);

-- ST 状态历史（PIT，不能只存当前状态）
-- ★ 数据来源：Tushare 无直接接口，须从 namechange（股票曾用名）反推——
--   名称含 "ST"/"*ST" 的区间即为 ST 期。D5 的股票池剔除完全依赖此表，
--   反推逻辑必须有单测（覆盖：戴帽、摘帽、连续两次戴帽、退市整理期）。
stock_status(
  ts_code VARCHAR, start_date DATE, end_date DATE,   -- end_date NULL = 至今
  status VARCHAR,                        -- NORMAL / ST / *ST / DELIST_PERIOD
  PRIMARY KEY(ts_code, start_date)
);

-- 日线
daily_bar(
  ts_code VARCHAR, trade_date DATE,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,  -- 原始未复权
  vol DOUBLE, amount DOUBLE,             -- 手 / 千元
  adj_factor DOUBLE,                     -- 后复权因子 ★ D8
  limit_up DOUBLE, limit_down DOUBLE,    -- 当日涨跌停价 ★ D6
  is_suspended BOOLEAN,
  _ingested_at TIMESTAMP,
  PRIMARY KEY(ts_code, trade_date)
);

-- 日频估值与市值
daily_basic(
  ts_code VARCHAR, trade_date DATE,
  turnover_rate DOUBLE, turnover_rate_f DOUBLE, volume_ratio DOUBLE,
  pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE, ps DOUBLE, ps_ttm DOUBLE,
  dv_ratio DOUBLE, dv_ttm DOUBLE,
  total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
  total_mv DOUBLE, circ_mv DOUBLE,
  PRIMARY KEY(ts_code, trade_date)
);

-- ★ 财报 PIT（核心表）
financial_pit(
  ts_code VARCHAR,
  ann_date DATE,                         -- ★ 公告日 = PIT 键 (D3)
  end_date DATE,                         -- 报告期
  report_type VARCHAR,                   -- 1 合并 / 4 调整合并 ...
  update_flag INTEGER,                   -- 0 原始披露 / 1 重述（不覆盖原值）
  -- 利润表
  total_revenue DOUBLE, revenue DOUBLE, operate_profit DOUBLE,
  total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE,
  -- 资产负债表
  total_assets DOUBLE, total_liab DOUBLE, total_hldr_eqy_exc_min_int DOUBLE,
  -- 现金流量表
  n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE, n_cash_flows_fnc_act DOUBLE,
  -- 派生指标
  roe DOUBLE, roa DOUBLE, grossprofit_margin DOUBLE, netprofit_margin DOUBLE,
  debt_to_assets DOUBLE, current_ratio DOUBLE,
  or_yoy DOUBLE, netprofit_yoy DOUBLE, basic_eps DOUBLE, bps DOUBLE,
  _ingested_at TIMESTAMP,
  PRIMARY KEY(ts_code, ann_date, end_date, report_type, update_flag)
);

-- ★ 宏观（PIT）
macro_indicator(
  indicator VARCHAR,        -- m1_yoy/m2_yoy/tsf_stock_yoy/cpi/ppi/pmi_mfg/shibor_3m/cn10y/usdcny
  period DATE,              -- 数据所属期
  publish_date DATE,        -- ★ 实际公布日 (D4)
  value DOUBLE,
  _ingested_at TIMESTAMP,
  PRIMARY KEY(indicator, period, publish_date)
);

-- 资金流
money_flow(
  ts_code VARCHAR, trade_date DATE,
  hk_hold_ratio DOUBLE,                  -- 北向持股占流通股比
  margin_balance DOUBLE,                 -- 融资余额
  net_mf_amount DOUBLE,                  -- 主力净流入
  PRIMARY KEY(ts_code, trade_date)
);

-- 指数日线（000300.SH 沪深300 / 000905.SH 中证500 / 000852.SH 中证1000 / 000985.CSI 中证全指）
index_daily(
  ts_code VARCHAR, trade_date DATE,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE, amount DOUBLE,
  pe_ttm DOUBLE,                         -- 指数估值，用于 ERP 计算
  PRIMARY KEY(ts_code, trade_date)
);
```

### 4.3 股票池定义（PIT 函数）

`query.get_universe(as_of_date)` 逐条剔除：

| 剔除项 | 规则 | 防的是 |
|---|---|---|
| 已退市 | `delist_date IS NULL OR delist_date > as_of_date` | 幸存者偏差 |
| 未上市 | `list_date <= as_of_date` | 前视偏差 |
| 次新股 | `list_date <= as_of_date - 250 自然日` | 上市初期定价失真 |
| ST / *ST | `stock_status` 在 `as_of_date` 生效的状态不含 ST | 退市风险 + 5% 涨跌停规则不同 |
| 停牌 | `is_suspended = FALSE` | 不可交易 |
| 低流动性 | 20 日均成交额位于横截面后 20% | 冲击成本不可控 |

### 4.4 落地校验（`ashare/data/validate.py`，每次入库后自动跑）

| 校验 | 判据 | 失败动作 |
|---|---|---|
| 行数完整性 | `count(daily_bar)` == `Σ(每股在市交易日数)`，**误差为 0**（D9 要求停牌日补占位行） | 阻断，报告缺失区间 |
| 占位行合规 | 所有 `is_suspended=TRUE` 的行满足 `vol=0 AND amount=0 AND open=high=low=close` | 阻断 |
| 复权因子跳变 | 相邻交易日 `adj_factor` 比值 ∉ [0.5, 2.0] | 告警，需匹配到分红送转事件才放行 |
| 财报公告日 | `financial_pit.ann_date` 缺失率 = 0 | 阻断 |
| 宏观公布日 | `macro_indicator.publish_date` 缺失率 = 0 | 阻断 |
| 双源交叉 | 抽样 200 只股票 × 100 日，Tushare vs BaoStock 后复权收盘价偏差 < 0.5% | 告警 |
| 涨跌停价 | `limit_up/limit_down` 缺失率 = 0（主板 10%、创业板/科创板 20%、ST 5%、北交所 30%） | 阻断 |

---

## 5. P2 · 因子库与回测引擎

### 5.1 因子清单（周频有效）

**量价类**

| 因子 | 定义 | 方向 |
|---|---|---|
| `reversal_20` | −1 × 过去 20 交易日累计收益 | 正（A 股最强异象之一） |
| `momentum_120_20` | 过去 120 日收益 − 过去 20 日收益（跳过近月避开反转） | 正 |
| `volatility_60` | 过去 60 日日收益标准差 | 负（低波异象） |
| `turnover_20` | 过去 20 日平均换手率 | 负（A 股高换手负收益极显著） |
| `amihud_20` | `mean(\|ret\| / amount)` 20 日均值 | 正（非流动性溢价） |
| `max_ret_20` | 过去 20 日单日最大涨幅 | 负（彩票效应） |

**基本面类**（全部走 `financial_pit`，PIT 取数）

| 因子 | 定义 | 方向 |
|---|---|---|
| `ep_ttm` | 1 / PE_TTM | 正 |
| `bp` | 1 / PB | 正 |
| `sp_ttm` | 1 / PS_TTM | 正 |
| `roe_ttm` | TTM 归母净利 / 平均净资产 | 正 |
| `gross_margin` | 毛利率 | 正 |
| `accrual` | (净利润 − 经营现金流) / 总资产 | 负（盈利质量） |
| `np_yoy` | 归母净利同比 | 正 |
| `sue` | (本期净利 − 去年同期) / 过去 8 期同比增速标准差 | 正（盈余惯性） |

**资金/情绪类**

| 因子 | 定义 | 方向 |
|---|---|---|
| `north_hold_chg_20` | 北向持股比例 20 日变化 | 正 |
| `margin_chg_20` | 融资余额 20 日变化率 | 待检验 |

**风险因子**（用于中性化，**不作为 alpha**）：`log_mv`（ln 总市值）、`industry`（申万一级 dummy）、`beta_250`（对中证全指）。

### 5.2 因子处理链（顺序不可调换）

```
1. 计算原始值      f(as_of_date, universe) -> Series
2. MAD 去极值      median ± 3 × 1.4826 × MAD      ← 不用 3σ，A 股尾部太厚
3. 行业+市值中性化  对 [log_mv, 申万一级 dummy] 做横截面 OLS，取残差
4. zscore 标准化   (x - mean) / std
5. 缺失值填 0      中性化后 0 = 行业均值
```

多因子合成：默认**等权合成 zscore 之和**。理由：IC 加权 / 最大化 IR 的权重在样本外极不稳定，等权是最难被过拟合的基线。若要用加权，必须先通过 §5.5 的全部五闸。

### 5.3 回测引擎语义（核心 < 400 行）

```python
for rebalance_date in weekly_dates:              # 每周最后一个交易日
    universe   = query.get_universe(rebalance_date)
    scores     = combine_factors(rebalance_date, universe)
    target_pos = macro_timing(rebalance_date)                  # 0.2 ~ 1.0
    weights    = build_portfolio(scores, target_pos, constraints)

    exec_date  = next_trade_date(rebalance_date)               # T+1
    # D6：涨跌停 / 停牌不可交易
    tradable   = mask_untradable(exec_date, weights, prev_weights)
    fills      = apply_open_price(exec_date, tradable)         # 开盘价成交
    cost       = trade_cost(fills)
    equity    *= (1 + period_return(fills) - cost)
```

**成本模型**

| 项 | 费率 |
|---|---|
| 佣金（双边） | 0.025%（最低 5 元，组合级忽略） |
| 印花税（仅卖出） | 0.05% |
| 过户费（沪市） | 0.001% |
| 冲击成本 | `0.5 × (委托金额 / 当日成交额) × 当日振幅`，封顶 0.3% |

一个往返约 **0.3%**。周频年化换手 10~20 倍 → 成本拖累 3~6%。

**退市处理**：`delist_date` 前最后一个交易日按收盘价强制清仓。

### 5.4 评估指标

| 类别 | 指标 |
|---|---|
| 净值 | 年化收益、年化波动、Sharpe、Calmar、最大回撤及区间、月度胜率 |
| 相对 | 超额收益（vs 中证全指 / 沪深300）、信息比率 IR、跟踪误差 |
| 因子 | IC 均值、RankIC 均值、ICIR、IC 胜率、10 分层年化收益单调性 |
| 交易 | 年化双边换手、平均持仓数、成本拖累占比 |
| 归因 | Brinson 行业归因 + 风格因子回归归因 |

### 5.5 防自欺五闸（`ashare/backtest/guards.py`）

| 闸 | 做法 | 通过标准 |
|---|---|---|
| 1 · 样本外一次 | 2010–2019 调参，2020–至今样本外 | 样本外 Sharpe ≥ 样本内 × 0.6 |
| 2 · Walk-forward | 滚动 5 年训练 + 1 年测试，逐年前滚 | 各年份参数不发生量级跳变 |
| 3 · Shuffle 对照 | 每个调仓日横截面打乱因子分数，重跑 200 次 | 真实 Sharpe 在 shuffle 分布 95 分位之上 |
| 4 · 成本敏感 | 成本翻倍（往返 0.6%）重跑 | 策略仍为正超额 |
| 5 · 参数高原 | 核心参数 ±30% 网格扫描 | 收益呈平缓高原，非尖峰 |

任一闸未过 → 该策略**不得进入 P3 生产信号**。

### 5.6 引擎正确性验收（用已知异象反测引擎）

A 股短期反转（`reversal_20`）与高换手负收益（`turnover_20`）是文献中极其稳健的结论。
若 10 分层回测跑不出显著单调性，**判定为数据或引擎有 bug**，而不是"市场变了"。这是 P2 的硬性验收项。

**验收窗口锁定 2010-01-01 – 2019-12-31。** 2020 年后（注册制、量化拥挤、小盘因子衰减）
这些异象确有真实弱化，若把验收窗口开到今天，等于要求工程师"调数据直到跑出结果"——
把一个防自欺的检验变成自欺的来源。2020 年后的衰减单独记为**观察项**，不作为验收门槛。

---

## 6. P3 · 策略层

### 6.1 宏观择时层（决定总仓位）

输入 5 个 PIT 宏观指标，各自转为**滚动 5 年历史分位数**（禁用全样本分位数 —— 那是前视）：

| 指标 | 含义 | 方向 |
|---|---|---|
| `ERP` = 1/PE_TTM(中证全指) − 10Y 国债 | 股债性价比 | 高 → 加仓 |
| `m1_m2_gap` = M1同比 − M2同比 | 资金活化程度（A 股经典领先指标） | 高 → 加仓 |
| `tsf_yoy_chg` = 社融存量同比的 3 月变化 | 信用扩张/收缩 | 上升 → 加仓 |
| `north_flow_60` = 北向 60 日净流入 / 流通市值 | 外资流向 | 高 → 加仓 |
| `trend_ma200` = 中证全指 close / MA200 | 趋势确认，防下跌趋势中抄底 | > 1 → 加仓 |

每个指标按分位映射为 0 / 0.5 / 1 分，等权求和得 `score ∈ [0,1]`：

```
目标总仓位 = 20% + 80% × score
```

下限 20% 的作用是防止完全空仓错过 V 型反弹（A 股底部反弹极快，空仓成本很高）。
**明确禁止在这一层引入任何 ML 模型（N5）。**

#### 关停规则（本层必须自证价值，否则默认关闭）

宏观层是全系统**统计功效最低**的模块（月频 15 年约 180 个样本点），却控制着**最大的杠杆**
（总仓位 20%–100%）。因此它不享有「默认启用」的待遇：

> 样本外区间内，若「宏观择时版」的 **Sharpe 与 Calmar 均未超过「固定 80% 仓位版」**，
> 则 $\pi_t \equiv 0.8$ 常数化，宏观层降级为纯展示（仪表盘仍显示 5 项分位，但不参与仓位计算）。

判定在样本外只做一次，结果记入 `docs/oos-runs.md`。**这不是保守，是承认 180 个样本点撑不起强主张。**

### 6.2 横截面选股层（决定个股权重）

- 合成因子分数取前 N 名（N 默认 50，需通过参数高原检验）
- 权重：默认**等权**；可选风险平价（1/σ 归一）
- 约束：单股 ≤ 5%；单申万一级行业 ≤ 20%；单周换手 ≤ 30%；剔除次日预期一字涨停标的

### 6.3 输出契约（这就是"买卖清单含时点"）

```json
{
  "as_of": "2026-08-14",
  "as_of_note": "周五收盘后计算",
  "execute_on": "2026-08-17T09:15:00+08:00",
  "execute_note": "周一 09:15-09:25 集合竞价阶段挂限价单，09:25 统一按开盘价撮合",
  "target_position": 0.65,
  "macro_score": {"ERP": 1.0, "m1_m2_gap": 0.5, "tsf_yoy_chg": 0.5,
                  "north_flow_60": 0.0, "trend_ma200": 1.0},
  "orders": [
    {
      "ts_code": "600519.SH", "name": "贵州茅台",
      "action": "BUY",
      "current_weight": 0.0, "target_weight": 0.045,
      "limit_price_range": [1580.0, 1620.0],
      "price_basis": "前收 ± 0.5 × ATR20",
      "factor_contrib": {"roe_ttm": 1.82, "ep_ttm": 0.91, "reversal_20": -0.34},
      "urgency": "normal"
    }
  ],
  "strategy_version": "v1.0.0",
  "param_hash": "sha256:a3f2..."
}
```

`param_hash` 用于确认样本外运行时参数未被改动（D7）。

### 6.4 成交回写与持仓校准（闭环，缺此项系统三个月即失真）

「不做自动下单」（N2）是设计取舍；「不知道实际成交了什么」是**缺陷**。两者必须拆开。

若无回写，第 t+1 期计算 `current_weight` 时用的是「假设全部按目标成交」的虚拟持仓。而现实中一定有：
一字涨停买不进、限价未触及、用户主动跳过某笔、部分成交。误差逐周累积且**单向放大**
（没买进的继续被当成持有 → 下期不再买入 → 永久缺仓）。

最小可行方案（不引入券商 API）：

| 方式 | 说明 |
|---|---|
| 对账单导入 | 用户从券商 App 导出当日成交/持仓 CSV → `POST /api/portfolio/reconcile` |
| 手工确认 | 信号看板每笔 order 提供「已成交 / 部分成交 / 未执行」三态勾选，默认未确认 |

`position_ledger(as_of_date, ts_code, shares, avg_cost, source)` 为**实际持仓的唯一真值**。
未回写的期次，系统必须在下期信号顶部显著标注「持仓未校准，本期信号可靠性下降」，
**不得静默按虚拟持仓继续推演**。

归属 P3，但表结构在 P1 阶段一并建好（后补迁移成本高于预建）。

---

## 7. P4 · 知识库

在现有 `rag/`（chromadb + sentence-transformers）基础上扩展：

- 语料：券商研报 PDF、财报全文、交易所公告、行业报告
- **每个 chunk 的 metadata 必须含 `publish_date`** —— 入库时写入几乎零成本，事后回补则需重新解析全部语料，**这一项不可延后**
- 检索接口的 `as_of_date` 过滤（`publish_date <= as_of_date`）成本较高，可延后到真正需要历史复盘时再做
- 未打 `publish_date` 的 chunk 在回测/复盘场景中直接排除，不做兜底猜测

**不做这一步的后果**：复盘 2021 年信号时，LLM 会读到 2023 年的研报，归因全部作废。

---

## 8. P5 · LLM 投研层（只读）

| 能力 | 优先级 | 输入 | 输出 |
|---|---|---|---|
| 信号解释 | P0 | 单条 order 的 `factor_contrib` | 因子贡献的自然语言表述 |
| 风险否决清单 | P0 | 财报异常 + 公告 + 质押/商誉数据 | 持仓个股的负面信号排查 |
| 组合周报 | P0 | 本周净值 + 归因结果 | 涨跌来源、行业/风格暴露变化 |
| 个股全量级报告 | P1 | P1 全量数据 + P2 因子分解 + P4 检索 | 财务/估值/行业对比/因子暴露/风险提示 |

**「风险否决清单」为什么不违反 D1**：它是**单向**的 —— 只能触发减仓或剔除候选，
永远不能触发加仓或提升权重。单向否决权不构成决策权。加仓路径依然只由数学模型掌握。

**个股全量级报告的真实价值**不在于影响单次买卖（那确实违反 D1 精神），而在于
**让用户理解自己持有什么，从而不在回撤中恐慌性偏离策略**。行为偏离是量化个人投资者
最大的收益漏损来源，远大于因子选择的差异。这是它值得做、但排在 P1 而非 P0 的理由。

**约束（D1）**：
- 解释文本由 `factor_contrib` 数值驱动，不允许 LLM 自行编造买入理由
- LLM 层不持有任何数据库写句柄
- 新增的 Agent 工具全部为只读，**不加入 `TRADING_TOOLS`**（它们本来就不能动钱）

---

## 9. P6 · 前端

复用现有 `static/`：

| 页面 | 内容 |
|---|---|
| 每日信号看板 | 目标仓位 + 调仓清单表格 + 宏观分项得分 |
| 回测报告页 | 净值曲线、回撤、分层收益、IC 时序、五闸检验结果 |
| 个股详情页 | K 线 + 因子暴露雷达 + 财务趋势 + LLM 深度报告 |
| 宏观仪表盘 | 5 个宏观指标的分位时序 + 历史仓位曲线 |

---

## 10. 与现有代码的整合

### 复用（不改动）
`server.py` FastAPI+SSE 框架、`auth/`、`cache.py`、`rag/` 检索链路、`memory/`

### 扩展
| 模块 | 改动 |
|---|---|
| `server.py` | 新增 `/api/signal/*`、`/api/backtest/*`、`/api/stock/{code}/report` |
| `rag/indexer.py` | chunk metadata 增加 `publish_date` |
| `rag/retriever.py` | 检索接口增加 `as_of_date` 过滤 |
| `quant_agent.py` | 新增**只读**工具：`query_universe` / `get_factor_exposure` / `get_signal_list` / `run_stock_report` |

### 不动
`brokers/` 全部（含 `risk_gate.py`、`intent_store.py`、各 adapter）—— 本期 A 股不接下单通道（N2）。

### 新增目录
```
ashare/
├── data/                 # P1
│   ├── sources/          # tushare.py / baostock.py / akshare.py  (可插拔 adapter)
│   ├── schema.sql
│   ├── ingest.py         # 增量落库
│   ├── validate.py       # 落地校验（§4.4）
│   └── query.py          # ★ 唯一数据出口，强制 as_of_date (D2)
├── factors/              # P2
│   ├── base.py           # Factor 抽象 + 注册表
│   ├── price.py / fundamental.py / flow.py / risk.py
│   └── pipeline.py       # 去极值 / 中性化 / 标准化
├── backtest/             # P2
│   ├── engine.py         # < 400 行核心
│   ├── cost.py / metrics.py
│   └── guards.py         # 防自欺五闸
├── strategy/             # P3
│   ├── macro_timing.py / stock_selection.py / portfolio.py
└── report/               # P5
    └── stock_deep.py
```

`CLAUDE.md` 需新增一节「A 股数据与回测铁律」，把 §2 的 D1–D8 写进去。

---

## 11. 交付顺序与验收

**本期范围：P1 + P2。** P1/P2 不对，P3–P6 全是沙上城堡。

### P1 验收（全部为可执行断言）

- [ ] 全 A（含已退市）2010-01-01 至今日线入库，行数与「交易日历 × 在市股票数」**完全相等**（D9：停牌日补占位行）
- [ ] `adj_factor` 跳变记录全部能匹配到分红送转事件
- [ ] `financial_pit.ann_date` 缺失率 = 0
- [ ] `macro_indicator` 覆盖 8 个指标，`publish_date` 缺失率 = 0
- [ ] `query.get_universe('2015-06-12')` 结果中不含 `list_date > 2014-10-05` 或 `delist_date <= 2015-06-12` 的股票（单测断言）
- [ ] `query.get_financial('600519.SH', as_of='2021-04-01')` 返回的 `end_date` 为 2020-12-31 且 `ann_date <= 2021-04-01`（PIT 断言）
- [ ] 双源交叉校验：200 只 × 100 日，后复权收盘价偏差 < 0.5%
- [ ] 停牌占位行断言：任取一只有长期停牌史的股票（如 `000562.SZ`），其 `daily_bar` 行数
      == 在市区间交易日数；且 `rolling(20)` 窗口跨越的日历跨度对全横截面一致（D9）
- [ ] ST 反推断言：`stock_status` 由 `namechange` 反推，覆盖戴帽/摘帽/二次戴帽/退市整理四种情形的单测
- [ ] 涨跌停兜底断言：`limit_up` 缺失时规则计算正确（创业板 2020-08-24 前后 10%/20% 切换、ST 5%）；
      规则也算不出时返回「不可交易」而非默认可交易
- [ ] 每次回测产物同时带 `param_hash` 与 `data_snapshot_id`（D7）

### P2 验收

- [ ] `reversal_20` 与 `turnover_20` 的 10 分层回测呈现显著单调性（§5.6 引擎正确性反测）
- [ ] 涨停买不进 / 跌停卖不出 / 停牌不可交易，各有单测
- [ ] Shuffle 对照 200 次可运行，输出 Sharpe 分布
- [ ] 全市场 15 年周频回测单次运行 < 60 秒
- [ ] 所有因子函数签名首参为 `as_of_date`（静态检查脚本）

---

## 12. 已知风险与限制

| 风险 | 说明 | 缓解 |
|---|---|---|
| 数据源单点 | Tushare 积分限流 / 接口变更 | adapter 可插拔 + BaoStock 交叉校验 |
| A 股结构变迁 | 注册制、涨跌停规则变化、退市新规 | walk-forward 检验参数稳定性；样本外区间恰好覆盖注册制时期 |
| 策略容量 | 小市值股容量有限 | 流动性剔除后 20%；资金规模超千万需重估 |
| 宏观择时天然弱 | 5 指标投票，样本点少 | 仓位下限 20%，避免择时失败造成极端后果 |
| 无自动执行 | 需人工在券商 App 下单 | 输出含明确挂单区间与执行时点（N2 为设计取舍，非缺陷） |
| 无审计日志 | 沿用现有仓库的已知缺口 | 信号输出带 `param_hash`，样本外运行记入 `docs/oos-runs.md` |
