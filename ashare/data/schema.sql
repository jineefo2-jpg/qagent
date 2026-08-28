-- ashare/data/schema.sql
-- 表结构定义见 docs/specs/2026-08-18-ashare-quant-platform-design.md §4.2
-- 本文件是唯一 DDL 来源；改动必须同步 bump _db.SCHEMA_VERSION

CREATE TABLE IF NOT EXISTS calendar (
  trade_date      DATE PRIMARY KEY,
  is_open         BOOLEAN NOT NULL,
  pre_trade_date  DATE
);

CREATE TABLE IF NOT EXISTS stock_basic (
  ts_code      VARCHAR PRIMARY KEY,
  symbol       VARCHAR,
  name         VARCHAR,
  sw_l1        VARCHAR, sw_l2 VARCHAR, sw_l3 VARCHAR,
  market       VARCHAR,
  list_date    DATE,
  delist_date  DATE,               -- NULL = 在市 (D5)
  is_hs        VARCHAR,
  _ingested_at TIMESTAMP DEFAULT current_timestamp
);
-- v2：Tushare 自带行业分类（申万成分无权限时 industry_member 的降级来源）
ALTER TABLE stock_basic ADD COLUMN IF NOT EXISTS industry VARCHAR;

-- ST 状态历史。数据来源：由 namechange 反推（Tushare 无直接接口）
CREATE TABLE IF NOT EXISTS stock_status (
  ts_code    VARCHAR,
  start_date DATE,
  end_date   DATE,                 -- NULL = 至今
  status     VARCHAR,              -- NORMAL | ST | *ST | DELIST_PERIOD | S
  PRIMARY KEY (ts_code, start_date)
);

CREATE TABLE IF NOT EXISTS daily_bar (
  ts_code      VARCHAR,
  trade_date   DATE,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
  vol DOUBLE, amount DOUBLE,
  adj_factor   DOUBLE,             -- 后复权因子 (D8)
  limit_up     DOUBLE,
  limit_down   DOUBLE,
  limit_source VARCHAR,            -- 'api' | 'rule' | 'unknown'  (B2)
  is_suspended BOOLEAN NOT NULL DEFAULT FALSE,   -- 占位行标记 (D9)
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS daily_basic (
  ts_code VARCHAR, trade_date DATE,
  turnover_rate DOUBLE, turnover_rate_f DOUBLE, volume_ratio DOUBLE,
  pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE, ps DOUBLE, ps_ttm DOUBLE,
  dv_ratio DOUBLE, dv_ttm DOUBLE,
  total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
  total_mv DOUBLE, circ_mv DOUBLE,
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (ts_code, trade_date)
);

-- ★ PIT 核心表：主键含 ann_date (D3)
CREATE TABLE IF NOT EXISTS financial_pit (
  ts_code     VARCHAR,
  ann_date    DATE,
  end_date    DATE,
  report_type VARCHAR,
  update_flag INTEGER,
  total_revenue DOUBLE, revenue DOUBLE, operate_profit DOUBLE,
  total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE,
  total_assets DOUBLE, total_liab DOUBLE, total_hldr_eqy_exc_min_int DOUBLE,
  n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE, n_cash_flows_fnc_act DOUBLE,
  roe DOUBLE, roa DOUBLE, grossprofit_margin DOUBLE, netprofit_margin DOUBLE,
  debt_to_assets DOUBLE, current_ratio DOUBLE,
  or_yoy DOUBLE, netprofit_yoy DOUBLE, basic_eps DOUBLE, bps DOUBLE,
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (ts_code, ann_date, end_date, report_type, update_flag)
);

-- ★ 宏观 PIT (D4)
CREATE TABLE IF NOT EXISTS macro_indicator (
  indicator            VARCHAR,
  period               DATE,
  publish_date         DATE,
  value                DOUBLE,
  publish_date_source  VARCHAR,    -- 'observed' | 'rule'
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (indicator, period, publish_date)
);

-- 本期只入 hk_hold_ratio（架构师 A1：margin_* 待因子过闸后再补，回补幂等）
CREATE TABLE IF NOT EXISTS money_flow (
  ts_code VARCHAR, trade_date DATE,
  hk_hold_ratio DOUBLE,
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (ts_code, trade_date)
);

-- 行业成分历史（PIT）：申万成分 in_date/out_date；无权限时由 stock_basic.industry 降级生成，
-- 且 _meta.industry_source 必须写 'tushare_static' 显式记录降级
CREATE TABLE IF NOT EXISTS industry_member (
  ts_code  VARCHAR,
  sw_l1    VARCHAR, sw_l2 VARCHAR, sw_l3 VARCHAR,
  in_date  DATE,
  out_date DATE,                   -- NULL = 至今
  PRIMARY KEY (ts_code, in_date)
);

CREATE TABLE IF NOT EXISTS index_daily (
  ts_code VARCHAR, trade_date DATE,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE, amount DOUBLE,
  pe_ttm DOUBLE,
  _ingested_at TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (ts_code, trade_date)
);

-- ★ 拉取状态机（架构师 B1）：断点续传 + 数据快照可追溯
CREATE TABLE IF NOT EXISTS ingest_log (
  job_id      VARCHAR,             -- '<table>:<partition_key>'  例 'daily_bar:2015'
  table_name  VARCHAR,
  partition   VARCHAR,
  state       VARCHAR,             -- PENDING|RUNNING|DONE|RETRY|SUSPECT|FAILED
  attempts    INTEGER DEFAULT 0,
  rows_written INTEGER DEFAULT 0,
  last_error  VARCHAR,
  started_at  TIMESTAMP,
  finished_at TIMESTAMP,
  PRIMARY KEY (job_id)
);

CREATE TABLE IF NOT EXISTS _meta (
  key VARCHAR PRIMARY KEY,
  value VARCHAR
);

-- 发布台账（v3，2026-08-28）。每次 promote 追加一行，记录**这次发布影响的最早可见日期**。
--
-- ★ 它存在的唯一理由：`snapshot_id` 是【全库】指纹，表达不了「历史那部分没变」。
--   每日增量只往尾部追加新交易日（实测：8-24~8-27，老数据一行没动），却会让指纹改变，
--   于是 3300 万行历史因子集体被判陈旧、需要十几小时全量重建 —— 每天一次，不可持续。
--   有了这张表，因子行的有效性判据变成：
--       行有效 ⟺ 快照与当前相同  或  行的 trade_date < 它建成之后每一次发布的 min_affected_date
--   日增量的 min_affected_date 是「新交易日」，历史因子因此不必重建。
--
-- ★ 一个必须说准的例外（不要写成「历史因子永远有效」）：财报按周节流全市场扫描时
--   fin_start = start − 120 天，而 _upsert 是 INSERT OR REPLACE —— 内容没变的行也会
--   刷新 _ingested_at，于是那一周的 min_affected 会被拽回约 120 天前，失效一个滚动
--   四个月的尾巴（约 17 个周频日期 / 3% 的因子行）。判据本身没错（重述值确实可能变），
--   但代价要如实说：**每周失效一个 120 天尾巴，其余永远有效**。
--
-- ★ 已知天花板（有意）：参考表（calendar / stock_basic / stock_status / industry_member）
--   每次 ingest 都被整表 INSERT OR REPLACE，本判据覆盖不到它们的【历史】修正
--   （申万成分回溯调整、namechange 补录历史 ST 段、list_date 勘误 —— 一年一两次，
--   但后果是 processed_value 静默错，因为中性化/zscore 全是横截面统计量）。
--   ⚠ 别按「给这四张表加 _ingested_at（schema v4）」去补 —— 那条路是错的：它们每天
--   被整表重写，加了时间戳只会让 min_affected 每天回到 2010，整个机制当场作废。
--   正确的补法是在 promote 时 ATTACH 上一版 market 做一次内容 diff（四张表最大 1 万行，
--   毫秒级），把「历史行真的变了」压成那一天。见 promote._log_snapshot 的 TODO。
CREATE TABLE IF NOT EXISTS snapshot_log (
  snapshot_id       VARCHAR PRIMARY KEY,   -- 这次发布之后的数据指纹
  promoted_at       TIMESTAMP NOT NULL,
  min_affected_date DATE                   -- 本次写入影响的最早【可见】日期；NULL = 无法判定（保守：全部失效）
);
