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
