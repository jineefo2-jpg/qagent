-- ashare/data/ledger_schema.sql
-- ledger 库（信号清单 + 实际持仓 + 逐单确认）的唯一 DDL 来源。
--
-- ★ 为什么是第三个库（P3 Task 0 对计划 V1 的修正）：
--   market 由 promote 整体替换（用户数据跟着换 = 丢）；derived 的既有契约是
--   「缓存，可随时 rm 重算」（derived_schema.sql 头注释）—— 而这里的三张表是
--   【不可重算的用户资产】：清单是某时刻的决策记录，持仓/确认是用户的手工输入。
--   所以本库与 market 同款 schema_version 守卫（_ledger.init_schema），永不可 rm。
--
-- ★ 本库只增改不删：状态修正 = 再写一行/覆盖同键，历史由 created_at 见证。

CREATE TABLE IF NOT EXISTS _meta (
  key   VARCHAR PRIMARY KEY,
  value VARCHAR
);

-- 调仓清单（规格 §6.3 的 JSON 全文 + 提出来便于索引的列）。
-- D7：param_hash 与 data_snapshot_id 缺一不可，用 NOT NULL 约束而不是写入方自觉。
CREATE TABLE IF NOT EXISTS signal_plan (
  as_of_date       DATE,
  param_hash       VARCHAR NOT NULL,
  execute_on       VARCHAR NOT NULL,      -- ISO 带时区的执行时点（§6.3 execute_on 原文）
  data_snapshot_id VARCHAR NOT NULL,
  strategy_version VARCHAR NOT NULL,
  plan_json        VARCHAR NOT NULL,      -- §6.3 契约全文
  created_at       TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (as_of_date, param_hash)    -- 同参数重生成 = 幂等覆盖；参数变了是新行（D7 台账连续）
);

-- 实际持仓的唯一真值（规格 §6.4）。source 标注这行是怎么来的 ——
-- 'signal_assumed'（按清单假定成交，最弱）必须能与人工来源区分开，
-- 否则「持仓未校准」的警示无从判起。
CREATE TABLE IF NOT EXISTS position_ledger (
  as_of_date  DATE,
  ts_code     VARCHAR,
  shares      DOUBLE NOT NULL,
  avg_cost    DOUBLE,                     -- 券商对账单可能不给成本，允许 NULL
  source      VARCHAR NOT NULL CHECK (source IN ('reconcile_csv', 'manual_confirm', 'signal_assumed')),
  created_at  TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (as_of_date, ts_code)
);

-- 逐单三态确认（规格 §6.4：已成交 / 部分成交 / 未执行）。
CREATE TABLE IF NOT EXISTS order_confirm (
  as_of_date    DATE,
  ts_code       VARCHAR,
  state         VARCHAR NOT NULL CHECK (state IN ('filled', 'partial', 'skipped')),
  filled_shares DOUBLE,                   -- partial 才有意义；filled/skipped 允许 NULL
  note          VARCHAR,
  created_at    TIMESTAMP DEFAULT current_timestamp,
  PRIMARY KEY (as_of_date, ts_code)
);
