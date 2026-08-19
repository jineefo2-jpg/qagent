-- ashare/data/derived_schema.sql
-- derived 库（因子值 + 回测运行）的唯一 DDL 来源。
-- 与 market 库分文件：market 由 promote.py 用 os.replace 整体替换，
-- 因子值若住在 market 里，每次 promote 都会被一起换掉。

-- ★ 主键【不含】snapshot_id：同因子 + 同参数 + 同交易日 + 同股票只有一行。
--   换数据快照重算 = 覆盖旧值，snapshot_id 列记录这行是哪份数据算出来的。
--   snapshot_id 进主键的后果是同一天堆 N 份因子值，取数得先挑快照，
--   挑错了回测静默用了另一份数据 —— 溯源要的是一列，不是分身。
CREATE TABLE IF NOT EXISTS factor_value (
  factor_name     VARCHAR,
  param_hash      VARCHAR,          -- 因子参数的哈希（同因子不同窗口期是不同的行）
  trade_date      DATE,
  ts_code         VARCHAR,
  raw_value       DOUBLE,           -- 原始因子值
  processed_value DOUBLE,           -- 去极值 / 标准化 / 中性化之后
  snapshot_id     VARCHAR,          -- 算这行时用的 market 库快照 (D7 溯源)
  PRIMARY KEY (factor_name, param_hash, trade_date, ts_code)
);

-- 回测运行台账。param_hash 与 data_snapshot_id 必须同时记录 —— D7 缺一不可：
-- 只有参数没有数据快照，等于换了一批数据跑出的结果被当成同一次实验。
CREATE TABLE IF NOT EXISTS backtest_run (
  run_id           VARCHAR PRIMARY KEY,
  param_hash       VARCHAR,
  data_snapshot_id VARCHAR,
  engine_version   VARCHAR,
  started_at       TIMESTAMP,
  elapsed_sec      DOUBLE,
  config_json      VARCHAR,
  metrics_json     VARCHAR,
  is_oos           BOOLEAN          -- 样本外：同一 (param_hash, data_snapshot_id) 只该有一次 (D7)
);
