# 样本外运行台账（D7）

> 样本外区间 2020-01-01 至今**只跑一次**。每次运行由 `run_backtest` **自动追加**一行（不靠人工）。
> 改任何参数后重跑，等于把样本外污染成样本内 —— 必须在此标注。
> `param_hash` 锁参数，`data_snapshot_id` 锁数据；两者任一不同，结果不可与历史行直接比较。

| 运行时间 (UTC) | strategy_version | param_hash | data_snapshot_id | engine_version | 区间 | Sharpe(IS) | Sharpe(OOS) | 五闸 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
