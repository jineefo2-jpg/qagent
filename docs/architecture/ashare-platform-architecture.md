# 架构方案: A 股全市场量化策略平台

- 日期：2026-08-18 · 状态：待评审
- 上游契约：[`docs/specs/2026-08-18-ashare-quant-platform-design.md`](../specs/2026-08-18-ashare-quant-platform-design.md)（下称「规格」）
- 关联：[`docs/technical-architecture.md`](../technical-architecture.md)（现有 QuantAgent 系统）、[ADR-0003](../adr/0003-ashare-data-and-backtest.md)
- 本期范围：**P1 数据底座 + P2 因子库/回测引擎**。P3/P4/P5/P6 只定契约边界，不实现。

> **本文不重复规格已写清的内容**：表结构（规格 §4.2）、因子清单（§5.1）、铁律 D1–D8（§2）、
> 成本模型（§5.3）、五闸标准（§5.5）、验收断言（§11）—— 一律引用，不复制。
> 本文只写规格**没写透的工程部分**：接口契约、调度时序、集成契约、隔离机制、部署运维，
> 外加 §12 的架构评审意见（该砍的砍、该补的补）。

---

## 0. 文档导览

| 章节 | 内容 | 主要读者 |
|---|---|---|
| §1 | 需求摘要与量化指标（容量/性能预算） | 全员 |
| §2 | 总体架构图 + 信任边界 | 全员 |
| §3 | 模块划分、职责、依赖方向 DAG | 全员 |
| §4 | **模块接口契约**（query / factor / backtest，可直接照着实现） | 后端 + 量化 |
| §5 | **数据流与调度时序**（增量、重试、限流、全量回补） | 数据工程 |
| §6 | **与 server.py / quant_agent.py 的集成契约** | 后端 |
| §7 | **D1 物理隔离的工程落地**（可被 CI 检查） | 全员 |
| §8 | 安全设计 | 后端 |
| §9 | 性能与容量 | 全员 |
| §10 | 部署与运维（DuckDB 并发、备份、回滚） | 运维 |
| §11 | 演进路径（台阶触发条件） | 架构 |
| §12 | **架构风险评审：砍掉什么 / 补上什么** | PM + 全员 |
| §13 | 风险与未决项 | 全员 |

---

## 1. 需求摘要与量化指标

### 1.1 从规格推导的容量指标

| 项 | 量级 | 推导 |
|---|---|---|
| 股票全集（含已退市） | ≈ 5,700 只 | 在市 ≈ 5,400 + 2010 以来退市 ≈ 300 |
| 交易日 | ≈ 4,030 天 | 2010-01-01 ~ 2026-08，16.6 年 × 243 |
| 平均在市数 | ≈ 3,000 只 | 2010 年 ≈ 1,700 → 2026 年 ≈ 5,400 |
| `daily_bar` | ≈ 1,250 万行（+3% 停牌补行） | 4,030 × 3,000 |
| `daily_basic` | ≈ 1,210 万行 | 同上（停牌日无估值行） |
| `financial_pit` | ≈ 44 万行 | 5,700 × 67 报告期 × 1.15（重述） |
| `money_flow`（仅北向） | ≈ 500 万行 | 2016-12 起，2,300 日 × 2,200 只 |
| `macro_indicator` | ≈ 1,700 行 | 8 指标 × 200 期 |
| `index_daily` | ≈ 1.6 万行 | 4 指数 × 4,030 |
| **`factor_value`（派生，最大表）** | **≈ 3,900 万行** | 16 因子 × 780 周 × 平均池 3,100 |
| `market.duckdb` 体积 | 0.6 ~ 0.9 GB | 列存 + 压缩 |
| `derived.duckdb` 体积 | 0.4 ~ 0.7 GB | factor_value 为主 |
| 周频调仓日 | 780 个 | 4,030 / 5.17 |

> 规格 §4.1 写「Parquet < 1GB」，量级判断正确。**但规格漏算了 `factor_value`** —— 它是全库最大的表，
> 且是派生数据，必须与原始数据分文件存储（见 §10.2）。

### 1.2 性能预算（每条都是可测断言）

| 链路 | 预算 | 依据 / 备注 |
|---|---|---|
| 单次全量周频回测（含 IC / 分层 / 归因诊断） | **< 60 s** | 规格 §11 硬指标。前提：因子已预计算落库（§12.2 补 B4） |
| 回测「净值-only」模式（`compute_diagnostics=False`） | **< 8 s** | Shuffle 200 次 / 参数网格扫描的前提 |
| Shuffle 对照 200 次（闸 3） | < 30 min | 200 × 8 s + 打乱开销 |
| 因子全量预计算（16 因子 × 780 周） | < 20 min | 一次性；数据快照变更时增量重算 |
| 每日增量 ingest（拉取→校验→入库→算因子） | **< 20 min** | 收盘后 18:00–18:30 窗口 |
| 首次全量回补 15 年 | 3 ~ 9 h | 见 §5.4，受 Tushare 限频支配 |
| `GET /api/ashare/universe`、`/factor-exposure` P99 | < 300 ms | 单日查询，DuckDB 单表扫描 |
| `GET /api/ashare/health` P99 | < 100 ms | 读 `ingest_log` 摘要，不扫数据表 |
| Agent 只读工具单次调用 | < 1.5 s | 不阻塞 LLM 流式输出感知 |

### 1.3 规模前提（决定「不做什么」）

| 维度 | 本期规模 | 直接后果 |
|---|---|---|
| 用户数 | **1**（作者本人） | 不做多租户隔离、不做配额、不做作业队列服务 |
| 并发回测 | **≤ 1** | 单线程 executor 足够，不引入 Celery/RQ/多进程池 |
| 部署 | **单机**（macOS 开发机 / 一台 VPS） | 不引入 K8s / Docker Compose 多服务 / 消息队列 |
| 数据更新频率 | 每交易日 1 次 | 不做流式管道，`cron` + 单脚本足够 |
| 资金规模 | 个人账户 | 容量约束按规格 §12「千万以下」处理，不做容量建模 |

---

## 2. 总体架构图

```
                        ┌─────────────── 信任边界 A：外部数据（不可信输入）───────────────┐
                        │  Tushare Pro API   BaoStock   akshare   研报 PDF（人工投放）   │
                        └───────┬──────────────┬───────────┬──────────────┬─────────────┘
                                │              │           │              │
  ══════════════════════════════▼══════════════▼═══════════▼══════════════▼══════════════
   写侧（离线，cron 驱动，独占写锁）                                       │
  ┌──────────────────────────────────────────────────────────────────┐     │
  │ ashare/data/ingest.py         ← 唯一持有 DuckDB 写句柄的模块       │     │
  │   sources/tushare.py · baostock.py · akshare.py（可插拔 adapter） │     │
  │   → 影子文件写入 → validate.py（§4.4 六项）→ 原子替换             │     │
  │   → ingest_log 记录每个 (table, partition) 的状态（断点续传）      │     │
  └────────────────────────────┬─────────────────────────────────────┘     │
                               │ os.replace()（原子）                       │
  ┌────────────────────────────▼─────────────────────────────────────┐     │
  │ data/ashare/market.duckdb   （原始数据，read_only 对所有读者）     │     │
  │ data/ashare/derived.duckdb  （factor_value / backtest_run）       │     │
  │ data/ashare/parquet/        （冷备导出，按年分区，灾备用）         │     │
  └────────────────────────────┬─────────────────────────────────────┘     │
  ═════════════════════════════▼═══════════════════════════════════════════│═════════
   读侧（在线 + 离线计算，全部 read_only 连接）                              │
                               │                                           │
  ┌────────────────────────────▼─────────────────────────────────────┐     │
  │ ashare/data/query.py  ★ 唯一数据出口 · 强制 as_of_date（D2）      │     │
  │   对外只暴露 §4.1 的函数；内部 duckdb.connect(read_only=True)     │     │
  └───────┬──────────────────────────────────┬───────────────────────┘     │
          │                                  │                             │
  ┌───────▼────────────────┐    ┌────────────▼────────────────────┐        │
  │ ashare/factors/        │    │ ashare/backtest/                │        │
  │  base(注册表)/pipeline │───▶│  engine · cost · metrics ·guards│        │
  │  price/fundamental/... │    │  → BacktestResult               │        │
  └───────┬────────────────┘    └────────────┬────────────────────┘        │
          │                                  │                             │
  ┌───────▼──────────────────────────────────▼───────────────────┐         │
  │ ashare/strategy/  （P3，本期不实现，契约见 §4.4）              │         │
  └───────────────────────────┬──────────────────────────────────┘         │
                              │                                            │
  ══════════════════════════ 信任边界 B：LLM 只读（D1）══════════════════════│═══════
                              │ 只读，无写句柄，无写路径                     │
  ┌───────────────────────────▼──────────────────────────────────┐  ┌──────▼──────┐
  │ ashare/agent_tools.py   4 个只读工具                          │  │ rag/        │
  │   query_universe · get_factor_exposure                       │◀─│ 检索按       │
  │   (get_signal_list · run_stock_report → P3/P5 后注册)         │  │ publish_date│
  └───────────────────────────┬──────────────────────────────────┘  └─────────────┘
                              │ 合并进 TOOL_REGISTRY / TOOL_SCHEMAS
  ┌───────────────────────────▼──────────────────────────────────────────────────┐
  │ 现有 QuantAgent（不改动核心）                                                  │
  │   server.py（FastAPI + SSE）· quant_agent.py（主循环 + dispatch_tool）        │
  │   + 新增 /api/ashare/* 只读端点 + POST /api/ashare/backtest                    │
  │   ✕ brokers/ 完全不接触（规格 N6）                                            │
  └──────────────────────────────────────────────────────────────────────────────┘
```

**两条信任边界的含义**

| 边界 | 规则 | 强制手段 |
|---|---|---|
| A · 外部数据 | 任何外部返回值先经 `validate.py`，不通过不进主库 | 影子文件 + 原子替换（校验失败则主库不变） |
| B · LLM 只读（D1） | LLM 侧模块无写句柄、无写路径、无写依赖 | read_only 连接（运行时硬闸）+ 导入方向静态检查（CI）+ 注册表隔离测试（§7） |

---

## 3. 模块划分与职责

### 3.1 模块表

| 模块 | 职责 | 对外接口 | 允许写 | 负责角色 |
|---|---|---|---|---|
| `ashare/data/sources/*` | 各数据源 adapter，把外部 API 结果规范化成统一 DataFrame | `fetch_<dataset>(**partition) -> pd.DataFrame` | ✗ | 数据工程 |
| `ashare/data/ingest.py` | 分区拉取、幂等落库、`ingest_log` 状态机、影子文件与原子替换 | CLI `python -m ashare.data.ingest` | **✓ 唯一写者** | 数据工程 |
| `ashare/data/validate.py` | 规格 §4.4 六项校验 + 阻断/告警分级 | `validate(conn, scope) -> ValidationReport` | ✗ | 数据工程 |
| `ashare/data/limits.py` | 涨跌停价规则兜底计算（§12.2 补 B2） | `limit_prices(...)` | ✗ | 数据工程 |
| `ashare/data/query.py` | **唯一数据出口**，PIT 语义、股票池、复权、TTM | §4.1 全部签名 | ✗ | 后端 |
| `ashare/factors/base.py` | `@factor` 注册表 + `FactorSpec` | §4.2 | ✗ | 量化 |
| `ashare/factors/pipeline.py` | 去极值 → 中性化 → 标准化（顺序固定） | `process()` | ✗ | 量化 |
| `ashare/factors/{price,fundamental,flow,risk}.py` | 因子实现（签名固定） | `@factor` 装饰的函数 | ✗ | 量化 |
| `ashare/data/derived_store.py` | 派生库读写（**独家持 duckdb**） | `write_factor_values()/read_factor_values()/coverage_report()` | ✓ derived 库 | 量化 |
| `ashare/factors/store.py` | 编排「算 → 写」，**不碰 duckdb** | `build()` | ✗（经 derived_store） | 量化 |
| `ashare/backtest/engine.py` | 主循环（≤ 400 行，仅编排） | `run_backtest(config)` | ✗ | 量化 |
| `ashare/backtest/{cost,portfolio,metrics,guards}.py` | 成本 / 组合构建 / 指标 / 五闸 | §4.3 | ✗ | 量化 |
| `ashare/backtest/store.py` | 回测结果持久化 | `save()/load()/list_runs()` | ✓ derived 库 | 量化 |
| `ashare/agent_tools.py` | 4 个 LLM 只读工具，返回 dict 不抛异常 | `ASHARE_TOOL_REGISTRY/SCHEMAS` | ✗ | 后端 |
| `server.py`（扩展） | `/api/ashare/*` 端点 | §6.1 | ✗ | 后端 |

### 3.2 依赖方向 DAG（这是 D1 与分层的地基）

```
sources ──▶ ingest ──▶ validate            [写侧，独立进程]
                │
                ▼  (仅通过文件)
             query  ◀──── limits
               ▲ ▲ ▲
               │ │ └──────── agent_tools ──▶ server.py
               │ └────────── factors ──▶ factors.store
               └──────────── backtest ──▶ backtest.store
                                  ▲
                              strategy (P3)
```

**硬约束（CI 检查，见 §7.2）**

| # | 规则 | 违反后果 |
|---|---|---|
| L1 | **只有 `ashare/data/**` 可以 `import duckdb`**（2026-08-21 收紧：原来给 `factors/store.py`、`backtest/store.py` 开的口子已关闭 —— 一旦这两层拿到 duckdb，它们同样能连 market.duckdb 读未掩码的原始行） | 绕过唯一出口 |
| L2 | `ashare/data/**` 不得 import `ashare.factors` / `ashare.backtest` / `ashare.strategy` / `ashare.agent_tools` | 循环依赖 |
| L3 | `ashare/factors/**` 只能 import `ashare.data.query`、`ashare.factors.*` | 因子直连数据库（违 D2） |
| L4 | `ashare/agent_tools.py` 只能 import `ashare.data.query`、`ashare.factors`、`ashare.backtest.{store,metrics}`；**禁止 import `ashare.data.ingest`、`ashare.factors.store`、`ashare.backtest.store` 的写函数、`brokers.*`** | 违 D1 |
| L5 | `ashare/**` 不得 import `server` / `quant_agent` | 反向依赖 |
| L6 | `ashare/backtest/engine.py` 行数 ≤ 400（不含空行注释） | 规格 §5.3；见 §12.1 砍 A5 的口径修正 |

---
## 4. 模块接口契约

> **语言约定**：`ashare/**` 每个文件首行 `from __future__ import annotations`，因此注解可用 `list[str] | None`
> 新语法（Python 3.9 兼容，注解不求值）。
> **例外**：`server.py` 里的 pydantic 模型**必须**用 `Optional[X]` / `List[X]` —— pydantic 在 3.9 下
> 会运行时求值注解，`X | None` 直接抛 `TypeError`。这是本仓库 3.9.6 环境下的真实坑。

### 4.1 `ashare/data/query.py` —— 全系统唯一数据出口

```python
from __future__ import annotations
import datetime as _dt
from typing import Any, Callable, Iterable, Mapping, Sequence, Union
import pandas as pd

DateLike = Union[str, _dt.date]        # "2015-06-12" | datetime.date(2015, 6, 12)

# ══════════════════════════════════════════════════════════════════
# 异常（query 层向调用方 raise；只有 agent_tools 层才包成 dict）
# ══════════════════════════════════════════════════════════════════
class QueryError(Exception): ...
class AsOfDateError(QueryError): ...    # as_of 越界 / 超出数据覆盖 / 非法格式
class DataGapError(QueryError): ...     # 请求区间内数据缺失超过容忍度
class UnknownFieldError(QueryError): ...

# ══════════════════════════════════════════════════════════════════
# 0. 连接与快照生命周期
# ══════════════════════════════════════════════════════════════════
def open_db(market_path: str | None = None,
            derived_path: str | None = None) -> None:
    """惰性建立 read_only 连接（ATTACH derived 库）。幂等。
    若底层文件的 realpath 发生变化（原子替换发生过）→ 自动重连。"""

def close_db() -> None: ...

def snapshot_id() -> str:
    """当前数据快照指纹 = sha256(market 文件名 + 各表 max(_ingested_at) + schema_version)[:16]。
    ★ 必须写进 BacktestResult 与 docs/oos-runs.md —— param_hash 只锁参数，
      数据快照锁数据；两者都相同才叫「可复现」（补强 D7，见 §12.2 补 B1）。"""

def preload(start: DateLike, end: DateLike,
            tables: Sequence[str] = ("daily_bar", "daily_basic")) -> None:
    """把区间数据一次性物化进进程内 DataFrame 缓存。
    回测入口调用一次，之后所有 get_* 优先命中缓存切片而非发 SQL。
    Live（每日信号）路径不调用，直接走 SQL。"""

def clear_preload() -> None: ...

# ══════════════════════════════════════════════════════════════════
# 1. 日历
#    交易日历是提前公布的公共信息，查询未来日历不构成前视偏差。
# ══════════════════════════════════════════════════════════════════
def is_trade_date(as_of_date: DateLike) -> bool: ...

def prev_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date: ...

def next_trade_date(as_of_date: DateLike, n: int = 1) -> _dt.date | None:
    """日历未覆盖到时返回 None（不抛）。回测末端必须处理 None。"""

def get_trade_dates(as_of_date: DateLike, *,
                    start: DateLike | None = None,
                    freq: str = "D") -> list[_dt.date]:
    """返回 [start, as_of_date] 闭区间内的交易日。
    freq: 'D' 全部 | 'W' 每周最后一个交易日 | 'M' 每月最后一个交易日。
    'W' 即规格 §5.3 的 weekly_dates 定义，唯一实现点，禁止各处自己算周末。"""

# ══════════════════════════════════════════════════════════════════
# 2. 股票池与元数据（D5）
# ══════════════════════════════════════════════════════════════════
def get_universe(as_of_date: DateLike, *,
                 min_list_days: int = 250,
                 exclude_st: bool = True,
                 exclude_suspended: bool = True,
                 liquidity_drop_pct: float = 0.20,
                 markets: Sequence[str] | None = None) -> list[str]:
    """返回 as_of_date 当日可交易股票池（ts_code 升序）。
    ★ 剔除顺序固定，不可调换（规格 §4.3 列了规则但没定顺序，此处定死）：
        1) delist_date IS NULL OR delist_date > as_of_date       退市
        2) list_date <= as_of_date - min_list_days（自然日）      未上市 + 次新
        3) stock_status 在 as_of_date 生效状态 ∉ {ST,*ST,DELIST_PERIOD}
        4) daily_bar 当日存在且 is_suspended = FALSE
        5) markets 过滤（None = 全市场）
        6) ★ 在 1–5 剩余池内 ★ 计算 20 日均成交额横截面分位，剔除后 liquidity_drop_pct
    先硬性剔除、后算流动性分位 —— 顺序颠倒会让退市股/次新股参与分位计算，结果不同。"""

def explain_universe(as_of_date: DateLike, **kwargs) -> pd.DataFrame:
    """调试与验收用。index=ts_code，columns=[step1_listed, step2_seasoned, step3_not_st,
    step4_tradable, step5_market, step6_liquid, included, drop_reason]。
    规格 §11 的 P1 验收断言直接跑这个函数，比断言 get_universe 的差集可读。"""

def get_stock_basic(as_of_date: DateLike,
                    ts_codes: Sequence[str] | None = None) -> pd.DataFrame:
    """index=ts_code，columns=[symbol,name,sw_l1,sw_l2,sw_l3,market,list_date,delist_date,is_hs]。
    name / sw_* 取 as_of_date 时点的历史值（来自 namechange 与行业变更历史），不是今天的值。"""

def get_industry(as_of_date: DateLike,
                 ts_codes: Sequence[str] | None = None,
                 level: str = "l1") -> pd.Series:
    """index=ts_code → 申万行业代码。★ 成分数 < 5 的行业统一归入 '__OTHER__'，
    否则中性化 OLS 的行业 dummy 会奇异（规格未提，见 §12.2 补 B7）。"""

# ══════════════════════════════════════════════════════════════════
# 3. 行情（D8：对外只给后复权；原始价不出 query 层）
# ══════════════════════════════════════════════════════════════════
def get_bars(as_of_date: DateLike,
             ts_codes: Sequence[str],
             *,
             lookback: int | None = None,
             start: DateLike | None = None,
             fields: Sequence[str] = ("open", "high", "low", "close", "vol", "amount"),
             adjust: str = "hfq") -> pd.DataFrame:
    """长表，MultiIndex (ts_code, trade_date)，闭区间上界 = as_of_date。
    lookback = 交易日条数（与 start 二选一，都给则取交集）。
    adjust: 'hfq'（默认，价格列 × adj_factor）| 'none'（原始价，仅 ingest/validate 可用）。
    ★ 永远额外返回 is_suspended 列。停牌日有行但 OHLC 为 NaN（见 §12.2 补 B3）。
    ★ 本函数【不返回】limit_up / limit_down —— 复权价与原始涨跌停价比较是 bug 温床，
      涨跌停信息只能通过 get_tradable_mask 获取。"""

def get_price_panel(as_of_date: DateLike,
                    ts_codes: Sequence[str],
                    field: str = "close",
                    lookback: int = 250,
                    adjust: str = "hfq") -> pd.DataFrame:
    """宽表：index=trade_date，columns=ts_code。因子计算的主力入口（向量化友好）。
    停牌日为 NaN，不做前向填充 —— 填充与否由因子自己决定并写进注释。"""

def get_daily_basic(as_of_date: DateLike,
                    ts_codes: Sequence[str],
                    fields: Sequence[str] = ("pe_ttm", "pb", "ps_ttm", "total_mv",
                                             "circ_mv", "turnover_rate_f"),
                    lookback: int = 1) -> pd.DataFrame:
    """lookback=1 → 单日，index=ts_code；lookback>1 → MultiIndex (ts_code, trade_date)。"""

def get_index_bars(as_of_date: DateLike,
                   index_code: str,
                   lookback: int = 250,
                   fields: Sequence[str] = ("close", "pe_ttm")) -> pd.DataFrame:
    """index=trade_date。用于 beta_250 中性化与 P3 宏观 ERP / trend_ma200。"""

# ══════════════════════════════════════════════════════════════════
# 4. 执行时点专用（D6）—— 唯一首参不是 as_of_date 的函数
# ══════════════════════════════════════════════════════════════════
def get_tradable_mask(exec_date: DateLike,
                      ts_codes: Sequence[str]) -> pd.DataFrame:
    """★ 唯一合法的「首参非 as_of_date」函数，静态检查白名单里显式列出这一个名字。
    语义：回测时钟已推进到 exec_date（T+1），此刻读 exec_date 的盘口是「当下」不是「未来」。
    调用方限定：只有 ashare/backtest/** 可以调（L4 规则的一部分）。

    index=ts_code，columns：
      can_buy    bool    可买入
      can_sell   bool    可卖出
      reason     str     '' | 'suspended' | 'limit_up_seal' | 'limit_down_seal'
                         | 'no_quote' | 'delisted' | 'limit_unknown'
      open_hfq   float   后复权开盘价（成交价）
      close_hfq  float   后复权收盘价（退市清仓用）
      amount     float   当日成交额（冲击成本用）
      amplitude  float   (high-low)/pre_close（冲击成本用）

    判定（全部用【原始价】在函数内部完成，原始价不外泄）：
      停牌 / 无行情           → can_buy=False, can_sell=False
      open==limit_up  且 high==low → can_buy=False（一字涨停买不进）
      open==limit_down 且 high==low → can_sell=False（一字跌停卖不出）
      limit_up IS NULL（数据缺失） → 两者皆 False，reason='limit_unknown'
                                     ★ 保守：宁可不交易，不可假设可交易
      delist_date <= exec_date      → can_buy=False, can_sell=True（强制清仓路径）"""

# ══════════════════════════════════════════════════════════════════
# 5. 财报 PIT（D3）
# ══════════════════════════════════════════════════════════════════
def get_financial(as_of_date: DateLike,
                  ts_codes: Sequence[str],
                  fields: Sequence[str],
                  *,
                  n_periods: int = 1,
                  include_restated: bool = False) -> pd.DataFrame:
    """PIT 取数：WHERE ann_date <= as_of_date，按 end_date 分组取 ann_date 最大者。
    n_periods=1 → index=ts_code；>1 → MultiIndex (ts_code, end_date) 倒序 n 期。
    include_restated=False（默认）→ 只取 update_flag=0 的原始披露值。
      True 仅供研究「重述影响」，任何进入回测的因子都必须用 False。
    额外返回列：ann_date、end_date、report_type、lag_days(= as_of - ann_date)。"""

def get_financial_ttm(as_of_date: DateLike,
                      ts_codes: Sequence[str],
                      field: str) -> pd.Series:
    """★ TTM 拼接必须在 query 层（规格未指定归属，此处定死）。
    理由：A 股财报是【累计口径】，TTM = 最新累计 + 上年年报 − 上年同期累计，
    这段逻辑放因子层会被 roe_ttm / ep_ttm / sp_ttm / gross_margin 各抄一遍，
    抄错一处就是静默错误。
    流量科目（revenue / n_income / n_cashflow_act）走上式；
    存量科目（total_assets / equity）走「期初期末均值」，函数内按 field 白名单分派。
    不足 4 期或跨期缺失 → NaN（不外推）。"""

# ══════════════════════════════════════════════════════════════════
# 6. 宏观 PIT（D4）与资金流
# ══════════════════════════════════════════════════════════════════
def get_macro(as_of_date: DateLike,
              indicators: Sequence[str],
              lookback_periods: int = 60) -> pd.DataFrame:
    """WHERE publish_date <= as_of_date；同 (indicator, period) 取 publish_date 最大者。
    index=period，columns=indicator，附加列 <indicator>__publish_date 便于审计。"""

def get_money_flow(as_of_date: DateLike,
                   ts_codes: Sequence[str],
                   fields: Sequence[str] = ("hk_hold_ratio",),
                   lookback: int = 20) -> pd.DataFrame:
    """MultiIndex (ts_code, trade_date)。
    ★ hk_hold_ratio 仅 2016-12 起有数据；早于该日返回 NaN，
      调用方（north_hold_chg_20 因子）必须靠 FactorSpec.available_from 声明而非静默填 0。"""
```

**query 层的四条不变量**

| # | 不变量 | 理由 |
|---|---|---|
| Q1 | 所有 SQL 参数化（`?` 占位），**禁止字符串拼接日期/代码** | 注入 + 类型歧义（DuckDB DATE vs str） |
| Q2 | 日期入参统一在函数入口 `_norm_date()` → `datetime.date`，越界抛 `AsOfDateError` | 类型统一 |
| Q3 | 空结果返回**带正确列名的空 DataFrame/Series**，绝不返回 None | 调用方免写 None 分支 |
| Q4 | query 层 **raise**，不返回 `{"error": ...}`。dict 化只发生在 `agent_tools.py` | 分层职责；见 §6.2 |

### 4.2 因子契约 —— `@factor` 装饰器而非抽象基类

> **对规格的偏离说明**：规格提到「Factor 抽象基类」，本方案改为
> **`@factor` 装饰器 + `FactorSpec` dataclass**。理由：
> ① 规格 §2 D2 要求的签名字面量就是 `f(as_of_date, universe) -> Series`，函数形式直接满足，
>    类方法会多出 `self` 参数，静态检查要额外处理；
> ② 16 个因子没有一个需要继承共享状态或多态；ABC 在这里是纯样板；
> ③ 参数化因子（闸 5 的 ±30% 网格）用关键字参数 + `param_hash` 表达，比子类实例化更轻。
> 元数据契约不弱化 —— 全部落在 `FactorSpec` 上，可被静态检查读取。

```python
# ashare/factors/base.py
from __future__ import annotations
from dataclasses import dataclass, field
import datetime as _dt
from typing import Any, Callable, Mapping
import pandas as pd

FactorFn = Callable[..., pd.Series]

@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: FactorFn
    direction: int                      # +1 值越大越好 / -1 越小越好
    category: str                       # 'price' | 'fundamental' | 'flow' | 'risk'
    lookback_days: int                  # 声明需要多少【交易日】历史 → 驱动 preload 区间
    neutralize: bool = True             # risk 类（log_mv/industry/beta_250）设 False
    available_from: _dt.date | None = None   # 数据可得起始日（north_* = 2016-12-05）
    min_coverage: float = 0.60          # 池内非空占比低于此值 → 该日该因子作废
    default_params: Mapping[str, Any] = field(default_factory=dict)

    def param_hash(self, **override) -> str:
        """sha256(name + canonical_json(default_params | override))[:12]
        进 factor_value 主键，保证参数不同的同名因子不互相覆盖。"""

FACTOR_REGISTRY: dict[str, FactorSpec] = {}

def factor(*, name: str, direction: int, category: str, lookback_days: int,
           neutralize: bool = True,
           available_from: _dt.date | None = None,
           min_coverage: float = 0.60,
           **default_params) -> Callable[[FactorFn], FactorFn]:
    """注册装饰器。重名直接 raise（不允许静默覆盖）。"""

# ── 因子函数签名契约（CI 强制检查前两个位置参数名）──
#
#   @factor(name="reversal_20", direction=+1, category="price", lookback_days=30)
#   def reversal_20(as_of_date, universe, *, window: int = 20) -> pd.Series: ...
#
#   1) 前两个位置参数必须严格是 as_of_date, universe（名字与顺序）
#   2) 其余参数必须是 keyword-only（* 之后），且有默认值
#   3) 返回 index=ts_code 的 Series，允许是 universe 的子集（缺的记 NaN）
#   4) 只能通过 ashare.data.query 取数；无副作用；不得写文件/网络/全局状态
#   5) 不得引用 as_of_date 之后的任何数据（靠 query 层签名 + 人工评审保证）

def get_factor(name: str) -> FactorSpec: ...
def list_factors(category: str | None = None) -> list[FactorSpec]: ...

def compute_factor(name: str, as_of_date, universe: list[str], *,
                   processed: bool = True, **param_override) -> tuple[pd.Series, list[str]]:
    """raw → （processed=True 时）走 pipeline.process。
    available_from 之前的日期直接返回全 NaN Series（不静默填 0）。"""

def compute_panel(names: list[str], as_of_date, universe: list[str], *,
                  processed: bool = True) -> pd.DataFrame:
    """index=ts_code，columns=names。**一律现算**，返回 `(DataFrame, warnings)`。

    **2026-08-21 修正两处（初稿两条都错）：**
    · 「优先从 factor_value 表读，缺的现算」与 Task 8 的裁决直接矛盾 ——
      `derived_store.read` 未命中就返回空、**不静默现算**，因为「现算与落库的口径分歧」
      是最难查的一类 bug。读缓存还是现算由**调用方**决定，不在这一层偷偷混合。
    · 裸返回类型没有地方放本文档自己要求记录的 warning（§4.2 通例）。
    """

def combine(weights: Mapping[str, float], as_of_date, universe: list[str]) -> tuple[pd.Series, list[str]]:
    """合成分数 = Σ wᵢ × directionᵢ × processedᵢ。默认全 1.0 等权（规格 §5.2）。
    ★ 覆盖率不足（< min_coverage）或 available_from 未到的因子，
      从当日分母中【剔除】并按剩余因子重新归一，而不是当 0 参与 ——
      当 0 参与等于静默降权，会让 2017 年前的合成分数悄悄变味（§12.2 补 B5）。"""
```

```python
# ashare/factors/pipeline.py —— 顺序固定，不可调换（规格 §5.2）
def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series: ...
def neutralize(s: pd.Series, as_of_date, universe, *,
               by: tuple[str, ...] = ("log_mv", "industry")) -> tuple[pd.Series, list[str]]:
    """横截面 OLS 取残差（为什么不是 Barra 的 WLS：见算法说明书 §3.2 的裁决 ——
    组合是等权 top-N，相关的是【无权】正交，那是 OLS 的恒等式）。
    行业 dummy 来自 get_industry（已并小行业）；industry_source != 'sw' 直接抛（回填行业 = 前视）。
    有效样本 < 30 或设计矩阵秩亏 → 返回原 Series + warning，不静默返回 NaN。
    返回 (残差, warnings)：warnings 上浮到 BacktestResult.warnings，降级的那一天在运行记录里看得见。"""
def zscore(s: pd.Series) -> pd.Series: ...
def process(s: pd.Series, as_of_date, universe, *, spec: FactorSpec) -> tuple[pd.Series, list[str]]:
    """1 winsorize_mad → 2 (spec.neutralize 时) neutralize → 3 zscore → 4 fillna(0)"""
```

```python
# ashare/factors/store.py（L3，不 import duckdb）—— 编排「算 → 写」
#   落库本身在 ashare/data/derived_store.py（L1）。拆两层不是为了整洁，是 L1 闸的要求。
#   ★ read 的键是 (name, param_hash) 不是 name：PK 含 param_hash，而 L1 够不到
#     FACTOR_REGISTRY 解析不出它；只按名字读，在闸 5 的 ±30% 参数网格下会同时命中
#     两代值，pivot 要么抛要么静默留一个。
def build(names: list[str], dates: list[_dt.date], *,
          overwrite: bool = False, progress=None) -> tuple[dict, list[str]]:
    """算 → 写。写入本身走 derived_store.write_factor_values。

    PRIMARY KEY (factor_name, param_hash, trade_date, ts_code)
    ★ snapshot_id 变化 → 该快照下的因子值失效，**原地 UPSERT 覆盖**。
      2026-08-21 修正：本行初稿写「不删旧行，加新行」，那与上面的 PK 直接冲突
      （PK 不含 snapshot_id，加不了新行），照着写就会退化成 DO NOTHING ——
      留下陈旧值配陈旧快照，而 read 会把它当成当前快照放行。这是本层存在的意义所在。"""

# ── ashare/data/derived_store.py（L1，独家持 duckdb）──
def write_factor_values(df: pd.DataFrame) -> int: ...
def read_factor_values(name_to_hash: dict[str, str], date: _dt.date,
                       universe: list[str], processed: bool = True
                       ) -> tuple[pd.DataFrame, list[str]]: ...
def current_factor_dates(name_to_hash: dict[str, str]) -> pd.DataFrame: ...
def coverage_report(name_to_hash: dict[str, str]) -> pd.DataFrame:
    """每因子的已算日期区间、覆盖率（**从 raw_value 算**）与 n_stale_dates，
    供 /api/ashare/health 展示。覆盖率必须只统计当前快照的行 ——
    混代统计会让报出来的数越过 min_coverage，而 read 实际只能服务其中一部分。"""
```

### 4.3 回测引擎输入输出

> **★ 引擎侧五条裁决（2026-08-21，Task 13 落地后）**
>
> **① `BacktestResult.equity` 必须是【日频】，不是调仓频率。** 周频采样会**低估最大回撤**
> （看不见周内的低点），而 MDD 喂 Calmar —— 这是朝着好看方向的偏差。
> 策略周频调仓，但账本每个交易日盯市；成本是稀疏序列，对齐到日频索引上即可。
> 年化按各自频率：净值 252、IC 52（§9 的「按入参序列自己的频率」）。
>
> **② `compute_diagnostics=True` 时 `ic` / `layers` / `attribution` 必须真的产出。**
> 尤其 `attribution` —— §3.2 的 OLS-vs-WLS 裁决**只能靠它被证伪**，不接等于那条裁决
> 永远说不清对错。不产出而只发一条 warning，是把一个可检验的断言变成一句空话。
>
> **③ `build_targets` 返回三元组 `(final, intended, warnings)`。** 现在引擎为了拿到裁剪前的
> 意图账本，用 `max_turnover=inf` **再调一次** —— 既多做一遍功，又必须**丢弃第二次的
> warnings**（否则每条降级报两遍）。丢弃 warning 这件事本身就违反「降级必须可见」。
>
> **④ `compute_diagnostics` 只闸住那三个【新增】块；`metrics.compute(full=True)` 恒真。**
> 这解开了 `types.py` 与 `metrics.compute(full=)` 的矛盾：豁免条件写的是
> 「只能新增、不得改动 metrics 里的任何一个数」，而 `full=False` 会**删掉**
> turnover / cost-drag / D6-gap —— 同一个 `param_hash` 产出不同的 metrics 键集。
> `full=False` 只留给临时分析，绝不由 `compute_diagnostics` 驱动。
>
> **⑥ ICIR 摘要（含 Newey-West t 值）进 `ic` 诊断块，不进 `metrics`（2026-08-22 补裁）。**
> Task 13 指出 ① 与 ④ 在这里互撞，读对了：把 ICIR 放进 `metrics` 会让
> `compute_diagnostics` 改变 metrics 的键集，而 ④ 禁止这件事。
> 但只放 `ic_series` 的逐日值同样不行 —— 实测 **`icir()` 在生产代码里无人调用**，
> 只有测试在调。规格 §4.2 把 NW 调整的 t 值称作「把噪声因子判成有效因子的头号原因」的解药，
> 而它在真实运行里一个数都不产出。
> 落点是 `result.ic`：它本身就受 `compute_diagnostics` 闸住，加进去不动 `metrics` 的键集，
> ④ 与 ① 同时成立。
>
> **⑦ 引擎不得在 `cfg.end` 之后成交（2026-08-22 裁决）。**
> 现在最后一个周频信号日若落在区间末尾，$\tau = T{+}1$ 会掉到 `[start, end]` 之外而成交照做。
> 那是一笔**结果永远不被度量**的交易：扣了成本却没有对应的收益，净值被低估 ——
> 方向上保守，但一样是错的。
> 落法：执行日落在区间外的信号**丢弃并告警**；期末账本按 `end` 收盘**盯市**，不强制清仓
> （回测本来就不该以清仓收尾）。
>
> **⑤ `backtest_run` 的读写归 `derived_store`，`backtest/store.py` 只做转发。**
> 表在 schema 里躺着却没有写入方，比没有这张表更糟。pickle 是权宜之计
> （L1 不许 backtest 层碰 duckdb，而当时 `derived_store` 只覆盖了 `factor_value`）。

> **★ `equity` 在本系统里是两个不同的量，必须分清（2026-08-21 裁决）**
>
> · `simulate(equity=...)` —— **货币**口径的组合权益（现金 + 持仓市值），
>   因为它要做 `shares = Δw × equity / price` 的权重↔股数换算。
> · `BacktestResult.equity` —— **净值**指数，初始 1.0。
>
> `charge` 产出的 `total_cost` 跟着前者，是**货币**；而 `metrics.compute` 收到的是后者。
> 两者相除得到的不是比例而是钱 —— 实测 `cost_drag_annual = 98502.98`，
> 而 §5.4 说这个数应该落在 **3%–6%**。**成本模型是否接对了，本来只有这一个便宜的体检指标，
> 单位一错它就废了。**
>
> **落法**：`metrics.compute(..., initial_capital: float)`，
> `cost_frac_t = cost_t / (net_value_t × initial_capital)`。定额本金回测下这个换算是精确的。
> 并加一条量纲守卫：`cost_drag_annual > 1.0` 就告警 ——
> 年化成本拖累超过 100% 永远不是真结果，而这是「两条曲线不在同一量纲上」唯一的信号。

> **派生库的写入口在数据层，不在 backtest / factors 层（2026-08-20 裁决）。**
>
> `BacktestResult.save()/load()` 与 Task 8 的因子落库都要写 `derived.duckdb`，
> 而分层闸 **L1 只允许 `ashare/data/**` `import duckdb`** —— 两处都撞上同一堵墙。
>
> **不放宽 L1。** 这道闸故意是粗粒度的：「这个文件能不能 import duckdb」AST 查得出来，
> 「能 import 但只准连 derived.duckdb」查不出来。一旦 `factors/store.py` 能 import duckdb，
> 它同样能 `duckdb.connect('market.duckdb')` 然后 SELECT 未经掩码的原始行 ——
> 那正是 D2 要挡的东西。
>
> **落法**：新建**公开**模块 `ashare/data/derived_store.py`，独家持有派生库的读写
> （因子值 + 回测运行记录）。它只收发 DataFrame 与基础类型，**绝不 import
> `ashare.backtest` 或 `ashare.factors`** —— 否则低层反向依赖高层。
> `BacktestResult.save()` 负责把自己序列化成 dict 再交给它。
> **2026-08-21 修正**：本行初稿说「不额外包一层 `factors/store.py` 转发」——
> 那句话在「转发」的前提下是对的，但 `build` 不是转发，它是「算 → 写」的编排。
> 实际落地是两层：`derived_store`（L1，持 duckdb，收发 DataFrame）
> + `factors/store.py`（L3，调 `compute_panel` 与 `write_factor_values`，不碰 duckdb）。
>
> 注意 L2（首参 `as_of_date`）**不延伸到** `derived_store`：回测引擎读因子面板天然是
> 读一个日期区间（2010–2024 一次读完），强行套 `as_of_date` 形状就不对。
> 这里的前视保护在**写入时**——因子值本身是按 PIT 纪律算出来的。


```python
# ashare/backtest/types.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import datetime as _dt
import pandas as pd

@dataclass(frozen=True)
class CostConfig:                       # 数值来源：规格 §5.3，本文不重复解释
    commission_bps: float = 2.5         # 双边
    stamp_duty_bps: float = 5.0         # 仅卖出
    transfer_bps: float = 0.1           # 沪市
    impact_coef: float = 0.5            # 0.5 × (委托额/当日成交额) × 振幅
    impact_cap_bps: float = 30.0
    multiplier: float = 1.0             # 闸 4（成本敏感）设 2.0

@dataclass(frozen=True)
class PortfolioConstraints:
    top_n: int = 50
    weighting: str = "equal"            # 'equal' | 'risk_parity'
    max_single: float = 0.05
    max_industry: float = 0.20
    max_turnover: float = 0.30          # 单周双边换手上限

@dataclass(frozen=True)
class BacktestConfig:
    start: _dt.date
    end: _dt.date
    factors: tuple[tuple[str, float], ...]      # 有序元组以便 hash；((name, weight), ...)
    constraints: PortfolioConstraints = PortfolioConstraints()
    cost: CostConfig = CostConfig()
    macro_timing: bool = False                   # False → 恒定满仓（P2 阶段默认 False）
    position_floor: float = 0.20
    position_cap: float = 1.00
    benchmark: str = "000985.CSI"
    initial_capital: float = 1_000_000.0
    # ── 运行模式开关（不进 param_hash，见下）──
    compute_diagnostics: bool = True    # False → 跳过 IC/分层/归因，只算净值（8s 档）
    shuffle_seed: int | None = None     # 非 None → 每个调仓日横截面打乱分数（闸 3）
    factor_param_override: Mapping[str, Mapping] = field(default_factory=dict)

    def param_hash(self) -> str:
        """sha256(canonical_json(策略语义字段))[:16]。
        ★ 只包含影响结果的字段；compute_diagnostics 不进 hash（它只影响算不算诊断），
          shuffle_seed 进 hash（它改变结果）。
        ★ 这是 D7 的参数指纹，与 query.snapshot_id() 一起写进 docs/oos-runs.md。"""

@dataclass
class BacktestResult:
    config: BacktestConfig
    param_hash: str
    data_snapshot_id: str               # ★ 补强 D7
    engine_version: str                 # 引擎语义变更时手动 bump
    started_at: _dt.datetime
    elapsed_sec: float

    equity: pd.Series                   # index=trade_date（日频），组合净值（初始 1.0）
    positions: pd.DataFrame             # MultiIndex (rebalance_date, ts_code)
                                        #   cols: score, target_weight, filled_weight,
                                        #         shares, price_hfq, industry
    trades: pd.DataFrame                # cols: exec_date, ts_code, side('BUY'|'SELL'),
                                        #   shares, price_hfq, amount, commission,
                                        #   stamp_duty, transfer_fee, impact, total_cost
    blocked: pd.DataFrame               # ★ 不可交易明细，D6 的证据链
                                        #   cols: exec_date, ts_code, intended_side,
                                        #         intended_weight, reason
    metrics: dict                       # 规格 §5.4 的净值/相对/交易三类指标
    ic: pd.DataFrame | None             # compute_diagnostics=True 时：
                                        #   index=rebalance_date, cols=<factor>__ic/__rank_ic
    layers: pd.DataFrame | None         # 10 分层：index=rebalance_date, cols=L1..L10
    attribution: pd.DataFrame | None    # Brinson 行业 + 风格回归
    warnings: list[str]                 # 中性化秩亏 / 覆盖率不足 / 日历缺口等非致命问题

    def save(self, run_id: str) -> None:     # → derived.duckdb + parquet
    @classmethod
    def load(cls, run_id: str) -> "BacktestResult": ...
    def summary(self) -> dict:               # 供 REST / Agent 工具返回的精简版（< 3 KB）

# ashare/backtest/engine.py —— 唯一公开入口
def run_backtest(config: BacktestConfig,
                 *, on_progress: Callable[[int, int], None] | None = None
                 ) -> BacktestResult: ...
```

**引擎内部分工（`engine.py` ≤ 400 行只做编排，实体逻辑在同级模块）**

| 函数 | 归属文件 | 签名 |
|---|---|---|
| 组合构建（top_n / 权重 / 三条约束 / 换手裁剪） | `portfolio.py` | `build_targets(scores, target_position, prev_weights, industry, constraints) -> pd.Series` |
| 成交模拟（掩码 + 开盘价 + 部分成交） | `execution.py` | `simulate(exec_date, targets, prev_holdings, equity, cost) -> (trades, holdings, blocked)` |
| 费用 | `cost.py` | `charge(trade_rows, cost_cfg) -> pd.DataFrame` |
| 指标 | `metrics.py` | `compute(equity, trades, positions, benchmark_series, *, full: bool) -> dict` |
| 五闸 | `guards.py` | 见下 |

```python
# ashare/backtest/guards.py
@dataclass
class GateResult:
    name: str; passed: bool; detail: dict; note: str

def gate1_out_of_sample(cfg) -> GateResult          # 样本外 Sharpe ≥ 样本内 × 0.6
def gate2_walk_forward(cfg, train_years=5, test_years=1) -> GateResult
def gate3_shuffle(cfg, n: int = 200, seed: int = 0) -> GateResult
    """★ 内部强制 compute_diagnostics=False，否则 200 次 × 60s = 3.3 小时（§12.1 砍 A3）。"""
def gate4_cost_stress(cfg, multiplier: float = 2.0) -> GateResult
def gate5_param_plateau(cfg, grid: Mapping[str, Sequence]) -> GateResult
def run_all_gates(cfg, *, gates: Sequence[str] | None = None) -> dict[str, GateResult]
```

### 4.4 P3 输出契约（本期不实现，仅锁边界）

规格 §6.3 的调仓清单 JSON 即 P3 对外契约。本期只做两件事：
1. 在 `ashare/strategy/__init__.py` 里放函数签名占位
   `def build_rebalance_plan(as_of_date, config) -> dict`，返回结构 = 规格 §6.3；
2. 该 JSON **必须**额外带 `data_snapshot_id` 字段（与 `param_hash` 并列）。

---
## 5. 数据流与调度时序

### 5.1 日常增量时序（交易日 18:00 触发）

```
时刻    执行者              动作                                        失败处理
─────  ──────────────────  ──────────────────────────────────────────  ─────────────────────
18:00  cron                python -m ashare.data.ingest --daily        —
       │
18:00  ingest.preflight    ① trade_cal 确认 T 是交易日                  非交易日 → exit 0（静默）
                           ② 检查 ingest_log 有无未完成分区             有 → 先补跑
                           ③ cp market.duckdb → market.staging.duckdb  磁盘不足 → 告警 exit 1
       │
18:02  ingest.fetch        并发度 = 1（Tushare 限频友好），按序拉：      每接口独立重试(§5.3)
                           trade_cal / stock_basic(周一才拉)
                           daily(T) → adj_factor(T) → daily_basic(T)
                           stk_limit(T) 或 limits.py 规则计算
                           suspend_d(T) → hk_hold(T)
                           income/balancesheet/cashflow/fina_indicator(ann_date=T)
                           macro：每月固定日窗口拉（见 §5.2）
                           index_daily(T) × 4
       │
18:12  ingest.normalize    ★ 停牌补行：日历有 T、股票在市、daily 无行    —
                           → 插入 OHLC=NULL, is_suspended=TRUE 的占位行
                           （不补这一行，因子的 lookback 窗口会静默错位）
       │
18:14  ingest.write        对 staging 库执行 DELETE+INSERT 事务          事务失败 → 丢弃 staging
                           （按 (table, trade_date) 幂等）
       │
18:16  validate.run        规格 §4.4 六项，作用于 staging 库             阻断项失败 → 保留 staging
                           分级：BLOCK / WARN                            供人工检查，主库不动，
                                                                         告警邮件，exit 2
       │
18:18  ingest.promote      fsync(staging) → os.replace(staging, market) 替换失败 → 主库仍是旧的
                           → 写 db_version 记录 → 保留最近 3 个历史版本   （原子性由 rename 保证）
       │
18:19  factors.store.build 增量算 T 日全部因子（若 T 是周五还要算        因子算错不影响 market 库，
                           调仓日横截面）→ 写 derived.staging → 替换      单独回滚
       │
18:26  (仅周五) P3 signal  build_rebalance_plan(T) → signal 表 + JSON    P3 未上线时跳过
       │
18:28  报告                写 data/ashare/runs/<T>.json：               —
                           各表增量行数、校验结果、耗时、warnings
                           失败时复用 scripts/alert_worker.py 发邮件
```

**关键点**

| # | 决策 | 理由 |
|---|---|---|
| S1 | 拉取并发度 = 1 | Tushare 按分钟限频，并发只会更快撞限流然后退避，净吞吐不增反降 |
| S2 | 先写 staging 再原子替换 | 同时解决 DuckDB 单写限制、备份、回滚（§10.2） |
| S3 | 校验作用于 staging 而非主库 | 校验不过时主库仍是上一个已知良好版本，服务不中断 |
| S4 | 因子库与行情库分开替换 | 因子重算不必复制 0.9 GB 行情数据；回滚粒度分离 |
| S5 | 停牌补行在 normalize 阶段做 | 补行是数据语义的一部分，不能留给因子层各自判断 |

### 5.2 各数据集的更新节奏

| 数据集 | 分区键 | 频率 | 备注 |
|---|---|---|---|
| `calendar` | 年 | 每年 12 月 + 每周一校对 | 交易所公布次年日历后 |
| `stock_basic` | 全量 | 每周一 | 4 次调用（list_status = L/D/P/全部）合并 |
| `stock_status`(ST) | 全量 | 每周一 | ★ 由 `namechange` 反推，见 §12.2 补 B6 |
| `daily_bar`/`adj_factor`/`daily_basic` | trade_date | 每交易日 | 各 1 次调用 |
| 涨跌停价 | trade_date | 每交易日 | `stk_limit` 或 `limits.py` 规则兜底 |
| 停牌 | trade_date | 每交易日 | `suspend_d` + 与 daily 缺行交叉验证 |
| `money_flow`(北向) | trade_date | 每交易日 | 2016-12 起 |
| `financial_pit` | ann_date | 每交易日 | 4 接口 × ann_date=T；披露季非交易日也要补拉 |
| `macro_indicator` | 指标 | 每月 3 个窗口 | PMI 月末/次月 1 日、CPI-PPI 次月 9–11 日、社融-M1M2 次月 10–15 日；**窗口内每日重拉直到该期出现，publish_date = 首次出现日**（D4 的落地做法） |
| `index_daily` | trade_date | 每交易日 | 4 指数 |

> **`publish_date` 怎么来**：Tushare 宏观接口不返回公布日。做法是**每日轮询 + 记录首见日**：
> 某指标某 `period` 第一次在库外接口出现的那个自然日 = `publish_date`。
> 回补历史时没有这个日志，只能用**官方发布日历的经验规则**回填（社融次月 10–15、CPI 次月 9、PMI 月末），
> 并在 `macro_indicator` 加一列 `publish_date_source ∈ {'observed','rule'}`。
> ★ 这是 D4 在历史区间上的**已知精度损失**，必须在 ADR 与 `oos-runs.md` 中显式承认，
>   不能假装 `publish_date` 是精确的。

### 5.3 失败与重试策略

**任务状态机**（每个 `(dataset, partition_key)` 一行）

```
             ┌──────────┐
             │ PENDING  │
             └────┬─────┘
                  ▼
             ┌──────────┐   限频/超时/5xx      ┌──────────┐
             │ RUNNING  │────────────────────▶│  RETRY   │─┐
             └────┬─────┘   退避 1/4/16/64 s   └──────────┘ │ ≤4 次
                  │                                 ▲       │
                  │ 成功                            └───────┘
                  ▼                                         │ >4 次
             ┌──────────┐                              ┌────▼─────┐
             │    OK    │                              │  FAILED  │
             └──────────┘                              └──────────┘
                  ▲
                  │ 二次确认有数据
             ┌────┴─────┐   日历说是交易日但返回空
             │ SUSPECT  │◀──────────────────────────  30 min 后单次重跑
             └──────────┘   仍空 → FAILED
```

**`ingest_log` 表（规格缺失，必须补 —— 见 §12.2 补 B1）**

```sql
ingest_log(
  dataset       VARCHAR,      -- 'daily_bar' | 'financial_pit' | ...
  partition_key VARCHAR,      -- '2026-08-18' | '2026Q2' | 'ALL'
  status        VARCHAR,      -- PENDING/RUNNING/OK/RETRY/SUSPECT/FAILED
  attempt       INTEGER,
  rows_written  BIGINT,
  source        VARCHAR,      -- 'tushare' | 'baostock' | 'akshare'
  started_at    TIMESTAMP,
  finished_at   TIMESTAMP,
  error         VARCHAR,      -- 截断到 500 字符，不含 token
  PRIMARY KEY (dataset, partition_key, attempt)
);
```

**错误分类与动作**

| 错误类型 | 判据 | 动作 |
|---|---|---|
| 限频 | Tushare 抛「每分钟最多访问该接口 N 次」 | 退避后重试；令牌桶速率自动下调 20% |
| 网络/超时/5xx | requests 异常 | 指数退避 1/4/16/64 s，≤ 4 次 |
| 积分不足 | Tushare 抛权限错误 | **不重试**，标 FAILED，走降级路径（如 `stk_limit` → `limits.py`），告警 |
| 返回空但应有数据 | 日历是交易日 + 该表历史同期非空 | SUSPECT → 30 min 后单次重跑 |
| 字段缺失/类型变更 | normalize 阶段 schema 断言失败 | **不重试**，FAILED + 告警（接口变更需人工改 adapter） |
| 校验 BLOCK 项失败 | `validate.py` | 不替换主库，保留 staging，告警 |
| 校验 WARN 项失败 | `validate.py` | 记 warnings，继续替换 |

**幂等**：所有写入是 `DELETE FROM t WHERE <partition_key>` + `INSERT`，包在单事务里。
重跑任意分区任意次，结果一致。

### 5.4 Tushare 限流下的拉取节奏与首次全量回补

**限流适配（不硬编码具体积分档位）**

Tushare 的每分钟调用上限随积分档位与接口而变，且官方会调整。因此**不把速率写死**，而是：

```
令牌桶（tokens/min，初始 120）
  ├─ 连续 60 次成功         → 速率 +10%（上限 480/min）
  ├─ 命中一次限频错误       → 速率 −20%，退避 60 s
  └─ 每次调用前 sleep 到桶允许为止
状态持久化到 data/ashare/rate_state.json —— 跨进程复用学习到的速率，
避免每次重启都从头试探再撞墙。
```

**首次全量回补分批方案**（每批独立可重跑，断点续传靠 `ingest_log`）

| 批次 | 数据集 | 分片键 | 调用次数 | 单次行数 | @120/min | @300/min |
|---|---|---|---|---|---|---|
| B0 | trade_cal / stock_basic / index_daily / macro | 全量 | ≈ 30 | — | < 1 min | < 1 min |
| B1 | `daily` | trade_date | 4,030 | ~3,000 | 34 min | 13 min |
| B2 | `adj_factor` | trade_date | 4,030 | ~3,000 | 34 min | 13 min |
| B3 | `daily_basic` | trade_date | 4,030 | ~3,000 | 34 min | 13 min |
| B4 | 涨跌停（`stk_limit` 或规则） | trade_date | 4,030 / 0 | ~3,000 | 34 min / 0 | 13 min / 0 |
| B5 | `suspend_d` | trade_date | 4,030 | 小 | 34 min | 13 min |
| B6 | `hk_hold`（2016-12 起） | trade_date | 2,300 | ~2,200 | 19 min | 8 min |
| B7 | 财报 4 表 | ann_date | 4,030 × 4 = 16,120 | 小 | 134 min | 54 min |
| B8 | BaoStock 交叉校验抽样 | 200 只 × 100 日 | 200 | — | ~5 min | ~5 min |
| **合计** | | | **≈ 38,800 次** | | **≈ 5.5 h** | **≈ 2.2 h** |

> **估算口径**：调用次数是确定的；耗时 = 调用次数 / 有效速率，有效速率取决于积分档与网络。
> 给出 120/min 与 300/min 两档区间。实测速率写进 `rate_state.json` 后，二次回补按实测重估。
> **执行建议**：B1–B7 分三夜跑（B1–B3 / B4–B6 / B7），每批完成后立刻跑该批的校验项；
> 全部完成后再跑一次全量 `validate` + B8 交叉校验。

**回补期的特殊约束**

| # | 约束 | 理由 |
|---|---|---|
| R1 | 回补写入时**不建主键/唯一索引**，全部写完后再 `CREATE UNIQUE INDEX` | DuckDB 大批量插入时维护 ART 索引会显著拖慢（千万行级差异明显） |
| R2 | 回补直接写 `market.staging.duckdb`，不碰主库 | 回补跑几小时，期间服务照常读旧库 |
| R3 | 每 200 个分区 `CHECKPOINT` 一次 | 控制 WAL 体积 |
| R4 | 回补脚本必须支持 `--from-partition` 与 `--only-failed` | 断点续传 |
| R5 | 全量回补完成后导出一次 Parquet 冷备 | 之后重建 DuckDB 不必再拉一遍 Tushare |

---
## 6. 与现有 server.py / quant_agent.py 的集成契约

### 6.1 新增 REST 端点清单

全部挂在现有 FastAPI app 上，路径前缀 `/api/ashare/`。**不新增服务、不新增中间件。**

| 端点 | 方法 | 鉴权 | 请求 | 响应（成功） |
|---|---|---|---|---|
| `/api/ashare/health` | GET | 匿名 | — | `AshareHealth` |
| `/api/ashare/universe` | GET | 匿名 | `as_of: str`（必填）、`explain: bool=false`、`limit: int=200` | `UniverseResponse` |
| `/api/ashare/factors` | GET | 匿名 | — | `list[FactorMeta]` |
| `/api/ashare/factor-exposure` | GET | 匿名 | `as_of`、`ts_code?`、`factor?`、`top: int=50` | `FactorExposureResponse` |
| `/api/ashare/backtest` | POST | **`require_user`** | `BacktestRequest` | `202 {run_id, status:"queued"}` |
| `/api/ashare/backtest/{run_id}` | GET | 匿名 | — | `BacktestRunResponse` |
| `/api/ashare/backtest/runs` | GET | 匿名 | `limit: int=20` | `list[BacktestRunSummary]` |
| `/api/ashare/gates/{run_id}` | GET | 匿名 | — | `dict[str, GateResult]`（未跑则 `{"status":"not_run"}`） |

> **鉴权口径**：读端点匿名可用（与现有 `/api/rag/search`、`/api/kb/stats` 一致，数据是公开市场数据，
> 无用户维度）。**只有 `POST /api/ashare/backtest` 要求登录** —— 它消耗几十秒 CPU，是唯一可被
> 滥用成算力的端点。不为其它端点加登录，是因为加了没有安全收益只有摩擦。

**Pydantic 模型（注意 3.9 必须用 `Optional[...]`）**

```python
class AshareHealth(BaseModel):
    ok: bool
    db_version: str                     # market.duckdb 的版本文件名
    data_snapshot_id: str
    latest_trade_date: Optional[str]
    table_rows: Dict[str, int]
    last_ingest: Optional[Dict[str, Any]]   # runs/<date>.json 摘要
    failed_partitions: int                  # ingest_log status='FAILED' 计数
    factor_coverage: Dict[str, str]         # factor -> "2010-01-04..2026-08-15"

class UniverseResponse(BaseModel):
    as_of: str
    count: int
    ts_codes: List[str]                     # 截断到 limit
    truncated: bool
    reasons: Optional[List[Dict[str, Any]]] = None   # explain=true 时

class FactorMeta(BaseModel):
    name: str; direction: int; category: str
    lookback_days: int; neutralize: bool
    available_from: Optional[str]; description: str

class FactorExposureResponse(BaseModel):
    as_of: str
    mode: str                               # 'by_stock' | 'by_factor'
    rows: List[Dict[str, Any]]              # by_stock: 一只股票的全因子 z 值
                                            # by_factor: 该因子 top N 股票
class BacktestRequest(BaseModel):
    start: str
    end: str
    factors: Dict[str, float]               # 白名单校验：key ∈ FACTOR_REGISTRY
    top_n: int = Field(50, ge=10, le=500)
    weighting: str = Field("equal", pattern="^(equal|risk_parity)$")
    max_single: float = Field(0.05, gt=0, le=0.2)
    max_industry: float = Field(0.20, gt=0, le=1.0)
    max_turnover: float = Field(0.30, gt=0, le=1.0)
    cost_multiplier: float = Field(1.0, ge=1.0, le=5.0)
    compute_diagnostics: bool = True
    shuffle_seed: Optional[int] = None

class BacktestRunResponse(BaseModel):
    run_id: str
    status: str                             # queued | running | done | failed
    progress: Optional[float]               # 0..1
    param_hash: Optional[str]
    data_snapshot_id: Optional[str]
    metrics: Optional[Dict[str, Any]]
    equity_curve: Optional[List[List[Any]]] # [[date, value], ...] 降采样到 ≤ 1500 点
    warnings: List[str] = []
    error: Optional[str] = None
```

**回测异步执行（不引入任何新依赖）**

```
POST /api/ashare/backtest
  ├─ 参数 clamp + factors 白名单校验（防 top_n=1e9 / start=1990 拖死机器）
  ├─ run_id = uuid4().hex[:12]
  ├─ cache.set(f"ashare:bt:{run_id}", {"status":"queued", ...}, ttl=86400)   ← 复用 cache.py
  ├─ _BT_EXECUTOR.submit(_run_backtest_job, run_id, cfg)                     ← ThreadPoolExecutor(max_workers=1)
  └─ 202 {"run_id": ..., "status": "queued"}

GET /api/ashare/backtest/{run_id}  → 从 cache 读状态；done 时从 derived 库读结果摘要
```

| 决策 | 理由 |
|---|---|
| `ThreadPoolExecutor(max_workers=1)` 而非 Celery/RQ/多进程 | 单用户、并发 ≤ 1。numpy/pandas 在计算内核里释放 GIL，不会卡死 event loop |
| 进度用轮询而非 SSE | 已有 SSE 是为 LLM token 流设计的；回测进度 3 秒轮询一次足够，少一套事件桥 |
| 状态存 `cache.py` | 已有抽象，Redis/内存自动降级，零新增 |
| `max_workers=1` 是硬上限 | 第二个回测请求排队而不是并行 —— 两个回测同时 preload 会吃 2 份内存 |

### 6.2 Agent 只读工具的注册方式

**新工具不写进 `quant_agent.py`**（遵循 CLAUDE.md G2：新工具进独立文件）。

```python
# ashare/agent_tools.py  —— 本文件是 LLM 唯一能触达 A 股数据的入口
from __future__ import annotations
from ashare.data import query
from ashare import factors
from ashare.backtest import store as bt_store   # 只 import read 侧

def query_universe(as_of: str, limit: int = 50) -> dict: ...
def get_factor_exposure(as_of: str, ts_code: str | None = None,
                        factor: str | None = None, top: int = 20) -> dict: ...
def get_signal_list(as_of: str) -> dict: ...        # P3 落地后才注册
def run_stock_report(ts_code: str, as_of: str) -> dict: ...  # P5 落地后才注册

ASHARE_TOOL_REGISTRY: dict[str, Callable] = {
    "query_universe":       query_universe,
    "get_factor_exposure":  get_factor_exposure,
    # get_signal_list / run_stock_report 在 P3 / P5 落地前【不注册】
    # —— 注册一个必然返回 not_available 的工具，只会污染 LLM 的工具选择。
}
ASHARE_TOOL_SCHEMAS: list[dict] = [ ... ]   # Anthropic 风格，与现有 TOOL_SCHEMAS 同构

# ★ 冻结的只读集合，供隔离测试断言
ASHARE_READONLY_TOOLS = frozenset(ASHARE_TOOL_REGISTRY)
```

`quant_agent.py` 的改动**只有一处，三行**，且遵循本仓库「可选依赖」惯例：

```python
# quant_agent.py，紧跟 TOOL_SCHEMAS 定义之后
try:
    from ashare.agent_tools import ASHARE_TOOL_REGISTRY, ASHARE_TOOL_SCHEMAS
    TOOL_REGISTRY.update(ASHARE_TOOL_REGISTRY)
    TOOL_SCHEMAS.extend(ASHARE_TOOL_SCHEMAS)
except ImportError:      # duckdb 未安装 / ashare 未初始化 → A 股功能整体不可用，其余照常
    pass
```

> **必须放在 `_OPENAI_TOOLS = _to_openai_tools(TOOL_SCHEMAS)` 那两行之前**
> （`quant_agent.py:4773`）—— 那两个列表在模块加载时就固化了，之后 extend `TOOL_SCHEMAS` 不会生效。
> 这是本次集成唯一的顺序陷阱。
>
> LangGraph 版（`quant_agent_lg.py`）从 `quant_agent` 导入 `TOOL_SCHEMAS` / `TRADING_TOOLS`，
> **自动继承，无需改动**。

**工具实现的三条铁律**（沿用本仓库既有约定）

| # | 规则 |
|---|---|
| T1 | 永远返回 JSON-可序列化 dict；query 层的 `QueryError` 在工具边界捕获转 `{"success": False, "error": ...}` |
| T2 | 返回体 < 3 KB（`_truncate_observation` 上游自保），股票列表默认截断到 50 条并带 `truncated` 标记 |
| T3 | 工具描述里**不出现**任何暗示可执行交易的措辞；明确写「只读、不下单、不修改任何数据」 |

### 6.3 如何保证这 4 个工具永远不被误加进 `TRADING_TOOLS`

先把语义说准，否则以后一定有人加错：

> **`TRADING_TOOLS` 不是「危险工具」标签，而是「需要登录用户身份」的闸。**
> 现状里 `get_user_profile` / `record_memory` 也在其中（`quant_agent.py:3595`），
> 它们并不危险，只是**必须知道是谁**。
> A 股只读工具查的是全市场公开数据，**没有用户维度**，因此不进 `TRADING_TOOLS`。
> 判据是「是否需要用户身份」，不是「是否重要」。

四道机制（前两道是硬闸，后两道是回归防线）：

| # | 机制 | 位置 | 失败表现 |
|---|---|---|---|
| M1 | 导入时断言 `ASHARE_READONLY_TOOLS.isdisjoint(TRADING_TOOLS)` | `tests/ashare/test_tool_registration.py` | 有人加进去 → 测试红 |
| M2 | 断言匿名用户可见：`set(ASHARE_READONLY_TOOLS) <= {s["name"] for s in get_tool_schemas_for(False)}` | 同上 | 加进 `TRADING_TOOLS` 会让匿名用户看不到 → 同一条测试也红 |
| M3 | AST 检查：`ashare/agent_tools.py` 禁止 import `brokers.*`、`ashare.data.ingest`、任何 `*.store.build/save` | `scripts/check_ashare_layering.py` | 有人在只读工具里引入写路径 → 检查红 |
| M4 | 断言这 4 个工具的 schema description 含「只读」且不含「下单/买入/卖出/委托」等动词 | 同 M1 文件 | 描述被「优化」成诱导性文案 → 红 |

M1/M2/M4 都写在**同一个测试文件**里，跑现有 `python3 -m pytest tests/ -q` 就覆盖，
不需要新建 CI 系统。

---

## 7. D1（LLM 与决策层物理隔离）的工程落地

规格 D1 只写了「不存在任何代码路径能让 LLM 输出写回」。这是**约定**，不是机制。
下面是四层机制，从硬到软，任何一层都能独立发现违规。

### 7.1 层 1（最硬）· 运行时只读连接

```python
# ashare/data/query.py 模块级，唯一连接构造点
_CONN = duckdb.connect(MARKET_DB_PATH, read_only=True)
_CONN.execute("ATTACH ? AS derived (READ_ONLY)", [DERIVED_DB_PATH])
```

DuckDB 在 `read_only=True` 的连接上执行任何 DDL/DML 会直接抛
`InvalidInputException: Cannot execute statement of type "INSERT" ... database is attached in read-only mode`。
**这一层不依赖任何检查工具，也不依赖人的自觉。** 只要 LLM 侧模块只能通过 `query.py` 拿数据，
写操作在物理上就不可能发生。

写句柄只存在于三处，且都是**离线 CLI 进程**，与 FastAPI 进程不共享地址空间：
`ashare/data/ingest.py`、`ashare/factors/store.py`、`ashare/backtest/store.py`。

### 7.2 层 2 · 静态检查脚本 `scripts/check_ashare_layering.py`

纯 AST 扫描，零依赖（`ast` + `pathlib`），退出码非 0 即失败。检查四类规则：

| 规则组 | 内容 |
|---|---|
| **A · 导入方向** | §3.2 的 L1–L6 全部规则。实现：遍历 `ast.Import` / `ast.ImportFrom`，按 (源目录 → 被导入模块) 查禁止表 |
| **B · 首参名** | `ashare/data/query.py` 中所有 public `def`（不以 `_` 开头）首参必须是 `as_of_date`。白名单**只有三个名字**：`get_tradable_mask`（首参 `exec_date`）、`preload`、`open_db`/`close_db`/`snapshot_id`（无参） |
| **C · 因子签名** | `ashare/factors/{price,fundamental,flow,risk}.py` 中被 `@factor` 装饰的函数，前两个位置参数必须严格是 `as_of_date, universe`，其余参数必须 keyword-only 且有默认值 |
| **D · 写操作黑名单** | `ashare/agent_tools.py` + `ashare/report/**` 中：<br>① 不得出现 `duckdb` 标识符；<br>② 字符串字面量不得匹配 `(?i)\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|ATTACH|COPY)\b`；<br>③ 不得调用名为 `build` / `save` / `write` / `promote` / `ingest` 的属性 |

**接入方式（不建 CI 系统）**

```python
# tests/ashare/test_layering.py
def test_layering_rules():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/check_ashare_layering.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
```

现有仓库已有 `.claude/settings.json` 的 `PostToolUse` → `py_compile` 钩子；
把 layering 检查挂进 pytest 而不是钩子，是因为它需要看全仓库而不是单文件。

### 7.3 层 3 · 目录级职责隔离

| 目录 | 能否 import `query` | 能否 import `*.store` 写函数 | 备注 |
|---|---|---|---|
| `ashare/data/ingest.py` | ✗（自己是写侧） | — | 唯一持 market 写句柄 |
| `ashare/factors/**` | ✓ | 只有 `store.py` 自己 | |
| `ashare/backtest/**` | ✓ | 只有 `store.py` 自己 | 额外获准调 `get_tradable_mask` |
| `ashare/strategy/**` (P3) | ✓ | ✗ | 信号写库由 CLI 触发，不在策略模块内 |
| `ashare/agent_tools.py` (LLM) | ✓ | **✗** | 层 2 规则 D 强制 |
| `ashare/report/**` (P5 LLM) | ✓ | **✗** | 同上 |

### 7.4 层 4 · 可观测证据

- `BacktestResult.data_snapshot_id` + `param_hash` 写进 `docs/oos-runs.md`（D7）。
- `factor_value` / `signal` 表带 `snapshot_id` 与 `writer`（`'ingest'|'factor_build'|'backtest'`）列。
  **不存在 `writer='llm'` 的合法取值** —— 出现即事故，等价于 `brokers` 子系统里
  `actor='llm' AND action='bind'` 的告警语义（ADR-0001 §8）。
- `/api/ashare/health` 暴露 `db_version` 与 `data_snapshot_id`，人工可核对。

### 7.5 明确不做的隔离手段

| 手段 | 不做的理由 |
|---|---|
| 文件系统权限（`chmod 444` + 独立 unix 账户跑 ingest） | 单机个人项目，ingest 与 server 同一用户。收益 ≈ 0，运维摩擦大。**规模到「多人可登录同一台机器」时再做** |
| 独立只读副本进程 / 数据库代理 | read_only 连接已经是物理隔离，再加一层是纯开销 |
| SELinux / 容器只读挂载 | 同上，超出当前台阶 |

---
## 8. 安全设计

本系统不碰钱（规格 N2/N6，`brokers/` 完全不动），安全面比现有交易子系统小得多，
但仍有四类必须在方案阶段解决的问题。

| 类别 | 措施 |
|---|---|
| **凭证** | `TUSHARE_TOKEN` 只进 `.env`（`.env.example` 加一行说明）。绝不进日志、异常 message、`ingest_log.error`、LLM 返回值。`sources/tushare.py` 的异常处理必须先 `str(e).replace(token, "***")` 再落库 |
| **SQL 注入** | query 层全部参数化（不变量 Q1）。`as_of_date` 经 `_norm_date()` 后是 `datetime.date` 对象，不可能携带 SQL 片段 |
| **输入校验边界** | `POST /api/ashare/backtest` 是唯一用户可控的重计算入口：`factors` 的 key 必须 ∈ `FACTOR_REGISTRY`（白名单，不接受任意字符串 → 杜绝通过因子名触发任意模块加载）；`top_n`/`max_*`/`cost_multiplier` 用 pydantic `Field(ge/le)` clamp；`start` 不早于 `2010-01-01`，`end` 不晚于最新交易日 |
| **拒绝服务** | 回测 executor `max_workers=1` + 队列长度上限 3（超出返回 429）。`GET /universe` 的 `limit` 上限 500 |
| **LLM 提示注入** | 沿用 CLAUDE.md 既有规则：RAG 检索文本、公告文本是**不可信内容**。本期新增的只读工具返回的是**数值**（因子 z 值、股票代码），不含自由文本，注入面比 RAG 更小。`run_stock_report`（P5）会拼接 RAG 文本，届时必须在 ADR 里单独处理 |
| **文件权限** | DuckDB 文件与 Parquet 冷备位于 `data/ashare/`，加进 `.gitignore`（与现有 `data/brokers.db` 同规格）。**不做** unix 权限隔离（§7.5） |

**与现有系统安全模型的关系**：不新增认证机制、不新增 threadlocal、不碰 `auth/`（CLAUDE.md G5）。
`POST /api/ashare/backtest` 直接用现成的 `Depends(require_user)`。

---

## 9. 性能与容量

### 9.1 回测 60 秒预算分解（全量 15 年 × 780 调仓日 × 含诊断）

| 阶段 | 预算 | 做法 |
|---|---|---|
| `preload` 区间行情 | 8 s | 一次 SQL 拉宽表，float32，只取用得到的列 |
| 因子读取（`factor_value` 命中） | 6 s | 780 日 × 16 因子从 derived 库一次性读入内存 |
| 组合构建 780 次 | 10 s | 纯 numpy 向量化，行业约束用分组累加而非循环 |
| 成交模拟 780 次（含掩码） | 12 s | `get_tradable_mask` 在 preload 后走内存切片 |
| 净值/指标 | 4 s | — |
| 诊断（IC / 10 分层 / Brinson 归因） | 18 s | `compute_diagnostics=False` 时整段跳过 |
| **合计** | **58 s** | 净值-only 模式 ≈ 40 s → 需再优化到 < 8 s，见下 |

> **诚实提示**：净值-only 目标 8 s 需要「组合构建 + 成交模拟」降到 ~5 s，
> 意味着这两段必须是纯数组运算，不能有 per-stock 的 Python 循环。
> 这是**引擎实现的硬性约束**，写进 `engine.py` 的模块 docstring。
> 若首版做不到 8 s，闸 3 的 200 次 shuffle 就要降到 100 次并在 `oos-runs.md` 记录口径变更。

### 9.2 内存预算

| 项 | 峰值 | 备注 |
|---|---|---|
| `preload` 宽表（6 列 × 4030 日 × 5700 只 × float32） | ≈ 550 MB | 每列 ≈ 92 MB |
| 因子面板（16 × 780 × 3100 × float32） | ≈ 155 MB | |
| 回测中间态（持仓/交易明细） | < 100 MB | |
| FastAPI 进程基线（含 chromadb + sentence-transformers） | ≈ 800 MB | 现有系统既有开销 |
| **目标峰值 RSS** | **< 2.5 GB** | 超出即触发「按列/按年分块 preload」降级 |

### 9.3 存储写入约束（影响查询性能，必须遵守）

| # | 约束 | 理由 |
|---|---|---|
| P1 | 所有事实表**按 `trade_date` 升序写入** | DuckDB 行组 zone map 靠 min/max 剪枝；乱序写入会让单日查询退化成全表扫描 |
| P2 | 回补期不建索引，完成后统一 `CREATE UNIQUE INDEX` | 见 §5.4 R1 |
| P3 | 每次 promote 前 `CHECKPOINT` 并关闭连接，确保无残留 `.wal` | 带 `.wal` 的文件被 `os.replace` 换走会损坏（§10.2） |
| P4 | `factor_value` 单独放 derived 库 | 它比行情表还大，且是可重算的派生数据 |

### 9.4 热点与降级

| 热点 | 策略 |
|---|---|
| `get_universe` 在回测里被调 780 次 | preload 后从内存算；另加进程内 LRU（key = as_of + 参数 hash），命中率 ~100%（同一次回测参数不变） |
| `get_financial_ttm` 跨 4 期拼接，因子层反复调 | 结果进 `factor_value` 前置缓存；单次回测内 LRU |
| `/api/ashare/factor-exposure` 单日查询 | 直接读 `factor_value`（已预计算），无需现算 |
| 数据库正在 promote（几十毫秒窗口） | 读者持旧 fd 继续服务旧快照，下次 `open_db()` 检测 inode 变化后重连。**没有不可用窗口** |

---

## 10. 部署与运维

### 10.1 单机拓扑

```
一台机器（开发机或单台 VPS，4 核 / 16 GB / 100 GB SSD）

  ┌── 常驻进程 ─────────────────────────────────────────┐
  │ uvicorn server:app     （FastAPI + SSE + /api/ashare/*）│
  │   └─ query.py: duckdb read_only × 2 (market, derived) │
  │   └─ ThreadPoolExecutor(max_workers=1)  回测作业       │
  │ (可选) redis-server    现有 cache.py 用                │
  └────────────────────────────────────────────────────┘

  ┌── 定时任务（crontab，各自独立进程，跑完即退）──────────┐
  │ 0 18 * * 1-5   python -m ashare.data.ingest --daily   │
  │ 30 18 * * 5    python -m ashare.factors.store --week  │
  │ 0 3  * * 6     python -m ashare.data.export_parquet   │
  │ 0 4  * * 6     python -m ashare.data.gc_versions      │
  └────────────────────────────────────────────────────┘

  ┌── 目录 ────────────────────────────────────────────┐
  │ data/ashare/market.duckdb        当前主库（read_only 对外）│
  │ data/ashare/derived.duckdb       因子 + 回测产物            │
  │ data/ashare/staging/             ingest 工作区              │
  │ data/ashare/versions/            历史版本（hardlink，留 3 份）│
  │ data/ashare/parquet/<table>/year=YYYY/*.parquet  冷备        │
  │ data/ashare/runs/<date>.json     每日 ingest 报告            │
  │ data/ashare/rate_state.json      Tushare 限速学习状态         │
  └────────────────────────────────────────────────────┘
```

**不部署的东西（以及为什么）**

| 不用 | 理由 |
|---|---|
| Docker Compose / K8s | 单进程 + cron，容器化只增加调试摩擦 |
| Kafka / RabbitMQ / Celery | 每天 1 次批任务，队列的价值为负 |
| Airflow / Prefect | 8 行 crontab + `ingest_log` 状态机已覆盖依赖与重跑 |
| Postgres / TimescaleDB | 见 ADR-0003 §决策一 |
| Prometheus / Grafana | 复用现有 `/metrics/brokers` 同款进程内计数 + `/api/ashare/health` |

### 10.2 DuckDB 并发读写限制与应对 ★

**限制（这是选型的真实代价，必须正面处理）**

| 场景 | DuckDB 行为 |
|---|---|
| 同一进程内多线程共享一个连接 | ✅ 支持（内部有并发控制） |
| 多进程同时 `read_only=True` | ✅ 支持 |
| 一个进程读写 + 另一个进程只读 | ❌ **不支持**。读写模式会取文件独占锁，其它进程连不上；反之只读进程存在时，写进程也拿不到锁 |

→ 直接后果：**`server.py` 常驻持有只读连接，`ingest.py` 就永远无法直接写主库。**
这不是可以「以后再说」的问题，是选型当天就必须给出方案的问题。

**应对：影子文件 + 原子替换（一个机制解决三个问题）**

```
① cp market.duckdb → staging/market.staging.duckdb    （0.9 GB，NVMe 上 1~2 s）
② 对 staging 独占读写：DELETE+INSERT → CHECKPOINT → close   （无锁冲突：不同文件）
③ validate(staging)：BLOCK 项失败则就地终止，主库不动
④ os.link(market.duckdb, versions/market-<snapshot>.duckdb)  （硬链接，0 拷贝，保留旧 inode）
⑤ os.replace(staging/market.staging.duckdb, market.duckdb)   （同一文件系统内原子）
⑥ 读者：下次 open_db() 比较路径 inode（≤ 每 10 s 一次），变了就重连
```

| 这个机制顺带解决的问题 | 怎么解决的 |
|---|---|
| 并发读写 | 写永远发生在另一个文件上，零锁冲突 |
| 备份 | 步骤 ④ 的硬链接就是零成本快照 |
| 回滚 | `os.replace(versions/market-<x>.duckdb, market.duckdb)`，秒级 |
| 部分失败 | 校验不过就不执行 ⑤，主库停留在上一个已知良好版本 |
| 读者一致性 | POSIX 语义下，持有旧 fd 的读者继续看到完整的旧快照，不会读到半个库 |

**三个必须遵守的细节**

1. **promote 前必须 `CHECKPOINT` + 关闭 staging 连接**，确保没有 `market.staging.duckdb.wal`
   残留。带 WAL 的库文件被单独换走 = 数据损坏。
2. **staging 与 market 必须在同一文件系统**，否则 `os.replace` 不是原子的。
3. **`derived.duckdb` 走同一套流程**，但独立版本序列 —— 因子重算不应回滚行情。

### 10.3 备份策略

| 层 | 内容 | 频率 | 保留 | 恢复耗时 |
|---|---|---|---|---|
| L0 热备 | `versions/` 硬链接快照 | 每次 promote | 3 份（market）+ 3 份（derived） | 秒级 |
| L1 冷备 | Parquet 按表按年分区导出 | 每周六 03:00 | 最近 4 周 | 15~30 min（重建 DuckDB） |
| L2 异地 | `rsync data/ashare/parquet/` 到外部盘 / 对象存储 | 每周 | 最近 3 份 | 取决于带宽 |
| L3 源头 | Tushare 原始数据（终极兜底） | — | — | 3~9 h（全量回补） |

Parquet 冷备的角色**只有两个**：灾备重建、跨工具分析。
**明确不做**「DuckDB 直接联邦查询 Parquet 当主查询层」—— 见 §12.1 砍 A2。

磁盘预算：主库 0.9 GB × 4（当前 + 3 版本）+ derived 0.7 GB × 4 + Parquet 4 周 × 1.2 GB ≈ **11 GB**。
`gc_versions` 每周清理超出保留数的版本。

### 10.4 数据回滚方案

区分两种场景，用不同手段（混用是运维事故的常见来源）：

| 场景 | 手段 | 命令 |
|---|---|---|
| **今天这次 promote 出了问题**（校验漏检、字段错位） | 文件级回滚 | `python -m ashare.data.rollback --to <version>` → `os.replace` + 通知读者重连 |
| **发现 N 天前某个分区数据错了**（数据源修正、复权因子回溯调整） | 分区级重拉（**常态**） | `python -m ashare.data.ingest --dataset daily_bar --from 2026-08-01 --to 2026-08-05 --force`，幂等 DELETE+INSERT |
| **因子逻辑改了** | 派生数据重建 | `python -m ashare.factors.store --rebuild <factor_name>`，只动 derived 库 |
| **数据快照变了但回测结果还在引用旧快照** | 不回滚，**标记失效** | `backtest_run.data_snapshot_id != 当前` → `/api/ashare/backtest/runs` 上标 `stale`；D7 的样本外记录必须重跑 |

> 最后一条是 D7 的真实执行细节：**参数没改但数据变了，样本外结果同样不可比。**
> 规格只锁了 `param_hash`，这是漏洞，见 §12.2 补 B1。

### 10.5 监控（复用现有模式，不引新组件）

| 信号 | 来源 | 阈值 / 动作 |
|---|---|---|
| 最新交易日落后 | `/api/ashare/health.latest_trade_date` | 落后 > 1 个交易日 → 邮件告警 |
| `ingest_log` FAILED 分区数 | 同上 | `> 0` → 邮件告警 |
| 校验 BLOCK 项 | `runs/<date>.json` | 任意一项失败 → 邮件（复用 `scripts/alert_worker.py`） |
| 双源交叉偏差 | 同上 | 偏差 > 0.5% 的样本占比 > 1% → 邮件 |
| `writer='llm'` 出现在任何派生表 | SQL 巡检（每周） | **critical**，等同 ADR-0001 §8 的 LLM 越权告警 |
| 回测队列积压 | 进程内计数 | > 3 → 返回 429 |

---
## 11. 演进路径（台阶与触发条件）

每一步都写清**触发条件**，没到触发条件就不做。这是防止今天过度设计的唯一有效手段。

| 台阶 | 触发条件（可观测） | 演进动作 | 预估成本 |
|---|---|---|---|
| **T0 · 本期** | — | 单机 / DuckDB 两文件 / cron / 单线程回测 | — |
| T1 · 数据量增长 | `market.duckdb` > 20 GB（例如引入分钟线或逐笔） | 停止 `preload` 全量入内存，改为按年分块 + 物化视图；DuckDB 仍够用 | 2~3 天 |
| T2 · 回测并发 | 同时排队的回测请求持续 > 2 | `ThreadPoolExecutor` → `ProcessPoolExecutor(2)` + 文件锁；**不引入 Celery** | 1 天 |
| T3 · 多用户 | 真有第 2 个人用这套系统 | `/api/ashare/*` 加 `require_user`；回测按用户配额；`backtest_run` 加 `user_id` | 2 天 |
| T4 · 需要事务并发写 | 出现「盘中实时更新 + 同时查询」需求（本期不存在，周频用不着） | 才考虑 Postgres/Timescale。见 ADR-0003 的退出成本评估 | 1~2 周 |
| T5 · 策略进生产 | 五闸全过，开始按信号真实下单 | 才谈 A 股下单通道、审计日志、`brokers/` 扩展 —— **本期明确不做（N2/N6）** | 另开 ADR |
| T6 · 知识库时点隔离 | P4 开工 | `rag/retriever.py` 加 `as_of_date` 过滤（chroma `where` 支持 `$lte`） | 1 天（前提：B11 已提前做） |

---

## 12. 架构风险评审（对设计规格的意见）

以下是以架构师视角对规格的评审结论。**PM 拥有最终裁决权**，但每条都给出了依据与成本。

### 12.1 判定为过度设计 —— 建议砍

| # | 规格位置 | 问题 | 建议 | 省下的成本 |
|---|---|---|---|---|
| **A1** | §4.2 `money_flow` 表的 `margin_balance` / `net_mf_amount` | 两个字段各自只服务一个因子，其中 `margin_chg_20` 规格自己标注「待检验」，`net_mf_amount` 根本没进因子清单。为一个待检验因子每天多拉 2 个接口、回补多 8,000 次调用 | 本期 `money_flow` **只入 `hk_hold_ratio`**（`north_hold_chg_20` 与 P3 的 `north_flow_60` 都要用它）。`margin_*` / 主力净流入等因子过闸后再补库，回补是幂等的，随时可加 | 回补少 ≈ 70 min + Tushare 积分；每日 ingest 少 2 个接口 |
| **A2** | §4.1「DuckDB 单文件 + Parquet 冷备」 | 「Parquet」这个词很容易被后续实现理解成「DuckDB 直接查 Parquet 的联邦查询层」。那会慢 5~10 倍（无 zone map 剪枝、无主键约束）且引入两套真值 | 明确写死：**Parquet 只是导出产物**，用于灾备重建与跨工具分析，**不参与任何在线查询路径**。已落在 §10.3 | 避免一次架构返工 |
| **A3** | §5.5 闸 3「Shuffle 对照 200 次」 | 单次全量回测 60 s × 200 = **3.3 小时**。这不是设计缺陷，但成本被严重低估，会导致闸 3 实际上没人跑 | 闸 3 内部强制 `compute_diagnostics=False`（shuffle 只需要 Sharpe 分布，不需要 IC/分层/归因），单次降到 8 s → 200 次 ≈ 27 min。已落进 §4.3 的 `BacktestConfig` | 3.3 h → 27 min |
| **A4** | §10「新增 4 个只读工具」 | `get_signal_list`（依赖 P3）与 `run_stock_report`（依赖 P5）在本期没有数据可返回。注册一个永远返回「功能未上线」的工具，只会污染 LLM 的工具选择、白白占 prompt token | 本期只注册 `query_universe`、`get_factor_exposure` 两个。另两个的 schema 先写好放着，P3/P5 落地当天再加进 registry | 每轮 LLM 调用少 2 个工具 schema |
| **A5** | §5.3「引擎核心 < 400 行」 | 把掩码/成本/约束/指标全算上，真实实现量在 1,200~1,500 行。硬凑「核心 400 行」的结果通常是把逻辑挪到别的文件再声称核心很小 —— 文字游戏，不产生任何工程收益 | 改成可检查的口径：**`engine.py` ≤ 400 行且只做编排**，实体逻辑按 §4.3 的表分到 `portfolio/execution/cost/metrics.py`，每个文件职责单一。已落进 §3.2 规则 L6 | 避免自欺式的模块划分 |
| **A6** | §7 P4 知识库 | 本期不交付 P4，但 `rag/indexer.py` 的 metadata 变更是**破坏性**的：现有 chunk 没有 `publish_date`，加字段必须重建索引，语料越多越贵 | 分两半：**现在就做**「indexer 写入 `publish_date` 字段」（约 5 行 + 文件名解析 + 手工 override JSON），**本期不做**检索过滤逻辑。这是少数「提前一步反而更便宜」的情况 | 未来重建索引的成本随语料线性增长 |

**考虑过但决定不砍的**

| 项 | 为什么不砍 |
|---|---|
| `macro_indicator` 的 9 个指标（P3 只用到 5 个） | 每个指标 1~2 次调用、总共 1,700 行。成本可忽略，而宏观数据补历史比补行情麻烦（`publish_date` 只能靠观测积累），早入库反而对 |
| `index_daily.pe_ttm`（只服务 P3 的 ERP） | 4 次调用。同上 |
| 申万三级行业（`sw_l2` / `sw_l3`） | 一次性字段，不增加日常拉取成本；三级行业在个股报告里有用 |
| 双数据源交叉校验（BaoStock） | 这是 P1 验收硬指标，且抽样只有 200 次调用。数据正确性是整个系统的地基，这里省钱是最坏的省钱 |

### 12.2 判定为欠设计 —— 必须补（不补会留坑）

| # | 缺口 | 后果 | 补法 | 严重度 |
|---|---|---|---|---|
| **B1** | 没有 `ingest_log`，也没有**数据快照标识** | ① 拉取中断无法断点续传，只能从头再来；② **D7 被架空** —— `param_hash` 只锁参数，但重述财报、复权因子回溯修正会让同参数跑出不同结果，样本外污染无法追溯 | 补 `ingest_log` 表（§5.3）+ `query.snapshot_id()`（§4.1）；`BacktestResult` 与 `docs/oos-runs.md` 必须同时记 `param_hash` **和** `data_snapshot_id` | **高（威胁 D7）** |
| **B2** | 涨跌停价的**数据可得性**没有兜底方案 | D6 直接依赖 `limit_up/limit_down`。Tushare `stk_limit` 有积分门槛，拿不到就没法判定一字板 | 补 `ashare/data/limits.py` 规则兜底，并把已知规则边界硬编码 + 列出**已知不准确场景**：新股上市首日（主板无涨跌幅/科创创业前 5 日无限制）、退市整理期、ST 戴帽摘帽当日、2019-07-22 科创板开板、2020-08-24 创业板改 20%、北交所 30%。规则算不出的当日 → `limit_unknown` → **该股当日不可交易**（保守，不假设可交易） | **高（威胁 D6）** |
| **B3** | **停牌日没有占位行** | Tushare `daily` 在停牌日**不返回该股的行**。若不补行，`get_bars(lookback=20)` 拿到的是「最近 20 条记录」而不是「最近 20 个交易日」—— 一只停牌 5 天的股票，它的 `reversal_20` 实际覆盖 25 个交易日。**横截面因子被静默污染，且完全没有报错**。这是全套设计里最隐蔽的 bug 源 | ingest 的 normalize 阶段补行：`在市 AND 日历交易日 AND daily 无行 → 插入 OHLC=NULL, is_suspended=TRUE`。`get_bars` 的 lookback 一律按**日历交易日**计数而非记录数 | **高（静默数据污染）** |
| **B4** | 因子值没有落库表 | 五闸的闸 3（200 次 shuffle）与闸 5（±30% 参数网格）都要反复用同一批因子值。每次现算 16 因子 × 780 日 = 全量回测最贵的部分，60 s 目标不可能达成 | 补 `factor_value(factor_name, param_hash, trade_date, ts_code, raw_value, processed_value, snapshot_id)` 与 `ashare/factors/store.py`（§4.2）。缓存 key 必须含 `param_hash` + `snapshot_id`，否则会用旧数据算出的因子跑新回测 | 中高（威胁 60 s 指标） |
| **B5** | 因子的**数据可得起始日**没有表达方式 | `north_hold_chg_20` 依赖北向持股，2016-12 陆股通才有数据。**2026-08-21 修正：本栏原来的「后果」说错了危险在哪。** 一个因子【整体】缺失时，「填 0」与「剔除后重新归一」只差一个逐日常数，而 §7.2 只拿分数做排序 —— 300 只票实测目标权重**逐位相同**，B5 原本描述的那种失真不存在。真正的损害在【部分】覆盖：`process` 末尾的 `fillna(0)` 会把缺失的那 40% 判成「行业内平均水平」，同一测试下 50 只选中的票**换掉 13 只**。规则不变（仍是剔除+归一），但理由要改对，否则下一个人会去防错的东西 | `FactorSpec.available_from` + `combine()` 在覆盖率不足或未到起始日时**把该因子从当日分母中剔除并重新归一**（§4.2） | 中高 |
| **B6** | `stock_status`（ST 历史）**没有数据来源** | 规格 §4.2 凭空定义了这张表，但 Tushare 没有「ST 状态历史区间」接口。没有它，D5 的 ST 剔除只能用「今天是不是 ST」→ 又一个幸存者偏差 | 由 `namechange`（历史名称变更）反推：名称含 `ST`/`*ST` 的区间即 ST 区间。**必须写明边界**：用变更**生效日**而非公告日；名称里的 `S`（未股改）、`退` 单独归类；反推结果与当前 `stock_basic.name` 交叉校验，不一致的进人工清单 | 中高（威胁 D5） |
| **B7** | 中性化 OLS 的**退化情形**没有规定 | 申万一级 31 个行业里，早年部分行业成分股 < 5 只；某些日期整个池子 < 100 只。设计矩阵秩亏 → OLS 抛异常或返回全 NaN | `get_industry` 把成分 < 5 的行业并入 `__OTHER__`；`neutralize()` 在有效样本 < 30 或秩亏时**返回原始 Series 并记 warning**，而不是静默返回 NaN（§4.2） | 中 |
| **B8** | 退市清仓的成交价假设**过于乐观** | 规格 §5.3 说「`delist_date` 前最后一个交易日按收盘价强制清仓」。退市整理期股票连续跌停、几乎无流动性，按收盘价成交是系统性乐观偏差 | ① `get_universe` 剔除 `DELIST_PERIOD` 状态（大部分已被 ST 规则覆盖）；② 若持仓中仍出现，按**最后一个有效后复权收盘价 × 0.5** 清仓入账，并在 `BacktestResult.warnings` 记录。宁可低估也不能高估。**2026-08-20 修正**：初稿写「整理期首日收盘价」，但掩码只看得到当前 bar、不知道整理期哪天开始 —— 拿不到那个值。末日收盘价在连续跌停序列的末端，因此比首日方案更保守，与本行自己的「宁可低估」同向 | 中 |
| **B9** | 回测 `< 60 s` 没有定义**前置条件** | 「含不含因子预计算」「含不含诊断」口径不同，差 10 倍 | 定死为：**因子已落库 + `compute_diagnostics=True` + 全市场 15 年周频**。因子冷启动预计算另计（< 20 min）。已落进 §1.2 / §9.1 | 中 |

### 12.3 规格含糊、本文已定死（无需 PM 决策，备案即可）

| 规格位置 | 含糊点 | 本文的裁决 |
|---|---|---|
| §4.3 股票池 | 六条剔除规则没定**顺序**，「20 日均额后 20%」在剔除前算还是剔除后算，结果不同 | 先做 1–5 硬性剔除，在**剩余池内**算流动性分位（§4.1 `get_universe`） |
| §5.1 基本面因子 | TTM 拼接逻辑归属未定（query 层还是因子层） | **query 层** `get_financial_ttm`，理由见 §4.1 |
| §5.2 因子处理链 | 「缺失值填 0」是在 zscore 前还是后 | zscore 之后（顺序 1→2→3→4，中性化后 0 = 行业均值，这也是规格自己的解释） |
| §5.3 引擎伪码 | `weekly_dates` 的定义（自然周五 vs 每周最后一个交易日） | **每周最后一个交易日**，唯一实现点 `query.get_trade_dates(freq='W')` |
| §2 D2 | 「Factor 抽象基类」 | 改为 `@factor` 装饰器 + `FactorSpec` dataclass，理由见 §4.2 |
| §8 | 「新增 Agent 工具不加入 `TRADING_TOOLS`」只是约定 | 落成 4 道可执行机制（§6.3），并澄清 `TRADING_TOOLS` 的真实语义是「需要用户身份」而非「危险」 |

### 12.4 工作量量级（供 PM 排期，不是承诺）

| 阶段 | 内容 | 量级 |
|---|---|---|
| P1-a | schema + sources adapter + ingest 状态机 + limits 兜底 | 4~6 人日 |
| P1-b | query.py 全部签名 + PIT 语义 + 单测 | 3~4 人日 |
| P1-c | validate 六项 + 双源交叉 + 首次全量回补（含跑数等待） | 3~4 人日 |
| P2-a | 因子框架 + 16 个因子 + pipeline + factor_value 落库 | 5~7 人日 |
| P2-b | 回测引擎 + 成本 + 指标 + D6 三个单测 | 5~7 人日 |
| P2-c | 五闸 + 引擎正确性反测（§5.6） | 3~4 人日 |
| 集成 | REST 端点 + 2 个 Agent 工具 + 隔离检查脚本 | 2~3 人日 |
| **合计** | | **25~35 人日** |

> 若要压缩：可先砍 P2-c 的闸 2（walk-forward）与闸 5（参数高原），保留闸 1/3/4，
> 但**必须在 `docs/oos-runs.md` 里写明「本次未跑闸 2/5」**，否则等于自欺。

---

## 13. 风险与未决项

### 13.1 风险

| 风险 | 影响 | 缓解 | 剩余风险 |
|---|---|---|---|
| Tushare 积分不足以覆盖 `stk_limit` / 财报接口 | D6 无法精确实现 | B2 规则兜底 + `limit_unknown` 保守不交易 | 一字板判定在少数边界日不准，会低估收益（保守方向） |
| 宏观 `publish_date` 历史值只能靠规则回填 | D4 在 2010–2026 历史区间精度受限 | `publish_date_source ∈ {observed, rule}` 列 + ADR 显式承认 | 回测的宏观择时层存在数日级的时点误差；下限 20% 仓位设计已在一定程度上吸收 |
| 净值-only 回测降不到 8 s | 闸 3 成本失控 | 降到 100 次 shuffle 并记录口径 | 统计功效下降 |
| DuckDB 版本升级导致文件格式不兼容 | 旧版本文件读不了 | `requirements.txt` 对 duckdb 做**大版本锁定**（同 tigeropen 的处理）；Parquet 冷备是跨版本的逃生通道 | 低 |
| 数据源接口字段变更 | ingest 静默错位 | normalize 阶段做 schema 断言，字段缺失即 FAILED 不重试 | 低 |
| `preload` 内存超出 | OOM | 峰值预算 2.5 GB + 按列/按年分块降级路径 | 低 |
| 引擎有 bug 但回测「看起来不错」 | 最危险的一类 | 规格 §5.6 的已知异象反测（`reversal_20`/`turnover_20` 分层单调性）作为**硬性验收**，跑不出来判定为 bug | 中 —— 这条必须真的执行，不能因为「先跑通」而跳过 |

### 13.2 未决项（需要 PM / 用户拍板）

| # | 未决项 | 选项 | 建议 |
|---|---|---|---|
| U1 | Tushare 积分档位 | 现有积分是多少？决定 §5.4 的耗时区间与 B2 是否必须走兜底 | 开工第一天先用 30 分钟实测各接口限频，写进 `rate_state.json` |
| U2 | A1（砍 `margin_balance`/`net_mf_amount`）是否接受 | 接受 / 保留全量 | 建议接受，随时可幂等补回 |
| U3 | A4（本期只注册 2 个 Agent 工具）是否接受 | 接受 / 全注册 | 建议接受 |
| U4 | A6（现在就给 RAG chunk 加 `publish_date`）是否纳入本期 | 纳入（+0.5 人日）/ 推迟 | 建议纳入 |
| U5 | 回测结果保留策略 | 全留 / 只留通过五闸的 | 建议每个 `param_hash` 只留最新一次 + 所有样本外运行永久保留 |
| U6 | `docs/oos-runs.md` 的写入方式 | 手工 / 由 `run_backtest` 自动追加 | 建议**自动追加**（人工记录必然漏记，D7 就失效了） |

---

## 附录 A · 新增目录（相对规格 §10 的增量部分已标 ★）

```
ashare/
├── data/
│   ├── sources/{tushare,baostock,akshare}.py
│   ├── schema.sql                  # 含 ★ ingest_log 表
│   ├── ingest.py                   # 唯一 market 写者 + 状态机 + 影子替换
│   ├── validate.py
│   ├── limits.py             ★ 涨跌停规则兜底（B2）
│   ├── promote.py            ★ staging → 原子替换 / 版本 / 回滚（§10.2）
│   ├── export_parquet.py     ★ 冷备导出
│   └── query.py                    # 唯一数据出口
├── factors/
│   ├── base.py                     # @factor + FactorSpec + FACTOR_REGISTRY
│   ├── pipeline.py
│   ├── store.py              ★ factor_value 预计算落库（B4）
│   └── {price,fundamental,flow,risk}.py
├── backtest/
│   ├── engine.py                   # ≤ 400 行，只编排
│   ├── portfolio.py          ★ 组合构建（从 engine 拆出）
│   ├── execution.py          ★ 成交模拟（从 engine 拆出）
│   ├── cost.py / metrics.py / guards.py
│   ├── types.py              ★ BacktestConfig / BacktestResult
│   └── store.py              ★ 回测结果持久化
├── strategy/                       # P3，本期只放签名占位
└── agent_tools.py            ★ LLM 只读工具（原规格放在 quant_agent.py，此处独立成文件以满足 G2）

scripts/
└── check_ashare_layering.py  ★ D1 / 分层 / 签名静态检查（§7.2）

tests/ashare/                 ★
├── test_layering.py                # 调用上面的检查脚本
├── test_tool_registration.py       # M1/M2/M4（§6.3）
├── test_query_pit.py               # D2/D3/D4 断言
├── test_universe.py                # D5 断言（规格 §11）
└── test_engine_semantics.py        # D6 三个单测：涨停买不进/跌停卖不出/停牌
```

## 附录 B · CLAUDE.md 需要新增的一节

规格 §10 要求把 D1–D8 写进 `CLAUDE.md`。除此之外建议追加三条**本文新增的硬约束**：

1. `ashare/data/query.py` 是唯一数据出口；任何模块 `import duckdb` 都要过 `scripts/check_ashare_layering.py`。
2. 任何回测/信号产物必须同时记录 `param_hash` 与 `data_snapshot_id`，缺一不可（D7 补强）。
3. `ashare/agent_tools.py` 的工具**永远不进 `TRADING_TOOLS`**；`TRADING_TOOLS` 的语义是「需要用户身份」，不是「危险」。

---

文档结束。维护责任：§4 接口契约变更必须同步改 `scripts/check_ashare_layering.py` 的白名单；
§6.1 端点变更同步更新 `docs/technical-architecture.md` §12.2 端点速查。
