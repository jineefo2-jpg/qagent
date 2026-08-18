# P1 · A 股 PIT 数据底座 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成一个全 A 股（含已退市）2010-01-01 至今的 PIT 数据仓库，对外只暴露 `ashare/data/query.py` 一个出口，且所有前视偏差/幸存者偏差在 API 签名层面就难以写出。

**Architecture:** DuckDB 单文件（`data/ashare_market.duckdb`）+ 影子文件原子替换。Tushare Pro 为主源、BaoStock 交叉校验。`ingest.py` 是唯一写者，`query.py` 是唯一读者（`read_only=True` 连接）。停牌日补占位行、财报按 `ann_date` 取数、宏观按 `publish_date` 取数。

**Tech Stack:** Python 3.9.6、DuckDB、pandas、tushare、baostock、pytest

## Global Constraints

- Python **3.9.6**。`ashare/**` 每个文件首行 `from __future__ import annotations`（可用 `list[str] | None` 注解）。`server.py` 的 pydantic 模型**必须**用 `Optional[X]`/`List[X]` —— pydantic 在 3.9 运行时求值注解，`X | None` 抛 `TypeError`。
- 铁律 D1–D9 见 `CLAUDE.md` 的「A 股数据与回测铁律」一节，全部适用。
- `query.py` 所有公开函数首参名必须是 `as_of_date`。**唯一豁免**：`get_tradable_mask(exec_date, ...)`。
- `query.py` 所有 SQL 用 `?` 参数化，**禁止字符串拼接日期/代码**。
- `query.py` 空结果返回**带正确列名的空 DataFrame/Series**，绝不返回 `None`。
- `query.py` 层 **raise** 异常（`QueryError` 子类），不返回 `{"error": ...}`。
- 数据库路径：market = `data/ashare_market.duckdb`，derived = `data/ashare_derived.duckdb`。均加入 `.gitignore`。
- 依赖新增：`duckdb>=1.0.0,<2.0.0`（**大版本锁定**，同 tigeropen 处理）、`tushare>=1.4.0`、`baostock>=0.8.8`。全部走仓库既有的 `try: import X except ImportError` 可选依赖模式。
- 提交信息格式：`feat(ashare): ...` / `test(ashare): ...` / `fix(ashare): ...`

## 未决项裁决（架构师 §13.2 的 U1–U6）

| # | 裁决 | 理由 |
|---|---|---|
| U1 | **Task 0 实测**，不阻塞开工 | 30 分钟能测出来的事不值得停工等 |
| U2 | **接受砍** `margin_balance`/`net_mf_amount` | 服务于一个规格自标「待检验」的因子。回补幂等，过闸后随时补 |
| U3 | **接受**本期只注册 2 个 Agent 工具 | 永远返回「未上线」的工具只污染 LLM 工具选择 |
| U4 | **接受纳入** RAG chunk 加 `publish_date`（Task 15） | 重建索引成本随语料线性增长，提前做更便宜 |
| U5 | 每个 `param_hash` 留最新一次 + **样本外运行永久保留** | 样本外记录是 D7 的唯一凭据 |
| U6 | `docs/oos-runs.md` **由代码自动追加** | 人工记录必然漏记，漏记即 D7 失效 |

---

### Task 0: Tushare 接口权限与限频实测

**Files:**
- Create: `scripts/probe_tushare.py`
- Create: `data/rate_state.json`（gitignored）
- Create: `docs/ashare-datasource-probe.md`

**Interfaces:**
- Consumes: 无
- Produces: `docs/ashare-datasource-probe.md` 记录各接口可用性与实测限频；后续 Task 3 的令牌桶读 `data/rate_state.json`

- [ ] **Step 1: 写探测脚本**

```python
# scripts/probe_tushare.py
"""实测 Tushare 各接口的可用性与限频。开工第一天跑一次，30 分钟内出结果。
不写任何业务数据，只写 data/rate_state.json 与探测报告。"""
from __future__ import annotations
import json, os, pathlib, time
import tushare as ts

# (接口名, 调用 lambda, 是否 P1 必需)
PROBES = [
    ("trade_cal",     lambda p: p.trade_cal(exchange="SSE", start_date="20240101", end_date="20240131"), True),
    ("stock_basic",   lambda p: p.stock_basic(exchange="", list_status="L", fields="ts_code,name,list_date"), True),
    ("namechange",    lambda p: p.namechange(ts_code="600519.SH"), True),
    ("daily",         lambda p: p.daily(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("adj_factor",    lambda p: p.adj_factor(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("stk_limit",     lambda p: p.stk_limit(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("daily_basic",   lambda p: p.daily_basic(ts_code="600519.SH", start_date="20240101", end_date="20240131"), True),
    ("fina_indicator",lambda p: p.fina_indicator(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("income",        lambda p: p.income(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("balancesheet",  lambda p: p.balancesheet(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("cashflow",      lambda p: p.cashflow(ts_code="600519.SH", start_date="20200101", end_date="20240101"), True),
    ("hk_hold",       lambda p: p.hk_hold(trade_date="20240102"), True),
    ("index_daily",   lambda p: p.index_daily(ts_code="000985.CSI", start_date="20240101", end_date="20240131"), True),
    ("cn_m",          lambda p: p.cn_m(start_m="202301", end_m="202312"), True),
    ("shibor",        lambda p: p.shibor(start_date="20240101", end_date="20240131"), True),
]

def main() -> int:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("ERROR: 未设置 TUSHARE_TOKEN 环境变量")
        return 1
    pro = ts.pro_api(token)

    results = []
    for name, call, required in PROBES:
        t0 = time.time()
        try:
            df = call(pro)
            results.append({"api": name, "ok": True, "rows": len(df),
                            "elapsed": round(time.time() - t0, 2), "required": required, "error": ""})
            print(f"  OK   {name:<16} rows={len(df):<6} {time.time()-t0:.2f}s")
        except Exception as exc:            # noqa: BLE001 — 探测脚本要看到全部错误类型
            results.append({"api": name, "ok": False, "rows": 0,
                            "elapsed": round(time.time() - t0, 2), "required": required,
                            "error": str(exc)[:200]})
            print(f"  FAIL {name:<16} {str(exc)[:120]}")
        time.sleep(0.5)

    blocked = [r for r in results if not r["ok"] and r["required"]]
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/rate_state.json").write_text(
        json.dumps({"probed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results,
                    "calls_per_min": 120}, ensure_ascii=False, indent=2))

    print(f"\n必需接口不可用: {len(blocked)} 个")
    for r in blocked:
        print(f"  - {r['api']}: {r['error']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行探测**

```bash
TUSHARE_TOKEN=<你的token> python3 scripts/probe_tushare.py
```

Expected: 逐行打印每个接口 OK/FAIL。末尾给出「必需接口不可用」计数。

- [ ] **Step 3: 把结果写成决策文档**

创建 `docs/ashare-datasource-probe.md`，内容为一张表（接口 / 可用 / 行数 / 耗时 / 备选方案），并对每个 FAIL 的接口写明替代路径：

| FAIL 的接口 | 替代方案 |
|---|---|
| `stk_limit` | 走 `ashare/data/limits.py` 规则兜底（Task 4），算不出的日期标 `limit_unknown` → 该股当日不可交易 |
| `fina_indicator` | 用 `income`/`balancesheet`/`cashflow` 三张原始表自行计算派生指标 |
| `hk_hold` | 砍掉 `north_hold_chg_20` 因子（P2）与 `north_flow_60` 宏观指标（P3），宏观层降为 4 指标 |
| `cn_m` / `shibor` | 从 akshare 对应接口取，`publish_date` 走规则回填并标 `publish_date_source='rule'` |

- [ ] **Step 4: 提交**

```bash
git add scripts/probe_tushare.py docs/ashare-datasource-probe.md
git commit -m "chore(ashare): Tushare 接口权限与限频实测脚本 + 探测结果"
```

---

### Task 1: 项目骨架 + schema + DuckDB 连接

**Files:**
- Create: `ashare/__init__.py`, `ashare/data/__init__.py`, `ashare/data/sources/__init__.py`
- Create: `ashare/data/schema.sql`
- Create: `ashare/data/_db.py`
- Create: `tests/ashare/__init__.py`, `tests/ashare/conftest.py`, `tests/ashare/test_schema.py`
- Modify: `requirements.txt`, `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces:
  - `ashare.data._db.connect_write(path: str) -> duckdb.DuckDBPyConnection`
  - `ashare.data._db.connect_read(path: str) -> duckdb.DuckDBPyConnection`（`read_only=True`）
  - `ashare.data._db.init_schema(conn) -> None`
  - `ashare.data._db.SCHEMA_VERSION: int`

- [ ] **Step 1: 写失败测试**

```python
# tests/ashare/conftest.py
from __future__ import annotations
import pathlib, tempfile
import pytest

@pytest.fixture
def tmp_db(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "test_market.duckdb")
```

```python
# tests/ashare/test_schema.py
from __future__ import annotations
import pytest
duckdb = pytest.importorskip("duckdb")
from ashare.data import _db

EXPECTED_TABLES = {
    "calendar", "stock_basic", "stock_status", "daily_bar", "daily_basic",
    "financial_pit", "macro_indicator", "money_flow", "index_daily", "ingest_log",
}

def test_init_schema_creates_all_tables(tmp_db):
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    got = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert EXPECTED_TABLES <= got, f"缺表: {EXPECTED_TABLES - got}"
    conn.close()

def test_init_schema_is_idempotent(tmp_db):
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    _db.init_schema(conn)          # 第二次不得抛
    conn.close()

def test_financial_pit_pk_includes_ann_date(tmp_db):
    """D3：ann_date 必须在主键里，否则同报告期的多次披露会互相覆盖。"""
    conn = _db.connect_write(tmp_db)
    _db.init_schema(conn)
    conn.execute("""INSERT INTO financial_pit (ts_code, ann_date, end_date, report_type,
                    update_flag, n_income_attr_p) VALUES
                    ('600519.SH','2021-04-01','2020-12-31','1',0, 100.0),
                    ('600519.SH','2021-04-28','2020-12-31','1',0, 101.0)""")
    n = conn.execute("SELECT count(*) FROM financial_pit").fetchone()[0]
    assert n == 2, "同报告期不同公告日必须能共存，否则 PIT 无从谈起"
    conn.close()

def test_read_only_connection_rejects_write(tmp_db):
    """D1 的最硬一层：只读连接上任何 DML 都必须抛异常。"""
    w = _db.connect_write(tmp_db); _db.init_schema(w); w.close()
    r = _db.connect_read(tmp_db)
    with pytest.raises(Exception):
        r.execute("INSERT INTO calendar (trade_date, is_open) VALUES ('2024-01-02', TRUE)")
    r.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/ashare/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ashare'`

- [ ] **Step 3: 加依赖**

在 `requirements.txt` 末尾追加：

```
# A 股数据底座（P1）
# duckdb 大版本锁定：文件格式跨大版本不保证兼容，Parquet 冷备是逃生通道
duckdb>=1.0.0,<2.0.0
tushare>=1.4.0
baostock>=0.8.8
```

在 `.gitignore` 末尾追加：

```
# A 股数据底座
data/ashare_market.duckdb*
data/ashare_derived.duckdb*
data/ashare_staging/
data/rate_state.json
data/parquet_export/
```

装依赖：

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: 写 schema.sql**

```sql
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
```

- [ ] **Step 5: 写 `_db.py`**

```python
# ashare/data/_db.py
"""DuckDB 连接与 schema 管理。ingest 是唯一写者，query 只用 read_only 连接。"""
from __future__ import annotations
import pathlib
from typing import Any

import duckdb

SCHEMA_VERSION = 1
_SCHEMA_SQL = pathlib.Path(__file__).with_name("schema.sql")

DEFAULT_MARKET_PATH = "data/ashare_market.duckdb"
DEFAULT_DERIVED_PATH = "data/ashare_derived.duckdb"


def connect_write(path: str) -> duckdb.DuckDBPyConnection:
    """可写连接。只有 ingest.py / promote.py 允许调用。"""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def connect_read(path: str) -> duckdb.DuckDBPyConnection:
    """只读连接。D1 的最硬一层 —— DuckDB 在 read_only 连接上执行任何 DML 直接抛异常，
    不依赖任何人的自觉。query.py 只能用这个。"""
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"数据库不存在: {path}（先跑 python -m ashare.data.ingest）")
    return duckdb.connect(path, read_only=True)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """建表。幂等（全部 CREATE TABLE IF NOT EXISTS）。"""
    conn.execute(_SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [str(SCHEMA_VERSION)],
    )
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python3 -m pytest tests/ashare/test_schema.py -v`
Expected: 4 passed

- [ ] **Step 7: 提交**

```bash
git add ashare/ tests/ashare/ requirements.txt .gitignore
git commit -m "feat(ashare): P1 骨架 + DuckDB schema + 只读连接守卫"
```

---

### Task 2: 分层静态检查脚本（先立守门人）

**Files:**
- Create: `scripts/check_ashare_layering.py`
- Create: `tests/ashare/test_layering.py`

**Interfaces:**
- Consumes: `ashare/**` 的源码（AST 解析，不导入）
- Produces: `check_ashare_layering.check(root: str = "ashare") -> list[str]`（返回违规列表，空 = 通过）

> 这个任务排在写业务代码之前：守门人先到位，后面每个任务的最后一步都能跑它。

- [ ] **Step 1: 写失败测试**

```python
# tests/ashare/test_layering.py
from __future__ import annotations
import pathlib, sys, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import check_ashare_layering as chk

def test_real_codebase_passes():
    """真实 ashare/ 必须零违规。这是每个任务的收尾闸门。"""
    assert chk.check("ashare") == []

def _write(tmp_path, rel, src):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src))
    return p

def test_detects_direct_duckdb_import(tmp_path):
    _write(tmp_path, "factors/price.py", "import duckdb\n")
    v = chk.check(str(tmp_path))
    assert any("duckdb" in x for x in v)

def test_allows_duckdb_in_data_layer(tmp_path):
    _write(tmp_path, "data/_db.py", "import duckdb\n")
    assert chk.check(str(tmp_path)) == []

def test_detects_wrong_first_param_in_query(tmp_path):
    _write(tmp_path, "data/query.py", "def get_bars(date, codes): ...\n")
    v = chk.check(str(tmp_path))
    assert any("as_of_date" in x for x in v)

def test_allows_whitelisted_get_tradable_mask(tmp_path):
    _write(tmp_path, "data/query.py", "def get_tradable_mask(exec_date, ts_codes): ...\n")
    assert chk.check(str(tmp_path)) == []

def test_detects_write_in_report_layer(tmp_path):
    """D1：LLM 层（report/、agent_tools.py）不得出现写操作。"""
    _write(tmp_path, "report/stock_deep.py", "def f(c):\n    c.execute('INSERT INTO x VALUES (1)')\n")
    v = chk.check(str(tmp_path))
    assert any("INSERT" in x.upper() for x in v)
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/ashare/test_layering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'check_ashare_layering'`

- [ ] **Step 3: 写检查脚本**

```python
# scripts/check_ashare_layering.py
"""ashare/ 分层与签名静态检查（AST，不导入被检查模块）。

四类规则：
  L1 导入方向：只有 ashare/data/** 可以 import duckdb
  L2 首参名  ：ashare/data/query.py 的公开函数首参必须是 as_of_date（白名单除外）
  L3 因子签名：ashare/factors/{price,fundamental,flow,risk}.py 的公开函数前两个位置参数
               必须是 (as_of_date, universe)
  L4 写操作  ：ashare/report/**、ashare/agent_tools.py 不得出现 DML 字符串（D1）
"""
from __future__ import annotations
import ast, pathlib, sys

DUCKDB_ALLOWED_PREFIX = ("data/",)
QUERY_FIRST_PARAM_WHITELIST = {"get_tradable_mask"}     # ★ 唯一豁免，见规格 D2
READONLY_LAYERS = ("report/", "agent_tools.py")
DML = ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "DROP ", "ALTER ", "REPLACE ")
FACTOR_FILES = {"price.py", "fundamental.py", "flow.py", "risk.py"}


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def check(root: str = "ashare") -> list[str]:
    rootp = pathlib.Path(root)
    if not rootp.exists():
        return []
    violations: list[str] = []

    for py in sorted(rootp.rglob("*.py")):
        rel = _rel(py, rootp)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as exc:
            violations.append(f"{rel}: 语法错误 {exc}")
            continue
        src = py.read_text(encoding="utf-8")

        # L1 导入方向
        if not rel.startswith(DUCKDB_ALLOWED_PREFIX):
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                if any(m.split(".")[0] == "duckdb" for m in mods):
                    violations.append(
                        f"{rel}:{node.lineno}: L1 直接 import duckdb —— 一切取数经 ashare/data/query.py（D2）")

        # L2 query.py 首参名
        if rel == "data/query.py":
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not node.name.startswith("_") \
                        and node.name not in QUERY_FIRST_PARAM_WHITELIST \
                        and node.args.args:
                    first = node.args.args[0].arg
                    if first != "as_of_date":
                        violations.append(
                            f"{rel}:{node.lineno}: L2 {node.name}() 首参是 '{first}'，必须是 'as_of_date'（D2）")

        # L3 因子签名
        if rel.startswith("factors/") and py.name in FACTOR_FILES:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    names = [a.arg for a in node.args.args]
                    if names[:2] != ["as_of_date", "universe"]:
                        violations.append(
                            f"{rel}:{node.lineno}: L3 因子 {node.name}() 前两个参数必须是 "
                            f"(as_of_date, universe)，实际 {names[:2]}")

        # L4 只读层不得有 DML
        if rel.startswith(READONLY_LAYERS) or rel == "agent_tools.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    upper = node.value.upper().lstrip()
                    if any(upper.startswith(k) for k in DML):
                        violations.append(
                            f"{rel}:{node.lineno}: L4 只读层出现 DML 字符串 —— 违反 D1（LLM 层无写权限）")
            if "connect_write" in src:
                violations.append(f"{rel}: L4 只读层引用了 connect_write —— 违反 D1")

    return violations


def main() -> int:
    v = check(sys.argv[1] if len(sys.argv) > 1 else "ashare")
    for line in v:
        print(f"VIOLATION  {line}")
    print(f"\n{len(v)} 处违规" if v else "\n分层检查通过")
    return 1 if v else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/ashare/test_layering.py -v`
Expected: 6 passed

- [ ] **Step 5: 手动跑一次脚本**

```bash
python3 scripts/check_ashare_layering.py ashare
```

Expected: `分层检查通过`

- [ ] **Step 6: 提交**

```bash
git add scripts/check_ashare_layering.py tests/ashare/test_layering.py
git commit -m "feat(ashare): 分层与签名静态检查（D1/D2 的可执行守卫）"
```

---

### Task 3: Tushare adapter + 持久化令牌桶

**Files:**
- Create: `ashare/data/sources/tushare.py`
- Create: `ashare/data/sources/_ratelimit.py`
- Create: `tests/ashare/test_ratelimit.py`

**Interfaces:**
- Consumes: `data/rate_state.json`（Task 0 产出）
- Produces:
  - `ashare.data.sources.tushare.TushareSource(token: str | None = None)`
  - `.trade_cal(start, end) -> pd.DataFrame`
  - `.stock_basic() -> pd.DataFrame`（含已退市：`list_status='L','D','P'` 三次合并）
  - `.namechange(ts_code=None) -> pd.DataFrame`
  - `.daily(ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame`
  - `.adj_factor(...)`, `.stk_limit(...)`, `.daily_basic(...)`, `.fina_indicator(...)`,
    `.income(...)`, `.balancesheet(...)`, `.cashflow(...)`, `.hk_hold(...)`,
    `.index_daily(...)`, `.cn_m(start_m, end_m)`, `.shibor(start, end)`（宏观逐接口暴露，不做统一 `.macro()`）
  - `ashare.data.sources._ratelimit.TokenBucket(calls_per_min: int, state_path: str)`，方法 `.acquire() -> None`

- [ ] **Step 1: 写令牌桶失败测试**

```python
# tests/ashare/test_ratelimit.py
from __future__ import annotations
import json, time
from ashare.data.sources._ratelimit import TokenBucket

def test_bucket_allows_burst_up_to_capacity(tmp_path):
    b = TokenBucket(calls_per_min=60, state_path=str(tmp_path / "s.json"))
    t0 = time.time()
    for _ in range(5):
        b.acquire()
    assert time.time() - t0 < 0.5, "首批调用不应被限速"

def test_bucket_throttles_beyond_rate(tmp_path):
    b = TokenBucket(calls_per_min=60, state_path=str(tmp_path / "s.json"), capacity=2)
    b.acquire(); b.acquire()
    t0 = time.time()
    b.acquire()                       # 第 3 次必须等 ≈1s（60/min → 1 token/s）
    assert time.time() - t0 >= 0.9

def test_bucket_state_persists_across_instances(tmp_path):
    """跨进程复用：ingest 分批跑，第二个进程不能把配额当全新的。"""
    p = str(tmp_path / "s.json")
    b1 = TokenBucket(calls_per_min=60, state_path=p, capacity=2)
    b1.acquire(); b1.acquire()
    b2 = TokenBucket(calls_per_min=60, state_path=p, capacity=2)
    t0 = time.time()
    b2.acquire()
    assert time.time() - t0 >= 0.9, "新实例必须读到已消耗的配额"
    assert json.loads(open(p).read())["tokens"] >= 0
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/ashare/test_ratelimit.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现令牌桶**

```python
# ashare/data/sources/_ratelimit.py
"""持久化令牌桶。Tushare 限频按账号算，跨进程共享 —— 状态必须落盘，
否则分批 ingest 时第二个进程会把配额当全新的，直接撞限频。"""
from __future__ import annotations
import json, pathlib, time


class TokenBucket:
    def __init__(self, calls_per_min: int, state_path: str, capacity: int | None = None) -> None:
        self.rate = calls_per_min / 60.0          # tokens/sec
        self.capacity = float(capacity if capacity is not None else calls_per_min)
        self.state_path = pathlib.Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> tuple[float, float]:
        try:
            d = json.loads(self.state_path.read_text())
            return float(d["tokens"]), float(d["updated_at"])
        except Exception:                          # 首次运行 / 文件损坏 → 满桶
            return self.capacity, time.time()

    def _save(self, tokens: float, now: float) -> None:
        self.state_path.write_text(json.dumps({"tokens": tokens, "updated_at": now}))

    def acquire(self) -> None:
        """阻塞直到拿到一个 token。"""
        while True:
            tokens, updated = self._load()
            now = time.time()
            tokens = min(self.capacity, tokens + (now - updated) * self.rate)
            if tokens >= 1.0:
                self._save(tokens - 1.0, now)
                return
            self._save(tokens, now)
            time.sleep(max(0.05, (1.0 - tokens) / self.rate))
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/ashare/test_ratelimit.py -v`
Expected: 3 passed

- [ ] **Step 5: 实现 Tushare adapter**

```python
# ashare/data/sources/tushare.py
"""Tushare Pro adapter。职责只有两件：限频 + 把 Tushare 的 YYYYMMDD 字符串日期
规范成 datetime.date。不做任何业务转换 —— 那是 ingest.normalize 的事。"""
from __future__ import annotations
import datetime as _dt
import json, os, pathlib
from typing import Any

import pandas as pd

from ._ratelimit import TokenBucket

try:
    import tushare as _ts
except ImportError:                     # 可选依赖，遵循本仓库既有模式
    _ts = None

_DATE_COLS = ("trade_date", "ann_date", "end_date", "list_date", "delist_date",
              "start_date", "f_ann_date", "cal_date")


def _to_date(df: pd.DataFrame) -> pd.DataFrame:
    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce").dt.date
    return df


def _fmt(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


class TushareSource:
    def __init__(self, token: str | None = None, state_path: str = "data/rate_state.json") -> None:
        if _ts is None:
            raise ImportError("需要 tushare：pip install tushare")
        tok = token or os.environ.get("TUSHARE_TOKEN")
        if not tok:
            raise ValueError("未提供 TUSHARE_TOKEN")
        self._pro = _ts.pro_api(tok)
        cpm = 120
        try:
            cpm = int(json.loads(pathlib.Path(state_path).read_text()).get("calls_per_min", 120))
        except Exception:
            pass
        self._bucket = TokenBucket(cpm, state_path)

    def _call(self, api: str, **kw) -> pd.DataFrame:
        self._bucket.acquire()
        df = getattr(self._pro, api)(**{k: v for k, v in kw.items() if v is not None})
        return _to_date(df if df is not None else pd.DataFrame())

    # ── 基础 ──
    def trade_cal(self, start, end, exchange: str = "SSE") -> pd.DataFrame:
        return self._call("trade_cal", exchange=exchange,
                          start_date=_fmt(start), end_date=_fmt(end))

    def stock_basic(self) -> pd.DataFrame:
        """★ 必须三种 list_status 都拉：L 在市 / D 已退市 / P 暂停上市。
        只拉 L 就是幸存者偏差的源头（D5）。"""
        fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs"
        parts = [self._call("stock_basic", exchange="", list_status=s, fields=fields)
                 for s in ("L", "D", "P")]
        return pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")

    def namechange(self, ts_code: str | None = None) -> pd.DataFrame:
        return self._call("namechange", ts_code=ts_code,
                          fields="ts_code,name,start_date,end_date,change_reason")

    # ── 行情 ──
    def daily(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("daily", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def adj_factor(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("adj_factor", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def stk_limit(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("stk_limit", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def daily_basic(self, ts_code=None, trade_date=None, start=None, end=None) -> pd.DataFrame:
        return self._call("daily_basic", ts_code=ts_code, trade_date=_fmt(trade_date),
                          start_date=_fmt(start), end_date=_fmt(end))

    def index_daily(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("index_daily", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    # ── 财报（PIT 的原料）──
    def fina_indicator(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("fina_indicator", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def income(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("income", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def balancesheet(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("balancesheet", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    def cashflow(self, ts_code: str, start=None, end=None) -> pd.DataFrame:
        return self._call("cashflow", ts_code=ts_code,
                          start_date=_fmt(start), end_date=_fmt(end))

    # ── 资金流 / 宏观 ──
    def hk_hold(self, trade_date) -> pd.DataFrame:
        return self._call("hk_hold", trade_date=_fmt(trade_date))

    def cn_m(self, start_m: str, end_m: str) -> pd.DataFrame:
        return self._call("cn_m", start_m=start_m, end_m=end_m)

    def shibor(self, start=None, end=None) -> pd.DataFrame:
        return self._call("shibor", start_date=_fmt(start), end_date=_fmt(end))
```

- [ ] **Step 6: 跑分层检查 + 全部测试**

```bash
python3 scripts/check_ashare_layering.py ashare && python3 -m pytest tests/ashare/ -q
```

Expected: `分层检查通过`，测试全绿

- [ ] **Step 7: 提交**

```bash
git add ashare/data/sources/ tests/ashare/test_ratelimit.py
git commit -m "feat(ashare): Tushare adapter + 跨进程持久化令牌桶"
```

---

### Task 4: 涨跌停规则兜底（B2 / D6）

**Files:**
- Create: `ashare/data/limits.py`
- Create: `tests/ashare/test_limits.py`

**Interfaces:**
- Consumes: `ashare.data.sources.tushare` 无（纯规则函数）
- Produces: `ashare.data.limits.compute_limits(ts_code, trade_date, pre_close, list_date, status) -> tuple[float | None, float | None, str]`
  返回 `(limit_up, limit_down, source)`，`source ∈ {'rule', 'unknown'}`

- [ ] **Step 1: 写失败测试**

```python
# tests/ashare/test_limits.py
from __future__ import annotations
import datetime as dt
import pytest
from ashare.data.limits import compute_limits

D = dt.date

def test_main_board_10pct():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "NORMAL")
    assert (round(up,2), round(dn,2), src) == (110.0, 90.0, "rule")

def test_st_5pct():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "ST")
    assert (round(up,2), round(dn,2)) == (105.0, 95.0)

def test_chinext_was_10pct_before_20200824():
    up, dn, _ = compute_limits("300750.SZ", D(2020,8,21), 100.0, D(2018,6,11), "NORMAL")
    assert round(up,2) == 110.0

def test_chinext_became_20pct_on_20200824():
    up, dn, _ = compute_limits("300750.SZ", D(2020,8,24), 100.0, D(2018,6,11), "NORMAL")
    assert round(up,2) == 120.0

def test_star_market_20pct():
    up, dn, _ = compute_limits("688981.SH", D(2021,5,10), 100.0, D(2020,7,16), "NORMAL")
    assert round(up,2) == 120.0

def test_bse_30pct():
    up, dn, _ = compute_limits("830799.BJ", D(2024,1,10), 100.0, D(2021,11,15), "NORMAL")
    assert round(up,2) == 130.0

def test_new_listing_returns_unknown():
    """新股上市首日无涨跌幅限制（主板）/ 前 5 日无限制（科创创业）——
    规则算不出，必须返回 unknown 让上层判为不可交易，绝不能假装 10%。"""
    up, dn, src = compute_limits("301000.SZ", D(2024,1,10), 100.0, D(2024,1,10), "NORMAL")
    assert (up, dn, src) == (None, None, "unknown")

def test_delist_period_returns_unknown():
    up, dn, src = compute_limits("600519.SH", D(2024,1,10), 100.0, D(2001,8,27), "DELIST_PERIOD")
    assert src == "unknown"
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/ashare/test_limits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ashare.data.limits'`

- [ ] **Step 3: 实现**

```python
# ashare/data/limits.py
"""涨跌停价规则兜底（架构师 B2）。

Tushare stk_limit 有积分门槛，拿不到时用规则算。D6 直接依赖这个函数。

★ 核心原则：算不出的一律返回 (None, None, 'unknown')，由上层判为【不可交易】。
  宁可少一次成交机会，绝不能假设可交易 —— 后者会在回测里凭空生成
  现实中不存在的成交，是单向乐观偏差。
"""
from __future__ import annotations
import datetime as _dt

CHINEXT_20PCT_FROM = _dt.date(2020, 8, 24)     # 创业板涨跌幅由 10% 改 20%
STAR_OPEN_DATE = _dt.date(2019, 7, 22)         # 科创板开板
NEW_LISTING_GRACE_DAYS = 10                    # 上市初期规则复杂，一律 unknown


def _board(ts_code: str) -> str:
    code, _, ex = ts_code.partition(".")
    if ex == "BJ" or code.startswith(("43", "83", "87", "88")):
        return "BSE"
    if code.startswith("688"):
        return "STAR"
    if code.startswith("300"):
        return "CHINEXT"
    return "MAIN"


def compute_limits(ts_code: str, trade_date: _dt.date, pre_close: float | None,
                   list_date: _dt.date | None, status: str) -> tuple[float | None, float | None, str]:
    """返回 (limit_up, limit_down, source)。source ∈ {'rule', 'unknown'}。"""
    if pre_close is None or pre_close <= 0:
        return None, None, "unknown"

    # 退市整理期：涨跌幅规则历经多次变更，且流动性枯竭，一律不交易
    if status == "DELIST_PERIOD":
        return None, None, "unknown"

    # 上市初期：主板首日无涨跌幅、科创创业前 5 日无限制、北交所首日无限制
    if list_date is not None and (trade_date - list_date).days < NEW_LISTING_GRACE_DAYS:
        return None, None, "unknown"

    board = _board(ts_code)
    if board == "BSE":
        pct = 0.30
    elif board == "STAR":
        if trade_date < STAR_OPEN_DATE:
            return None, None, "unknown"
        pct = 0.20
    elif board == "CHINEXT":
        pct = 0.20 if trade_date >= CHINEXT_20PCT_FROM else 0.10
    else:
        pct = 0.10

    # ST 一律 5%（各板块统一），且优先级高于板块规则
    if status in ("ST", "*ST"):
        pct = 0.05

    # A 股涨跌停价按四舍五入到 0.01 元
    return round(pre_close * (1 + pct), 2), round(pre_close * (1 - pct), 2), "rule"
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/ashare/test_limits.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
git add ashare/data/limits.py tests/ashare/test_limits.py
git commit -m "feat(ashare): 涨跌停规则兜底，算不出即判不可交易（B2/D6）"
```

---

### Task 5: ingest — 日历 + 股票基础信息 + ST 历史反推（D5/B6）

**Files:**
- Create: `ashare/data/ingest.py`
- Create: `tests/ashare/test_ingest_status.py`

**Interfaces:**
- Consumes: `TushareSource`、`_db.connect_write`、`_db.init_schema`
- Produces:
  - `ashare.data.ingest.derive_stock_status(namechange_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame`
    （列：`ts_code, start_date, end_date, status`）
  - `ashare.data.ingest.ingest_calendar(conn, src, start, end) -> int`
  - `ashare.data.ingest.ingest_stock_basic(conn, src) -> int`
  - `ashare.data.ingest.ingest_stock_status(conn, src) -> int`
  - `ashare.data.ingest.job_state(conn, job_id) -> str | None` / `set_job(conn, job_id, table, partition, state, **kw) -> None`

- [ ] **Step 1: 写 ST 反推的失败测试**

```python
# tests/ashare/test_ingest_status.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from ashare.data.ingest import derive_stock_status

D = dt.date

def _nc(rows):
    return pd.DataFrame(rows, columns=["ts_code", "name", "start_date", "end_date"])

def _basic(rows):
    return pd.DataFrame(rows, columns=["ts_code", "name", "list_date", "delist_date"])

def test_derives_st_period_from_name():
    nc = _nc([("000001.SZ", "深发展A",  D(2000,1,1), D(2011,12,31)),
              ("000001.SZ", "ST深发展", D(2012,1,1), D(2013,6,30)),
              ("000001.SZ", "平安银行",  D(2013,7,1), None)])
    out = derive_stock_status(nc, _basic([("000001.SZ", "平安银行", D(2000,1,1), None)]))
    st = out[out.status == "ST"]
    assert len(st) == 1
    assert st.iloc[0].start_date == D(2012,1,1) and st.iloc[0].end_date == D(2013,6,30)

def test_star_st_is_distinct_from_st():
    nc = _nc([("000002.SZ", "*ST某某", D(2015,5,1), D(2016,4,30))])
    out = derive_stock_status(nc, _basic([("000002.SZ", "某某", D(2000,1,1), None)]))
    assert set(out.status) == {"*ST"}

def test_second_st_period_is_separate_row():
    """二次戴帽：两段 ST 之间隔着 NORMAL，不能被合并成一段。"""
    nc = _nc([("000003.SZ", "ST甲", D(2012,1,1), D(2013,1,1)),
              ("000003.SZ", "甲",   D(2013,1,2), D(2015,1,1)),
              ("000003.SZ", "ST甲", D(2015,1,2), None)])
    out = derive_stock_status(nc, _basic([("000003.SZ", "ST甲", D(2000,1,1), None)]))
    assert len(out[out.status == "ST"]) == 2

def test_s_prefix_not_treated_as_st():
    """'S' 前缀是未股改，不是 ST。误判会错杀一批 2006-2007 的股票。"""
    nc = _nc([("000004.SZ", "S某某", D(2006,1,1), D(2007,1,1))])
    out = derive_stock_status(nc, _basic([("000004.SZ", "某某", D(2000,1,1), None)]))
    assert "ST" not in set(out.status) and "*ST" not in set(out.status)

def test_delist_period_from_name_suffix():
    nc = _nc([("000005.SZ", "某某退", D(2020,1,1), D(2020,2,1))])
    out = derive_stock_status(nc, _basic([("000005.SZ", "某某退", D(2000,1,1), D(2020,2,2))]))
    assert "DELIST_PERIOD" in set(out.status)

def test_stock_with_no_namechange_gets_normal_row():
    out = derive_stock_status(_nc([]), _basic([("600519.SH", "贵州茅台", D(2001,8,27), None)]))
    assert len(out) == 1 and out.iloc[0].status == "NORMAL"
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/ashare/test_ingest_status.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ashare.data.ingest'`

- [ ] **Step 3: 实现 ingest 的日历/基础信息/ST 反推部分**

```python
# ashare/data/ingest.py
"""market 库的唯一写者。

状态机（ingest_log）：PENDING → RUNNING → DONE | RETRY | SUSPECT | FAILED
  RETRY   网络/限频类错误，可重试
  SUSPECT 拉到了数据但校验存疑（行数异常），保留数据待人工确认
  FAILED  schema 断言失败（字段缺失/改名），不重试 —— 重试只会一直错
"""
from __future__ import annotations
import datetime as _dt
import re
from typing import Any

import pandas as pd

from . import _db

# ST 判定：'ST'/'*ST' 前缀。★ 'S' 单独前缀是未股改，不是 ST。
_RE_STAR_ST = re.compile(r"^\*ST")
_RE_ST = re.compile(r"^S?ST")          # 'SST' 也是 ST（未股改 + ST）
_RE_DELIST = re.compile(r"退$")


def _classify(name: str) -> str:
    n = (name or "").replace(" ", "")
    if _RE_DELIST.search(n):
        return "DELIST_PERIOD"
    if _RE_STAR_ST.match(n):
        return "*ST"
    if _RE_ST.match(n):
        return "ST"
    return "NORMAL"


def derive_stock_status(namechange_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """由历史名称变更反推 ST 状态区间（架构师 B6 —— Tushare 无此接口）。

    边界（必须遵守，否则 D5 的股票池就是错的）：
      - 用变更【生效日】start_date，不是公告日
      - 'S' 前缀 = 未股改，不是 ST
      - 名称含 '退' = 退市整理期，单独归类
      - 无 namechange 记录的股票，补一条覆盖全生命周期的 NORMAL
    """
    rows: list[dict[str, Any]] = []

    if len(namechange_df):
        nc = namechange_df.sort_values(["ts_code", "start_date"])
        for ts_code, grp in nc.groupby("ts_code", sort=True):
            for _, r in grp.iterrows():
                rows.append({"ts_code": ts_code,
                             "start_date": r["start_date"],
                             "end_date": r.get("end_date"),
                             "status": _classify(r["name"])})

    covered = {r["ts_code"] for r in rows}
    for _, b in basic_df.iterrows():
        if b["ts_code"] not in covered:
            rows.append({"ts_code": b["ts_code"],
                         "start_date": b["list_date"],
                         "end_date": b.get("delist_date"),
                         "status": _classify(b.get("name", ""))})

    out = pd.DataFrame(rows, columns=["ts_code", "start_date", "end_date", "status"])
    return out.sort_values(["ts_code", "start_date"]).reset_index(drop=True)


# ══════════════ ingest_log 状态机 ══════════════
def set_job(conn, job_id: str, table: str, partition: str, state: str,
            *, rows: int = 0, error: str = "") -> None:
    conn.execute(
        """INSERT INTO ingest_log (job_id, table_name, partition, state, attempts,
                                   rows_written, last_error, started_at, finished_at)
           VALUES (?, ?, ?, ?, 1, ?, ?, current_timestamp,
                   CASE WHEN ? IN ('DONE','FAILED') THEN current_timestamp END)
           ON CONFLICT (job_id) DO UPDATE SET
             state = excluded.state,
             attempts = ingest_log.attempts + 1,
             rows_written = excluded.rows_written,
             last_error = excluded.last_error,
             finished_at = excluded.finished_at""",
        [job_id, table, partition, state, rows, error, state])


def job_state(conn, job_id: str) -> str | None:
    r = conn.execute("SELECT state FROM ingest_log WHERE job_id = ?", [job_id]).fetchone()
    return r[0] if r else None


def _upsert(conn, table: str, df: pd.DataFrame, pk: list[str]) -> int:
    """按主键覆盖写入。DuckDB 的 INSERT OR REPLACE 走主键冲突。"""
    if df is None or df.empty:
        return 0
    cols = [c for c in df.columns if c != "_ingested_at"]
    conn.register("_stage", df[cols])
    conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) SELECT {','.join(cols)} FROM _stage")
    conn.unregister("_stage")
    return len(df)


# ══════════════ 各表 ingest ══════════════
def ingest_calendar(conn, src, start, end) -> int:
    df = src.trade_cal(start, end)
    df = df.rename(columns={"cal_date": "trade_date", "pretrade_date": "pre_trade_date"})
    df["is_open"] = df["is_open"].astype(int).astype(bool)
    n = _upsert(conn, "calendar", df[["trade_date", "is_open", "pre_trade_date"]], ["trade_date"])
    set_job(conn, "calendar:all", "calendar", "all", "DONE", rows=n)
    return n


def ingest_stock_basic(conn, src) -> int:
    df = src.stock_basic()
    for c in ("sw_l1", "sw_l2", "sw_l3"):
        if c not in df.columns:
            df[c] = None
    cols = ["ts_code", "symbol", "name", "sw_l1", "sw_l2", "sw_l3",
            "market", "list_date", "delist_date", "is_hs"]
    n = _upsert(conn, "stock_basic", df[cols], ["ts_code"])
    set_job(conn, "stock_basic:all", "stock_basic", "all", "DONE", rows=n)
    return n


def ingest_stock_status(conn, src) -> int:
    basic = conn.execute(
        "SELECT ts_code, name, list_date, delist_date FROM stock_basic").fetchdf()
    nc = src.namechange(ts_code=None)
    status = derive_stock_status(nc, basic)
    n = _upsert(conn, "stock_status", status, ["ts_code", "start_date"])
    set_job(conn, "stock_status:all", "stock_status", "all", "DONE", rows=n)
    return n
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/ashare/test_ingest_status.py -v`
Expected: 6 passed

- [ ] **Step 5: 跑分层检查 + 全量测试**

```bash
python3 scripts/check_ashare_layering.py ashare && python3 -m pytest tests/ashare/ -q
```

Expected: 分层检查通过，全部测试绿

- [ ] **Step 6: 提交**

```bash
git add ashare/data/ingest.py tests/ashare/test_ingest_status.py
git commit -m "feat(ashare): ingest 日历/基础信息 + ST 历史由 namechange 反推（D5/B6）"
```

---

### Task 6: ingest 日线 + 停牌占位行补齐（D9 / B3）★ 本计划最关键的一个任务

**Files:**
- Modify: `ashare/data/ingest.py`（追加 `normalize_daily_bar` 与 `ingest_daily_bar`）
- Create: `tests/ashare/test_ingest_daily.py`

**Interfaces:**
- Consumes: Task 5 的 `_upsert`/`set_job`；Task 4 的 `compute_limits`
- Produces:
  - `ashare.data.ingest.normalize_daily_bar(daily, adj, limit, calendar_dates, basic_row, status_rows) -> pd.DataFrame`
  - `ashare.data.ingest.ingest_daily_bar(conn, src, ts_code, start, end) -> int`

- [ ] **Step 1: 写失败测试 —— 占位行是核心断言**

```python
# tests/ashare/test_ingest_daily.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from ashare.data.ingest import normalize_daily_bar

D = dt.date
CAL = [D(2024,1,2), D(2024,1,3), D(2024,1,4), D(2024,1,5), D(2024,1,8)]
BASIC = {"ts_code": "600519.SH", "list_date": D(2001,8,27), "delist_date": None}
STATUS = [{"start_date": D(2001,8,27), "end_date": None, "status": "NORMAL"}]

def _daily(dates_prices):
    return pd.DataFrame(
        [{"ts_code": "600519.SH", "trade_date": d, "open": p, "high": p, "low": p,
          "close": p, "pre_close": p, "vol": 100.0, "amount": 1000.0} for d, p in dates_prices])

def _adj(dates):
    return pd.DataFrame([{"ts_code": "600519.SH", "trade_date": d, "adj_factor": 1.0} for d in dates])

def test_suspended_days_get_placeholder_rows():
    """★ D9 核心断言：Tushare 停牌日不返回行，normalize 必须补齐。
    不补则 rolling(20) 拿到的是 20 条记录而非 20 个交易日，因子静默污染。"""
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,5), 105.0), (D(2024,1,8), 106.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,5), D(2024,1,8)]),
                              None, CAL, BASIC, STATUS)
    assert len(out) == len(CAL), f"必须补齐到 {len(CAL)} 个交易日，实际 {len(out)}"
    sus = out[out.is_suspended]
    assert set(sus.trade_date) == {D(2024,1,3), D(2024,1,4)}

def test_placeholder_rows_carry_prev_close_and_zero_volume():
    daily = _daily([(D(2024,1,2), 100.0), (D(2024,1,5), 105.0), (D(2024,1,8), 106.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2), D(2024,1,5), D(2024,1,8)]),
                              None, CAL, BASIC, STATUS).set_index("trade_date")
    r = out.loc[D(2024,1,3)]
    assert r.vol == 0 and r.amount == 0
    assert r.open == r.high == r.low == r.close == 100.0, "占位行 OHLC 全取前收"
    assert r.adj_factor == 1.0, "占位行沿用前一日复权因子"

def test_no_rows_outside_listing_window():
    """未上市/已退市区间不得补行 —— 补了就是幸存者偏差的反面（凭空造出行情）。"""
    basic = {"ts_code": "600519.SH", "list_date": D(2024,1,4), "delist_date": None}
    out = normalize_daily_bar(_daily([(D(2024,1,5), 105.0)]), _adj([D(2024,1,5)]),
                              None, CAL, basic, STATUS)
    assert out.trade_date.min() >= D(2024,1,4)
    assert len(out) == 3                       # 1/4 占位 + 1/5 实际 + 1/8 占位（1/2、1/3 在上市前，不得出现）
    assert list(out.is_suspended) == [True, False, True]
    assert pd.isna(out.iloc[0].close), "上市首日即停牌且无前收 → OHLC 为 NaN，不得编造价格"

def test_limit_falls_back_to_rule_when_api_missing():
    daily = _daily([(D(2024,1,2), 100.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), None, [D(2024,1,2)], BASIC, STATUS)
    r = out.iloc[0]
    assert r.limit_source == "rule" and round(r.limit_up, 2) == 110.0

def test_limit_prefers_api_over_rule():
    daily = _daily([(D(2024,1,2), 100.0)])
    lim = pd.DataFrame([{"ts_code": "600519.SH", "trade_date": D(2024,1,2),
                         "up_limit": 111.11, "down_limit": 88.88}])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), lim, [D(2024,1,2)], BASIC, STATUS)
    r = out.iloc[0]
    assert r.limit_source == "api" and r.limit_up == 111.11

def test_row_count_equals_trading_days_in_listing_window():
    """P1 验收断言的单元版：行数 == 在市区间交易日数，误差为 0（不是 0.1%）。"""
    daily = _daily([(D(2024,1,2), 100.0)])
    out = normalize_daily_bar(daily, _adj([D(2024,1,2)]), None, CAL, BASIC, STATUS)
    assert len(out) == 5
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/ashare/test_ingest_daily.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_daily_bar'`

- [ ] **Step 3: 实现 normalize + ingest**

在 `ashare/data/ingest.py` 末尾追加：

```python
from .limits import compute_limits


def _status_at(status_rows: list[dict], d: _dt.date) -> str:
    for r in status_rows:
        start = r["start_date"]
        end = r.get("end_date")
        if start is not None and start <= d and (end is None or d <= end):
            return r["status"]
    return "NORMAL"


def normalize_daily_bar(daily: pd.DataFrame,
                        adj: pd.DataFrame,
                        limit: pd.DataFrame | None,
                        calendar_dates: list[_dt.date],
                        basic_row: dict,
                        status_rows: list[dict]) -> pd.DataFrame:
    """把 Tushare 三张表合成 daily_bar，并【按交易日历补齐停牌占位行】（D9 / 架构师 B3）。

    ★ 这是全套设计里最隐蔽的一个坑：Tushare `daily` 在停牌日不返回该股的行。
      不补行的话，get_bars(lookback=20) 拿到的是「最近 20 条记录」而不是
      「最近 20 个交易日」—— 一只停牌 5 天的股票，它的 reversal_20 实际覆盖
      25 个交易日。横截面因子被静默污染，且完全不报错。
    """
    ts_code = basic_row["ts_code"]
    list_date = basic_row.get("list_date")
    delist_date = basic_row.get("delist_date")

    # 1. 只保留在市区间内的交易日
    dates = [d for d in sorted(calendar_dates)
             if (list_date is None or d >= list_date)
             and (delist_date is None or d <= delist_date)]
    if not dates:
        return pd.DataFrame(columns=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "adj_factor", "limit_up", "limit_down",
            "limit_source", "is_suspended"])

    # 2. 以完整交易日历为骨架左连接
    frame = pd.DataFrame({"trade_date": dates})
    d = daily.copy() if daily is not None and len(daily) else pd.DataFrame(
        columns=["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"])
    frame = frame.merge(d.drop(columns=["ts_code"], errors="ignore"), on="trade_date", how="left")

    a = adj.copy() if adj is not None and len(adj) else pd.DataFrame(
        columns=["trade_date", "adj_factor"])
    frame = frame.merge(a.drop(columns=["ts_code"], errors="ignore"), on="trade_date", how="left")

    frame = frame.sort_values("trade_date").reset_index(drop=True)

    # 3. 标记停牌（daily 无行 = 停牌）并填占位值
    frame["is_suspended"] = frame["close"].isna()
    frame["adj_factor"] = frame["adj_factor"].ffill().bfill()

    prev_close = None
    for i in frame.index:
        if frame.at[i, "is_suspended"]:
            fill = prev_close
            for c in ("open", "high", "low", "close", "pre_close"):
                frame.at[i, c] = fill
            frame.at[i, "vol"] = 0.0
            frame.at[i, "amount"] = 0.0
        prev_close = frame.at[i, "close"]

    # 4. 涨跌停：API 优先，缺失走规则兜底（B2）
    lim_map: dict[_dt.date, tuple[float, float]] = {}
    if limit is not None and len(limit):
        for _, r in limit.iterrows():
            lim_map[r["trade_date"]] = (r.get("up_limit"), r.get("down_limit"))

    ups, downs, srcs = [], [], []
    for _, r in frame.iterrows():
        td = r["trade_date"]
        if td in lim_map and lim_map[td][0] is not None:
            ups.append(lim_map[td][0]); downs.append(lim_map[td][1]); srcs.append("api")
            continue
        u, dn, src = compute_limits(ts_code, td, r.get("pre_close"),
                                    list_date, _status_at(status_rows, td))
        ups.append(u); downs.append(dn); srcs.append(src)

    frame["ts_code"] = ts_code
    frame["limit_up"], frame["limit_down"], frame["limit_source"] = ups, downs, srcs
    return frame[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                  "vol", "amount", "adj_factor", "limit_up", "limit_down",
                  "limit_source", "is_suspended"]]


def ingest_daily_bar(conn, src, ts_code: str, start, end) -> int:
    job = f"daily_bar:{ts_code}:{start}"
    if job_state(conn, job) == "DONE":
        return 0
    set_job(conn, job, "daily_bar", ts_code, "RUNNING")
    try:
        daily = src.daily(ts_code=ts_code, start=start, end=end)
        adj = src.adj_factor(ts_code=ts_code, start=start, end=end)
        try:
            limit = src.stk_limit(ts_code=ts_code, start=start, end=end)
        except Exception:
            limit = None                     # 无权限 → 走规则兜底，不是错误

        cal = [r[0] for r in conn.execute(
            "SELECT trade_date FROM calendar WHERE is_open AND trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date", [start, end]).fetchall()]
        basic = conn.execute(
            "SELECT ts_code, list_date, delist_date FROM stock_basic WHERE ts_code = ?",
            [ts_code]).fetchdf().to_dict("records")[0]
        status = conn.execute(
            "SELECT start_date, end_date, status FROM stock_status WHERE ts_code = ? "
            "ORDER BY start_date", [ts_code]).fetchdf().to_dict("records")

        out = normalize_daily_bar(daily, adj, limit, cal, basic, status)
        n = _upsert(conn, "daily_bar", out, ["ts_code", "trade_date"])
        set_job(conn, job, "daily_bar", ts_code, "DONE", rows=n)
        return n
    except Exception as exc:                 # noqa: BLE001
        set_job(conn, job, "daily_bar", ts_code, "RETRY", error=str(exc)[:500])
        raise
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/ashare/test_ingest_daily.py -v`
Expected: 6 passed

- [ ] **Step 5: 全量测试 + 分层检查**

```bash
python3 scripts/check_ashare_layering.py ashare && python3 -m pytest tests/ashare/ -q
```

Expected: 全绿

- [ ] **Step 6: 提交**

```bash
git add ashare/data/ingest.py tests/ashare/test_ingest_daily.py
git commit -m "feat(ashare): 日线入库 + 停牌日按交易日历补占位行（D9/B3）"
```

---

### Task 7–15（Task 6 通过后展开；写到接口 + 决策 + 验收断言粒度）

> 前 6 个任务的经验：计划里逐字写实现，评审照样能找出规则/类型 bug，等于双倍工作。
> 从 Task 7 起，计划只锁三样东西：**接口签名、关键决策、验收断言**。实现按 TDD 现写现测。
> 所有 query.py 签名以 `docs/architecture/ashare-platform-architecture.md` §4.1 为准，此处只列偏离与补充。

---

### Task 7: ingest 财报 PIT + daily_basic + index_daily + 行业成分历史

**Files:** Modify `ashare/data/ingest.py`, `ashare/data/schema.sql`(v2)；Create `tests/ashare/test_ingest_financial.py`

**决策**
- 财报 PIT 三张原始表 `income` / `balancesheet` / `cashflow` + `fina_indicator` 合并进 `financial_pit`，
  以 `(ts_code, ann_date, end_date, report_type, update_flag)` 为键；`fina_indicator` 无 `report_type`/`update_flag`
  时按 `('1', 0)` 归并到同一 (ts_code, ann_date, end_date)。**只要 `ann_date` 为空的行一律丢弃并计数**（D3：无公告日 = 无法 PIT）。
- `f_ann_date`（实际公告日）存在时优先于 `ann_date`——Tushare 的 `ann_date` 有时是"预约披露日"。
- 新表 `industry_member(ts_code, sw_l1, sw_l2, sw_l3, in_date, out_date)`：来源 `index_member`(申万成分历史)。
  无权限时降级：用 `stock_basic.industry`（Tushare 自有分类）写成 `in_date=list_date, out_date=NULL`，
  并在 `_meta` 写 `industry_source='tushare_static'`。**降级必须显式记录**，不得静默。
- `SCHEMA_VERSION` → 2；`init_schema` 读到旧版本且缺表 → 直接 `CREATE TABLE IF NOT EXISTS`（加表向前兼容）；
  读到**更高**版本 → raise（防旧代码写新库）。

**接口**
- `ingest_financial(conn, src, ts_code, start, end) -> int`
- `ingest_daily_basic(conn, src, trade_date) -> int`（按日全市场，一次调用一天）
- `ingest_index_daily(conn, src, index_code, start, end) -> int`
- `ingest_industry_member(conn, src) -> int`
- `merge_financial_frames(income, balance, cashflow, fina) -> pd.DataFrame`（纯函数，可测）

**验收断言**
- 同一 `(ts_code, end_date)` 两个不同 `ann_date` 的行共存；`update_flag=1` 独立成行不覆盖原始行
- `ann_date` 为空的行不入库，返回值中 `dropped_no_ann_date` 计数可见
- `f_ann_date` 存在时 `financial_pit.ann_date == f_ann_date`
- daily_basic 单日全市场入库后 `count == len(src 返回)`，重跑幂等
- 降级路径：`index_member` 抛权限错误 → `industry_member` 由 `stock_basic.industry` 生成且 `_meta.industry_source='tushare_static'`

---

### Task 8: ingest 宏观 PIT + 北向持股

**Files:** Modify `ashare/data/ingest.py`, `ashare/data/sources/tushare.py`（补 cn_cpi/cn_ppi/cn_pmi/sf_month/hk_hold）；Create `tests/ashare/test_ingest_macro.py`

**决策（D4 的具体化）**
- 指标与来源：`m1_yoy`/`m2_yoy` ← `cn_m`；`cpi_yoy` ← `cn_cpi`；`ppi_yoy` ← `cn_ppi`；`pmi_mfg` ← `cn_pmi`；
  `tsf_stock_yoy` ← `sf_month`；`shibor_3m` ← `shibor`；`cn10y` ← akshare `bond_zh_us_rate`（Tushare 无稳定接口，adapter 层可插拔）。
- **历史 `publish_date` 只能按规则回填**，且规则取【保守晚值】——宁可晚几天可见，绝不提前：
  M1/M2/社融 = 次月 15 日；CPI/PPI = 次月 10 日；PMI = 次月 1 日；shibor/cn10y = 当日。
  `publish_date_source='rule'`。每日增量 ingest 时若拉到新 period，`publish_date=今天, source='observed'`。
- `cn_m.month` 是 `YYYYMM`，在 normalize 里单独转 `period = 该月最后一天`。
- `hk_hold` 按 `trade_date` 全市场拉取，`hk_hold_ratio = ratio` 列；2016-12-05 前无数据，**不补 0**。

**接口**
- `ingest_macro(conn, src, indicator, start, end) -> int`
- `ingest_hk_hold(conn, src, trade_date) -> int`
- `rule_publish_date(indicator, period) -> date`（纯函数，可测）

**验收断言**
- `rule_publish_date('m2_yoy', 2024-07-31) == 2024-08-15`；`('cpi_yoy', 2024-07-31) == 2024-08-10`；`('pmi_mfg', 2024-07-31) == 2024-08-01`
- 所有 `macro_indicator` 行 `publish_date >= period`（不可能先公布再发生）
- `publish_date_source` 非空
- 同 `(indicator, period)` 先 rule 后 observed 两行共存（PIT：不覆盖）

---

### Task 9: query.py 骨架 — 连接 / snapshot_id / 日历 / preload

**Files:** Create `ashare/data/query.py`；Create `tests/ashare/test_query_calendar.py`

**决策**
- 模块级单例只读连接（`_conn`），`open_db()` 幂等；文件 realpath 变化（影子替换）→ 自动重连。
- `snapshot_id() = sha256(realpath basename + schema_version + Σ 各表 max(_ingested_at) + count)[:16]`。
- `get_trade_dates(as_of_date, *, start=None, freq='D'|'W'|'M')`：`W` = 每周最后一个交易日（`weekly_dates` 唯一实现点，禁止别处自算）。
- `next_trade_date` 日历未覆盖 → 返回 `None`（不抛）。
- `preload(start, end, tables=(...))` 把区间物化进进程内 dict-of-DataFrame；`get_*` 优先命中缓存切片。本任务只做骨架，Task 11 接上。
- 异常层级：`QueryError` ← `AsOfDateError` / `DataGapError` / `UnknownFieldError`。

**验收断言**
- 只读连接上 `_conn.execute("INSERT ...")` 抛 `duckdb.InvalidInputException`（D1）
- `snapshot_id()` 在 ingest 一行后变化，在无写入时稳定
- `get_trade_dates('2024-01-31', start='2024-01-01', freq='W')` == 每周最后交易日（用真实 2024 年 1 月日历硬编码断言）
- 分层检查：`query.py` 所有公开函数首参 `as_of_date`（L2 自动覆盖）

---

### Task 10: query.get_universe / explain_universe / get_stock_basic / get_industry

**Files:** Modify `ashare/data/query.py`；Create `tests/ashare/test_universe.py`

**决策（架构文档 §4.1 定死的顺序）**
1) 退市 2) 上市满 `min_list_days`(250 自然日) 3) `stock_status` 在 `as_of_date` 生效状态 ∉ {ST, *ST, DELIST_PERIOD}
4) `daily_bar` 当日存在且 `is_suspended=FALSE` 5) markets 过滤 6) **在 1–5 剩余池内**按 20 日均成交额剔后 20%。
- `get_industry`：成分 < 5 的行业归 `__OTHER__`（中性化秩亏保护，B7）；按 `industry_member.in_date <= as_of < out_date` 取 PIT 行业。
- `explain_universe` 返回逐步布尔列 + `drop_reason`，验收断言直接跑它。

**验收断言（规格 §11）**
- `get_universe('2015-06-12')` 不含 `list_date > 2014-10-05` 或 `delist_date <= 2015-06-12` 的股票
- ST 区间内的股票不在池内；区间外在池内（同一只股票、两个日期）
- 流动性分位在硬性剔除之后算：构造 3 只股票，一只已退市但成交额最低，断言退市股不影响其余两只的分位
- `explain_universe` 的 `included` 列与 `get_universe` 结果一致

---

### Task 11: query 行情 — get_bars / get_price_panel / get_daily_basic / get_index_bars / get_tradable_mask

**Files:** Modify `ashare/data/query.py`；Create `tests/ashare/test_query_bars.py`, `tests/ashare/test_tradable_mask.py`

**决策**
- `get_bars` 默认 `adjust='hfq'`：价格列 × `adj_factor`；**不返回 `limit_up/limit_down`**；`lookback` 按**交易日历条数**计
  （不是记录数——D9 已保证两者相等，此处再断言一次）。停牌日 OHLC 输出为 **NaN**（架构 §4.1：由因子自己决定是否填充），`is_suspended` 列恒返回。
- `get_price_panel` 宽表 index=trade_date, columns=ts_code；停牌 NaN 不 ffill。
- `get_tradable_mask(exec_date, ts_codes)`：**唯一首参非 as_of_date 的函数**；内部用原始价判定，只输出 `can_buy/can_sell/reason/open_hfq/close_hfq/amount/amplitude`；
  `limit_up IS NULL`（`limit_source='unknown'`）→ 两侧皆 False，`reason='limit_unknown'`；停牌 → `suspended`；
  `open==limit_up AND high==low` → `can_buy=False, reason='limit_up_seal'`；对称跌停；`delist_date <= exec_date` → `can_buy=False, can_sell=True, reason='delisted'`。
- 空结果返回带正确列名的空 DataFrame（Q3）；`as_of_date` 越界抛 `AsOfDateError`（Q2）。

**验收断言**
- `get_bars(..., lookback=20)` 对一只中间停牌 5 天的股票返回 20 行（不是 25 行）且 5 行 `is_suspended=True` 且 OHLC NaN
- `get_bars` 返回列不含 `limit_up`/`limit_down`
- `get_tradable_mask`：一字涨停 `can_buy=False`；一字跌停 `can_sell=False`；停牌两侧 False；`limit_source='unknown'` 两侧 False；退市日 `can_sell=True, can_buy=False`
- `get_bars(adjust='hfq').close == daily_bar.close * adj_factor`（数值断言）
- `get_bars(as_of_date > 数据最大日期)` 抛 `AsOfDateError`

---

### Task 12: query 财报 / 宏观 / 资金流 — get_financial / get_financial_ttm / get_macro / get_money_flow

**Files:** Modify `ashare/data/query.py`；Create `tests/ashare/test_query_pit.py`

**决策**
- `get_financial`：`WHERE ann_date <= as_of_date AND update_flag = 0`（默认 `include_restated=False`），按 `end_date` 分组取 `ann_date` 最大者；附加列 `ann_date, end_date, report_type, lag_days`。
- `get_financial_ttm(as_of_date, ts_codes, field)`：**TTM 拼接在 query 层**（架构 §4.1 定死）。
  流量科目：`最新累计 + 上年年报 − 上年同期累计`；存量科目：期初期末均值；按 field 白名单分派；不足 4 期或跨期缺失 → NaN，**不外推**。
- `get_macro`：`WHERE publish_date <= as_of_date`，同 `(indicator, period)` 取 `publish_date` 最大者；附加 `<indicator>__publish_date` 列。
- `get_money_flow`：`hk_hold_ratio` 2016-12-05 前返回 NaN，不填 0（B5）。

**验收断言（规格 §11 + 算法说明书 §1）**
- `get_financial('600519.SH', as_of='2021-04-01')` 的 `end_date == 2020-12-31` 且 `ann_date <= 2021-04-01`
- 同一 `end_date` 先后两次披露：`as_of` 在两次之间取第一次值，之后取第二次值
- `include_restated=False` 时 `update_flag=1` 的行不可见
- TTM 跨年重置：构造 Q1/H1/Q3/FY 累计值，`get_financial_ttm` 在 Q1 后 = Q1 + FY − 上年 Q1
- `get_macro('2024-08-01', ['m2_yoy'])` 拿不到 2024-07 的值（publish_date 08-15）；`'2024-08-15'` 拿得到（D4）

---

### Task 13: validate.py 六项校验 + BaoStock 双源交叉

**Files:** Create `ashare/data/validate.py`, `ashare/data/sources/baostock.py`；Create `tests/ashare/test_validate.py`

**决策（规格 §4.4）**
- 六项：行数完整性（**误差为 0**）/ 占位行合规 / 复权因子跳变 [0.5, 2.0] / `financial_pit.ann_date` 缺失率 0 /
  `macro.publish_date` 缺失率 0 / 涨跌停缺失率（`limit_source='unknown'` 占比报告，不阻断）。
- 双源交叉：抽样 200 只 × 100 日，BaoStock 后复权收盘价 vs Tushare 偏差 < 0.5%；BaoStock 不可用时**标 SKIPPED 而非 PASS**。
- 返回 `list[CheckResult(name, passed, detail, blocking)]`；任一 `blocking and not passed` → `run_all()` 抛 `ValidationError`。
- 校验只读：`validate.py` 只用 `connect_read`。

**验收断言**
- 人为删掉一行 daily_bar → 行数完整性 FAIL 且 detail 指出 (ts_code, 缺失区间)
- 占位行 `vol != 0` → FAIL
- adj_factor 跳变 3.0 → 告警级（非阻断）
- BaoStock 不可用 → 该项 `SKIPPED`，`run_all()` 不抛

---

### Task 14: 驱动 + 影子文件替换 + 全量回补 + P1 验收

**Files:** Create `ashare/data/promote.py`, `ashare/data/__main__.py`（`python -m ashare.data.ingest --full|--daily`）；Create `tests/ashare/test_p1_acceptance.py`；Modify `docs/oos-runs.md`（新建）

**决策**
- 写者写 `data/ashare_staging/market.duckdb`，校验通过后 `CHECKPOINT` → `os.replace` 到 `data/ashare_market.duckdb`；旧文件 `os.link` 成 `.bak.<snapshot_id>` 零拷贝快照，保留 3 份。
- `--daily`：`end` **clamp 到日历中最后一个 `<= today` 的交易日**（Task 6 Minor：否则当天被冻结成停牌占位）；`start = 上次 DONE 的最大 trade_date + 1`。
- `--full`：按年分批、按表分组；每批结束写 `ingest_log`；中断可续。全量回补是**操作员手动触发**，需要 `TUSHARE_TOKEN`。
- `test_p1_acceptance.py`：规格 §11 七条断言写成 pytest；若 `data/ashare_market.duckdb` 不存在则 `pytest.skip`（CI 无数据）。

**验收断言**
- 影子替换后 `query.snapshot_id()` 变化且旧连接自动重连
- `--daily` 的 end 不超过最后已发布交易日
- 规格 §11 P1 七条全部通过（对真实数据）

---

### Task 15: RAG chunk 增加 publish_date（U4）

**Files:** Modify `rag/indexer.py`；Create `tests/ashare/test_rag_publish_date.py`

**决策**
- 从文件名解析 `YYYY-MM-DD` / `YYYYMMDD`；解析不到读同目录 `publish_dates.json` 覆盖；都没有 → `publish_date=None, publish_date_source='unknown'`，**不猜**。
- 只写 metadata，**本期不做检索侧 `as_of_date` 过滤**（架构 A6）。

**验收断言**
- 文件名 `研报_茅台_2026-05-18.pdf` → chunk metadata `publish_date='2026-05-18'`
- 无日期文件 → `publish_date=None, publish_date_source='unknown'`
- 现有 `search_research_docs` 行为不变

## 自查

**规格覆盖**：D1（Task 1 只读连接 + Task 2 L4 检查）、D2（Task 2 L2 检查 + Task 9-12）、D3（Task 7、12）、D4（Task 8、12）、D5（Task 5、10）、D6（Task 4、11）、D7（Task 9 `snapshot_id`）、D8（Task 6 `adj_factor` + Task 11 `adjust='hfq'`）、D9（Task 6）—— 九条铁律各有归属任务。

**占位符扫描**：Task 0–6 的每个代码步骤都是可直接粘贴运行的完整实现，无 TBD。Task 7–15 明确标注为「Task 6 通过后展开」，这是刻意的分批而非占位。

**类型一致性**：`compute_limits` 在 Task 4 定义、Task 6 调用，签名一致（返回三元组）。`_upsert`/`set_job`/`job_state` 在 Task 5 定义、Task 6 复用。`normalize_daily_bar` 的返回列与 `schema.sql` 的 `daily_bar` 列一一对应（14 列）。
