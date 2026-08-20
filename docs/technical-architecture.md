# QuantAgent · 技术架构文档

> 版本: 2026-05-29 · 对应代码状态: 主分支
> 适用范围: 工程师 onboarding、架构评审、新功能设计参考

---

## 0. 文档导览

| 章节 | 内容 |
|---|---|
| §1 项目概览 | 一句话定位 + 关键数据指标 |
| §2 技术栈 | 全量依赖清单 + 选型理由 |
| §3 整体架构 | 系统分层 + 数据流 + 关键组件图 |
| §4 请求生命周期 | 一次 HTTP 请求从浏览器到 LLM 再回到浏览器的全链路 |
| §5 模块分层详解 | 每个 Python 包/模块的职责、边界、关键 API |
| §6 **LangGraph 架构深度详解** | 节点 / 边 / State / 调度 / 事件流 / 与原版差异 ⭐ |
| §7 工具链（Tool Registry） | 工具注册 / 调度 / Schema / 鉴权门 |
| §8 安全护栏 | 交易两阶段 / 风控 / ADR-0002 实盘五层 |
| §9 数据持久化 | Session / Memory / Cache / Broker DB / Vector DB |
| §10 配置体系 | env vars / ADR / CLAUDE.md 三层约束 |
| §11 测试 harness | 单元 / e2e / Mock 策略 |
| §12 部署与运维 | 启动命令 / Redis 可选 / 监控指标 |
| §13 已知技术债与演进路线 | G1–G5 北极星目标 |

---

## 1. 项目概览

**QuantAgent** 是一个**流式量化交易 AI Agent**：FastAPI Web 服务 + LLM 驱动的工具使用主循环 + RAG 研报检索 + 多券商接入 + 半自动下单。

| 指标 | 数值 |
|---|---|
| 代码总规模 | ~13 100 行 Python（主项目） |
| LLM 工具数 | 30+ 个（行情/因子/技术/期权/RAG/记忆/交易） |
| 支持券商 | Mock / Alpaca (paper) / Tiger (paper + live) |
| 支持 LLM Protocol | DeepSeek/OpenAI 原生（默认）+ LangGraph 切换 |
| 默认端口 | 5000（macOS 用 8001/8080 因 ControlCenter 占用） |
| 测试规模 | 214 passed (unit + e2e) |

**核心设计哲学**：
1. **LLM 输出即不可信** — 所有副作用走鉴权 + 意图 + 风控 + 审计
2. **工具是契约** — 永远返回 dict，绝不抛异常跨 LLM 边界
3. **配置即约束** — env vars + ADR + CLAUDE.md 三层叠加

---

## 2. 技术栈

### 2.1 后端

| 层 | 选型 | 版本 | 用途 |
|---|---|---|---|
| Web 框架 | FastAPI | 0.110+ | 异步 + SSE 流式 + OpenAPI |
| ASGI 服务器 | uvicorn[standard] | 0.27+ | HTTP/1.1 + WebSocket |
| 数据模型 | pydantic | 2.0+ | 请求体校验 + 类型推断 |
| LLM SDK (默认) | openai | 1.40+ | DeepSeek 走 OpenAI 协议 |
| LLM SDK (可选) | langchain-openai + langgraph + langchain-core | 0.2+ / 0.2+ / 0.3+ | LangGraph 版主循环 |
| HTTP 中间件 | starlette.SessionMiddleware | — | OAuth state cookie session |

### 2.2 数据与缓存

| 层 | 选型 | 用途 | Fallback |
|---|---|---|---|
| 内存/分布式缓存 | redis ≥ 5.0 | 会话、意图、Answer Cache | 内存字典（无 REDIS_URL 自动降级） |
| 关系存储 | sqlite3（标准库） | Broker credentials、audit log、users | — |
| 向量数据库 | chromadb ≥ 1.5 | RAG + episodic memory + profile | — |
| 嵌入模型 | sentence-transformers ≥ 5.0 (BAAI/bge-*) | 中英文向量化 + 重排 | — |
| PDF 解析 | PyMuPDF ≥ 1.26 | 研报 PDF → chunk | — |

### 2.3 行情/交易/数据源

| 数据源 | SDK | 用途 | 是否需要密钥 |
|---|---|---|---|
| yfinance | yfinance ≥ 0.2 | 美股/港股延迟报价 + 期权链兜底 | ❌ |
| akshare | akshare ≥ 1.18 | A 股行情/财报 | ❌ |
| Alpaca | alpaca-py ≥ 0.30 | 美股 paper trading | ✅ |
| Tiger Brokers | tigeropen 3.2.x | 美/港/A 股 paper/live + 实时推送 | ✅（per-user RSA key） |
| 新闻聚合 | urllib/requests + RSS/HTML 解析 | 7 个新闻源 fallback | 部分需密钥 |

### 2.4 认证 / 安全

| 组件 | 选型 | 用途 |
|---|---|---|
| OAuth | authlib ≥ 1.3 | Google / GitHub 第三方登录 |
| Cookie 签名 | itsdangerous ≥ 2.1 | Web session token 签名 |
| 凭证加密 | cryptography ≥ 42 | KEK + DEK 信封加密券商凭证 |

### 2.5 前端

| 层 | 选型 | 理由 |
|---|---|---|
| 渲染 | 原生 HTML + CSS + Vanilla JS | 单文件 SPA 风格，零构建链 |
| Markdown | server-side markdown 渲染 | LLM 输出直接服务端渲染好 HTML 推下来 |
| 流式 | SSE（Server-Sent Events） | 比 WebSocket 简单、单向流式天然契合 LLM token 输出 |
| 图表 | static/charts/ 下的轻量 JS（自维护） | 不依赖大型图表库 |

### 2.6 测试

| 层 | 工具 | 范围 |
|---|---|---|
| 单元 | pytest | 各模块 `tests/<module>/` |
| e2e | pytest + monkeypatch | `tests/e2e/`，mock LLM 跑全 dispatch |
| 网络隔离 | monkeypatch fixture | 强制铁律：无真实网络 |

---

## 3. 整体架构

### 3.1 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│ Browser  (static/index.html · 单文件 SPA · SSE 客户端)        │
└────────────────────────────┬────────────────────────────────┘
                             │  HTTPS  (REST + SSE)
┌────────────────────────────▼────────────────────────────────┐
│ FastAPI Server  (server.py · 1840+ 行)                       │
│   ├─ Auth Middleware  (auth/)                                │
│   ├─ Session Layer    (cache.py 抽象)                        │
│   ├─ Threadlocal 注入 (device_id + authenticated)            │
│   └─ SSE 桥           (event → SSE 字符串)                    │
└────────┬─────────────────────────────────────────┬───────────┘
         │                                         │
         │  stream_quant_agent(messages)           │  REST endpoints
         │  (sync gen / async gen)                 │  /api/broker/*
         │                                         │  /api/orders/*
┌────────▼────────────┐         ┌──────────────────▼───────────┐
│ Agent 主循环         │         │ Broker / Order 端点          │
│ ├─ quant_agent.py    │         │ ├─ confirm-order (5 层闸门)  │
│ │   (原生 ReAct)      │         │ ├─ bind / unbind             │
│ └─ quant_agent_lg.py │         │ ├─ live-orders toggle        │
│   (LangGraph)        │         │ └─ symbols/{search,brief,    │
│                      │         │     option-expiries,chain}   │
└──────┬───────────────┘         └──────────────────────────────┘
       │
       │  dispatch_tool(name, input)
       │
┌──────▼──────────────────────────────────────────────────────┐
│ TOOL_REGISTRY  (30+ tools, 单一事实源)                        │
│   ├─ 行情:  market_quote / historical_prices / market_news    │
│   ├─ 因子:  factor_score / technical_indicator                │
│   ├─ 期权:  black_scholes / implied_volatility /              │
│   │         search_symbol_options                             │
│   ├─ 风险:  risk_metrics / correlation_matrix /               │
│   │         portfolio_optimizer                               │
│   ├─ RAG:   search_research_docs                              │
│   ├─ 记忆:  get_user_profile / update_user_profile /          │
│   │         record_memory                                     │
│   └─ 交易:  broker_account / place_order_intent /             │
│             cancel_order  (⚠️ 鉴权门)                          │
└──────┬────────────────────────┬─────────────────────────────┘
       │                        │
       │ 数据/计算工具调用       │ 交易工具:意图存盘,不直接下单
       │                        │
┌──────▼─────────┐   ┌──────────▼──────────────────────────────┐
│ 外部数据源     │   │ Brokers / Trading                       │
│ yfinance/      │   │ ├─ BrokerRegistry  (per-user adapter)   │
│ akshare/       │   │ ├─ MockAdapter     (虚拟 $100k)         │
│ Tiger SDK/     │   │ ├─ AlpacaAdapter   (paper trading)      │
│ news APIs/     │   │ ├─ TigerAdapter    (paper + live)       │
│ Yahoo RSS      │   │ ├─ intent_store    (两阶段订单意图)     │
│                │   │ ├─ risk_gate       (预交易风控)         │
│                │   │ ├─ credentials_store (信封加密)         │
│                │   │ └─ audit            (append-only 审计)  │
└────────────────┘   └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ State / Storage                                              │
│ ├─ Redis / Memory cache  (sessions / intents / answer cache) │
│ ├─ SQLite (data/brokers.db)  (encrypted credentials + audit) │
│ ├─ ChromaDB (rag_db/)        (RAG + profile + episodic)      │
│ └─ Markdown/HTML 静态资源    (static/)                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计选择

| 选择 | 原因 |
|---|---|
| **同一 TOOL_REGISTRY 服务两套 Agent 主循环** | 协议层可换（DeepSeek/OpenAI vs LangGraph），工具层不变 |
| **SSE 而非 WebSocket** | LLM 输出本质单向流式；SSE 重连/兼容性更好；无需协议升级 |
| **Redis 可选** | 开发/CI 零依赖；生产可加 Redis 横向扩展 |
| **SQLite 而非 Postgres** | 中小规模 SaaS（百用户级）够用；ACID 文件级，运维 0 |
| **Chroma 而非 Pinecone/Weaviate** | 本地嵌入式，零运维；中文 BGE 模型效果足够 |
| **凭证信封加密** | KEK 轮换不用重加密 credentials；AWS KMS 同款模式 |
| **意图/提交分离** | LLM 不可直接动钱；UI 显式确认是不可绕过的安全闸 |

---

## 4. 请求生命周期

以"用户问『NVDA 适合买吗』"为例，完整链路：

```
1. 浏览器
   ├─ JS 调 /api/sessions/<sid>/chat
   ├─ 携带 X-Device-Id header（匿名用户）或 Cookie（登录用户）
   └─ 建立 SSE 连接

2. server.py: chat(sid, body, x_device_id, user)
   ├─ _scope_id(user, x_device_id) → "u:42" 或 device id
   ├─ _set_request_device_id(scope) ←┐
   ├─ _set_request_authenticated(...) ├ 写入 threading.local
   ├─ session = get_session(scope, sid)
   ├─ session.messages.append({"role":"user", "content":"NVDA 适合买吗"})
   ├─ save_session(...)  # 持久化第一道
   └─ return StreamingResponse(event_stream())

3. event_stream() (跑在 starlette 线程池)
   ├─ ⚠️ 关键：threadlocal 在新线程里要重新设
   ├─ _set_request_device_id(...)
   ├─ _set_request_authenticated(...)
   ├─ 命中 answer cache? → _replay_events_from_turn → 早退
   └─ for event in stream_quant_agent(session.messages):
         yield f"data: {json.dumps(event)}\n\n"

4. stream_quant_agent(messages)
   ├─ _build_system_prompt(messages)  ← Layer 1 profile + Layer 2 episodic
   ├─ messages[0] = {"role":"system", "content": sys}
   ├─ _sanitize_messages(messages)
   └─ for iter in range(1, 16):
        ├─ stream = client.chat.completions.create(..., tools=..., stream=True)
        ├─ for chunk in stream:
        │   ├─ delta.content → content_buf += ; yield content_delta
        │   └─ delta.tool_calls → tool_calls_buf 累积 (按 index)
        ├─ if finish=='stop':
        │   ├─ yield final
        │   ├─ _generate_followups(messages)  ← 独立轻量 LLM 调用
        │   └─ yield suggestions; return
        └─ if finish=='tool_calls':
            ├─ for tc in tool_calls_buf:
            │   ├─ yield tool_call
            │   ├─ obs = dispatch_tool(name, args)
            │   ├─ yield tool_result
            │   └─ tool_msgs.append({"role":"tool", ...})
            ├─ messages.append(assistant_msg)
            └─ messages.extend(tool_msgs)   # 原子提交,continue

5. dispatch_tool(name, input)
   ├─ if name in TRADING_TOOLS and not authenticated: return auth_error
   ├─ fn = TOOL_REGISTRY[name]
   └─ try: return fn(**input)
      except TypeError: return {"success":False, "error": "参数错误..."}
      except Exception: return {"success":False, "error": "工具执行异常..."}

6. 工具内部（如 market_quote）
   ├─ 命中 _tool_cache？→ 直接返回
   ├─ yfinance → 失败 → sina → 失败 → mock + 标 [MOCK DATA]
   ├─ 整理成 {"symbol":..., "price":..., "source":...}
   └─ return dict（永不抛）

7. 终态写回 + 持久化
   ├─ assistant_turn 写入 session.display
   ├─ save_session(scope, session)  ← 持久化第二道
   ├─ if cacheable: cache.set(answer_cache_key, turn, ttl=300)
   └─ yield {"type":"done"}

8. 浏览器
   ├─ EventSource 收到 final → 渲染 HTML
   ├─ 收到 suggestions → 渲染推荐问题卡
   └─ 收到 done → 关闭 SSE 连接
```

**关键不变量**：
- 每一轮 LLM 输出都先经过完整的 `messages` 喂回（包括 system 重算）
- 工具返回 dict 永远 JSON-序列化成功
- 任何异常都会 yield `error` 事件而非直接中断 SSE
- session 在用户消息和最终回复两个时点都持久化（防丢）

---

## 5. 模块分层详解

### 5.1 顶层 Python 文件

| 文件 | 行数 | 角色 | 边界 |
|---|---|---|---|
| `server.py` | 1840+ | **唯一真实入口**。FastAPI 应用、所有 REST/SSE 端点、auth 胶水 | 不包含业务逻辑（业务 → quant_agent / brokers） |
| `quant_agent.py` | 5234 | **核心库**。Tool registry + schemas + system prompt + ReAct 主循环 | 不调 FastAPI；不持有进程级状态（用 threadlocal 接住请求态） |
| `quant_agent_lg.py` | 511 | **LangGraph 版主循环**。复用 quant_agent 的工具/schema/prompt | 见 §6 详解 |
| `cache.py` | 100+ | **缓存抽象层**。Redis/Memory 鸭子接口 | drop-in compatible；业务方调用永远写 `from cache import cache` |
| `agent_demo.py` | 936 | **遗留教学 demo**。Anthropic SDK + 3 工具最小示例 | ❌ 不在产品调用链；只供学习 |
| `LoopDemo.py` | — | **用户实验文件**。任意 Python 学习/测试用，不在主链路 | — |

### 5.2 包/目录

```
demo/
├── auth/                    OAuth + Cookie session + email login
│   ├── __init__.py          公共 API: User / current_user / require_user
│   ├── oauth.py             Google / GitHub OAuth 流程
│   ├── email_login.py       邮箱验证码登录
│   ├── sessions.py          Web session token (signed cookie)
│   ├── users.py             User CRUD (SQLite 表)
│   ├── deps.py              FastAPI 依赖项
│   └── admin.py             管理员后台 endpoints
│
├── brokers/                 券商抽象 + 凭证 + 风控 + 审计
│   ├── base.py              BrokerAdapter ABC + Credentials dataclass + redact
│   ├── mock_adapter.py      虚拟 $100k 模拟（无外部依赖）
│   ├── alpaca_adapter.py    Alpaca paper trading
│   ├── tiger_adapter.py     Tiger paper/live, RSA 认证
│   ├── tiger_push.py        Tiger WebSocket 实时推送
│   ├── tiger_quote.py       Tiger 行情查询（实时/延迟兜底）
│   ├── quote_fallback.py    yfinance 兜底（无券商权限时用）
│   ├── realtime_state.py    Push state store (per user_scope 内存累积)
│   ├── registry.py          BrokerRegistry: per-user adapter 实例化
│   ├── credentials_store.py 信封加密 + SQLite 持久化
│   ├── crypto.py            KEK/DEK 操作 + 轮换
│   ├── audit.py             append-only 审计表
│   ├── metrics.py           /metrics/brokers 进程内计数
│   ├── intent_store.py      OrderIntent 两阶段存储
│   ├── risk_gate.py         单笔/日总/市价/白名单风控
│   ├── _db.py               SQLite 连接管理
│   ├── _schema.sql          DB DDL
│   ├── rotate_kek.py        KEK 轮换工具脚本
│   ├── gen_kek.py           生成新 KEK 工具脚本
│   └── cli.py               bind/unbind CLI（运维用）
│
├── memory/                  长期记忆（用户档案 + 情景记忆）
│   ├── __init__.py          公共 API
│   ├── profile.py           结构化档案（chromadb 存 JSON）
│   └── episodic.py          情景记忆原文（chromadb + 向量检索）
│
├── rag/                     研报 PDF 检索
│   ├── config.py            embedding 模型 / chunk 参数
│   ├── indexer.py           PDF → chunk → 向量化 → chromadb
│   └── retriever.py         query → 检索 + 重排
│
├── data/                    SQLite DB 文件（gitignored）
│   └── brokers.db
│
├── rag_db/                  chromadb 持久化目录（gitignored）
├── docs/                    架构文档 + ADR + 研报样本
├── scripts/                 运维/诊断脚本（reindex / diag_* / setup_venv）
├── static/                  前端
│   ├── index.html           主聊天 SPA（~3300 行 HTML+JS+CSS）
│   ├── brokers.html         券商绑定管理页
│   └── charts/              自维护轻量图表 JS
└── tests/
    ├── auth/
    ├── brokers/
    └── e2e/                 端到端（mock LLM）
```

### 5.3 模块边界规约

CLAUDE.md G5 明文：
- **认证/鉴权只能走 `auth/`** — 业务模块不准引入新的 threadlocal 全局或 hardcoded 用户集合
- **`brokers/risk_gate.py` 是订单路径必经** — 不允许"绕开测试"
- **每个 broker 必须有 paper/simulator 模式** — CI 能跑

---

## 6. LangGraph 架构深度详解 ⭐

本节是用户特别要求的重点。这是 `quant_agent_lg.py` 的完整解析。

### 6.1 设计目标

| 目标 | 通过 LangGraph 怎么解决 |
|---|---|
| 替代 `while iter < 15` 命令式循环 | `StateGraph` 声明式编排 |
| tool_calls 并行执行 | `tools_node` 用 `asyncio.gather` 并行 |
| `render` + `followup` 并行（最终答案出来后两个独立任务） | 条件路由返回 list → LangGraph fanout |
| 节点级 retry / checkpoint（待启用） | 内建能力 |
| **不破坏 SSE 协议** | 用 `astream_events(version="v2")` + 自定义事件桥接 |

**关键约束**：跟原 `stream_quant_agent` **yield 协议完全一致**，前端无需改一行代码即可切换。

### 6.2 节点定义（4 个）

```
┌────────────┐
│   START    │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────────┐
│  llm                                │
│  ─ 异步流式调 ChatOpenAI            │
│  ─ token 批处理(30ms / 40字)        │
│  ─ dispatch round_start / llm_token │
│  ─ 累积 AIMessage(content+tool_calls)│
└─────┬───────────────────────────────┘
      │
      │ _should_finalize(state)
      │
      ├─ has tool_calls? ──→ ["tools"]
      │
      └─ 无 tool_calls ────→ ["render", "followup"]  ← list 触发并行
                                  │              │
                                  ▼              ▼
                          ┌──────────┐    ┌──────────┐
                          │  render  │    │ followup │
                          │ MD→HTML  │    │ 生成3个   │
                          └────┬─────┘    │ 推荐问题  │
                               │          └────┬─────┘
                               │               │
                               ▼               ▼
                            ┌────────────────────┐
                            │       END          │
                            └────────────────────┘

      ┌─────────────────────────────────┐
      │  tools (从 llm 路由进来)         │
      │  ─ 并行执行 tool_calls           │
      │  ─ dispatch tool_call_start/end │
      │  ─ ToolMessage 写回 state       │
      └─────┬───────────────────────────┘
            │
            └──→ 回到 llm
```

#### Node 1 · `llm` (lines 138-182)

**职责**：异步流式调 LLM，token 边到边推送给前端。

**State 读**：`state["messages"]`（含历史 user/assistant/tool messages + system prompt）

**State 写**：
- `messages: [full_AIMessage]`（追加一条新的 assistant 消息）
- `iteration: state.get("iteration", 0) + 1`

**关键实现细节**：

```python
async def _llm_node(state: AgentState, config=None) -> dict:
    # 新一轮通知前端清缓冲(上一轮如果说了"我来查行情..."这种过渡话会被覆盖)
    await adispatch_custom_event("round_start", {}, config=config)

    full = None
    buf, buf_len = [], 0
    last_flush = time.monotonic()

    async def _flush():
        if buf:
            await adispatch_custom_event("llm_token", {"text": "".join(buf)}, config=config)
            buf.clear(); buf_len = 0; last_flush = time.monotonic()

    async for chunk in _llm_for_request().astream(state["messages"], config=config):
        full = chunk if full is None else full + chunk
        content = getattr(chunk, "content", "") or ""
        if content:
            buf.append(content); buf_len += len(content)
            if (time.monotonic() - last_flush) * 1000 >= _TOKEN_FLUSH_MS \
               or buf_len >= _TOKEN_FLUSH_CHARS:
                await _flush()
    await _flush()   # 尾部残余

    return {"messages": [full] if full else [],
            "iteration": state.get("iteration", 0) + 1}
```

**为什么要 token 批处理**：每个 token 一次 `adispatch_custom_event` 经过 LangChain callback 链路开销不可忽略，实测可消耗 1.5-2.5s。30ms / 40 字阈值是经验值，可通过 `LG_TOKEN_FLUSH_MS` / `LG_TOKEN_FLUSH_CHARS` env 覆盖。

#### Node 2 · `tools` (lines 185-250)

**职责**：并行执行 AIMessage.tool_calls 里所有工具调用。

**State 读**：`state["messages"][-1]`（必须是带 tool_calls 的 AIMessage）

**State 写**：`messages: [ToolMessage, ...]`（每个工具一条）

**并行 + 鉴权透传**：

```python
async def _tools_node(state, config=None):
    last = state["messages"][-1]
    if not getattr(last, 'tool_calls', None):
        return {"messages": []}

    # 全部先 dispatch tool_call_start(用户立刻看到状态)
    for tc in last.tool_calls:
        await adispatch_custom_event("tool_call_start", {...}, config=config)

    # ⚠️ 关键: 主线程读 threadlocal,传给 worker 线程重新注入
    _ctx_authed = _is_request_authenticated()
    _ctx_device = _get_request_device_id()

    def _run_one(tc):
        # worker 线程 threadlocal 是空的,必须重设
        _set_request_device_id(_ctx_device)
        _set_request_authenticated(_ctx_authed)
        result = dispatch_tool(name, args)
        ...

    # 并行执行
    coros = [loop.run_in_executor(None, _run_one, tc) for tc in last.tool_calls]
    completed = await asyncio.gather(*coros)

    for name, tc_id, result, msg in completed:
        await adispatch_custom_event("tool_call_end", {...}, config=config)
        tool_msgs.append(msg)

    return {"messages": tool_msgs}
```

**关键约束**：
1. **threadlocal 跨线程问题**：`run_in_executor` 把 `_run_one` 扔到 ThreadPool 跑，那个线程的 threadlocal 是空的，所以工具调用时 `_is_request_authenticated()` 会返回 False → 交易工具被错误拒绝。**必须显式重新注入**。
2. **工具返回 dict 大小裁剪**：超 3000 字符的结果裁剪掉 `results/news/quotes` 数组的尾部，避免下一轮 prompt 爆掉。
3. **错误归一化**：`dispatch_tool` 已确保返回 dict，但 LangGraph 节点层再加一道 try/except 兜底，规避 `dispatch_tool` 自身的 bug。

#### Node 3 · `render` (lines 253-268)

**职责**：从最后一条 AIMessage 提文本，**服务端 Markdown → HTML 渲染**。

**State 读**：`state["messages"]` 倒着找最后一条非空 AIMessage

**State 写**：
- `final_text: str`
- `final_html: str`

为什么放服务端渲染：
- 前端不引 markdown 库，包体积小
- HTML 出来时就可以加自定义图表占位、白名单转义
- Cache 命中时重放也能直接拿 HTML

#### Node 4 · `followup` (lines 271-283)

**职责**：基于会话末尾生成 3 个推荐问题（独立轻量 LLM 调用，跟主答案并行）。

**State 读**：把 LC 消息转回 OpenAI dict 形式

**State 写**：`suggestions: list[str]`

**关键**：复用 `quant_agent._generate_followups` 实现（原版那套），但用 `run_in_executor` 跑在线程里**避免阻塞 event loop**——因为它内部是同步 OpenAI 调用。

### 6.3 边定义

```python
def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("llm",       _llm_node)
    builder.add_node("tools",     _tools_node)
    builder.add_node("render",    _render_node)
    builder.add_node("followup",  _followup_node)

    builder.set_entry_point("llm")
    # 条件边: 返回 list → LangGraph 同时调度
    builder.add_conditional_edges("llm", _should_finalize,
                                   ["tools", "render", "followup"])
    builder.add_edge("tools", "llm")        # tools 完了回 llm
    builder.add_edge("render", END)         # render 完了结束
    builder.add_edge("followup", END)       # followup 完了结束
    return builder.compile()
```

| 边 | 类型 | 触发条件 |
|---|---|---|
| `START → llm` | entry | 总是进入 llm |
| `llm → tools` | conditional | 最新 AIMessage 有 tool_calls |
| `llm → [render, followup]` | conditional fanout | 最新 AIMessage 无 tool_calls（终态） |
| `tools → llm` | static | 工具执行完总是回 llm |
| `render → END` | static | 渲染完结束 |
| `followup → END` | static | follow-up 完结束 |

**条件路由函数**：

```python
def _should_finalize(state: AgentState) -> list:
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
        return ["tools"]
    return ["render", "followup"]   # ⭐ list 触发并行
```

**返回 list 的语义**：LangGraph 看到 list 会**同时调度多个节点**。`render` 和 `followup` 没有数据依赖（一个看 messages 文本，一个看消息列表），可并行；最终都汇入 END。

### 6.4 State 定义

```python
class AgentState(TypedDict, total=False):
    messages:    Annotated[Sequence[BaseMessage], add_messages]
    final_text:  str
    final_html:  str
    suggestions: list
    iteration:   int
```

| 字段 | 用途 | reducer |
|---|---|---|
| `messages` | 完整对话历史 | `add_messages`（追加，不覆盖） |
| `final_text` | 最终答案纯文本 | 覆盖 |
| `final_html` | 最终答案 HTML | 覆盖 |
| `suggestions` | 3 个推荐问题 | 覆盖 |
| `iteration` | 已迭代轮次 | 节点显式 +1（防止 recursion_limit 误判） |

**`add_messages` reducer 的核心作用**：节点返回 `{"messages": [...]}` 时，**自动追加到 state["messages"]**，而不是覆盖。这让 `llm → tools → llm` 这种循环里每轮新增的消息自然累积。

### 6.5 事件流：astream_events v2 桥接 SSE

最关键的兼容性设计在 `stream_quant_agent_lg`：

```python
async for event in _graph.astream_events(state, config=config, version="v2"):
    kind = event.get("event", "")     # 'on_custom_event' / 'on_chat_model_end' / 'on_chain_end' / ...
    name = event.get("name", "")
    data = event.get("data", {})

    if kind == "on_custom_event":
        if name == "round_start":
            yield {"type": "content_reset"}
        elif name == "llm_token":
            yield {"type": "content_delta", "text": data["text"], ...}
        elif name == "tool_call_start":
            yield {"type": "tool_call", "name":..., "input":..., "id":...}
        elif name == "tool_call_end":
            yield {"type": "tool_result", "name":..., "result":..., "is_error":...}

    elif kind == "on_chat_model_end":
        iteration += 1

    elif kind == "on_chain_end":
        node = metadata.get("langgraph_node", "")
        if node == "render":
            yield {"type": "final", "text":..., "text_html":..., "iterations":...}
        elif node == "followup":
            yield {"type": "suggestions", "items":...}
        elif name == "LangGraph":
            final_state_output = output       # 全图结束,保留终态
```

事件映射表：

| LangGraph 内部事件 | 转成 SSE | 触发节点 |
|---|---|---|
| `round_start` (custom) | `content_reset` | llm 进入 |
| `llm_token` (custom) | `content_delta` | llm token 批 |
| `tool_call_start` (custom) | `tool_call` | tools 启动 |
| `tool_call_end` (custom) | `tool_result` | tools 完成 |
| `on_chain_end` of `render` | `final` | render 完成 |
| `on_chain_end` of `followup` | `suggestions` | followup 完成 |
| `on_chain_end` of `LangGraph` | （仅保留终态，不 yield） | 全图收尾 |

**事件去重**：

```python
seen_tool_runs = set()     # LangChain 偶发会重放事件
if tc_id and tc_id not in seen_tool_runs:
    seen_tool_runs.add(tc_id)
    yield {"type": "tool_call", ...}
```

### 6.6 消息格式转换

LangGraph 用 `BaseMessage` 子类，server.py 持久化的是 OpenAI dict 格式。两者要互转：

| OpenAI dict | LangChain |
|---|---|
| `{"role":"system","content":"..."}` | `SystemMessage(content=...)` |
| `{"role":"user","content":"..."}` | `HumanMessage(content=...)` |
| `{"role":"assistant","content":"...", "tool_calls":[...]}` | `AIMessage(content=..., tool_calls=[{name,args,id}])` |
| `{"role":"tool","tool_call_id":"...","content":"..."}` | `ToolMessage(content=..., tool_call_id=...)` |

转换函数：
- `_openai_to_lc_messages` (line 329) — 入口转
- `_lc_to_openai_messages` (line 365) — 出口转，写回 server.py 的 session.messages

**入口 `stream_quant_agent_lg` 末尾的回写**：

```python
if final_state_output and final_state_output.get("messages"):
    new_openai = _lc_to_openai_messages(list(final_state_output["messages"]))
    messages.clear()
    messages.extend(new_openai)
```

这一段保证 LangGraph 跑完后，**server.py 那个原始 messages list 被原地更新**——server.py 继续按 OpenAI 格式持久化即可，下次再调 LangGraph 还能继续。

### 6.7 鉴权透传机制

```python
# 模块加载时绑定两套工具
_llm_with_tools      = _llm.bind_tools(_to_openai_tools(TOOL_SCHEMAS))
_llm_with_tools_anon = _llm.bind_tools(
    _to_openai_tools([s for s in TOOL_SCHEMAS if s["name"] not in TRADING_TOOLS]))

def _llm_for_request():
    return _llm_with_tools if _is_request_authenticated() else _llm_with_tools_anon
```

`_llm_node` 调用时 `_llm_for_request()` 按当前 threadlocal 取对应版本——**未登录用户的 LLM 直接看不到交易工具的 schema**，从源头就调不到。

### 6.8 与原版 ReAct 实现的对比

| 维度 | quant_agent.py (原版) | quant_agent_lg.py (LangGraph) |
|---|---|---|
| 主循环 | `for iter in range(1, max_iterations+1)` | `StateGraph` 编排 + 循环边 `tools → llm` |
| 流式 | OpenAI 流式 API + 手动累积 delta | `_llm.astream(...)` + token 批处理 |
| Tool 并行 | ❌ for tc in tool_calls_buf 顺序 | ✅ asyncio.gather 并行 |
| Render | 主循环内 yield final 时同步渲染 | 独立 `render` 节点，可与 followup 并行 |
| Followup | yield final 后顺序调一次 LLM | 独立 `followup` 节点，并行 |
| 鉴权 | dispatch_tool 守门 + schema 屏蔽 | 同 + LangGraph 节点内重设 threadlocal |
| 错误处理 | try/except 在 stream loop | 节点级 try/except + LangGraph 内建 retry（待启用） |
| Checkpoint | ❌ | ✅ LangGraph 内建（待启用） |
| 代码行数 | ~300 行（含 dispatch + tools） | ~510 行（含转换 + 事件桥） |

**选哪个**：
- 默认走原版 — 代码简单、依赖少、易调试
- 切 LangGraph (`USE_LANGGRAPH=1`) — 并行执行收益明显（多工具调用场景）、想用 checkpoint 时

### 6.9 LangGraph 版本的扩展点

设计上预留但**尚未启用**的能力：

| 能力 | 启用方式 | 价值 |
|---|---|---|
| 节点级 retry | `builder.add_node(..., retry=RetryPolicy(...))` | 网络抖动自动重试 |
| Checkpointer | `_graph.compile(checkpointer=SqliteSaver(...))` | 长任务断点续跑、回溯调试 |
| Human-in-the-loop | `interrupt_before=["tools"]` | 工具执行前人工审核 |
| State persistence | 跨进程会话恢复 | 多实例部署 |

---

## 7. 工具链（Tool Registry）

### 7.1 双注册结构

```python
TOOL_REGISTRY = {                          # name → callable
    "market_quote": market_quote,
    "factor_score": factor_score,
    ...
}

TOOL_SCHEMAS = [                           # Anthropic 风格 schema
    {"name": "market_quote",
     "description": "...",
     "input_schema": {"type":"object", ...}},
    ...
]

TRADING_TOOLS = {                          # 鉴权白名单
    "broker_account", "place_order_intent", "cancel_order",
    "get_user_profile", "update_user_profile", "record_memory",
}
```

模块加载时**一次性预算**两套 OpenAI 风格工具列表：

```python
_OPENAI_TOOLS      = _to_openai_tools(TOOL_SCHEMAS)
_OPENAI_TOOLS_ANON = _to_openai_tools([s for s in TOOL_SCHEMAS
                                         if s["name"] not in TRADING_TOOLS])

def _current_openai_tools():
    return _OPENAI_TOOLS if _is_request_authenticated() else _OPENAI_TOOLS_ANON
```

### 7.2 dispatch_tool 守门

```python
def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    # 1. 鉴权
    if tool_name in TRADING_TOOLS and not _is_request_authenticated():
        return {"success": False, "error_type": "auth_required", ...}

    # 2. 注册检查
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"success": False, "error": f"工具 '{tool_name}' 不存在", ...}

    # 3. 调用 + 异常归一化
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {e}"}
    except Exception as e:
        return {"success": False, "error": f"工具执行异常: {e}"}
```

**铁律**：**dispatch_tool 永远返回 dict**，从不让异常跨越到 LLM。

### 7.3 工具内部错误模式（Multi-source fallback）

`market_quote` 是典型代表：

```python
def market_quote(symbol, skip_cache=False):
    # 1. 缓存
    cached = _get_cached(symbol)
    if cached and not skip_cache: return cached

    # 2. 多源 fallback (顺序尝试)
    for source_fn in (_quote_yfinance, _quote_sina, _quote_akshare):
        try:
            result = source_fn(symbol)
            if result: return {**result, "source": source_fn.__name__}
        except Exception:
            continue

    # 3. 都失败 → mock + 显式标识
    return {**_mock_quote(symbol), "data_source": "[MOCK DATA]"}
```

**这个模式被全项目复用**：
- `tiger_quote.get_brief` (实时 → 延迟)
- `option-chain` endpoint (Tiger → yfinance)
- News (7 个新闻源依次尝试)

### 7.4 大返回值裁剪

```python
def _truncate_observation(obs: dict, obs_str: str, max_len=2500) -> str:
    if len(obs_str) <= max_len: return obs_str

    # 语义化裁剪: 知道 results/news/quotes 是列表
    for k in ("results", "news", "quotes"):
        if isinstance(obs.get(k), list) and len(obs[k]) > 3:
            obs[k] = obs[k][:3]
            obs["truncated"] = True

    # 字符串字段截 200
    for list_field in ("results", "news"):
        for item in obs.get(list_field, []):
            for tk in ("content", "summary", "title"):
                v = item.get(tk)
                if isinstance(v, str) and len(v) > 200:
                    item[tk] = v[:200] + "..."

    return json.dumps(obs, ensure_ascii=False, default=str)
```

**注意**：`historical_prices` 的 `close/dates` 数组**不裁剪**——它们是下游 `technical_indicator` 的必需输入，裁了就断链。

---

## 8. 安全护栏

### 8.1 交易两阶段 (Intent → Confirm)

```
[1] LLM 决定下单
        │
        ▼
[2] place_order_intent(symbol, qty, side, limit_price, ...)
    ├─ 生成 OrderIntent (id, ts, ttl=5min)
    └─ save_intent(scope, intent)
                │
                │  LLM 拿不到 broker_order_id
                ▼
[3] LLM 回复用户「已生成意图 X,确认下单？」
        │
        ▼
[4] 前端展示意图详情 + 红色按钮「确认下单」
        │
        ▼
[5] 用户点击 → POST /api/orders/confirm {intent_id, double_confirmed}
        │
        ▼
[6] server.confirm_order():
    ├─ pop_intent(scope, id)             ← 意图只能用一次
    ├─ _live_order_blocked(...)          ← ADR-0002 Layer 5
    ├─ broker.get_account()              ← 拉账户快照
    ├─ check_order(intent, acc, ...)     ← risk_gate
    ├─ broker.place_order(intent)        ← 真正落单
    └─ record_order_placed(scope)
```

### 8.2 risk_gate 规则

| 规则 | 默认值 | env 覆盖 |
|---|---|---|
| 单笔上限占净值 % | 0.20 | `RISK_MAX_SINGLE_PCT` |
| ETF 单笔上限占净值 % | 0.50 | `RISK_MAX_SINGLE_PCT_ETF` |
| 二次确认阈值 % | 0.50 | `RISK_DOUBLE_CONFIRM_PCT` |
| 日内下单笔数上限 | 20 | `RISK_MAX_DAILY_ORDERS` |
| 日内撤单率上限 | 0.40 | `RISK_MAX_DAILY_CANCEL_RATE` |
| 禁用市价单 | True | `RISK_BLOCK_MARKET_ORDER` |
| 白名单（symbol） | DEFAULT_WHITELIST | `RISK_WHITELIST` |

任意一条不过 → **意图丢弃** + 返回 `blocked_by_risk_gate: true`。

### 8.3 ADR-0002 实盘五层

实盘下单**五层全部 True** 才允许：

```
1. binding.env = 'live'                  ← bind 时显式选
2. binding.live_orders_enabled = 1       ← 单独 POST + 用户输入「我确认开启下单」
3. risk_gate.check_order(...).passed     ← 单笔/日内/市价/白名单
4. UI 二阶段红色 banner + 用户点击确认    ← /api/orders/confirm
5. server.confirm_order() 二次校验 env+flag ← 防中途篡改
```

任一缺失 → 422 拒绝，意图回写（用户可重试）。

**关键不变量**：LLM 工具层**没有任何代码路径**能翻 `live_orders_enabled` 开关——只能由认证用户在 UI 上操作。

### 8.4 凭证保护（信封加密）

```
.env: BROKER_KEK_v1=<base64-32bytes>     ← 主密钥(KEK),可轮换
                       │
                       │ 解密
                       ▼
                 DEK (per-binding)        ← 数据密钥,密文存 SQLite
                       │
                       │ 解密
                       ▼
                 credentials JSON         ← 内存明文,使用后归零,缓存 ≤ 60s
```

**铁律**：
- 凭证**永远不进**日志、异常 message、LLM 返回值、system prompt
- `redact_credentials()` 正则匹配 PEM block / 长 base64，强制过滤
- 每个 adapter 的 `__repr__` 显式不打 private_key

### 8.5 审计（append-only）

`brokers/audit.py` 写 SQLite 表 `broker_audit_log`：

| 字段 | 说明 |
|---|---|
| `ts` | 时间戳 |
| `actor` | `user` / `llm` / `system` |
| `action` | `bind` / `unbind` / `use` / `rotate` / `fail` / `read` |
| `user_id`, `binding_id`, `broker_type`, `label` | 上下文 |
| `detail` | JSON 详情 |

**关键告警**：任何 `actor='llm' AND action ∈ {bind, unbind, rotate}` 的行 = critical incident。LLM 只应该 trigger `use`，其它都意味着守门破了。

---

## 9. 数据持久化

| 数据 | 存储 | Key 形式 | TTL |
|---|---|---|---|
| Web session token | Redis / 内存 | `auth:session:<token>` | 7d |
| Quant session (messages + display) | Redis / 内存 | `quant:session:<scope>:<sid>` | 7d |
| Session index | Redis / 内存 | `quant:session:index:<scope>` | 永久 |
| Order intent | Redis / 内存 | `intent:<scope>:<id>` | 5min |
| Answer cache | Redis / 内存 | `answer:<hash>` | 5min |
| User profile | ChromaDB | collection: `user_profiles` | 永久 |
| Episodic memory | ChromaDB | collection: `episodic_<user>` | 永久 |
| RAG 研报 | ChromaDB | collection: `research_docs` | 永久 |
| User accounts | SQLite | table: `users` | 永久 |
| Broker bindings | SQLite | table: `broker_bindings` (信封加密) | 永久 |
| Audit log | SQLite | table: `broker_audit_log` | 永久 append-only |
| Realtime broker state | 进程内存 | `state[user_scope]` | 进程生命周期 |

---

## 10. 配置体系（三层）

### 10.1 环境变量（运行时配置）

| 变量 | 默认 | 用途 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | 主 LLM 密钥 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型选择 |
| `ANTHROPIC_API_KEY` | — | 仅 LoopDemo / 教学 demo |
| `APP_SECRET_KEY` | 临时生成 | starlette session middleware |
| `REDIS_URL` | — | 设了用 Redis，否则内存 |
| `USE_LANGGRAPH` | `0` | 切换主循环实现 |
| `BROKER_MODE` | `mock` | `mock` / `alpaca`（共享 paper） |
| `BROKER_KEK_v1`, `v2`, ... | — | 凭证加密主密钥 |
| `ALPACA_API_KEY/SECRET` | — | 仅供 diag 脚本，不进生产代码 |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | **必须是 paper** |
| `RISK_*` | 见 §8.2 | 风控阈值 |
| `ANSWER_CACHE_ENABLE` | `1` | 同 query+ctx 缓存 |
| `ANSWER_CACHE_TTL` | `300` | 答案缓存 TTL |
| `WARMUP` | `1` | 启动预热 LLM 连接 + embedding 模型 |
| `LG_TOKEN_FLUSH_MS` | `30` | LangGraph token 批阈值 |
| `LG_TOKEN_FLUSH_CHARS` | `40` | LangGraph token 批阈值 |
| `BROKER_TIGER_CACHE_TTL_SEC` | `30` | Tiger REST 限速缓存 |
| `GOOGLE_CLIENT_ID/SECRET`, `GITHUB_CLIENT_ID/SECRET` | — | OAuth 提供商凭证 |
| `EMAIL_LOGIN_ENABLED`, SMTP 配置 | — | 邮箱验证码登录 |

### 10.2 ADR（架构决策记录）

| ADR | 主题 | 关键约束 |
|---|---|---|
| `0001-broker-abstraction.md` | 多券商抽象 + 多租户凭证 | per-binding, 信封加密, env 不进生产路径 |
| `0002-live-trading.md` | 实盘下单五层闸 | live_orders_enabled flag, 中文 ack 短语 |

### 10.3 CLAUDE.md（开发宪法）

这是给开发者 / AI 看的**操作护栏**，不是被代码读的。关键不变量：
- 工具不能跨 LLM 边界抛异常
- 订单路径不能 `except: pass`
- 凭证不能进日志
- `live_orders_enabled` LLM 不能翻
- `agent_demo.py` 不要扩展（G1）
- 新工具必须有 mock 测试（G3）
- 协议层（DeepSeek/LangGraph/Anthropic）可换，工具层不动（G4）
- 鉴权只走 `auth/`（G5）

---

## 11. 测试 Harness

### 11.1 三层结构

```
tests/
├── auth/              OAuth flow, session signing, user CRUD
├── brokers/           BrokerAdapter mock-based 测试
│   ├── test_tiger_adapter.py
│   ├── test_tiger_quote.py
│   ├── test_tiger_push.py
│   ├── test_alpaca_adapter.py
│   ├── test_credentials_store.py
│   ├── test_risk_gate.py
│   ├── test_intent_store.py
│   ├── test_audit.py
│   ├── test_live_trading.py       (ADR-0002 五层)
│   ├── test_quote_fallback.py     (yfinance 兜底)
│   └── ...
└── e2e/
    └── test_smoke.py              (mock LLM,真实 dispatch)
```

### 11.2 网络隔离铁律

所有依赖外部 API 的测试必须 mock：

```python
def test_yf_get_brief(monkeypatch):
    import brokers.quote_fallback as qf
    monkeypatch.setattr(qf, "_YF_OK", True)

    class _FakeYf:
        @staticmethod
        def Ticker(_sym):
            return NS(fast_info=NS(last_price=213.5, previous_close=210.0, ...))

    monkeypatch.setattr(qf, "_yf", _FakeYf)
    b = qf.yf_get_brief("AAPL")
    assert b["latest_price"] == 213.5
```

### 11.3 e2e Canary 模式

`tests/e2e/test_smoke.py`：
- mock 掉 DeepSeek LLM 响应（返回固定 "调 black_scholes" 指令）
- 工具用真实 pure-math 实现（`black_scholes` 不需要网络）
- 端到端验证：`stream_quant_agent → dispatch_tool → 工具 → 返回 → 写回 messages`
- 零网络、零成本、零密钥

### 11.4 当前状态

214 passed (跨 unit + e2e)，作为合并门槛。

---

## 12. 部署与运维

### 12.1 启动命令

```bash
# 1. 准备
pip install -r requirements.txt
cp .env.example .env
# 填 DEEPSEEK_API_KEY + APP_SECRET_KEY + BROKER_KEK_v1

# 2. (可选) Redis
redis-server &
export REDIS_URL=redis://localhost:6379/0

# 3. 启动
python server.py                          # 默认 5000 端口
PORT=8001 python server.py                # macOS 避开 AirPlay
USE_LANGGRAPH=1 python server.py          # 切 LangGraph 主循环

# 4. (可选) 索引研报
python scripts/reindex_research_docs.py

# 5. 诊断
python scripts/test_alpaca_connect.py
python scripts/diag_alpaca_key.py
```

### 12.2 端点速查

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 主聊天 SPA |
| `/brokers` | GET | 券商绑定管理页 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | OpenAPI |
| `/api/sessions/{sid}/chat` | POST (SSE) | 聊天主入口 |
| `/api/sessions` | GET / POST / DELETE | 会话 CRUD |
| `/api/auth/login`, `/api/auth/logout`, `/api/auth/oauth/*` | — | 登录流 |
| `/api/orders/confirm` | POST | 订单确认（最关键闸） |
| `/api/orders/cancel/{id}` | POST | 撤单 |
| `/api/broker/bindings` | GET / POST / DELETE | 券商绑定 |
| `/api/broker/bindings/{id}/live-orders` | POST | ADR-0002 实盘开关 |
| `/api/broker/symbols/search` | GET | 股票模糊搜索（Tiger） |
| `/api/broker/symbols/brief` | GET | 行情快照（Tiger + 延迟兜底） |
| `/api/broker/symbols/option-expiries` | GET | 期权到期日（Tiger + yfinance 兜底） |
| `/api/broker/symbols/option-chain` | GET | 期权链（同上） |
| `/api/broker/stream` | GET (SSE) | Tiger 实时推送 |
| `/api/memory/profile` | GET / POST | 用户档案 |
| `/api/memory/episodic` | GET / POST / DELETE | 情景记忆 |
| `/api/rag/search` | GET | RAG 检索 |
| `/metrics/brokers` | GET | 进程内 broker 指标 |

### 12.3 监控指标（最少必看）

| 指标 | 阈值 | 来源 |
|---|---|---|
| LLM 请求 P99 延迟 | < 8s | server.py 自录 |
| 工具调用失败率 | < 5% | dispatch_tool 失败计数 |
| risk_gate 拒绝率 | 监控异常飙升 | check_order |
| 凭证使用速率 | 异常突增 = 攻击 | audit.log |
| **LLM-actor 非 use action** | **== 0**（任何 > 0 都是 critical） | broker_audit_log |
| 实盘订单数 | 监控 | confirm_order 成功路径 |

---

## 13. 已知技术债与演进路线（CLAUDE.md G1–G5）

### G1 · 单一 Agent 入口
- 现状：`agent_demo.py` 是早期教学版，不在产品调用链
- 目标：彻底移除，所有 demo 复用 `quant_agent.py` 工具

### G2 · quant_agent.py 拆分
- 现状：5234 行单文件，工具实现 + 主循环 + system prompt + 用户态注入混在一起
- 目标：拆到 < 500 行（只剩主循环 orchestration），工具按域拆到：
  ```
  tools/
  ├── market/      行情
  ├── factors/     因子
  ├── technical/   技术指标
  ├── options/     期权
  ├── risk/        风险/相关性
  ├── trading/     交易
  ├── memory/      记忆 CRUD
  └── registry.py  唯一 TOOL_REGISTRY 装配点
  ```

### G3 · 每个工具必须有 smoke test
- 现状：覆盖率不均；新加工具偶尔漏测
- 目标：CI 强制覆盖

### G4 · LLM Protocol 可换性
- 现状：DeepSeek/OpenAI + LangGraph 双协议，Anthropic native 未对接
- 目标：抽象 `LLMRunner` 接口，下面三个并列实现
- 当前已对：工具/schema/system prompt 由 protocol-agnostic 模块管理（quant_agent.py 已经做到）

### G5 · 鉴权统一走 auth/
- 现状：threadlocal + `_is_request_authenticated` 临时方案
- 目标：把 threadlocal 替换成 `contextvars` + 显式 FastAPI dependency 注入

### 其它待办（已规划）
- 长对话压缩：滑动窗口 + 早期 messages LLM 摘要 + 原文 episodic 镜像
- 多 binding 切换：`broker_label` 参数透出到前端
- LangGraph checkpointer：长任务断点续跑
- broker 子系统外的全局审计层：交易工具 / 用户操作的 append-only 日志

---

## 附录 A · 关键文件速查

| 想了解什么 | 看哪个文件 |
|---|---|
| 主循环原版 | `quant_agent.py:4972` `stream_quant_agent` |
| 主循环 LangGraph 版 | `quant_agent_lg.py:391` `stream_quant_agent_lg` |
| 工具调度 | `quant_agent.py:4734` `dispatch_tool` |
| 工具 schema | `quant_agent.py:3883` `TOOL_SCHEMAS` |
| 工具注册 | `quant_agent.py:3854` `TOOL_REGISTRY` |
| 鉴权白名单 | `quant_agent.py:3595` `TRADING_TOOLS` |
| 系统提示 | `quant_agent.py:4651` `SYSTEM_PROMPT` |
| 系统提示动态注入 | `quant_agent.py:4916` `_build_system_prompt` |
| 订单确认闸 | `server.py:802` `confirm_order` |
| 实盘开关 | `server.py:1504` `api_set_live_orders` |
| Risk gate | `brokers/risk_gate.py:150` `check_order` |
| 信封加密 | `brokers/credentials_store.py:140` `CredentialsStore` |
| Tiger 实时推送 | `brokers/tiger_push.py:62` `TigerPushClientWrapper` |
| yfinance 兜底 | `brokers/quote_fallback.py` |
| LangGraph 节点 | `quant_agent_lg.py:138-283` |
| LangGraph 边 | `quant_agent_lg.py:305-319` `_build_graph` |
| SSE 桥 | `quant_agent_lg.py:421-499` |

## 附录 B · ADR 快查

- [`0001-broker-abstraction.md`](adr/0001-broker-abstraction.md) — 券商抽象 / 凭证 / 多租户
- [`0002-live-trading.md`](adr/0002-live-trading.md) — 实盘下单五层闸门

## 附录 C · 学习路径建议

如果你是新工程师 onboarding，按这个顺序读：

1. `CLAUDE.md` — 先理解约束
2. `docs/adr/0001` + `0002` — 理解为什么是这套架构
3. 本文 §3-§4 — 整体架构 + 请求生命周期
4. `server.py:400-540` `chat()` — 入口的完整 hook
5. `quant_agent.py:4972` `stream_quant_agent` — 原版主循环
6. `quant_agent.py:4734-4774` `dispatch_tool` + `_to_openai_tools` — 工具桥接
7. `market_quote` (line 654) + `place_order_intent` — 看一只无副作用工具 + 一只有副作用工具
8. `server.py:802` `confirm_order` — 安全总闸
9. `brokers/registry.py` + `credentials_store.py` — 多租户凭证
10. **本文 §6 + `quant_agent_lg.py`** — 对比 LangGraph 实现
11. `tests/brokers/test_live_trading.py` + `test_tiger_quote.py` — 看测试 mock 怎么写

---

文档结束。维护责任：每次新增模块 / 改 ADR 时同步更新 §5（模块表）、§10.1（env）、§12.2（端点速查）。
