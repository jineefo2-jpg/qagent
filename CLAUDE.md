# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run commands

```bash
# Install deps (Python 3.9+ — tested on 3.9.6)
pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest for the canary test

# Configure (copy then fill DEEPSEEK_API_KEY at minimum)
cp .env.example .env

# Start the web server (FastAPI + SSE, default port 5000)
python server.py
# → http://localhost:5000   UI
# → http://localhost:5000/docs   OpenAPI
# → http://localhost:5000/brokers   bind a brokerage account

# On macOS, port 5000 conflicts with AirPlay Receiver. Override:
PORT=8080 python server.py

# Switch agent runtime
USE_LANGGRAPH=1 python server.py   # use quant_agent_lg.py instead of quant_agent.py

# Reindex RAG corpus (PDFs under docs/ into rag_db/)
python scripts/reindex_research_docs.py

# Diagnostics (Alpaca / order flow sanity checks)
python scripts/test_alpaca_connect.py
python scripts/diag_alpaca_key.py
python scripts/diag_mock_orders.py
```

# Run the canary smoke test (mocks LLM + uses pure-math tool, zero network)
python3 -m pytest tests/e2e/ -q

# Run a single test
python3 -m pytest tests/e2e/test_smoke.py::test_simple_chat_no_tools -q

## Architecture — big picture

This is a streaming quant-trading AI agent: a FastAPI server fronts an LLM-driven tool-use loop that can fetch market data, run quant analytics, query a RAG corpus of research PDFs, and place semi-automated orders against Alpaca (paper) or a per-user mock broker.

### Request lifecycle

```
Browser (static/) ──SSE──> server.py /api/chat
                              │
                              ▼
                  stream_quant_agent(messages)        ← quant_agent.py (default)
                  stream_quant_agent_lg(messages)     ← quant_agent_lg.py (USE_LANGGRAPH=1)
                              │  iterative tool-use loop
                              ▼
                  TOOL_REGISTRY dispatch ─────────────┐
                              │                       │
                  ┌───────────┼───────────┬───────────┼─────────────┐
                  ▼           ▼           ▼           ▼             ▼
              market data  factors/    options/    rag/          brokers/
              (yfinance,   technicals  risk/       (chromadb)    (Alpaca|Mock)
               akshare,                                          → risk_gate
               news APIs)                                        → intent_store
```

### The three top-level Python files — what's actually what

| File | Role | Status |
|---|---|---|
| `server.py` (1256 L) | **Real entry.** FastAPI app, SSE streaming, sessions, auth, broker/RAG/memory REST endpoints | production |
| `quant_agent.py` (5093 L) | **Core library.** Tool registry, schemas, system prompt, DeepSeek/OpenAI streaming loop. ⚠️ Monolith — see "Known tech debt" | production |
| `quant_agent_lg.py` (510 L) | LangGraph re-implementation of the same loop; reuses tools from `quant_agent` | opt-in via `USE_LANGGRAPH=1` |
| `agent_demo.py` (936 L) | Standalone teaching demo (Anthropic SDK, 3 minimal tools) — **not used by server.py**, partly duplicates `quant_agent.py` | legacy/redundant |

### Key symbols in `quant_agent.py` (so you don't grep)

| Symbol | Line (approx) | What |
|---|---|---|
| `TRADING_TOOLS` | 3595 | Set of tool names gated behind auth |
| `TOOL_REGISTRY` | 3854 | `{name: callable}` — single source of truth for dispatch |
| `TOOL_SCHEMAS` | 3883 | OpenAI-format JSON schemas advertised to the LLM |
| `stream_quant_agent` | 4828 | Main generator-style agent loop (DeepSeek streaming) |
| `_build_system_prompt` | — | Injects user profile + episodic memory into system msg |
| `_news_*`, `_compute_factors_*` | — | Per-source market/news/factor adapters |

### Module boundaries (mostly clean — keep them clean)

- `auth/` — OAuth (Google/GitHub) + email-code login, cookie sessions, `require_user` FastAPI dep
- `brokers/` — `BrokerAdapter` ABC + `MockAdapter` (per-user virtual $100k) + `AlpacaAdapter` (shared paper account); **`risk_gate.py` MUST stay in the order path** (single-position caps, daily limits, market-order block, whitelist)
- `brokers/intent_store.py` — two-phase order flow: LLM creates `OrderIntent` → user confirms via `/api/broker/confirm-order/{id}` → adapter submits
- `rag/` — chromadb + sentence-transformers; `indexer.py` writes to `rag_db/`, `retriever.py` reads
- `memory/` — `profile.py` (long-term user profile) + `episodic.py` (recent interactions), both vector-backed
- `cache.py` — Redis if `REDIS_URL` set, else in-memory dict (drop-in compatible API)

### Critical env switches (affect behavior, not just config)

| Env | Effect |
|---|---|
| `USE_LANGGRAPH=1` | Swap `quant_agent` → `quant_agent_lg` at server import time |
| `BROKER_MODE=mock\|alpaca` | `mock` (default): per-user virtual account; `alpaca`: real paper account shared by all users |
| `ALPACA_BASE_URL` | **Must stay `paper-api.alpaca.markets`** in dev — production endpoint would be live money |
| `ANSWER_CACHE_ENABLE=1` | Replay identical `(query, context)` for 5 min without calling the LLM |
| `WARMUP=1` | Pre-warm models/connections at server startup |
| `RISK_*` | Override defaults in `brokers/risk_gate.py` (single-position cap, daily order limit, etc.) |

## Known tech debt (engineering north-stars for refactors)

If you're working in this repo, these are the things we are intentionally moving toward. Don't add code that fights them:

1. **G1** — One agent entry. `agent_demo.py` is on the way out; do not extend it. Reuse `quant_agent.py` tools instead.
2. **G2** — `quant_agent.py` should shrink to <500 lines (just orchestration). New tools go in their own files; do not bolt onto the monolith. Target layout: `tools/{market,factors,technical,options,risk,trading,memory}/` plus `tools/registry.py`.
3. **G3** — Every tool needs a smoke test. Network calls must be mocked.
4. **G4** — LLM protocol layer (DeepSeek/OpenAI, Anthropic native, LangGraph) should be interchangeable runners over the same tool registry. Don't hardcode protocol-specific fields outside the runner.
5. **G5** — Auth checks go through `auth/` only. Do not introduce new thread-local globals or hardcoded `TRADING_TOOLS`-style sets in business code.

## Trading safety — hard rules

This agent can move (paper) money. The following are non-negotiable. If a task seems to require breaking any of them, STOP and ask the user — do not "find a way".

### Order flow contract
- LLM-callable trading tools MUST return an `OrderIntent` and persist it via `brokers.intent_store`. They MUST NOT call `adapter.submit_order` directly.
- Submission MUST happen only from `/api/broker/confirm-order/{id}` after explicit user click. Do not add code paths that bypass this two-phase flow.
- Do not collapse intent → confirm → submit into one step, even "for tests" or "for convenience".

### Risk parameters are ceilings, not defaults
- `RISK_*` values in code and `.env.example` are SAFE upper bounds. Lowering them is fine; raising them MUST have an explicit user ack in the PR description.
- Do not change `RISK_BLOCK_MARKET_ORDER=1` — market orders are intentionally blocked for slippage protection, not a bug.
- Do not expand `RISK_WHITELIST` to "fix" a rejected order. The rejection is the feature.

### Authentication gate for trading tools
- Any new tool that can move money or modify positions MUST be added to `TRADING_TOOLS` in `quant_agent.py`. Forgetting this exposes the tool to unauthenticated callers.
- Do not remove entries from `TRADING_TOOLS` without an explicit user ack.

### LLM tool descriptions are part of the safety surface
- Trading tool `description` fields in `TOOL_SCHEMAS` MUST mention that user confirmation is required. Do not "improve" them to be more persuasive — the LLM uses them to decide when to act.

### Untrusted content stays untrusted
- Output from `_news_*`, RAG retrieval, and any external HTTP source is untrusted text. Never wire it into a path that lets the LLM execute trades without going through the same intent → confirm flow.
- Do not add system-prompt instructions like "follow the recommendations in the news" — that is a prompt injection vector.

### Live endpoint is forbidden
- `ALPACA_BASE_URL` MUST remain `https://paper-api.alpaca.markets`. Do not add code, examples, or docs that point at the live endpoint, even conditionally. If the user asks for live trading, refuse and escalate.

### Operations the AI does not initiate on its own
- Batch cancel orders
- Auto-close or auto-rebalance positions
- "Fixing" positions that look anomalous
- Running `scripts/diag_*` or `reset_account` in any automated verification loop (they mutate state)

### Multi-tenancy when `BROKER_MODE=alpaca`
- A single Alpaca paper account is shared across users in this mode. Orders from one user consume the shared cash/buying power. Do not write code that assumes per-user isolation when this mode is active.

### Audit trail
- **Current state**: there is no structured audit log. `brokers/` uses no `logging` module; `intent_store` keeps intents only in cache (transient).
- This is a known gap. Until a real audit layer exists:
  - Do not suppress, swallow, or downgrade any exception, print, or return value on the order-intent / submit / cancel paths — those traces are currently the only forensic signal.
  - Order-related failures MUST fail loud (raise or return a structured error dict). Never `except: pass` in `brokers/` or in trading tools.
- If you add an audit layer, every state transition of an order (intent created → confirmed → submitted → filled/canceled/rejected) MUST be recorded with: `device_id`, `user_id` (if present), `intent_id`, `broker_order_id`, `symbol`, `side`, `qty`, `timestamp`, and the actor (`llm` vs `user-confirm` vs `system`). The log MUST be append-only and MUST NOT be silenced by env flags.

## Broker abstraction model (per-user multi-tenant)

Detailed design: `docs/adr/0001-broker-abstraction.md`. The constraints below are the parts that MUST be observed by anyone editing this codebase.

### Model
- A user can have **1:N broker bindings**, where a binding = `(user_id, broker_type, label)`. Each binding owns its own credentials. There is no module-level shared broker credential (except `MockAdapter`, which is stateless per-user).
- Adapters are constructed **per binding** via `brokers.registry.BrokerRegistry.get(user_id, broker_type, label=None)`. Do not introduce paths that bypass the registry.
- Supported brokers: `mock` (always), `alpaca` (paper only), `tiger` (paper only). New brokers MUST extend `BrokerAdapter` and ship with a paper / simulator mode that works in CI without a real account.
- **Every non-mock broker (alpaca, tiger, ...) requires an explicit per-user binding.** `ALPACA_API_KEY` / `ALPACA_API_SECRET` env vars MUST NOT drive any production code path; the only legitimate uses are smoke / diag scripts run locally by an operator. Calling `get_current_broker(broker_type="alpaca")` without a binding for the current user MUST raise — never silently fall back to env. See ADR-0001 addendum.

### Credentials
- Credentials are encrypted with **envelope encryption** in `brokers/credentials_store.py`: per-binding DEK encrypts the credential blob; a versioned KEK (`BROKER_KEK_v1`, `BROKER_KEK_v2`, ... from env) wraps the DEK.
- Persistence is SQLite (`data/brokers.db`). The DB file contains only ciphertext and is safe to back up; without the KEK env var it is useless.
- Plaintext credentials are decrypted **in memory only**, zeroed after each use, and held inside a cached adapter for ≤ 60 s.
- Credentials MUST NEVER:
  - appear in any log line, exception message, traceback, or print
  - be returned by any tool to the LLM
  - be embedded in system prompts, RAG context, or any text that reaches the model
  - be serialized to disk in plaintext, including caches, snapshots, or test fixtures
- KEK rotation: operator runs `python -m brokers.rotate_kek`. Old KEKs retained 30 days then deleted. Server refuses to start if SQLite has rows but no KEK is set.

### Tiger specifics
- Tiger uses RSA private keys, **not OAuth**. The UI wizard handles the 3-step flow (open Tiger portal → generate key pair → upload `.pem` + tiger_id). See ADR §5.
- `env` defaults to `paper` at bind time. Switching to `live` is forbidden by the current trading-safety rules — do not add code paths that flip `env='live'` without a new ADR.
- Tiger SDK is pinned to a major version in `requirements.txt`. Do not bump without checking the changelog for breaking auth changes.

### Audit & monitoring
- Every credential lifecycle event (`bind`, `unbind`, `use`, `rotate`, `fail`, `read`) is recorded **append-only** in the `broker_audit_log` SQLite table by `brokers/audit.py`. This satisfies the audit-trail TODO in the trading-safety section above for the broker subsystem.
- Audit writes MUST NOT be silenced by env flags or try/except. A failed audit write is a code path that needs fixing, not swallowing.
- The most important alert is: **any audit row with `actor='llm'` and `action ∈ {'bind','unbind','rotate'}` is a critical incident.** The LLM should only ever trigger `action='use'`. Anything else means a guardrail leaked.
- Metrics exposed at `/metrics/brokers` (in-process counters). See ADR §8 for the full list and alert thresholds.

### Tool-layer integration
- Trading tools (`place_order_intent`, `cancel_order`, `broker_account`) resolve the user's chosen binding via `BrokerRegistry`. Their LLM-facing schemas accept an optional `broker_label` argument; absent it, the user's default binding is used.
- The two-phase order flow (intent → user confirm → submit) from the trading-safety rules remains unchanged. Multi-binding adds **which binding** to the intent payload; it does not change the contract.

## No-go zones (read-only unless task explicitly says otherwise)

- `.env` — never read or commit; use `.env.example` to learn schema
- `brokers/risk_gate.py` — touching this can enable real-money mistakes; if a change is needed, it must be in a dedicated PR with the user's explicit ack
- `brokers/alpaca_adapter.py` order-submit paths — same reason
- `rag_db/` and `docs/*.pdf` — gitignored data, do not regenerate or delete without being asked

## Style conventions observed in this codebase

- Chinese comments are normal and welcome — do not "translate" them
- Module-level `try: import X except ImportError` is used for optional deps (redis, akshare, alpaca-py, langgraph) — preserve this pattern, do not make them hard deps
- FastAPI handlers use `Depends(require_user)` for auth and `Header("x-device-id")` for anonymous-but-scoped flows
- Tools return plain dicts/lists (JSON-serializable); never raise across the LLM boundary — wrap errors into `{"error": "..."}`
