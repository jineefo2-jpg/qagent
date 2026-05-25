# ADR-0002 · Live (Real-Money) Trading — Multi-Layer Safety Model

- **Status**: Accepted · 2026-05-26
- **Phase**: X7 in the harness-coding roadmap
- **Amends**: ADR-0001 §1 "default broker environment stays paper" — now
  live is allowed under strict opt-in.

## 1. Context

Until this ADR, the codebase enforced "paper only" as a hard rule:
- Bind API rejected `env='live'`
- Tiger account number prefix wasn't inspected at bind
- Result: a user who entered a real (D-prefix) Tiger account got it
  stored as `env='paper'`, while the SDK still routed orders to the real
  account. **A user-bound real account would execute real-money orders
  with NO safety distinction from paper.**

This was a latent safety hole, not a working safeguard. This ADR:
1. Acknowledges the hole.
2. Replaces the "always reject live" rule with an **explicit, multi-layer
   opt-in model** that bears the safety burden honestly.

## 2. Decision Drivers

- A user with a real brokerage account MUST be able to monitor it
  through this app (positions, orders, balance).
- A user must be able to **trade live** if they explicitly choose to,
  through the same risk-gated two-phase flow paper uses — not via a
  parallel "unsafe path".
- The path from "user binds a real account" to "real money moves" must
  cross multiple, independent guards. Failing any one of them must
  prevent execution.
- The LLM (via tool calls) MUST NOT be able to flip a binding from
  read-only to trading-enabled. Only the human user via UI can.

## 3. Decision — Multi-Layer Safety Model

A live order requires **all five** of the following to be true. Failing any
one MUST stop execution and surface a clear error.

### Layer 1 — Binding `env='live'`
- The credentials are for a real brokerage account.
- The user picks `env` explicitly at bind time (UI radio, paper default).
- Default after upgrade from earlier schema: `env='paper'` for all rows
  (existing rows remain safe; no automatic re-classification).

### Layer 2 — Per-binding `live_orders_enabled=1`
- New column on `broker_bindings`. Default `0` (false).
- Even with `env='live'`, the binding is READ-ONLY until this flag is
  flipped to `1` via the UI by the authenticated user.
- Flipping requires:
  - User authenticated via `require_user`
  - Explicit POST to a separate endpoint
    `/api/broker/bindings/{id}/live-orders`
  - Confirmation acknowledgement in the request body
    (`acknowledge: "I understand this will use real money"`)
- The LLM has no tool that can set this flag. Tools resolve adapters
  via `BrokerRegistry`; setting the flag is a UI / API surface, not a
  tool surface.
- An audit row (`actor='user', action='use'`, detail tagged 'enable_live')
  is written on every flip.

### Layer 3 — `risk_gate` rules apply UNCHANGED
- The same single-position-cap / daily-order-cap / market-order-block /
  whitelist rules that protect paper apply to live.
- A future ADR may tighten limits *specifically* for live (e.g. lower
  single-position cap, tighter whitelist) — but the current limits are
  the floor, not the ceiling.
- `RISK_BLOCK_MARKET_ORDER=1` stays the default for live, same as paper.

### Layer 4 — Two-phase confirmation
- LLM generates `OrderIntent` only. Same flow as paper.
- User must click "确认下单" in the UI.
- The confirm modal for live orders MUST render a red `⚠️ 实盘 - 真金白银`
  banner so the user cannot mistake the action.

### Layer 5 — Server-side env recheck at confirm
- `/api/broker/confirm-order/{id}` resolves the binding for the user.
- If `binding.env='live'` AND `live_orders_enabled=0` → reject with HTTP
  422 and a clear error message. The intent stays in the store so the
  user can re-attempt after enabling live.
- If both conditions met → submit through the adapter, audit log
  `actor='user', action='use'` with extra detail tagging 'live'.

## 4. Out of Scope

- **Alpaca live trading**: Alpaca's `base_url` distinction is preserved.
  `ALPACA_BASE_URL` stays `paper-api.alpaca.markets`. Enabling Alpaca
  live requires a separate ADR (URL change has its own safety surface).
- **Auto-detection of live by account number**: We considered inferring
  `env='live'` from Tiger's D-prefix vs U-prefix. Rejected: account
  number formats vary across Tiger products and may change. Better to
  ask the user explicitly than be subtly wrong.
- **Tightened risk_gate for live**: Stays at the existing thresholds for
  this ADR. Tightening warrants its own review with concrete data.

## 5. Migration

- SQLite schema: ADD COLUMN `live_orders_enabled INTEGER NOT NULL DEFAULT 0`.
- Run idempotently on each `_db.init()` — check `PRAGMA table_info`
  first.
- Existing rows (all `env='paper'`) are unaffected and stay paper.

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Operator runs `UPDATE broker_bindings SET live_orders_enabled=1` directly in SQLite to bypass UI | DB has no constraint against it — but every order goes through audit. A live order without a matching `enable_live` audit row is a red flag for forensics. |
| User sets flag, walks away, malicious actor on shared computer issues live orders via LLM | Same risk as paper (the LLM session is authenticated as the user). UI session timeout exists. Recommendation: add `live_orders_enabled` auto-expiry (e.g. 1 hour) — punt to a follow-up ADR. |
| Schema migration fails on prod, server starts without the column | `_db.init()` fails loud; refuses to serve requests until DB is correct. |
| Cross-binding leak: user A's `live_orders_enabled` somehow applies to user B | The flag is on the row, which is filtered by `user_id` in every read. Pinned by an existing isolation test (`test_isolation_*`) — we add one more specifically for the live flag. |

## 7. References

- ADR-0001 — broker abstraction
- CLAUDE.md → "Trading safety — hard rules" (this ADR amends it)
- Layer-4 UI: order confirm modal in `static/index.html`
- Layer-5 server: `confirm_order` in `server.py`
