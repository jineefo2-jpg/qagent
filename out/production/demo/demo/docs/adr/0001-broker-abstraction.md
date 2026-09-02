# ADR-0001 · Broker Abstraction & Per-User Credential Storage

- **Status**: Accepted · 2026-05-24
- **Phase**: X1 in the harness-coding roadmap (precedes the `quant_agent.py` decomposition in Phase C)
- **Supersedes**: the implicit "module-level shared credentials" pattern currently in `brokers/`

## 1. Context & Decision Drivers

The codebase currently supports two broker modes (`BROKER_MODE=mock|alpaca`) using
**module-level credentials** — every user shares the same Alpaca paper account.
The new requirement is to let each end-user **link their own brokerage account**
(initially Tiger Brokers / 老虎证券) so trades route to the right wallet.

Constraints, in order of priority:

1. **CLAUDE.md trading safety rules are unchanged.** Default broker environment
   stays paper. Live trading remains forbidden until a future, separate ADR.
2. **Target audience**: Chinese retail users → Tiger and Futu first. Alpaca
   stays as a secondary option and as the default in tests.
3. **Deployment scale**: small SaaS, dozens to low hundreds of users, single
   self-hosted instance. Not enterprise; not single-user-desktop.
4. **Credential class**: Tiger uses an **RSA private key** (PEM), not an OAuth
   token. A leaked private key is equivalent to a leaked login — credential
   storage must reflect that severity.

## 2. Decision

- Adapters are constructed **per binding**, where a binding = `(user_id, broker_type, label)`. Module-level shared adapters are removed except for `MockAdapter` (which remains stateless per-user via cache).
- Credentials are stored with **envelope encryption**: a per-binding Data
  Encryption Key (DEK) encrypts the credential blob; a global Key Encryption
  Key (KEK), versioned and rotatable, encrypts the DEK.
- Persistence: **SQLite** (file-based, ACID, zero ops) for the source of truth
  + **Redis** for hot-path cache of decrypted adapter instances (short TTL,
  invalidated on unbind/rotate).
- Relationship: **users 1:N broker bindings**. A user may bind multiple
  accounts at the same broker (disambiguated by user-supplied `label`).
- Audit: every credential lifecycle event (bind, unbind, use, rotate, failed
  auth) is written **append-only** to a SQLite audit table. Audit cannot be
  silenced by env flags.
- LLM never sees raw credentials. Tool calls resolve `user_id` → adapter
  internally; the LLM only receives broker call results (already JSON-safe).

## 3. Data Model

### SQLite schema

```sql
CREATE TABLE broker_bindings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL,                  -- from auth/
    broker_type          TEXT NOT NULL,                  -- 'tiger' | 'alpaca' | 'mock' | 'futu'
    label                TEXT NOT NULL,                  -- user nickname, e.g. "我的主账户"
    env                  TEXT NOT NULL CHECK (env IN ('paper','live')) DEFAULT 'paper',
    encrypted_credential BLOB NOT NULL,                  -- ciphertext of credential JSON
    dek_wrapped          BLOB NOT NULL,                  -- DEK wrapped by KEK
    kek_version          INTEGER NOT NULL,               -- which KEK wrapped the DEK
    created_at           INTEGER NOT NULL,
    last_used_at         INTEGER,
    UNIQUE(user_id, broker_type, label)
);

CREATE TABLE broker_audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    user_id     TEXT,
    binding_id  INTEGER,                                  -- nullable: failed binds have no id yet
    actor       TEXT NOT NULL CHECK (actor IN ('user','llm','system','rotation')),
    action      TEXT NOT NULL CHECK (action IN ('bind','unbind','use','rotate','fail','read')),
    detail      TEXT,                                     -- short human-readable detail; no secrets
    success     INTEGER NOT NULL CHECK (success IN (0,1))
);
CREATE INDEX idx_audit_user_ts ON broker_audit_log(user_id, ts DESC);
CREATE INDEX idx_audit_ts ON broker_audit_log(ts DESC);
```

File location: `data/brokers.db` (new `data/` directory, gitignored).

### Credential JSON shapes (per `broker_type`, before encryption)

```jsonc
// tiger
{ "tiger_id": "12345678", "private_key_pem": "-----BEGIN ...-----", "account": "u9999999", "account_type": "GLOBAL" }

// alpaca (paper)
{ "api_key": "PK...", "api_secret": "...", "base_url": "https://paper-api.alpaca.markets" }

// mock
{ "initial_cash": 100000 }
```

## 4. Credential Lifecycle

| Event | Flow |
|---|---|
| **Bind** | POST `/api/broker/bindings` → server validates by calling broker's "get account" → encrypts → inserts row → audit `bind/success`. On validation failure: audit `bind/fail`, no row written. |
| **Use** | Tool layer asks `BrokerRegistry.get(user_id, broker_type, label?)` → cache hit returns hot adapter; cache miss reads SQLite, decrypts in-memory, constructs adapter, caches with TTL (60 s). Audit `read/success` debounced (1 per binding per hour). |
| **Rotate KEK** | Operator runs `python -m brokers.rotate_kek` → creates new KEK version → re-wraps every DEK → bumps `kek_version` → audit `rotate/success` per binding. Old KEK retained for 30 days then deleted. |
| **Unbind** | DELETE `/api/broker/bindings/{id}` → row deleted → cache invalidated → audit `unbind/success`. |
| **Failed auth at use** | Adapter raises `BrokerAuthError` → cache invalidated → audit `fail` → return error to LLM as `{"success": false, "error_type": "broker_auth_failed"}` (no token in message). |

**KEK storage**: env var `BROKER_KEK_v1`, `BROKER_KEK_v2`, ... (multiple
versions can coexist during rotation). Generated by `openssl rand -base64 32`.
Operator MUST set at least one; server refuses to start if none present and
SQLite has rows.

**Plaintext lifetime in memory**: decrypted credential is kept inside the
adapter instance only for the duration of one tool call, then explicitly
zeroed via `bytearray` mutation. Cached adapters hold the decrypted bytes;
cache TTL 60 s.

## 5. Tiger Integration Specifics

Tiger does **not** offer OAuth. Authentication is RSA-key based. The user must
generate a key pair on Tiger's OpenAPI portal, then upload the private key to
our app. We optimize the UX to make this as smooth as possible.

### UX · 3-step wizard

```
[Settings → Linked Brokers → Add Tiger]
  │
  ├─ Step 1: Open Tiger OpenAPI portal (link + screenshot)
  │          https://www.itigerup.com/openapi/
  │          → user applies for OpenAPI access (one-time)
  │
  ├─ Step 2: Generate key pair on Tiger portal
  │          → user downloads private_key.pem
  │          → user copies tiger_id
  │          (clear screenshots + 1-min video link)
  │
  └─ Step 3: Upload to QuantAgent
             - Tiger ID:          [input]
             - Private key file:  [file picker, .pem only]
             - Account:           [dropdown: paper / global / standard]
             - Label:             [input, e.g. "我的主账户"]
             [Test connection]   ← calls Tiger get_account; shows balance on success
             [Bind]              ← only enabled after test passes
```

### Defaults & guardrails

- **`env` defaults to `paper`** at binding time. Switching to `live` is a
  separate operation that will require a future ADR + additional confirmation
  (see CLAUDE.md trading safety rules — current rules forbid live).
- The Tiger SDK call to validate is `TradeClient.get_account()` — read-only,
  no order side effects.
- Account type dropdown: `paper` ("环球账户模拟"), `global` ("环球账户"),
  `standard` ("综合账户"). Only `paper` is selectable while live trading is
  disabled.

### Tiger SDK version pin

Add to `requirements.txt`:
```
tigeropen>=3.2.0,<4.0.0
```

(Tiger's API has had occasional breaking changes; we pin a major to avoid
silent breakage.)

## 6. Alternatives Considered

| Alternative | Why rejected |
|---|---|
| **AWS KMS / HashiCorp Vault** | Operationally heavy for a single-host SaaS; introduces a cross-service dependency and a paid AWS bill for low user count. Revisit if scale grows. |
| **User-password-derived encryption** (PBKDF2 → key) | Loses access if user forgets password; incompatible with autonomous agent paths that need decrypt without prompting the user. |
| **Plaintext API keys in env vars** (current model) | Violates multi-tenancy; one user's key reachable from another user's session via any global-variable bug. |
| **Postgres** | Better for scale but adds a service to deploy. SQLite is sufficient for ≤ low-thousands rows and zero-ops; can swap if we cross that threshold. |
| **OAuth-only** | Tiger doesn't support it. Forcing OAuth would exclude our primary user base. |

## 7. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| KEK leak (env compromised) | Versioned KEKs + rotation playbook in `docs/runbooks/`. Operator alert if `kek_version > 1` but old KEK is still readable after 30 days. |
| Process memory dump | Decrypted bytes zeroed after use; adapter cache TTL ≤ 60 s. We accept residual risk — full mitigation needs hardware isolation, out of scope. |
| LLM prompt injection asking for credentials | Tool layer never returns or logs credentials. System prompt explicitly says "credentials are never visible to you" (per CLAUDE.md). |
| SQLite file leak (backup, snapshot) | DB file contains only ciphertext + DEK ciphertext; useless without KEK env. Backups can be unencrypted. |
| User uploads wrong key, locks themselves out | "Test connection" gate on bind; failed binds don't write a row. Unbind always available. |

**Resolved open questions** (decided in review):
- ✅ Persistence: SQLite (not Redis-only — durability required for credentials)
- ✅ Multiplicity: 1:N (a user may bind multiple accounts at the same broker)

**Still open** (defer to later ADR):
- Live trading enablement flow (separate ADR; current rules forbid)
- Futu integration model (FutuOpenD gateway is a different topology; defer)

## 8. Monitoring & Alerting

Small-SaaS-appropriate stack — no Prometheus/Grafana dependency required;
extensible if needed later.

### Structured logs (always on)

A dedicated `broker_audit` logger writes one-line JSON to stdout for every
audit table row. Format:

```json
{"ts":1716579600,"user_id":"u_42","binding_id":7,"actor":"llm","action":"use","broker":"tiger","success":true}
```

These lines double as the persistent audit trail (SQLite) and as the stream
that monitoring tools can tail.

### Metrics (in-process counter, exposed at `/metrics/brokers`)

| Metric | Type | Purpose |
|---|---|---|
| `broker_bind_total{broker, result}` | counter | bind success/failure by broker |
| `broker_use_total{broker, result}` | counter | use success/failure by broker |
| `broker_auth_fail_total{broker}` | counter | drives the alert below |
| `broker_kek_age_days` | gauge | days since current KEK was set |
| `broker_active_bindings` | gauge | count by broker |

Implementation: simple dict in process, snapshot read by endpoint. No new
dependency. If team adopts Prometheus later, swap to `prometheus_client`.

### Alerts (driven by existing `scripts/alert_worker.py` pattern)

| Condition | Channel | Severity |
|---|---|---|
| `broker_auth_fail_total{broker=x}` rate > 5/min for 5 min | email (SMTP already configured) | warn |
| Any `actor='llm'` with `action='bind'` or `action='unbind'` | email + log | **critical** — LLM should never touch credential lifecycle |
| `broker_kek_age_days > 90` | email | warn |
| Process restart with rows present but no KEK env var | email + refuse to start | **critical** |
| Audit log write failure | log loud + retry | warn |

The "LLM touched credentials" alert is the key safety signal — by design, the
LLM should only ever trigger `action='use'`. Anything else means a code path
escaped the guardrails.

## 9. Migration Plan

Maps to Phase X in the roadmap:

- **X2**: Refactor `BrokerAdapter` ABC and existing `MockAdapter` / `AlpacaAdapter` to per-instance auth. Update tool layer (`place_order_intent`, `broker_account`, `cancel_order`) to resolve via `BrokerRegistry`. Canary smoke test must remain green.
- **X3**: Implement `brokers/credentials_store.py` (Fernet envelope), `brokers/registry.py` (cache + lookup), `brokers/audit.py` (SQLite append). Add `data/brokers.db` migration script. Tests for crypto round-trip, KEK rotation, audit append.
- **X4**: Implement `brokers/tiger_adapter.py`. SDK pinned. Tests cover the paper-environment path. Real Tiger account testing falls to the user (you).
- **X5**: Server endpoints (`POST /api/broker/bindings`, `DELETE /api/broker/bindings/{id}`, `GET /api/broker/bindings`, `POST /api/broker/bindings/{id}/test`) + UI wizard.

Each step is its own commit, behind a worktree. Canary stays green at every commit.

## 10. References

- CLAUDE.md → "Trading safety — hard rules" (immutable during Phase X)
- Tiger OpenAPI docs: https://quant.itigerup.com/
- Tiger Python SDK: https://github.com/tigerfintech/openapi-python-sdk
- `cryptography.fernet` (stdlib of choice for symmetric encryption)

---

## Addendum · 2026-05-24 · Alpaca is also per-user

**Decision**: every non-mock broker (Alpaca, Tiger, future) requires an
explicit per-user binding through the UI / API. The legacy
`BROKER_MODE=alpaca` shared-paper-account mode is **removed**.

### What changes vs the original ADR
- §2 originally said "Module-level shared adapters are removed except for
  `MockAdapter`". This stays true — but the original text could be read as
  permitting an env-key fallback for Alpaca. **It does not.** As of X3:
  - `ALPACA_API_KEY` / `ALPACA_API_SECRET` env vars MUST NOT drive any
    production code path. They may be referenced only by smoke / diag
    scripts that the operator runs locally.
  - Calling `get_current_broker(broker_type="alpaca")` without a binding
    in `credentials_store` MUST raise `BrokerError("requires user binding")`.
  - The error surfaces to the LLM as
    `{"success": false, "error_type": "no_broker_binding"}` so the agent
    can prompt the user toward the UI bind flow.
- §5 (Tiger specifics) describes the 3-step upload wizard. The same UX
  pattern applies to Alpaca (simpler — just `api_key` + `api_secret`).

### Migration impact
- Existing operators who relied on `BROKER_MODE=alpaca` with env keys
  must now run the bind flow once per user. The `.env.example` Alpaca
  block stays for transparency but its docstring will note
  "deprecated, use UI bind instead".

### Why this addendum (not ADR-0002)
Original ADR is < 24 h old and unreleased. This is a clarification of
the same decision, not a new one. ADR-0002 will land when something
genuinely overturns ADR-0001 (e.g. when we eventually enable live trading
behind a separate flag).
