-- broker credential store + audit log (ADR-0001 §3)
-- All statements are idempotent so _db.init() can be called repeatedly.

CREATE TABLE IF NOT EXISTS broker_bindings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL,
    broker_type          TEXT NOT NULL,
    label                TEXT NOT NULL,
    env                  TEXT NOT NULL DEFAULT 'paper'
                              CHECK (env IN ('paper','live')),
    encrypted_credential BLOB NOT NULL,
    dek_wrapped          BLOB NOT NULL,
    kek_version          INTEGER NOT NULL,
    created_at           INTEGER NOT NULL,
    last_used_at         INTEGER,
    UNIQUE(user_id, broker_type, label)
);

CREATE INDEX IF NOT EXISTS idx_bindings_user
    ON broker_bindings(user_id);

CREATE TABLE IF NOT EXISTS broker_audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    user_id     TEXT,
    binding_id  INTEGER,
    actor       TEXT NOT NULL
                    CHECK (actor IN ('user','llm','system','rotation')),
    action      TEXT NOT NULL
                    CHECK (action IN ('bind','unbind','use','rotate','fail','read')),
    detail      TEXT,
    success     INTEGER NOT NULL CHECK (success IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_audit_user_ts
    ON broker_audit_log(user_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_audit_ts
    ON broker_audit_log(ts DESC);
