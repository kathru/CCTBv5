-- CCTBv5 — Database Schema
-- Run once on first deploy (or via migration tool)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Orders ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    client_order_id     TEXT PRIMARY KEY,
    exchange_order_id   TEXT,
    symbol              TEXT        NOT NULL,
    side                TEXT        NOT NULL,
    order_type          TEXT        NOT NULL,
    mode                TEXT        NOT NULL,
    status              TEXT        NOT NULL,
    quantity            NUMERIC     NOT NULL,
    filled_quantity     NUMERIC     NOT NULL DEFAULT 0,
    avg_fill_price      NUMERIC     NOT NULL DEFAULT 0,
    limit_price         NUMERIC,
    stop_loss           NUMERIC,
    take_profit         NUMERIC,
    fees_paid           NUMERIC     NOT NULL DEFAULT 0,
    strategy_id         TEXT        NOT NULL,
    signal_id           TEXT        NOT NULL,
    retry_count         INTEGER     NOT NULL DEFAULT 0,
    last_error          TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    submitted_at        TIMESTAMPTZ,
    filled_at           TIMESTAMPTZ,
    cancelled_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol   ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status   ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_created  ON orders(created_at DESC);

-- ── Fills ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fills (
    fill_id             TEXT PRIMARY KEY,
    client_order_id     TEXT        NOT NULL REFERENCES orders(client_order_id),
    exchange_order_id   TEXT        NOT NULL,
    symbol              TEXT        NOT NULL,
    side                TEXT        NOT NULL,
    quantity            NUMERIC     NOT NULL,
    price               NUMERIC     NOT NULL,
    fee                 NUMERIC     NOT NULL,
    fee_currency        TEXT        NOT NULL,
    is_maker            BOOLEAN     NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fills_order   ON fills(client_order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol  ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_time    ON fills(timestamp DESC);

-- ── Positions ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol              TEXT        NOT NULL,
    side                TEXT        NOT NULL,
    status              TEXT        NOT NULL,
    strategy_id         TEXT        NOT NULL,
    quantity            NUMERIC     NOT NULL DEFAULT 0,
    avg_entry_price     NUMERIC     NOT NULL DEFAULT 0,
    total_fees          NUMERIC     NOT NULL DEFAULT 0,
    stop_loss           NUMERIC,
    take_profit         NUMERIC,
    realized_pnl        NUMERIC     NOT NULL DEFAULT 0,
    unrealized_pnl      NUMERIC     NOT NULL DEFAULT 0,
    opened_at           TIMESTAMPTZ NOT NULL,
    closed_at           TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol  ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status  ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);

-- ── Signals (audit trail) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    event_id            TEXT        PRIMARY KEY,
    strategy_id         TEXT        NOT NULL,
    symbol              TEXT        NOT NULL,
    direction           TEXT        NOT NULL,
    score               NUMERIC     NOT NULL,
    calibrated_score    NUMERIC     NOT NULL,
    confidence          NUMERIC     NOT NULL,
    expected_value      NUMERIC     NOT NULL,
    kelly_fraction      NUMERIC     NOT NULL,
    regime              TEXT        NOT NULL,
    timeframe           TEXT        NOT NULL,
    factors             JSONB,
    timestamp           TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_signals_symbol   ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_time     ON signals(timestamp DESC);

-- ── Risk events (audit trail) ────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_events (
    event_id            TEXT        PRIMARY KEY,
    action              TEXT        NOT NULL,
    reason              TEXT,
    drawdown_pct        NUMERIC,
    var_pct             NUMERIC,
    timestamp           TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_risk_events_time ON risk_events(timestamp DESC);
