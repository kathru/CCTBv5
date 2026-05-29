"""
StrategyTradeLogger — rastreamento de trades das estratégias v5.20.

As estratégias FundingHarvest, ShortSqueezeDetector, SectorPairDetector e
RegimeAwarePairEngine executam ordens diretamente via OKX (bypass OMS).
Este módulo registra seus trades na tabela `strategy_trades` do PostgreSQL,
alimentando analytics, WFO, edge alpha, advanced risk e reality check.

Tabela `strategy_trades`:
  id             SERIAL PK
  strategy_id    VARCHAR  (funding_harvest | short_squeeze | sector_pair | regime_pair)
  symbol         VARCHAR  (símbolo primário, ex: BTC-USDT)
  side           VARCHAR  (long | short | pair_long_short)
  entry_price    NUMERIC
  exit_price     NUMERIC  (NULL se posição aberta)
  notional       NUMERIC  (USDT alocados)
  pnl_usdt       NUMERIC  (NULL se aberta)
  pnl_pct        NUMERIC  (NULL se aberta)
  reason         VARCHAR  (motivo de saída)
  opened_at      TIMESTAMPTZ
  closed_at      TIMESTAMPTZ  (NULL se aberta)
  extra          JSONB    (dados extras: z-score, spread, mode, etc.)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# DDL auto-run on first use
_DDL = """
CREATE TABLE IF NOT EXISTS strategy_trades (
    id          SERIAL PRIMARY KEY,
    strategy_id VARCHAR(64)   NOT NULL,
    symbol      VARCHAR(32)   NOT NULL,
    side        VARCHAR(32)   NOT NULL DEFAULT 'long',
    entry_price NUMERIC(24,8),
    exit_price  NUMERIC(24,8),
    notional    NUMERIC(20,4),
    pnl_usdt    NUMERIC(20,4),
    pnl_pct     NUMERIC(12,8),
    reason      VARCHAR(256),
    opened_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    closed_at   TIMESTAMPTZ,
    extra       JSONB
);
CREATE INDEX IF NOT EXISTS idx_st_strategy ON strategy_trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_st_opened   ON strategy_trades(opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_st_closed   ON strategy_trades(closed_at DESC NULLS LAST);
"""


class StrategyTradeLogger:
    """
    Singleton ligado a uma conexão PostgreSQL.

    Uso:
        logger = StrategyTradeLogger(db)
        await logger.log_trade(
            strategy_id="funding_harvest",
            symbol="BTC-USDT",
            side="short",
            entry_price=50000.0,
            exit_price=49500.0,
            notional=200.0,
            pnl_pct=-0.005,
            pnl_usdt=-1.0,
            reason="low_funding",
            opened_at=<datetime>,
            closed_at=<datetime>,
            extra={"funding_rate": 0.0002},
        )
    """

    def __init__(self, db) -> None:
        self._db = db
        self._migrated = False

    async def _ensure_table(self) -> None:
        if self._migrated:
            return
        try:
            await self._db.execute(_DDL)
            self._migrated = True
        except Exception as exc:
            logger.warning("StrategyTradeLogger: falha na migração: %s", exc)

    async def log_trade(
        self,
        *,
        strategy_id: str,
        symbol: str,
        side: str = "long",
        entry_price: float | None = None,
        exit_price: float | None = None,
        notional: float | None = None,
        pnl_pct: float | None = None,
        pnl_usdt: float | None = None,
        reason: str | None = None,
        opened_at: datetime | None = None,
        closed_at: datetime | None = None,
        extra: dict | None = None,
    ) -> None:
        """Persiste um trade completo (entrada + saída) no banco."""
        await self._ensure_table()
        try:
            await self._db.execute(
                """
                INSERT INTO strategy_trades
                    (strategy_id, symbol, side, entry_price, exit_price,
                     notional, pnl_usdt, pnl_pct, reason,
                     opened_at, closed_at, extra)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                strategy_id,
                symbol,
                side,
                entry_price,
                exit_price,
                notional,
                pnl_usdt,
                pnl_pct,
                reason,
                opened_at or datetime.now(UTC),
                closed_at,
                json.dumps(extra) if extra else None,
            )
        except Exception as exc:
            logger.warning(
                "StrategyTradeLogger: falha ao persistir trade %s/%s: %s",
                strategy_id, symbol, exc,
            )

    async def get_trades(
        self,
        strategy_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Retorna trades fechados, opcionalmente filtrados por strategy_id."""
        await self._ensure_table()
        try:
            if strategy_id:
                rows = await self._db.fetch(
                    "SELECT * FROM strategy_trades "
                    "WHERE strategy_id=$1 AND closed_at IS NOT NULL "
                    "ORDER BY closed_at DESC LIMIT $2",
                    strategy_id, limit,
                )
            else:
                rows = await self._db.fetch(
                    "SELECT * FROM strategy_trades "
                    "WHERE closed_at IS NOT NULL "
                    "ORDER BY closed_at DESC LIMIT $1",
                    limit,
                )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("StrategyTradeLogger: falha ao buscar trades: %s", exc)
            return []

    async def get_stats_by_strategy(self) -> dict[str, dict]:
        """
        Agrega métricas por strategy_id:
          n_trades, win_rate, avg_pnl_pct, total_pnl_usdt,
          profit_factor, avg_hold_hours
        """
        await self._ensure_table()
        try:
            rows = await self._db.fetch(
                """
                SELECT
                    strategy_id,
                    COUNT(*)                                      AS n_trades,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS n_wins,
                    SUM(pnl_usdt)                                 AS total_pnl,
                    AVG(pnl_pct)                                  AS avg_pnl_pct,
                    SUM(CASE WHEN pnl_usdt > 0 THEN pnl_usdt ELSE 0 END) AS gross_profit,
                    ABS(SUM(CASE WHEN pnl_usdt < 0 THEN pnl_usdt ELSE 0 END)) AS gross_loss,
                    AVG(EXTRACT(EPOCH FROM (closed_at - opened_at))/3600) AS avg_hold_hours
                FROM strategy_trades
                WHERE closed_at IS NOT NULL
                GROUP BY strategy_id
                """
            )
        except Exception as exc:
            logger.warning("StrategyTradeLogger: get_stats_by_strategy error: %s", exc)
            return {}

        result: dict[str, dict] = {}
        for r in rows:
            n    = int(r["n_trades"])
            wins = int(r["n_wins"])
            gp   = float(r["gross_profit"] or 0)
            gl   = float(r["gross_loss"]   or 0)
            result[r["strategy_id"]] = {
                "n_trades":       n,
                "n_wins":         wins,
                "n_losses":       n - wins,
                "win_rate":       round(wins / n, 4) if n else None,
                "total_pnl_usdt": round(float(r["total_pnl"] or 0), 2),
                "avg_pnl_pct":    round(float(r["avg_pnl_pct"] or 0), 6),
                "profit_factor":  round(gp / gl, 3) if gl > 0 else None,
                "avg_hold_hours": round(float(r["avg_hold_hours"] or 0), 2),
            }
        return result


# ── Singleton de aplicação ────────────────────────────────────────────────────
# Inicializado em startup (app.py) com: strategy_trade_log.init(db)
# As estratégias importam `strategy_trade_log` e chamam await strategy_trade_log.log_trade(...)

class _LazyLogger:
    """Proxy que aceita chamadas antes da inicialização (no-op seguro)."""

    def __init__(self) -> None:
        self._inner: StrategyTradeLogger | None = None

    def init(self, db) -> None:
        self._inner = StrategyTradeLogger(db)

    async def log_trade(self, **kwargs) -> None:
        if self._inner:
            await self._inner.log_trade(**kwargs)

    async def get_trades(self, **kwargs) -> list[dict]:
        if self._inner:
            return await self._inner.get_trades(**kwargs)
        return []

    async def get_stats_by_strategy(self) -> dict[str, dict]:
        if self._inner:
            return await self._inner.get_stats_by_strategy()
        return {}


strategy_trade_log = _LazyLogger()
