"""Position repository — PostgreSQL CRUD."""

from ...core.models import Position
from ..postgres import Database


class PositionRepository:

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, position: Position, position_id: str) -> None:
        await self._db.execute("""
            INSERT INTO positions (
                id, symbol, side, status, strategy_id,
                quantity, avg_entry_price, total_fees,
                stop_loss, take_profit,
                realized_pnl, unrealized_pnl,
                opened_at, closed_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (id) DO UPDATE SET
                status          = EXCLUDED.status,
                quantity        = EXCLUDED.quantity,
                avg_entry_price = EXCLUDED.avg_entry_price,
                total_fees      = EXCLUDED.total_fees,
                stop_loss       = EXCLUDED.stop_loss,
                take_profit     = EXCLUDED.take_profit,
                realized_pnl    = EXCLUDED.realized_pnl,
                unrealized_pnl  = EXCLUDED.unrealized_pnl,
                closed_at       = EXCLUDED.closed_at
        """,
            position_id,
            position.symbol,
            position.side.value,
            position.status.value,
            position.strategy_id,
            position.quantity,
            position.avg_entry_price,
            position.total_fees,
            position.stop_loss,
            position.take_profit,
            position.realized_pnl,
            position.unrealized_pnl,
            position.opened_at,
            position.closed_at,
        )

    async def get_open(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = await self._db.fetch(
                "SELECT * FROM positions WHERE status = 'open' AND symbol = $1"
                " ORDER BY opened_at DESC",
                symbol,
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM positions WHERE status = 'open'"
                " ORDER BY opened_at DESC"
            )
        return [dict(r) for r in rows]

    async def get_by_strategy(
        self, strategy_id: str, limit: int = 100
    ) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT * FROM positions WHERE strategy_id = $1"
            " ORDER BY opened_at DESC LIMIT $2",
            strategy_id, limit,
        )
        return [dict(r) for r in rows]

    async def close_stale_by_symbol(self, symbol: str) -> int:
        """Fecha todas as posições abertas de um símbolo (sync com OKX zerou o saldo)."""
        from datetime import datetime, timezone
        result = await self._db.execute(
            """UPDATE positions
               SET status = 'closed', closed_at = $1
               WHERE status = 'open' AND symbol = $2""",
            datetime.now(timezone.utc),
            symbol,
        )
        # asyncpg retorna "UPDATE N" — extrai o N
        try:
            return int(str(result).split()[-1])
        except Exception:
            return 0

    async def get_closed(
        self, symbol: str | None = None, limit: int = 200
    ) -> list[dict]:
        if symbol:
            rows = await self._db.fetch(
                "SELECT * FROM positions WHERE status = 'closed' AND symbol = $1"
                " ORDER BY closed_at DESC LIMIT $2",
                symbol, limit,
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM positions WHERE status = 'closed'"
                " ORDER BY closed_at DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]
