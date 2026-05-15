"""Fill repository — PostgreSQL CRUD."""

from ..postgres import Database
from ...core.models import Fill, OrderSide


class FillRepository:

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, fill: Fill) -> None:
        await self._db.execute("""
            INSERT INTO fills (
                fill_id, client_order_id, exchange_order_id,
                symbol, side, quantity, price, fee,
                fee_currency, is_maker, timestamp
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (fill_id) DO NOTHING
        """,
            fill.fill_id,
            fill.client_order_id,
            fill.exchange_order_id,
            fill.symbol,
            fill.side.value,
            fill.quantity,
            fill.price,
            fill.fee,
            fill.fee_currency,
            fill.is_maker,
            fill.timestamp,
        )

    async def get_by_order(self, client_order_id: str) -> list[Fill]:
        rows = await self._db.fetch(
            "SELECT * FROM fills WHERE client_order_id = $1"
            " ORDER BY timestamp ASC",
            client_order_id,
        )
        return [self._to_model(r) for r in rows]

    async def get_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> list[Fill]:
        rows = await self._db.fetch(
            "SELECT * FROM fills WHERE symbol = $1"
            " ORDER BY timestamp DESC LIMIT $2",
            symbol, limit,
        )
        return [self._to_model(r) for r in rows]

    def _to_model(self, row) -> Fill:
        return Fill(
            fill_id=row["fill_id"],
            client_order_id=row["client_order_id"],
            exchange_order_id=row["exchange_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            fee=float(row["fee"]),
            fee_currency=row["fee_currency"],
            is_maker=row["is_maker"],
            timestamp=row["timestamp"],
        )
