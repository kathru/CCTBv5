"""Order repository — PostgreSQL CRUD."""

from ...core.models import Order, OrderMode, OrderSide, OrderStatus, OrderType
from ..postgres import Database


class OrderRepository:

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, order: Order) -> None:
        """Insert or update an order (upsert)."""
        await self._db.execute("""
            INSERT INTO orders (
                client_order_id, exchange_order_id, symbol, side,
                order_type, mode, status, quantity, filled_quantity,
                avg_fill_price, limit_price, stop_loss, take_profit,
                fees_paid, strategy_id, signal_id, retry_count,
                last_error, created_at, submitted_at, filled_at, cancelled_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22
            )
            ON CONFLICT (client_order_id) DO UPDATE SET
                exchange_order_id = EXCLUDED.exchange_order_id,
                status            = EXCLUDED.status,
                filled_quantity   = EXCLUDED.filled_quantity,
                avg_fill_price    = EXCLUDED.avg_fill_price,
                fees_paid         = EXCLUDED.fees_paid,
                retry_count       = EXCLUDED.retry_count,
                last_error        = EXCLUDED.last_error,
                submitted_at      = EXCLUDED.submitted_at,
                filled_at         = EXCLUDED.filled_at,
                cancelled_at      = EXCLUDED.cancelled_at
        """,
            order.client_order_id,
            order.exchange_order_id,
            order.symbol,
            order.side.value,
            order.order_type.value,
            order.mode.value,
            order.status.value,
            order.quantity,
            order.filled_quantity,
            order.avg_fill_price,
            order.limit_price,
            order.stop_loss,
            order.take_profit,
            order.fees_paid,
            order.strategy_id,
            order.signal_id,
            order.retry_count,
            order.last_error,
            order.created_at,
            order.submitted_at,
            order.filled_at,
            order.cancelled_at,
        )

    async def get(self, client_order_id: str) -> Order | None:
        row = await self._db.fetchrow(
            "SELECT * FROM orders WHERE client_order_id = $1",
            client_order_id,
        )
        return self._to_model(row) if row else None

    async def get_open(self, symbol: str | None = None) -> list[Order]:
        open_statuses = ("new", "pending", "submitted", "partial")
        if symbol:
            rows = await self._db.fetch(
                "SELECT * FROM orders WHERE status = ANY($1) AND symbol = $2"
                " ORDER BY created_at DESC",
                list(open_statuses), symbol,
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM orders WHERE status = ANY($1)"
                " ORDER BY created_at DESC",
                list(open_statuses),
            )
        return [self._to_model(r) for r in rows]

    async def get_recent(self, limit: int = 500) -> list[Order]:
        """Retorna ordens recentes (todos os status), mais recentes primeiro.
        Usa filled_at quando disponível, senão created_at."""
        rows = await self._db.fetch(
            "SELECT * FROM orders "
            "ORDER BY COALESCE(filled_at, submitted_at, created_at) DESC "
            "LIMIT $1",
            limit,
        )
        return [self._to_model(r) for r in rows]

    async def get_filled(self, limit: int = 50) -> list[Order]:
        """Retorna apenas ordens preenchidas (filled), mais recentes primeiro."""
        rows = await self._db.fetch(
            "SELECT * FROM orders WHERE status = 'filled' "
            "ORDER BY COALESCE(filled_at, created_at) DESC "
            "LIMIT $1",
            limit,
        )
        return [self._to_model(r) for r in rows]

    async def count_filled(self) -> int:
        """Conta o total de ordens preenchidas no DB."""
        row = await self._db.fetchrow("SELECT COUNT(*) FROM orders WHERE status = 'filled'")
        return int(row[0]) if row else 0

    async def get_by_strategy(
        self, strategy_id: str, limit: int = 100
    ) -> list[Order]:
        rows = await self._db.fetch(
            "SELECT * FROM orders WHERE strategy_id = $1"
            " ORDER BY created_at DESC LIMIT $2",
            strategy_id, limit,
        )
        return [self._to_model(r) for r in rows]

    def _to_model(self, row) -> Order:
        return Order(
            client_order_id=row["client_order_id"],
            exchange_order_id=row["exchange_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            order_type=OrderType(row["order_type"]),
            mode=OrderMode(row["mode"]),
            status=OrderStatus(row["status"]),
            quantity=float(row["quantity"]),
            filled_quantity=float(row["filled_quantity"]),
            avg_fill_price=float(row["avg_fill_price"]),
            limit_price=float(row["limit_price"]) if row["limit_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            fees_paid=float(row["fees_paid"]),
            strategy_id=row["strategy_id"],
            signal_id=row["signal_id"],
            retry_count=row["retry_count"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            submitted_at=row["submitted_at"],
            filled_at=row["filled_at"],
            cancelled_at=row["cancelled_at"],
        )
