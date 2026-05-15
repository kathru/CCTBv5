"""Orders endpoints — read-only."""
from fastapi import APIRouter, Request

from src.persistence.repositories.orders import OrderRepository

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.get("/open")
async def get_open_orders(request: Request) -> list[dict]:
    repo = OrderRepository(request.app.state.db)
    orders = await repo.get_open()
    return [_serialize(o) for o in orders]


@router.get("/recent")
async def get_recent_orders(request: Request, strategy_id: str = "", limit: int = 50) -> list[dict]:
    repo = OrderRepository(request.app.state.db)
    if strategy_id:
        orders = await repo.get_by_strategy(strategy_id, limit=limit)
    else:
        orders = await repo.get_open()
    return [_serialize(o) for o in orders]


def _serialize(order) -> dict:
    return {
        "client_order_id": order.client_order_id,
        "exchange_order_id": order.exchange_order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "status": order.status.value,
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": order.avg_fill_price,
        "strategy_id": order.strategy_id,
        "created_at": str(order.created_at),
    }
