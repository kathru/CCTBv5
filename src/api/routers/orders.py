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
        orders = await repo.get_recent(limit=limit)
    return [_serialize(o) for o in orders]


@router.get("/filled")
async def get_filled_orders(request: Request, limit: int = 50) -> dict:
    """Retorna ordens preenchidas para o histórico de trades + total real."""
    repo = OrderRepository(request.app.state.db)
    orders = await repo.get_filled(limit=limit)
    total  = await repo.count_filled()
    return {"orders": [_serialize(o) for o in orders], "total": total}


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
        "fees_paid": order.fees_paid,
        "strategy_id": order.strategy_id,
        "created_at": str(order.created_at),
        "filled_at": str(order.filled_at) if order.filled_at else None,
    }
