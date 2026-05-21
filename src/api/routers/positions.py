"""Positions endpoints — read-only."""
from fastapi import APIRouter, Request

from src.persistence.repositories.positions import PositionRepository

router = APIRouter(prefix="/api/positions", tags=["positions"])


@router.get("/open")
async def get_open_positions(request: Request) -> list[dict]:
    repo  = PositionRepository(request.app.state.db)
    cache = request.app.state.cache
    rows  = await repo.get_open()
    result = []
    for r in rows:
        pos = _serialize(r)
        sym   = pos.get("symbol", "")
        qty   = float(pos.get("quantity", 0) or 0)
        entry = float(pos.get("avg_entry_price", 0) or 0)
        # Enriquece com preço atual e P&L
        price_raw = await cache.get_price(sym) if cache and sym else None
        price = float(price_raw) if price_raw else entry
        notional = qty * price
        unrealized = (price - entry) * qty if entry > 0 else 0.0
        pos["current_price"]  = round(price, 4)
        pos["notional"]       = round(notional, 4)
        pos["unrealized_pnl"] = round(unrealized, 4)
        pos["pnl_pct"]        = round(unrealized / (entry * qty) * 100, 2) if entry * qty > 0 else 0.0
        # Formata qty sem decimais excessivos
        pos["quantity"]       = round(qty, 6)
        pos["avg_entry_price"]= round(entry, 4)
        result.append(pos)
    return result


@router.get("/closed")
async def get_closed_positions(request: Request, limit: int = 50) -> list[dict]:
    repo = PositionRepository(request.app.state.db)
    rows = await repo.get_closed(limit=limit)
    return [_serialize(r) for r in rows]


def _serialize(row) -> dict:
    import decimal, uuid
    from datetime import datetime
    def _s(v):
        if isinstance(v, (decimal.Decimal, uuid.UUID, datetime)):
            return str(v)
        if isinstance(v, dict):
            return {kk: _s(vv) for kk, vv in v.items()}
        return v
    if isinstance(row, dict):
        return {k: _s(v) for k, v in row.items()}
    return {k: _s(v) for k, v in dict(row).items()}
