"""Metrics endpoints — read-only."""

from fastapi import APIRouter, Request

from ...core.version import get_version
from ...core.config import settings
from ...persistence.repositories.fills import FillRepository

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


@router.get("/system")
async def get_system_metrics(request: Request) -> dict:
    cache = request.app.state.cache
    status = await cache.get_system_status() or "unknown"
    return {
        "system_status": status,
        "version": get_version(),
        "mode": "paper" if settings.okx_paper_trading else "live",
    }


@router.get("/prices")
async def get_prices(request: Request) -> dict:
    """Return cached prices for all known symbols."""
    cache = request.app.state.cache
    prices = {}
    for symbol in SYMBOLS:
        price = await cache.get_price(symbol)
        if price:
            prices[symbol] = price
    return prices


@router.get("/performance")
async def get_performance(request: Request) -> dict:
    """
    Estatísticas de performance calculadas a partir dos fills no PostgreSQL.
    Retorna: total_trades, win_rate, total_fees, pnl_by_symbol, etc.
    """
    db = request.app.state.db
    repo = FillRepository(db)

    # Agrega fills de todos os símbolos
    total_trades = 0
    total_fees   = 0.0
    total_volume = 0.0
    pnl_by_symbol: dict[str, float] = {}

    for symbol in SYMBOLS:
        fills = await repo.get_by_symbol(symbol, limit=1000)
        if not fills:
            continue

        for fill in fills:
            total_trades += 1
            fee  = getattr(fill, "fee", 0.0) or 0.0
            qty  = getattr(fill, "quantity", 0.0) or 0.0
            price = getattr(fill, "price", 0.0) or 0.0
            total_fees   += fee
            total_volume += qty * price

        # P&L por símbolo: soma fills SELL - soma fills BUY (simplificado)
        buys  = sum(f.quantity * f.price for f in fills
                    if str(getattr(f, "side", "")).upper() in ("BUY", "LONG"))
        sells = sum(f.quantity * f.price for f in fills
                    if str(getattr(f, "side", "")).upper() in ("SELL", "SHORT"))
        if buys > 0 or sells > 0:
            pnl_by_symbol[symbol] = round(sells - buys, 2)

    total_pnl = sum(pnl_by_symbol.values())

    return {
        "total_trades":   total_trades,
        "total_fees":     round(total_fees, 4),
        "total_volume":   round(total_volume, 2),
        "total_pnl":      round(total_pnl, 2),
        "pnl_by_symbol":  pnl_by_symbol,
    }
