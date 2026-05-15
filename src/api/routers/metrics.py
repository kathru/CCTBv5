"""Metrics endpoints — read-only."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/system")
async def get_system_metrics(request: Request) -> dict:
    cache = request.app.state.cache
    status = await cache.get_system_status() or "unknown"
    return {
        "system_status": status,
        "version": "5.0.0",
    }


@router.get("/prices")
async def get_prices(request: Request) -> dict:
    """Return cached prices for all known symbols."""
    cache = request.app.state.cache
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    prices = {}
    for symbol in symbols:
        price = await cache.get_price(symbol)
        if price:
            prices[symbol] = price
    return prices
