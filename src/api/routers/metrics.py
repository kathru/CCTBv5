"""Metrics endpoints — read-only."""

from fastapi import APIRouter, Request

from ...core.config import settings
from ...core.version import get_version
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
    Estatísticas de performance calculadas a partir das ordens filled.
    """
    db = request.app.state.db

    # Lê ordens filled diretamente (fills table pode estar vazia em paper trading)
    rows = await db.fetch(
        "SELECT symbol, side, filled_quantity, avg_fill_price, fees_paid, filled_at "
        "FROM orders WHERE status='filled' ORDER BY filled_at ASC"
    )

    total_trades = len(rows)
    total_fees   = 0.0
    total_volume = 0.0
    pnl_by_symbol: dict[str, float] = {}
    buys_by_sym:  dict[str, float] = {}
    sells_by_sym: dict[str, float] = {}

    for r in rows:
        sym   = r["symbol"]
        qty   = float(r["filled_quantity"] or 0)
        price = float(r["avg_fill_price"] or 0)
        fees  = float(r["fees_paid"] or 0)
        side  = str(r["side"]).upper()
        notional = qty * price

        total_fees   += fees
        total_volume += notional

        if side in ("BUY", "LONG"):
            buys_by_sym[sym]  = buys_by_sym.get(sym, 0) + notional
        elif side in ("SELL", "SHORT"):
            sells_by_sym[sym] = sells_by_sym.get(sym, 0) + notional

    for sym in set(list(buys_by_sym) + list(sells_by_sym)):
        b = buys_by_sym.get(sym, 0)
        s = sells_by_sym.get(sym, 0)
        if b > 0 or s > 0:
            pnl_by_symbol[sym] = round(s - b, 2)

    total_pnl = sum(pnl_by_symbol.values())

    return {
        "total_trades": total_trades,
        "total_fees":   round(total_fees, 4),
        "total_volume": round(total_volume, 2),
        "total_pnl":    round(total_pnl, 2),
        "pnl_by_symbol": pnl_by_symbol,
    }
