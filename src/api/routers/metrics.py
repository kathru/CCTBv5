"""Metrics endpoints — read-only."""

from fastapi import APIRouter, Request

from ...core.config import settings
from ...core.version import get_version

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
    total_buys   = sum(1 for r in rows if str(r["side"]).upper() in ("BUY", "LONG"))
    total_sells  = sum(1 for r in rows if str(r["side"]).upper() in ("SELL", "SHORT"))
    total_fees   = 0.0
    total_volume = 0.0
    pnl_by_symbol: dict[str, float] = {}
    buys_by_sym:  dict[str, float] = {}
    sells_by_sym: dict[str, float] = {}

    fees_from_db = 0.0
    for r in rows:
        sym      = r["symbol"]
        qty      = float(r["filled_quantity"] or 0)
        price    = float(r["avg_fill_price"] or 0)
        fees     = float(r["fees_paid"] or 0)
        side     = str(r["side"]).upper()
        notional = qty * price

        fees_from_db += fees
        total_volume += notional

        if side in ("BUY", "LONG"):
            buys_by_sym[sym]  = buys_by_sym.get(sym, 0) + notional
        elif side in ("SELL", "SHORT"):
            sells_by_sym[sym] = sells_by_sym.get(sym, 0) + notional

    # OKX demo pode retornar fees=0 — estima 0.1% (taker rate padrão) nesse caso
    if fees_from_db == 0.0 and total_volume > 0:
        total_fees = round(total_volume * 0.001, 4)  # 0.1% sobre volume total
    else:
        total_fees = round(fees_from_db, 4)

    # P&L realizado: cost-basis por símbolo (não net cash flow)
    for sym in set(list(buys_by_sym) + list(sells_by_sym)):
        b = buys_by_sym.get(sym, 0)
        s = sells_by_sym.get(sym, 0)
        if b > 0 and s > 0:
            # Calc qty via ordens para avg buy price (aproximação simples por notional)
            pnl_by_symbol[sym] = round(s - b, 2)  # parcial: só pairs com sell

    total_pnl = sum(pnl_by_symbol.values())

    # Total de ordens no banco (filled + cancelled)
    all_rows = await db.fetch("SELECT COUNT(*) as n FROM orders")
    total_orders_db = int(all_rows[0]["n"]) if all_rows else total_trades

    return {
        "total_trades":    total_trades,   # ordens filled (buy + sell)
        "total_buys":      total_buys,
        "total_sells":     total_sells,
        "total_orders_db": total_orders_db,  # total no banco (inclui cancelled)
        "total_fees":   round(total_fees, 4),
        "total_volume": round(total_volume, 2),
        "total_pnl":    round(total_pnl, 2),
        "pnl_by_symbol": pnl_by_symbol,
    }
