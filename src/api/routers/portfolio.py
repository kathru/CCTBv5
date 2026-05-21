"""
Portfolio router — expõe estado do PortfolioEngine e métricas de performance.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _serialize(v):
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: _serialize(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_serialize(i) for i in v]
    return str(v)


@router.get("/exits")
async def exit_plans(request: Request) -> dict:
    """Status dos planos de saída ativos (PositionMonitor)."""
    monitor = getattr(request.app.state, "position_monitor", None)
    if monitor is None:
        return {"available": False, "active_plans": 0, "plans": {}}
    return monitor.status()


@router.get("/summary")
async def portfolio_summary(request: Request) -> dict:
    """
    Estado completo do portfolio: capital, P&L, risco, exposição.
    Retorna zeros se o TradingLoop ainda não iniciou.
    """
    portfolio = getattr(request.app.state, "portfolio", None)
    if portfolio is None:
        return {
            "available": False,
            "initial_capital":    10000.0,
            "total_value":        10000.0,
            "cash_available":     10000.0,
            "total_exposure_pct": 0.0,
            "open_position_count": 0,
            "realized_pnl":       0.0,
            "unrealized_pnl":     0.0,
            "daily_pnl":          0.0,
            "total_return_pct":   0.0,
            "drawdown_pct":       0.0,
            "portfolio_beta":     0.0,
            "avg_correlation":    0.0,
            "concentration_risk": 0.0,
            "updated_at":         datetime.now(UTC).isoformat(),
        }

    s = portfolio.state

    # Conta posições abertas do DB (inclui posições sincronizadas da exchange)
    from ...persistence.repositories.positions import PositionRepository
    db = getattr(request.app.state, "db", None)
    db_open_positions = []
    if db:
        try:
            repo = PositionRepository(db)
            db_open_positions = await repo.get_open()
        except Exception:
            pass
    open_count = max(s.open_position_count, len(db_open_positions))

    # Calcula exposição e P&L com base nas posições do DB + preços do cache
    cache = getattr(request.app.state, "cache", None)
    unrealized = s.unrealized_pnl
    notional_total = 0.0
    positions_data = {}
    if cache and db_open_positions:
        for pos in db_open_positions:
            sym = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", "")
            qty = float(pos.get("quantity", 0) if isinstance(pos, dict) else getattr(pos, "quantity", 0))
            entry = float(pos.get("avg_entry_price", 0) if isinstance(pos, dict) else getattr(pos, "avg_entry_price", 0))
            price_raw = await cache.get_price(sym)
            price = float(price_raw) if price_raw else entry
            notional = qty * price
            unreal = (price - entry) * qty if entry > 0 else 0.0
            notional_total += notional
            unrealized += unreal
            positions_data[sym] = {
                "quantity": qty, "avg_entry": entry,
                "current_price": price, "notional": notional,
                "unrealized_pnl": unreal,
            }

    # Portfolio em USDT: cash disponível + notional das posições abertas do bot
    # NÃO inclui BTC/ETH/SOL pré-existentes — só o capital operacional do bot
    cash_value = s.cash_available
    total_value = cash_value + notional_total if notional_total > 0 else s.total_value
    exposure_pct = notional_total / total_value if total_value > 0 and notional_total > 0 else s.total_exposure_pct

    # Retorno relativo ao capital inicial (em USDT)
    initial = s.initial_capital if s.initial_capital > 0 else total_value
    total_return_pct = (total_value - initial) / initial if initial > 0 else 0.0

    # P&L realizado: soma das ordens SELL - ordens BUY fechadas (em USDT)
    realized_pnl = s.realized_pnl
    if db:
        try:
            rows = await db.fetch(
                "SELECT side, SUM(filled_quantity * avg_fill_price) as vol "
                "FROM orders WHERE status='filled' GROUP BY side"
            )
            sells = sum(float(r["vol"] or 0) for r in rows if str(r["side"]).upper() in ("SELL","SHORT"))
            buys  = sum(float(r["vol"] or 0) for r in rows if str(r["side"]).upper() in ("BUY","LONG"))
            if buys > 0 or sells > 0:
                realized_pnl = round(sells - buys, 4)
        except Exception:
            pass

    return {
        "available":           True,
        "initial_capital":     s.initial_capital,
        "total_value":         total_value,
        "cash_available":      s.cash_available,
        "total_exposure_pct":  round(exposure_pct, 4),
        "open_position_count": open_count,
        "positions":           positions_data,
        "realized_pnl":        realized_pnl,
        "unrealized_pnl":      round(unrealized, 4),
        "daily_pnl":           round(unrealized, 4),   # approximation
        "total_return_pct":    round(total_return_pct, 4),
        "drawdown_pct":        round(s.drawdown_pct, 4),
        "portfolio_beta":      round(s.portfolio_beta, 3),
        "avg_correlation":     round(s.avg_correlation, 3),
        "concentration_risk":  round(s.concentration_risk, 3),
        "is_overexposed":      s.is_overexposed,
        "updated_at":          _serialize(s.updated_at),
    }
