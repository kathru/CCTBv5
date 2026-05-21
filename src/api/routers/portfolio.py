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
            "initial_capital":    85_000.0,
            "total_value":        85_000.0,
            "cash_available":     85_000.0,
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

    # Calcula P&L com base nas posições do DB + preços do cache
    cache = getattr(request.app.state, "cache", None)
    unrealized = 0.0
    notional_bot = 0.0   # só posições do bot (momentum_v2), não exchange_sync
    notional_total = 0.0 # todas as posições para exposição
    positions_data = {}
    if cache and db_open_positions:
        for pos in db_open_positions:
            sym     = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", "")
            qty     = float(pos.get("quantity", 0) if isinstance(pos, dict) else getattr(pos, "quantity", 0))
            entry   = float(pos.get("avg_entry_price", 0) if isinstance(pos, dict) else getattr(pos, "avg_entry_price", 0))
            strat   = pos.get("strategy_id", "") if isinstance(pos, dict) else getattr(pos, "strategy_id", "")
            price_raw = await cache.get_price(sym)
            price   = float(price_raw) if price_raw else entry
            notional = qty * price
            unreal   = (price - entry) * qty if entry > 0 else 0.0
            notional_total += notional
            # Só conta unrealized P&L de posições abertas pelo bot (não as do sync inicial)
            is_bot_position = str(strat) not in ("exchange_sync", "")
            if is_bot_position:
                notional_bot += notional
                unrealized   += unreal
            positions_data[sym] = {
                "quantity": qty, "avg_entry": entry,
                "current_price": price, "notional": notional,
                "unrealized_pnl": unreal,
                "strategy_id": strat,
            }

    # P&L realizado: SELL - BUY (em USDT) de ordens do bot
    realized_pnl = 0.0
    if db:
        try:
            rows = await db.fetch(
                "SELECT side, SUM(filled_quantity * avg_fill_price) as vol "
                "FROM orders WHERE status='filled' AND strategy_id != 'exchange_sync' GROUP BY side"
            )
            sells = sum(float(r["vol"] or 0) for r in rows if str(r["side"]).upper() in ("SELL","SHORT"))
            buys  = sum(float(r["vol"] or 0) for r in rows if str(r["side"]).upper() in ("BUY","LONG"))
            realized_pnl = round(sells - buys, 4)
        except Exception:
            pass

    # Portfolio em USDT = capital inicial + P&L realizado + P&L não realizado (bot)
    initial = s.initial_capital if s.initial_capital > 1000.0 else 85_000.0
    cash_value = s.cash_available
    # total = cash + valor das posições compradas pelo bot (a preço atual)
    total_value = cash_value + notional_bot if notional_bot > 0 else cash_value + unrealized + initial + realized_pnl
    total_value = max(total_value, cash_value)  # nunca menor que o cash disponível
    exposure_pct = notional_bot / total_value if total_value > 0 and notional_bot > 0 else (
        notional_total / (initial + realized_pnl + unrealized) if initial > 0 else 0.0
    )
    total_return_pct = (total_value - initial) / initial if initial > 0 else 0.0

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
