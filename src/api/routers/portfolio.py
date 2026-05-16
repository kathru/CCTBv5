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
    return {
        "available":           True,
        "initial_capital":     s.initial_capital,
        "total_value":         s.total_value,
        "cash_available":      s.cash_available,
        "total_exposure_pct":  round(s.total_exposure_pct, 4),
        "open_position_count": s.open_position_count,
        "realized_pnl":        round(s.realized_pnl, 4),
        "unrealized_pnl":      round(s.unrealized_pnl, 4),
        "daily_pnl":           round(s.daily_pnl, 4),
        "total_return_pct":    round(s.total_return_pct, 4),
        "drawdown_pct":        round(s.drawdown_pct, 4),
        "portfolio_beta":      round(s.portfolio_beta, 3),
        "avg_correlation":     round(s.avg_correlation, 3),
        "concentration_risk":  round(s.concentration_risk, 3),
        "is_overexposed":      s.is_overexposed,
        "updated_at":          _serialize(s.updated_at),
    }
