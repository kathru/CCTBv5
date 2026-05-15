"""
Strategy Metrics — professional performance statistics.

Computed from a list of trade P&Ls (returns).
All metrics are standard quantitative finance definitions.

Metrics:
  - Sharpe Ratio    : risk-adjusted return (vs risk-free rate)
  - Sortino Ratio   : like Sharpe but penalizes only downside vol
  - Profit Factor   : gross wins / gross losses
  - Win Rate        : % of winning trades
  - Expectancy      : avg P&L per trade (in R-multiples)
  - Max Drawdown    : largest peak-to-trough decline
  - Calmar Ratio    : annualized return / max drawdown
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05   # 5% annual (conservative)


@dataclass
class StrategyMetrics:
    """All performance metrics for a strategy."""

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0

    # Volatility
    volatility_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_return": f"{self.total_return_pct:.2%}",
            "annualized_return": f"{self.annualized_return_pct:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.3f}",
            "sortino_ratio": f"{self.sortino_ratio:.3f}",
            "calmar_ratio": f"{self.calmar_ratio:.3f}",
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.3f}",
            "expectancy": f"{self.expectancy:.4f}",
            "avg_win": f"{self.avg_win:.4f}",
            "avg_loss": f"{self.avg_loss:.4f}",
            "largest_win": f"{self.largest_win:.4f}",
            "largest_loss": f"{self.largest_loss:.4f}",
            "max_drawdown": f"{self.max_drawdown_pct:.2%}",
            "volatility": f"{self.volatility_pct:.2%}",
        }

    def grade(self) -> str:
        """Quick quality grade for the strategy."""
        if self.sharpe_ratio >= 2.0 and self.profit_factor >= 2.0:
            return "A"
        if self.sharpe_ratio >= 1.0 and self.profit_factor >= 1.5:
            return "B"
        if self.sharpe_ratio >= 0.5 and self.profit_factor >= 1.0:
            return "C"
        return "D"


def compute_metrics(
    returns: list[float],
    initial_capital: float = 10000.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> StrategyMetrics:
    """
    Compute all metrics from a list of per-trade returns (as fractions).

    Args:
        returns: List of per-trade returns, e.g. [0.02, -0.01, 0.03, ...]
        initial_capital: Starting capital (for drawdown calculation)
        periods_per_year: For annualization (252 for daily, 52 for weekly)

    Returns:
        StrategyMetrics with all computed values.
    """
    if not returns:
        return StrategyMetrics()

    n = len(returns)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    # Basic stats
    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Total and annualized return
    total_return = sum(returns)
    annualized = total_return * (periods_per_year / n) if n > 0 else 0.0

    # Volatility (std of returns)
    mean_r = total_return / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    volatility = math.sqrt(variance) * math.sqrt(periods_per_year) if variance > 0 else 0.0

    # Sharpe ratio
    daily_rf = RISK_FREE_RATE / periods_per_year
    excess_returns = [r - daily_rf for r in returns]
    mean_excess = sum(excess_returns) / n
    std_excess = math.sqrt(sum((r - mean_excess) ** 2 for r in excess_returns) / n)
    sharpe = (mean_excess / std_excess) * math.sqrt(periods_per_year) if std_excess > 0 else 0.0

    # Sortino ratio (downside deviation only)
    downside = [min(r - daily_rf, 0) ** 2 for r in returns]
    downside_std = math.sqrt(sum(downside) / n) * math.sqrt(periods_per_year)
    sortino = (mean_excess * periods_per_year) / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    max_dd, avg_dd = _compute_drawdowns(returns)

    # Calmar ratio
    calmar = annualized / abs(max_dd) if max_dd != 0 else 0.0

    return StrategyMetrics(
        total_return_pct=total_return,
        annualized_return_pct=annualized,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=max(wins) if wins else 0.0,
        largest_loss=min(losses) if losses else 0.0,
        max_drawdown_pct=max_dd,
        avg_drawdown_pct=avg_dd,
        volatility_pct=volatility,
    )


def _compute_drawdowns(returns: list[float]) -> tuple[float, float]:
    """Compute max and average drawdown from a returns series."""
    if not returns:
        return 0.0, 0.0

    # Build equity curve
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    # Compute drawdowns
    peak = equity[0]
    drawdowns = []

    for value in equity:
        if value > peak:
            peak = value
        dd = (value - peak) / peak if peak > 0 else 0.0
        drawdowns.append(dd)

    max_dd = min(drawdowns)
    avg_dd = sum(d for d in drawdowns if d < 0) / max(len([d for d in drawdowns if d < 0]), 1)

    return max_dd, avg_dd
