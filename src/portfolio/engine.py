"""
Portfolio Engine — manages allocation, exposure, correlation and capital.

Responsibilities:
  1. Maintain PortfolioState from open positions + fills
  2. Gate new entries based on portfolio-level risk
  3. Calculate beta, correlation, concentration
  4. Publish PortfolioState to bus for strategies to consume

This engine is the final gate before a signal becomes an order.
Even if Risk Engine says NORMAL, Portfolio Engine can veto an entry
if portfolio-level risk is too high.
"""

import logging
from datetime import UTC, datetime

from ..core.bus import EventBus
from ..core.models import Position, PositionStatus
from .state import PortfolioState

logger = logging.getLogger(__name__)


# BTC correlation benchmarks (approximate)
# Used to estimate portfolio beta
SYMBOL_BETA: dict[str, float] = {
    "BTC-USDT": 1.0,
    "ETH-USDT": 1.2,
    "SOL-USDT": 1.4,
    "AVAX-USDT": 1.3,
    "LINK-USDT": 1.1,
    "DOGE-USDT": 0.9,
}

# Approximate correlation matrix (simplified)
SYMBOL_CORRELATION: dict[tuple[str, str], float] = {
    ("BTC-USDT", "ETH-USDT"): 0.85,
    ("BTC-USDT", "SOL-USDT"): 0.80,
    ("ETH-USDT", "SOL-USDT"): 0.82,
    ("BTC-USDT", "AVAX-USDT"): 0.75,
    ("ETH-USDT", "AVAX-USDT"): 0.78,
}


def _now() -> datetime:
    return datetime.now(UTC)


class PortfolioEngine:
    """
    Manages portfolio state and gates new entries.
    """

    MAX_PORTFOLIO_BETA = 1.8
    MAX_AVG_CORRELATION = 0.85
    MAX_CONCENTRATION = 0.60    # max single position as % of portfolio

    def __init__(
        self,
        bus: EventBus,
        initial_capital: float = 10000.0,
    ) -> None:
        self._bus = bus
        self._state = PortfolioState(
            initial_capital=initial_capital,
            total_value=initial_capital,
            cash_available=initial_capital,
        )

    def update(
        self,
        positions: list[Position],
        current_prices: dict[str, float],
        cash: float,
    ) -> PortfolioState:
        """
        Recalculate portfolio state from current positions and prices.
        Call every cycle after receiving market data.
        """
        open_positions = [p for p in positions if p.status == PositionStatus.OPEN]

        # Update unrealized P&L
        total_unrealized = 0.0
        positions_dict = {}
        total_notional = 0.0

        for pos in open_positions:
            price = current_prices.get(pos.symbol, pos.avg_entry_price)
            pos.update_unrealized(price)
            notional = pos.quantity * price
            total_notional += notional
            total_unrealized += pos.unrealized_pnl
            positions_dict[pos.symbol] = {
                "side": pos.side.value,
                "quantity": pos.quantity,
                "avg_entry": pos.avg_entry_price,
                "current_price": price,
                "notional": notional,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "strategy_id": pos.strategy_id,
            }

        total_realized = sum(p.realized_pnl for p in positions)
        total_value = cash + total_notional
        exposure_pct = total_notional / total_value if total_value > 0 else 0.0

        self._state = PortfolioState(
            total_value=total_value,
            cash_available=cash,
            initial_capital=self._state.initial_capital,
            total_exposure_pct=exposure_pct,
            open_position_count=len(open_positions),
            realized_pnl=total_realized,
            unrealized_pnl=total_unrealized,
            daily_pnl=total_unrealized + total_realized,
            total_return_pct=(
                (total_value - self._state.initial_capital) / self._state.initial_capital
                if self._state.initial_capital > 0 else 0.0
            ),
            portfolio_beta=self._calc_beta(open_positions, positions_dict),
            avg_correlation=self._calc_avg_correlation(open_positions),
            concentration_risk=self._calc_concentration(
                open_positions, positions_dict, total_notional
            ),
            positions=positions_dict,
            updated_at=_now(),
        )

        logger.debug(
            "Portfolio updated value=%.2f exposure=%.1f%% positions=%d beta=%.2f",
            total_value, exposure_pct * 100,
            len(open_positions), self._state.portfolio_beta,
        )
        return self._state

    def can_open_position(
        self,
        symbol: str,
        notional: float,
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Final portfolio-level gate before new entry.
        """
        state = self._state

        # Check beta limit
        new_beta = state.portfolio_beta + (
            SYMBOL_BETA.get(symbol, 1.0) * notional / max(state.total_value, 1)
        )
        if new_beta > self.MAX_PORTFOLIO_BETA:
            return False, f"Portfolio beta too high: {new_beta:.2f} > {self.MAX_PORTFOLIO_BETA}"

        # Check correlation
        if state.avg_correlation > self.MAX_AVG_CORRELATION:
            return False, f"Positions too correlated: {state.avg_correlation:.2f}"

        # Check concentration
        if state.total_value > 0:
            new_concentration = notional / state.total_value
            if new_concentration > self.MAX_CONCENTRATION:
                return False, (
                    f"Position too concentrated: {new_concentration:.1%} "
                    f"> {self.MAX_CONCENTRATION:.1%}"
                )

        return True, ""

    def _calc_beta(
        self,
        positions: list[Position],
        positions_dict: dict,
    ) -> float:
        """Weighted average beta vs BTC."""
        total_notional = sum(
            positions_dict.get(p.symbol, {}).get("notional", 0)
            for p in positions
        )
        if total_notional <= 0:
            return 0.0

        weighted_beta = sum(
            SYMBOL_BETA.get(p.symbol, 1.0) *
            positions_dict.get(p.symbol, {}).get("notional", 0)
            for p in positions
        )
        return weighted_beta / total_notional

    def _calc_avg_correlation(self, positions: list[Position]) -> float:
        """Average pairwise correlation between open positions."""
        symbols = [p.symbol for p in positions]
        if len(symbols) < 2:
            return 0.0

        pairs = []
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                key = (min(s1, s2), max(s1, s2))
                corr = SYMBOL_CORRELATION.get(key, 0.7)  # default 70%
                pairs.append(corr)

        return sum(pairs) / len(pairs) if pairs else 0.0

    def _calc_concentration(
        self,
        positions: list[Position],
        positions_dict: dict,
        total_notional: float,
    ) -> float:
        """Herfindahl-Hirschman Index — concentration measure."""
        if total_notional <= 0 or not positions:
            return 0.0
        shares = [
            positions_dict.get(p.symbol, {}).get("notional", 0) / total_notional
            for p in positions
        ]
        return sum(s ** 2 for s in shares)

    @property
    def state(self) -> PortfolioState:
        return self._state
