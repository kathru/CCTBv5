"""
Meta Strategy Layer — self-optimizing strategy manager.

Migrated and improved from V4's meta_layer.py.

Responsibilities:
  1. Track performance of each strategy per regime
  2. Adjust signal weights based on recent P&L
  3. Activate/deactivate strategies based on edge health
  4. Detect regime changes and adapt accordingly

Strategy states:
  HEALTHY    → edge stable, normal weight
  EXPANDING  → edge growing → increase weight (max 55%)
  DEGRADING  → edge falling → reduce weight
  SUSPENDED  → edge negative 2+ windows → weight = 0
  RECOVERING → was suspended, now positive → min weight (5%)

Update frequency: per trade (online learning)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import StrEnum

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class StrategyEdgeState(StrEnum):
    HEALTHY    = "healthy"
    EXPANDING  = "expanding"
    DEGRADING  = "degrading"
    SUSPENDED  = "suspended"
    RECOVERING = "recovering"


@dataclass
class StrategyPerformance:
    """Tracks rolling performance for one strategy."""
    strategy_id: str
    edge_state: StrategyEdgeState = StrategyEdgeState.HEALTHY
    current_weight: float = 1.0         # multiplier on signal score
    win_rate_window: list[bool] = field(default_factory=list)
    pnl_window: list[float] = field(default_factory=list)
    negative_windows: int = 0
    total_trades: int = 0
    window_size: int = 20

    @property
    def recent_win_rate(self) -> float:
        if not self.win_rate_window:
            return 0.5
        return sum(self.win_rate_window) / len(self.win_rate_window)

    @property
    def recent_pnl(self) -> float:
        return sum(self.pnl_window)

    @property
    def edge(self) -> float:
        """Simple edge metric: win_rate - 0.5 (positive = has edge)."""
        return self.recent_win_rate - 0.5

    def record_trade(self, pnl: float) -> None:
        is_win = pnl > 0
        self.win_rate_window.append(is_win)
        self.pnl_window.append(pnl)
        self.total_trades += 1

        # Keep rolling window
        if len(self.win_rate_window) > self.window_size:
            self.win_rate_window.pop(0)
            self.pnl_window.pop(0)


# Weight bounds
MIN_WEIGHT = 0.05
MAX_WEIGHT = 1.55    # 55% bonus
DEFAULT_WEIGHT = 1.0
STEP_UP = 0.10
STEP_DOWN = 0.15


class MetaStrategyLayer:
    """
    Self-optimizing meta-layer that adapts strategy weights
    based on recent performance per regime.

    Works with the StrategyRunner — adjusts signal scores
    before they reach the OMS.
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._performances: dict[str, StrategyPerformance] = {}
        self._regime_performance: dict[str, dict[str, float]] = {}
        self._updated_at: datetime | None = None

    def register(self, strategy_id: str) -> None:
        """Register a strategy for tracking."""
        if strategy_id not in self._performances:
            self._performances[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                window_size=self._window_size,
            )
            logger.info("MetaLayer: registered strategy=%s", strategy_id)

    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        regime: str = "UNKNOWN",
    ) -> None:
        """
        Record a trade result and update strategy weights.
        Call after every closed position.
        """
        if strategy_id not in self._performances:
            self.register(strategy_id)

        perf = self._performances[strategy_id]
        perf.record_trade(pnl)

        # Track per-regime performance
        if regime not in self._regime_performance:
            self._regime_performance[regime] = {}
        reg_perf = self._regime_performance[regime]
        reg_perf[strategy_id] = reg_perf.get(strategy_id, 0.0) + pnl

        # Update edge state and weight
        self._update_state(perf)
        self._updated_at = _now()

        logger.debug(
            "MetaLayer: strategy=%s pnl=%.4f edge=%.3f state=%s weight=%.2f",
            strategy_id, pnl, perf.edge, perf.edge_state, perf.current_weight,
        )

    def _update_state(self, perf: StrategyPerformance) -> None:
        """State machine: update edge state and weight."""
        edge = perf.edge

        if perf.edge_state == StrategyEdgeState.SUSPENDED:
            if edge > 0:
                perf.edge_state = StrategyEdgeState.RECOVERING
                perf.current_weight = MIN_WEIGHT
                perf.negative_windows = 0
                logger.info("MetaLayer: %s RECOVERING", perf.strategy_id)
            return

        if perf.edge_state == StrategyEdgeState.RECOVERING:
            if edge > 0.05:
                perf.edge_state = StrategyEdgeState.HEALTHY
                perf.current_weight = DEFAULT_WEIGHT
            elif edge < 0:
                perf.edge_state = StrategyEdgeState.SUSPENDED
                perf.current_weight = 0.0
            return

        # Normal states
        if edge < 0:
            perf.negative_windows += 1
            if perf.negative_windows >= 2:
                perf.edge_state = StrategyEdgeState.SUSPENDED
                perf.current_weight = 0.0
                logger.warning("MetaLayer: %s SUSPENDED (negative edge)", perf.strategy_id)
            else:
                perf.edge_state = StrategyEdgeState.DEGRADING
                perf.current_weight = max(MIN_WEIGHT, perf.current_weight - STEP_DOWN)
        elif edge > 0.10:
            perf.negative_windows = 0
            perf.edge_state = StrategyEdgeState.EXPANDING
            perf.current_weight = min(MAX_WEIGHT, perf.current_weight + STEP_UP)
        else:
            perf.negative_windows = 0
            perf.edge_state = StrategyEdgeState.HEALTHY
            # Drift back toward default
            if perf.current_weight > DEFAULT_WEIGHT:
                perf.current_weight = max(DEFAULT_WEIGHT, perf.current_weight - STEP_DOWN / 2)

    def get_weight(self, strategy_id: str) -> float:
        """Return current weight multiplier for a strategy."""
        perf = self._performances.get(strategy_id)
        if perf is None:
            return DEFAULT_WEIGHT
        return perf.current_weight

    def is_active(self, strategy_id: str) -> bool:
        """Return False if strategy is SUSPENDED."""
        perf = self._performances.get(strategy_id)
        if perf is None:
            return True
        return perf.edge_state != StrategyEdgeState.SUSPENDED

    def apply_weight(self, strategy_id: str, score: float) -> float:
        """Apply meta-weight to a signal score. Clamps to [0, 1]."""
        weight = self.get_weight(strategy_id)
        return min(1.0, max(0.0, score * weight))

    def status(self) -> dict:
        return {
            sid: {
                "edge_state": p.edge_state.value,
                "weight": round(p.current_weight, 3),
                "win_rate": round(p.recent_win_rate, 3),
                "edge": round(p.edge, 3),
                "total_trades": p.total_trades,
            }
            for sid, p in self._performances.items()
        }
