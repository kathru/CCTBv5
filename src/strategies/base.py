"""
BaseStrategy — interface that ALL strategies must implement.

Rules (enforced by ABC):
  1. Receives data by injection (candles, ticker, context)
  2. Returns Signal | None — nothing else
  3. NEVER executes orders
  4. NEVER accesses the database
  5. NEVER accesses the exchange

This makes strategies:
  - Fully testable in isolation (no mocks needed for infra)
  - Replayable (same input → same output, deterministic)
  - Safe to run in backtest with identical code as live
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.models import Candle, Signal, Ticker


@dataclass
class StrategyContext:
    """
    Everything a strategy needs to make a decision.
    Injected by the strategy runner — strategy never fetches this itself.
    """
    symbol: str
    candles_1h: list[Candle]
    candles_6h: list[Candle]
    ticker: Ticker | None
    portfolio_value: float
    # mantido por compatibilidade (vazio em ciclo 1H)
    candles_30m: list[Candle] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)   # regime, breadth, etc.


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.
    Subclasses must implement evaluate().
    """

    def __init__(self, strategy_id: str, symbols: list[str]) -> None:
        self._strategy_id = strategy_id
        self._symbols = symbols
        self._enabled = True

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @abstractmethod
    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        """
        Core evaluation method.

        Args:
            ctx: All market data and portfolio state needed.

        Returns:
            Signal if there's an actionable opportunity, None otherwise.

        Constraints:
            - Must be deterministic given the same ctx
            - Must NOT have side effects (no DB writes, no API calls)
            - Must NOT access global state
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._strategy_id})"
