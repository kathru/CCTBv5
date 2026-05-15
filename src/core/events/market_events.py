from dataclasses import dataclass

from ..models import Candle, Ticker
from .base import BaseEvent


@dataclass(frozen=True)
class CandleEvent(BaseEvent):
    """New candle received from exchange (confirmed or forming)."""
    candle: Candle | None = None


@dataclass(frozen=True)
class TickerEvent(BaseEvent):
    """Real-time price update."""
    ticker: Ticker | None = None
