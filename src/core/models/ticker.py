from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Ticker:
    """Real-time price snapshot — immutable, exchange-agnostic."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume_24h: float
    open_24h: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0.0
