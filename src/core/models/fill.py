from dataclasses import dataclass
from datetime import datetime

from .order import OrderSide


@dataclass(frozen=True)
class Fill:
    """
    Confirmed execution — immutable record of what actually happened.
    One Order can produce multiple Fills (partial fills).
    """

    fill_id: str
    client_order_id: str          # links to Order
    exchange_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime
    is_maker: bool                # maker fee vs taker fee

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        return self.notional + self.fee
