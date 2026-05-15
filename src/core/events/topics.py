"""
Event topics — every event belongs to exactly one topic.
Components subscribe to topics, not to individual event types.

Flow:
  MARKET → SIGNAL → RISK → ORDER → FILL → PORTFOLIO
"""

from enum import StrEnum


class Topic(StrEnum):
    MARKET    = "market"      # candles, tickers, orderbook updates
    SIGNAL    = "signal"      # strategy signals
    RISK      = "risk"        # risk engine decisions
    ORDER     = "order"       # order lifecycle (new, submitted, cancelled...)
    FILL      = "fill"        # confirmed executions
    PORTFOLIO = "portfolio"   # portfolio state updates
    SYSTEM    = "system"      # heartbeat, kill switch, reconciliation
