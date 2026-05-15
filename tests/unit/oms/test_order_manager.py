from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.bus import EventBus
from src.core.events import SignalEvent
from src.core.models import Fill, OrderSide, OrderStatus, Signal, SignalDirection
from src.oms.order_manager import OrderManager


def make_signal() -> Signal:
    return Signal(
        strategy_id="test_strategy",
        symbol="BTC-USDT",
        direction=SignalDirection.LONG,
        timestamp=datetime.now(UTC),
        score=0.75,
        calibrated_score=0.72,
        confidence=0.8,
        expected_value=3.5,
        kelly_fraction=0.1,
        regime="TREND_EXPANSION",
        timeframe="1H",
    )


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def router():
    mock = AsyncMock()
    mock.submit.return_value = "EXC-ORDER-123"
    mock.cancel.return_value = True
    return mock


@pytest.fixture
def oms(bus, router):
    manager = OrderManager(bus=bus, router=router)
    manager.open_gate()
    return manager


@pytest.mark.asyncio
async def test_gate_blocks_orders(bus, router):
    oms = OrderManager(bus=bus, router=router)
    # Gate starts CLOSED
    signal = make_signal()
    await oms.on_signal(SignalEvent(signal=signal))
    assert oms.open_order_count == 0


@pytest.mark.asyncio
async def test_signal_creates_submitted_order(oms):
    signal = make_signal()
    await oms.on_signal(SignalEvent(signal=signal))
    assert oms.open_order_count == 1
    order = oms.get_open_orders()[0]
    assert order.status == OrderStatus.SUBMITTED
    assert order.exchange_order_id == "EXC-ORDER-123"


@pytest.mark.asyncio
async def test_fill_transitions_to_filled(oms):
    signal = make_signal()
    await oms.on_signal(SignalEvent(signal=signal))
    order = oms.get_open_orders()[0]
    order.quantity = 0.01

    fill = Fill(
        fill_id="fill-1",
        client_order_id=order.client_order_id,
        exchange_order_id=order.exchange_order_id,
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        quantity=0.01,
        price=60000.0,
        fee=0.006,
        fee_currency="USDT",
        timestamp=datetime.now(UTC),
        is_maker=True,
    )
    await oms.on_fill(fill)
    assert order.status == OrderStatus.FILLED
    assert order.avg_fill_price == 60000.0


@pytest.mark.asyncio
async def test_cancel_all(oms):
    signal = make_signal()
    await oms.on_signal(SignalEvent(signal=signal))
    assert oms.open_order_count == 1
    await oms.cancel_all(reason="test")
    assert oms.open_order_count == 0
