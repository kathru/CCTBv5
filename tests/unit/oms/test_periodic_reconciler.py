from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.bus import EventBus
from src.recovery.periodic_reconciler import PeriodicReconciler


def make_exchange():
    exchange = AsyncMock()
    exchange.get_order_status.return_value = {"status": "submitted"}
    exchange.get_open_positions.return_value = []
    return exchange


def make_order_manager(open_orders=None):
    om = MagicMock()
    om.get_open_orders.return_value = open_orders or []
    om._accepting_orders = True
    om._orders = {}
    return om


@pytest.fixture
def bus():
    return EventBus()


@pytest.mark.asyncio
async def test_start_and_stop(bus):
    pr = PeriodicReconciler(
        bus=bus,
        db=AsyncMock(),
        exchange=make_exchange(),
        interval_seconds=300,
    )
    await pr.start()
    assert pr._running is True
    await pr.stop()
    assert pr._running is False


@pytest.mark.asyncio
async def test_no_cycle_when_no_open_orders(bus):
    pr = PeriodicReconciler(
        bus=bus,
        db=AsyncMock(),
        exchange=make_exchange(),
        order_manager=make_order_manager(open_orders=[]),
        interval_seconds=300,
    )
    # Run one cycle directly
    await pr._run_cycle()
    assert pr._run_count == 1
    assert pr._divergence_count == 0


@pytest.mark.asyncio
async def test_status_report(bus):
    pr = PeriodicReconciler(
        bus=bus,
        db=AsyncMock(),
        exchange=make_exchange(),
        interval_seconds=120,
    )
    status = pr.status()
    assert status["running"] is False
    assert status["run_count"] == 0
    assert status["interval_seconds"] == 120
