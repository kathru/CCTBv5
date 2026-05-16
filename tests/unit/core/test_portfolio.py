from datetime import UTC, datetime

import pytest

from src.core.bus import EventBus
from src.core.models import Position, PositionSide, PositionStatus
from src.portfolio.engine import PortfolioEngine


def make_position(symbol: str, quantity: float, entry: float) -> Position:
    return Position(
        symbol=symbol,
        side=PositionSide.LONG,
        strategy_id="test",
        quantity=quantity,
        avg_entry_price=entry,
        status=PositionStatus.OPEN,
        opened_at=datetime.now(UTC),
    )


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def engine(bus):
    return PortfolioEngine(bus=bus, initial_capital=10000.0)


def test_initial_state(engine):
    state = engine.state
    assert state.total_value == 10000.0
    assert state.cash_available == 10000.0
    assert state.open_position_count == 0


def test_update_with_no_positions(engine):
    state = engine.update(positions=[], current_prices={}, cash=10000.0)
    assert state.total_value == 10000.0
    assert state.total_exposure_pct == 0.0
    assert state.portfolio_beta == 0.0


def test_update_with_open_position(engine):
    pos = make_position("BTC-USDT", quantity=0.1, entry=40000.0)
    prices = {"BTC-USDT": 42000.0}
    cash = 6000.0

    state = engine.update(positions=[pos], current_prices=prices, cash=cash)

    assert state.open_position_count == 1
    assert state.total_value == pytest.approx(6000 + 0.1 * 42000, rel=1e-3)
    assert state.unrealized_pnl == pytest.approx((42000 - 40000) * 0.1, rel=1e-3)
    assert state.total_exposure_pct > 0


def test_portfolio_beta_calculated(engine):
    pos = make_position("ETH-USDT", quantity=1.0, entry=2000.0)
    prices = {"ETH-USDT": 2000.0}
    state = engine.update(positions=[pos], current_prices=prices, cash=8000.0)
    # ETH beta = 1.2, fully deployed within open positions → beta = 1.2
    # (weighted by notional/total_notional = 2000/2000 = 1.0)
    assert state.portfolio_beta == pytest.approx(1.2, rel=0.1)


def test_can_open_position_allowed(engine):
    engine.update(positions=[], current_prices={}, cash=10000.0)
    allowed, reason = engine.can_open_position("BTC-USDT", notional=1000.0)
    assert allowed is True
    assert reason == ""


def test_can_open_position_too_concentrated(engine):
    engine.update(positions=[], current_prices={}, cash=10000.0)
    # 70% of portfolio in one position
    allowed, reason = engine.can_open_position("BTC-USDT", notional=7000.0)
    assert allowed is False
    assert "concentrated" in reason


def test_portfolio_state_drawdown(engine):
    # Simulate loss
    engine._state.initial_capital = 10000.0
    engine._state.total_value = 9000.0
    assert engine.state.drawdown_pct == pytest.approx(0.10, rel=1e-3)


def test_correlation_with_multiple_positions(engine):
    positions = [
        make_position("BTC-USDT", 0.1, 40000.0),
        make_position("ETH-USDT", 1.0, 2000.0),
    ]
    prices = {"BTC-USDT": 40000.0, "ETH-USDT": 2000.0}
    state = engine.update(positions=positions, current_prices=prices, cash=2000.0)
    # BTC-ETH correlation should be ~0.85
    assert 0.7 <= state.avg_correlation <= 1.0
