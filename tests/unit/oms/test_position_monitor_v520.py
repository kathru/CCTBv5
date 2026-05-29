"""Tests for PositionMonitor v5.20 — new fields and public APIs."""

from datetime import UTC, datetime

from src.oms.position_monitor import ExitPlan, ShortSwapPlan

# ── ExitPlan — default fields ─────────────────────────────────────────────────

def make_exit_plan(**kwargs):
    defaults = dict(
        symbol="BTC-USDT",
        quantity=0.01,
        qty_remaining=0.01,
        stop_loss=49000.0,
        take_profit=55000.0,
        entry_price=50000.0,
        entry_time=datetime.now(UTC),
        entry_regime="TREND_EXPANSION",
        atr=500.0,
        strategy_id="momentum_v2",
    )
    defaults.update(kwargs)
    return ExitPlan(**defaults)


def test_exit_plan_chill_mode_defaults_false():
    plan = make_exit_plan()
    assert plan.chill_mode is False


def test_exit_plan_running_mode_defaults_false():
    plan = make_exit_plan()
    assert plan.running_mode is False


def test_exit_plan_trailing_activated_defaults_false():
    plan = make_exit_plan()
    assert plan.trailing_activated is False


def test_exit_plan_chill_mode_can_be_set():
    plan = make_exit_plan()
    plan.chill_mode = True
    assert plan.chill_mode is True


def test_exit_plan_summary_includes_chill():
    plan = make_exit_plan()
    plan.chill_mode = True
    summary = plan.summary()
    assert "chill" in summary.lower() or "YES" in summary


def test_exit_plan_summary_includes_running():
    plan = make_exit_plan()
    plan.running_mode = True
    summary = plan.summary()
    assert "running" in summary.lower() or "YES" in summary


# ── ShortSwapPlan — harvest_mode ──────────────────────────────────────────────

def make_short_plan(**kwargs):
    defaults = dict(
        symbol="BTC-USDT",
        swap_sym="BTC-USDT-SWAP",
        contracts=2,
        entry_price=50000.0,
        stop_loss=55000.0,
        take_profit=45000.0,
        backstop_at=datetime.now(UTC),
        strategy_id="test",
    )
    defaults.update(kwargs)
    return ShortSwapPlan(**defaults)


def test_short_plan_harvest_mode_defaults_false():
    plan = make_short_plan()
    assert plan.harvest_mode is False


def test_short_plan_harvest_mode_can_be_true():
    plan = make_short_plan(harvest_mode=True)
    assert plan.harvest_mode is True


def test_short_plan_summary_includes_harvest():
    plan = make_short_plan(harvest_mode=True)
    summary = plan.summary()
    assert "HARVEST" in summary or "harvest" in summary.lower()


def test_short_plan_summary_no_harvest_label_when_false():
    plan = make_short_plan(harvest_mode=False)
    summary = plan.summary()
    assert "HARVEST" not in summary


# ── PositionMonitor public API: iter_running_plans + unregister_short_plan ────

def make_mock_pm():
    """Creates a minimal PositionMonitor-like object for unit testing."""
    from unittest.mock import MagicMock
    pm = MagicMock()
    pm._plans       = {}
    pm._short_plans = {}
    pm._exits_today = 0

    # Wire the real methods
    from src.oms.position_monitor import PositionMonitor
    pm.iter_running_plans    = PositionMonitor.iter_running_plans.__get__(pm)
    pm.unregister_short_plan = PositionMonitor.unregister_short_plan.__get__(pm)
    return pm


def test_iter_running_plans_empty():
    pm = make_mock_pm()
    result = list(pm.iter_running_plans())
    assert result == []


def test_iter_running_plans_filters_non_running():
    pm = make_mock_pm()
    plan_running = make_exit_plan()
    plan_running.running_mode = True
    plan_stopped = make_exit_plan(symbol="ETH-USDT")
    plan_stopped.running_mode = False
    pm._plans["BTC-USDT"] = plan_running
    pm._plans["ETH-USDT"] = plan_stopped

    result = dict(pm.iter_running_plans())
    assert "BTC-USDT" in result
    assert "ETH-USDT" not in result


def test_iter_running_plans_returns_all_running():
    pm = make_mock_pm()
    for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
        p = make_exit_plan(symbol=sym)
        p.running_mode = True
        pm._plans[sym] = p

    result = list(pm.iter_running_plans())
    assert len(result) == 3


def test_unregister_short_plan_removes_plan():
    pm = make_mock_pm()
    pm._short_plans["BTC-USDT"] = make_short_plan()
    pm.unregister_short_plan("BTC-USDT")
    assert "BTC-USDT" not in pm._short_plans


def test_unregister_short_plan_increments_exits_today():
    pm = make_mock_pm()
    pm._short_plans["ETH-USDT"] = make_short_plan(symbol="ETH-USDT")
    initial = pm._exits_today
    pm.unregister_short_plan("ETH-USDT")
    assert pm._exits_today == initial + 1


def test_unregister_short_plan_noop_if_missing():
    """Should not raise if symbol not in _short_plans."""
    pm = make_mock_pm()
    pm.unregister_short_plan("NONEXISTENT-USDT")  # should not raise
    assert pm._exits_today == 0
