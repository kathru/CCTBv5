import pytest

from src.strategies.meta_layer import MetaStrategyLayer, StrategyEdgeState


@pytest.fixture
def meta():
    m = MetaStrategyLayer(window_size=10)
    m.register("strat_a")
    return m


def test_initial_weight_is_one(meta):
    assert meta.get_weight("strat_a") == 1.0
    assert meta.is_active("strat_a") is True


def test_expanding_after_wins(meta):
    for _ in range(8):
        meta.record_trade("strat_a", pnl=0.02)
    assert meta._performances["strat_a"].edge_state == StrategyEdgeState.EXPANDING
    assert meta.get_weight("strat_a") > 1.0


def test_degrading_after_losses(meta):
    for _ in range(8):
        meta.record_trade("strat_a", pnl=-0.01)
    assert meta._performances["strat_a"].edge_state in {
        StrategyEdgeState.DEGRADING, StrategyEdgeState.SUSPENDED
    }


def test_suspended_after_consecutive_negative_windows(meta):
    # Fill window with losses
    for _ in range(10):
        meta.record_trade("strat_a", pnl=-0.02)
    assert meta._performances["strat_a"].edge_state == StrategyEdgeState.SUSPENDED
    assert meta.get_weight("strat_a") == 0.0
    assert meta.is_active("strat_a") is False


def test_recovering_after_suspension(meta):
    for _ in range(10):
        meta.record_trade("strat_a", pnl=-0.02)
    assert meta.is_active("strat_a") is False
    # Record multiple wins to push window into positive territory
    for _ in range(6):
        meta.record_trade("strat_a", pnl=0.05)
    state = meta._performances["strat_a"].edge_state
    assert state in {StrategyEdgeState.RECOVERING, StrategyEdgeState.HEALTHY}
    assert meta.is_active("strat_a") is True


def test_apply_weight_clamps_to_zero_one(meta):
    # Suspend strategy
    for _ in range(10):
        meta.record_trade("strat_a", pnl=-0.05)
    result = meta.apply_weight("strat_a", score=0.8)
    assert result == 0.0


def test_apply_weight_amplifies_good_strategy(meta):
    for _ in range(8):
        meta.record_trade("strat_a", pnl=0.05)
    result = meta.apply_weight("strat_a", score=0.7)
    assert result > 0.7   # weight > 1.0 amplifies score
    assert result <= 1.0  # clamped


def test_unregistered_strategy_gets_default_weight(meta):
    assert meta.get_weight("unknown_strat") == 1.0
    assert meta.is_active("unknown_strat") is True


def test_status_contains_all_strategies(meta):
    meta.register("strat_b")
    status = meta.status()
    assert "strat_a" in status
    assert "strat_b" in status
    assert "edge_state" in status["strat_a"]
    assert "weight" in status["strat_a"]
