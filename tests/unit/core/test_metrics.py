import pytest

from src.metrics.infra_metrics import InfraMetrics
from src.metrics.strategy_metrics import StrategyMetrics, compute_metrics

# ── Strategy Metrics ──────────────────────────────────────────

def test_empty_returns_gives_zero_metrics():
    m = compute_metrics([])
    assert m.total_trades == 0
    assert m.sharpe_ratio == 0.0
    assert m.win_rate == 0.0


def test_all_wins():
    returns = [0.02] * 10
    m = compute_metrics(returns)
    assert m.win_rate == 1.0
    assert m.profit_factor == float("inf")
    assert m.total_return_pct == pytest.approx(0.20, rel=1e-3)


def test_all_losses():
    returns = [-0.01] * 10
    m = compute_metrics(returns)
    assert m.win_rate == 0.0
    assert m.total_return_pct == pytest.approx(-0.10, rel=1e-3)


def test_mixed_returns():
    returns = [0.03, -0.01, 0.02, -0.01, 0.04, -0.02, 0.01]
    m = compute_metrics(returns)
    assert 0.0 < m.win_rate < 1.0
    assert m.profit_factor > 0
    assert m.sharpe_ratio != 0.0
    assert m.max_drawdown_pct <= 0.0


def test_sharpe_positive_for_good_strategy():
    # Consistently positive returns → positive Sharpe
    returns = [0.01] * 50 + [-0.005] * 10
    m = compute_metrics(returns)
    assert m.sharpe_ratio > 0


def test_sortino_better_than_sharpe_for_asymmetric_returns():
    # Strategy with big wins, small losses → Sortino > Sharpe
    returns = [0.05, -0.005, 0.05, -0.005, 0.05, -0.005]
    m = compute_metrics(returns)
    assert m.sortino_ratio >= m.sharpe_ratio


def test_max_drawdown_is_negative():
    returns = [0.02, 0.02, -0.05, -0.03, 0.01]
    m = compute_metrics(returns)
    assert m.max_drawdown_pct < 0


def test_grade_returns_valid_grade():
    m = StrategyMetrics(sharpe_ratio=2.5, profit_factor=2.5)
    assert m.grade() == "A"
    m2 = StrategyMetrics(sharpe_ratio=0.3, profit_factor=0.8)
    assert m2.grade() == "D"


def test_to_dict_has_required_keys():
    m = compute_metrics([0.01, -0.005, 0.02])
    d = m.to_dict()
    required = {"sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor",
                "expectancy", "max_drawdown", "total_trades"}
    assert required.issubset(d.keys())


# ── Infra Metrics ─────────────────────────────────────────────

def test_infra_starts_healthy():
    m = InfraMetrics()
    m.record_ws_connect()
    m.record_ws_message()
    # Freshly started — uptime should be high
    snap = m.snapshot()
    assert snap.error_count == 0
    assert snap.reconnect_count == 0


def test_infra_reconnect_count():
    m = InfraMetrics()
    m.record_ws_connect()
    m.record_ws_disconnect()
    m.record_ws_connect()
    snap = m.snapshot()
    assert snap.reconnect_count >= 1


def test_infra_records_latencies():
    m = InfraMetrics()
    m.record_event_lag(12.5)
    m.record_event_lag(15.0)
    m.record_order_latency(45.0)
    snap = m.snapshot()
    assert abs(snap.avg_event_lag_ms - 13.75) < 0.1
    assert snap.avg_order_latency_ms == 45.0


def test_infra_snapshot_to_dict():
    m = InfraMetrics()
    m.record_ws_connect()
    snap = m.snapshot()
    d = snap.to_dict()
    assert "ws_uptime_pct" in d
    assert "reconnect_count" in d
    assert "avg_event_lag_ms" in d
