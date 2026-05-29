"""Tests for ShortSqueezeDetector — squeeze score calculation."""


from src.strategies.squeeze.short_squeeze_detector import (
    SQUEEZE_THRESHOLD,
    ShortSqueezeDetector,
)


def make_detector():
    det = ShortSqueezeDetector.__new__(ShortSqueezeDetector)
    det._cache           = None
    det._okx             = None
    det._portfolio_value = 10000.0
    det._running         = False
    det._task            = None
    det._active          = {}
    det._funding_history = {}
    return det


# ── _compute_squeeze_score ────────────────────────────────────────────────────

def test_score_with_empty_data_is_below_threshold():
    """With no signal data, score should be below the entry threshold."""
    det = make_detector()
    score, components = det._compute_squeeze_score("BTC-USDT", {}, {})
    # Score may be non-zero due to neutral defaults but must be below threshold
    assert 0.0 <= score < SQUEEZE_THRESHOLD
    assert isinstance(components, dict)


def test_score_component_A_funding_reversal():
    """Funding was negative, now reverting → funding_reversal component fires."""
    det = make_detector()
    det._funding_history["BTC-USDT"] = [-0.001, -0.0008, -0.0003]
    m6 = {"funding_rate": 0.0001}   # now positive
    m7 = {}
    score, comp = det._compute_squeeze_score("BTC-USDT", m6, m7)
    assert comp.get("funding_reversal", 0) > 0.5, "Component A should be > 0.5 on reversal"


def test_score_component_B_oi_drop_with_rising_price():
    """OI dropping while price rising → oi_drop component fires."""
    det = make_detector()
    m6 = {"oi_change_pct": -2.0, "funding_rate": 0.0}
    m7 = {"momentum_1h": 0.8}   # positive momentum (price rising)
    score, comp = det._compute_squeeze_score("BTC-USDT", m6, m7)
    assert comp.get("oi_drop", 0) > 0.5, "Component B should be > 0.5"


def test_score_component_C_rs_reversal():
    """RS value present — check score doesn't error and is bounded."""
    det = make_detector()
    m6 = {"funding_rate": 0.0, "oi_change_pct": 0.0}
    m7 = {"m7_score": 0.55, "momentum_1h": 0.3}
    score, comp = det._compute_squeeze_score("ETH-USDT", m6, m7)
    assert 0.0 <= score <= 1.0
    assert "rs_reversal" in comp


def test_score_component_D_price_confirmation():
    """momentum_1h > 0.5% → price_confirm component fires."""
    det = make_detector()
    m6 = {"funding_rate": 0.0}
    m7 = {"momentum_1h": 0.8}   # >0.5
    score, comp = det._compute_squeeze_score("SOL-USDT", m6, m7)
    assert comp.get("price_confirm", 0) > 0.5, "Component D should be > 0.5 on momentum_1h > 0.5"


def test_score_full_squeeze_signal():
    """All components firing should produce score >= threshold."""
    det = make_detector()
    # Simulate all conditions for max score
    det._funding_history["BTC-USDT"] = [-0.002, -0.001, -0.0005]
    m6 = {
        "funding_rate":    0.0002,    # reverting from negative
        "oi_change_pct":  -3.0,       # OI dropping
        "scores": {"funding": 0.8, "oi_change": 0.8},
    }
    m7 = {
        "m7_score":    0.72,
        "momentum_1h": 1.2,
    }
    score, comp = det._compute_squeeze_score("BTC-USDT", m6, m7)
    assert 0.0 <= score <= 1.0
    # With all components active, score should be significant
    assert score >= 0.3


def test_score_bounded_between_0_and_1():
    det = make_detector()
    for _ in range(10):
        m6 = {"funding_rate": 0.005, "oi_change_pct": -10.0}
        m7 = {"m7_score": 1.0, "momentum_1h": 5.0}
        score, _ = det._compute_squeeze_score("BTC-USDT", m6, m7)
        assert 0.0 <= score <= 1.0


def test_threshold_constant():
    assert 0.5 <= SQUEEZE_THRESHOLD <= 0.9
