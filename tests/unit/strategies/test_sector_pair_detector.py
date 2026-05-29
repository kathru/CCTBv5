"""Tests for SectorPairDetector — Z-score and pair logic."""


from src.strategies.pairs.sector_pair_detector import (
    DIV_THRESHOLD,
    PAIRS,
    Z_ENTRY,
    Z_EXIT,
    SectorPairDetector,
)


def make_detector():
    det = SectorPairDetector.__new__(SectorPairDetector)
    det._cache           = None
    det._okx             = None
    det._portfolio_value = 10000.0
    det._running         = False
    det._task            = None
    det._active          = {}
    det._ratio_history   = {}
    return det


# ── Z-score calculation ───────────────────────────────────────────────────────

def test_zscore_of_mean_is_zero():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = sum(data) / len(data)
    z = det_zscore(data, mean)
    assert abs(z) < 1e-9


def det_zscore(history: list, current: float) -> float:
    """Replicate the zscore logic from SectorPairDetector."""
    if len(history) < 2:
        return 0.0
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    std = variance ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


def test_zscore_high_value():
    data = [1.0] * 20
    z = det_zscore(data, 3.0)
    # std ~= 0 → should return 0 to avoid division by zero
    assert z == 0.0


def test_zscore_normal_distribution():
    data = [float(i) for i in range(20)]
    z = det_zscore(data, 25.0)
    # 25 is far from mean (~9.5) → high z-score
    assert z > 2.0


def test_zscore_negative():
    data = [float(i) for i in range(20)]
    z = det_zscore(data, -5.0)
    assert z < -2.0


# ── PAIRS configuration ───────────────────────────────────────────────────────

def test_pairs_are_defined():
    assert len(PAIRS) >= 3


def test_pairs_contain_btc_eth_sol():
    pair_keys = set()
    for a, b in PAIRS:
        pair_keys.add(a)
        pair_keys.add(b)
    assert "BTC-USDT" in pair_keys
    assert "ETH-USDT" in pair_keys
    assert "SOL-USDT" in pair_keys


def test_pairs_no_self_reference():
    for a, b in PAIRS:
        assert a != b


# ── _resolve_original_pair ────────────────────────────────────────────────────

def test_resolve_original_pair_finds_canonical_pair():
    """Finds the canonical PAIRS order regardless of long/short assignment."""
    from src.strategies.pairs.sector_pair_detector import PairTrade
    det = make_detector()
    trade = PairTrade(
        long_sym="ETH-USDT",
        short_sym="BTC-USDT",
        short_swap_sym="BTC-USDT-SWAP",
        long_qty=0.5,
        short_contracts=1,
        entry_long_price=2000.0,
        entry_short_price=50000.0,
        entry_ratio=0.04,
        entry_zscore=2.5,
        notional=1000.0,
    )
    sym_a, sym_b = det._resolve_original_pair(trade)
    # One of the two PAIRS orderings should be returned
    assert {sym_a, sym_b} == {"BTC-USDT", "ETH-USDT"}


def test_resolve_original_pair_fallback():
    """Falls back to trade order when pair not in PAIRS."""
    from src.strategies.pairs.sector_pair_detector import PairTrade
    det = make_detector()
    trade = PairTrade(
        long_sym="FAKE-USDT",
        short_sym="COIN-USDT",
        short_swap_sym="COIN-USDT-SWAP",
        long_qty=1.0,
        short_contracts=1,
        entry_long_price=100.0,
        entry_short_price=200.0,
        entry_ratio=0.5,
        entry_zscore=2.0,
        notional=500.0,
    )
    sym_a, sym_b = det._resolve_original_pair(trade)
    assert sym_a == "FAKE-USDT"
    assert sym_b == "COIN-USDT"


# ── Constants sanity ──────────────────────────────────────────────────────────

def test_constants_sane():
    assert Z_ENTRY > Z_EXIT >= 0
    assert 0 < DIV_THRESHOLD < 0.5
