"""Tests for RegimeAwarePairEngine — mode params and pair selection."""

from datetime import UTC, datetime

import pytest

from src.strategies.cross_asset.regime_aware_pair_engine import (
    MODE_BEAR,
    MODE_BULL,
    MODE_CHOP,
    MODE_IDLE,
    MODE_PARAMS,
    REGIME_TO_MODE,
    SWAP_CONTRACT_SIZE,
    SWAP_SYMBOLS,
    RegimeAwarePairEngine,
    RegimePairPosition,
)

# ── MODE_PARAMS sanity ────────────────────────────────────────────────────────

def test_all_modes_have_required_keys():
    required = {"spread_threshold", "spread_exit", "max_hold_hours",
                "max_drawdown", "allocation_pct", "reverse_direction"}
    for mode, params in MODE_PARAMS.items():
        missing = required - params.keys()
        assert not missing, f"Mode {mode} missing keys: {missing}"


def test_bear_has_tighter_params_than_bull():
    """BEAR regime should have tighter drawdown and shorter hold than BULL."""
    assert MODE_PARAMS[MODE_BEAR]["max_drawdown"] < MODE_PARAMS[MODE_BULL]["max_drawdown"]
    assert MODE_PARAMS[MODE_BEAR]["max_hold_hours"] < MODE_PARAMS[MODE_BULL]["max_hold_hours"]
    assert MODE_PARAMS[MODE_BEAR]["allocation_pct"] < MODE_PARAMS[MODE_BULL]["allocation_pct"]


def test_chop_reverses_direction():
    """CHOP mode trades in reverse (mean reversion)."""
    assert MODE_PARAMS[MODE_CHOP]["reverse_direction"] is True
    assert MODE_PARAMS[MODE_BULL]["reverse_direction"] is False
    assert MODE_PARAMS[MODE_BEAR]["reverse_direction"] is False


def test_spread_exit_below_spread_entry():
    for mode, p in MODE_PARAMS.items():
        assert p["spread_exit"] < p["spread_threshold"], (
            f"Mode {mode}: spread_exit must be below spread_threshold"
        )


# ── REGIME_TO_MODE mapping ────────────────────────────────────────────────────

def test_trend_expansion_maps_to_bull():
    assert REGIME_TO_MODE["TREND_EXPANSION"] == MODE_BULL


def test_bear_trend_maps_to_bear():
    assert REGIME_TO_MODE["BEAR_TREND"] == MODE_BEAR


def test_mean_reverting_chop_maps_to_chop():
    assert REGIME_TO_MODE["MEAN_REVERTING_CHOP"] == MODE_CHOP


def test_high_correlation_maps_to_idle():
    assert REGIME_TO_MODE.get("HIGH_CORRELATION_RISK") == MODE_IDLE


def test_all_mapped_modes_exist_in_params():
    """Every non-IDLE regime mode must have entries in MODE_PARAMS."""
    for regime, mode in REGIME_TO_MODE.items():
        if mode != MODE_IDLE:
            assert mode in MODE_PARAMS, f"Mode {mode} for regime {regime} not in MODE_PARAMS"


# ── RegimePairPosition ────────────────────────────────────────────────────────

def make_position(long_px=50000.0, short_px=2000.0):
    return RegimePairPosition(
        mode=MODE_BULL,
        long_symbol="BTC-USDT",
        short_symbol="ETH-USDT",
        short_swap_sym="ETH-USDT-SWAP",
        entry_long_price=long_px,
        entry_short_price=short_px,
        long_qty=0.02,
        short_contracts=1,
        notional=1000.0,
        entry_spread=0.12,
    )


def test_position_pnl_zero_at_entry():
    pos = make_position()
    pnl = pos.pnl_pct(50000.0, 2000.0)
    assert abs(pnl) < 1e-9


def test_position_pnl_positive_when_spread_converges():
    """Long goes up, short goes down → both legs profitable."""
    pos = make_position(long_px=50000.0, short_px=2000.0)
    pnl = pos.pnl_pct(52000.0, 1900.0)
    assert pnl > 0


def test_position_age_hours_is_non_negative():
    pos = make_position()
    assert pos.age_hours >= 0.0


# ── SWAP_SYMBOLS and contract sizes ──────────────────────────────────────────

def test_swap_symbols_complete():
    for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
        assert sym in SWAP_SYMBOLS
        assert SWAP_SYMBOLS[sym].endswith("-SWAP")


def test_contract_sizes_positive():
    for swap_sym, cs in SWAP_CONTRACT_SIZE.items():
        assert cs > 0, f"Contract size for {swap_sym} must be > 0"


# ── Recovery from cache (no crash on empty cache) ────────────────────────────

@pytest.mark.asyncio
async def test_recover_from_empty_cache():
    """Recovery with no cached position should not raise."""
    class FakeCache:
        async def get(self, key):
            return None
    eng = RegimeAwarePairEngine.__new__(RegimeAwarePairEngine)
    eng._cache           = FakeCache()
    eng._okx             = None
    eng._portfolio_value = 10000.0
    eng._running         = False
    eng._task            = None
    eng._position        = None
    eng._current_mode    = MODE_IDLE
    await eng._recover_position_from_cache()
    assert eng._position is None   # nothing to recover


@pytest.mark.asyncio
async def test_recover_from_valid_cache():
    """Recovery with cached position should reconstruct _position."""
    import json
    cached = {
        "mode": MODE_BULL,
        "long": "BTC-USDT",
        "short": "ETH-USDT",
        "spread": 0.12,
        "notional": 5000.0,
        "opened_at": datetime.now(UTC).isoformat(),
        "regime": "TREND_EXPANSION",
    }

    class FakeCache:
        async def get(self, key):
            return json.dumps(cached) if key == "regime_pair:position" else None

    eng = RegimeAwarePairEngine.__new__(RegimeAwarePairEngine)
    eng._cache           = FakeCache()
    eng._okx             = None
    eng._portfolio_value = 10000.0
    eng._running         = False
    eng._task            = None
    eng._position        = None
    eng._current_mode    = MODE_IDLE

    await eng._recover_position_from_cache()
    assert eng._position is not None
    assert eng._position.long_symbol == "BTC-USDT"
    assert eng._position.short_symbol == "ETH-USDT"
    assert eng._current_mode == MODE_BULL
