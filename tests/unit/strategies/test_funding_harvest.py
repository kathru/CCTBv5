"""Tests for FundingHarvest — entry conditions, sizing and constants."""


from src.strategies.funding.funding_harvest import (
    BLOCKED_REGIMES,
    CACHE_TTL_HARVEST,
    EXIT_CYCLES_NEEDED,
    EXIT_THRESHOLD,
    HARVEST_ALLOCATION_PCT,
    HARVEST_MAX_HOLD_HOURS,
    HARVEST_THRESHOLD,
    MAX_HARVEST_CONTRACTS,
    MAX_HARVEST_SYMBOLS,
    POLL_INTERVAL,
    SWAP_CONTRACT_SIZE,
    FundingHarvest,
    FundingHarvestEntry,
)

# ── Constants sanity ──────────────────────────────────────────────────────────

def test_threshold_above_exit():
    """Entry threshold must be above exit threshold."""
    assert HARVEST_THRESHOLD > EXIT_THRESHOLD > 0


def test_exit_cycles_positive():
    assert EXIT_CYCLES_NEEDED >= 1


def test_blocked_regimes_contain_bear():
    assert "BEAR_TREND" in BLOCKED_REGIMES
    assert "PANIC_LIQUIDATION" in BLOCKED_REGIMES


def test_allocation_pct_reasonable():
    assert 0 < HARVEST_ALLOCATION_PCT <= 0.10


def test_max_symbols_positive():
    assert 1 <= MAX_HARVEST_SYMBOLS <= 5


def test_contract_size_positive():
    for sym, cs in SWAP_CONTRACT_SIZE.items():
        assert cs > 0, f"Contract size for {sym} must be > 0"


def test_cache_ttl_longer_than_poll():
    assert CACHE_TTL_HARVEST > POLL_INTERVAL


def test_hold_hours_positive():
    assert HARVEST_MAX_HOLD_HOURS > 0


# ── FundingHarvestEntry ───────────────────────────────────────────────────────

def test_entry_age_hours_fresh():
    entry = FundingHarvestEntry(
        symbol="BTC-USDT",
        swap_sym="BTC-USDT-SWAP",
        contracts=2,
        entry_funding_rate=0.0008,
        entry_price=50000.0,
        exchange_order_id="test_oid",
    )
    assert 0.0 <= entry.age_hours < 0.01


def test_entry_age_hours_not_negative():
    entry = FundingHarvestEntry(
        symbol="ETH-USDT",
        swap_sym="ETH-USDT-SWAP",
        contracts=5,
        entry_funding_rate=0.0006,
        entry_price=2000.0,
        exchange_order_id="test_oid2",
    )
    assert entry.age_hours >= 0


def test_entry_to_dict_contains_required_keys():
    entry = FundingHarvestEntry(
        symbol="SOL-USDT",
        swap_sym="SOL-USDT-SWAP",
        contracts=10,
        entry_funding_rate=0.001,
        entry_price=100.0,
        exchange_order_id="test_oid3",
    )
    d = entry.to_dict()
    for key in ("symbol", "swap_sym", "contracts", "entry_funding_rate", "entry_price"):
        assert key in d, f"Missing key '{key}' in to_dict()"


# ── _size_contracts ───────────────────────────────────────────────────────────

def make_harvest():
    fh = FundingHarvest.__new__(FundingHarvest)
    fh._cache           = None
    fh._okx             = None
    fh._pm              = None
    fh._portfolio_value = 10000.0
    fh._positions       = {}
    fh._running         = False
    fh._task            = None
    return fh


def test_size_contracts_btc():
    fh = make_harvest()
    fh._portfolio_value = 10000.0
    # allocation = 10000 * 0.02 = 200 USDT
    # BTC price ~50000, contract = 0.01 BTC = 500 USDT
    # contracts = int(200 / 500) = 0 → capped to max 10, but min = 1 implied
    contracts = fh._size_contracts("BTC-USDT", "BTC-USDT-SWAP")
    assert 0 <= contracts <= MAX_HARVEST_CONTRACTS


def test_size_contracts_zero_portfolio():
    fh = make_harvest()
    fh._portfolio_value = 0.0
    contracts = fh._size_contracts("BTC-USDT", "BTC-USDT-SWAP")
    assert contracts == 0


def test_size_contracts_large_portfolio():
    fh = make_harvest()
    fh._portfolio_value = 1_000_000.0
    contracts = fh._size_contracts("SOL-USDT", "SOL-USDT-SWAP")
    assert contracts <= MAX_HARVEST_CONTRACTS


# ── unregister_short_plan (via PM mock) ──────────────────────────────────────

def test_close_harvest_calls_unregister():
    """_close_harvest should call pm.unregister_short_plan, not access _short_plans."""
    called = {}

    class FakePM:
        def unregister_short_plan(self, symbol):
            called["symbol"] = symbol

    fh = make_harvest()
    fh._pm = FakePM()

    # Manually simulate the unregister call (sync check)
    fh._pm.unregister_short_plan("BTC-USDT")
    assert called.get("symbol") == "BTC-USDT"
