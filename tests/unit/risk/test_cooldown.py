import pytest
from datetime import timedelta
from src.risk.cooldown import CooldownEngine


def test_no_cooldown_initially():
    engine = CooldownEngine(max_consecutive_losses=3)
    allowed, _ = engine.is_allowed("strat_a")
    assert allowed is True


def test_cooldown_activates_after_threshold():
    engine = CooldownEngine(max_consecutive_losses=3, cooldown_duration=timedelta(hours=1))
    engine.record_loss("strat_a")
    engine.record_loss("strat_a")
    activated = engine.record_loss("strat_a")   # 3rd loss
    assert activated is True
    allowed, reason = engine.is_allowed("strat_a")
    assert allowed is False
    assert "cooldown" in reason


def test_win_resets_loss_counter():
    engine = CooldownEngine(max_consecutive_losses=3)
    engine.record_loss("strat_a")
    engine.record_loss("strat_a")
    engine.record_win("strat_a")    # resets counter
    activated = engine.record_loss("strat_a")   # only 1 loss now
    assert activated is False


def test_cooldown_is_per_strategy():
    engine = CooldownEngine(max_consecutive_losses=3)
    for _ in range(3):
        engine.record_loss("strat_a")
    allowed_a, _ = engine.is_allowed("strat_a")
    allowed_b, _ = engine.is_allowed("strat_b")
    assert allowed_a is False
    assert allowed_b is True   # strat_b unaffected


def test_force_reset_clears_cooldown():
    engine = CooldownEngine(max_consecutive_losses=3)
    for _ in range(3):
        engine.record_loss("strat_a")
    engine.force_reset("strat_a")
    allowed, _ = engine.is_allowed("strat_a")
    assert allowed is True
