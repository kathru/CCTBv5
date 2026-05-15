from src.risk.kill_switch import KillSwitch, KillSwitchState


def test_initial_state_is_armed():
    ks = KillSwitch()
    assert ks.state == KillSwitchState.ARMED
    assert ks.is_armed is True
    assert ks.allows_new_entries is True
    assert ks.allows_any_operation is True


def test_soft_kill_blocks_entries():
    ks = KillSwitch()
    ks.trigger_soft("test_reason")
    assert ks.state == KillSwitchState.SOFT
    assert ks.allows_new_entries is False
    assert ks.allows_any_operation is True  # SOFT still allows operations (closing)


def test_hard_kill_blocks_everything():
    ks = KillSwitch()
    ks.trigger_hard("critical_error")
    assert ks.state == KillSwitchState.HARD
    assert ks.allows_new_entries is False
    assert ks.allows_any_operation is False


def test_soft_can_be_reset():
    ks = KillSwitch()
    ks.trigger_soft("reason")
    result = ks.reset_soft()
    assert result is True
    assert ks.is_armed is True


def test_hard_cannot_be_reset_by_soft_reset():
    ks = KillSwitch()
    ks.trigger_hard("critical")
    result = ks.reset_soft()
    assert result is False
    assert ks.state == KillSwitchState.HARD


def test_hard_requires_manual_reset():
    ks = KillSwitch()
    ks.trigger_hard("critical")
    ks.reset_hard("manual_intervention")
    assert ks.is_armed is True


def test_soft_ignored_when_hard_active():
    ks = KillSwitch()
    ks.trigger_hard("hard_reason")
    ks.trigger_soft("soft_reason")  # should be ignored
    assert ks.state == KillSwitchState.HARD
