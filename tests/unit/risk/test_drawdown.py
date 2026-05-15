from src.risk.drawdown import DrawdownEngine


def test_no_drawdown_on_flat_portfolio():
    engine = DrawdownEngine()
    engine.update(10000.0)
    engine.update(10000.0)
    assert engine.daily_drawdown == 0.0


def test_daily_drawdown_calculated_correctly():
    engine = DrawdownEngine(max_daily_dd=0.03)
    engine.update(10000.0)  # sets day_start_value
    engine.update(9700.0)   # 3% down
    assert abs(engine.daily_drawdown - 0.03) < 0.001


def test_daily_limit_breached():
    engine = DrawdownEngine(max_daily_dd=0.03)
    engine.update(10000.0)
    engine.update(9600.0)   # 4% down — breach
    breached, reason = engine.is_daily_limit_breached()
    assert breached is True
    assert "breached" in reason


def test_daily_limit_not_breached():
    engine = DrawdownEngine(max_daily_dd=0.03)
    engine.update(10000.0)
    engine.update(9800.0)   # 2% down — ok
    breached, _ = engine.is_daily_limit_breached()
    assert breached is False


def test_peak_updates_correctly():
    engine = DrawdownEngine()
    engine.update(10000.0)
    engine.update(11000.0)  # new peak
    engine.update(10000.0)  # back down
    assert abs(engine.peak_drawdown - 1000/11000) < 0.001


def test_acceleration_detected():
    engine = DrawdownEngine(acceleration_threshold=0.01)
    # Declining values
    for v in [10000, 9900, 9800, 9700, 9600, 9500, 9400, 9300, 9200, 9100]:
        engine.update(float(v))
    accelerating, reason = engine.is_accelerating()
    assert accelerating is True
