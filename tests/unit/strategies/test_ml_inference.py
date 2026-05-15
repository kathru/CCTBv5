import pytest
import numpy as np
from datetime import datetime, UTC
from pathlib import Path

from src.core.models import Candle
from src.strategies.ml.inference import (
    PlattCalibrator, FeatureExtractor, MLInferenceEngine
)


def make_candles(n: int = 30) -> list[Candle]:
    candles = []
    price = 42000.0
    for i in range(n):
        candles.append(Candle(
            symbol="BTC-USDT",
            granularity="1H",
            timestamp=datetime.now(UTC),
            open=price - 50,
            high=price + 100,
            low=price - 100,
            close=price + i * 10,
            volume=500.0 + i,
            confirmed=True,
        ))
    return candles


# ── PlattCalibrator ───────────────────────────────────────────

def test_calibrator_uses_defaults_when_no_file(tmp_path):
    cal = PlattCalibrator(coef_path=tmp_path / "missing.json")
    assert cal.is_using_defaults is True


def test_calibrator_loads_from_file(tmp_path):
    import json
    coef_path = tmp_path / "coef.json"
    coef_path.write_text(json.dumps({"A": -3.0, "B": 1.5}))
    cal = PlattCalibrator(coef_path=coef_path)
    assert cal.is_using_defaults is False
    assert cal._A == -3.0


def test_calibrate_score_between_zero_and_one():
    cal = PlattCalibrator()
    for score in [0.0, 0.3, 0.5, 0.7, 1.0]:
        result = cal.calibrate(score)
        assert 0.0 <= result <= 1.0


def test_higher_score_gives_higher_calibrated():
    cal = PlattCalibrator()
    assert cal.calibrate(0.8) > cal.calibrate(0.3)


# ── FeatureExtractor ─────────────────────────────────────────

def test_extractor_returns_none_with_insufficient_candles():
    extractor = FeatureExtractor()
    result = extractor.extract(make_candles(5))
    assert result is None


def test_extractor_returns_correct_size():
    extractor = FeatureExtractor()
    result = extractor.extract(make_candles(30))
    assert result is not None
    assert len(result) == FeatureExtractor.FEATURE_SIZE


def test_extractor_returns_finite_values():
    extractor = FeatureExtractor()
    result = extractor.extract(make_candles(30))
    assert result is not None
    assert np.all(np.isfinite(result))


# ── MLInferenceEngine ─────────────────────────────────────────

def test_engine_creates_without_model_files(tmp_path):
    engine = MLInferenceEngine(models_dir=tmp_path)
    assert engine is not None
    assert engine.calibrator_status["using_defaults"] is True


def test_engine_calibrate_score(tmp_path):
    engine = MLInferenceEngine(models_dir=tmp_path)
    result = engine.calibrate_score(0.7)
    assert 0.0 <= result <= 1.0


def test_engine_predict_regime_fallback(tmp_path):
    engine = MLInferenceEngine(models_dir=tmp_path)
    candles = make_candles(30)
    prob = engine.predict_regime_probability(candles, "TREND_EXPANSION")
    assert 0.0 <= prob <= 1.0


def test_engine_predict_returns_half_insufficient_data(tmp_path):
    engine = MLInferenceEngine(models_dir=tmp_path)
    prob = engine.predict_regime_probability(make_candles(5), "TREND_EXPANSION")
    assert prob == 0.5
