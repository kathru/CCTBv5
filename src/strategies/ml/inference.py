"""
ML Inference Layer — lightweight inference only.

Design principles:
  - Training happens LOCALLY or on Colab/RunPod
  - Oracle Free Tier runs INFERENCE ONLY (no training)
  - Models are serialized to JSON/numpy files and loaded at startup
  - Inference must be fast (< 10ms per call)
  - Falls back gracefully if model files are missing

What lives here:
  1. Model loader (loads from data/models/)
  2. Feature extractor (candles → feature vector)
  3. Regime classifier (predicts market regime probability)
  4. Score calibrator (re-calibrates signal scores — extends Platt scaling)

What does NOT live here:
  - Training code (in scripts/train_*.py, run locally)
  - Heavy compute (NumPy only, no PyTorch/TF on Oracle)
"""

import json
import logging
import math
from pathlib import Path

import numpy as np

from ...core.models import Candle

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data") / "models"


class PlattCalibrator:
    """
    Platt scaling calibration — migrated from V4.

    Converts raw signal scores (0-1) to calibrated probabilities.
    Coefficients (A, B) are trained locally and saved to JSON.

    Formula: P = 1 / (1 + exp(-(A * score + B)))
    """

    DEFAULT_A = 2.5    # placeholder — replace with trained values from calibrate.py
    DEFAULT_B = -1.2   # score=0.5 → ~0.5 probability with these defaults

    def __init__(self, coef_path: Path | None = None) -> None:
        self._A = self.DEFAULT_A
        self._B = self.DEFAULT_B
        self._loaded = False
        self._regime_thresholds: dict[str, float] = {}

        if coef_path and coef_path.exists():
            self._load(coef_path)
        else:
            logger.warning(
                "PlattCalibrator: no coef file found — using defaults. "
                "Run calibrate.py locally to generate coefficients."
            )

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            # Support both V4 format (platt_a/platt_b) and generic (A/B)
            self._A = float(data.get("platt_a", data.get("A", self.DEFAULT_A)))
            self._B = float(data.get("platt_b", data.get("B", self.DEFAULT_B)))
            self._regime_thresholds = data.get("regime_thresholds", {})
            self._loaded = True
            calibrated_at = data.get("calibrated_at", "desconhecido")
            win_rate = data.get("win_rate", None)
            logger.info(
                "PlattCalibrator loaded A=%.4f B=%.4f win_rate=%s calibrated_at=%s from %s",
                self._A, self._B,
                f"{win_rate:.1%}" if win_rate else "n/a",
                calibrated_at[:10] if calibrated_at else "?",
                path,
            )
        except Exception as exc:
            logger.error("PlattCalibrator load failed: %s — using defaults", exc)

    def calibrate(self, raw_score: float) -> float:
        """Convert raw score to calibrated probability."""
        return 1.0 / (1.0 + math.exp(-(self._A * raw_score + self._B)))

    def get_regime_threshold(self, regime: str, default: float) -> float:
        """Return calibrated threshold for a regime, or the hardcoded default."""
        return self._regime_thresholds.get(regime, default)

    @property
    def is_using_defaults(self) -> bool:
        return not self._loaded


class FeatureExtractor:
    """
    Extracts a fixed-size feature vector from candles.
    Same features used in training must be used in inference.

    Features (21 total):
      - Returns: 1, 3, 5, 10, 20 period
      - Volatility: rolling std of returns (5, 10, 20)
      - Volume ratio: current / mean (5, 10)
      - Price position: (close - low) / (high - low) for 1, 5
      - Momentum: RSI-like oscillator (5, 10, 14)
      - ATR ratio: ATR / close (5, 10)
    """

    FEATURE_SIZE = 18

    def extract(self, candles: list[Candle]) -> np.ndarray | None:
        """
        Extract features from most recent candles (newest first).
        Returns None if not enough candles.
        """
        if len(candles) < 21:
            return None

        closes = np.array([c.close for c in candles[:21]])
        highs  = np.array([c.high for c in candles[:21]])
        lows   = np.array([c.low for c in candles[:21]])
        vols   = np.array([c.volume for c in candles[:21]])

        returns = np.diff(closes) / closes[1:]  # 20 returns

        features = [
            # Returns at different horizons
            float(returns[0]),
            float(np.mean(returns[:3])),
            float(np.mean(returns[:5])),
            float(np.mean(returns[:10])),
            float(np.mean(returns[:20])),

            # Volatility
            float(np.std(returns[:5])),
            float(np.std(returns[:10])),
            float(np.std(returns[:20])),

            # Volume ratios
            float(vols[0] / np.mean(vols[:5]) if np.mean(vols[:5]) > 0 else 1.0),
            float(vols[0] / np.mean(vols[:10]) if np.mean(vols[:10]) > 0 else 1.0),

            # Price position (Williams %R style)
            float((closes[0] - lows[0]) / (highs[0] - lows[0]) if highs[0] > lows[0] else 0.5),
            float((closes[0] - np.min(lows[:5])) / (np.max(highs[:5]) - np.min(lows[:5]))
                  if np.max(highs[:5]) > np.min(lows[:5]) else 0.5),

            # Simple momentum (normalized)
            float(np.mean(returns[:5]) / (np.std(returns[:5]) + 1e-8)),
            float(np.mean(returns[:10]) / (np.std(returns[:10]) + 1e-8)),
            float(np.mean(returns[:14]) / (np.std(returns[:14]) + 1e-8)),

            # ATR ratio
            float(np.mean(highs[:5] - lows[:5]) / closes[0] if closes[0] > 0 else 0),
            float(np.mean(highs[:10] - lows[:10]) / closes[0] if closes[0] > 0 else 0),

            # Skewness of returns
            float(_skewness(returns[:10])),
        ]

        return np.array(features, dtype=np.float32)


def _skewness(x: np.ndarray) -> float:
    """Simple skewness calculation."""
    if len(x) < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


class MLInferenceEngine:
    """
    Lightweight ML inference for the trading bot.
    Loads pre-trained models from data/models/.
    Falls back to rule-based methods if models are missing.
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self._models_dir = models_dir
        self._extractor = FeatureExtractor()
        self._calibrator = PlattCalibrator(
            coef_path=models_dir / "calibration_coef.json"
        )
        self._regime_weights: np.ndarray | None = None
        self._load_models()

    def _load_models(self) -> None:
        """Load model weights from files."""
        weights_path = self._models_dir / "regime_weights.npy"
        if weights_path.exists():
            try:
                self._regime_weights = np.load(str(weights_path))
                logger.info("ML: regime weights loaded from %s", weights_path)
            except Exception as exc:
                logger.warning("ML: could not load regime weights: %s", exc)
        else:
            logger.info(
                "ML: no regime weights found at %s — "
                "using rule-based fallback", weights_path
            )

    def calibrate_score(self, raw_score: float) -> float:
        """Apply Platt scaling calibration to a raw signal score."""
        return self._calibrator.calibrate(raw_score)

    def extract_features(self, candles: list[Candle]) -> np.ndarray | None:
        """Extract feature vector from candles."""
        return self._extractor.extract(candles)

    def predict_regime_probability(
        self,
        candles: list[Candle],
        regime: str,
    ) -> float:
        """
        Return probability (0-1) that the current regime is 'regime'.
        Falls back to 0.5 if model not loaded or insufficient data.
        """
        features = self._extractor.extract(candles)
        if features is None:
            return 0.5

        if self._regime_weights is None:
            # Fallback: rule-based estimate from momentum
            momentum = features[0]   # 1-period return
            if regime == "TREND_EXPANSION":
                return float(np.clip(0.5 + momentum * 10, 0, 1))
            return 0.5

        # Simple linear model: sigmoid(features @ weights)
        try:
            score = float(np.dot(features, self._regime_weights[:len(features)]))
            return float(1.0 / (1.0 + math.exp(-score)))
        except Exception:
            return 0.5

    @property
    def calibrator_status(self) -> dict:
        return {
            "using_defaults": self._calibrator.is_using_defaults,
            "A": self._calibrator._A,
            "B": self._calibrator._B,
        }
