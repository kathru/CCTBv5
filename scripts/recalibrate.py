#!/usr/bin/env python3
"""
Recalibração do sistema de sinais V4 com dados recentes da OKX.

O que faz:
  1. Baixa candles históricos da OKX (paginado, sem limite prático)
  2. Desliza uma janela sobre o histórico e computa score para cada ponto
  3. Rotula: 1 se os próximos N candles fecham acima do custo (fee+slippage), 0 se não
  4. Ajusta novos coeficientes Platt (A, B) via gradiente descendente
  5. Analisa performance por regime e sugere novos thresholds
  6. Grava data/models/calibration_coef.json pronto para deploy

Uso:
  python scripts/recalibrate.py
  python scripts/recalibrate.py --start 2025-01-01 --forward 6 --min-score 0.45

Requisitos:
  pip install numpy requests python-dotenv
"""

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recalibrate")

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "models" / "calibration_coef.json"
CACHE_DIR = ROOT / "data" / "cache"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS     = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
DEFAULT_START       = "2025-01-01"      # início do período de recalibração
DEFAULT_GRANULARITY = "1H"
DEFAULT_FORWARD     = 5                 # candles à frente para medir o resultado
DEFAULT_FEE         = 0.005            # 0.5% round-trip (maker+taker+slippage)
DEFAULT_MIN_SCORE   = 0.40             # score mínimo para incluir na amostra

# ── OKX Fetcher ───────────────────────────────────────────────────────────────

OKX_BASE = "https://www.okx.com"
GRAN_MS   = {"1H": 3_600_000, "4H": 14_400_000, "6H": 21_600_000, "1D": 86_400_000}


def fetch_candles_range(
    symbol: str,
    granularity: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """
    Baixa todos os candles confirmados entre start_dt e end_dt da OKX.
    Usa paginação via parâmetro 'after' (busca do mais recente para o mais antigo).
    Endpoint: /api/v5/market/history-candles  (histórico completo, sem auth)
    """
    ms_per_candle = GRAN_MS.get(granularity, 3_600_000)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    after_ms = end_ms   # começa do fim, vai para trás

    candles = []
    page = 0

    log.info("Baixando %s %s de %s até %s…",
             symbol, granularity,
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    while True:
        url = (
            f"{OKX_BASE}/api/v5/market/history-candles"
            f"?instId={symbol}&bar={granularity}&limit=100&after={after_ms}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Fetch error page=%d: %s — aguardando 5s", page, exc)
            time.sleep(5)
            continue

        rows = data.get("data", [])
        if not rows:
            break

        for row in rows:
            ts_ms    = int(row[0])
            confirmed = row[8] == "1"
            if ts_ms < start_ms:
                candles.sort(key=lambda c: c["ts"])
                log.info("  %s: %d candles baixados", symbol, len(candles))
                return candles
            if confirmed and start_ms <= ts_ms <= end_ms:
                candles.append({
                    "ts":     ts_ms,
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })

        # Próxima página: after = ts do candle mais antigo desta página - 1ms
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        after_ms = oldest_ts - 1
        page += 1
        time.sleep(0.25)   # rate limit gentil

    candles.sort(key=lambda c: c["ts"])
    log.info("  %s: %d candles baixados", symbol, len(candles))
    return candles


def save_cache(symbol: str, granularity: str, candles: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol.replace('-','_')}_{granularity}.json"
    path.write_text(json.dumps(candles))
    log.info("Cache salvo: %s", path)


def load_cache(symbol: str, granularity: str) -> list[dict] | None:
    path = CACHE_DIR / f"{symbol.replace('-','_')}_{granularity}.json"
    if path.exists():
        data = json.loads(path.read_text())
        log.info("Cache carregado: %s (%d candles)", path, len(data))
        return data
    return None


# ── V4 Signal Logic (espelho do v4_strategy.py) ───────────────────────────────

REGIME_THRESHOLDS_ORIGINAL: dict[str, float] = {
    "TREND_EXPANSION":        0.56,
    "VOLATILITY_COMPRESSION": 0.60,
    "TREND_EXHAUSTION":       0.68,
    "MEAN_REVERTING_CHOP":    0.72,
    "HIGH_CORRELATION_RISK":  0.75,
    "PANIC_LIQUIDATION":      0.99,
    "LIQUIDITY_VACUUM":       0.99,
}

PLATT_A_V4 = 0.419
PLATT_B_V4 = -1.147


def detect_regime(closes: list[float], volumes: list[float]) -> str:
    if len(closes) < 20:
        return "MEAN_REVERTING_CHOP"

    sma_fast = float(np.mean(closes[:5]))
    sma_slow = float(np.mean(closes[:20]))
    avg_vol  = float(np.mean(volumes[:20]))
    last_vol = volumes[0]

    drop = (closes[0] - closes[1]) / closes[1] if closes[1] > 0 else 0
    if drop < -0.05:
        return "PANIC_LIQUIDATION"

    if sma_fast > sma_slow:
        if last_vol > avg_vol * 1.2:
            return "TREND_EXPANSION"
        if last_vol < avg_vol * 0.8:
            return "TREND_EXHAUSTION"
        return "VOLATILITY_COMPRESSION"

    # Abaixo da SMA lenta
    atr_5 = float(np.mean([
        abs(closes[i] - closes[i + 1]) for i in range(min(5, len(closes) - 1))
    ]))
    rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0

    if rel_atr < 0.005:
        return "LIQUIDITY_VACUUM"
    if rel_atr > 0.025:
        return "HIGH_CORRELATION_RISK"

    return "MEAN_REVERTING_CHOP"


def score_signal(closes: list[float], highs: list[float],
                 volumes: list[float], regime: str) -> tuple[float, dict]:
    """Espelho exato de V4MomentumStrategy._score_signal."""
    # M1 — Momentum (30%)
    momentum = (closes[0] - closes[-1]) / closes[-1] if closes[-1] > 0 else 0
    m1 = min(max((momentum + 0.05) / 0.10, 0.0), 1.0)

    # M2 — Estrutura (30%)
    m2 = 1.0 if (len(highs) >= 3 and highs[0] > highs[1] > highs[2]) else 0.4

    # M3 — Volume (20%)
    avg_v = float(np.mean(volumes[:5])) if volumes else 1.0
    m3 = min(volumes[0] / avg_v, 2.0) / 2.0 if avg_v > 0 else 0.5

    # M4 — Alinhamento de regime (20%)
    m4 = 0.8 if regime == "TREND_EXPANSION" else 0.5

    score = m1 * 0.3 + m2 * 0.3 + m3 * 0.2 + m4 * 0.2
    return score, {"m1_momentum": m1, "m2_structure": m2, "m3_volume": m3, "m4_regime": m4}


def platt_calibrate(score: float, A: float, B: float) -> float:
    return 1.0 / (1.0 + math.exp(-(A * score + B)))


# ── Labeling ──────────────────────────────────────────────────────────────────

def label_outcome(candles: list[dict], entry_idx: int,
                  forward: int, fee: float) -> int:
    """
    1 se o close do candle entry_idx+forward é > entry_price * (1 + fee), senão 0.
    Simula: comprar no close atual, vender N candles depois, lucro após fee.
    """
    if entry_idx + forward >= len(candles):
        return -1   # sem dados suficientes para rotular
    entry_price  = candles[entry_idx]["close"]
    exit_price   = candles[entry_idx + forward]["close"]
    net_return   = (exit_price - entry_price) / entry_price - fee
    return 1 if net_return > 0 else 0


# ── Platt Fitting ─────────────────────────────────────────────────────────────

def fit_platt(scores: np.ndarray, labels: np.ndarray,
              lr: float = 0.1, epochs: int = 2000) -> tuple[float, float]:
    """
    Ajusta A e B por gradiente descendente (cross-entropy loss).
    Equivalente a regressão logística com 1 feature.
    """
    A = PLATT_A_V4
    B = PLATT_B_V4

    for epoch in range(epochs):
        p     = 1.0 / (1.0 + np.exp(-(A * scores + B)))
        p     = np.clip(p, 1e-7, 1 - 1e-7)
        error = p - labels
        grad_A = float(np.mean(error * scores))
        grad_B = float(np.mean(error))
        A -= lr * grad_A
        B -= lr * grad_B
        if (epoch + 1) % 500 == 0:
            loss = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
            log.debug("  epoch=%d loss=%.4f A=%.4f B=%.4f", epoch + 1, loss, A, B)

    return float(A), float(B)


# ── Threshold Analysis ────────────────────────────────────────────────────────

def analyze_regime_thresholds(
    samples: list[dict],
    A: float,
    B: float,
) -> dict[str, float]:
    """
    Para cada regime, encontra o threshold de score calibrado que maximiza
    a precisão (P(win | score >= threshold)) com mínimo de 60% de precisão.
    """
    by_regime: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for s in samples:
        cal = platt_calibrate(s["score"], A, B)
        by_regime[s["regime"]].append((cal, s["label"]))

    new_thresholds: dict[str, float] = {}

    log.info("\n── Análise por regime ──────────────────────────────────────")
    for regime, pairs in sorted(by_regime.items()):
        if len(pairs) < 10:
            log.info("  %-28s  amostras insuficientes (%d)", regime, len(pairs))
            new_thresholds[regime] = REGIME_THRESHOLDS_ORIGINAL.get(regime, 0.65)
            continue

        pairs.sort(key=lambda x: x[0])
        cals   = np.array([p[0] for p in pairs])
        labels = np.array([p[1] for p in pairs])

        # Testa thresholds de 0.45 a 0.85 em passos de 0.01
        best_thresh = REGIME_THRESHOLDS_ORIGINAL.get(regime, 0.65)
        best_f1     = 0.0

        for thresh in np.arange(0.45, 0.86, 0.01):
            mask = cals >= thresh
            if mask.sum() < 5:
                continue
            precision = labels[mask].mean()
            recall    = mask[labels == 1].mean() if (labels == 1).sum() > 0 else 0
            if precision + recall == 0:
                continue
            f1 = 2 * precision * recall / (precision + recall)
            if precision >= 0.55 and f1 > best_f1:
                best_f1    = f1
                best_thresh = float(thresh)

        old_thresh = REGIME_THRESHOLDS_ORIGINAL.get(regime, 0.65)
        precision_all = labels.mean()
        log.info(
            "  %-28s  n=%4d  win_rate=%.1f%%  threshold: %.2f→%.2f  (Δ%+.2f)",
            regime, len(pairs),
            precision_all * 100,
            old_thresh, best_thresh,
            best_thresh - old_thresh,
        )
        new_thresholds[regime] = round(best_thresh, 2)

    return new_thresholds


# ── Main ──────────────────────────────────────────────────────────────────────

def build_samples(
    candles: list[dict],
    symbol: str,
    forward: int,
    fee: float,
    min_score: float,
) -> list[dict]:
    samples = []
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    volumes = [c["volume"] for c in candles]
    n = len(candles)

    for i in range(20, n - forward):
        # Janela mais recente primeiro (como o engine faz)
        w_close  = list(reversed(closes[max(0, i - 20):i + 1]))
        w_high   = list(reversed(highs[max(0, i - 20):i + 1]))
        w_vol    = list(reversed(volumes[max(0, i - 20):i + 1]))

        regime = detect_regime(w_close, w_vol)
        if regime in {"PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"}:
            continue

        score, _ = score_signal(w_close, w_high, w_vol, regime)
        if score < min_score:
            continue

        label = label_outcome(candles, i, forward, fee)
        if label == -1:
            continue

        samples.append({
            "symbol": symbol,
            "ts":     candles[i]["ts"],
            "score":  score,
            "regime": regime,
            "label":  label,
        })

    return samples


def print_summary(samples: list[dict], A_new: float, B_new: float) -> None:
    n = len(samples)
    wins = sum(s["label"] for s in samples)
    log.info("\n══ RESUMO ══════════════════════════════════════════════════")
    log.info("  Amostras totais : %d", n)
    log.info("  Win rate real   : %.1f%%", 100 * wins / n if n else 0)
    log.info("  A (antigo→novo) : %.4f → %.4f", PLATT_A_V4, A_new)
    log.info("  B (antigo→novo) : %.4f → %.4f", PLATT_B_V4, B_new)

    # Calibração em scores típicos
    log.info("\n  Calibração por score (novos coef):")
    for s in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        p_old = platt_calibrate(s, PLATT_A_V4, PLATT_B_V4)
        p_new = platt_calibrate(s, A_new, B_new)
        log.info("    score=%.2f  prob_old=%.3f  prob_new=%.3f  Δ=%+.3f",
                 s, p_old, p_new, p_new - p_old)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra o sistema de sinais V4")
    parser.add_argument("--symbols",   nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start",     default=DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end",       default=None, help="YYYY-MM-DD (default: hoje)")
    parser.add_argument("--gran",      default=DEFAULT_GRANULARITY, choices=["1H","4H","6H","1D"])
    parser.add_argument("--forward",   type=int,   default=DEFAULT_FORWARD)
    parser.add_argument("--fee",       type=float, default=DEFAULT_FEE)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--no-cache",  action="store_true", help="Ignora cache local")
    parser.add_argument("--dry-run",   action="store_true", help="Não grava o arquivo")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt   = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
        if args.end else datetime.now(UTC)
    )

    log.info("═" * 60)
    log.info("  CCTBv5 — Recalibração de Sinais")
    log.info("  Período  : %s → %s", start_dt.date(), end_dt.date())
    log.info("  Símbolos : %s", ", ".join(args.symbols))
    log.info("  Janela   : %d candles %s à frente", args.forward, args.gran)
    log.info("  Fee r/t  : %.2f%%", args.fee * 100)
    log.info("═" * 60)

    # ── 1. Baixar / carregar candles ──────────────────────────
    all_samples: list[dict] = []

    for symbol in args.symbols:
        candles = None
        if not args.no_cache:
            candles = load_cache(symbol, args.gran)

        if candles is None:
            candles = fetch_candles_range(symbol, args.gran, start_dt, end_dt)
            save_cache(symbol, args.gran, candles)

        if len(candles) < 50:
            log.warning("%s: candles insuficientes (%d) — ignorando", symbol, len(candles))
            continue

        # ── 2. Gerar amostras ─────────────────────────────────
        samples = build_samples(candles, symbol, args.forward, args.fee, args.min_score)
        log.info("%s: %d amostras geradas (win=%d, loss=%d)",
                 symbol, len(samples),
                 sum(s["label"] for s in samples),
                 sum(1 for s in samples if s["label"] == 0))
        all_samples.extend(samples)

    if not all_samples:
        log.error("Nenhuma amostra gerada. Verifique conexão ou parâmetros.")
        return

    # ── 3. Ajustar Platt scaling ──────────────────────────────
    log.info("\nAjustando Platt scaling com %d amostras…", len(all_samples))
    scores_arr = np.array([s["score"] for s in all_samples], dtype=np.float64)
    labels_arr = np.array([s["label"] for s in all_samples], dtype=np.float64)

    A_new, B_new = fit_platt(scores_arr, labels_arr)

    # ── 4. Analisar thresholds por regime ─────────────────────
    new_thresholds = analyze_regime_thresholds(all_samples, A_new, B_new)

    # ── 5. Imprimir resumo ────────────────────────────────────
    print_summary(all_samples, A_new, B_new)

    log.info("\n  Novos thresholds por regime:")
    for regime, thresh in sorted(new_thresholds.items()):
        log.info("    %-28s  %.2f", regime, thresh)

    # ── 6. Gravar resultado ───────────────────────────────────
    result = {
        "platt_a":          round(A_new, 6),
        "platt_b":          round(B_new, 6),
        "n":                len(all_samples),
        "win_rate":         round(float(labels_arr.mean()), 4),
        "calibrated_at":    datetime.now(UTC).isoformat(),
        "period_start":     args.start,
        "period_end":       end_dt.strftime("%Y-%m-%d"),
        "symbols":          args.symbols,
        "granularity":      args.gran,
        "forward_candles":  args.forward,
        "regime_thresholds": new_thresholds,
        "previous": {
            "platt_a": PLATT_A_V4,
            "platt_b": PLATT_B_V4,
        },
    }

    if args.dry_run:
        log.info("\n[DRY RUN] Resultado (não gravado):")
        log.info(json.dumps(result, indent=2))
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2))
        log.info("\n✅ Gravado em: %s", OUTPUT)
        log.info("\nPróximo passo: deploy")
        log.info("  git add data/models/calibration_coef.json")
        log.info("  git commit -m 'chore: recalibração %s'", datetime.now(UTC).strftime("%Y-%m"))
        log.info("  git push")
        log.info("  # No Oracle: git pull && sudo docker compose restart cctb")


if __name__ == "__main__":
    main()
