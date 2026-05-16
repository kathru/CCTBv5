#!/usr/bin/env python3
"""
Feature Analysis — relatório completo de governance de features.

Executa:
  1. Valida o schema (sem violações de leakage ou peso)
  2. Carrega candles históricos (do cache do recalibrate.py)
  3. Computa features para todo o período
  4. Verifica leakage temporalmente
  5. Calcula baseline de distribuição (para drift monitor)
  6. Calcula importância de features (permutação + correlação)
  7. Gera relatório e salva baseline em data/models/feature_baseline.json

Uso:
  python scripts/feature_analysis.py
  python scripts/feature_analysis.py --symbol BTC-USDT --start 2025-01-01
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_analysis")

from src.monitoring.feature_governance import (  # noqa: E402
    CURRENT_SCHEMA,
    FeatureImportance,
    FeatureStats,
    LeakageGuard,
    governance,
)

CACHE_DIR       = ROOT / "data" / "cache"
BASELINE_OUT    = ROOT / "data" / "models" / "feature_baseline.json"
IMPORTANCE_OUT  = ROOT / "data" / "models" / "feature_importance.json"


# ── Scoring v2 (espelho de v4_strategy._score_signal) ─────────────────────────

def compute_features(candles_window: list[dict]) -> dict[str, float] | None:
    """
    Computa todas as features do modelo v2 para um ponto.
    Idêntico a V4MomentumStrategy._score_signal().
    """
    if len(candles_window) < 21:
        return None

    closes  = [c["close"]  for c in candles_window[:21]]
    opens   = [c["open"]   for c in candles_window[:10]]
    highs   = [c["high"]   for c in candles_window[:10]]
    lows    = [c["low"]    for c in candles_window[:10]]
    volumes = [c["volume"] for c in candles_window[:20]]

    # M1 Adaptive Momentum
    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r5   = (closes[0]-closes[5])/closes[5]   if len(closes)>5  and closes[5]>0  else 0
    r10  = (closes[0]-closes[10])/closes[10] if len(closes)>10 and closes[10]>0 else 0
    r20  = (closes[0]-closes[20])/closes[20] if len(closes)>20 and closes[20]>0 else 0
    mw   = r5*0.5 + r10*0.3 + r20*0.2
    m1   = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    # M2 Trend Consistency
    n    = min(5, len(closes)-1)
    bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    m2   = bull*0.5 + (hh+hl)/2*0.5

    # M3 Volume Confirmation
    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr    = min(volumes[0]/avg5, 3.0)/3.0 if avg5>0 else 0.5
    vt    = min(max(sum(volumes[:3])/sum(volumes[3:6]), 0.3), 2.0) if len(volumes)>=6 and sum(volumes[3:6])>0 else 1.0
    cc    = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    m3    = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4 Regime Strength
    sma5  = sum(closes[:5])/5
    sma20 = sum(closes[:20])/20
    dist  = (sma5-sma20)/sma20 if sma20>0 else 0
    m4r   = min(max((dist+0.02)/0.04, 0.0), 1.0)
    m4f   = 0.85 if sma5>sma20 else 0.45
    m4    = m4r*0.6 + m4f*0.4

    # M5 Candle Structure
    cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
          for i in range(min(3, len(closes)))]
    m5 = sum(cs)/len(cs) if cs else 0.5

    return {
        "m1_momentum":    round(m1, 4),
        "m2_consistency": round(m2, 4),
        "m3_volume":      round(m3, 4),
        "m4_regime_str":  round(m4, 4),
        "m5_candle":      round(m5, 4),
    }


def load_cache(symbol: str) -> list[dict]:
    path = CACHE_DIR / f"{symbol.replace('-','_')}_1H.json"
    if not path.exists():
        log.error("Cache não encontrado: %s. Rode recalibrate.py primeiro.", path)
        return []
    candles = json.loads(path.read_text())
    log.info("Cache carregado: %d candles para %s", len(candles), symbol)
    return candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Analysis CCTBv5")
    parser.add_argument("--symbol",  default="BTC-USDT")
    parser.add_argument("--start",   default=None, help="YYYY-MM-DD filtro de início")
    parser.add_argument("--forward", type=int, default=5)
    parser.add_argument("--fee",     type=float, default=0.005)
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  CCTBv5 Feature Governance Analysis")
    log.info("  Schema: %s  |  Símbolo: %s", CURRENT_SCHEMA.version, args.symbol)
    log.info("=" * 65)

    # ── 1. Validar schema ────────────────────────────────────
    log.info("\n[1/5] Validando schema de features...")
    violations = LeakageGuard.validate_schema(CURRENT_SCHEMA)
    if violations:
        log.error("  VIOLAÇÕES:")
        for v in violations:
            log.error("    - %s", v)
    else:
        log.info("  Schema v%s: VÁLIDO — sem violações de leakage ou peso",
                 CURRENT_SCHEMA.version)
        for name, fdef in CURRENT_SCHEMA.features.items():
            log.info("    %-20s  peso=%.0f%%  lookback=%d  leakage_safe=%s",
                     name, fdef.weight*100, fdef.lookback,
                     "SIM" if fdef.leakage_safe else "NAO")

    # ── 2. Carregar dados ────────────────────────────────────
    log.info("\n[2/5] Carregando candles...")
    candles = load_cache(args.symbol)
    if not candles:
        return

    if args.start:
        start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                       .replace(tzinfo=UTC).timestamp() * 1000)
        candles = [c for c in candles if c["ts"] >= start_ms]
        log.info("  Filtrado para %s→: %d candles", args.start, len(candles))

    # ── 3. Computar features e verificar leakage temporal ────
    log.info("\n[3/5] Computando features e verificando leakage temporal...")
    features_list, labels, errors = [], [], []

    for i in range(21, len(candles) - args.forward):
        window = list(reversed(candles[max(0, i-21):i+1]))
        feats  = compute_features(window)
        if feats is None:
            continue

        # Leakage check por feature
        for fname, fval in feats.items():
            ok, reason = LeakageGuard.validate_feature_window(
                fname, fval, i, len(candles), args.forward
            )
            if not ok:
                errors.append(f"i={i} {fname}: {reason}")

        # Label: forward return >= fee
        entry  = candles[i]["close"]
        exit_  = candles[i + args.forward]["close"]
        label  = 1 if (exit_ - entry) / entry > args.fee else 0

        features_list.append(feats)
        labels.append(label)

    win_rate = sum(labels) / len(labels) if labels else 0
    log.info("  %d amostras  |  Win Rate: %.1f%%  |  Erros de leakage: %d",
             len(features_list), win_rate * 100, len(errors))
    if errors:
        log.warning("  Primeiros 3 erros: %s", errors[:3])

    if len(features_list) < 50:
        log.error("  Amostras insuficientes. Abortando.")
        return

    # ── 4. Baseline de distribuição ──────────────────────────
    log.info("\n[4/5] Calculando baseline de distribuição...")
    feature_names = list(features_list[0].keys())
    baselines: dict[str, FeatureStats] = {}

    for fname in feature_names:
        values = [f[fname] for f in features_list]
        stats  = FeatureStats.from_values(values)
        baselines[fname] = stats
        log.info("  %-20s  mean=%.3f  std=%.3f  p50=%.3f  [p10=%.3f p90=%.3f]",
                 fname, stats.mean, stats.std, stats.p50, stats.p10, stats.p90)

    # Salva baseline
    BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    baseline_data = {
        "schema_version": CURRENT_SCHEMA.version,
        "symbol":         args.symbol,
        "n_samples":      len(features_list),
        "win_rate":       round(win_rate, 4),
        "computed_at":    datetime.now(UTC).isoformat(),
        "features":       {k: v.to_dict() for k, v in baselines.items()},
    }
    BASELINE_OUT.write_text(json.dumps(baseline_data, indent=2))
    log.info("  Baseline salvo em: %s", BASELINE_OUT)

    # Define baseline no drift monitor
    governance.drift.set_baseline(baselines)

    # ── 5. Feature importance ────────────────────────────────
    log.info("\n[5/5] Calculando importância por permutação...")
    importance = FeatureImportance.from_samples(
        features_list, labels, n_permutations=50
    )
    log.info("\n" + FeatureImportance.format_report(importance))

    # Verifica se os pesos do schema refletem a importância real
    log.info("\n  ALINHAMENTO Peso×Importância:")
    for fname, metrics in importance.items():
        schema_w = metrics["schema_weight"]
        rel_imp  = metrics["relative_importance"]
        delta    = rel_imp - schema_w
        status   = "OK" if abs(delta) < 0.10 else "DESALINHADO"
        log.info("  %-20s  schema=%.0f%%  medido=%.0f%%  delta=%+.0f%%  [%s]",
                 fname, schema_w*100, rel_imp*100, delta*100, status)

    # Salva feature importance
    importance_data = {
        "schema_version": CURRENT_SCHEMA.version,
        "symbol":         args.symbol,
        "n_samples":      len(features_list),
        "win_rate":       round(win_rate, 4),
        "computed_at":    datetime.now(UTC).isoformat(),
        "features": {
            fname: {
                "spearman_corr":     round(m["spearman_corr"], 4),
                "perm_importance":   round(m["perm_importance"], 4),
                "relative_importance": round(m["relative_importance"], 4),
                "schema_weight":     m["schema_weight"],
                "aligned":           abs(m["relative_importance"] - m["schema_weight"]) < 0.10,
            }
            for fname, m in importance.items()
        },
    }
    IMPORTANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_OUT.write_text(json.dumps(importance_data, indent=2))
    log.info("  Feature importance salvo em: %s", IMPORTANCE_OUT)

    log.info("\n" + "=" * 65)
    log.info("  Análise completa.")
    log.info("  Próximos passos:")
    log.info("    1. Baseline gravado → drift monitor está configurado")
    log.info("    2. Execute periodicamente para monitorar drift ao vivo")
    log.info("    3. Se RelImp >> Weight → considerar ajustar pesos")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
