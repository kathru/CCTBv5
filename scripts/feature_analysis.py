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

Features computáveis de candles históricos (v2.5.0 — 9 features):
  M1-M5: totalmente computáveis
  M8:    totalmente computável (ATR + Bollinger + dir_consistency)
  M6:    requer API de futuros OKX → default 0.5 neutro (documentado)
  M7:    requer multi-símbolo → default 0.5 neutro (documentado)
  M9:    requer FinNLP/Fear&Greed externo → default 0.5 neutro (documentado)
  M4:    peso=0% desde v2.5.0 — mantido para monitoramento/diagnóstico

Uso:
  python scripts/feature_analysis.py
  python scripts/feature_analysis.py --symbol BTC-USDT --start 2025-01-01
"""

import argparse
import json
import logging
import math
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

LOOKBACK = 25   # candles de janela (25 para M8 Bollinger 20 + buffer)


# ── M8: Volatility State (computável de candles) ─────────────────────────────

def _atr_hist(highs: list[float], lows: list[float], closes: list[float],
              period: int = 14) -> float:
    n = min(period, len(highs) - 1)
    if n <= 0:
        return (highs[0] - lows[0]) if highs else 0.0
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i+1]),
               abs(lows[i] - closes[i+1])) for i in range(n)]
    return sum(trs) / len(trs) if trs else 0.0


def _bollinger_hist(closes: list[float], period: int = 20) -> tuple[float, float, float]:
    n = min(period, len(closes))
    if n < 2:
        c = closes[0]
        return c, c, c
    window = closes[:n]
    mid = sum(window) / n
    std = math.sqrt(sum((x - mid) ** 2 for x in window) / n)
    return mid + 2 * std, mid, mid - 2 * std


def _compute_m8(closes: list[float], highs: list[float], lows: list[float]) -> float:
    """
    Computa M8 (Volatility State score) de candles históricos.
    Espelho de volatility_state.compute_vol_state() sem as classes externas.
    """
    if len(closes) < 22:
        return 0.5
    price    = closes[0]
    atr_now  = _atr_hist(highs, lows, closes, period=14)
    atr_prev = _atr_hist(highs[7:], lows[7:], closes[7:], period=14)
    bb_upper, _bb_mid, bb_lower = _bollinger_hist(closes, period=20)
    bb_width_pct = (bb_upper - bb_lower) / price if price > 0 else 0.0
    atr_pct      = atr_now / price if price > 0 else 0.0
    atr_change   = atr_now / atr_prev if atr_prev > 0 else 1.0
    n_dir = min(10, len(closes) - 1)
    ups   = sum(1 for i in range(n_dir) if closes[i] > closes[i+1])
    dns   = n_dir - ups
    dir_c = max(ups, dns) / n_dir if n_dir > 0 else 0.5

    STATE_M8 = {
        "EXPANDING":      0.80,
        "TREND":          0.70,
        "COMPRESSED":     0.65,
        "MEAN_REVERTING": 0.35,
        "CHAOTIC":        0.20,
    }
    if atr_pct > 0.025 and dir_c < 0.45:
        state = "CHAOTIC"
    elif atr_change > 1.15 and dir_c > 0.55:
        state = "EXPANDING"
    elif atr_pct < 0.008 or bb_width_pct < 0.015:
        state = "COMPRESSED"
    elif dir_c > 0.60 and 0.008 <= atr_pct <= 0.025:
        state = "TREND"
    else:
        state = "MEAN_REVERTING"
    return STATE_M8[state]


# ── Scoring v2.1 (espelho de momentum_strategy._score_signal) ────────────────

def compute_features(candles_window: list[dict]) -> dict[str, float] | None:
    """
    Computa todas as features computáveis de candles (M1-M5, M8).
    Idêntico a MomentumStrategy._score_signal() para os fatores históricos.

    M6 (Futures Flow): default 0.5 — requer API OKX em tempo real
    M7 (Rel. Strength): default 0.5 — requer múltiplos símbolos
    """
    if len(candles_window) < LOOKBACK:
        return None

    closes  = [c["close"]  for c in candles_window[:LOOKBACK]]
    opens   = [c["open"]   for c in candles_window[:10]]
    highs   = [c["high"]   for c in candles_window[:LOOKBACK]]
    lows    = [c["low"]    for c in candles_window[:LOOKBACK]]
    volumes = [c["volume"] for c in candles_window[:20]]

    # M1 Adaptive Momentum — blend r1+r5 (curto) + r10+r20 (médio)
    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r1   = (closes[0]-closes[1])/closes[1]   if len(closes)>1  and closes[1]>0  else 0
    r5   = (closes[0]-closes[5])/closes[5]   if len(closes)>5  and closes[5]>0  else 0
    r10  = (closes[0]-closes[10])/closes[10] if len(closes)>10 and closes[10]>0 else 0
    r20  = (closes[0]-closes[20])/closes[20] if len(closes)>20 and closes[20]>0 else 0
    mw   = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
    m1   = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    # M2 Trend Consistency
    n    = min(6, len(closes)-1)
    bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    m2   = bull*0.5 + (hh+hl)/2*0.5

    # M3 Volume Confirmation
    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr    = min(volumes[0]/avg5, 3.0)/3.0 if avg5>0 else 0.5
    vt    = (
        min(max(sum(volumes[:3])/sum(volumes[3:6]), 0.3), 2.0)
        if len(volumes) >= 6 and sum(volumes[3:6]) > 0 else 1.0
    )
    cc    = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    m3    = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4 Regime Strength — MONITORAMENTO APENAS (peso=0% desde v2.5.0)
    # Mantido no output para diagnóstico/auditoria, mas NÃO entra no score.
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

    # M6 Futures Flow — external API, neutro no backtest histórico
    m6 = 0.5

    # M7 Relative Strength — requer multi-símbolo, neutro no backtest histórico
    m7 = 0.5

    # M8 Volatility State — computável de candles
    m8 = _compute_m8(closes, highs[:LOOKBACK], lows[:LOOKBACK])

    # M9 News Sentiment — requer Fear&Greed externo → neutro 0.5
    m9 = 0.5

    return {
        "m1_momentum":    round(m1, 4),
        "m2_consistency": round(m2, 4),
        "m3_volume":      round(m3, 4),
        "m4_regime_str":  round(m4, 4),   # monitoring only — peso=0% desde v2.5.0
        "m5_candle":      round(m5, 4),
        "m6_futures":     round(m6, 4),   # neutro — API externa
        "m7_rel_strength":round(m7, 4),   # neutro — multi-símbolo
        "m8_vol_state":   round(m8, 4),
        "m9_sentiment":   round(m9, 4),   # neutro — Fear&Greed externo
    }


def load_cache(symbol: str) -> list[dict]:
    for gran in ("1H", "4H"):
        path = CACHE_DIR / f"{symbol.replace('-','_')}_{gran}.json"
        if path.exists():
            candles = json.loads(path.read_text())
            log.info("Cache carregado: %d candles (%s) para %s", len(candles), gran, symbol)
            return candles
    log.error("Cache não encontrado para %s. Rode recalibrate.py primeiro.", symbol)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Analysis CCTBv5")
    parser.add_argument("--symbol",  default="BTC-USDT")
    parser.add_argument("--start",   default=None, help="YYYY-MM-DD filtro de início")
    parser.add_argument("--forward", type=int, default=10)  # 10×1H = 10h
    parser.add_argument("--fee",     type=float, default=0.005)
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  CCTBv5 Feature Governance Analysis — Schema v%s", CURRENT_SCHEMA.version)
    log.info("  Símbolo: %s  |  M6/M7/M9: neutro (0.5)  |  M8: calculado  |  M4: peso=0%%", args.symbol)
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
            log.info("    %-22s  peso=%.0f%%  lookback=%d  leakage_safe=%s",
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

    for i in range(LOOKBACK, len(candles) - args.forward):
        window = list(reversed(candles[max(0, i-LOOKBACK):i+1]))
        feats  = compute_features(window)
        if feats is None:
            continue

        for fname, fval in feats.items():
            # M6/M7 são neutros por design — não verificar leakage deles
            if fname in ("m6_futures", "m7_rel_strength"):
                continue
            ok, reason = LeakageGuard.validate_feature_window(
                fname, fval, i, len(candles), args.forward
            )
            if not ok:
                errors.append(f"i={i} {fname}: {reason}")

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
        note = " [neutro fixo]" if fname in ("m6_futures", "m7_rel_strength") else ""
        log.info("  %-24s  mean=%.3f  std=%.3f  p50=%.3f  [p10=%.3f p90=%.3f]%s",
                 fname, stats.mean, stats.std, stats.p50, stats.p10, stats.p90, note)

    BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    baseline_data = {
        "schema_version": CURRENT_SCHEMA.version,
        "symbol":         args.symbol,
        "n_samples":      len(features_list),
        "win_rate":       round(win_rate, 4),
        "computed_at":    datetime.now(UTC).isoformat(),
        "notes": {
            "m6_futures":      "neutro (0.5) — requer API OKX de futuros em tempo real",
            "m7_rel_strength": "neutro (0.5) — requer dados multi-símbolo simultâneos",
            "m8_vol_state":    "calculado de candles históricos via ATR+Bollinger+dir_consistency",
        },
        "features": {k: v.to_dict() for k, v in baselines.items()},
    }
    BASELINE_OUT.write_text(json.dumps(baseline_data, indent=2))
    log.info("  Baseline salvo em: %s", BASELINE_OUT)
    governance.drift.set_baseline(baselines)

    # ── 5. Feature importance ────────────────────────────────
    log.info("\n[5/5] Calculando importância por permutação...")
    log.info("  AVISO: M6/M7 têm importância 0 por construção (fixos em 0.5 no backtest)")
    importance = FeatureImportance.from_samples(
        features_list, labels, n_permutations=50
    )
    log.info("\n" + FeatureImportance.format_report(importance))

    log.info("\n  ALINHAMENTO Peso×Importância:")
    for fname, metrics in importance.items():
        schema_w = metrics["schema_weight"]
        rel_imp  = metrics["relative_importance"]
        delta    = rel_imp - schema_w
        if fname in ("m6_futures", "m7_rel_strength"):
            log.info("  %-24s  schema=%.0f%%  [importância não mensurável — dado externo]",
                     fname, schema_w*100)
            continue
        status = "OK" if abs(delta) < 0.10 else "DESALINHADO"
        log.info("  %-24s  schema=%.0f%%  medido=%.0f%%  delta=%+.0f%%  [%s]",
                 fname, schema_w*100, rel_imp*100, delta*100, status)

    importance_data = {
        "schema_version": CURRENT_SCHEMA.version,
        "symbol":         args.symbol,
        "n_samples":      len(features_list),
        "win_rate":       round(win_rate, 4),
        "computed_at":    datetime.now(UTC).isoformat(),
        "features": {
            fname: {
                "spearman_corr":       round(m["spearman_corr"], 4),
                "perm_importance":     round(m["perm_importance"], 4),
                "relative_importance": round(m["relative_importance"], 4),
                "schema_weight":       m["schema_weight"],
                "aligned": (
                    None if fname in ("m6_futures", "m7_rel_strength", "m9_sentiment")
                    else True if fname == "m4_regime_str"  # peso=0%, alinhado por definição
                    else abs(m["relative_importance"] - m["schema_weight"]) < 0.10
                ),
                **({"note": "não mensurável — dado externo ao backtest"}
                   if fname in ("m6_futures", "m7_rel_strength", "m9_sentiment") else {}),
            }
            for fname, m in importance.items()
        },
    }
    IMPORTANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_OUT.write_text(json.dumps(importance_data, indent=2))
    log.info("  Feature importance salvo em: %s", IMPORTANCE_OUT)

    log.info("\n" + "=" * 65)
    log.info("  Análise completa. Schema v%s  |  %d features  |  %d amostras",
             CURRENT_SCHEMA.version, len(feature_names), len(features_list))
    log.info("  Próximos passos:")
    log.info("    1. docker cp data/models/feature_baseline.json cctb_app:/app/data/models/")
    log.info("    2. docker cp data/models/feature_importance.json cctb_app:/app/data/models/")
    log.info("    3. git add data/models/ && git commit -m 'chore: feature analysis %s'",
             datetime.now(UTC).strftime("%Y-%m-%d"))
    log.info("=" * 65)


if __name__ == "__main__":
    main()
