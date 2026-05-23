#!/usr/bin/env python3
"""
Ensemble Robustness Test — Phase 15.4

Valida que a complexidade do modelo M1-M8 realmente agrega valor
e identifica quais fatores são essenciais vs dispensáveis.

Metodologia:
  1. Testa todas as combinações C(8, K) de K fatores dos 8
     - K=3: C(8,3)=56 combinações (rápido, visão ampla)
     - K=5: C(8,5)=56 combinações (padrão)
     - K=7: C(8,7)=8 combinações (ablation — remove 1 fator por vez)
  2. Para cada combinação: calcula score e Sharpe OOS (WFO simplificado)
  3. Pergunta: quantas combinações têm Sharpe > 0?
     - Todas > 0 → modelo robusto, complexidade justificada
     - < 50% > 0 → dependência de poucos fatores → simplificar
  4. Feature ablation: qual fator quando REMOVIDO piora mais o Sharpe?
     → mais importante quando ausente

Saída:
  - data/models/ensemble_robustness.json (resultados completos)
  - Ranking de features por importância de ablation

Uso:
  python scripts/ensemble_robustness.py
  python scripts/ensemble_robustness.py --k 7 --symbol BTC-USDT
  python scripts/ensemble_robustness.py --k 3 --min-sharpe 0.0
"""

import argparse
import itertools
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
log = logging.getLogger("ensemble_robustness")

CACHE_DIR  = ROOT / "data" / "cache"
OUTPUT     = ROOT / "data" / "models" / "ensemble_robustness.json"

# ── Todos os 8 fatores do modelo atual ───────────────────────────────────────
ALL_FEATURES = [
    "m1_momentum",
    "m2_consistency",
    "m3_volume",
    "m4_regime_str",
    "m5_candle",
    "m6_futures",        # fixo em 0.5 no backtest histórico
    "m7_rel_strength",   # fixo em 0.5 no backtest histórico
    "m8_vol_state",
]

# Pesos originais por feature (usados para normalização no sub-ensemble)
ORIGINAL_WEIGHTS = {
    "m1_momentum":    0.10,
    "m2_consistency": 0.20,
    "m3_volume":      0.25,
    "m4_regime_str":  0.05,
    "m5_candle":      0.08,
    "m6_futures":     0.10,
    "m7_rel_strength":0.09,
    "m8_vol_state":   0.13,
}

LOOKBACK = 25


# ── Score com sub-conjunto de features (pesos renormalizados) ─────────────────

def _compute_score_subset(candles_window: list[dict],
                           features_subset: tuple[str, ...]) -> float | None:
    """
    Computa o score usando apenas as features do sub-conjunto.
    Pesos são renormalizados para somar 1.0.
    """
    if len(candles_window) < LOOKBACK:
        return None

    closes  = [c["close"]  for c in candles_window[:LOOKBACK]]
    opens   = [c["open"]   for c in candles_window[:10]]
    highs   = [c["high"]   for c in candles_window[:LOOKBACK]]
    lows    = [c["low"]    for c in candles_window[:LOOKBACK]]
    volumes = [c["volume"] for c in candles_window[:20]]

    # Computa todos os fatores
    all_vals: dict[str, float] = {}

    # M1
    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r1   = (closes[0]-closes[1])/closes[1]   if len(closes)>1  and closes[1]>0  else 0
    r5   = (closes[0]-closes[5])/closes[5]   if len(closes)>5  and closes[5]>0  else 0
    r10  = (closes[0]-closes[10])/closes[10] if len(closes)>10 and closes[10]>0 else 0
    r20  = (closes[0]-closes[20])/closes[20] if len(closes)>20 and closes[20]>0 else 0
    mw   = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
    all_vals["m1_momentum"] = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    # M2
    n    = min(6, len(closes)-1)
    bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    all_vals["m2_consistency"] = bull*0.5 + (hh+hl)/2*0.5

    # M3
    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr    = min(volumes[0]/avg5, 3.0)/3.0 if avg5>0 else 0.5
    vt    = (
        min(max(sum(volumes[:3])/sum(volumes[3:6]), 0.3), 2.0)
        if len(volumes) >= 6 and sum(volumes[3:6]) > 0 else 1.0
    )
    cc    = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    all_vals["m3_volume"] = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4
    sma5  = sum(closes[:5])/5
    sma20 = sum(closes[:20])/20 if len(closes)>=20 else sma5
    dist  = (sma5-sma20)/sma20 if sma20>0 else 0
    m4r   = min(max((dist+0.02)/0.04, 0.0), 1.0)
    m4f   = 0.85 if sma5>sma20 else 0.45
    all_vals["m4_regime_str"] = m4r*0.6 + m4f*0.4

    # M5
    cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
          for i in range(min(3, len(closes)))]
    all_vals["m5_candle"] = sum(cs)/len(cs) if cs else 0.5

    # M6/M7 — neutros no backtest
    all_vals["m6_futures"]      = 0.5
    all_vals["m7_rel_strength"] = 0.5

    # M8 — simplificado (proxy ATR)
    atr_pct = atr / closes[0] if closes[0] > 0 else 0.01
    all_vals["m8_vol_state"] = 0.35 if atr_pct < 0.008 else (0.65 if atr_pct < 0.015 else 0.5)

    # Score com sub-conjunto renormalizado
    total_w = sum(ORIGINAL_WEIGHTS[f] for f in features_subset)
    if total_w <= 0:
        return None

    score = sum(all_vals[f] * ORIGINAL_WEIGHTS[f] / total_w for f in features_subset)
    return min(max(score, 0.0), 1.0)


def _sharpe_from_returns(returns: list[float]) -> float | None:
    if len(returns) < 5:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = variance ** 0.5
    return (mean / std) * (252 ** 0.5) if std > 0 else None


def _evaluate_combination(
    candles: list[dict],
    features: tuple[str, ...],
    forward: int = 10,
    fee: float = 0.005,
    min_score: float = 0.45,
    is_split: float = 0.70,
) -> dict:
    """
    Avalia uma combinação de features via IS/OOS split simplificado.
    """
    n_is = int(len(candles) * is_split)
    is_data  = candles[:n_is]
    oos_data = candles[n_is:]

    def _run(data: list[dict]) -> dict:
        scores, labels = [], []
        for i in range(LOOKBACK, len(data) - forward):
            window = list(reversed(data[max(0, i-LOOKBACK):i+1]))
            sc = _compute_score_subset(window, features)
            if sc is None or sc < min_score:
                continue
            entry = data[i]["close"]
            exit_ = data[i + forward]["close"]
            net   = (exit_ - entry) / entry - fee
            labels.append(1 if net > 0 else 0)
            scores.append(net)
        if not scores:
            return {"n": 0, "wr": 0, "sharpe": None, "total_ret": 0}
        wr = sum(labels) / len(labels)
        sh = _sharpe_from_returns(scores)
        return {"n": len(scores), "wr": round(wr, 4),
                "sharpe": round(sh, 3) if sh else None,
                "total_ret": round(sum(scores), 4)}

    is_m  = _run(is_data)
    oos_m = _run(oos_data)
    return {"is": is_m, "oos": oos_m}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Robustness Test — Phase 15.4")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--k",      type=int, default=7,
                        help="Tamanho do sub-ensemble (5=combinações, 7=ablation, default 7)")
    parser.add_argument("--forward",type=int, default=10)
    parser.add_argument("--min-sharpe", type=float, default=0.0,
                        help="Sharpe mínimo OOS para considerar 'positivo'")
    args = parser.parse_args()

    # Carrega candles
    cache_file = CACHE_DIR / f"{args.symbol.replace('-','_')}_1H.json"
    if not cache_file.exists():
        log.error("Cache não encontrado: %s — rode recalibrate.py primeiro", cache_file)
        return

    candles = json.loads(cache_file.read_text(encoding="utf-8"))
    log.info("Candles: %d | Símbolo: %s | K=%d", len(candles), args.symbol, args.k)
    log.info("")

    if args.k > len(ALL_FEATURES):
        log.error("K=%d > %d features disponíveis", args.k, len(ALL_FEATURES))
        return

    combos = list(itertools.combinations(ALL_FEATURES, args.k))
    log.info("Testando C(%d,%d) = %d combinações...", len(ALL_FEATURES), args.k, len(combos))

    results: list[dict] = []
    positive_oos = 0

    for i, combo in enumerate(combos):
        res = _evaluate_combination(candles, combo, forward=args.forward)
        oos_sharpe = res["oos"].get("sharpe")
        is_positive = oos_sharpe is not None and oos_sharpe > args.min_sharpe
        if is_positive:
            positive_oos += 1

        results.append({
            "combo":      list(combo),
            "is":         res["is"],
            "oos":        res["oos"],
            "oos_positive": is_positive,
        })

        if (i + 1) % max(1, len(combos) // 10) == 0:
            log.info("  [%d/%d] %s → OOS Sharpe=%s %s",
                     i + 1, len(combos),
                     "+".join(f.split("_")[0].upper() for f in combo),
                     f"{oos_sharpe:.3f}" if oos_sharpe else "N/A",
                     "✓" if is_positive else "✗")

    # Robustez global
    pct_positive = positive_oos / len(combos) * 100 if combos else 0
    robustness = (
        "ROBUSTO"      if pct_positive >= 60 else
        "MODERADO"     if pct_positive >= 30 else
        "FRAGIL"
    )

    # Feature ablation (K=7: quem faz mais falta quando removido)
    ablation: dict[str, dict] = {}
    if args.k == len(ALL_FEATURES) - 1:
        for r in results:
            missing = [f for f in ALL_FEATURES if f not in r["combo"]]
            if missing:
                fname = missing[0]
                sh = r["oos"].get("sharpe")
                ablation[fname] = {
                    "oos_sharpe":     sh,
                    "oos_positive":   r["oos_positive"],
                    "missing_impact": sh,   # quanto piora sem esse fator
                }
        # Ordena: pior Sharpe quando ausente = mais importante
        ablation_sorted = dict(
            sorted(ablation.items(),
                   key=lambda x: (x[1]["oos_sharpe"] or -99))
        )
        log.info("\n── Feature Ablation (K=%d: remove 1 por vez) ──", args.k)
        log.info("  (menor Sharpe quando ausente = fator mais importante)")
        for fname, v in ablation_sorted.items():
            sh = v["oos_sharpe"]
            log.info("  SEM %-22s → OOS Sharpe=%s  %s",
                     fname, f"{sh:.3f}" if sh else "N/A",
                     "✗ prejudica" if sh and sh < 0 else "~ neutro")
    else:
        ablation_sorted = {}

    # Melhor sub-ensemble
    results_with_sharpe = [r for r in results if r["oos"].get("sharpe") is not None]
    best = (
        max(results_with_sharpe, key=lambda r: r["oos"]["sharpe"]) if results_with_sharpe else None
    )
    worst = (
        min(results_with_sharpe, key=lambda r: r["oos"]["sharpe"]) if results_with_sharpe else None
    )

    log.info("\n══ RESULTADO FINAL ══════════════════════════════════════════")
    log.info("  Símbolo       : %s", args.symbol)
    log.info("  K             : %d (de %d features)", args.k, len(ALL_FEATURES))
    log.info("  Combinações   : %d", len(combos))
    log.info("  OOS positivos : %d (%.0f%%)", positive_oos, pct_positive)
    log.info("  Robustez      : %s", robustness)
    if best:
        log.info("  Melhor combo  : %s → Sharpe=%.3f",
                 "+".join(f.split("_")[0].upper() for f in best["combo"]),
                 best["oos"]["sharpe"])
    if worst:
        log.info("  Pior combo    : %s → Sharpe=%.3f",
                 "+".join(f.split("_")[0].upper() for f in worst["combo"]),
                 worst["oos"]["sharpe"])

    if ablation_sorted:
        most_important = list(ablation_sorted.keys())[0]
        log.info("  Fator mais importante (ablation): %s", most_important)

    # Salva resultado
    output = {
        "symbol":         args.symbol,
        "k":              args.k,
        "n_combinations": len(combos),
        "positive_oos":   positive_oos,
        "pct_positive":   round(pct_positive, 1),
        "robustness":     robustness,
        "best_combo":     best["combo"] if best else None,
        "best_oos_sharpe":best["oos"]["sharpe"] if best else None,
        "ablation":       ablation_sorted,
        "all_results":    results,
        "computed_at":    datetime.now(UTC).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Salvo em: %s", OUTPUT)
    log.info("═" * 60)

    # Recomendação
    log.info("\n── Recomendação ──────────────────────────────────────────────")
    if pct_positive >= 60:
        log.info("  ✅ Modelo ROBUSTO — complexidade justificada.")
        log.info("     A maioria dos sub-ensembles tem edge positivo.")
    elif pct_positive >= 30:
        log.info("  ⚠ Modelo MODERADO — alguns sub-ensembles funcionam.")
        log.info("     Considere remover features com ablation negativo.")
    else:
        log.info("  ❌ Modelo FRÁGIL — poucos sub-ensembles têm edge.")
        log.info("     Rever scoring completo — possível overfitting.")
    log.info("─" * 60)


if __name__ == "__main__":
    main()
