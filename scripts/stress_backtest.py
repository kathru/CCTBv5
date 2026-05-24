#!/usr/bin/env python3
"""
Stress Backtest — Phase 16: Regime Robustness

Testa o modelo completo (M1-M8 equivalente + detecção de regime + circuit breakers)
contra os 3 cenários adversos definidos no plano de validação:

  16.1 — Lateralidade Prolongada
         Identifica automaticamente períodos range < 5% por > 30 dias.
         Sistema deve reduzir atividade, não sangrar.

  16.2 — Bear Market Prolongado
         Detecta as piores quedas de 30 dias no histórico disponível.
         Pergunta: BEAR_TREND ativa antes ou depois do crash?
         Calcula lag de regime em horas e perdas no lag.

  16.3 — Volatility Regimes
         Identifica compressões prolongadas (M8=COMPRESSED > 5 dias)
         e explosões de vol (ATR dobra em < 4H).
         Verifica se M8 classifica corretamente e se sizing adapta.

Saída:
  data/models/stress_backtest.json  (resultados completos)
  Relatório no console com conclusões objetivas.

Uso:
  python scripts/stress_backtest.py
  python scripts/stress_backtest.py --symbol BTC-USDT
  python scripts/stress_backtest.py --start 2025-01-01 --end 2025-06-30
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
log = logging.getLogger("stress_backtest")

CACHE_DIR = ROOT / "data" / "cache"
OUTPUT    = ROOT / "data" / "models" / "stress_backtest.json"

LOOKBACK  = 25
FORWARD   = 10
FEE       = 0.005
MIN_SCORE = 0.45

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ts(candle: dict) -> datetime:
    return datetime.fromtimestamp(candle["ts"] / 1000, tz=UTC)


def _ret(closes: list[float], h: int) -> float:
    if len(closes) <= h or closes[h] <= 0:
        return 0.0
    return (closes[0] - closes[h]) / closes[h]


def _atr(highs, lows, closes, period=14) -> float:
    n = min(period, len(highs) - 1)
    if n <= 0:
        return (highs[0] - lows[0]) if highs else 0.0
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i+1]),
               abs(lows[i]-closes[i+1])) for i in range(n)]
    return sum(trs) / len(trs) if trs else 0.0


def _bollinger(closes, period=20):
    n = min(period, len(closes))
    if n < 2:
        c = closes[0]; return c, c, c
    mid = sum(closes[:n]) / n
    std = math.sqrt(sum((x-mid)**2 for x in closes[:n]) / n)
    return mid + 2*std, mid, mid - 2*std


# ── Score M1-M8 (mesma lógica do walk_forward.py) ────────────────────────────

def _score(window: list[dict]) -> float | None:
    if len(window) < LOOKBACK:
        return None
    closes  = [c["close"]  for c in window[:LOOKBACK]]
    opens   = [c["open"]   for c in window[:10]]
    highs   = [c["high"]   for c in window[:LOOKBACK]]
    lows    = [c["low"]    for c in window[:LOOKBACK]]
    volumes = [c["volume"] for c in window[:20]]

    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r1   = _ret(closes, 1); r5 = _ret(closes, 5)
    r10  = _ret(closes, 10); r20 = _ret(closes, 20)
    mw   = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
    m1   = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    n    = min(6, len(closes)-1)
    bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    m2   = bull*0.5 + (hh+hl)/2*0.5

    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr    = min(volumes[0]/avg5,3.0)/3.0 if avg5>0 else 0.5
    vt    = (
        min(max(sum(volumes[:3])/sum(volumes[3:6]), 0.3), 2.0)
        if len(volumes) >= 6 and sum(volumes[3:6]) > 0 else 1.0
    )
    cc    = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    m3    = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4 — REMOVIDO (peso 0% desde v2.5.0) — mantido computado para diagnóstico
    sma5  = sum(closes[:5])/5
    sma20 = sum(closes[:20])/20 if len(closes)>=20 else sma5

    # M5 — Candle Structure (6%)
    cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
          for i in range(min(3,len(closes)))]
    m5 = sum(cs)/len(cs) if cs else 0.5

    # M6/M7/M9 neutros (sem API) | M8 calculado de candles
    # Score v2.5.0: M1=2% M2=11% M3=33% M4=0% M5=6% M6=12% M7=11% M8=19% M9=6%
    atr_pct = atr/closes[0] if closes[0] > 0 else 0.01
    atr_ratio = 1.0  # sem histórico de ATR prev no stress_backtest
    dir_consistency = (
        sum(1 for i in range(min(5, len(closes)-1)) if closes[i] > closes[i+1]) / 5
    ) if len(closes) > 5 else 0.5
    bb_width = (max(closes[:20]) - min(closes[:20])) / closes[0] if len(closes) >= 20 else 0.02
    m8 = _compute_m8_state_from_candles(atr_pct, atr_ratio, dir_consistency, bb_width)

    return (m1*0.02 + m2*0.11 + m3*0.33 +
            m5*0.06 + 0.5*0.12 + 0.5*0.11 + m8*0.19 + 0.5*0.06)


# ── Detecção de regime BEAR_TREND ─────────────────────────────────────────────

def _detect_bear(window: list[dict]) -> bool:
    """Replica a lógica simplificada de BEAR_TREND do MomentumStrategy."""
    if len(window) < 20:
        return False
    closes = [c["close"] for c in window[:21]]
    vols   = [c["volume"] for c in window[:20]]
    sma5   = sum(closes[:5])/5
    sma20  = sum(closes[:20])/20
    drop_1h = (closes[0]-closes[1])/closes[1] if closes[1] > 0 else 0
    # BEAR: SMA5 < SMA20 por tendência + queda acelerada
    sma_bear = sma5 < sma20 * 0.99
    avg_vol  = sum(vols)/len(vols)
    high_vol = vols[0] > avg_vol * 1.5
    return sma_bear and (drop_1h < -0.02 or high_vol)


def _compute_m8_state_from_candles(
    atr_pct: float, atr_ratio: float, dir_consistency: float, bb_width: float
) -> float:
    """Converte estado M8 em score 0-1 (espelha backtest_engine._vol_state_from_candles)."""
    if atr_pct > 0.025 and dir_consistency < 0.45:
        state = "CHAOTIC"
    elif atr_ratio > 1.15 and dir_consistency > 0.55:
        state = "EXPANDING"
    elif atr_pct < 0.008 or bb_width < 0.015:
        state = "COMPRESSED"
    elif dir_consistency > 0.60 and 0.008 <= atr_pct <= 0.025:
        state = "TREND"
    else:
        state = "MEAN_REVERTING"
    # Espelho exato de volatility_state.STATE_M8_SCORE (fonte autoritativa do M8)
    return {"EXPANDING": 0.80, "TREND": 0.70, "COMPRESSED": 0.65,
            "MEAN_REVERTING": 0.35, "CHAOTIC": 0.20, "UNKNOWN": 0.50}.get(state, 0.50)


def _detect_m8_state(window: list[dict]) -> str:
    if len(window) < 22:
        return "UNKNOWN"
    closes = [c["close"] for c in window[:26]]
    highs  = [c["high"]  for c in window[:26]]
    lows   = [c["low"]   for c in window[:26]]
    price    = closes[0]
    atr_now  = _atr(highs, lows, closes, 14)
    atr_prev = _atr(highs[7:], lows[7:], closes[7:], 14)
    bb_u, _, bb_l = _bollinger(closes, 20)
    atr_pct    = atr_now/price if price > 0 else 0
    atr_change = atr_now/atr_prev if atr_prev > 0 else 1
    bb_w       = (bb_u-bb_l)/price if price > 0 else 0
    n_dir = min(10, len(closes)-1)
    ups   = sum(1 for i in range(n_dir) if closes[i]>closes[i+1])
    dir_c = max(ups, n_dir-ups)/n_dir if n_dir > 0 else 0.5
    if atr_pct > 0.025 and dir_c < 0.45:           return "CHAOTIC"
    elif atr_change > 1.15 and dir_c > 0.55:        return "EXPANDING"
    elif atr_pct < 0.008 or bb_w < 0.015:          return "COMPRESSED"
    elif dir_c > 0.60 and 0.008 <= atr_pct <= 0.025: return "TREND"
    else:                                             return "MEAN_REVERTING"


# ── Backtest em janela de tempo ───────────────────────────────────────────────

def _run_window(candles: list[dict], label: str) -> dict:
    """Roda o modelo num período específico e retorna métricas."""
    trades = []
    regime_switches: list[dict] = []
    vol_states: list[str] = []
    prev_bear = False

    for i in range(LOOKBACK, len(candles) - FORWARD):
        window = list(reversed(candles[max(0, i-LOOKBACK):i+1]))

        is_bear = _detect_bear(window)
        vol_state = _detect_m8_state(window)
        vol_states.append(vol_state)

        # Detecta transição de regime
        if is_bear != prev_bear:
            regime_switches.append({
                "ts":    _ts(candles[i]).isoformat(),
                "to":    "BEAR_TREND" if is_bear else "RECOVERY",
                "price": candles[i]["close"],
                "idx":   i,
            })
        prev_bear = is_bear

        # Não entra em BEAR_TREND (replica o bloqueio)
        if is_bear:
            continue

        sc = _score(window)
        if sc is None or sc < MIN_SCORE:
            continue

        entry = candles[i]["close"]
        exit_ = candles[i + FORWARD]["close"]
        net   = (exit_ - entry) / entry - FEE
        trades.append({
            "ts":    _ts(candles[i]).isoformat(),
            "score": round(sc, 4),
            "entry": entry,
            "exit":  exit_,
            "pnl":   round(net, 6),
            "win":   net > 0,
            "vol_state": vol_state,
        })

    # Métricas
    n = len(trades)
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    pnls   = [t["pnl"] for t in trades]

    wr   = round(len(wins)/n, 4)         if n > 0 else 0
    avg_w = sum(t["pnl"] for t in wins)/len(wins) if wins else 0
    avg_l = sum(t["pnl"] for t in losses)/len(losses) if losses else 0
    exp   = round(wr*avg_w + (1-wr)*avg_l, 6) if n > 0 else 0
    pf    = round(sum(t["pnl"] for t in wins)/abs(sum(t["pnl"] for t in losses)), 3) \
            if losses and sum(t["pnl"] for t in losses) != 0 else None
    total_ret = round(sum(pnls), 4)

    # Drawdown
    eq, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        eq += p; peak = max(peak, eq)
        dd = (peak-eq)/peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Vol state distribution
    from collections import Counter
    vol_dist = dict(Counter(vol_states))

    # Bear regime coverage
    bear_candles = sum(1 for i in range(LOOKBACK, len(candles)-FORWARD)
                       if _detect_bear(list(reversed(candles[max(0,i-LOOKBACK):i+1]))))
    bear_pct = round(bear_candles / max(1, len(candles)-LOOKBACK-FORWARD) * 100, 1)

    return {
        "label":           label,
        "n_candles":       len(candles),
        "n_trades":        n,
        "win_rate":        wr,
        "expectancy":      exp,
        "profit_factor":   pf,
        "total_return_pct":round(total_ret*100, 3),
        "max_drawdown_pct":round(max_dd*100, 3),
        "avg_win_pct":     round(avg_w*100, 4),
        "avg_loss_pct":    round(avg_l*100, 4),
        "bear_regime_pct": bear_pct,
        "regime_switches": regime_switches,
        "vol_state_dist":  vol_dist,
        "verdict": _verdict(n, wr, total_ret, max_dd, bear_pct),
    }


def _verdict(n: int, wr: float, total_ret: float, max_dd: float, bear_pct: float) -> str:
    if n < 5:
        return "INSUFICIENTE — poucos trades no período"
    if bear_pct > 60:
        return f"BEAR DOMINANTE — {bear_pct}% do período bloqueado ✓ (sistema correto)"
    if wr > 0.45 and total_ret > 0:
        return "ROBUSTO — edge mantido no período adverso"
    if wr > 0.35 and total_ret > -0.05:
        return "MODERADO — drawdown controlado"
    return f"FRÁGIL — WR={wr:.0%} ret={total_ret:.1%} dd={max_dd:.1%}"


# ── Identificadores de períodos de stress ────────────────────────────────────

def _find_lateral_periods(candles: list[dict], min_days: int = 30,
                           max_range_pct: float = 0.05) -> list[dict]:
    """Encontra períodos de lateralidade (range < max_range_pct por min_days)."""
    window_h = min_days * 24
    periods  = []
    i = window_h
    while i < len(candles):
        window = candles[i-window_h:i]
        highs  = [c["high"]  for c in window]
        lows   = [c["low"]   for c in window]
        h_max  = max(highs)
        l_min  = min(lows)
        rng    = (h_max - l_min) / l_min if l_min > 0 else 1.0
        if rng < max_range_pct:
            start = _ts(candles[i-window_h])
            end   = _ts(candles[i])
            if not periods or (start - datetime.fromisoformat(periods[-1]["end"])).days > 7:
                periods.append({
                    "start": start.isoformat(),
                    "end":   end.isoformat(),
                    "range_pct": round(rng*100, 2),
                    "days":  min_days,
                    "start_idx": i - window_h,
                    "end_idx":   i,
                })
            i += window_h // 2   # avança meio período
        else:
            i += 24   # avança 1 dia
    return periods[:5]   # máximo 5 períodos


def _find_bear_periods(candles: list[dict], top_n: int = 3) -> list[dict]:
    """Encontra os top N piores bear markets de 30 dias."""
    window_h = 720  # 30 dias
    results  = []
    for i in range(window_h, len(candles), 24):
        window = candles[i-window_h:i]
        ret    = (window[-1]["close"] - window[0]["close"]) / window[0]["close"]
        if ret < -0.10:   # só quedas > 10%
            results.append({
                "start":     _ts(candles[i-window_h]).isoformat(),
                "end":       _ts(candles[i]).isoformat(),
                "ret_pct":   round(ret*100, 2),
                "start_idx": i-window_h,
                "end_idx":   min(i + 240, len(candles)-1),  # +10 dias OOS
            })
    results.sort(key=lambda x: x["ret_pct"])
    # Deduplica (remove sobreposições)
    dedup = []
    for r in results:
        if not dedup or r["start_idx"] > dedup[-1]["end_idx"]:
            dedup.append(r)
    return dedup[:top_n]


def _find_vol_regimes(candles: list[dict]) -> dict:
    """Classifica CADA candle com M8 e analisa transições."""
    states = []
    for i in range(LOOKBACK, len(candles)):
        window = list(reversed(candles[max(0, i-LOOKBACK):i+1]))
        states.append(_detect_m8_state(window))

    from collections import Counter
    dist = dict(Counter(states))
    total = sum(dist.values()) or 1

    # Detecta compressões prolongadas (> 5 dias = 120 candles)
    compressions: list[dict] = []
    streak = 0
    streak_start = 0
    for i, s in enumerate(states):
        if s == "COMPRESSED":
            if streak == 0:
                streak_start = i
            streak += 1
        else:
            if streak >= 120:
                compressions.append({
                    "duration_h": streak,
                    "duration_days": round(streak/24, 1),
                    "start":  _ts(candles[streak_start + LOOKBACK]).isoformat(),
                })
            streak = 0
    # Detecta explosões de vol (CHAOTIC por > 12h)
    chaotic_streaks: list[dict] = []
    streak = 0
    streak_start = 0
    for i, s in enumerate(states):
        if s == "CHAOTIC":
            if streak == 0:
                streak_start = i
            streak += 1
        else:
            if streak >= 12:
                chaotic_streaks.append({
                    "duration_h": streak,
                    "start": _ts(candles[streak_start + LOOKBACK]).isoformat(),
                })
            streak = 0

    return {
        "distribution":    {k: {"n": v, "pct": round(v/total*100, 1)} for k, v in dist.items()},
        "compressions":    compressions[:5],
        "chaotic_streaks": chaotic_streaks[:5],
        "n_analyzed":      len(states),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 — Stress Backtest")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--start",  default=None, help="YYYY-MM-DD")
    parser.add_argument("--end",    default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    cache_file = CACHE_DIR / f"{args.symbol.replace('-','_')}_1H.json"
    if not cache_file.exists():
        log.error("Cache não encontrado: %s", cache_file)
        return

    all_candles = json.loads(cache_file.read_text(encoding="utf-8"))

    # Filtro de datas
    if args.start:
        start_ms = int(
            datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
        )
        all_candles = [c for c in all_candles if c["ts"] >= start_ms]
    if args.end:
        end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()*1000)
        all_candles = [c for c in all_candles if c["ts"] <= end_ms]

    first_dt = _ts(all_candles[0])
    last_dt  = _ts(all_candles[-1])

    log.info("═" * 65)
    log.info("  Phase 16 — Regime Robustness Stress Test")
    log.info("  Símbolo: %s | Período: %s → %s | %d candles",
             args.symbol, first_dt.date(), last_dt.date(), len(all_candles))
    log.info("═" * 65)

    results = {
        "symbol":  args.symbol,
        "period":  {"start": first_dt.isoformat(), "end": last_dt.isoformat()},
        "computed_at": datetime.now(UTC).isoformat(),
    }

    # ── 16.1 Lateralidade ──────────────────────────────────────────────────
    log.info("\n── 16.1 Lateralidade Prolongada (range < 5%% por >= 30 dias) ──")
    lateral_periods = _find_lateral_periods(all_candles)
    lateral_results = []

    if not lateral_periods:
        log.info("  Nenhum período de lateralidade > 30 dias encontrado no cache.")
    else:
        for p in lateral_periods:
            window = all_candles[p["start_idx"]:p["end_idx"] + 240]
            res = _run_window(window, f"Lateral {p['start'][:10]} ({p['range_pct']}% range)")
            res["range_pct"] = p["range_pct"]
            lateral_results.append(res)
            log.info("  %s → %s | range=%.2f%% | trades=%d WR=%.0f%% ret=%.2f%% | %s",
                     p["start"][:10], p["end"][:10], p["range_pct"],
                     res["n_trades"], res["win_rate"]*100,
                     res["total_return_pct"], res["verdict"])

    results["lateral"] = lateral_results

    # ── 16.2 Bear Markets ──────────────────────────────────────────────────
    log.info("\n── 16.2 Bear Markets (top 3 piores 30 dias) ──────────────────")
    bear_periods = _find_bear_periods(all_candles)
    bear_results = []

    for p in bear_periods:
        window = all_candles[max(0, p["start_idx"]-240) : p["end_idx"]]
        res = _run_window(window, f"Bear {p['start'][:10]} ({p['ret_pct']}%)")
        res["bear_ret_pct"] = p["ret_pct"]

        # Análise do lag de regime
        regime_switches = res["regime_switches"]
        bear_switch = next((s for s in regime_switches if s["to"] == "BEAR_TREND"), None)
        if bear_switch:
            # Quantas horas após o início da queda o BEAR_TREND ativou?
            crash_start = datetime.fromisoformat(p["start"])
            switch_ts   = datetime.fromisoformat(bear_switch["ts"])
            lag_h = (switch_ts - crash_start).total_seconds() / 3600
            res["regime_lag_hours"] = round(lag_h, 1)
            res["bear_activated"] = True
        else:
            res["regime_lag_hours"] = None
            res["bear_activated"]   = False

        bear_results.append(res)
        lag_info = f"lag={res['regime_lag_hours']}h" if res["bear_activated"] else "NAO ATIVOU"
        log.info("  %s | bear=%.1f%% | BEAR_TREND: %s | trades=%d WR=%.0f%% blocked=%.0f%%",
                 p["start"][:10], p["ret_pct"], lag_info,
                 res["n_trades"], res["win_rate"]*100, res["bear_regime_pct"])

    results["bear"] = bear_results

    # ── 16.3 Volatility Regimes ────────────────────────────────────────────
    log.info("\n── 16.3 Volatility State Machine — distribuição completa ──────")
    vol_analysis = _find_vol_regimes(all_candles)
    results["volatility"] = vol_analysis

    log.info("  Distribuição de estados M8:")
    for state, info in sorted(vol_analysis["distribution"].items(), key=lambda x: -x[1]["pct"]):
        log.info("    %-18s %5.1f%%  (%d candles)", state, info["pct"], info["n"])

    if vol_analysis["compressions"]:
        log.info("  Compressões > 5 dias:")
        for c in vol_analysis["compressions"]:
            log.info("    %s → %.1f dias", c["start"][:10], c["duration_days"])
    else:
        log.info("  Sem compressões > 5 dias no período.")

    if vol_analysis["chaotic_streaks"]:
        log.info("  Explosões de vol (CHAOTIC > 12h):")
        for c in vol_analysis["chaotic_streaks"]:
            log.info("    %s → %dh", c["start"][:10], c["duration_h"])

    # ── Full period baseline ───────────────────────────────────────────────
    log.info("\n── Baseline: modelo completo no período inteiro ──────────────")
    full = _run_window(all_candles, f"Full period {first_dt.date()} → {last_dt.date()}")
    results["full_period"] = full
    log.info("  Trades=%d  WR=%.1f%%  Ret=%.2f%%  MaxDD=%.2f%%  BEAR_blocked=%.0f%%",
             full["n_trades"], full["win_rate"]*100, full["total_return_pct"],
             full["max_drawdown_pct"], full["bear_regime_pct"])
    log.info("  Veredicto: %s", full["verdict"])

    # ── Sumário final ──────────────────────────────────────────────────────
    log.info("\n═" * 65 + "═")
    log.info("  SUMÁRIO PHASE 16 — %s", args.symbol)
    log.info("═" * 65 + "═")

    # Circuit breaker analysis
    any_bear_failed = any(not r.get("bear_activated", False) for r in bear_results)
    if any_bear_failed:
        log.info("  ⚠ BEAR_TREND NAO ATIVOU em alguns períodos de queda")
    else:
        log.info("  ✓ BEAR_TREND ativou em todos os bear markets detectados")

    avg_lag = [r["regime_lag_hours"] for r in bear_results if r.get("regime_lag_hours")]
    if avg_lag:
        log.info("  Lag médio de ativação: %.0fh (%.0f candles de exposição)",
                 sum(avg_lag)/len(avg_lag), sum(avg_lag)/len(avg_lag))

    # CB all-assets-crash: verifica se -3% em 1H teria disparado
    crash_events = 0
    for i in range(1, len(all_candles)):
        ret_1h = (all_candles[i]["close"] - all_candles[i-1]["close"]) / all_candles[i-1]["close"]
        if ret_1h < -0.03:
            crash_events += 1
    log.info("  Eventos de -3%%+ em 1H (CB all-crash): %d vezes no período", crash_events)

    log.info("  M8 COMPRESSED: %.1f%% do tempo (%.0f candles)",
             vol_analysis["distribution"].get("COMPRESSED", {}).get("pct", 0),
             vol_analysis["distribution"].get("COMPRESSED", {}).get("n", 0))
    log.info("  M8 CHAOTIC: %.1f%% do tempo",
             vol_analysis["distribution"].get("CHAOTIC", {}).get("pct", 0))
    log.info("═" * 65 + "═")

    # Salva
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Salvo em: %s", OUTPUT)


if __name__ == "__main__":
    main()
