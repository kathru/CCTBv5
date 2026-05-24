#!/usr/bin/env python3
"""
Walk-Forward Optimization (WFO) — CCTBv5

Valida a estratégia V4 contra dados OOS (out-of-sample) reais para
garantir que a calibração não está overfitada ao período de treino.

Metodologia:
  - Janela expandida (anchored): treino sempre começa na mesma data
  - Cada fold adiciona mais dados de treino e testa no período seguinte
  - Resultado final é a média das métricas OOS de todos os folds válidos

Output:
  - Tabela de folds com métricas IS e OOS
  - Score de estabilidade (quão consistente é a performance OOS)
  - Recomendação: os coeficientes Platt do fold mais recente com dados reais

Uso:
  python scripts/walk_forward.py
  python scripts/walk_forward.py --symbol BTC-USDT --train-months 6 --test-months 2
  python scripts/walk_forward.py --start 2024-01-01 --no-cache
"""

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ── Bootstrap path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.core.models import Candle  # noqa: E402
from src.replay.backtest_engine import BacktestEngine  # noqa: E402
from src.strategies.ml.inference import PlattCalibrator  # noqa: E402
from src.strategies.momentum.momentum_strategy import MomentumStrategy  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,   # silencia logs da estratégia durante backtest
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wfo")
log.setLevel(logging.INFO)

# ── Constantes ────────────────────────────────────────────────────────────────
CACHE_DIR   = ROOT / "data" / "cache"
OUTPUT_DIR  = ROOT / "data" / "wfo"
OKX_BASE    = "https://www.okx.com"
GRAN_MS     = {"1H": 3_600_000, "4H": 14_400_000}
DEFAULT_FEE = 0.005   # round-trip 0.5%

PLATT_A_INIT = 0.378188
PLATT_B_INIT = -1.075301


# ── Candle fetching (reusa cache do recalibrate.py) ───────────────────────────

def fetch_candles(symbol: str, start_dt: datetime, end_dt: datetime,
                  use_cache: bool = True) -> list[dict]:
    """Baixa candles 1H da OKX ou usa cache local (fallback 4H)."""
    cache_file = CACHE_DIR / f"{symbol.replace('-','_')}_1H.json"
    if not cache_file.exists():
        cache_file = CACHE_DIR / f"{symbol.replace('-','_')}_4H.json"

    if use_cache and cache_file.exists():
        all_candles = json.loads(cache_file.read_text())
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp()   * 1000)
        filtered = [c for c in all_candles if start_ms <= c["ts"] <= end_ms]
        if filtered:
            log.info("Cache: %d candles para %s (%s → %s)",
                     len(filtered), symbol,
                     start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            return filtered

    log.info("Baixando %s de %s até %s...",
             symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    end_ms    = int(end_dt.timestamp()   * 1000)
    start_ms  = int(start_dt.timestamp() * 1000)
    after_ms  = end_ms
    candles   = []

    while True:
        url = (f"{OKX_BASE}/api/v5/market/history-candles"
               f"?instId={symbol}&bar=1H&limit=100&after={after_ms}")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            rows = resp.json().get("data", [])
        except Exception as exc:
            log.warning("Erro na API: %s — aguardando 5s", exc)
            time.sleep(5)
            continue

        if not rows:
            break

        for row in rows:
            ts_ms = int(row[0])
            if row[8] == "1" and start_ms <= ts_ms <= end_ms:
                candles.append({
                    "ts":     ts_ms,
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })
            if ts_ms <= start_ms:
                candles.sort(key=lambda c: c["ts"])
                return candles

        oldest = int(rows[-1][0])
        if oldest <= start_ms:
            break
        after_ms = oldest - 1
        time.sleep(0.25)

    candles.sort(key=lambda c: c["ts"])
    return candles


def to_candle_objects(raw: list[dict], symbol: str) -> list[Candle]:
    """Converte dicts para objetos Candle."""
    result = []
    for r in raw:
        result.append(Candle(
            symbol=symbol,
            granularity="1H",
            timestamp=datetime.fromtimestamp(r["ts"] / 1000, tz=UTC),
            open=r["open"], high=r["high"], low=r["low"],
            close=r["close"], volume=r["volume"],
            confirmed=True,
        ))
    return result


# ── Calibração Platt (mesmo algoritmo do recalibrate.py) ─────────────────────

def _atr_wfo(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> float:
    n = min(period, len(highs) - 1)
    if n <= 0:
        return (highs[0] - lows[0]) if highs else 0.0
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i+1]),
               abs(lows[i] - closes[i+1])) for i in range(n)]
    return sum(trs) / len(trs) if trs else 0.0


def _bollinger_wfo(closes: list[float], period: int = 20) -> tuple[float, float, float]:
    import math as _math
    n = min(period, len(closes))
    if n < 2:
        c = closes[0]
        return c, c, c
    window = closes[:n]
    mid = sum(window) / n
    std = _math.sqrt(sum((x - mid) ** 2 for x in window) / n)
    return mid + 2 * std, mid, mid - 2 * std


def _compute_m8_wfo(closes: list[float], highs: list[float], lows: list[float]) -> float:
    """M8 Volatility State — espelho de volatility_state.compute_vol_state()."""
    if len(closes) < 22:
        return 0.5
    price    = closes[0]
    atr_now  = _atr_wfo(highs, lows, closes, 14)
    atr_prev = _atr_wfo(highs[7:], lows[7:], closes[7:], 14)
    bb_upper, _, bb_lower = _bollinger_wfo(closes, 20)
    atr_pct    = atr_now / price if price > 0 else 0.0
    atr_change = atr_now / atr_prev if atr_prev > 0 else 1.0
    bb_w_pct   = (bb_upper - bb_lower) / price if price > 0 else 0.0
    n_dir = min(10, len(closes) - 1)
    ups   = sum(1 for i in range(n_dir) if closes[i] > closes[i+1])
    dir_c = max(ups, n_dir - ups) / n_dir if n_dir > 0 else 0.5
    STATE_M8 = {"EXPANDING":0.80,"TREND":0.70,"COMPRESSED":0.65,
                "MEAN_REVERTING":0.35,"CHAOTIC":0.20}
    if atr_pct > 0.025 and dir_c < 0.45:
        state = "CHAOTIC"
    elif atr_change > 1.15 and dir_c > 0.55:
        state = "EXPANDING"
    elif atr_pct < 0.008 or bb_w_pct < 0.015:
        state = "COMPRESSED"
    elif dir_c > 0.60 and 0.008 <= atr_pct <= 0.025:
        state = "TREND"
    else:
        state = "MEAN_REVERTING"
    return STATE_M8[state]


def score_raw(candles_window: list[dict]) -> float | None:
    """
    Computa score bruto M1-M8 para um ponto.
    DEVE ser idêntico ao _score_signal() em momentum_strategy.py.

    Pesos: M1=18% M2=18% M3=14% M4=14% M5=6% M6=10%(neutro) M7=9%(neutro) M8=11%

    M6 (Futures Flow): neutro 0.5 — requer API OKX em tempo real
    M7 (Rel.Strength): neutro 0.5 — requer multi-símbolo simultâneo
    M8 (Vol.State):    calculado de candles
    """
    if len(candles_window) < 25:
        return None

    closes  = [c["close"]  for c in candles_window[:25]]
    opens   = [c["open"]   for c in candles_window[:10]]
    highs   = [c["high"]   for c in candles_window[:25]]
    lows    = [c["low"]    for c in candles_window[:25]]
    volumes = [c["volume"] for c in candles_window[:20]]

    # ── Score v2.5.0 — espelha momentum_strategy._score_signal() ────────────
    # M1  2% | M2 11% | M3 33% | M4 0% (removido) | M5 6%
    # M6 12% | M7 11% | M8 19% | M9 6%  → Soma=100%
    # M6/M7/M9 neutros no backtest histórico (sem dados de futuros/RS/news)

    # M1 — Adaptive Momentum (2%)
    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r1   = (closes[0]-closes[1])/closes[1]   if len(closes)>1  and closes[1]>0  else 0
    r5   = (closes[0]-closes[5])/closes[5]   if len(closes)>5  and closes[5]>0  else 0
    r10  = (closes[0]-closes[10])/closes[10] if len(closes)>10 and closes[10]>0 else 0
    r20  = (closes[0]-closes[20])/closes[20] if len(closes)>20 and closes[20]>0 else 0
    mw   = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
    m1   = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    # M2 — Trend Consistency (11%)
    n    = min(6, len(closes)-1)
    bull = sum(1 for i in range(n) if closes[i]>opens[i])/n if n>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    m2   = bull*0.5 + (hh+hl)/2*0.5

    # M3 — Volume Confirmation (33%)
    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr  = min(volumes[0]/avg5, 3.0)/3.0 if avg5>0 else 0.5
    vt  = (
        min(max(sum(volumes[:3]) / sum(volumes[3:6]), 0.3), 2.0)
        if len(volumes) >= 6 and sum(volumes[3:6]) > 0 else 1.0
    )
    cc  = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    m3  = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4 — REMOVIDO (peso 0% desde v2.5.0) — mantido computado para diagnóstico

    # M5 — Candle Structure (6%)
    cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
          for i in range(min(3, len(closes)))]
    m5 = sum(cs)/len(cs) if cs else 0.5

    # M6 — Futures Flow (12%) — neutro no backtest histórico (sem API futuros)
    m6 = 0.5

    # M7 — Relative Strength (11%) — neutro no backtest histórico
    m7 = 0.5

    # M8 — Volatility State (19%) — calculado de candles
    m8 = _compute_m8_wfo(closes, highs, lows)

    # M9 — News Sentiment (6%) — neutro no backtest histórico (sem API)
    m9 = 0.5

    return (m1*0.02 + m2*0.11 + m3*0.33 +
            m5*0.06 + m6*0.12 + m7*0.11 + m8*0.19 + m9*0.06)


def fit_platt_on_period(candles: list[dict], forward: int = 5,
                        fee: float = 0.005,
                        min_score: float = 0.50) -> tuple[float, float, int, float]:
    """
    Calibra coeficientes Platt em um período de treino.
    Usa apenas amostras com score >= min_score (igual ao threshold real).
    Isso evita contaminar a calibração com sinais fracos que a estratégia
    real nunca executaria.
    Retorna (A, B, n_samples, win_rate).
    """
    scores, labels = [], []
    n = len(candles)

    for i in range(25, n - forward):
        window = list(reversed(candles[max(0, i-25):i+1]))
        score  = score_raw(window)
        if score is None or score < min_score:   # era 0.3 — contaminava com sinais fracos
            continue

        entry = candles[i]["close"]
        exit_ = candles[i + forward]["close"]
        net   = (exit_ - entry) / entry - fee
        labels.append(1 if net > 0 else 0)
        scores.append(score)

    if len(scores) < 10:
        return PLATT_A_INIT, PLATT_B_INIT, 0, 0.0

    # Gradiente descendente
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.float64)
    A, B = PLATT_A_INIT, PLATT_B_INIT

    for _ in range(1000):
        p     = 1.0 / (1.0 + np.exp(-(A * scores_arr + B)))
        p     = np.clip(p, 1e-7, 1 - 1e-7)
        error = p - labels_arr
        A    -= 0.1 * float(np.mean(error * scores_arr))
        B    -= 0.1 * float(np.mean(error))

    win_rate = float(np.mean(labels_arr))
    return float(A), float(B), len(scores), win_rate


# ── Métricas de resultado ─────────────────────────────────────────────────────

def compute_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    std  = statistics.stdev(returns)
    return (mean / std) * math.sqrt(252) if std > 0 else 0.0


def compute_max_dd(pnls: list[float]) -> float:
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak if peak > 0 else 0.0
        max_dd  = max(max_dd, dd)
    return max_dd


# ── Walk-Forward engine ───────────────────────────────────────────────────────

class WalkForwardResult:
    def __init__(self) -> None:
        self.folds: list[dict] = []

    def add(self, fold: dict) -> None:
        self.folds.append(fold)

    def summary(self) -> dict:
        valid = [f for f in self.folds if f["oos_trades"] >= 3]
        if not valid:
            return {"valid_folds": 0, "message": "Nenhum fold com trades suficientes"}

        oos_wr   = [f["oos_win_rate"]    for f in valid]
        oos_exp  = [f["oos_expectancy"]  for f in valid]
        oos_pf   = [f["oos_pf"]          for f in valid]
        oos_sh   = [f["oos_sharpe"]      for f in valid]
        oos_dd   = [f["oos_max_dd"]      for f in valid]
        oos_ret  = [f["oos_return"]      for f in valid]

        # Stability: consistência dos retornos OOS
        stability = 1.0 - statistics.stdev(oos_ret) / (abs(statistics.mean(oos_ret)) + 1e-6) \
                    if len(oos_ret) > 1 else 0.5

        return {
            "valid_folds":     len(valid),
            "total_folds":     len(self.folds),
            "avg_win_rate":    round(statistics.mean(oos_wr),  4),
            "avg_expectancy":  round(statistics.mean(oos_exp), 4),
            "avg_pf":          round(statistics.mean(oos_pf),  4),
            "avg_sharpe":      round(statistics.mean(oos_sh),  3),
            "avg_max_dd":      round(statistics.mean(oos_dd),  4),
            "avg_oos_return":  round(statistics.mean(oos_ret), 4),
            "stability_score": round(max(0, min(1, stability)), 4),
            "best_A": valid[-1]["platt_a"],   # fold mais recente
            "best_B": valid[-1]["platt_b"],
        }


async def run_fold_backtest(
    symbol: str,
    test_candles: list[Candle],
    platt_a: float,
    platt_b: float,
    initial_capital: float = 10000.0,
) -> dict:
    """Cria estratégia com coeficientes do fold e roda backtest."""

    class CalibratedStrategy(MomentumStrategy):
        """Strategy com Platt coefficients injetados para este fold.
        Replica o comportamento de produção: BEAR_TREND e PANIC bloqueiam entradas.
        Isso garante que o WFO mede performance nas mesmas condições que o sistema real.
        """
        def __init__(self):
            super().__init__(symbols=[symbol])
            self._platt = PlattCalibrator.__new__(PlattCalibrator)
            self._platt._A       = platt_a
            self._platt._B       = platt_b
            self._platt._loaded  = True
            self._platt._regime_thresholds = {}

        def _confirm_regime_mtf(self, ctx, regime_1h):
            # No WFO: sem candles_6h disponíveis — retorna regime_1h diretamente.
            # Mantém BEAR_TREND e PANIC_LIQUIDATION bloqueados como em produção.
            return regime_1h

    strategy = CalibratedStrategy()
    engine   = BacktestEngine(
        strategy=strategy,
        symbol=symbol,
        initial_capital=initial_capital,
        position_size_pct=0.08,   # 8% por trade (CHOP sizing)
        seed=42,
    )

    if len(test_candles) < 30:
        return {"trades": 0, "win_rate": 0, "expectancy": 0,
                "pf": 0, "sharpe": 0, "max_dd": 0, "return": 0}

    result = await engine.run(test_candles, warmup=25)

    pnls    = [t.pnl for t in result.trades]
    returns = [t.pnl_pct for t in result.trades]

    return {
        "trades":     result.total_trades,
        "win_rate":   round(result.win_rate, 4),
        "expectancy": round(result.expectancy, 4),
        "pf":         round(result.profit_factor, 3),
        "sharpe":     round(compute_sharpe(returns), 3),
        "max_dd":     round(compute_max_dd(pnls), 4),
        "return":     round(result.total_return_pct, 4),
    }


async def walk_forward(
    symbol: str,
    all_candles: list[dict],
    train_months: int = 6,
    test_months:  int = 2,
    initial_capital: float = 10000.0,
    forward_candles: int = 10,   # 10×1H = 10h
    purged: bool = False,        # Phase 15.3: Purged CV
    embargo_candles: int = 5,    # candles de embargo entre folds
) -> WalkForwardResult:
    """
    Executa WFO com janela expandida.
    Para cada fold: treina Platt no período IS, testa backtest no OOS.
    """
    wfo = WalkForwardResult()
    if not all_candles:
        return wfo

    # Determina os limites de tempo
    start_ts = all_candles[0]["ts"]
    end_ts   = all_candles[-1]["ts"]

    train_ms = train_months * 30 * 24 * 3_600_000
    test_ms  = test_months  * 30 * 24 * 3_600_000

    fold_num    = 0
    test_start  = start_ts + train_ms

    # Embargo em ms (purged CV — 15.3)
    embargo_ms = embargo_candles * GRAN_MS.get("1H", 3_600_000) if purged else 0

    while test_start + test_ms <= end_ts:
        fold_num  += 1
        test_end   = test_start + test_ms

        # Partição IS (treino) e OOS (teste)
        # Purged CV (15.3): remove candles contaminados
        # (forward_candles após último candle de treino)
        # e adiciona embargo period antes do início do teste
        purge_ms   = forward_candles * GRAN_MS.get("1H", 3_600_000) if purged else 0
        train_end  = test_start - embargo_ms
        train_data = [c for c in all_candles if c["ts"] < (train_end - purge_ms)]
        test_data  = [c for c in all_candles if test_start <= c["ts"] < test_end]

        if len(train_data) < 100 or len(test_data) < 50:
            test_start += test_ms
            continue

        train_start_dt = datetime.fromtimestamp(train_data[0]["ts"]/1000, tz=UTC)
        train_end_dt   = datetime.fromtimestamp(train_data[-1]["ts"]/1000, tz=UTC)
        test_start_dt  = datetime.fromtimestamp(test_data[0]["ts"]/1000, tz=UTC)
        test_end_dt    = datetime.fromtimestamp(test_data[-1]["ts"]/1000, tz=UTC)

        log.info("Fold %2d │ IS: %s→%s (%d candles) │ OOS: %s→%s (%d candles)",
                 fold_num,
                 train_start_dt.strftime("%Y-%m"),
                 train_end_dt.strftime("%Y-%m"),
                 len(train_data),
                 test_start_dt.strftime("%Y-%m"),
                 test_end_dt.strftime("%Y-%m"),
                 len(test_data))

        # 1. Calibra Platt no IS
        A, B, n_samples, is_wr = fit_platt_on_period(
            train_data, forward=forward_candles
        )

        # 2. Backtest no OOS com os coeficientes calibrados
        test_candles_obj = to_candle_objects(test_data, symbol)
        oos = await run_fold_backtest(symbol, test_candles_obj, A, B, initial_capital)

        fold = {
            "fold":           fold_num,
            "is_start":       train_start_dt.strftime("%Y-%m-%d"),
            "is_end":         train_end_dt.strftime("%Y-%m-%d"),
            "oos_start":      test_start_dt.strftime("%Y-%m-%d"),
            "oos_end":        test_end_dt.strftime("%Y-%m-%d"),
            "is_n_samples":   n_samples,
            "is_win_rate":    round(is_wr, 4),
            "platt_a":        round(A, 6),
            "platt_b":        round(B, 6),
            "oos_trades":     oos["trades"],
            "oos_win_rate":   oos["win_rate"],
            "oos_expectancy": oos["expectancy"],
            "oos_pf":         oos["pf"],
            "oos_sharpe":     oos["sharpe"],
            "oos_max_dd":     oos["max_dd"],
            "oos_return":     oos["return"],
            "valid":          oos["trades"] >= 3,
        }
        wfo.add(fold)

        # Avança para o próximo fold
        test_start += test_ms

    return wfo


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(wfo: WalkForwardResult, symbol: str) -> None:
    summary = wfo.summary()

    log.info("")
    log.info("═" * 100)
    log.info("  CCTBv5 Walk-Forward Results — %s", symbol)
    log.info("═" * 100)

    # Header da tabela
    log.info("  %-4s │ %-12s │ %-12s │ %-8s %-8s %-8s │ %-6s %-6s %-6s %-7s %-6s %s",
             "Fold", "IS", "OOS", "IS_Samp", "IS_WR", "Platt A",
             "Trades", "WinR", "Exp", "PF", "Sharpe", "OOS Ret")
    log.info("  " + "─" * 98)

    for f in wfo.folds:
        valid_mark = "✓" if f["valid"] else "✗"
        log.info(
            "  %2d%s  │ %s → %s │ %s → %s │ %7d  %5.1f%%  %7.4f │ "
            "%5d  %5.1f%%  %+6.2f  %5.2f  %6.2f  %+6.2f%%",
            f["fold"], valid_mark,
            f["is_start"][:7], f["is_end"][:7],
            f["oos_start"][:7], f["oos_end"][:7],
            f["is_n_samples"], f["is_win_rate"]*100, f["platt_a"],
            f["oos_trades"],    f["oos_win_rate"]*100, f["oos_expectancy"],
            f["oos_pf"],        f["oos_sharpe"],       f["oos_return"]*100,
        )

    log.info("  " + "─" * 98)
    log.info("")

    if "message" in summary:
        log.info("  %s", summary["message"])
        return

    # Resumo OOS
    log.info("  RESUMO OOS (%d/%d folds válidos):",
             summary["valid_folds"], summary["total_folds"])
    log.info("  Win Rate médio    : %.1f%%", summary["avg_win_rate"] * 100)
    log.info("  Expectancy médio  : %+.2f", summary["avg_expectancy"])
    log.info("  Profit Factor     : %.2f",  summary["avg_pf"])
    log.info("  Sharpe OOS médio  : %.2f",  summary["avg_sharpe"])
    log.info("  Max DD OOS médio  : %.1f%%", summary["avg_max_dd"] * 100)
    log.info("  Retorno OOS médio : %+.2f%%", summary["avg_oos_return"] * 100)
    log.info("  Stability Score   : %.3f  (1.0=perfeito, >0.5 aceitável)",
             summary["stability_score"])
    log.info("")

    # Interpretação
    log.info("  INTERPRETAÇÃO:")
    if summary["avg_oos_return"] > 0:
        log.info("  ✓ Retorno OOS positivo em média — estratégia tem edge real")
    else:
        log.info("  ✗ Retorno OOS negativo — rever thresholds ou scoring")

    if summary["avg_pf"] > 1.2:
        log.info("  ✓ Profit Factor > 1.2 — edge estatístico presente")
    else:
        log.info("  ✗ Profit Factor baixo — poucos trades ou edge fraco")

    if summary["stability_score"] > 0.5:
        log.info("  ✓ Performance estável entre folds — não overfitado")
    else:
        log.info("  ✗ Alta variância entre folds — possível overfitting")

    # Coeficientes recomendados
    log.info("")
    log.info("  COEFICIENTES RECOMENDADOS (fold mais recente válido):")
    log.info("  platt_a = %.6f  (atual: %.6f)", summary["best_A"], PLATT_A_INIT)
    log.info("  platt_b = %.6f  (atual: %.6f)", summary["best_B"], PLATT_B_INIT)
    log.info("═" * 100)


def save_results(wfo: WalkForwardResult, symbol: str, output_path: Path) -> None:
    """Salva resultados em JSON (timestamped + latest para o dashboard)."""
    output_path.mkdir(parents=True, exist_ok=True)
    data = {
        "symbol":    symbol,
        "run_at":    datetime.now(UTC).isoformat(),
        "summary":   wfo.summary(),
        "folds":     wfo.folds,
    }
    # Arquivo timestamped (histórico)
    ts_str = datetime.now(UTC).strftime('%Y%m%d_%H%M')
    fname = output_path / f"wfo_{symbol.replace('-','_')}_{ts_str}.json"
    fname.write_text(json.dumps(data, indent=2))
    log.info("Resultados salvos em: %s", fname)

    # Arquivo latest por símbolo (para o dashboard via API)
    latest = output_path / f"wfo_{symbol.replace('-','_')}_latest.json"
    latest.write_text(json.dumps(data, indent=2))
    log.info("Latest atualizado: %s", latest)


def auto_apply_wfo(all_results: dict[str, WalkForwardResult], models_dir: Path) -> bool:
    """
    Se OOS positivo E estável em TODOS os símbolos avaliados,
    atualiza calibration_coef.json com os coeficientes do fold mais recente.

    Critérios de segurança para auto-apply:
      - avg_oos_return > 0 (edge positivo OOS)
      - stability_score > 0.5 (performance consistente entre folds)
      - valid_folds >= 2 (mínimo de evidência)
      - Pelo menos 1 símbolo aprovado

    Retorna True se atualizou.
    """
    coef_path = models_dir / "calibration_coef.json"
    if not coef_path.exists():
        log.warning("auto-apply: calibration_coef.json não encontrado em %s", models_dir)
        return False

    # Coleta sumários aprovados
    approved: list[dict] = []
    for sym, wfo in all_results.items():
        s = wfo.summary()
        if not s.get("valid_folds"):
            log.info("auto-apply: %s — sem folds válidos, ignorando", sym)
            continue
        oos_ok    = s.get("avg_oos_return", 0) > 0
        stab_ok   = s.get("stability_score", 0) > 0.5
        folds_ok  = s.get("valid_folds", 0) >= 2
        if oos_ok and stab_ok and folds_ok:
            approved.append({"symbol": sym, **s})
            log.info(
                "auto-apply: %s APROVADO — OOS=%.2f%% stability=%.3f folds=%d",
                sym, s["avg_oos_return"] * 100, s["stability_score"], s["valid_folds"],
            )
        else:
            log.info(
                "auto-apply: %s REJEITADO — OOS=%.2f%% stability=%.3f folds=%d",
                sym, s.get("avg_oos_return", 0) * 100,
                s.get("stability_score", 0), s.get("valid_folds", 0),
            )

    if not approved:
        log.info("auto-apply: nenhum símbolo aprovado — coeficientes mantidos")
        return False

    # Usa os coeficientes do símbolo com maior Sharpe OOS (mais robusto)
    best = max(approved, key=lambda x: x.get("avg_sharpe", 0))
    new_a = best["best_A"]
    new_b = best["best_B"]

    # Lê coef atual para salvar como previous
    try:
        existing = json.loads(coef_path.read_text())
    except Exception:
        existing = {}

    updated = {
        **existing,
        "platt_a":       new_a,
        "platt_b":       new_b,
        "wfo_applied_at": datetime.now(UTC).isoformat(),
        "wfo_source":    best["symbol"],
        "wfo_oos_return": round(best["avg_oos_return"], 6),
        "wfo_stability":  round(best["stability_score"], 4),
        "wfo_valid_folds": best["valid_folds"],
        "previous": {
            "platt_a": existing.get("platt_a"),
            "platt_b": existing.get("platt_b"),
        },
    }
    coef_path.write_text(json.dumps(updated, indent=2))

    log.info(
        "auto-apply: coeficientes atualizados A=%.6f→%.6f B=%.6f→%.6f (fonte: %s)",
        existing.get("platt_a", 0), new_a,
        existing.get("platt_b", 0), new_b,
        best["symbol"],
    )
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    symbols = args.symbols if args.symbols else ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt   = datetime.now(UTC) if not args.end else \
               datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    log.info("Walk-Forward CCTBv5")
    log.info("Período   : %s → %s", args.start, end_dt.strftime("%Y-%m-%d"))
    log.info("Símbolos  : %s", ", ".join(symbols))
    log.info("Treino    : %d meses  │  Teste: %d meses  │  Step: %d meses",
             args.train_months, args.test_months, args.test_months)
    log.info("")

    all_results: dict[str, WalkForwardResult] = {}

    for symbol in symbols:
        log.info("── %s ──────────────────────────────────────", symbol)
        raw = fetch_candles(symbol, start_dt, end_dt, use_cache=not args.no_cache)

        if len(raw) < 200:
            log.warning("%s: candles insuficientes (%d) — ignorando", symbol, len(raw))
            continue

        wfo = await walk_forward(
            symbol=symbol,
            all_candles=raw,
            train_months=args.train_months,
            test_months=args.test_months,
            initial_capital=args.capital,
            purged=getattr(args, "purged", False),
            embargo_candles=getattr(args, "embargo_candles", 5),
        )

        all_results[symbol] = wfo
        print_report(wfo, symbol)

        if not args.dry_run:
            save_results(wfo, symbol, OUTPUT_DIR)

    # Auto-apply: atualiza calibration_coef.json se critérios de segurança passarem
    if args.auto_apply and not args.dry_run:
        models_dir = ROOT / "data" / "models"
        applied = auto_apply_wfo(all_results, models_dir)
        if not applied:
            log.info("auto-apply: edge insuficiente — coeficientes não atualizados")
    elif not args.dry_run and all_results:
        # Modo manual: mostra sugestão
        log.info("")
        log.info("Para aplicar coeficientes WFO automaticamente use: --auto-apply")
        log.info(
            "Para recalibrar manualmente: python scripts/recalibrate.py --start %s",
            args.start,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization CCTBv5")
    parser.add_argument("--symbols",       nargs="+", default=None)
    parser.add_argument("--start",         default="2024-01-01",
                        help="Início do período (YYYY-MM-DD)")
    parser.add_argument("--end",           default=None)
    parser.add_argument("--train-months",  type=int,   default=6)
    parser.add_argument("--test-months",   type=int,   default=2)
    parser.add_argument("--capital",       type=float, default=10000.0)
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Não salva arquivos")
    parser.add_argument("--auto-apply",    action="store_true",
                        help="Atualiza calibration_coef.json se OOS positivo e estável")
    parser.add_argument("--purged",        action="store_true",
                        help="Phase 15.3: Purged CV — remove contaminação temporal entre folds")
    parser.add_argument("--embargo-candles", type=int, default=5,
                        help=(
                            "Candles de embargo entre treino e teste "
                            "no modo --purged (default: 5)"
                        ))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
