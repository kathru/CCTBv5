#!/usr/bin/env python3
"""
Recalibração do sistema de sinais — com suporte incremental via SQLite.

Modos:
  python scripts/recalibrate.py --incremental
      Busca apenas candles novos desde o último registro no banco,
      gera amostras incrementais e refita Platt em todo o histórico.
      Ideal para rodar diariamente.

  python scripts/recalibrate.py --rebuild
      Baixa tudo do zero (usa --start como ponto de partida),
      reconstrói o banco e refita. Use após mudanças no score_signal.

  python scripts/recalibrate.py --dry-run
      Simula sem gravar nada.

Requisitos:
  pip install numpy requests python-dotenv
"""

import argparse
import json
import logging
import math
import sqlite3
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

ROOT       = Path(__file__).parent.parent
OUTPUT     = ROOT / "data" / "models" / "calibration_coef.json"
DB_PATH    = ROOT / "data" / "calibration.db"
CACHE_DIR  = ROOT / "data" / "cache"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS     = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
DEFAULT_START       = "2024-01-01"
DEFAULT_GRANULARITY = "1H"
DEFAULT_FORWARD     = 10   # 10 × 1H = 10h (janela de avaliação)
DEFAULT_FEE         = 0.005
DEFAULT_MIN_SCORE   = 0.40

LOOKBACK = 25   # candles de janela para score_signal (25 para M8 Bollinger 20 + buffer)

REGIME_THRESHOLDS_DEFAULT: dict[str, float] = {
    "TREND_EXPANSION":        0.56,
    "VOLATILITY_COMPRESSION": 0.60,
    "TREND_EXHAUSTION":       0.68,
    "MEAN_REVERTING_CHOP":    0.72,
    "HIGH_CORRELATION_RISK":  0.75,
    "PANIC_LIQUIDATION":      0.99,
    "LIQUIDITY_VACUUM":       0.99,
}

PLATT_A_PRIOR = 0.3567
PLATT_B_PRIOR = -1.0067


# ── SQLite ────────────────────────────────────────────────────────────────────

def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol   TEXT NOT NULL,
            gran     TEXT NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL NOT NULL,
            high     REAL NOT NULL,
            low      REAL NOT NULL,
            close    REAL NOT NULL,
            volume   REAL NOT NULL,
            PRIMARY KEY (symbol, gran, ts)
        );
        CREATE TABLE IF NOT EXISTS samples (
            symbol   TEXT    NOT NULL,
            gran     TEXT    NOT NULL,
            ts       INTEGER NOT NULL,
            score    REAL    NOT NULL,
            regime   TEXT    NOT NULL,
            label    INTEGER NOT NULL,
            forward  INTEGER NOT NULL,
            fee      REAL    NOT NULL,
            PRIMARY KEY (symbol, gran, ts, forward, fee)
        );
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol   TEXT    NOT NULL,
            ts       INTEGER NOT NULL,
            rate     REAL    NOT NULL,
            PRIMARY KEY (symbol, ts)
        );
    """)
    conn.commit()
    return conn


def db_last_ts(conn: sqlite3.Connection, symbol: str, gran: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM candles WHERE symbol=? AND gran=?", (symbol, gran)
    ).fetchone()
    return row[0] if row and row[0] else None


def db_insert_candles(conn: sqlite3.Connection, symbol: str, gran: str,
                      candles: list[dict]) -> int:
    rows = [
        (symbol, gran, c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"])
        for c in candles
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


def db_load_candles(conn: sqlite3.Connection, symbol: str, gran: str,
                    since_ts: int | None = None) -> list[dict]:
    if since_ts:
        rows = conn.execute(
            "SELECT ts,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND gran=? AND ts>=? ORDER BY ts",
            (symbol, gran, since_ts),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND gran=? ORDER BY ts",
            (symbol, gran),
        ).fetchall()
    return [
        {"ts": r[0], "open": r[1], "high": r[2], "low": r[3],
         "close": r[4], "volume": r[5]}
        for r in rows
    ]


def db_insert_samples(conn: sqlite3.Connection, samples: list[dict],
                      forward: int, fee: float) -> int:
    rows = [
        (s["symbol"], s["gran"], s["ts"], s["score"],
         s["regime"], s["label"], forward, fee)
        for s in samples
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


def db_load_samples(conn: sqlite3.Connection,
                    forward: int, fee: float) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol,ts,score,regime,label FROM samples "
        "WHERE forward=? AND fee=? ORDER BY ts",
        (forward, fee),
    ).fetchall()
    return [
        {"symbol": r[0], "ts": r[1], "score": r[2], "regime": r[3], "label": r[4]}
        for r in rows
    ]


def db_clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM candles")
    conn.execute("DELETE FROM samples")
    conn.commit()
    log.info("Banco limpo para rebuild completo.")


def db_insert_funding_rates(conn: sqlite3.Connection, symbol: str,
                             rows: list[tuple[int, float]]) -> int:
    conn.executemany(
        "INSERT OR IGNORE INTO funding_rates (symbol, ts, rate) VALUES (?,?,?)",
        [(symbol, ts, rate) for ts, rate in rows],
    )
    conn.commit()
    return len(rows)


def db_load_funding_lookup(conn: sqlite3.Connection, symbol: str) -> dict[int, float]:
    """Retorna {ts_ms: funding_rate} para lookup rápido por timestamp."""
    rows = conn.execute(
        "SELECT ts, rate FROM funding_rates WHERE symbol=? ORDER BY ts",
        (symbol,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def db_last_funding_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM funding_rates WHERE symbol=?", (symbol,)
    ).fetchone()
    return row[0] if row and row[0] else None


# ── OKX Fetcher ───────────────────────────────────────────────────────────────

OKX_BASE = "https://www.okx.com"
GRAN_MS   = {"1H": 3_600_000, "4H": 14_400_000, "6H": 21_600_000, "1D": 86_400_000}


def fetch_candles_range(symbol: str, granularity: str,
                        start_dt: datetime, end_dt: datetime) -> list[dict]:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    after_ms = end_ms
    candles: list[dict] = []
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
            ts_ms     = int(row[0])
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

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        after_ms = oldest_ts - 1
        page += 1
        time.sleep(0.25)

    candles.sort(key=lambda c: c["ts"])
    log.info("  %s: %d candles baixados", symbol, len(candles))
    return candles


def fetch_funding_rates(symbol: str, since_ts: int | None = None) -> list[tuple[int, float]]:
    """
    Baixa histórico de funding rates da OKX (max ~3 meses disponíveis).
    Retorna lista de (ts_ms, rate).
    """
    # OKX swap instrument = BTC-USDT-SWAP
    swap = symbol.replace("-USDT", "-USDT-SWAP")
    url  = f"{OKX_BASE}/api/v5/public/funding-rate-history?instId={swap}&limit=100"
    rows: list[tuple[int, float]] = []
    after_ms = None
    page = 0

    log.info("Baixando funding rates para %s…", symbol)
    while True:
        req_url = url + (f"&after={after_ms}" if after_ms else "")
        try:
            resp = requests.get(req_url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as exc:
            log.warning("Funding rate fetch error: %s", exc)
            break

        if not data:
            break

        for d in data:
            ts   = int(d.get("fundingTime", 0))
            rate = float(d.get("realizedRate", d.get("fundingRate", 0)) or 0)
            if since_ts and ts <= since_ts:
                rows.sort(key=lambda x: x[0])
                log.info("  %s: %d funding rates baixados", symbol, len(rows))
                return rows
            rows.append((ts, rate))

        after_ms = int(data[-1].get("fundingTime", 0)) - 1
        page += 1
        time.sleep(0.3)
        if page > 50:  # segurança
            break

    rows.sort(key=lambda x: x[0])
    log.info("  %s: %d funding rates baixados", symbol, len(rows))
    return rows


# ── Signal logic (espelho de momentum_strategy.py) ────────────────────────────

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

    atr_5 = float(np.mean([
        abs(closes[i] - closes[i + 1]) for i in range(min(5, len(closes) - 1))
    ]))
    rel_atr = atr_5 / closes[0] if closes[0] > 0 else 0

    if rel_atr < 0.005:
        return "LIQUIDITY_VACUUM"
    if rel_atr > 0.025:
        return "HIGH_CORRELATION_RISK"

    return "MEAN_REVERTING_CHOP"


def _atr_recal(highs: list[float], lows: list[float], closes: list[float],
               period: int = 14) -> float:
    n = min(period, len(highs) - 1)
    if n <= 0:
        return (highs[0] - lows[0]) if highs else 0.0
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i+1]),
               abs(lows[i] - closes[i+1])) for i in range(n)]
    return sum(trs) / len(trs) if trs else 0.0


def _bollinger_recal(closes: list[float], period: int = 20) -> tuple[float, float, float]:
    import math as _math
    n = min(period, len(closes))
    if n < 2:
        c = closes[0]
        return c, c, c
    window = closes[:n]
    mid = sum(window) / n
    std = _math.sqrt(sum((x - mid) ** 2 for x in window) / n)
    return mid + 2 * std, mid, mid - 2 * std


def _compute_m8_recal(closes: list[float], highs: list[float], lows: list[float]) -> float:
    """M8 Volatility State — espelho de volatility_state.compute_vol_state()."""
    if len(closes) < 22:
        return 0.5
    price    = closes[0]
    atr_now  = _atr_recal(highs, lows, closes, 14)
    atr_prev = _atr_recal(highs[7:], lows[7:], closes[7:], 14)
    bb_upper, _, bb_lower = _bollinger_recal(closes, 20)
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


def _compute_m7_recal(closes_sym: list[float], closes_btc: list[float],
                      alt_closes_list: list[list[float]]) -> float:
    """
    M7 Relative Strength — espelho de RelativeStrengthCollector.
    Horizontes: 1h (40%), 5h (35%), 24h (25%).
    """
    import math as _math2

    def _ret(closes: list[float], h: int) -> float | None:
        if len(closes) <= h or closes[h] <= 0:
            return None
        return (closes[0] - closes[h]) / closes[h]

    def _score_rs(rs: float | None) -> float:
        if rs is None:
            return 0.5
        x = max(-50.0, min(50.0, (rs - 1.0) * 8.0))  # clamp para evitar overflow
        return round(1.0 / (1.0 + _math2.exp(-x)), 4)

    # RS ponderado multi-horizonte
    btc_rets = {h: _ret(closes_btc, h) for h in [1, 5, 24]}
    sym_rets = {h: _ret(closes_sym, h) for h in [1, 5, 24]}
    weights  = {1: 0.40, 5: 0.35, 24: 0.25}

    rs_scores = []
    for h, w in weights.items():
        br, sr = btc_rets[h], sym_rets[h]
        if br is None or sr is None:
            continue
        rs = sr / br if abs(br) > 1e-8 else 1.0
        rs_scores.append(_score_rs(rs) * w)
    rs_weighted = sum(rs_scores) / sum(weights[h] for h in [1,5,24]
                      if btc_rets[h] is not None and sym_rets[h] is not None) \
                  if rs_scores else 0.5

    # BTC leadership
    btc_1h = _ret(closes_btc, 1) or 0.0
    btc_5h = _ret(closes_btc, 5) or 0.0
    lead   = min(max(0.5 + btc_1h * 20 + btc_5h * 5, 0.0), 1.0)
    if alt_closes_list:
        alt_rets = [_ret(a, 1) for a in alt_closes_list if _ret(a, 1) is not None]
        if alt_rets:
            avg_alt = sum(alt_rets) / len(alt_rets)
            if btc_1h > 0 and avg_alt > 0:
                lead = min(lead + 0.1, 1.0)
            elif btc_1h < 0:
                lead = max(lead - 0.1, 0.0)

    # RS trend (usando 1h como proxy — apenas 1 ponto disponível por candle)
    rs_trend = 0.5  # neutro sem histórico de RS

    m7 = rs_weighted * 0.50 + lead * 0.30 + rs_trend * 0.20
    return round(min(max(m7, 0.0), 1.0), 4)


def _compute_m6_recal(ts_ms: int, funding_lookup: dict[int, float]) -> float:
    """
    M6 Futures Flow simplificado para backtest:
    Usa apenas funding rate (OI não disponível historicamente).
    Funding rate → score: negativo=bearish(0.2), neutro=0.5, positivo=bullish(0.8).
    """
    if not funding_lookup:
        return 0.5
    # Busca o funding rate mais próximo antes de ts_ms (janela de 8h)
    EIGHT_HOURS = 8 * 3_600_000
    best_ts, best_rate = None, None
    for fts, rate in funding_lookup.items():
        if fts <= ts_ms and (best_ts is None or fts > best_ts):
            best_ts, best_rate = fts, rate

    if best_ts is None or (ts_ms - best_ts) > EIGHT_HOURS * 1.5:
        return 0.5  # sem dado próximo

    rate = best_rate
    # Normaliza: funding positivo = longs pagam = bullish momentum
    # Faixa típica: [-0.001, +0.003] por 8h
    if rate > 0.001:
        score = 0.70   # funding alto = muito bullish
    elif rate > 0.0003:
        score = 0.60
    elif rate > -0.0003:
        score = 0.50   # neutro
    elif rate > -0.001:
        score = 0.40
    else:
        score = 0.30   # funding muito negativo = bearish

    return score


def score_signal(closes: list[float], highs: list[float],
                 lows: list[float], opens: list[float],
                 volumes: list[float], regime: str,
                 m6: float = 0.5, m7: float = 0.5) -> float:
    """
    Score M1-M8 espelho de MomentumStrategy._score_signal().
    Pesos v2.3.0: M1=2% M2=20% M3=30% M4=0% M5=8% M6=10% M7=9% M8=21%
    M6 e M7 são injetados externamente por build_samples.
    """
    n = len(closes)

    # M1 Adaptive Momentum
    atr  = sum(highs[i]-lows[i] for i in range(min(10,len(highs))))/min(10,len(highs))
    norm = max(atr*2, closes[0]*0.005)
    r1   = (closes[0]-closes[1])/closes[1]   if n>1  and closes[1]>0  else 0
    r5   = (closes[0]-closes[5])/closes[5]   if n>5  and closes[5]>0  else 0
    r10  = (closes[0]-closes[10])/closes[10] if n>10 and closes[10]>0 else 0
    r20  = (closes[0]-closes[20])/closes[20] if n>20 and closes[20]>0 else 0
    mw   = r1*0.30 + r5*0.30 + r10*0.25 + r20*0.15
    m1   = min(max((mw/(norm/closes[0]))*0.5+0.5, 0.0), 1.0)

    # M2 Trend Consistency
    nb   = min(6, n-1)
    bull = sum(1 for i in range(nb) if closes[i]>opens[i])/nb if nb>0 else 0.5
    hh   = sum(1 for i in range(min(4,len(highs)-1)) if highs[i]>highs[i+1])/4
    hl   = sum(1 for i in range(min(4,len(lows)-1))  if lows[i]>lows[i+1])/4
    m2   = bull*0.5 + (hh+hl)/2*0.5

    # M3 Volume Confirmation
    avg5  = sum(volumes[:5])/5   if len(volumes)>=5  else volumes[0] if volumes else 1
    avg20 = sum(volumes[:20])/20 if len(volumes)>=20 else avg5
    vr    = min(volumes[0]/avg5, 3.0)/3.0 if avg5>0 else 0.5
    vt    = min(max(sum(volumes[:3])/sum(volumes[3:6]),0.3),2.0) if len(volumes)>=6 and sum(volumes[3:6])>0 else 1.0
    cc    = 1.0 if closes[0]>opens[0] and volumes[0]>avg20 else 0.4
    m3    = vr*0.4 + (vt-0.3)/1.7*0.3 + cc*0.3

    # M4 Regime Strength
    sma5  = sum(closes[:5])/5
    sma20 = sum(closes[:20])/20 if n>=20 else sma5
    dist  = (sma5-sma20)/sma20 if sma20>0 else 0
    m4r   = min(max((dist+0.02)/0.04, 0.0), 1.0)
    m4f   = 0.85 if sma5>sma20 else 0.45
    m4    = m4r*0.6 + m4f*0.4

    # M5 Candle Structure
    cs = [(closes[i]-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0.5
          for i in range(min(3, n))]
    m5 = sum(cs)/len(cs) if cs else 0.5

    # M6 e M7 injetados externamente (calculados em build_samples)
    # M8 Volatility State — calculado de candles
    m8 = _compute_m8_recal(closes, highs, lows)

    # Pesos v2.3.0 — espelho exato de momentum_strategy.py linha 445
    # M4=0 (removido), soma = 2+20+30+0+8+10+9+21 = 100%
    return (m1*0.02 + m2*0.20 + m3*0.30 +
            m5*0.08 + m6*0.10 + m7*0.09 + m8*0.21)


def platt_calibrate(score: float, A: float, B: float) -> float:
    return 1.0 / (1.0 + math.exp(-(A * score + B)))


# ── Sample builder ────────────────────────────────────────────────────────────

def build_samples(candles: list[dict], symbol: str, gran: str,
                  forward: int, fee: float, min_score: float,
                  start_idx: int = LOOKBACK,
                  btc_candles: list[dict] | None = None,
                  alt_candles_map: dict[str, list[dict]] | None = None,
                  funding_lookup: dict[int, float] | None = None) -> list[dict]:
    """
    Gera amostras para candles[start_idx : len-forward].
    btc_candles: candles do BTC para calcular M7 (Relative Strength).
    alt_candles_map: {symbol: candles} dos outros ativos para BTC leadership.
    funding_lookup: {ts_ms: rate} para M6 (Funding Rate).
    """
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    opens   = [c["open"]   for c in candles]
    volumes = [c["volume"] for c in candles]
    n = len(candles)
    samples: list[dict] = []

    # Pré-computa lookup de BTC por ts para M7
    btc_ts_map: dict[int, int] = {}  # ts -> índice em btc_candles
    btc_closes: list[float] = []
    if btc_candles:
        btc_closes = [c["close"] for c in btc_candles]
        btc_ts_map = {c["ts"]: i for i, c in enumerate(btc_candles)}

    # Pré-computa lookup de alts para BTC leadership
    alt_closes_by_ts: dict[int, list[list[float]]] = {}
    if alt_candles_map:
        # Para cada timestamp, coleta os closes das alts
        all_ts = set(c["ts"] for c in candles)
        for alt_sym, alt_cands in alt_candles_map.items():
            alt_map = {c["ts"]: i for i, c in enumerate(alt_cands)}
            alt_cls = [c["close"] for c in alt_cands]
            for ts in all_ts:
                if ts in alt_map:
                    idx = alt_map[ts]
                    window = list(reversed(alt_cls[max(0, idx - LOOKBACK):idx + 1]))
                    alt_closes_by_ts.setdefault(ts, []).append(window)

    for i in range(start_idx, n - forward):
        w_close = list(reversed(closes[max(0, i - LOOKBACK):i + 1]))
        w_high  = list(reversed(highs[max(0, i - LOOKBACK):i + 1]))
        w_low   = list(reversed(lows[max(0, i - LOOKBACK):i + 1]))
        w_open  = list(reversed(opens[max(0, i - LOOKBACK):i + 1]))
        w_vol   = list(reversed(volumes[max(0, i - LOOKBACK):i + 1]))

        regime = detect_regime(w_close, w_vol)
        if regime in {"PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"}:
            continue

        ts_now = candles[i]["ts"]

        # M7: Relative Strength vs BTC
        m7 = 0.5
        if btc_candles and ts_now in btc_ts_map:
            btc_i = btc_ts_map[ts_now]
            btc_w = list(reversed(btc_closes[max(0, btc_i - LOOKBACK):btc_i + 1]))
            alts  = alt_closes_by_ts.get(ts_now, [])
            # Exclui o próprio símbolo das alts
            alts_filtered = alts  # já filtrado pois alt_candles_map não inclui symbol
            m7 = _compute_m7_recal(w_close, btc_w, alts_filtered)

        # M6: Funding Rate
        m6 = _compute_m6_recal(ts_now, funding_lookup or {})

        score = score_signal(w_close, w_high, w_low, w_open, w_vol, regime, m6=m6, m7=m7)
        if score < min_score:
            continue

        entry_price = candles[i]["close"]
        exit_price  = candles[i + forward]["close"]
        net_return  = (exit_price - entry_price) / entry_price - fee
        label       = 1 if net_return > 0 else 0

        samples.append({
            "symbol": symbol,
            "gran":   gran,
            "ts":     ts_now,
            "score":  score,
            "regime": regime,
            "label":  label,
        })

    return samples


# ── Platt fitting ─────────────────────────────────────────────────────────────

def fit_platt(scores: np.ndarray, labels: np.ndarray,
              lr: float = 0.1, epochs: int = 2000) -> tuple[float, float]:
    A = PLATT_A_PRIOR
    B = PLATT_B_PRIOR

    for epoch in range(epochs):
        p     = 1.0 / (1.0 + np.exp(-(A * scores + B)))
        p     = np.clip(p, 1e-7, 1 - 1e-7)
        error = p - labels
        A    -= lr * float(np.mean(error * scores))
        B    -= lr * float(np.mean(error))
        if (epoch + 1) % 500 == 0:
            loss = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
            log.debug("  epoch=%d loss=%.4f A=%.4f B=%.4f", epoch + 1, loss, A, B)

    return float(A), float(B)


def analyze_regime_thresholds(samples: list[dict],
                               A: float, B: float) -> dict[str, float]:
    by_regime: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for s in samples:
        cal = platt_calibrate(s["score"], A, B)
        by_regime[s["regime"]].append((cal, s["label"]))

    new_thresholds: dict[str, float] = {}
    log.info("\n── Análise por regime ──────────────────────────────────────")

    for regime, pairs in sorted(by_regime.items()):
        if len(pairs) < 10:
            new_thresholds[regime] = REGIME_THRESHOLDS_DEFAULT.get(regime, 0.65)
            continue

        pairs.sort(key=lambda x: x[0])
        cals   = np.array([p[0] for p in pairs])
        labels = np.array([p[1] for p in pairs])

        best_thresh = REGIME_THRESHOLDS_DEFAULT.get(regime, 0.65)
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
                best_f1     = f1
                best_thresh = float(thresh)

        old_thresh = REGIME_THRESHOLDS_DEFAULT.get(regime, 0.65)
        log.info(
            "  %-28s  n=%5d  win_rate=%.1f%%  threshold: %.2f→%.2f  (Δ%+.2f)",
            regime, len(pairs), labels.mean() * 100,
            old_thresh, best_thresh, best_thresh - old_thresh,
        )
        new_thresholds[regime] = round(best_thresh, 2)

    return new_thresholds


# ── Main ──────────────────────────────────────────────────────────────────────

def _log_experiment(result: dict, mode: str = "incremental") -> None:
    """
    Phase 15.1 — Experiment Tracking.
    Appends calibration run to experiment_log.json para rastreabilidade histórica.
    Compara automaticamente com o run anterior (gate de qualidade).
    """
    exp_path = ROOT / "data" / "models" / "experiment_log.json"
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    # Lê log existente
    try:
        log_data = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else {}
    except Exception:
        log_data = {}

    experiments = log_data.get("experiments", [])
    prev = experiments[-1] if experiments else {}

    # Versão incremental
    version = f"platt_v{len(experiments) + 1}"

    # Comparação vs anterior
    delta_wr = None
    delta_a  = None
    if prev:
        delta_wr = round(result["win_rate"] - prev.get("win_rate", result["win_rate"]), 4)
        delta_a  = round(result["platt_a"]   - prev.get("platt_a",  result["platt_a"]),  6)

    entry = {
        "version":      version,
        "run_at":       result["calibrated_at"],
        "mode":         mode,
        "platt_a":      result["platt_a"],
        "platt_b":      result["platt_b"],
        "n_samples":    result["n"],
        "win_rate":     result["win_rate"],
        "symbols":      result["symbols"],
        "delta_wr":     delta_wr,
        "delta_a":      delta_a,
        "forward":      result["forward_candles"],
        "quality_gate": {
            "wr_vs_prev":  "+" + str(delta_wr) if delta_wr and delta_wr > 0 else str(delta_wr),
            "passed":      (delta_wr is None or delta_wr >= -0.02),  # aceita queda de até -2%
        },
    }
    experiments.append(entry)

    log_data = {"experiments": experiments, "updated_at": result["calibrated_at"]}
    exp_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("\n── Phase 15.1: Experiment Tracking ──────────────────────────")
    log.info("  Versão   : %s", version)
    log.info("  WR atual : %.1f%%  (delta %+.1f%%)",
             result["win_rate"] * 100,
             (delta_wr or 0) * 100)
    log.info("  A atual  : %.6f  (delta %+.6f)", result["platt_a"], delta_a or 0)
    log.info("  Gate     : %s", "✅ PASSED" if entry["quality_gate"]["passed"] else "❌ FAILED")
    log.info("  Log      : %s (%d runs)", exp_path, len(experiments))
    log.info("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra o sistema de sinais CCTBv5")
    parser.add_argument("--symbols",     nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start",       default=DEFAULT_START, help="YYYY-MM-DD (só no rebuild)")
    parser.add_argument("--gran",        default=DEFAULT_GRANULARITY, choices=["1H","4H","6H","1D"])
    parser.add_argument("--forward",     type=int,   default=DEFAULT_FORWARD)
    parser.add_argument("--fee",         type=float, default=DEFAULT_FEE)
    parser.add_argument("--min-score",   type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--incremental", action="store_true",
                        help="Busca só candles novos e adiciona ao banco (padrão)")
    parser.add_argument("--rebuild",     action="store_true",
                        help="Limpa o banco e reconstrói do zero via OKX")
    parser.add_argument("--from-cache",  action="store_true",
                        help="Bootstrap rápido: importa cache JSON local para o banco")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Não grava nada")
    args = parser.parse_args()

    # Padrão: incremental
    if not args.rebuild and not args.from_cache:
        args.incremental = True

    end_dt = datetime.now(UTC)

    log.info("═" * 60)
    log.info("  CCTBv5 — Recalibração %s", "INCREMENTAL" if args.incremental else "REBUILD")
    log.info("  Símbolos : %s", ", ".join(args.symbols))
    log.info("  Janela   : %d candles %s à frente | fee %.2f%%",
             args.forward, args.gran, args.fee * 100)
    log.info("  Banco    : %s", DB_PATH)
    log.info("═" * 60)

    conn = open_db()

    if args.rebuild:
        db_clear(conn)

    # ── Bootstrap a partir dos caches JSON locais ─────────────────────────────
    if args.from_cache:
        log.info("Bootstrap a partir dos caches JSON locais…")
        for symbol in args.symbols:
            cache_path = CACHE_DIR / f"{symbol.replace('-','_')}_{args.gran}.json"
            if not cache_path.exists():
                log.warning("Cache não encontrado: %s", cache_path)
                continue
            candles = json.loads(cache_path.read_text())
            log.info("%s: %d candles no cache", symbol, len(candles))
            if not args.dry_run:
                inserted = db_insert_candles(conn, symbol, args.gran, candles)
                log.info("%s: %d candles inseridos no banco.", symbol, inserted)
        log.info("Bootstrap concluído — continuando para gerar amostras…\n")

    # ── 1. Buscar e armazenar candles novos via OKX ───────────────────────────
    if args.from_cache:
        log.info("Modo --from-cache: skip fetch OKX — usando dados do banco.\n")
    for symbol in args.symbols if not args.from_cache else []:
        last_ts = db_last_ts(conn, symbol, args.gran)

        if last_ts:
            # Incremental: busca a partir do próximo candle
            gran_ms  = GRAN_MS.get(args.gran, 3_600_000)
            start_dt = datetime.fromtimestamp((last_ts + gran_ms) / 1000, tz=UTC)
            hours_behind = (end_dt.timestamp() * 1000 - last_ts) / gran_ms
            log.info("%s: banco tem dados até %s (%.0f candles atrás)",
                     symbol,
                     datetime.fromtimestamp(last_ts / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M"),
                     hours_behind)
        else:
            # Primeiro run: bootstrap completo
            start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
            log.info("%s: banco vazio — bootstrap desde %s", symbol, args.start)

        if start_dt >= end_dt - timedelta(hours=1):
            log.info("%s: já atualizado — nada a buscar.", symbol)
            continue

        new_candles = fetch_candles_range(symbol, args.gran, start_dt, end_dt)
        if not new_candles:
            log.warning("%s: sem candles novos retornados pela OKX.", symbol)
            continue

        if not args.dry_run:
            inserted = db_insert_candles(conn, symbol, args.gran, new_candles)
            log.info("%s: %d candles novos inseridos no banco.", symbol, inserted)

    # ── 2. Buscar funding rates (M6) — OKX histórico ─────────────────────────
    funding_lookups: dict[str, dict[int, float]] = {}
    if not args.from_cache:
        for symbol in args.symbols:
            last_fr_ts = db_last_funding_ts(conn, symbol)
            fr_rows = fetch_funding_rates(symbol, since_ts=last_fr_ts)
            if fr_rows and not args.dry_run:
                db_insert_funding_rates(conn, symbol, fr_rows)
            funding_lookups[symbol] = db_load_funding_lookup(conn, symbol)
            log.info("%s: %d funding rates no banco", symbol, len(funding_lookups[symbol]))
    else:
        for symbol in args.symbols:
            funding_lookups[symbol] = db_load_funding_lookup(conn, symbol)

    # ── 3. Pré-carregar candles de todos os símbolos (M7 cross-symbol) ────────
    all_candles_map: dict[str, list[dict]] = {}
    for symbol in args.symbols:
        all_candles_map[symbol] = db_load_candles(conn, symbol, args.gran)

    btc_candles = all_candles_map.get("BTC-USDT", [])

    # ── 4. Gerar amostras incrementais ────────────────────────────────────────
    for symbol in args.symbols:
        all_candles = all_candles_map[symbol]
        if len(all_candles) < LOOKBACK + args.forward + 1:
            log.warning("%s: candles insuficientes no banco (%d).", symbol, len(all_candles))
            continue

        # Descobre o último ts já processado como amostra
        row = conn.execute(
            "SELECT MAX(ts) FROM samples WHERE symbol=? AND gran=? AND forward=? AND fee=?",
            (symbol, args.gran, args.forward, args.fee),
        ).fetchone()
        last_sample_ts = row[0] if row and row[0] else None

        if last_sample_ts:
            ts_list = [c["ts"] for c in all_candles]
            try:
                last_idx = ts_list.index(last_sample_ts)
                start_idx = max(last_idx + 1, LOOKBACK)
            except ValueError:
                start_idx = LOOKBACK
            log.info("%s: gerando amostras a partir do índice %d (de %d candles)",
                     symbol, start_idx, len(all_candles))
        else:
            start_idx = LOOKBACK
            log.info("%s: gerando amostras do zero (%d candles).", symbol, len(all_candles))

        # Alts para BTC leadership (todos os outros símbolos)
        alt_map = {s: c for s, c in all_candles_map.items() if s != symbol}

        new_samples = build_samples(
            all_candles, symbol, args.gran,
            args.forward, args.fee, args.min_score,
            start_idx=start_idx,
            btc_candles=btc_candles if symbol != "BTC-USDT" else None,
            alt_candles_map=alt_map,
            funding_lookup=funding_lookups.get(symbol, {}),
        )

        if new_samples:
            wins = sum(s["label"] for s in new_samples)
            log.info("%s: %d novas amostras (win=%d loss=%d)",
                     symbol, len(new_samples), wins, len(new_samples) - wins)
            if not args.dry_run:
                db_insert_samples(conn, new_samples, args.forward, args.fee)
        else:
            log.info("%s: sem novas amostras.", symbol)

    # ── 5. Carregar todas as amostras e refitar ────────────────────────────────
    all_samples = db_load_samples(conn, args.forward, args.fee)
    if not all_samples:
        log.error("Nenhuma amostra no banco. Rode com --rebuild para bootstrap.")
        return

    scores_arr = np.array([s["score"] for s in all_samples], dtype=np.float64)
    labels_arr = np.array([s["label"] for s in all_samples], dtype=np.float64)

    wins     = int(labels_arr.sum())
    win_rate = float(labels_arr.mean())

    log.info("\nAjustando Platt scaling com %d amostras (win_rate=%.1f%%)…",
             len(all_samples), win_rate * 100)

    A_new, B_new = fit_platt(scores_arr, labels_arr)
    new_thresholds = analyze_regime_thresholds(all_samples, A_new, B_new)

    # ── 4. Resumo ─────────────────────────────────────────────────────────────
    log.info("\n══ RESUMO ══════════════════════════════════════════════════")
    log.info("  Amostras totais : %d", len(all_samples))
    log.info("  Win rate real   : %.1f%%", win_rate * 100)
    log.info("  A (prior→novo)  : %.4f → %.4f", PLATT_A_PRIOR, A_new)
    log.info("  B (prior→novo)  : %.4f → %.4f", PLATT_B_PRIOR, B_new)
    log.info("\n  Calibração por score:")
    for s in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        p_old = platt_calibrate(s, PLATT_A_PRIOR, PLATT_B_PRIOR)
        p_new = platt_calibrate(s, A_new, B_new)
        log.info("    score=%.2f  prob_prior=%.3f  prob_novo=%.3f  Δ=%+.3f",
                 s, p_old, p_new, p_new - p_old)

    log.info("\n  Novos thresholds por regime:")
    for regime, thresh in sorted(new_thresholds.items()):
        log.info("    %-28s  %.2f", regime, thresh)

    # ── 5. Gravar ─────────────────────────────────────────────────────────────
    result = {
        "platt_a":           round(A_new, 6),
        "platt_b":           round(B_new, 6),
        "n":                 len(all_samples),
        "win_rate":          round(win_rate, 4),
        "calibrated_at":     datetime.now(UTC).isoformat(),
        "period_end":        end_dt.strftime("%Y-%m-%d"),
        "symbols":           args.symbols,
        "granularity":       args.gran,
        "forward_candles":   args.forward,
        "regime_thresholds": new_thresholds,
        "previous": {
            "platt_a": PLATT_A_PRIOR,
            "platt_b": PLATT_B_PRIOR,
        },
    }

    if args.dry_run:
        log.info("\n[DRY RUN] Resultado (não gravado):\n%s", json.dumps(result, indent=2))
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2))
        log.info("\n✅ Gravado em: %s", OUTPUT)
        log.info("✅ Banco     : %s (%.1f MB)",
                 DB_PATH, DB_PATH.stat().st_size / 1024 / 1024)

        # ── Phase 15.1: Experiment Tracking ───────────────────────────────────
        _log_experiment(result, mode="rebuild" if args.rebuild else "incremental")

        log.info("\nPróximo passo:")
        log.info("  git add data/models/calibration_coef.json data/models/experiment_log.json")
        log.info("  git commit -m 'chore: recalibração %s'",
                 datetime.now(UTC).strftime("%Y-%m-%d"))
        log.info("  git push && ./deploy.ps1")

    conn.close()


if __name__ == "__main__":
    main()
