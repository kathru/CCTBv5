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
DEFAULT_GRANULARITY = "30m"
DEFAULT_FORWARD     = 10   # 10 × 30min = 5h (mesma janela de avaliação)
DEFAULT_FEE         = 0.005
DEFAULT_MIN_SCORE   = 0.40

LOOKBACK = 20   # candles de janela para score_signal

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


# ── OKX Fetcher ───────────────────────────────────────────────────────────────

OKX_BASE = "https://www.okx.com"
GRAN_MS   = {"30m": 1_800_000, "1H": 3_600_000, "4H": 14_400_000, "6H": 21_600_000, "1D": 86_400_000}


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


def score_signal(closes: list[float], highs: list[float],
                 volumes: list[float], regime: str) -> float:
    momentum = (closes[0] - closes[-1]) / closes[-1] if closes[-1] > 0 else 0
    m1 = min(max((momentum + 0.05) / 0.10, 0.0), 1.0)
    m2 = 1.0 if (len(highs) >= 3 and highs[0] > highs[1] > highs[2]) else 0.4
    avg_v = float(np.mean(volumes[:5])) if volumes else 1.0
    m3 = min(volumes[0] / avg_v, 2.0) / 2.0 if avg_v > 0 else 0.5
    m4 = 0.8 if regime == "TREND_EXPANSION" else 0.5
    return m1 * 0.3 + m2 * 0.3 + m3 * 0.2 + m4 * 0.2


def platt_calibrate(score: float, A: float, B: float) -> float:
    return 1.0 / (1.0 + math.exp(-(A * score + B)))


# ── Sample builder ────────────────────────────────────────────────────────────

def build_samples(candles: list[dict], symbol: str, gran: str,
                  forward: int, fee: float, min_score: float,
                  start_idx: int = LOOKBACK) -> list[dict]:
    """
    Gera amostras para candles[start_idx : len-forward].
    start_idx permite processar apenas candles novos passando o offset certo.
    """
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    volumes = [c["volume"] for c in candles]
    n = len(candles)
    samples: list[dict] = []

    for i in range(start_idx, n - forward):
        w_close = list(reversed(closes[max(0, i - LOOKBACK):i + 1]))
        w_high  = list(reversed(highs[max(0, i - LOOKBACK):i + 1]))
        w_vol   = list(reversed(volumes[max(0, i - LOOKBACK):i + 1]))

        regime = detect_regime(w_close, w_vol)
        if regime in {"PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"}:
            continue

        score = score_signal(w_close, w_high, w_vol, regime)
        if score < min_score:
            continue

        entry_price = candles[i]["close"]
        exit_price  = candles[i + forward]["close"]
        net_return  = (exit_price - entry_price) / entry_price - fee
        label       = 1 if net_return > 0 else 0

        samples.append({
            "symbol": symbol,
            "gran":   gran,
            "ts":     candles[i]["ts"],
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra o sistema de sinais CCTBv5")
    parser.add_argument("--symbols",     nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start",       default=DEFAULT_START, help="YYYY-MM-DD (só no rebuild)")
    parser.add_argument("--gran",        default=DEFAULT_GRANULARITY, choices=["30m","1H","4H","6H","1D"])
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

    # ── 2. Gerar amostras incrementais ────────────────────────────────────────
    for symbol in args.symbols:
        # Carrega todos os candles do banco (necessário para a janela de lookback)
        all_candles = db_load_candles(conn, symbol, args.gran)
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
            # Encontra o índice do primeiro candle após o último processado
            # Precisa de LOOKBACK candles anteriores para a janela
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

        new_samples = build_samples(
            all_candles, symbol, args.gran,
            args.forward, args.fee, args.min_score,
            start_idx=start_idx,
        )

        if new_samples:
            wins = sum(s["label"] for s in new_samples)
            log.info("%s: %d novas amostras (win=%d loss=%d)",
                     symbol, len(new_samples), wins, len(new_samples) - wins)
            if not args.dry_run:
                db_insert_samples(conn, new_samples, args.forward, args.fee)
        else:
            log.info("%s: sem novas amostras.", symbol)

    # ── 3. Carregar todas as amostras e refitar ────────────────────────────────
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
        log.info("\nPróximo passo:")
        log.info("  git add data/models/calibration_coef.json")
        log.info("  git commit -m 'chore: recalibração %s'",
                 datetime.now(UTC).strftime("%Y-%m-%d"))
        log.info("  git push && ./deploy.ps1")

    conn.close()


if __name__ == "__main__":
    main()
