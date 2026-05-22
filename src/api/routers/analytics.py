"""
Quantitative Analytics — métricas institucionais calculadas dos fills reais.

Phase 10 — Futures Flow:
  - /api/analytics/futures_flow  : funding rate, OI, M6 score por símbolo

Phase 6 — Equity Analytics:
  - Equity Curve       (série temporal por trade, high-water mark, underwater)
  - Rolling Metrics    (Sharpe/WR/Expectancy janela deslizante 20 trades)
  - Drawdown Analytics (max DD persistido, duração, recovery factor)
  - Regime Breakdown   (performance por regime)
  - Stability (R²)     (quão linear é a equity curve)
  - Export CSV         (equity curve para análise externa)
"""

import json
import math
import random as _random
import statistics
from collections import defaultdict
from datetime import UTC, datetime

from fastapi import APIRouter, Query, Request

from ...monitoring.signal_log import signal_audit_log

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

RISK_FREE_RATE = 0.05  # 5% ao ano (taxa livre de risco)
MIN_TRADES     = 3     # mínimo para mostrar métrica


# ── Helpers matemáticos ───────────────────────────────────────────────────────

def _sharpe(returns: list[float]) -> float | None:
    if len(returns) < MIN_TRADES:
        return None
    mean = statistics.mean(returns)
    std  = statistics.stdev(returns) if len(returns) > 1 else 0
    if std == 0:
        return None
    daily_rf = RISK_FREE_RATE / 252
    return round((mean - daily_rf) / std * math.sqrt(252), 3)


def _sortino(returns: list[float]) -> float | None:
    if len(returns) < MIN_TRADES:
        return None
    mean     = statistics.mean(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return None
    ds_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
    if ds_std == 0:
        return None
    daily_rf = RISK_FREE_RATE / 252
    return round((mean - daily_rf) / ds_std * math.sqrt(252), 3)


def _calmar(total_return_pct: float, max_dd_pct: float, days: int) -> float | None:
    if max_dd_pct <= 0 or days < 1:
        return None
    annual_factor  = 365 / days
    annual_return  = total_return_pct * annual_factor
    return round(annual_return / max_dd_pct, 3)


def _stability(values: list[float]) -> float | None:
    """R² da regressão linear sobre a equity curve (1.0 = linha perfeita)."""
    if len(values) < MIN_TRADES:
        return None
    n  = len(values)
    xs = list(range(n))
    xm = sum(xs) / n
    ym = sum(values) / n
    ss_tot = sum((y - ym) ** 2 for y in values)
    if ss_tot == 0:
        return 1.0
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return None
    slope     = sum((xs[i] - xm) * (values[i] - ym) for i in range(n)) / denom
    intercept = ym - slope * xm
    ss_res = sum((values[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    return round(max(0.0, 1.0 - ss_res / ss_tot), 4)


async def _get_filled_orders(db) -> list[dict]:
    """Retorna ordens filled como lista de dicts padronizados."""
    rows = await db.fetch(
        "SELECT * FROM orders WHERE status='filled' ORDER BY filled_at ASC NULLS LAST"
    )
    result = []
    for r in rows:
        result.append({
            "symbol":    r["symbol"],
            "side":      r["side"],
            "qty":       float(r["filled_quantity"] or 0),
            "price":     float(r["avg_fill_price"] or 0),
            "fee":       float(r["fees_paid"] or 0),
            "timestamp": r["filled_at"] or r["created_at"],
        })
    return result


def _pair_trades(orders: list[dict]) -> list[dict]:
    """
    Emparelha BUY→SELL por símbolo (FIFO) para obter trades completos.
    Aceita lista de dicts com keys: symbol, side, qty, price, fee, timestamp.
    """
    by_sym: dict[str, list] = defaultdict(list)
    for o in orders:
        by_sym[o["symbol"]].append(o)

    trades = []
    for sym, sym_orders in by_sym.items():
        sym_orders.sort(key=lambda o: o["timestamp"] or 0)
        buy_queue: list = []
        for o in sym_orders:
            side = str(o.get("side", "")).upper()
            if side in ("BUY", "LONG"):
                buy_queue.append(o)
            elif side in ("SELL", "SHORT") and buy_queue:
                entry = buy_queue.pop(0)
                entry_px = float(entry["price"])
                exit_px  = float(o["price"])
                qty      = float(min(entry["qty"], o["qty"]))
                fee      = float(entry.get("fee", 0)) + float(o.get("fee", 0))
                pnl      = (exit_px - entry_px) * qty - fee
                pnl_pct  = (exit_px - entry_px) / entry_px if entry_px > 0 else 0
                trades.append({
                    "symbol":   sym,
                    "pnl":      pnl,
                    "pnl_pct":  pnl_pct,
                    "entry_px": entry_px,
                    "exit_px":  exit_px,
                    "qty":      qty,
                    "fee":      fee,
                    "entry_ts": entry["timestamp"],
                    "exit_ts":  o["timestamp"],
                })
    return sorted(trades, key=lambda t: t["exit_ts"] or 0)


# ── Endpoint principal ────────────────────────────────────────────────────────

@router.get("/quantitative")
async def get_quantitative(request: Request) -> dict:
    db   = request.app.state.db

    # Usa ordens filled como fonte de verdade (fills table pode estar vazia)
    all_orders = await _get_filled_orders(db)
    trades     = _pair_trades(all_orders)
    n_trades  = len(trades)
    pnls      = [t["pnl"]     for t in trades]
    pnl_pcts  = [t["pnl_pct"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]

    # ── Hit Rate ────────────────────────────────────────────
    hit_rate = round(len(wins) / n_trades, 4) if n_trades > 0 else None

    # ── Expectancy ──────────────────────────────────────────
    avg_win  = sum(wins)   / len(wins)   if wins   else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    expectancy = None
    if n_trades >= MIN_TRADES:
        hr = len(wins) / n_trades
        expectancy = round(hr * avg_win + (1 - hr) * avg_loss, 4)

    # ── Profit Factor ───────────────────────────────────────
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None

    # ── Sharpe / Sortino ────────────────────────────────────
    sharpe  = _sharpe(pnl_pcts)
    sortino = _sortino(pnl_pcts)

    # ── Max Drawdown & Calmar ───────────────────────────────
    equity = [0.0]
    for p in pnls:
        equity.append(equity[-1] + p)

    peak       = equity[0]
    max_dd     = 0.0
    cur_dd_dur = 0
    max_dd_dur = 0

    for val in equity[1:]:
        if val > peak:
            peak       = val
            cur_dd_dur = 0
        else:
            dd = (peak - val) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
            cur_dd_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dd_dur)

    # Calmar: retorno total / max_dd (precisa de pelo menos 1 trade)
    total_return_pct = equity[-1] / 10000 if equity[-1] != 0 else 0  # assume capital 10k
    days_active = 1
    if trades:
        d0 = trades[0]["entry_ts"]
        d1 = trades[-1]["exit_ts"]
        if hasattr(d0, "timestamp"):
            days_active = max(1, int((d1.timestamp() - d0.timestamp()) / 86400))

    calmar = _calmar(total_return_pct, max_dd, days_active)

    # ── Drawdown duration (trades consecutivos em perda) ────
    current_dd_streak = 0
    for p in reversed(pnls):
        if p < 0:
            current_dd_streak += 1
        else:
            break

    # ── Stability (R² equity curve) ─────────────────────────
    stability = _stability(equity[1:]) if len(equity) > MIN_TRADES else None

    # ── Regime Breakdown ────────────────────────────────────
    entries = list(signal_audit_log._entries)
    regime_stats: dict[str, dict] = {}
    for e in entries:
        r = e.regime or "UNKNOWN"
        s = regime_stats.setdefault(r, {"total": 0, "signals": 0, "avg_score": 0, "scores": []})
        s["total"]   += 1
        s["scores"].append(e.score)
        if e.result == "SIGNAL":
            s["signals"] += 1
    for _r, s in regime_stats.items():
        s["avg_score"]  = round(sum(s["scores"]) / len(s["scores"]), 3) if s["scores"] else 0
        s["signal_rate"] = round(s["signals"] / s["total"], 3) if s["total"] else 0
        del s["scores"]

    # ── Resposta ────────────────────────────────────────────
    return {
        "n_trades":       n_trades,
        "n_wins":         len(wins),
        "n_losses":       len(losses),

        # Core metrics
        "hit_rate":       hit_rate,
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
        "avg_win":        round(avg_win,  2) if wins   else None,
        "avg_loss":       round(avg_loss, 2) if losses else None,
        "avg_win_pct":    round(sum(t["pnl_pct"] for t in trades if t["pnl"] > 0) / len(wins), 4) if wins else None,
        "avg_loss_pct":   round(sum(t["pnl_pct"] for t in trades if t["pnl"] <= 0) / len(losses), 4) if losses else None,

        # Risk-adjusted
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "stability_r2":   stability,

        # Drawdown
        "max_drawdown_pct":      round(max_dd, 4),
        "max_dd_duration_trades": max_dd_dur,
        "current_dd_streak":     current_dd_streak,

        # MAE/MFE — requer dados intra-trade (N/A sem tick data)
        "mae_available": False,
        "mfe_available": False,

        # Regime
        "regime_breakdown": regime_stats,

        # Meta
        "days_active":    days_active,
        "min_trades_req": MIN_TRADES,
        "computed_at":    datetime.now(UTC).isoformat(),
    }


# ── Phase 6.1 — Equity Curve ─────────────────────────────────────────────────

@router.get("/equity_curve")
async def get_equity_curve(request: Request) -> dict:
    """
    Phase 6.1 — Equity Curve completa.

    Retorna:
    - equity[]        : série de capital acumulado por trade (em $)
    - hwm[]           : high-water mark em cada ponto
    - underwater[]    : drawdown atual em % (0 = em máxima, -0.10 = -10%)
    - by_symbol{}     : contribuição de P&L por símbolo
    - trades[]        : lista completa de trades para exportação CSV
    - summary         : max_dd, recovery_factor, total_pnl, stability_r2
    """
    db      = request.app.state.db
    cache   = request.app.state.cache
    initial = 96_592.87  # capital inicial configurado

    all_orders = await _get_filled_orders(db)
    trades     = _pair_trades(all_orders)

    # ── Série temporal de equity ──────────────────────────────────────────────
    equity     = [initial]
    hwm        = [initial]
    underwater = [0.0]
    peak       = initial

    for t in trades:
        val = equity[-1] + t["pnl"]
        equity.append(round(val, 2))
        peak = max(peak, val)
        hwm.append(round(peak, 2))
        dd   = (val - peak) / peak if peak > 0 else 0.0
        underwater.append(round(dd, 6))

    # ── High-water mark persistido no Redis ───────────────────────────────────
    if cache and equity:
        stored_hwm = await cache.get("analytics:hwm")
        global_hwm = float(stored_hwm) if stored_hwm else initial
        current_hwm = max(global_hwm, max(equity))
        await cache.set("analytics:hwm", str(current_hwm), ttl=0)  # TTL 0 = persistente
    else:
        current_hwm = max(equity) if equity else initial

    # ── Breakdown por símbolo ─────────────────────────────────────────────────
    by_symbol: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        s   = by_symbol.setdefault(sym, {"pnl": 0.0, "trades": 0, "wins": 0})
        s["pnl"]    += t["pnl"]
        s["trades"] += 1
        if t["pnl"] > 0:
            s["wins"] += 1
    for _sym, s in by_symbol.items():
        s["pnl"]     = round(s["pnl"], 2)
        s["win_rate"] = round(s["wins"] / s["trades"], 3) if s["trades"] else 0

    # ── Métricas de drawdown ──────────────────────────────────────────────────
    max_dd_pct   = round(abs(min(underwater)), 4) if underwater else 0.0
    total_pnl    = equity[-1] - initial if len(equity) > 1 else 0.0
    total_ret_pct = total_pnl / initial if initial > 0 else 0.0
    recovery_factor = round(total_ret_pct / max_dd_pct, 3) if max_dd_pct > 0 else None

    # ── Stability R² ──────────────────────────────────────────────────────────
    stability = _stability(equity[1:]) if len(equity) > MIN_TRADES + 1 else None

    # ── Labels de tempo (índice de trade ou data) ─────────────────────────────
    labels = [f"T{i}" for i in range(len(equity))]
    if trades:
        labels[0] = "Início"
        for i, t in enumerate(trades, 1):
            ts = t.get("exit_ts")
            if hasattr(ts, "strftime"):
                labels[i] = ts.strftime("%d/%m %H:%M")

    return {
        "equity":          equity,
        "hwm":             hwm,
        "underwater":      underwater,
        "labels":          labels,
        "by_symbol":       by_symbol,
        "n_trades":        len(trades),
        "initial_capital": initial,
        "current_hwm":     round(current_hwm, 2),
        "total_pnl":       round(total_pnl, 2),
        "total_return_pct": round(total_ret_pct, 4),
        "max_drawdown_pct": max_dd_pct,
        "recovery_factor": recovery_factor,
        "stability_r2":    stability,
        "computed_at":     datetime.now(UTC).isoformat(),
    }


# ── Phase 6.2 — Rolling Metrics ──────────────────────────────────────────────

@router.get("/rolling")
async def get_rolling_metrics(
    request: Request,
    window: int = Query(default=20, ge=5, le=100, description="Janela deslizante de N trades"),
) -> dict:
    """
    Phase 6.2 — Rolling metrics com janela deslizante.

    Calcula para cada ponto com >= window trades:
    - rolling_sharpe     : Sharpe anualizado na janela
    - rolling_win_rate   : Win rate na janela
    - rolling_expectancy : Expectancy ($) na janela
    - rolling_vol        : Volatilidade de retornos na janela

    Permite detectar:
    - Degradação de edge (Sharpe caindo)
    - Regime shift (WR mudando)
    - Períodos de sobre/sub-performance
    """
    db     = request.app.state.db
    orders = await _get_filled_orders(db)
    trades = _pair_trades(orders)

    if len(trades) < window:
        return {
            "window":            window,
            "n_trades":          len(trades),
            "min_required":      window,
            "sufficient_data":   False,
            "rolling_sharpe":    [],
            "rolling_win_rate":  [],
            "rolling_expectancy": [],
            "rolling_vol":       [],
            "labels":            [],
        }

    rolling_sharpe:     list[float | None] = []
    rolling_wr:         list[float] = []
    rolling_expectancy: list[float] = []
    rolling_vol:        list[float] = []
    labels: list[str] = []

    for i in range(window - 1, len(trades)):
        w_trades = trades[i - window + 1 : i + 1]
        pnls     = [t["pnl"]     for t in w_trades]
        pcts     = [t["pnl_pct"] for t in w_trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]

        # Rolling Sharpe
        s = _sharpe(pcts)
        rolling_sharpe.append(s)

        # Rolling Win Rate
        rolling_wr.append(round(len(wins) / len(pnls), 3))

        # Rolling Expectancy
        avg_w = sum(wins)   / len(wins)   if wins   else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        hr    = len(wins) / len(pnls)
        rolling_expectancy.append(round(hr * avg_w + (1 - hr) * avg_l, 2))

        # Rolling Vol (desvio padrão dos retornos %)
        rolling_vol.append(
            round(statistics.stdev(pcts) * 100, 3) if len(pcts) > 1 else 0.0
        )

        # Label
        ts = w_trades[-1].get("exit_ts")
        labels.append(ts.strftime("%d/%m") if hasattr(ts, "strftime") else f"T{i}")

    # Resumo: tendência dos últimos 5 pontos (melhora ou piora?)
    def _trend(series: list) -> str:
        valid = [x for x in series[-5:] if x is not None]
        if len(valid) < 2:
            return "insufficient"
        return "improving" if valid[-1] > valid[0] else "degrading"

    return {
        "window":             window,
        "n_trades":           len(trades),
        "sufficient_data":    True,
        "rolling_sharpe":     [round(x, 3) if x is not None else None for x in rolling_sharpe],
        "rolling_win_rate":   rolling_wr,
        "rolling_expectancy": rolling_expectancy,
        "rolling_vol":        rolling_vol,
        "labels":             labels,
        "trends": {
            "sharpe":     _trend(rolling_sharpe),
            "win_rate":   _trend(rolling_wr),
            "expectancy": _trend(rolling_expectancy),
        },
        "latest": {
            "sharpe":     rolling_sharpe[-1] if rolling_sharpe else None,
            "win_rate":   rolling_wr[-1]     if rolling_wr     else None,
            "expectancy": rolling_expectancy[-1] if rolling_expectancy else None,
            "vol_pct":    rolling_vol[-1]    if rolling_vol    else None,
        },
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ── Phase 6.1 — Exportar CSV ─────────────────────────────────────────────────

@router.get("/equity_curve/csv")
async def export_equity_csv(request: Request):
    """Exporta equity curve como CSV para análise externa."""
    from fastapi.responses import PlainTextResponse
    db     = request.app.state.db
    orders = await _get_filled_orders(db)
    trades = _pair_trades(orders)
    initial = 96_592.87

    lines = ["trade,symbol,entry_ts,exit_ts,pnl,pnl_pct,equity,drawdown_pct"]
    equity = initial
    peak   = initial
    for i, t in enumerate(trades, 1):
        equity += t["pnl"]
        peak    = max(peak, equity)
        dd      = (equity - peak) / peak if peak > 0 else 0
        entry_s = t["entry_ts"].isoformat() if hasattr(t.get("entry_ts"), "isoformat") else ""
        exit_s  = t["exit_ts"].isoformat()  if hasattr(t.get("exit_ts"),  "isoformat") else ""
        lines.append(
            f"{i},{t['symbol']},{entry_s},{exit_s},"
            f"{t['pnl']:.2f},{t['pnl_pct']:.6f},{equity:.2f},{dd:.6f}"
        )
    return PlainTextResponse("\n".join(lines), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=equity_curve.csv"})


# ── Phase 7 — Distribution Analytics ─────────────────────────────────────────

def _histogram(values: list[float], n_bins: int = 10) -> dict:
    """Gera histograma simples: retorna bins, counts e labels."""
    if not values:
        return {"bins": [], "counts": [], "labels": []}
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return {"bins": [vmin], "counts": [len(values)], "labels": [f"{vmin:.3f}"]}
    width = (vmax - vmin) / n_bins
    bins   = [vmin + i * width for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - vmin) / width), n_bins - 1)
        counts[idx] += 1
    labels = [f"{bins[i]:.3f}" for i in range(n_bins)]
    return {"bins": [round(b, 6) for b in bins[:-1]], "counts": counts, "labels": labels}


def _holding_hours(entry_ts, exit_ts) -> float | None:
    """Duração em horas entre entry e exit."""
    try:
        if entry_ts is None or exit_ts is None:
            return None
        e = entry_ts.timestamp() if hasattr(entry_ts, "timestamp") else float(entry_ts)
        x = exit_ts.timestamp()  if hasattr(exit_ts,  "timestamp") else float(exit_ts)
        return round((x - e) / 3600, 2)
    except Exception:
        return None


def _streak_analysis(pnls: list[float]) -> dict:
    """Calcula sequências de wins/losses consecutivos."""
    if not pnls:
        return {"max_win_streak": 0, "max_loss_streak": 0,
                "current_streak": 0, "current_streak_type": "none"}
    max_win = max_loss = cur = 0
    cur_type = "win" if pnls[0] > 0 else "loss"

    for p in pnls:
        kind = "win" if p > 0 else "loss"
        if kind == cur_type:
            cur += 1
        else:
            cur = 1
            cur_type = kind
        if kind == "win":
            max_win  = max(max_win,  cur)
        else:
            max_loss = max(max_loss, cur)

    return {
        "max_win_streak":    max_win,
        "max_loss_streak":   max_loss,
        "current_streak":    cur,
        "current_streak_type": cur_type,
    }


def _regime_contribution(trades: list[dict]) -> dict:
    """
    Cruza trades com signal_audit_log pelo símbolo para obter regime no momento
    da entrada. Estratégia: busca o audit entry mais próximo (≤ 2h antes) do entry_ts.
    """
    entries = list(signal_audit_log._entries)
    # Indexa por símbolo para busca eficiente
    by_sym: dict[str, list] = {}
    for e in entries:
        by_sym.setdefault(e.symbol, []).append(e)

    regime_pnl: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        entry_ts = t.get("entry_ts")
        if entry_ts is None:
            regime = "UNKNOWN"
        else:
            # Busca audit entry mais próximo antes do entry_ts
            entry_epoch = entry_ts.timestamp() if hasattr(entry_ts, "timestamp") else float(entry_ts)
            candidates  = [
                e for e in by_sym.get(sym, [])
                if hasattr(e.timestamp, "timestamp")
                and abs(e.timestamp.timestamp() - entry_epoch) <= 7200  # ± 2h
            ]
            if candidates:
                closest = min(candidates, key=lambda e: abs(e.timestamp.timestamp() - entry_epoch))
                regime  = closest.regime or "UNKNOWN"
            else:
                regime = "UNKNOWN"

        r = regime_pnl.setdefault(regime, {"pnl": 0.0, "trades": 0, "wins": 0})
        r["pnl"]    += t["pnl"]
        r["trades"] += 1
        if t["pnl"] > 0:
            r["wins"] += 1

    # Normaliza
    for _rname, r in regime_pnl.items():
        r["pnl"]      = round(r["pnl"], 2)
        r["win_rate"] = round(r["wins"] / r["trades"], 3) if r["trades"] else 0
        r["avg_pnl"]  = round(r["pnl"] / r["trades"], 2)  if r["trades"] else 0

    return regime_pnl


@router.get("/distribution")
async def get_distribution(request: Request) -> dict:
    """
    Phase 7 — Distribution Analytics.

    Retorna:
    - return_dist       : histograma de P&L% por trade (10 bins)
    - holding_dist      : histograma de duração em horas
    - regime_contrib    : P&L, trades, WR por regime de mercado
    - streaks           : max win/loss streak, streak atual
    - extremes          : top 5 melhores e piores trades
    - summary           : skewness, kurtosis, % acima da média
    """
    db     = request.app.state.db
    orders = await _get_filled_orders(db)
    trades = _pair_trades(orders)

    pnl_pcts = [t["pnl_pct"] * 100 for t in trades]   # em %
    pnls     = [t["pnl"]     for t in trades]
    holdings = [
        h for t in trades
        if (h := _holding_hours(t.get("entry_ts"), t.get("exit_ts"))) is not None
    ]

    # ── Histograma de retornos (%) ────────────────────────────────────────────
    return_dist = _histogram(pnl_pcts, n_bins=10)

    # ── Histograma de holding time (horas) ───────────────────────────────────
    holding_dist = _histogram(holdings, n_bins=8)

    # ── Regime contribution ───────────────────────────────────────────────────
    regime_contrib = _regime_contribution(trades)

    # ── Streaks ───────────────────────────────────────────────────────────────
    streaks = _streak_analysis(pnls)

    # ── Extremos ──────────────────────────────────────────────────────────────
    sorted_trades = sorted(trades, key=lambda t: t["pnl"])
    def _fmt(t: dict) -> dict:
        ts = t.get("exit_ts")
        label = ts.strftime("%d/%m %H:%M") if hasattr(ts, "strftime") else "?"
        return {
            "symbol":  t["symbol"],
            "pnl":     round(t["pnl"], 2),
            "pnl_pct": round(t["pnl_pct"] * 100, 3),
            "holding_h": _holding_hours(t.get("entry_ts"), t.get("exit_ts")),
            "exit_at": label,
        }
    worst = [_fmt(t) for t in sorted_trades[:5]]
    best  = [_fmt(t) for t in sorted_trades[-5:][::-1]]

    # ── Estatísticas de distribuição ──────────────────────────────────────────
    skewness = kurtosis = above_avg_pct = None
    if len(pnl_pcts) >= MIN_TRADES:
        mean = statistics.mean(pnl_pcts)
        std  = statistics.stdev(pnl_pcts) if len(pnl_pcts) > 1 else 0
        if std > 0:
            # Pearson skewness (3 * (mean - median) / std)
            med      = statistics.median(pnl_pcts)
            skewness = round(3 * (mean - med) / std, 3)
            # Excess kurtosis (Fisher)
            n = len(pnl_pcts)
            if n >= 4:
                kurt_num = sum(((x - mean) / std) ** 4 for x in pnl_pcts) / n
                kurtosis = round(kurt_num - 3, 3)
        above_avg_pct = round(100 * sum(1 for x in pnl_pcts if x > mean) / len(pnl_pcts), 1)

    avg_holding_h = round(statistics.mean(holdings), 2) if holdings else None
    med_holding_h = round(statistics.median(holdings), 2) if holdings else None

    return {
        "n_trades":         len(trades),
        "return_dist":      return_dist,
        "holding_dist":     holding_dist,
        "regime_contrib":   regime_contrib,
        "streaks":          streaks,
        "best_trades":      best,
        "worst_trades":     worst,
        "summary": {
            "skewness":        skewness,
            "excess_kurtosis": kurtosis,
            "above_avg_pct":   above_avg_pct,
            "avg_holding_h":   avg_holding_h,
            "median_holding_h": med_holding_h,
            "avg_pnl_pct":     round(statistics.mean(pnl_pcts), 4) if pnl_pcts else None,
            "median_pnl_pct":  round(statistics.median(pnl_pcts), 4) if pnl_pcts else None,
        },
        "mae_mfe_note": "MAE/MFE full tracking requires intra-trade candle storage (Phase 10+)",
        "computed_at":  datetime.now(UTC).isoformat(),
    }


# ── Phase 8 — Reality Check ───────────────────────────────────────────────────

N_BOOTSTRAP  = 1000   # iterações bootstrap
N_PERMUTE    = 1000   # iterações permutação
ALPHA        = 0.05   # nível de significância → IC 95%


def _bootstrap_metric(values: list[float], metric_fn, n: int = N_BOOTSTRAP) -> dict:
    """
    Bootstrap com reposição: retorna média, IC 95% inferior/superior e std.
    metric_fn recebe list[float] e retorna float|None.
    """
    if len(values) < MIN_TRADES:
        return {"mean": None, "ci_low": None, "ci_high": None, "std": None}

    results = []
    for _ in range(n):
        sample = [_random.choice(values) for _ in range(len(values))]
        v = metric_fn(sample)
        if v is not None:
            results.append(v)

    if not results:
        return {"mean": None, "ci_low": None, "ci_high": None, "std": None}

    results.sort()
    lo = int(ALPHA / 2 * len(results))
    hi = int((1 - ALPHA / 2) * len(results)) - 1
    mean = sum(results) / len(results)
    std  = (sum((x - mean) ** 2 for x in results) / len(results)) ** 0.5
    return {
        "mean":    round(mean, 4),
        "ci_low":  round(results[max(0, lo)], 4),
        "ci_high": round(results[min(len(results)-1, hi)], 4),
        "std":     round(std, 4),
    }


def _permutation_pvalue(pnl_pcts: list[float], real_sharpe: float | None,
                        n: int = N_PERMUTE) -> dict:
    """
    Testa H0: a sequência real de P&Ls não é melhor que uma aleatória.
    p-value = fração de permutações com Sharpe >= real Sharpe.
    p < 0.05 → edge estatisticamente significativo.
    """
    if real_sharpe is None or len(pnl_pcts) < MIN_TRADES:
        return {"p_value": None, "significant": None, "n_permutations": n,
                "perm_sharpe_mean": None, "perm_sharpe_p95": None}

    shuffled = pnl_pcts[:]
    perm_sharpes: list[float] = []
    for _ in range(n):
        _random.shuffle(shuffled)
        s = _sharpe(shuffled)
        if s is not None:
            perm_sharpes.append(s)

    if not perm_sharpes:
        return {"p_value": None, "significant": None, "n_permutations": n,
                "perm_sharpe_mean": None, "perm_sharpe_p95": None}

    count_ge = sum(1 for s in perm_sharpes if s >= real_sharpe)
    p_value  = round(count_ge / len(perm_sharpes), 4)

    perm_sharpes.sort()
    p95_idx = int(0.95 * len(perm_sharpes))

    return {
        "p_value":          p_value,
        "significant":      p_value < ALPHA,
        "n_permutations":   n,
        "perm_sharpe_mean": round(sum(perm_sharpes) / len(perm_sharpes), 4),
        "perm_sharpe_p95":  round(perm_sharpes[min(p95_idx, len(perm_sharpes)-1)], 4),
    }


def _fee_stress(trades: list[dict], multipliers: list[float]) -> list[dict]:
    """
    Para cada multiplicador de fee, recalcula P&L total, WR e Expectancy.
    Permite saber em quantas vezes o custo de transação destruiria o edge.
    """
    results = []
    for mult in multipliers:
        stressed_pnls = []
        for t in trades:
            extra_fee = t["fee"] * (mult - 1)   # custo adicional vs baseline
            stressed_pnl = t["pnl"] - extra_fee
            stressed_pnls.append(stressed_pnl)

        wins   = [p for p in stressed_pnls if p > 0]
        losses = [p for p in stressed_pnls if p <= 0]
        n      = len(stressed_pnls)
        wr     = round(len(wins) / n, 4) if n else 0
        avg_w  = sum(wins)   / len(wins)   if wins   else 0
        avg_l  = sum(losses) / len(losses) if losses else 0
        exp    = round(wr * avg_w + (1 - wr) * avg_l, 2) if n >= MIN_TRADES else None
        total  = round(sum(stressed_pnls), 2)
        pcts   = [p / t["entry_px"] / t["qty"] if t["entry_px"] * t["qty"] > 0 else 0
                  for p, t in zip(stressed_pnls, trades, strict=False)]
        sh     = _sharpe(pcts)

        results.append({
            "fee_multiplier": mult,
            "total_pnl":      total,
            "win_rate":       wr,
            "expectancy":     exp,
            "sharpe":         sh,
            "profitable":     total > 0,
        })
    return results


@router.get("/reality_check")
async def get_reality_check(request: Request) -> dict:
    """
    Phase 8 — Reality Check: valida estatisticamente se o edge é real.

    Retorna:
    - bootstrap     : IC 95% de Sharpe, WR e Expectancy por reamostragem
    - permutation   : p-value (H0: sequência aleatória tão boa quanto real)
    - fee_stress    : impacto de 1×/2×/3×/5× fees na lucratividade
    - verdict       : resumo semáforo (GREEN/YELLOW/RED) por critério
    """
    db     = request.app.state.db
    orders = await _get_filled_orders(db)
    trades = _pair_trades(orders)

    n = len(trades)
    pnl_pcts = [t["pnl_pct"] for t in trades]
    pnls     = [t["pnl"]     for t in trades]
    wins     = [p for p in pnls if p > 0]

    if n < MIN_TRADES:
        return {
            "n_trades":       n,
            "min_required":   MIN_TRADES,
            "sufficient_data": False,
            "computed_at":    datetime.now(UTC).isoformat(),
        }

    # ── Métricas reais ──────────────────────────────────────────────────────
    real_sharpe     = _sharpe(pnl_pcts)
    real_wr         = round(len(wins) / n, 4)
    avg_w           = sum(wins) / len(wins) if wins else 0
    losses          = [p for p in pnls if p <= 0]
    avg_l           = sum(losses) / len(losses) if losses else 0
    real_expectancy = round(real_wr * avg_w + (1 - real_wr) * avg_l, 2)

    # ── Bootstrap ───────────────────────────────────────────────────────────
    def _wr(pcts):
        p = [x for x in pcts if x > 0]
        return len(p) / len(pcts) if pcts else None

    def _exp(pcts):
        # pcts aqui são retornos em fração (não $), precisamos de pnl_pct em $
        # reutilizamos a ideia mas mapeando de volta
        w = [x for x in pcts if x > 0]
        losses_ = [x for x in pcts if x <= 0]
        if not pcts:
            return None
        wr_ = len(w) / len(pcts)
        aw  = sum(w)       / len(w)       if w       else 0
        al  = sum(losses_) / len(losses_) if losses_ else 0
        return wr_ * aw + (1 - wr_) * al

    bs_sharpe     = _bootstrap_metric(pnl_pcts, _sharpe)
    bs_wr         = _bootstrap_metric(pnl_pcts, _wr)
    bs_expectancy = _bootstrap_metric(pnl_pcts, _exp)

    # ── Permutation test ────────────────────────────────────────────────────
    perm = _permutation_pvalue(pnl_pcts, real_sharpe)

    # ── Fee stress ──────────────────────────────────────────────────────────
    fee_stress = _fee_stress(trades, [1.0, 1.5, 2.0, 3.0, 5.0])

    # ── Verdict ─────────────────────────────────────────────────────────────
    def _signal(condition: bool | None, *, invert: bool = False) -> str:
        if condition is None:
            return "GREY"
        ok = not condition if invert else condition
        return "GREEN" if ok else "RED"

    sharpe_ok    = real_sharpe is not None and real_sharpe > 0
    pvalue_ok    = perm["p_value"] is not None and perm["p_value"] < ALPHA
    ci_positive  = bs_sharpe["ci_low"] is not None and bs_sharpe["ci_low"] > 0
    fee2x_ok     = fee_stress[2]["profitable"] if len(fee_stress) > 2 else None

    verdict = {
        "sharpe_positive":    {"status": _signal(sharpe_ok),   "value": real_sharpe,
                               "note": "Sharpe > 0"},
        "pvalue_significant": {"status": _signal(pvalue_ok),   "value": perm["p_value"],
                               "note": f"p < {ALPHA} → edge real"},
        "ci_positive":        {"status": _signal(ci_positive), "value": bs_sharpe["ci_low"],
                               "note": "IC 95% inferior > 0"},
        "survives_2x_fees":   {"status": _signal(fee2x_ok),    "value": fee_stress[2]["total_pnl"] if len(fee_stress)>2 else None,
                               "note": "Lucrativo com 2× fees"},
        "overall": "GREEN" if all(v["status"]=="GREEN" for v in [
                        {"status": _signal(sharpe_ok)},
                        {"status": _signal(pvalue_ok)},
                        {"status": _signal(ci_positive)},
                    ]) else ("YELLOW" if any(v["status"]=="GREEN" for v in [
                        {"status": _signal(sharpe_ok)},
                        {"status": _signal(pvalue_ok)},
                    ]) else "RED"),
    }

    return {
        "n_trades":        n,
        "sufficient_data": True,
        "real_metrics": {
            "sharpe":       real_sharpe,
            "win_rate":     real_wr,
            "expectancy":   real_expectancy,
        },
        "bootstrap": {
            "sharpe":     bs_sharpe,
            "win_rate":   bs_wr,
            "expectancy": bs_expectancy,
            "n_iterations": N_BOOTSTRAP,
        },
        "permutation":  perm,
        "fee_stress":   fee_stress,
        "verdict":      verdict,
        "alpha":        ALPHA,
        "computed_at":  datetime.now(UTC).isoformat(),
    }


# ── Phase 9 — Meta-Overfitting ────────────────────────────────────────────────
#
# Referência: López de Prado (2018) "Advances in Financial Machine Learning"
#   Cap. 8: The Deflated Sharpe Ratio
#   Cap. 11: Feature Importance
#
# Implementação 100% stdlib (sem scipy/numpy) usando math.erf para CDF normal.

_EULER_MASCHERONI = 0.5772156649


def _norm_cdf(x: float) -> float:
    """Função distribuição acumulada normal padrão via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Inversa da CDF normal (percent-point function) — aproximação racional.
    Erro < 4.5e-4 para p ∈ (0, 1).  Abramowitz & Stegun 26.2.17.
    """
    p = max(1e-10, min(1 - 1e-10, p))
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        sign = -1
    else:
        t = math.sqrt(-2.0 * math.log(1 - p))
        sign = 1
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num   = c0 + c1 * t + c2 * t * t
    denom = 1 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - num / denom)


def _moments(returns: list[float]) -> tuple[float, float, float, float]:
    """Retorna (mean, std, skewness, excess_kurtosis) de uma série."""
    n    = len(returns)
    mean = sum(returns) / n
    if n < 2:
        return mean, 0.0, 0.0, 0.0
    var  = sum((x - mean) ** 2 for x in returns) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 1e-10
    skew = sum(((x - mean) / std) ** 3 for x in returns) / n
    kurt = sum(((x - mean) / std) ** 4 for x in returns) / n - 3.0
    return mean, std, skew, kurt


def _deflated_sharpe(
    sr_hat: float,
    returns: list[float],
    n_trials: int,
    sr_benchmark: float | None = None,
) -> dict:
    """
    Deflated Sharpe Ratio (DSR) — López de Prado (2018).

    Ajusta o Sharpe observado por:
      1. Não-normalidade dos retornos (skewness + kurtosis)
      2. Viés de seleção (n_trials tentativas de estratégia/parâmetros)

    DSR = Φ( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - γ₃*SR_hat + γ₄/4 * SR_hat²) )

    SR* (expected max Sharpe sob H0) = E[max Sharpe de n_trials N(0,1)/sqrt(T)]
        ≈ Z(1 - 1/n) onde Z é a CDF inversa normal

    Returns: dict com dsr, sr_star, significant (DSR > 0.95)
    """
    n = len(returns)
    if n < MIN_TRADES or sr_hat is None:
        return {"dsr": None, "sr_star": None, "significant": None,
                "adjustment": None, "n_obs": n}

    _, _, skew, kurt = _moments(returns)

    # SR* benchmark: Sharpe máximo esperado de n_trials tentativas aleatórias
    if sr_benchmark is not None:
        sr_star = sr_benchmark
    else:
        # Fórmula simplificada: E[max Z | n_trials] ≈ Φ⁻¹(1 - 1/n_trials)
        # Escalado para a magnitude dos dados: divide por sqrt(n-1)
        z_max  = _norm_ppf(1.0 - 1.0 / max(n_trials, 2))
        z_max2 = _norm_ppf(1.0 - 1.0 / (max(n_trials, 2) * math.e))
        sr_star = ((1 - _EULER_MASCHERONI) * z_max + _EULER_MASCHERONI * z_max2) / math.sqrt(n - 1)

    # Ajuste de não-normalidade: σ² efetivo = 1 - γ₃*SR + (γ₄/4)*SR²
    variance_adj = 1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2
    if variance_adj <= 0:
        variance_adj = 1e-6

    # DSR
    dsr_z   = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(variance_adj)
    dsr     = round(_norm_cdf(dsr_z), 4)
    significant = dsr >= (1 - ALPHA)  # DSR ≥ 0.95 → edge real

    return {
        "dsr":          dsr,
        "sr_star":      round(sr_star, 4),
        "dsr_z":        round(dsr_z, 4),
        "significant":  significant,
        "skewness":     round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "variance_adj": round(variance_adj, 4),
        "n_obs":        n,
        "n_trials":     n_trials,
    }


def _min_track_record(sr_hat: float, returns: list[float],
                      alpha: float = ALPHA) -> dict:
    """
    Minimum Track Record Length — quantos trades são necessários para
    rejeitar H0 (Sharpe ≤ 0) com confiança (1-α).

    MinTRL = 1 + (1 - γ₃*SR + (γ₄/4)*SR²) * (Φ⁻¹(1-α)/SR)²
    """
    n = len(returns)
    if sr_hat is None or sr_hat == 0 or n < MIN_TRADES:
        return {"min_trl": None, "current_n": n, "sufficient": None}

    _, _, skew, kurt = _moments(returns)
    variance_adj = max(1e-6, 1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2)
    z_alpha      = _norm_ppf(1.0 - alpha)

    if sr_hat > 0:
        min_trl = 1 + variance_adj * (z_alpha / sr_hat) ** 2
    else:
        # Sharpe negativo: MinTRL "infinito" → sem track record suficiente
        min_trl = float("inf")

    sufficient = n >= min_trl if math.isfinite(min_trl) else False
    return {
        "min_trl":    round(min_trl, 1) if math.isfinite(min_trl) else None,
        "current_n":  n,
        "sufficient": sufficient,
        "gap":        None if not math.isfinite(min_trl) else max(0, round(min_trl - n, 1)),
    }


def _is_oos_split(trades: list[dict], split: float = 0.7) -> dict:
    """
    Divide trades em IS (primeiros split%) e OOS (restantes).
    Compara Sharpe, WR e Expectancy nos dois períodos.
    Ratio IS_Sharpe / OOS_Sharpe > 1 indica degradação (possível overfit).
    """
    n = len(trades)
    if n < MIN_TRADES * 2:
        return {"sufficient": False, "n_is": 0, "n_oos": 0}

    n_is  = max(MIN_TRADES, int(n * split))
    n_oos = n - n_is
    if n_oos < MIN_TRADES:
        return {"sufficient": False, "n_is": n_is, "n_oos": n_oos}

    def _metrics(subset: list[dict]) -> dict:
        pcts  = [t["pnl_pct"] for t in subset]
        pnls  = [t["pnl"]     for t in subset]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]
        wr    = len(wins) / len(pnls) if pnls else 0
        aw    = sum(wins)   / len(wins)   if wins   else 0
        al    = sum(losses) / len(losses) if losses else 0
        exp   = wr * aw + (1 - wr) * al
        sh    = _sharpe(pcts)
        return {"sharpe": sh, "win_rate": round(wr, 4),
                "expectancy": round(exp, 2), "n": len(subset)}

    is_m  = _metrics(trades[:n_is])
    oos_m = _metrics(trades[n_is:])

    # Degradation ratio: OOS_Sharpe / IS_Sharpe (1.0 = sem degradação)
    deg_ratio = None
    if is_m["sharpe"] and oos_m["sharpe"] and is_m["sharpe"] != 0:
        deg_ratio = round(oos_m["sharpe"] / is_m["sharpe"], 3)

    # Classificação
    if deg_ratio is None:
        overfit_signal = "INSUFFICIENT"
    elif deg_ratio >= 0.5:
        overfit_signal = "LOW"       # OOS retém ≥ 50% do IS Sharpe
    elif deg_ratio >= 0.0:
        overfit_signal = "MODERATE"  # degradação entre 0 e 50%
    else:
        overfit_signal = "HIGH"      # OOS inverte sinal → overfit severo

    return {
        "sufficient":    True,
        "split_pct":     split,
        "n_is":          n_is,
        "n_oos":         n_oos,
        "is_metrics":    is_m,
        "oos_metrics":   oos_m,
        "degradation_ratio": deg_ratio,
        "overfit_signal":    overfit_signal,
    }


@router.get("/meta_overfitting")
async def get_meta_overfitting(
    request: Request,
    n_trials: int = Query(default=10, ge=1, le=1000,
                          description="Nº de configurações/estratégias testadas"),
) -> dict:
    """
    Phase 9 — Meta-Overfitting Analysis (López de Prado framework).

    Retorna:
    - dsr           : Deflated Sharpe Ratio (ajustado por não-normalidade + n_trials)
    - min_trl       : Mínimo de trades para rejeitar H0 com 95% de confiança
    - is_oos        : Comparação In-Sample vs Out-of-Sample (split 70/30)
    - verdict       : Semáforo por critério + overall
    """
    db     = request.app.state.db
    orders = await _get_filled_orders(db)
    trades = _pair_trades(orders)

    n        = len(trades)
    pnl_pcts = [t["pnl_pct"] for t in trades]
    sr_hat   = _sharpe(pnl_pcts)

    if n < MIN_TRADES:
        return {
            "n_trades":        n,
            "min_required":    MIN_TRADES,
            "sufficient_data": False,
            "computed_at":     datetime.now(UTC).isoformat(),
        }

    # ── DSR ─────────────────────────────────────────────────────────────────
    dsr_result = _deflated_sharpe(sr_hat, pnl_pcts, n_trials)

    # ── MinTRL ──────────────────────────────────────────────────────────────
    min_trl_result = _min_track_record(sr_hat, pnl_pcts)

    # ── IS/OOS ──────────────────────────────────────────────────────────────
    is_oos_result = _is_oos_split(trades)

    # ── Verdict ─────────────────────────────────────────────────────────────
    dsr_ok    = dsr_result.get("significant") is True
    trl_ok    = min_trl_result.get("sufficient") is True
    overfit   = is_oos_result.get("overfit_signal", "INSUFFICIENT")
    overfit_ok = overfit == "LOW"

    def _v(ok): return "GREEN" if ok else ("GREY" if ok is None else "RED")

    n_green = sum([dsr_ok, trl_ok, overfit_ok])
    overall = "GREEN" if n_green == 3 else ("YELLOW" if n_green >= 1 else "RED")

    verdict = {
        "dsr_significant": {
            "status": _v(dsr_ok),
            "value":  dsr_result.get("dsr"),
            "note":   "DSR ≥ 0.95 → edge real após ajuste de seleção",
        },
        "track_record_sufficient": {
            "status": _v(trl_ok),
            "value":  min_trl_result.get("current_n"),
            "note":   f"Precisa ≥ {min_trl_result.get('min_trl')} trades",
        },
        "low_overfit": {
            "status": _v(overfit_ok),
            "value":  is_oos_result.get("degradation_ratio"),
            "note":   f"Overfit: {overfit}",
        },
        "overall": overall,
    }

    return {
        "n_trades":        n,
        "sufficient_data": True,
        "sr_hat":          sr_hat,
        "n_trials":        n_trials,
        "dsr":             dsr_result,
        "min_trl":         min_trl_result,
        "is_oos":          is_oos_result,
        "verdict":         verdict,
        "alpha":           ALPHA,
        "computed_at":     datetime.now(UTC).isoformat(),
    }


# ── Phase 10 — Futures Flow ───────────────────────────────────────────────────

SYMBOLS_FF = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


@router.get("/futures_flow")
async def get_futures_flow(request: Request) -> dict:
    """
    Phase 10 — Futures Flow: funding rate, Open Interest e M6 score.

    Lê do Redis os dados coletados pelo FuturesFlowCollector (cache 15min).
    Retorna dados por símbolo + sumário agregado.
    """
    cache = request.app.state.cache
    by_symbol: dict[str, dict] = {}

    for symbol in SYMBOLS_FF:
        raw = await cache.get(f"futures_flow:{symbol}")
        if raw:
            # cache.get() already parses JSON → may return dict or str
            if isinstance(raw, dict):
                by_symbol[symbol] = raw
            else:
                try:
                    by_symbol[symbol] = json.loads(raw)
                except Exception:
                    pass

    # Sumário agregado — M6 médio, funding médio
    m6_scores   = [d["scores"]["m6"]       for d in by_symbol.values() if d.get("scores")]
    fundings    = [d["funding_rate_pct"]    for d in by_symbol.values() if d.get("funding_rate_pct") is not None]
    oi_changes  = [d["oi_change_pct"]       for d in by_symbol.values() if d.get("oi_change_pct") is not None]

    summary = {
        "avg_m6":           round(sum(m6_scores)  / len(m6_scores),  4) if m6_scores  else None,
        "avg_funding_pct":  round(sum(fundings)   / len(fundings),   5) if fundings   else None,
        "avg_oi_change_pct":round(sum(oi_changes) / len(oi_changes), 3) if oi_changes else None,
        "n_symbols":        len(by_symbol),
        "has_data":         len(by_symbol) > 0,
    }

    # Interpretação do funding médio
    if summary["avg_funding_pct"] is not None:
        fr = summary["avg_funding_pct"]
        if fr < -0.02:
            summary["market_sentiment"] = "BEARISH (longs recebem)"
        elif fr < 0.0:
            summary["market_sentiment"] = "LEVE BEARISH"
        elif fr < 0.01:
            summary["market_sentiment"] = "NEUTRO"
        elif fr < 0.05:
            summary["market_sentiment"] = "BULLISH (bulls pagam moderado)"
        else:
            summary["market_sentiment"] = "CROWDED LONG (cuidado)"
    else:
        summary["market_sentiment"] = "SEM DADOS"

    return {
        "by_symbol":  by_symbol,
        "summary":    summary,
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ── Phase 11 — Relative Strength ─────────────────────────────────────────────

SYMBOLS_RS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


@router.get("/relative_strength")
async def get_relative_strength(request: Request) -> dict:
    """
    Phase 11 — Relative Strength: RS vs BTC, BTC leadership, M7 score.

    Lê do Redis os dados coletados pelo RelativeStrengthCollector (cache 15min).
    """
    cache = request.app.state.cache
    by_symbol: dict[str, dict] = {}

    for symbol in SYMBOLS_RS:
        raw = await cache.get(f"relative_strength:{symbol}")
        if raw:
            by_symbol[symbol] = raw if isinstance(raw, dict) else json.loads(raw)

    # Sumário: qual símbolo tem maior RS? BTC liderando?
    m7_scores    = {s: d["scores"]["m7"]        for s, d in by_symbol.items() if d.get("scores")}
    leaderships  = [d["scores"]["leadership"]   for d in by_symbol.values() if d.get("scores")]
    rs_1h_values = {s: d.get("rs_1h", 1.0)     for s, d in by_symbol.items()}

    btc_leading = None
    if leaderships:
        avg_lead = sum(leaderships) / len(leaderships)
        btc_leading = avg_lead >= 0.55

    # Símbolo com maior força relativa
    strongest = max(m7_scores, key=m7_scores.get) if m7_scores else None
    weakest   = min(m7_scores, key=m7_scores.get) if m7_scores else None

    summary = {
        "avg_m7":       round(sum(m7_scores.values()) / len(m7_scores), 4) if m7_scores else None,
        "btc_leading":  btc_leading,
        "strongest_rs": strongest,
        "weakest_rs":   weakest,
        "has_data":     len(by_symbol) > 0,
        "market_context": (
            "BTC liderando — maré alta favorece alts" if btc_leading
            else "BTC fraco — risco de alt underperformance" if btc_leading is False
            else "Sem dados"
        ),
    }

    return {
        "by_symbol":   by_symbol,
        "rs_1h":       rs_1h_values,
        "summary":     summary,
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ── Phase 12 — Volatility State ──────────────────────────────────────────────

SYMBOLS_VS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

VOL_STATE_M8 = {
    "EXPANDING":      0.80,
    "TREND":          0.70,
    "COMPRESSED":     0.65,
    "MEAN_REVERTING": 0.35,
    "CHAOTIC":        0.20,
}


@router.get("/volatility_state")
async def get_volatility_state(request: Request) -> dict:
    """
    Phase 12 — Volatility State Machine: estado atual da volatilidade por símbolo.

    Lê do Redis os dados calculados pelo VolatilityStateCollector (cache 15min).
    Estados: EXPANDING / TREND / COMPRESSED / MEAN_REVERTING / CHAOTIC
    """
    cache = request.app.state.cache
    by_symbol: dict[str, dict] = {}

    for symbol in SYMBOLS_VS:
        raw = await cache.get(f"vol_state:{symbol}")
        if raw:
            by_symbol[symbol] = raw if isinstance(raw, dict) else json.loads(raw)

    # Sumário
    states   = {s: d.get("state", "UNKNOWN")   for s, d in by_symbol.items()}
    m8scores = {s: d.get("m8_score", 0.5)      for s, d in by_symbol.items()}
    avg_m8   = round(sum(m8scores.values()) / len(m8scores), 4) if m8scores else None

    # Estado mais comum
    from collections import Counter
    state_counts = Counter(states.values())
    dominant_state = state_counts.most_common(1)[0][0] if state_counts else "UNKNOWN"

    summary = {
        "avg_m8":         avg_m8,
        "dominant_state": dominant_state,
        "states":         states,
        "has_data":       len(by_symbol) > 0,
        "market_vol_context": {
            "EXPANDING":      "Breakout — momentum favorável",
            "TREND":          "Tendência — mercado direcional",
            "COMPRESSED":     "Compressão — aguardar breakout",
            "MEAN_REVERTING": "Lateralização — momentum fraco",
            "CHAOTIC":        "Caótico — evitar novas entradas",
            "UNKNOWN":        "Sem dados",
        }.get(dominant_state, "Misto"),
    }

    return {
        "by_symbol":   by_symbol,
        "summary":     summary,
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ── Phase 14 — Advanced Risk ──────────────────────────────────────────────────

@router.get("/advanced_risk")
async def get_advanced_risk(request: Request) -> dict:
    """
    Phase 14 — Advanced Risk: correlação, VaR, stress, circuit breakers, liquidez.
    Lê do Redis o snapshot calculado pelo AdvancedRiskManager (cache 15min).
    """
    cache = request.app.state.cache
    raw   = await cache.get("advanced_risk")

    if not raw:
        return {
            "has_data":     False,
            "message":      "AdvancedRiskManager ainda não rodou — aguarde 15min após boot",
            "computed_at":  datetime.now(UTC).isoformat(),
        }

    data = raw if isinstance(raw, dict) else json.loads(raw)
    data["has_data"] = True
    return data


# ── Phase 13 — Meta Regime ────────────────────────────────────────────────────

@router.get("/meta_regime")
async def get_meta_regime(request: Request) -> dict:
    """
    Phase 13 — Meta Regime: regime macro cross-asset (BTC+ETH+SOL).

    Lê do Redis o dado calculado pelo MetaRegimeDetector (cache 15min).
    Retorna regime, features, distâncias aos arquétipos e multiplicador de threshold.
    """
    cache = request.app.state.cache
    raw   = await cache.get("meta_regime")

    if not raw:
        return {
            "regime":         "UNKNOWN",
            "description":    "Collector ainda não rodou — aguarde 15min após boot",
            "color":          "muted",
            "confidence":     0.0,
            "threshold_mult": 1.0,
            "features_raw":   {},
            "has_data":       False,
            "computed_at":    datetime.now(UTC).isoformat(),
        }

    data = raw if isinstance(raw, dict) else json.loads(raw)
    data["has_data"] = True
    return data
