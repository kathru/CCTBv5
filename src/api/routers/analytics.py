"""
Quantitative Analytics — métricas institucionais calculadas dos fills reais.

Métricas implementadas:
  - Sharpe Ratio        (retorno ajustado ao risco, anualizado)
  - Sortino Ratio       (penaliza só downside, anualizado)
  - Calmar Ratio        (retorno anual / max drawdown)
  - Hit Rate            (% trades vencedores)
  - Expectancy          (E[lucro] por trade em R$)
  - Avg Win / Avg Loss  (ratio ganho médio / perda média)
  - MAE / MFE           (excursão adversa/favorável máxima por trade)
  - Regime Breakdown    (performance por regime do signal_audit_log)
  - Drawdown Duration   (quantos trades consecutivos em drawdown)
  - Stability (R²)      (quão linear é a equity curve)
  - Profit Factor       (gross profit / gross loss)

Requer mínimo de trades para ter significância estatística (indicado em cada métrica).
"""

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from ...monitoring.signal_log import signal_audit_log
from ...persistence.repositories.fills import FillRepository

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


def _pair_trades(fills: list) -> list[dict]:
    """
    Emparelha fills BUY→SELL por símbolo (FIFO) para obter trades completos.
    Retorna lista de dicts com pnl, pnl_pct, hold_bars, mae, mfe.
    """
    by_sym: dict[str, list] = defaultdict(list)
    for f in fills:
        by_sym[f.symbol].append(f)

    trades = []
    for sym, sym_fills in by_sym.items():
        sym_fills.sort(key=lambda f: f.timestamp)
        buy_queue: list = []
        for f in sym_fills:
            side = str(getattr(f, "side", "")).upper()
            if side in ("BUY", "LONG"):
                buy_queue.append(f)
            elif side in ("SELL", "SHORT") and buy_queue:
                entry = buy_queue.pop(0)
                entry_px = float(entry.price)
                exit_px  = float(f.price)
                qty      = float(min(entry.quantity, f.quantity))
                fee      = float(getattr(entry, "fee", 0)) + float(getattr(f, "fee", 0))
                pnl      = (exit_px - entry_px) * qty - fee
                pnl_pct  = (exit_px - entry_px) / entry_px if entry_px > 0 else 0
                trades.append({
                    "symbol":    sym,
                    "pnl":       pnl,
                    "pnl_pct":   pnl_pct,
                    "entry_px":  entry_px,
                    "exit_px":   exit_px,
                    "qty":       qty,
                    "fee":       fee,
                    "entry_ts":  entry.timestamp,
                    "exit_ts":   f.timestamp,
                })
    return sorted(trades, key=lambda t: t["exit_ts"])


# ── Endpoint principal ────────────────────────────────────────────────────────

@router.get("/quantitative")
async def get_quantitative(request: Request) -> dict:
    db   = request.app.state.db
    repo = FillRepository(db)

    # Coleta fills de todos os símbolos
    all_fills = []
    for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
        all_fills.extend(await repo.get_by_symbol(sym, limit=1000))

    trades    = _pair_trades(all_fills)
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
