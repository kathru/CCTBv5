"""
Backtest Multi-Meses — ReversalStrategy1H
Valida a estratégia em 18 meses: Jan 2025 → Abr 2026

Objetivo: verificar se Apr 2026 foi sorte ou a estratégia é consistente.
Cada mês roda de forma independente (mesmo capital inicial).

Uso: docker exec cctb_app python3 /tmp/btmm.py
"""
import asyncio
import json
import sys
from calendar import monthrange
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.core.models import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy, StrategyContext
from src.replay.backtest_engine import BacktestEngine, BacktestResult, BacktestTrade
from src.strategies.ml.inference import PlattCalibrator

ROOT      = Path("/app")
CACHE_DIR = ROOT / "data" / "cache"
SYMBOLS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

INITIAL_CAPITAL = 96_592.87
CAPITAL_PER_SYM = INITIAL_CAPITAL / len(SYMBOLS)
WARMUP_H        = 40   # 40h de contexto antes de cada mês


# ── Meses a testar ──────────────────────────────────────────────────────────
# (ano, mês)
MONTHS = [
    (2025,  1), (2025,  2), (2025,  3), (2025,  4),
    (2025,  5), (2025,  6), (2025,  7), (2025,  8),
    (2025,  9), (2025, 10), (2025, 11), (2025, 12),
    (2026,  1), (2026,  2), (2026,  3), (2026,  4),
]


def month_bounds(year: int, month: int) -> tuple[int, int]:
    """Retorna (start_ms, end_ms) de um mês em UTC."""
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=UTC)
    last  = monthrange(year, month)[1]
    end   = datetime(year, month, last, 23, 59, 59, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def load_candles(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    key      = symbol.replace("-", "_")
    raw      = json.loads((CACHE_DIR / f"{key}_1H.json").read_text())
    warmup_start = start_ms - WARMUP_H * 3_600_000
    filtered = sorted(
        [c for c in raw if warmup_start <= c["ts"] <= end_ms],
        key=lambda c: c["ts"],
    )
    return [
        Candle(
            symbol=symbol, granularity="1H",
            timestamp=datetime.fromtimestamp(c["ts"] / 1000, tz=UTC),
            open=c["open"], high=c["high"], low=c["low"],
            close=c["close"], volume=c["volume"], confirmed=True,
        )
        for c in filtered
    ]


# ── Estratégia de Reversão 1H (inline — sem depender do módulo live) ────────

class ReversalStrategy1H(BaseStrategy):
    MIN_FALL_PCT = 0.030
    BASE_CANDLES = 6
    TREND_FLOOR  = 0.92
    TREND_CEIL   = 1.02
    MIN_SL_PCT   = 0.008
    MAX_SL_PCT   = 0.060
    MIN_RATIO    = 1.5
    KELLY_BASE   = 0.08

    def __init__(self, symbols):
        super().__init__(strategy_id="reversal_1h", symbols=symbols)
        try:
            self._platt = PlattCalibrator(
                coef_path=ROOT / "data" / "models" / "calibration_coef.json"
            )
        except Exception:
            self._platt = None

    async def evaluate(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles_1h
        if len(c) < 22:
            return None

        closes  = [x.close  for x in c[:22]]
        highs   = [x.high   for x in c[:22]]
        lows    = [x.low    for x in c[:22]]
        volumes = [x.volume for x in c[:22]]

        current = closes[0]
        sma20   = sum(closes[:20]) / 20

        if not (sma20 * self.TREND_FLOOR <= current <= sma20 * self.TREND_CEIL):
            return None

        lookback_high = max(closes[self.BASE_CANDLES + 1:20])
        fall_low      = min(closes[1:20])
        fall_pct      = (lookback_high - fall_low) / lookback_high if lookback_high > 0 else 0

        if fall_pct < self.MIN_FALL_PCT:
            return None
        if fall_low > sma20 * 1.01:
            return None

        n          = self.BASE_CANDLES
        base_high  = max(highs[1:n + 1])
        base_low   = min(lows[1:n + 1])
        base_range = base_high - base_low
        fall_mag   = lookback_high - fall_low

        if current <= base_high * 1.001:
            return None

        avg_vol   = sum(volumes[1:9]) / 8
        vol_ratio = volumes[0] / avg_vol if avg_vol > 0 else 0
        if avg_vol > 0 and vol_ratio < 0.8:
            return None

        sl_target = base_low * 0.999
        sl_dist   = current - sl_target
        if sl_dist <= 0:
            return None

        sl_pct = sl_dist / current
        if not (self.MIN_SL_PCT <= sl_pct <= self.MAX_SL_PCT):
            return None

        tp_pct    = self.MIN_RATIO * sl_pct
        ratio     = self.MIN_RATIO

        score_fall  = min(fall_pct / 0.10, 1.0)
        score_base  = 1.0 - min(base_range / fall_mag, 1.0) if fall_mag > 0 else 0
        score_vol   = min(vol_ratio / 2.0, 1.0)
        score_ratio = min(ratio / 8.0, 1.0)
        raw_score   = score_fall*0.30 + score_base*0.30 + score_vol*0.20 + score_ratio*0.20

        calibrated = self._platt.calibrate(raw_score) if self._platt else raw_score

        return Signal(
            strategy_id=self._strategy_id,
            symbol=ctx.symbol,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            score=raw_score,
            calibrated_score=calibrated,
            confidence=calibrated,
            expected_value=ratio * calibrated - (1 - calibrated),
            kelly_fraction=round(min(self.KELLY_BASE, 0.10), 4),
            regime="REVERSAL_1H",
            timeframe="1H",
            factors={
                "sl_pct":     round(sl_pct, 5),
                "tp_pct":     round(tp_pct, 5),
                "fall_pct":   round(fall_pct, 3),
                "base_range": round(base_range / fall_mag, 3) if fall_mag > 0 else 0,
                "vol_ratio":  round(vol_ratio, 2),
                "tp_ratio":   round(ratio, 2),
                "sl_ref":     round(sl_target, 4),
            },
        )


# ── Engine com TP/SL relativos ao entry_price real ──────────────────────────

class ReversalEngine(BacktestEngine):
    def _check_exit(self, trade, candle, history):
        f      = trade.signal.factors if trade.signal else {}
        sl_pct = f.get("sl_pct", 0)
        tp_pct = f.get("tp_pct", 0)

        if sl_pct and tp_pct:
            sl_price = trade.entry_price * (1.0 - sl_pct)
            tp_price = trade.entry_price * (1.0 + tp_pct)
        else:
            sl_price = f.get("sl_target", 0)
            tp_price = f.get("tp_target", 0)

        if sl_price > 0 and candle.low <= sl_price:
            return True, "stop_loss"
        if tp_price > 0 and candle.high >= tp_price:
            return True, "take_profit"

        candles_held = sum(1 for c in history if c.timestamp >= trade.entry_time)
        if candles_held >= 48:
            return True, "timeout_48h"

        return False, ""


# ── Runner mensal ────────────────────────────────────────────────────────────

async def run_month(year: int, month: int) -> dict:
    """Roda backtest de um mês para os 3 símbolos. Retorna métricas consolidadas."""
    start_ms, end_ms = month_bounds(year, month)
    month_pnl  = 0.0
    month_fees = 0.0
    month_trades = 0
    month_wins = 0
    month_results = []

    for sym in SYMBOLS:
        candles = load_candles(sym, start_ms, end_ms)
        if len(candles) < WARMUP_H + 5:
            continue

        apr_idx = next(
            (i for i, c in enumerate(candles) if c.timestamp.timestamp() * 1000 >= start_ms),
            WARMUP_H,
        )

        strategy = ReversalStrategy1H(symbols=[sym])
        engine   = ReversalEngine(
            strategy=strategy,
            symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08,
            seed=42,
        )
        result = await engine.run(candles, warmup=apr_idx)

        month_pnl    += result.total_pnl
        month_fees   += result.total_fees
        month_trades += result.total_trades
        month_wins   += result.winning_trades
        month_results.append(result)

    wr = month_wins / month_trades if month_trades > 0 else 0
    return {
        "year":    year,
        "month":   month,
        "pnl":     month_pnl,
        "fees":    month_fees,
        "trades":  month_trades,
        "wins":    month_wins,
        "wr":      wr,
        "results": month_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    sep  = "═" * 78
    sep2 = "─" * 78

    print(sep)
    print("  CCTBv5 — Backtest Multi-Meses | ReversalStrategy1H | Jan 2025 → Abr 2026")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f} | Fee: 0.10% taker | 3 símbolos")
    print(f"  Regras: queda≥3% + base≥6H + rompimento + volume + SMA20")
    print(f"  Exit: SL=base_low×0.999 | TP=1.5×SL (relativo ao fill price)")
    print(sep)

    monthly: list[dict] = []

    for year, month in MONTHS:
        m = await run_month(year, month)
        monthly.append(m)
        sign = "+" if m["pnl"] >= 0 else ""
        bar  = "▓" * min(int(abs(m["pnl"]) / 20), 30)
        col  = "✅" if m["pnl"] >= 0 else "❌"
        print(
            f"  {col} {year}/{month:02d}  "
            f"P&L: {sign}${m['pnl']:>8.2f}  "
            f"trades: {m['trades']:>3}  "
            f"WR: {m['wr']:>5.1%}  "
            f"fees: ${m['fees']:>6.2f}  "
            f"{bar}"
        )

    # ── Consolidado ──────────────────────────────────────────────────────────
    total_pnl    = sum(m["pnl"]    for m in monthly)
    total_fees   = sum(m["fees"]   for m in monthly)
    total_trades = sum(m["trades"] for m in monthly)
    total_wins   = sum(m["wins"]   for m in monthly)
    positive_months = sum(1 for m in monthly if m["pnl"] > 0)
    negative_months = sum(1 for m in monthly if m["pnl"] < 0)
    zero_months     = sum(1 for m in monthly if m["trades"] == 0)

    overall_wr  = total_wins / total_trades if total_trades > 0 else 0
    monthly_avg = total_pnl / len(monthly)

    # Melhor e pior mês
    best  = max(monthly, key=lambda m: m["pnl"])
    worst = min(monthly, key=lambda m: m["pnl"])

    # Consistência: % de meses positivos (excluindo meses sem trades)
    active_months = [m for m in monthly if m["trades"] > 0]
    pos_active = (
        sum(1 for m in active_months if m["pnl"] > 0) / len(active_months)
        if active_months else 0
    )

    print()
    print(sep)
    print("  CONSOLIDADO — 16 MESES")
    print(sep)
    print(f"  Período          : Jan/2025 → Abr/2026")
    print(f"  P&L total        : {'+'if total_pnl>=0 else ''}${total_pnl:,.2f}")
    print(f"  Retorno total    : {'+'if total_pnl>=0 else ''}{total_pnl/INITIAL_CAPITAL:.2%}")
    print(f"  P&L médio/mês    : {'+'if monthly_avg>=0 else ''}${monthly_avg:,.2f}")
    print(f"  Total trades     : {total_trades}")
    print(f"  Win Rate global  : {overall_wr:.1%}")
    print(f"  Total fees       : ${total_fees:,.2f}")
    print(sep2)
    print(
        f"  Meses positivos  : {positive_months} / {len(monthly)}"
        f"  ({positive_months/len(monthly):.0%})"
    )
    print(
        f"  Meses negativos  : {negative_months} / {len(monthly)}"
        f"  ({negative_months/len(monthly):.0%})"
    )
    print(f"  Meses sem trades : {zero_months} / {len(monthly)}")
    print(f"  Consist. (ativos): {pos_active:.0%} dos meses com trades foram positivos")
    print(sep2)
    print(
        f"  Melhor mês       : {best['year']}/{best['month']:02d}"
        f"  +${best['pnl']:,.2f}  ({best['trades']} trades)"
    )
    print(
        f"  Pior mês         : {worst['year']}/{worst['month']:02d}"
        f"  ${worst['pnl']:,.2f}  ({worst['trades']} trades)"
    )
    print(sep)

    # ── Análise por símbolo ───────────────────────────────────────────────────
    print()
    print("  ANÁLISE POR SÍMBOLO (16 meses)")
    print(sep2)
    sym_stats: dict[str, dict] = {s: {"pnl": 0.0, "trades": 0, "wins": 0, "fees": 0.0}
                                   for s in SYMBOLS}
    for m in monthly:
        for r in m["results"]:
            st = sym_stats[r.symbol]
            st["pnl"]    += r.total_pnl
            st["trades"] += r.total_trades
            st["wins"]   += r.winning_trades
            st["fees"]   += r.total_fees

    for sym, st in sym_stats.items():
        wr  = st["wins"] / st["trades"] if st["trades"] > 0 else 0
        avg = st["pnl"] / st["trades"] if st["trades"] > 0 else 0
        sign = "+" if st["pnl"] >= 0 else ""
        print(
            f"  {sym:<12}  P&L: {sign}${st['pnl']:>8.2f}  "
            f"trades: {st['trades']:>3}  WR: {wr:>5.1%}  "
            f"avg/trade: {sign}${avg:>6.2f}  fees: ${st['fees']:>6.2f}"
        )

    # ── Veredicto ──────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  VEREDICTO")
    print(sep)

    if pos_active >= 0.60 and total_pnl > 0:
        verdict = "✅ ESTRATÉGIA VÁLIDA — Consistente em múltiplos meses"
        detail  = f"  {pos_active:.0%} dos meses com trades foram positivos. Edge real detectado."
    elif pos_active >= 0.50 and total_pnl > 0:
        verdict = "⚠️  MARGINAL — Positivo mas inconsistente"
        detail  = f"  {pos_active:.0%} meses positivos. Mais validação necessária antes de confiar."
    elif zero_months >= len(monthly) * 0.5:
        verdict = "⚠️  POUCOS SETUPS — Estratégia muito seletiva para estes dados"
        detail  = f"  {zero_months} meses sem trades. Limiar de 3% pode ser muito alto."
    else:
        verdict = "❌ ESTRATÉGIA FRACA — Não consistente o suficiente"
        detail  = f"  Apenas {pos_active:.0%} meses positivos. Revisão necessária."

    print(f"  {verdict}")
    print(detail)
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
