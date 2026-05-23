"""
Backtest — Abril 2026 (1H)
Simula o comportamento completo do robô para 01–30/04/2026.

Uso: docker exec cctb_app python3 scripts/backtest_april2026.py
"""
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.core.models import Candle
from src.replay.backtest_engine import BacktestEngine, BacktestResult
from src.strategies.momentum.momentum_strategy import MomentumStrategy

ROOT      = Path("/app")
CACHE_DIR = ROOT / "data" / "cache"

SYMBOLS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
APR_START = 1743465600000  # 2026-04-01 00:00 UTC
APR_END   = 1746057600000  # 2026-04-30 23:59 UTC

# Capital dividido igualmente por símbolo (idêntico ao live)
INITIAL_CAPITAL   = 96_592.87
CAPITAL_PER_SYM   = INITIAL_CAPITAL / len(SYMBOLS)

# Kelly cap idêntico ao live (trading_loop.py)
KELLY_CAP = {
    "TREND_EXPANSION":        0.15,
    "VOLATILITY_COMPRESSION": 0.12,
    "TREND_EXHAUSTION":       0.10,
    "MEAN_REVERTING_CHOP":    0.08,
    "HIGH_CORRELATION_RISK":  0.05,
}


def load_candles(symbol: str) -> list[Candle]:
    key  = symbol.replace("-", "_")
    path = CACHE_DIR / f"{key}_1H.json"
    raw  = json.loads(path.read_text())
    # Inclui 21 candles de warmup antes de abril para contexto da estratégia
    warmup_start = APR_START - 21 * 3_600_000
    filtered = sorted(
        [c for c in raw if warmup_start <= c["ts"] <= APR_END],
        key=lambda c: c["ts"],
    )
    return [
        Candle(
            symbol=symbol,
            granularity="1H",
            timestamp=datetime.fromtimestamp(c["ts"] / 1000, tz=UTC),
            open=c["open"], high=c["high"],
            low=c["low"],   close=c["close"],
            volume=c["volume"],
            confirmed=True,
        )
        for c in filtered
    ]


def kelly_size_pct(regime: str) -> float:
    """Converte regime → position_size_pct idêntico ao live."""
    from src.strategies.momentum.momentum_strategy import REGIME_KELLY_MULT
    base = 0.08   # kelly base médio (calibrated_score ~0.37 × 0.25 = ~9%)
    mult = REGIME_KELLY_MULT.get(regime, 0.5)
    cap  = KELLY_CAP.get(regime, 0.08)
    return min(base * mult, cap)


async def run_backtest() -> None:
    sep = "═" * 68
    print(sep)
    print("  CCTBv5 — Backtest Abril 2026 (1H)")
    print(f"  Capital inicial : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Período         : 01/04/2026 → 30/04/2026")
    print(f"  Símbolos        : {', '.join(SYMBOLS)}")
    print(f"  Fee OKX demo    : 0.10% (taker)")
    print(sep)

    results: list[BacktestResult] = []

    for sym in SYMBOLS:
        candles = load_candles(sym)
        # Identifica primeiro candle de abril (após warmup)
        apr_idx = next(
            i for i, c in enumerate(candles)
            if c.timestamp.timestamp() * 1000 >= APR_START
        )
        print(
            f"\n  {sym}: {len(candles)} candles"
            f" ({candles[0].timestamp:%d/%m} → {candles[-1].timestamp:%d/%m}),"
            f" abril começa no índice {apr_idx}"
        )

        # Position size: regime-aware (usamos base 8% como proxy)
        # O engine respeita kelly via position_size_pct
        strategy = MomentumStrategy(symbols=[sym])
        engine   = BacktestEngine(
            strategy=strategy,
            symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08,   # base — strategy aplica regime_mult via kelly
            seed=42,
            fee_tier="standard",
        )

        result = await engine.run(candles, warmup=apr_idx)
        results.append(result)

        # Resumo por símbolo imediato
        pnl_s = (
            f"+${result.total_pnl:.2f}" if result.total_pnl >= 0
            else f"-${abs(result.total_pnl):.2f}"
        )
        print(
            f"  → {result.total_trades} trades | WR={result.win_rate:.1%}"
            f" | PnL={pnl_s} | PF={result.profit_factor:.2f}"
        )

    # ── Relatório detalhado por símbolo ───────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTADOS DETALHADOS POR SÍMBOLO")
    print(sep)

    grand_pnl    = 0.0
    grand_trades = 0
    grand_fees   = 0.0
    grand_wins   = 0

    for res in results:
        grand_pnl    += res.total_pnl
        grand_trades += res.total_trades
        grand_fees   += res.total_fees
        grand_wins   += res.winning_trades

        pnl_s = f"+${res.total_pnl:.2f}" if res.total_pnl >= 0 else f"-${abs(res.total_pnl):.2f}"
        print(f"\n  ── {res.symbol} ──")
        print(f"    Trades      : {res.total_trades}")
        if res.total_trades > 0:
            print(
                f"    Win Rate    : {res.win_rate:.1%}"
                f"  ({res.winning_trades}W / {res.total_trades - res.winning_trades}L)"
            )
            print(f"    Profit Fac. : {res.profit_factor:.2f}")
            print(f"    P&L         : {pnl_s}  ({res.total_return_pct:.2%})")
            print(f"    Fees totais : ${res.total_fees:.2f}")
            print(f"    Slippage    : ${res.total_slippage:.2f}")
            print(f"    Expectancy  : ${res.expectancy:.2f}/trade")
            print(f"    Avg Win     : ${res.avg_win:.2f}")
            print(f"    Avg Loss    : ${res.avg_loss:.2f}")
            print(f"    Avg Fill    : {res.avg_fill_ratio:.1%}")

            # Lista de trades
            print(
                f"\n    {'Entrada':<16} {'Saída':<16} {'Regime':<26}"
                f" {'Qty':>9} {'Entry':>9} {'Exit':>9} {'P&L':>9}"
            )
            print(f"    {'─'*16} {'─'*16} {'─'*26} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
            for t in sorted(res.trades, key=lambda x: x.entry_time):
                pnl_t = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
                exit_s = t.exit_time.strftime("%d/%m %H:%M") if t.exit_time else "timeout"
                regime = t.signal.regime if t.signal else "–"
                print(
                    f"    {t.entry_time.strftime('%d/%m %H:%M'):<16} {exit_s:<16}"
                    f" {regime:<26} {t.quantity:>9.5f}"
                    f" ${t.entry_price:>8,.2f} ${t.exit_price:>8,.2f} {pnl_t:>9}"
                )
        else:
            print(f"    Nenhum trade executado no período")

    # ── Consolidado final ─────────────────────────────────────────────────────
    final_capital  = INITIAL_CAPITAL + grand_pnl
    total_return   = grand_pnl / INITIAL_CAPITAL
    overall_wr     = grand_wins / grand_trades if grand_trades else 0

    print(f"\n{sep}")
    print("  CONSOLIDADO — ABRIL 2026")
    print(sep)
    print(f"  Capital inicial   : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Capital final     : ${final_capital:>12,.2f}")
    pnl_str = f"+${grand_pnl:,.2f}" if grand_pnl >= 0 else f"-${abs(grand_pnl):,.2f}"
    ret_str = f"+{total_return:.2%}" if total_return >= 0 else f"{total_return:.2%}"
    print(f"  P&L total         : {pnl_str:>13}")
    print(f"  Retorno total     : {ret_str:>13}")
    print(f"  Total de trades   : {grand_trades:>13}")
    print(f"  Win Rate global   : {overall_wr:>12.1%}")
    print(f"  Total fees pagas  : ${grand_fees:>12,.2f}")
    print(f"  Período           : 30 dias (abril/2026)")
    print(sep)


if __name__ == "__main__":
    asyncio.run(run_backtest())
