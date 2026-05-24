"""
Backtest — Janeiro a Abril 2026 (1H)
Simula o comportamento completo do robô para 01/01/2026 a 30/04/2026.
"""
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Adiciona o diretório raiz ao path para permitir importações do src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import Candle
from src.replay.backtest_engine import BacktestEngine, BacktestResult
from src.strategies.momentum.momentum_strategy import MomentumStrategy

ROOT      = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "cache"

SYMBOLS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
START_TS  = 1767225600000  # 2026-01-01 00:00 UTC
END_TS    = 1777593600000  # 2026-05-01 00:00 UTC

INITIAL_CAPITAL   = 96_592.87
CAPITAL_PER_SYM   = INITIAL_CAPITAL / len(SYMBOLS)

def load_candles(symbol: str) -> list[Candle]:
    key  = symbol.replace("-", "_")
    path = CACHE_DIR / f"{key}_1H.json"
    if not path.exists():
        print(f"Erro: Arquivo não encontrado {path}")
        return []
    raw  = json.loads(path.read_text())
    # Inclui 21 candles de warmup antes do início para contexto da estratégia
    warmup_start = START_TS - 21 * 3_600_000
    filtered = sorted(
        [c for c in raw if warmup_start <= c["ts"] <= END_TS],
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

async def run_backtest() -> None:
    sep = "═" * 68
    print(sep)
    print("  CCTBv5 — Backtest Jan-Abr 2026 (1H)")
    print(f"  Capital inicial : ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Período         : 01/01/2026 → 30/04/2026")
    print(f"  Símbolos        : {', '.join(SYMBOLS)}")
    print(f"  Fee OKX demo    : 0.15% (taker - paper fee default)")
    print(sep)

    results: list[BacktestResult] = []

    for sym in SYMBOLS:
        candles = load_candles(sym)
        if not candles:
            continue

        # Identifica primeiro candle de janeiro (após warmup)
        try:
            start_idx = next(
                i for i, c in enumerate(candles)
                if c.timestamp.timestamp() * 1000 >= START_TS
            )
        except StopIteration:
            print(f"  {sym}: Nenhum candle encontrado após a data de início.")
            continue

        print(
            f"\n  {sym}: {len(candles)} candles"
            f" ({candles[0].timestamp:%d/%m/%y} → {candles[-1].timestamp:%d/%m/%y}),"
            f" simulação começa no índice {start_idx}"
        )

        strategy = MomentumStrategy(symbols=[sym])
        engine   = BacktestEngine(
            strategy=strategy,
            symbol=sym,
            initial_capital=CAPITAL_PER_SYM,
            position_size_pct=0.08,
            seed=42,
            fee_tier="standard",
        )

        result = await engine.run(candles, warmup=start_idx)
        results.append(result)

        pnl_s = (
            f"+${result.total_pnl:.2f}" if result.total_pnl >= 0
            else f"-${abs(result.total_pnl):.2f}"
        )
        print(
            f"  → {result.total_trades} trades | WR={result.win_rate:.1%}"
            f" | PnL={pnl_s} | PF={result.profit_factor:.2f}"
        )

    # Consolidado final
    grand_pnl    = sum(r.total_pnl for r in results)
    grand_trades = sum(r.total_trades for r in results)
    grand_wins   = sum(r.winning_trades for r in results)
    grand_fees   = sum(r.total_fees for r in results)

    final_capital  = INITIAL_CAPITAL + grand_pnl
    total_return   = grand_pnl / INITIAL_CAPITAL
    overall_wr     = grand_wins / grand_trades if grand_trades else 0

    print(f"\n{sep}")
    print("  CONSOLIDADO — JAN-ABR 2026")
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
    print(f"  Período           : 120 dias")
    print(sep)

    # Imprime resultados individuais por crypto para o relatório
    for res in results:
        pnl_s = f"+${res.total_pnl:.2f}" if res.total_pnl >= 0 else f"-${abs(res.total_pnl):.2f}"
        print(f"\n[{res.symbol}] Final Capital: ${CAPITAL_PER_SYM + res.total_pnl:.2f} | PnL: {pnl_s} | WR: {res.win_rate:.1%} | PF: {res.profit_factor:.2f}")

if __name__ == "__main__":
    asyncio.run(run_backtest())
