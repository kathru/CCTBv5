#!/usr/bin/env python3
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Bootstrap path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.models.candle import Candle
from src.replay.backtest_engine import BacktestEngine
from src.strategies.momentum.momentum_strategy import MomentumStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("simulation_2026")

CACHE_DIR = ROOT / "data" / "cache"

def load_candles(symbol: str, start_dt: datetime, end_dt: datetime) -> list[Candle]:
    cache_file = CACHE_DIR / f"{symbol.replace('-', '_')}_30m.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    all_data = json.loads(cache_file.read_text())

    # We need some warmup before the start date
    warmup_count = 100
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Find the index of the first candle >= start_ms
    start_idx = -1
    for i, c in enumerate(all_data):
        if c["ts"] >= start_ms:
            start_idx = i
            break

    if start_idx == -1:
        return []

    # Take warmup + data until end_ms
    effective_start_idx = max(0, start_idx - warmup_count)

    candles = []
    for i in range(effective_start_idx, len(all_data)):
        c = all_data[i]
        if c["ts"] > end_ms:
            break
        candles.append(Candle(
            symbol=symbol,
            granularity="30m",
            timestamp=datetime.fromtimestamp(c["ts"] / 1000, tz=UTC),
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            confirmed=True
        ))

    return candles

async def run_simulation():
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)
    end_dt = datetime(2026, 4, 30, 23, 59, 59, tzinfo=UTC)

    initial_total_capital = 5000.0
    # For simulation, we allocate 1/3 of the capital to each symbol's backtest engine
    # to simulate they are running in parallel with their own allocated "pocket".
    # Or we can run them with 5000 each and use a position_size_pct that reflects the split.
    # Let's run with 5000 total and assume they share it, but BacktestEngine is per-symbol.
    # Easiest is to give each 1666.66 and 10% position size.

    capital_per_symbol = initial_total_capital / len(symbols)

    all_results = []

    for symbol in symbols:
        logger.info(f"Running simulation for {symbol}...")
        candles = load_candles(symbol, start_dt, end_dt)
        if not candles:
            logger.warning(f"No candles found for {symbol} in the specified period.")
            continue

        strategy = MomentumStrategy(symbols=[symbol])
        # BacktestEngine uses warmup internal to its run method too.
        # We provided candles starting 100 before the period.
        engine = BacktestEngine(
            strategy=strategy,
            symbol=symbol,
            initial_capital=capital_per_symbol,
            position_size_pct=0.10, # 10% of its allocated capital per trade
            seed=42
        )

        # In load_candles we already took 100 candles before start.
        # BacktestEngine.run(candles, warmup=21)
        # We need to find where the 2026-01-01 starts in the candles list to pass as warmup.
        warmup_val = 0
        for i, c in enumerate(candles):
            if c.timestamp >= start_dt:
                warmup_val = i
                break

        result = await engine.run(candles, warmup=warmup_val)
        all_results.append(result)

    # Aggregate results
    total_pnl = sum(r.total_pnl for r in all_results)
    final_capital = initial_total_capital + total_pnl

    total_trades = sum(r.total_trades for r in all_results)
    total_wins = sum(r.winning_trades for r in all_results)
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    total_gross_wins = sum(sum(t.pnl for t in r.trades if t.pnl > 0) for r in all_results)
    total_gross_losses = sum(abs(sum(t.pnl for t in r.trades if t.pnl < 0)) for r in all_results)
    profit_factor = total_gross_wins / total_gross_losses if total_gross_losses > 0 else float('inf')

    print("\n" + "="*50)
    print("SIMULATION RESULTS: 2026-01-01 to 2026-04-30")
    print("="*50)
    print(f"Initial Portfolio: R$ {initial_total_capital:.2f}")
    print(f"Final Portfolio:   R$ {final_capital:.2f}")
    print(f"Total PnL:         R$ {total_pnl:.2f} ({total_pnl/initial_total_capital:.2%})")
    print(f"Overall Win Rate:  {win_rate:.1%}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Total Trades:      {total_trades}")
    print("-" * 50)

    for r in all_results:
        summary = r.summary()
        print(f"Symbol: {summary['symbol']}")
        print(f"  Trades: {summary['total_trades']}")
        print(f"  Win Rate: {summary['win_rate']}")
        print(f"  PnL: {summary['total_pnl']}")
        print(f"  Profit Factor: {summary['profit_factor']}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(run_simulation())
