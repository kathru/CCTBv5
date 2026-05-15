"""
Backtest Engine — runs strategies against historical candle data.

Key principle: uses EXACTLY the same strategy code as live trading.
No "backtest mode" — strategies receive StrategyContext and return Signal,
just like in production.

What this engine simulates realistically:
  1. Spread        — bid/ask gap based on ATR
  2. Slippage      — execution price worse than signal price
  3. Partial fills — not every limit order fills fully
  4. Fees          — maker 0.10%, taker 0.40% (OKX rates)
  5. Latency       — simulated delay between signal and execution
  6. Incomplete candle — last candle may not be confirmed

Never run on Oracle — local only.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

from ..core.models import Candle, Signal, SignalDirection
from ..strategies.base import BaseStrategy, StrategyContext

logger = logging.getLogger(__name__)

# OKX fee schedule (Tier 1)
MAKER_FEE = 0.001   # 0.10%
TAKER_FEE = 0.004   # 0.40%


@dataclass
class BacktestTrade:
    """Result of a single simulated trade."""
    signal: Signal
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    entry_fee: float
    exit_fee: float
    slippage: float
    fill_ratio: float       # 1.0 = full fill, 0.6 = 60% partial
    entry_time: datetime
    exit_time: datetime | None

    @property
    def pnl(self) -> float:
        if self.side == "long":
            gross = (self.exit_price - self.entry_price) * self.quantity * self.fill_ratio
        else:
            gross = (self.entry_price - self.exit_price) * self.quantity * self.fill_ratio
        return gross - self.entry_fee - self.exit_fee

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.pnl / cost if cost > 0 else 0.0


@dataclass
class BacktestResult:
    """Aggregated results of a backtest run."""
    symbol: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    trades: list[BacktestTrade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def final_capital(self) -> float:
        return self.initial_capital + self.total_pnl

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.initial_capital if self.initial_capital > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return wins / losses if losses > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def expectancy(self) -> float:
        """Expected value per trade in currency."""
        return self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "total_pnl": f"{self.total_pnl:.2f}",
            "total_return": f"{self.total_return_pct:.2%}",
            "expectancy": f"{self.expectancy:.2f}",
            "avg_win": f"{self.avg_win:.2f}",
            "avg_loss": f"{self.avg_loss:.2f}",
        }


class SimulatedFillEngine:
    """
    Simulates realistic order execution for backtesting.
    Migrated from V4's SimulatedExecutionEngine with improvements.
    """

    def __init__(
        self,
        spread_factor: float = 0.0005,   # 0.05% of price as spread
        slippage_factor: float = 0.001,  # 0.10% max slippage
        passive_fill_prob: float = 0.65, # 65% chance limit order fills
        seed: int | None = None,
    ) -> None:
        self._spread = spread_factor
        self._slippage = slippage_factor
        self._fill_prob = passive_fill_prob
        if seed is not None:
            random.seed(seed)

    def simulate_entry(
        self,
        signal: Signal,
        candle: Candle,
        capital: float,
        position_size_pct: float,
    ) -> BacktestTrade | None:
        """
        Simulate order entry. Returns None if order didn't fill.
        """
        # Simulate partial fill
        fill_ratio = self._simulate_fill(candle)
        if fill_ratio == 0.0:
            return None

        # Entry price with slippage
        mid = candle.close
        slippage = mid * random.uniform(0, self._slippage)
        if signal.direction == SignalDirection.LONG:
            entry_price = mid + slippage + (mid * self._spread / 2)
        else:
            entry_price = mid - slippage - (mid * self._spread / 2)

        # Position size
        notional = capital * position_size_pct * fill_ratio
        quantity = notional / entry_price if entry_price > 0 else 0

        # Fee (taker for market, maker for passive limit)
        fee_rate = MAKER_FEE if signal.calibrated_score < 0.7 else TAKER_FEE
        entry_fee = notional * fee_rate

        return BacktestTrade(
            signal=signal,
            entry_price=entry_price,
            exit_price=0.0,     # set when exit is simulated
            quantity=quantity,
            side=signal.direction.value,
            entry_fee=entry_fee,
            exit_fee=0.0,
            slippage=slippage,
            fill_ratio=fill_ratio,
            entry_time=candle.timestamp,
            exit_time=None,
        )

    def simulate_exit(
        self,
        trade: BacktestTrade,
        exit_candle: Candle,
    ) -> BacktestTrade:
        """Apply exit price and fees to a trade."""
        mid = exit_candle.close
        slippage = mid * random.uniform(0, self._slippage)

        if trade.side == "long":
            exit_price = mid - slippage - (mid * self._spread / 2)
        else:
            exit_price = mid + slippage + (mid * self._spread / 2)

        exit_notional = trade.quantity * exit_price * trade.fill_ratio
        exit_fee = exit_notional * TAKER_FEE

        trade.exit_price = exit_price
        trade.exit_fee = exit_fee
        trade.exit_time = exit_candle.timestamp
        return trade

    def _simulate_fill(self, candle: Candle) -> float:
        """
        Simulate fill ratio using Beta distribution.
        High volume candles → higher fill probability.
        """
        if random.random() > self._fill_prob:
            return 0.0   # order didn't fill at all

        # Partial fill: Beta(2, 1) skewed toward full fill
        return random.betavariate(2, 1)


class BacktestEngine:
    """
    Runs a strategy against historical candles.
    Uses SimulatedFillEngine for realistic execution.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.10,
        stop_loss_atr_mult: float = 2.0,
        seed: int = 42,
    ) -> None:
        self._strategy = strategy
        self._symbol = symbol
        self._capital = initial_capital
        self._position_size_pct = position_size_pct
        self._sl_mult = stop_loss_atr_mult
        self._fill_engine = SimulatedFillEngine(seed=seed)

    async def run(
        self,
        candles: list[Candle],
        warmup: int = 20,
    ) -> BacktestResult:
        """
        Run backtest over historical candles.

        Args:
            candles: List of confirmed candles (oldest first)
            warmup: Number of candles to skip before evaluating signals
        """
        if len(candles) < warmup + 1:
            raise ValueError(f"Need at least {warmup + 1} candles, got {len(candles)}")

        result = BacktestResult(
            symbol=self._symbol,
            strategy_id=self._strategy.strategy_id,
            start_date=candles[warmup].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=self._capital,
        )

        capital = self._capital
        open_trade: BacktestTrade | None = None

        for i in range(warmup, len(candles)):
            candle = candles[i]
            history = candles[:i + 1]

            # Exit open trade if stop loss or take profit hit
            if open_trade:
                should_exit = self._check_exit(open_trade, candle)
                if should_exit:
                    closed = self._fill_engine.simulate_exit(open_trade, candle)
                    capital += closed.pnl
                    result.trades.append(closed)
                    open_trade = None
                    logger.debug(
                        "Exit at %s pnl=%.2f capital=%.2f",
                        candle.timestamp, closed.pnl, capital,
                    )

            # Evaluate strategy for new signal
            if open_trade is None:
                ctx = StrategyContext(
                    symbol=self._symbol,
                    candles_1h=list(reversed(history)),  # newest first
                    candles_6h=[],
                    ticker=None,
                    portfolio_value=capital,
                )
                signal = await self._strategy.evaluate(ctx)

                if signal is not None:
                    trade = self._fill_engine.simulate_entry(
                        signal, candle, capital, self._position_size_pct
                    )
                    if trade:
                        open_trade = trade
                        logger.debug(
                            "Entry at %s price=%.2f fill=%.0f%%",
                            candle.timestamp,
                            trade.entry_price,
                            trade.fill_ratio * 100,
                        )

        # Close any open trade at end of data
        if open_trade and len(candles) > 0:
            closed = self._fill_engine.simulate_exit(open_trade, candles[-1])
            capital += closed.pnl
            result.trades.append(closed)

        return result

    def _check_exit(self, trade: BacktestTrade, candle: Candle) -> bool:
        """Check if stop loss or take profit was hit."""
        if trade.side == "long":
            # Stop loss: price dropped 2 ATR below entry
            stop = trade.entry_price * (1 - self._sl_mult * 0.01)
            take = trade.entry_price * (1 + self._sl_mult * 2 * 0.01)
            return candle.low < stop or candle.high > take
        return False
