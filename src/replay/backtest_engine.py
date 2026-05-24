"""
Backtest Engine v2 — simulação realista com exit logic da Fase A/B/C.

Princípio fundamental: mesmo código de estratégia que o live trading.
Sem "backtest mode" — estratégias recebem StrategyContext e retornam Signal.

Componentes de simulação realista:
  1. Fees         — OKX Spot fee schedule (Maker/Taker por tier)
  2. Spread       — Proporcional ao ATR (alarga em volatilidade alta)
  3. Slippage     — Market impact proporcional a notional/volume
  4. Latency      — Sinal no close do candle N → execução no open do candle N+1
  5. Partial fills — Probabilidade baseada em volume relativo (não fixa)

Exit Logic v2 (Fase A — convexidade):
  6. Regime-aware SL/TP — multiplicadores por regime (TREND_EXPANSION, CHOP, etc.)
  7. TP Conversion — em TREND_EXPANSION o TP vira partial exit + running mode
  8. Trailing stop — conviction-adaptive (HOLD=2.0×, ALERT=0.5×)
  9. Conviction proxy — SMA + ATR + estrutura HH/HL substituem timeout

Sizing v2 (Fase B — sizing dinâmico):
  10. Usa signal.kelly_fraction calculado pela SizingEngine
  11. Vol state simplificado derivado dos candles (sem Redis)

Nunca rodar no Oracle — apenas localmente.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

from ..core.models import Candle, Signal, SignalDirection
from ..strategies.base import BaseStrategy, StrategyContext

logger = logging.getLogger(__name__)

# ── OKX Spot Fee Schedule (Tier 1 — conta padrão) ────────────────────────────
MAKER_FEE = 0.0010   # 0.10%
TAKER_FEE = 0.0015   # 0.15%
PAPER_FEE = TAKER_FEE

# ── Exit logic — espelha position_monitor.py ─────────────────────────────────

REGIME_MULT: dict[str, dict[str, float]] = {
    "TREND_EXPANSION":        {"sl": 1.5, "tp": 4.5},
    "VOLATILITY_COMPRESSION": {"sl": 1.0, "tp": 3.5},
    "MEAN_REVERTING_CHOP":    {"sl": 0.7, "tp": 2.0},
    "TREND_EXHAUSTION":       {"sl": 0.8, "tp": 2.5},
    "HIGH_CORRELATION_RISK":  {"sl": 0.8, "tp": 2.0},
    "BEAR_TREND":             {"sl": 0.5, "tp": 1.0},
    "PANIC_LIQUIDATION":      {"sl": 0.5, "tp": 1.0},
}

REGIME_TRAIL_ACTIVATE: dict[str, float] = {
    "TREND_EXPANSION":        1.2,
    "VOLATILITY_COMPRESSION": 1.0,
    "MEAN_REVERTING_CHOP":    1.5,
    "TREND_EXHAUSTION":       0.8,
    "HIGH_CORRELATION_RISK":  0.8,
    "BEAR_TREND":             0.3,
    "PANIC_LIQUIDATION":      0.2,
}

REGIME_PARTIAL_R: dict[str, float] = {
    "TREND_EXPANSION":        3.0,
    "VOLATILITY_COMPRESSION": 2.5,
    "MEAN_REVERTING_CHOP":    1.5,
    "TREND_EXHAUSTION":       1.5,
    "HIGH_CORRELATION_RISK":  1.5,
}

REGIMES_CONVERT_TP = {"TREND_EXPANSION"}
REGIMES_EMERGENCY  = {"PANIC_LIQUIDATION", "BEAR_TREND"}

CONVICTION_HOLD  = 70.0
CONVICTION_ALERT = 30.0
LOW_CONVICTION_STREAK_EXIT = 3     # ciclos consecutivos < 30% → exit
ABSOLUTE_BACKSTOP_CANDLES  = 720   # 30 dias de candles 1H


# ── Estado interno da posição durante simulação ───────────────────────────────

@dataclass
class _PositionState:
    """
    Estado de gestão de saída para uma posição aberta no backtest.
    Espelha ExitPlan do position_monitor.py — sem depender de Redis.
    """
    entry_regime:   str
    atr:            float
    stop_loss:      float
    take_profit:    float
    quantity:       float      # quantidade total original
    qty_remaining:  float      # quantidade ainda aberta

    trailing_stop:       float | None = None
    trailing_activated:  bool  = False
    partial_done:        bool  = False
    running_mode:        bool  = False
    tp_converted:        bool  = False

    low_conviction_streak: int = 0
    candles_held:          int = 0

    @property
    def r_value(self) -> float:
        return self.stop_loss - self.take_profit  # negativo — só para cálculo interno

    def sl_tp_from_regime(
        self, entry_price: float, atr: float, regime: str
    ) -> None:
        mults    = REGIME_MULT.get(regime, {"sl": 1.0, "tp": 3.0})
        self.stop_loss  = round(entry_price - atr * mults["sl"], 6)
        self.take_profit = round(entry_price + atr * mults["tp"], 6)


# ── BacktestTrade ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Resultado de um trade simulado completo."""
    signal:       Signal
    entry_price:  float
    exit_price:   float
    quantity:     float
    side:         str          # "long" | "short"
    entry_fee:    float
    exit_fee:     float
    slippage:     float
    fill_ratio:   float
    entry_time:   datetime
    exit_time:    datetime | None

    exit_reason:        str   = ""
    entry_regime:       str   = ""
    partial_pnl:        float = 0.0   # P&L da saída parcial (se houve)
    had_running_mode:   bool  = False
    candles_held:       int   = 0
    spread_cost:        float = 0.0
    market_impact:      float = 0.0
    latency_candles:    int   = 1

    @property
    def pnl(self) -> float:
        """P&L total incluindo saída parcial."""
        if self.side == "long":
            gross = (self.exit_price - self.entry_price) * self.quantity * self.fill_ratio
        else:
            gross = (self.entry_price - self.exit_price) * self.quantity * self.fill_ratio
        return gross - self.entry_fee - self.exit_fee + self.partial_pnl

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.pnl / cost if cost > 0 else 0.0

    @property
    def total_cost_pct(self) -> float:
        notional = self.entry_price * self.quantity
        return (
            (self.entry_fee + self.exit_fee + self.spread_cost + self.market_impact)
            / notional if notional > 0 else 0.0
        )


# ── BacktestResult ────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Resultado agregado de um backtest."""
    symbol:          str
    strategy_id:     str
    start_date:      datetime
    end_date:        datetime
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
        return self.winning_trades / self.total_trades if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.entry_fee + t.exit_fee for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage for t in self.trades)

    @property
    def total_spread_cost(self) -> float:
        return sum(t.spread_cost for t in self.trades)

    @property
    def final_capital(self) -> float:
        return self.initial_capital + self.total_pnl

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.initial_capital if self.initial_capital > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        wins   = sum(t.pnl for t in self.trades if t.pnl > 0)
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
        return self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss

    @property
    def avg_fill_ratio(self) -> float:
        return sum(t.fill_ratio for t in self.trades) / len(self.trades) if self.trades else 1.0

    @property
    def running_mode_exits(self) -> int:
        return sum(1 for t in self.trades if t.had_running_mode)

    @property
    def avg_candles_held(self) -> float:
        return sum(t.candles_held for t in self.trades) / len(self.trades) if self.trades else 0.0

    def summary(self) -> dict:
        pnls = [t.pnl_pct * 100 for t in self.trades]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Distribuição — detecta convexidade
        best3  = sorted(pnls, reverse=True)[:3]
        worst3 = sorted(pnls)[:3]

        return {
            "symbol":             self.symbol,
            "strategy_id":        self.strategy_id,
            "total_trades":       self.total_trades,
            "win_rate":           f"{self.win_rate:.1%}",
            "profit_factor":      f"{self.profit_factor:.2f}",
            "total_pnl":          f"${self.total_pnl:.2f}",
            "total_return":       f"{self.total_return_pct:.2%}",
            "expectancy":         f"${self.expectancy:.2f}",
            "total_fees":         f"${self.total_fees:.2f}",
            "total_slippage":     f"${self.total_slippage:.2f}",
            "avg_fill_ratio":     f"{self.avg_fill_ratio:.1%}",
            "avg_win":            f"${self.avg_win:.2f}",
            "avg_loss":           f"${self.avg_loss:.2f}",
            "avg_holding_candles": f"{self.avg_candles_held:.1f}h",
            "running_mode_exits": self.running_mode_exits,
            # Distribuição de retornos (convexidade)
            "best_3_trades_pct":  [f"{v:.2f}%" for v in best3],
            "worst_3_trades_pct": [f"{v:.2f}%" for v in worst3],
            "avg_win_pct":        f"{sum(wins)/len(wins):.2f}%" if wins else "–",
            "avg_loss_pct":       f"{sum(losses)/len(losses):.2f}%" if losses else "–",
        }


# ── SimulatedFillEngine ───────────────────────────────────────────────────────

class SimulatedFillEngine:
    """
    Simulação realista de execução de ordens para backtesting.
    Componentes: fees, spread, slippage, latência, partial fills.
    """

    SPREAD_BASE       = 0.0002
    SPREAD_ATR_FACTOR = 0.15
    SPREAD_MAX        = 0.002
    IMPACT_BASE       = 0.0001
    IMPACT_VOL_FACTOR = 0.50
    FILL_PROB_BASE    = 0.80
    FILL_PROB_VOL_MIN = 0.50

    def __init__(
        self,
        seed: int | None = 42,
        fee_tier: str = "standard",
    ) -> None:
        self._rng = random.Random(seed)
        fees = {
            "standard": (MAKER_FEE, TAKER_FEE),
            "vip1":     (0.0008,    0.0010),
            "vip2":     (0.0006,    0.0008),
        }
        self._maker_fee, self._taker_fee = fees.get(fee_tier, fees["standard"])

    def simulate_entry(
        self,
        signal:            Signal,
        execution_candle:  Candle,
        signal_candle:     Candle,
        capital:           float,
        position_size_pct: float,
    ) -> "BacktestTrade | None":
        fill_ratio = self._simulate_fill(execution_candle, signal_candle)
        if fill_ratio == 0.0:
            return None

        mid = execution_candle.open
        rel_atr = (signal_candle.high - signal_candle.low) / signal_candle.close \
                  if signal_candle.close > 0 else 0.001
        spread = min(self.SPREAD_BASE + self.SPREAD_ATR_FACTOR * rel_atr, self.SPREAD_MAX)
        spread_cost_pct = spread / 2

        notional      = capital * position_size_pct
        vol_notional  = execution_candle.volume * execution_candle.close
        participation = notional / vol_notional if vol_notional > 0 else 0.01
        impact_pct    = min(self.IMPACT_BASE + self.IMPACT_VOL_FACTOR * participation, 0.005)
        noise_factor  = self._rng.uniform(0.5, 1.5)
        actual_impact = impact_pct * noise_factor

        total_adverse = spread_cost_pct + actual_impact
        if signal.direction == SignalDirection.LONG:
            entry_price = mid * (1 + total_adverse)
        else:
            entry_price = mid * (1 - total_adverse)

        actual_notional = notional * fill_ratio
        quantity        = actual_notional / entry_price if entry_price > 0 else 0.0
        entry_fee       = actual_notional * self._taker_fee

        return BacktestTrade(
            signal=signal,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            side=signal.direction.value,
            entry_fee=entry_fee,
            exit_fee=0.0,
            slippage=actual_notional * actual_impact,
            fill_ratio=fill_ratio,
            entry_time=execution_candle.timestamp,
            exit_time=None,
            entry_regime=getattr(signal, "regime", ""),
            spread_cost=actual_notional * spread_cost_pct,
            market_impact=actual_notional * actual_impact,
        )

    def simulate_exit(
        self,
        trade:        "BacktestTrade",
        exit_candle:  Candle,
        qty_override: float | None = None,
    ) -> float:
        """
        Simula saída e retorna o P&L da saída.
        qty_override: para saídas parciais (quantidade a vender).
        Modifica trade.exit_price, trade.exit_fee, trade.exit_time.
        """
        qty = qty_override or (trade.quantity * trade.fill_ratio)

        rel_atr = (exit_candle.high - exit_candle.low) / exit_candle.close \
                  if exit_candle.close > 0 else 0.001
        spread = min(self.SPREAD_BASE + self.SPREAD_ATR_FACTOR * rel_atr, self.SPREAD_MAX)

        exit_base     = exit_candle.open
        vol_notional  = exit_candle.volume * exit_candle.close
        exit_notional = qty * exit_base
        participation = exit_notional / vol_notional if vol_notional > 0 else 0.005
        impact_pct    = min(self.IMPACT_BASE + self.IMPACT_VOL_FACTOR * participation, 0.003)
        noise_factor  = self._rng.uniform(0.5, 1.5)
        total_adverse = (spread / 2) + impact_pct * noise_factor

        if trade.side == "long":
            exit_price = exit_base * (1 - total_adverse)
        else:
            exit_price = exit_base * (1 + total_adverse)

        exit_fee = exit_notional * self._taker_fee

        # P&L desta saída
        if trade.side == "long":
            gross_pnl = (exit_price - trade.entry_price) * qty
        else:
            gross_pnl = (trade.entry_price - exit_price) * qty
        net_pnl = gross_pnl - exit_fee

        # Atualiza trade (última saída sobrescreve exit_price)
        trade.exit_price  = exit_price
        trade.exit_fee   += exit_fee
        trade.exit_time   = exit_candle.timestamp
        trade.slippage   += exit_notional * impact_pct * noise_factor
        trade.spread_cost += exit_notional * (spread / 2)

        return net_pnl

    def _simulate_fill(self, execution_candle: Candle, signal_candle: Candle) -> float:
        vol_ratio = execution_candle.volume / (signal_candle.volume + 1e-9)
        fill_prob = max(
            min(self.FILL_PROB_BASE * min(vol_ratio, 1.5), 0.95),
            self.FILL_PROB_VOL_MIN,
        )
        if self._rng.random() > fill_prob:
            return 0.0
        alpha = 2.0 + min(vol_ratio, 2.0)
        return self._rng.betavariate(alpha, 1.0)


# ── BacktestEngine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Roda uma estratégia contra candles históricos com simulação realista.

    Exit logic v2 (Fase A):
      - Regime-aware SL/TP
      - TP conversion em TREND_EXPANSION → running mode
      - Trailing conviction-adaptive (proxy via SMA + ATR)
      - Conviction proxy substitui timeout

    Sizing v2 (Fase B):
      - Usa signal.kelly_fraction (calculado pela SizingEngine na estratégia)
      - Fallback para position_size_pct se kelly não disponível
    """

    def __init__(
        self,
        strategy:          BaseStrategy,
        symbol:            str,
        initial_capital:   float = 10000.0,
        position_size_pct: float = 0.10,   # fallback se kelly não disponível
        seed:              int   = 42,
        fee_tier:          str   = "standard",
    ) -> None:
        self._strategy          = strategy
        self._symbol            = symbol
        self._capital           = initial_capital
        self._position_size_pct = position_size_pct
        self._fill_engine = SimulatedFillEngine(seed=seed, fee_tier=fee_tier)

    async def run(
        self,
        candles: list[Candle],
        warmup:  int = 21,
    ) -> BacktestResult:
        """
        Backtest completo com exit logic v2.

        Fluxo por candle i:
          1. Se posição aberta: avalia exit state (SL, TP, trailing, conviction)
             → saída parcial ou total
          2. Se sem posição: avalia sinal
          3. Entra no open do candle i+1 (latência 1 candle)
        """
        if len(candles) < warmup + 2:
            raise ValueError(
                f"Precisa de ao menos {warmup + 2} candles, recebeu {len(candles)}"
            )

        result = BacktestResult(
            symbol=self._symbol,
            strategy_id=self._strategy.strategy_id,
            start_date=candles[warmup].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=self._capital,
        )

        capital    = self._capital
        open_trade: BacktestTrade | None = None
        pos_state:  _PositionState | None = None

        for i in range(warmup, len(candles) - 1):
            candle      = candles[i]
            next_candle = candles[i + 1]
            history     = candles[:i + 1]    # oldest first
            newest_first = list(reversed(history))

            # ── 1. Gerenciar posição aberta ──────────────────────────────────
            if open_trade and pos_state:
                pos_state.candles_held += 1

                result_exit = self._evaluate_position(
                    trade=open_trade,
                    pos=pos_state,
                    candle=candle,
                    next_candle=next_candle,
                    history=newest_first,
                    capital=capital,
                )

                if result_exit == "partial":
                    # Saída parcial: 50% — calcula P&L e continua
                    qty_partial = round(pos_state.quantity * 0.50 * open_trade.fill_ratio, 8)
                    partial_pnl = self._fill_engine.simulate_exit(
                        open_trade, next_candle, qty_override=qty_partial
                    )
                    open_trade.partial_pnl += partial_pnl
                    pos_state.partial_done  = True
                    pos_state.qty_remaining = round(pos_state.qty_remaining - qty_partial, 8)
                    logger.debug(
                        "Partial exit %s qty=%.4f pnl=%.2f",
                        self._symbol, qty_partial, partial_pnl,
                    )

                elif result_exit in ("full", "stop", "trailing", "conviction",
                                     "backstop", "emergency"):
                    qty_rem = pos_state.qty_remaining * open_trade.fill_ratio
                    self._fill_engine.simulate_exit(
                        open_trade, next_candle, qty_override=qty_rem
                    )
                    open_trade.exit_reason      = result_exit
                    open_trade.had_running_mode = pos_state.running_mode
                    open_trade.candles_held     = pos_state.candles_held
                    capital += open_trade.pnl
                    result.trades.append(open_trade)
                    logger.debug(
                        "Exit [%s] %s @ %.4f  pnl=%.2f  cap=%.2f  held=%dh  run=%s",
                        result_exit, self._symbol,
                        open_trade.exit_price, open_trade.pnl,
                        capital, pos_state.candles_held,
                        "YES" if pos_state.running_mode else "no",
                    )
                    open_trade = None
                    pos_state  = None

            # ── 2. Avaliar novo sinal ────────────────────────────────────────
            if open_trade is None:
                # Vol state simplificado (sem Redis) para o SizingEngine
                vol_state_simple = _vol_state_from_candles(newest_first)

                ctx = StrategyContext(
                    symbol=self._symbol,
                    candles_1h=newest_first,
                    candles_6h=[],
                    candles_30m=newest_first,
                    ticker=None,
                    portfolio_value=capital,
                    extra={
                        "vol_state":    vol_state_simple,
                        "model_health": None,   # sem dados em backtest = neutro
                        "meta_regime":  None,
                        "futures_flow": None,
                        "relative_strength": None,
                        "news_sentiment":    None,
                    },
                )
                signal = await self._strategy.evaluate(ctx)

                if signal is not None:
                    # Sizing: usa kelly_fraction da SizingEngine (Fase B)
                    kelly = getattr(signal, "kelly_fraction", None) or self._position_size_pct

                    trade = self._fill_engine.simulate_entry(
                        signal=signal,
                        execution_candle=next_candle,
                        signal_candle=candle,
                        capital=capital,
                        position_size_pct=kelly,
                    )
                    if trade:
                        atr = _calc_atr(newest_first)
                        regime = getattr(signal, "regime", "MEAN_REVERTING_CHOP")
                        pos_state = _PositionState(
                            entry_regime=regime,
                            atr=atr,
                            stop_loss=0.0,
                            take_profit=0.0,
                            quantity=trade.quantity,
                            qty_remaining=trade.quantity,
                        )
                        pos_state.sl_tp_from_regime(trade.entry_price, atr, regime)
                        open_trade = trade
                        logger.debug(
                            "Entry %s @ %.4f  regime=%s  sl=%.4f  tp=%.4f  kelly=%.1f%%",
                            self._symbol, trade.entry_price, regime,
                            pos_state.stop_loss, pos_state.take_profit, kelly * 100,
                        )

        # ── 3. Fechar posição aberta ao fim dos dados ────────────────────────
        if open_trade and pos_state and len(candles) > 0:
            qty_rem = pos_state.qty_remaining * open_trade.fill_ratio
            self._fill_engine.simulate_exit(open_trade, candles[-1], qty_override=qty_rem)
            open_trade.exit_reason      = "end_of_data"
            open_trade.had_running_mode = pos_state.running_mode
            open_trade.candles_held     = pos_state.candles_held
            capital += open_trade.pnl
            result.trades.append(open_trade)

        logger.info(
            "Backtest %s: %d trades  WinR=%.1f%%  PnL=%.2f  "
            "PF=%.2f  RunMode=%d  AvgHold=%.1fh",
            self._symbol,
            result.total_trades,
            result.win_rate * 100,
            result.total_pnl,
            result.profit_factor,
            result.running_mode_exits,
            result.avg_candles_held,
        )
        return result

    # ── Avaliação de posição aberta ───────────────────────────────────────────

    def _evaluate_position(
        self,
        trade:       BacktestTrade,
        pos:         _PositionState,
        candle:      Candle,
        next_candle: Candle,
        history:     list[Candle],   # newest first
        capital:     float,
    ) -> str | None:
        """
        Avalia o estado de uma posição aberta em cada candle.

        Retorna:
          "stop"       — stop loss atingido
          "trailing"   — trailing stop atingido
          "full"       — take profit (outros regimes)
          "partial"    — saída parcial (50%), posição continua
          "conviction" — conviction proxy colapso
          "backstop"   — 30 dias (720 candles)
          "emergency"  — regime PANIC/BEAR
          None         — manter posição
        """
        price = candle.close   # proxy: avaliamos no close de cada candle

        # ── 1. Emergência ─────────────────────────────────────────────────────
        current_regime = _detect_regime(history)
        if current_regime in REGIMES_EMERGENCY:
            return "emergency"

        # ── 2. Stop Loss (safety net absoluta) ───────────────────────────────
        effective_sl = pos.trailing_stop if pos.trailing_stop else pos.stop_loss
        if candle.low <= effective_sl:
            return "stop" if not pos.trailing_stop else "trailing"

        # ── 3. Take Profit / TP Conversion ────────────────────────────────────
        if candle.high >= pos.take_profit and not pos.tp_converted:
            if pos.entry_regime in REGIMES_CONVERT_TP:
                # TREND_EXPANSION: converte em running mode
                if not pos.partial_done:
                    # Retorna "partial" — o loop fará a saída parcial
                    pos.tp_converted = True   # sinaliza conversão
                    pos.running_mode = True
                    # Trailing largo ao converter
                    running_trail = round(price - pos.atr * 2.5, 6)
                    pos.trailing_stop = running_trail
                    pos.trailing_activated = True
                    pos.take_profit = price * 99   # remove teto
                    return "partial"
                else:
                    # Partial já foi feita antes (em 3R) — apenas converte
                    pos.tp_converted = True
                    pos.running_mode = True
                    pos.take_profit  = price * 99
                    if not pos.trailing_activated:
                        pos.trailing_stop = round(price - pos.atr * 2.5, 6)
                        pos.trailing_activated = True
                    return None
            else:
                # Outros regimes: saída total no TP
                return "full"

        # ── 4. Backstop absoluto ──────────────────────────────────────────────
        if pos.candles_held >= ABSOLUTE_BACKSTOP_CANDLES:
            return "backstop"

        # ── 5. Saída parcial por R (antes do TP) ─────────────────────────────
        partial_r = REGIME_PARTIAL_R.get(pos.entry_regime, 2.5)
        r_value   = trade.entry_price - pos.stop_loss
        if r_value > 0:
            partial_price = trade.entry_price + r_value * partial_r
            if not pos.partial_done and candle.high >= partial_price:
                return "partial"

        # ── 6. Conviction proxy (substitui timeout) ───────────────────────────
        conviction = _conviction_proxy(history, trade.entry_price)

        if conviction < CONVICTION_ALERT:
            pos.low_conviction_streak += 1
            if pos.low_conviction_streak >= LOW_CONVICTION_STREAK_EXIT:
                return "conviction"
        else:
            pos.low_conviction_streak = 0

        # Anti-bag-holding P1: trade negativo + conviction < 50%
        if price < trade.entry_price and conviction < 50.0:
            return "conviction"

        # Anti-bag-holding P3: queda > 8% + conviction < 60%
        if price < trade.entry_price * 0.92 and conviction < 60.0:
            return "conviction"

        # ── 7. Trailing stop update ───────────────────────────────────────────
        trail_r    = REGIME_TRAIL_ACTIVATE.get(pos.entry_regime, 1.0)
        r_value    = trade.entry_price - pos.stop_loss
        trail_trigger = trade.entry_price + r_value * trail_r

        # ATR multiplier baseado na conviction (proxy)
        if conviction >= 85:
            atr_mult = 2.5
        elif conviction >= 70:
            atr_mult = 2.0
        elif conviction >= 50:
            atr_mult = 1.0
        else:
            atr_mult = 0.5
        if pos.running_mode:
            atr_mult += 0.5

        if price >= trail_trigger or pos.running_mode:
            new_trail = round(price - pos.atr * atr_mult, 6)
            current   = pos.trailing_stop or 0.0
            if not pos.trailing_activated:
                pos.trailing_stop = new_trail
                pos.trailing_activated = True
            elif new_trail > current:
                pos.trailing_stop = new_trail
            elif conviction < CONVICTION_HOLD:
                # Aperta trailing quando conviction degrada
                tighter = round(price - pos.atr * atr_mult, 6)
                if tighter > current:
                    pos.trailing_stop = tighter

        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calc_atr(candles: list[Candle], period: int = 14) -> float:
    """ATR dos últimos `period` candles (newest first)."""
    if len(candles) < period:
        return candles[0].close * 0.015 if candles else 100.0
    trs = []
    for i in range(period):
        c = candles[i]
        prev_close = candles[i + 1].close if i + 1 < len(candles) else c.close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


def _detect_regime(candles: list[Candle]) -> str:
    """Detecção de regime simplificada — espelha momentum_strategy._detect_regime_1h."""
    if len(candles) < 20:
        return "MEAN_REVERTING_CHOP"
    closes  = [c.close  for c in candles[:20]]
    volumes = [c.volume for c in candles[:20]]
    sma5    = sum(closes[:5]) / 5
    sma20   = sum(closes[:20]) / 20
    avg_vol = sum(volumes) / len(volumes)
    last_vol = volumes[0]

    if len(closes) >= 2 and closes[1] > 0:
        if (closes[0] - closes[1]) / closes[1] < -0.05:
            return "PANIC_LIQUIDATION"

    if sma5 > sma20:
        if last_vol > avg_vol * 1.2:
            return "TREND_EXPANSION"
        if last_vol < avg_vol * 0.8:
            return "TREND_EXHAUSTION"
        return "VOLATILITY_COMPRESSION"

    if len(closes) >= 11 and closes[10] > 0:
        if (closes[0] - closes[10]) / closes[10] < -0.02:
            return "BEAR_TREND"

    highs = [c.high for c in candles[:5]]
    lows  = [c.low  for c in candles[:5]]
    atr5  = sum(hi - lo for hi, lo in zip(highs, lows, strict=True)) / 5
    if closes[0] > 0 and atr5 / closes[0] > 0.030:
        return "HIGH_CORRELATION_RISK"

    return "MEAN_REVERTING_CHOP"


def _conviction_proxy(candles: list[Candle], entry_price: float) -> float:
    """
    Proxy de conviction 0–100 baseado em candles (sem Redis/cache).
    Replica EXATAMENTE os 6 componentes e pesos do HoldEngine (hold_engine.py).

    Componentes e pesos (idênticos ao WEIGHTS dict do HoldEngine):
      1. trend_persistence (25%): SMA score contínuo [-3%,+3%] + HH/HL (60%/40%)
      2. relative_strength (20%): fallback neutro 0.5 (sem M7 em backtest)
      3. vol_health        (20%): ATR ratio → mapeado em estados (VOL_STATE_HEALTH)
      4. breadth           (15%): fallback neutro 0.5 (sem MetaRegime em backtest)
      5. volume_behavior   (10%): volume vs média + direção relativa ao entry_price
      6. distribution      (10%): detecção de velas bearish + volume alto

    Fallbacks neutros (0.5) para componentes que dependem de dados externos
    (relative_strength e breadth) são o comportamento correto — identical ao
    HoldEngine quando Redis não retorna dados.
    """
    if len(candles) < 20:
        return 70.0   # sem dados suficientes = neutro HOLD

    closes  = [c.close  for c in candles[:25]]
    highs   = [c.high   for c in candles[:25]]
    lows    = [c.low    for c in candles[:25]]
    volumes = [c.volume for c in candles[:20]]
    opens   = [c.open   for c in candles[:4]]

    # 1. trend_persistence (25%) — cópia exata de HoldEngine._trend_persistence()
    sma5  = sum(closes[:5]) / 5
    sma20 = sum(closes[:20]) / 20
    margin = (sma5 - sma20) / sma20 if sma20 > 0 else 0.0
    sma_score = min(max((margin + 0.03) / 0.06, 0.0), 1.0)
    hh = sum(1 for i in range(min(4, len(highs)-1)) if highs[i] > highs[i+1])
    hl = sum(1 for i in range(min(4, len(lows)-1))  if lows[i]  > lows[i+1])
    structure_score = (hh + hl) / 8
    trend_persistence = sma_score * 0.60 + structure_score * 0.40

    # 2. relative_strength (20%) — neutro 0.5 (sem M7 no backtest, igual ao fallback do HoldEngine)
    relative_strength = 0.5

    # 3. vol_health (20%) — ATR ratio → estado → VOL_STATE_HEALTH (cópia do HoldEngine)
    _VOL_STATE_HEALTH = {
        "EXPANDING": 0.90, "TREND": 0.80, "COMPRESSED": 0.50,
        "MEAN_REVERTING": 0.30, "CHAOTIC": 0.05, "UNKNOWN": 0.50,
    }
    atr_now  = sum(highs[i] - lows[i] for i in range(min(5, len(highs)))) / 5
    atr_ref  = (sum(highs[i] - lows[i] for i in range(5, 20)) / 15
                if len(highs) >= 20 else atr_now)
    atr_ratio = atr_now / atr_ref if atr_ref > 0 else 1.0
    atr_pct   = atr_now / closes[0] if closes[0] > 0 else 0.01
    n_dir = min(10, len(closes) - 1)
    ups   = sum(1 for i in range(n_dir) if closes[i] > closes[i+1])
    dir_c = max(ups, n_dir - ups) / n_dir if n_dir > 0 else 0.5
    bb_width = (max(closes[:20]) - min(closes[:20])) / closes[0] if len(closes) >= 20 else 0.02
    if atr_pct > 0.025 and dir_c < 0.45:
        _state = "CHAOTIC"
    elif atr_ratio > 1.30 and dir_c > 0.55:
        _state = "EXPANDING"
    elif atr_pct < 0.008 or bb_width < 0.015:
        _state = "COMPRESSED"
    elif dir_c > 0.60 and 0.008 <= atr_pct <= 0.025:
        _state = "TREND"
    else:
        _state = "MEAN_REVERTING"
    vol_health = _VOL_STATE_HEALTH[_state]

    # 4. breadth (15%) — neutro 0.5 (sem MetaRegime no backtest, igual ao fallback)
    breadth = 0.5

    # 5. volume_behavior (10%) — cópia exata de HoldEngine._volume_behavior()
    avg_vol    = sum(volumes[1:]) / max(len(volumes) - 1, 1)
    curr_vol   = volumes[0] if volumes else avg_vol
    vol_ratio  = curr_vol / avg_vol if avg_vol > 0 else 1.0
    price_up   = closes[0] >= entry_price
    if price_up and vol_ratio > 1.10:
        volume_behavior = 0.85
    elif price_up and vol_ratio >= 0.80:
        volume_behavior = 0.65
    elif not price_up and vol_ratio > 1.20:
        volume_behavior = 0.15
    elif not price_up:
        volume_behavior = 0.35
    else:
        volume_behavior = 0.50

    # 6. distribution (10%) — cópia exata de HoldEngine._distribution()
    avg_vol4 = sum(volumes[1:4]) / max(len(volumes[1:4]), 1)
    dist_signals = sum(
        1 for i in range(min(3, len(closes)))
        if closes[i] < opens[i] and volumes[i] > avg_vol4 * 1.15
    )
    distribution = round(1.0 - (dist_signals / 3) * 0.90, 4)

    # Score final — pesos idênticos ao WEIGHTS dict do HoldEngine
    raw = (trend_persistence * 0.25 +
           relative_strength * 0.20 +
           vol_health        * 0.20 +
           breadth           * 0.15 +
           volume_behavior   * 0.10 +
           distribution      * 0.10)

    return round(max(0.0, min(100.0, raw * 100)), 1)


def _vol_state_from_candles(candles: list[Candle]) -> dict:
    """
    Deriva vol state simplificado dos candles para o SizingEngine em backtest.
    Sem Redis — usa apenas ATR ratio vs histórico recente.
    """
    if len(candles) < 10:
        return {"state": "UNKNOWN", "m8_score": 0.5}

    atr_now = sum(c.high - c.low for c in candles[:5]) / 5
    atr_ref = sum(c.high - c.low for c in candles[5:20]) / min(15, len(candles) - 5)
    ratio   = atr_now / atr_ref if atr_ref > 0 else 1.0

    # Scores espelham volatility_state.STATE_M8_SCORE (fonte autoritativa do M8)
    if ratio > 2.0:
        state, score = "CHAOTIC", 0.20
    elif ratio > 1.3:
        state, score = "EXPANDING", 0.80
    elif ratio > 0.9:
        state, score = "TREND", 0.70
    elif ratio > 0.6:
        state, score = "COMPRESSED", 0.65
    else:
        state, score = "MEAN_REVERTING", 0.35

    return {
        "state":    state,
        "m8_score": score,
        "metrics":  {"atr_pct": round(atr_now / candles[0].close, 4) if candles[0].close else 0},
    }
