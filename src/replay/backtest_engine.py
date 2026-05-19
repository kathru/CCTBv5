"""
Backtest Engine — simulação realista de execução para validação de estratégias.

Princípio fundamental: mesmo código de estratégia que o live trading.
Sem "backtest mode" — estratégias recebem StrategyContext e retornam Signal.

Componentes de simulação realista:
  1. Fees         — OKX Spot fee schedule (Maker/Taker por tier)
  2. Spread       — Proporcional ao ATR (alarga em volatilidade alta)
  3. Slippage     — Market impact proporcional a notional/volume
  4. Latency      — Sinal no close do candle N → execução no open do candle N+1
  5. Partial fills — Probabilidade baseada em volume relativo (não fixa)

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
# https://www.okx.com/fees
MAKER_FEE = 0.0010   # 0.10% maker (passive limit)
TAKER_FEE = 0.0015   # 0.15% taker (market order) ← corrigido: 0.15% não 0.40%

# Paper trading usa TAKER em todas as ordens (mercado)
PAPER_FEE = TAKER_FEE


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
    slippage:     float        # slippage total em $
    fill_ratio:   float        # 1.0 = cheio, 0.6 = 60% parcial
    entry_time:   datetime
    exit_time:    datetime | None

    # Componentes de custo detalhados (para análise)
    spread_cost:   float = 0.0
    market_impact: float = 0.0
    latency_candles: int = 1   # sempre 1 no backtest (execução no próximo candle)

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

    @property
    def total_cost_pct(self) -> float:
        """Custo total de execução como % do notional."""
        notional = self.entry_price * self.quantity
        return (self.entry_fee + self.exit_fee + self.spread_cost + self.market_impact) / notional \
               if notional > 0 else 0.0


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

    def summary(self) -> dict:
        return {
            "symbol":            self.symbol,
            "strategy_id":       self.strategy_id,
            "total_trades":      self.total_trades,
            "win_rate":          f"{self.win_rate:.1%}",
            "profit_factor":     f"{self.profit_factor:.2f}",
            "total_pnl":         f"${self.total_pnl:.2f}",
            "total_return":      f"{self.total_return_pct:.2%}",
            "expectancy":        f"${self.expectancy:.2f}",
            "total_fees":        f"${self.total_fees:.2f}",
            "total_slippage":    f"${self.total_slippage:.2f}",
            "avg_fill_ratio":    f"{self.avg_fill_ratio:.1%}",
            "avg_win":           f"${self.avg_win:.2f}",
            "avg_loss":          f"${self.avg_loss:.2f}",
        }


# ── SimulatedFillEngine ───────────────────────────────────────────────────────

class SimulatedFillEngine:
    """
    Simulação realista de execução de ordens para backtesting.

    Componentes:
      1. Fees         — OKX maker/taker por tipo de ordem
      2. Spread       — bid/ask gap proporcional ao ATR do candle
      3. Slippage     — market impact: f(notional / volume_notional)
      4. Latency      — execução no OPEN do próximo candle (não no close do sinal)
      5. Partial fills — probabilidade proporcional ao volume relativo

    Parâmetros calibrados para OKX Spot em ativos de alta liquidez (BTC/ETH/SOL).
    """

    # Spread base por volatilidade (ATR como % do preço)
    SPREAD_BASE       = 0.0002    # 0.02% em mercado calmo
    SPREAD_ATR_FACTOR = 0.15      # 15% do ATR relativo vira spread adicional
    SPREAD_MAX        = 0.002     # cap: nunca mais que 0.20%

    # Market impact: quanto do volume do candle a ordem consome
    IMPACT_BASE       = 0.0001    # 0.01% base sempre presente
    IMPACT_VOL_FACTOR = 0.50      # 50% da participação no volume vira slippage

    # Fill probability: função do volume relativo
    FILL_PROB_BASE    = 0.80      # 80% base de fill
    FILL_PROB_VOL_MIN = 0.50      # mínimo 50% em candles de volume baixo

    def __init__(
        self,
        seed: int | None = 42,
        fee_tier: str = "standard",    # "standard" | "vip1" | "vip2"
    ) -> None:
        self._rng = random.Random(seed)

        # Fee schedule por tier
        fees = {
            "standard": (MAKER_FEE, TAKER_FEE),
            "vip1":     (0.0008,    0.0010),
            "vip2":     (0.0006,    0.0008),
        }
        self._maker_fee, self._taker_fee = fees.get(fee_tier, fees["standard"])

    # ── Entry ────────────────────────────────────────────────────────────────

    def simulate_entry(
        self,
        signal:            Signal,
        execution_candle:  Candle,   # candle N+1 (próximo após o sinal)
        signal_candle:     Candle,   # candle N (onde o sinal foi gerado)
        capital:           float,
        position_size_pct: float,
    ) -> "BacktestTrade | None":
        """
        Simula entrada no candle N+1 (latência realista).
        O sinal é gerado no close do candle N.
        A ordem é executada no open do candle N+1.
        """
        # ── 1. Partial fill baseado em volume ─────────────────
        fill_ratio = self._simulate_fill(execution_candle, signal_candle)
        if fill_ratio == 0.0:
            return None

        # ── 2. Preço base: open do próximo candle (latência) ──
        mid = execution_candle.open

        # ── 3. Spread proporcional ao ATR ─────────────────────
        rel_atr = (signal_candle.high - signal_candle.low) / signal_candle.close \
                  if signal_candle.close > 0 else 0.001
        spread = min(
            self.SPREAD_BASE + self.SPREAD_ATR_FACTOR * rel_atr,
            self.SPREAD_MAX,
        )
        spread_cost_pct = spread / 2   # metade do spread por lado

        # ── 4. Market impact (slippage) ───────────────────────
        notional       = capital * position_size_pct
        vol_notional   = execution_candle.volume * execution_candle.close
        participation  = notional / vol_notional if vol_notional > 0 else 0.01
        impact_pct     = self.IMPACT_BASE + self.IMPACT_VOL_FACTOR * participation
        impact_pct     = min(impact_pct, 0.005)   # cap em 0.5% por ordem

        # Ruído: variação aleatória ±50% do impacto calculado
        noise_factor = self._rng.uniform(0.5, 1.5)
        actual_impact = impact_pct * noise_factor

        # ── 5. Preço de execução ──────────────────────────────
        total_adverse = spread_cost_pct + actual_impact
        if signal.direction == SignalDirection.LONG:
            entry_price = mid * (1 + total_adverse)  # compra acima do mid
        else:
            entry_price = mid * (1 - total_adverse)  # vende abaixo do mid

        # ── 6. Quantidade e fees ──────────────────────────────
        actual_notional = notional * fill_ratio
        quantity        = actual_notional / entry_price if entry_price > 0 else 0.0
        entry_fee       = actual_notional * self._taker_fee   # market order = taker

        # ── 7. Slippage total em $ (para métricas) ────────────
        slippage_total = actual_notional * actual_impact
        spread_cost    = actual_notional * spread_cost_pct

        logger.debug(
            "Entry %s %s @ %.4f (open=%.4f spread=%.4f%% impact=%.4f%% fill=%.0f%%)",
            signal.direction, signal.symbol,
            entry_price, mid, spread * 100, actual_impact * 100, fill_ratio * 100,
        )

        return BacktestTrade(
            signal=signal,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            side=signal.direction.value,
            entry_fee=entry_fee,
            exit_fee=0.0,
            slippage=slippage_total,
            fill_ratio=fill_ratio,
            entry_time=execution_candle.timestamp,
            exit_time=None,
            spread_cost=spread_cost,
            market_impact=slippage_total,
            latency_candles=1,
        )

    # ── Exit ─────────────────────────────────────────────────────────────────

    def simulate_exit(
        self,
        trade:       "BacktestTrade",
        exit_candle: Candle,
    ) -> "BacktestTrade":
        """
        Simula saída via stop/TP/timeout no candle de saída.
        Saídas (stop loss, take profit) executam no OPEN do candle que atingiu o nível.
        """
        # Saída no open do candle de saída (ou close se stop/TP intrabar)
        if trade.side == "long":
            # Stop hit: executa próximo ao low
            # TP hit: executa próximo ao high
            # Aproximação: usa close com spread/impact
            exit_base = exit_candle.open
        else:
            exit_base = exit_candle.open

        # Spread e impact na saída (geralmente menor — liquidez alta em stops)
        rel_atr = (exit_candle.high - exit_candle.low) / exit_candle.close \
                  if exit_candle.close > 0 else 0.001
        spread = min(self.SPREAD_BASE + self.SPREAD_ATR_FACTOR * rel_atr, self.SPREAD_MAX)

        vol_notional  = exit_candle.volume * exit_candle.close
        exit_notional = trade.quantity * exit_base * trade.fill_ratio
        participation = exit_notional / vol_notional if vol_notional > 0 else 0.005
        impact_pct    = min(self.IMPACT_BASE + self.IMPACT_VOL_FACTOR * participation, 0.003)
        noise_factor  = self._rng.uniform(0.5, 1.5)

        total_adverse = (spread / 2) + impact_pct * noise_factor

        if trade.side == "long":
            exit_price = exit_base * (1 - total_adverse)   # vende abaixo do mid
        else:
            exit_price = exit_base * (1 + total_adverse)   # cobre acima do mid

        exit_fee = exit_notional * self._taker_fee

        trade.exit_price = exit_price
        trade.exit_fee   = exit_fee
        trade.exit_time  = exit_candle.timestamp

        # Acumula slippage e spread da saída
        trade.slippage    += exit_notional * impact_pct * noise_factor
        trade.spread_cost += exit_notional * (spread / 2)

        return trade

    # ── Fill simulation ───────────────────────────────────────────────────────

    def _simulate_fill(
        self,
        execution_candle: Candle,
        signal_candle:    Candle,
    ) -> float:
        """
        Fill ratio baseado em volume relativo.
        Alta liquidez (volume alto) → fill quase certo e completo.
        Baixa liquidez → pode não preencher ou fill parcial.
        """
        # Probabilidade de fill proporcional ao volume relativo
        vol_ratio = execution_candle.volume / (signal_candle.volume + 1e-9)
        fill_prob = min(
            self.FILL_PROB_BASE * min(vol_ratio, 1.5),
            0.95,
        )
        fill_prob = max(fill_prob, self.FILL_PROB_VOL_MIN)

        # Decide se a ordem preenche
        if self._rng.random() > fill_prob:
            return 0.0   # sem fill

        # Quantidade do fill: Beta(3,1) — skewed para fill completo
        # Volume alto → parâmetro alpha maior → fill mais próximo de 1.0
        alpha = 2.0 + min(vol_ratio, 2.0)
        return self._rng.betavariate(alpha, 1.0)


# ── BacktestEngine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Roda uma estratégia contra candles históricos com simulação realista.
    Usa SimulatedFillEngine para execução com fees, spread, slippage, latency e partial fills.
    """

    def __init__(
        self,
        strategy:          BaseStrategy,
        symbol:            str,
        initial_capital:   float = 10000.0,
        position_size_pct: float = 0.10,
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
        Backtest completo com simulação realista.

        Fluxo por candle i:
          1. Verifica se posição aberta atingiu SL/TP (ATR-based)
          2. Se sim: executa saída no open do candle i+1
          3. Se posição fechada: avalia sinal com histórico até candle i
          4. Se sinal: entra no open do candle i+1 (latência 1 candle)

        Args:
          candles: lista de candles confirmados (mais antigo primeiro)
          warmup:  candles iniciais para aquecimento (não evaluados)
        """
        if len(candles) < warmup + 2:
            raise ValueError(f"Precisa de ao menos {warmup + 2} candles, recebeu {len(candles)}")

        result = BacktestResult(
            symbol=self._symbol,
            strategy_id=self._strategy.strategy_id,
            start_date=candles[warmup].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=self._capital,
        )

        capital    = self._capital
        open_trade: BacktestTrade | None = None

        for i in range(warmup, len(candles) - 1):   # -1: precisa do candle N+1
            candle      = candles[i]
            next_candle = candles[i + 1]
            history     = candles[:i + 1]

            # ── 1. Verificar saída da posição aberta ──────────
            if open_trade:
                should_exit, exit_reason = self._check_exit(open_trade, candle, history)
                if should_exit:
                    # Saída no open do PRÓXIMO candle (latência)
                    closed = self._fill_engine.simulate_exit(open_trade, next_candle)
                    capital += closed.pnl
                    result.trades.append(closed)
                    logger.debug(
                        "Exit [%s] @ %.4f  pnl=%.2f  cap=%.2f",
                        exit_reason, closed.exit_price, closed.pnl, capital,
                    )
                    open_trade = None

            # ── 2. Avaliar novo sinal ─────────────────────────
            if open_trade is None:
                candles_newest_first = list(reversed(history))
                ctx = StrategyContext(
                    symbol=self._symbol,
                    candles_1h=candles_newest_first,   # newest first
                    candles_6h=[],
                    candles_30m=candles_newest_first,  # backtest usa mesmos candles
                    ticker=None,
                    portfolio_value=capital,
                )
                signal = await self._strategy.evaluate(ctx)

                if signal is not None:
                    # Entrada no open do PRÓXIMO candle (latência 1 candle)
                    trade = self._fill_engine.simulate_entry(
                        signal=signal,
                        execution_candle=next_candle,
                        signal_candle=candle,
                        capital=capital,
                        position_size_pct=self._position_size_pct,
                    )
                    if trade:
                        open_trade = trade
                        logger.debug(
                            "Entry @ %.4f  fill=%.0f%%  impact=%.4f%%",
                            trade.entry_price,
                            trade.fill_ratio * 100,
                            trade.market_impact / (trade.entry_price * trade.quantity + 1e-9) * 100,
                        )

        # ── 3. Fechar posição aberta no fim dos dados ─────────
        if open_trade and len(candles) > 0:
            closed = self._fill_engine.simulate_exit(open_trade, candles[-1])
            capital += closed.pnl
            result.trades.append(closed)

        logger.info(
            "Backtest %s: %d trades  WinR=%.1f%%  PnL=%.2f  Fees=%.2f  Slip=%.2f",
            self._symbol,
            result.total_trades,
            result.win_rate * 100,
            result.total_pnl,
            result.total_fees,
            result.total_slippage,
        )
        return result

    def _check_exit(
        self,
        trade:   BacktestTrade,
        candle:  Candle,
        history: list[Candle],
    ) -> tuple[bool, str]:
        """
        Verifica se SL ou TP foram atingidos usando ATR dinâmico.
        Retorna (deve_sair, motivo).

        SL = entry - ATR_entry × 1.5
        TP = entry + ATR_entry × 3.0
        Timeout = 8h padrão (CHOP), verificado por contagem de candles
        """
        if trade.side != "long":
            return False, ""

        entry = trade.entry_price

        # ATR estimado: usa candles recentes (últimos 14)
        recent = history[-14:] if len(history) >= 14 else history
        if recent:
            atr = sum(c.high - c.low for c in recent) / len(recent)
        else:
            atr = entry * 0.01

        stop = entry - atr * 1.5
        take = entry + atr * 3.0

        # Candle atingiu o stop?
        if candle.low <= stop:
            return True, "stop_loss"

        # Candle atingiu o take profit?
        if candle.high >= take:
            return True, "take_profit"

        # Timeout: conta candles desde entrada
        hold_candles = sum(
            1 for c in history if c.timestamp >= trade.entry_time
        )
        if hold_candles >= 8:   # 8 candles 1H = 8 horas (CHOP default)
            return True, "timeout"

        return False, ""
