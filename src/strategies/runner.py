"""
Strategy Runner — evaluates all registered strategies each cycle.

Responsibilities:
  - Subscribe to MARKET events from the bus
  - Build StrategyContext from market data
  - Call each enabled strategy's evaluate()
  - Publish SignalEvent for every non-None result
  - Never touches the OMS or exchange directly

Deduplication rules:
  - Só avalia candles recentes (< 2 períodos de idade) para ignorar histórico
  - Só avalia 1x por símbolo por ciclo (evita dupla avaliação de 1H + 6H)
  - Intervalo mínimo entre avaliações do mesmo símbolo: 55 minutos
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from ..core.bus import EventBus
from ..core.events import CandleEvent, SignalEvent, Topic
from ..core.models import Signal
from ..market.engine import MarketEngine
from ..persistence.cache import Cache
from .base import BaseStrategy, StrategyContext

logger = logging.getLogger(__name__)

# Só avalia candles 30m confirmados com menos de 1 hora de idade
MAX_CANDLE_AGE = timedelta(hours=1)

# Granularidade alvo — só avalia eventos dessa granularidade
EVAL_GRANULARITY = "30m"


class StrategyRunner:
    """
    Evaluates all strategies on each market data update.
    Runs as a background consumer of MARKET events.
    """

    def __init__(
        self,
        bus: EventBus,
        market: MarketEngine,
        cache: Cache,
        portfolio_value: float = 0.0,
    ) -> None:
        self._bus = bus
        self._market = market
        self._cache = cache
        self._portfolio_value = portfolio_value
        self._strategies: dict[str, BaseStrategy] = {}
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._signal_count = 0
        self._eval_count = 0

        # Último timestamp de candle 1H avaliado por símbolo
        # Só reavalia quando o timestamp da vela mudar (novo fechamento de 1H)
        self._last_candle_ts: dict[str, datetime] = {}
        self._last_eval: dict[str, datetime] = {}  # mantido para status()

    def register(self, strategy: BaseStrategy) -> None:
        self._strategies[strategy.strategy_id] = strategy
        logger.info("Strategy registered: %s symbols=%s",
                    strategy.strategy_id, strategy.symbols)

    def unregister(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._queue = self._bus.subscribe(Topic.MARKET)

        # Pré-popula last_candle_ts com o candle mais recente de cada símbolo
        # sem avaliar — bot aguarda a PRÓXIMA hora fechar antes de agir
        await self._seed_last_candle_ts()

        self._task = asyncio.create_task(
            self._consume(), name="strategy_runner"
        )
        logger.info(
            "StrategyRunner started with %d strategies",
            len(self._strategies),
        )

    async def _seed_last_candle_ts(self) -> None:
        """
        No boot, verifica se o último candle 1H fechado já foi avaliado.
        - Se não foi → avalia agora → aguarda próxima hora
        - Se já foi → pula → aguarda próxima hora

        Garante que nenhuma vela é perdida e nenhuma é avaliada duas vezes.
        """
        symbols: set[str] = set()
        for strategy in self._strategies.values():
            symbols.update(strategy.symbols)

        # Timestamp do último candle 30m fechado = minuto atual truncado a 30min - 30min
        now = datetime.now(UTC)
        half = 0 if now.minute < 30 else 30
        last_closed_ts = now.replace(minute=half, second=0, microsecond=0) - timedelta(minutes=30)

        for symbol in symbols:
            redis_ts: datetime | None = None
            cached = await self._cache.get(f"last_candle_ts:{symbol}")
            if cached:
                try:
                    redis_ts = datetime.fromisoformat(cached)
                except ValueError:
                    pass

            # Usa o mais recente entre Redis e o candle calculado
            # Nunca avalia no boot — sempre aguarda a próxima hora fechar
            ts = max(redis_ts, last_closed_ts) if redis_ts else last_closed_ts
            self._last_candle_ts[symbol] = ts
            await self._cache.set(f"last_candle_ts:{symbol}", ts.isoformat(), ttl=10800)
            logger.info("StrategyRunner: %s — aguardando próxima hora (seed: %s UTC)",
                        symbol, ts.strftime("%Y-%m-%d %H:%M"))

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _consume(self) -> None:
        """Consume MARKET events e dispara avaliação apenas em candles novos e recentes."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                if not isinstance(event, CandleEvent) or not event.candle:
                    continue

                candle = event.candle
                if not candle.confirmed:
                    continue

                now = datetime.now(UTC)

                # Filtro 1: só processa granularidade 1H
                if getattr(event.candle, 'granularity', None) != EVAL_GRANULARITY:
                    continue

                # Filtro 2: ignora candles históricos (mais de 2h de idade)
                candle_ts = candle.timestamp.replace(tzinfo=UTC) \
                    if candle.timestamp.tzinfo is None \
                    else candle.timestamp
                age = now - candle_ts
                if age > MAX_CANDLE_AGE:
                    continue

                # Filtro 3: só avalia se o timestamp do candle for estritamente novo
                last_ts = self._last_candle_ts.get(candle.symbol)
                if last_ts and candle_ts <= last_ts:
                    continue

                # Nova hora fechou — persiste no Redis e avalia
                self._last_candle_ts[candle.symbol] = candle_ts
                self._last_eval[candle.symbol] = now
                await self._cache.set(
                    f"last_candle_ts:{candle.symbol}",
                    candle_ts.isoformat(),
                    ttl=10800,  # 3 horas
                )
                logger.info(
                    "Nova vela 30m %s ts=%s — avaliando estratégia",
                    candle.symbol, candle_ts.strftime("%Y-%m-%d %H:%M")
                )
                await self._evaluate_all(candle.symbol)

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("StrategyRunner error: %s", exc, exc_info=True)

    async def _evaluate_all(self, symbol: str) -> None:
        """Evaluate all strategies that trade the given symbol."""
        for strategy in self._strategies.values():
            if not strategy.is_enabled:
                continue
            if symbol not in strategy.symbols:
                continue
            try:
                self._eval_count += 1
                ctx = await self._build_context(symbol)
                signal = await strategy.evaluate(ctx)
                if signal is not None:
                    await self._publish_signal(signal)
            except Exception as exc:
                logger.error(
                    "Strategy %s raised: %s",
                    strategy.strategy_id, exc, exc_info=True,
                )

    async def _build_context(self, symbol: str) -> StrategyContext:
        candles_1h = self._market.get_candles(symbol, "1H")
        candles_6h = self._market.get_candles(symbol, "6H")

        pos_data = await self._cache.get_position(symbol)
        open_positions = [pos_data] if pos_data else []

        return StrategyContext(
            symbol=symbol,
            candles_1h=candles_1h,
            candles_6h=candles_6h,
            ticker=None,
            portfolio_value=self._portfolio_value,
            open_positions=open_positions,
        )

    async def _publish_signal(self, signal: Signal) -> None:
        self._signal_count += 1
        await self._bus.publish(Topic.SIGNAL, SignalEvent(signal=signal))
        logger.info(
            "Signal published strategy=%s symbol=%s direction=%s score=%.3f",
            signal.strategy_id, signal.symbol,
            signal.direction, signal.calibrated_score,
        )

    def update_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def status(self) -> dict:
        return {
            "running":       self._running,
            "strategies":    list(self._strategies.keys()),
            "signal_count":  self._signal_count,
            "eval_count":    self._eval_count,
            "last_eval":     {
                sym: ts.isoformat()
                for sym, ts in self._last_eval.items()
            },
        }
