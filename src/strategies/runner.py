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
import json
import logging
from datetime import UTC, datetime, timedelta

from ..core.bus import EventBus
from ..core.events import CandleEvent, SignalEvent, Topic
from ..core.models import Signal
from ..market.engine import MarketEngine
from ..persistence.cache import Cache
from .base import BaseStrategy, StrategyContext
from .weight_engine import weight_engine as _weight_engine
from ..portfolio.allocator import portfolio_allocator as _portfolio_allocator

logger = logging.getLogger(__name__)

# Só avalia candles 1H confirmados com menos de 2 horas de idade
MAX_CANDLE_AGE = timedelta(hours=2)

# Granularidade alvo — só avalia eventos dessa granularidade
EVAL_GRANULARITY = "1H"


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
        futures_flow=None,       # FuturesFlowCollector | None  (M6)
        relative_strength=None,  # RelativeStrengthCollector | None  (M7)
        vol_state=None,          # VolatilityStateCollector | None   (M8)
        meta_regime=None,        # MetaRegimeDetector | None         (Phase 13)
        news_sentiment=None,     # NewsSentimentCollector | None     (M9 Phase 4)
    ) -> None:
        self._bus = bus
        self._market = market
        self._cache = cache
        self._portfolio_value = portfolio_value
        self._futures_flow      = futures_flow       # injeta M6 no contexto
        self._relative_strength = relative_strength  # injeta M7 no contexto
        self._vol_state         = vol_state          # injeta M8 no contexto
        self._meta_regime       = meta_regime        # injeta meta regime no contexto
        self._news_sentiment    = news_sentiment     # injeta M9 no contexto
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

        # Pré-popula last_candle_ts e agenda warm-start para popular o dashboard
        await self._seed_last_candle_ts()

        self._task = asyncio.create_task(
            self._consume(), name="strategy_runner"
        )
        # Avalia 35s após boot (MarketEngine já terá candles) — popula signal_log
        asyncio.create_task(self._warm_start(), name="strategy_warm_start")
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

        # Timestamp do último candle 1H fechado = hora atual truncada - 1h
        now = datetime.now(UTC)
        last_closed_ts = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

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

    async def _warm_start(self) -> None:
        """Avalia todos os símbolos 35s após o boot para popular o dashboard imediatamente."""
        await asyncio.sleep(35)
        symbols: set[str] = set()
        for s in self._strategies.values():
            symbols.update(s.symbols)
        logger.info("StrategyRunner: warm-start — avaliando %d símbolos...", len(symbols))
        for symbol in sorted(symbols):
            try:
                candles = self._market.get_candles(symbol, "1H")
                if len(candles) < 22:
                    continue
                await self._evaluate_all(symbol)
                logger.info("StrategyRunner: warm-start %s — OK", symbol)
            except Exception as exc:
                logger.warning("StrategyRunner: warm-start %s falhou: %s", symbol, exc)

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
                    "Nova vela 1H %s ts=%s — avaliando estratégia",
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
                # Persiste contador diário de avaliações no Redis
                try:
                    raw = await self._cache.get("signals:daily_evals")
                    await self._cache.set(
                        "signals:daily_evals",
                        str((int(raw) if raw else 0) + 1),
                        ttl=90000,
                    )
                except Exception:
                    pass
                ctx = await self._build_context(symbol)
                signal = await strategy.evaluate(ctx)
                if signal is not None:
                    # Phase 15 — registra sinal no PortfolioAllocator para ranking cross-asset
                    from dataclasses import replace as _dc_replace
                    regime = getattr(signal, "regime", "UNKNOWN") or "UNKNOWN"
                    _portfolio_allocator.register_signal(
                        symbol=signal.symbol,
                        calibrated_score=float(signal.calibrated_score or 0),
                        regime=regime,
                        strategy_id=strategy.strategy_id,
                    )
                    # Aplica peso do regime ao kelly (WeightEngine)
                    # Signal é frozen dataclass → usa dataclasses.replace()
                    w = _weight_engine.get_weight(strategy.strategy_id, regime)
                    if w < 0.99:
                        logger.debug(
                            "WeightEngine: %s/%s kelly %.4f × %.2f = %.4f",
                            strategy.strategy_id, regime,
                            signal.kelly_fraction, w,
                            signal.kelly_fraction * w,
                        )
                    signal = _dc_replace(
                        signal,
                        kelly_fraction=round(signal.kelly_fraction * w, 4),
                        factors={
                            **(signal.factors or {}),
                            "we_weight": round(w, 3),
                            "we_regime": regime,
                        },
                    )
                    await self._publish_signal(signal)
            except Exception as exc:
                logger.error(
                    "Strategy %s raised: %s",
                    strategy.strategy_id, exc, exc_info=True,
                )

    async def _build_context(self, symbol: str) -> StrategyContext:
        candles_1h = self._market.get_candles(symbol, "1H")
        candles_6h = self._market.get_candles(symbol, "6H")
        # Ciclo 1H: candles_30m não coletados — estratégia usa candles_1h para tudo

        pos_data = await self._cache.get_position(symbol)
        open_positions = [pos_data] if pos_data else []

        # ── M6: Futures Flow (Phase 10) ───────────────────────────────────────
        futures_flow_data = None
        if self._futures_flow is not None:
            try:
                futures_flow_data = await self._futures_flow.get_flow(symbol)
            except Exception as exc:
                logger.debug("futures_flow.get_flow(%s) falhou: %s", symbol, exc)

        # ── M7: Relative Strength (Phase 11) ─────────────────────────────────
        # Lê do Redis o dado coletado pelo RelativeStrengthCollector (15min).
        # Fallback neutro (0.5) se dados indisponíveis.
        rs_data = None
        if self._relative_strength is not None:
            try:
                rs_data = await self._relative_strength.get_rs(symbol)
            except Exception as exc:
                logger.debug("relative_strength.get_rs(%s) falhou: %s", symbol, exc)

        # ── M8: Volatility State (Phase 12) ──────────────────────────────────
        vol_data = None
        if self._vol_state is not None:
            try:
                vol_data = await self._vol_state.get_state(symbol)
            except Exception as exc:
                logger.debug("vol_state.get_state(%s) falhou: %s", symbol, exc)

        # ── Meta Regime (Phase 13) — macro threshold modulator ───────────────
        meta_regime_data = None
        if self._meta_regime is not None:
            try:
                meta_regime_data = await self._meta_regime.get_regime()
            except Exception as exc:
                logger.debug("meta_regime.get_regime() falhou: %s", exc)

        # ── M9: News Sentiment (Phase 4) ─────────────────────────────────────
        news_data = None
        if self._news_sentiment is not None:
            try:
                news_data = await self._news_sentiment.get_scores(symbol)
            except Exception as exc:
                logger.debug("news_sentiment.get_scores(%s) falhou: %s", symbol, exc)

        # ── Model Health (Phase B — SizingEngine) ────────────────────────────
        # PSI de features + WR drift live vs calibrado.
        # Usado pelo SizingEngine para modular o Kelly dinamicamente.
        model_health_data = None
        try:
            raw_mh = await self._cache.get("model_health")
            if raw_mh:
                model_health_data = raw_mh if isinstance(raw_mh, dict) else json.loads(raw_mh)
        except Exception as exc:
            logger.debug("model_health cache read falhou: %s", exc)

        return StrategyContext(
            symbol=symbol,
            candles_1h=candles_1h,
            candles_6h=candles_6h,
            candles_30m=[],   # vazio — ciclo 1H não coleta granularidade 30m
            ticker=None,
            portfolio_value=self._portfolio_value,
            open_positions=open_positions,
            extra={
                "futures_flow":      futures_flow_data,
                "relative_strength": rs_data,
                "vol_state":         vol_data,
                "meta_regime":       meta_regime_data,
                "news_sentiment":    news_data,
                "model_health":      model_health_data,
            },
        )

    async def _publish_signal(self, signal: Signal) -> None:
        self._signal_count += 1
        # Persiste contador diário de sinais disparados no Redis
        try:
            raw = await self._cache.get("signals:daily_fired")
            await self._cache.set(
                "signals:daily_fired",
                str((int(raw) if raw else 0) + 1),
                ttl=90000,
            )
        except Exception:
            pass
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
