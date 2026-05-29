"""
Trading Loop — wires all engines together and runs the main cycle.

Pipeline completo (corrigido):
  MarketEngine → (CandleEvent) → StrategyRunner
  StrategyRunner → (SignalEvent) → [signal_consumer task]
  signal_consumer → RiskEngine.evaluate() → sizing → OrderManager
  OrderManager → ExecutionRouter → OKX
  OKX fill → (FillEvent) → PositionMonitor → exit orders

Parallel services:
  - PeriodicReconciler (every 2 min)
  - HeartbeatWatchdog (every 15s)
  - WebSocketWatchdog (every 5s)
  - AlertListener (continuous)
  - PositionMonitor (every 30s)

Boot sequence (BootSequence handles this):
  Connect → Schema → RECONCILING → Load state → Reconcile → RUNNING
"""

import asyncio
import logging
from pathlib import Path

from ..alerts.discord import create_alert_channel
from ..alerts.listener import AlertListener
from ..alerts.trading_alerts import TradingAlertsManager
from ..exchange.okx.client import OKXClient
from ..market.engine import MarketEngine
from ..market.futures_flow import FuturesFlowCollector
from ..market.meta_regime import MetaRegimeDetector
from ..market.news_sentiment import NewsSentimentCollector
from ..market.relative_strength import RelativeStrengthCollector
from ..market.volatility_state import VolatilityStateCollector
from ..metrics.infra_metrics import InfraMetrics
from ..monitoring.model_health import ModelHealthMonitor
from ..oms.execution_intelligence import execution_intelligence as _exec_intel
from ..oms.execution_router import ExecutionRouter
from ..oms.order_manager import OrderManager
from ..oms.position_monitor import PositionMonitor
from ..persistence import Cache, Database
from ..portfolio.allocator import portfolio_allocator as _portfolio_allocator
from ..portfolio.engine import PortfolioEngine
from ..recovery.boot import BootSequence
from ..recovery.periodic_reconciler import PeriodicReconciler
from ..risk.advanced_risk import AdvancedRiskManager
from ..risk.engine import RiskContext, RiskEngine
from ..risk.kill_switch import KillSwitch
from ..strategies.meta_layer import MetaStrategyLayer
from ..strategies.ml.inference import MLInferenceEngine
from ..strategies.momentum.momentum_strategy import MomentumStrategy
from ..strategies.runner import StrategyRunner
from ..watchdog.heartbeat import HeartbeatWatchdog
from ..watchdog.resource_watchdog import ResourceWatchdog
from ..watchdog.websocket_watchdog import WebSocketWatchdog
from .bus import EventBus
from .config import OKX_TAKER_FEE, settings
from .events import SignalEvent, Topic
from .events.risk_events import RiskAction

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
MODELS_DIR = Path("data") / "models"

# Quantidade mínima por símbolo (OKX spot minimums)
MIN_QTY = {
    "BTC-USDT": 0.00001,
    "ETH-USDT": 0.0001,
    "SOL-USDT": 0.01,
}
# Precisão de casas decimais por símbolo
QTY_PRECISION = {
    "BTC-USDT": 5,
    "ETH-USDT": 4,
    "SOL-USDT": 2,
}


# ── P2-1 REFACTOR ROADMAP ────────────────────────────────────────────────────
# TradingLoop é uma god-class de ~1300 linhas. Limites identificados para split:
#
#  BootService        (~lines 87-470):  __init__, start(), stop(), boot sequence
#  SignalConsumer     (~lines 495-710): _process_signal, _on_fill_update_portfolio
#  PaperFillService   (~lines 940-1060): _simulate_paper_fills, paper fill logic
#  DailyDigestService (~lines 778-940):  _maybe_send_daily_summary, daily stats
#
# Bloqueio atual: todas as partes acessam self._positions, self._cache,
# self._oms, self._portfolio, self._bus — necessário criar TradingState
# como objeto de estado compartilhado antes do split.
#
# Próximo passo: extrair PaperFillService (mais isolado) em src/core/paper_fill.py
# ─────────────────────────────────────────────────────────────────────────────


class TradingLoop:
    """
    Main trading loop — composes all engines and runs them.
    Call start() to begin, stop() to shut down gracefully.
    """

    def __init__(
        self,
        db: Database,
        cache: Cache,
        app_state=None,
    ) -> None:
        self._db = db
        self._cache = cache
        self._app_state = app_state
        self._running = False

        # ── Core infrastructure ───────────────────────────────
        self._bus = EventBus()
        self._kill_switch = KillSwitch()
        self._infra_metrics = InfraMetrics()

        # ── OKX client ────────────────────────────────────────
        # Use demo credentials when in paper mode and demo keys are configured
        _paper = settings.okx_paper_trading
        _has_demo = bool(settings.okx_demo_api_key and settings.okx_demo_secret_key)
        self._okx = OKXClient(
            api_key=settings.okx_demo_api_key if (_paper and _has_demo) else settings.okx_api_key,
            secret_key=(
                settings.okx_demo_secret_key if (_paper and _has_demo) else settings.okx_secret_key
            ),
            passphrase=(
                settings.okx_demo_passphrase if (_paper and _has_demo) else settings.okx_passphrase
            ),
            paper_trading=_paper,
        )
        if _paper and _has_demo:
            logger.info("OKXClient: using DEMO credentials with x-simulated-trading=1")
        elif _paper:
            logger.info("OKXClient: paper mode — local simulation (no demo keys)")

        # ── Market Engine ─────────────────────────────────────
        self._market = MarketEngine(
            bus=self._bus,
            okx=self._okx,
            cache=self._cache,
            symbols=SYMBOLS,
            granularities=["1H", "6H"],
            poll_interval=15,
        )

        # ── OMS ───────────────────────────────────────────────
        router = ExecutionRouter(exchange=self._okx)
        self._oms = OrderManager(
            bus=self._bus,
            router=router,
            db=self._db,   # persiste ordens no PostgreSQL imediatamente
        )

        # ── Risk Engine ───────────────────────────────────────
        self._risk = RiskEngine(
            bus=self._bus,
            kill_switch=self._kill_switch,
        )

        # ── Portfolio Engine ──────────────────────────────────
        self._portfolio = PortfolioEngine(
            bus=self._bus,
            initial_capital=82_515.77,  # capital operacional (USDT, excl. OKB/BRL)
        )
        # Rastreia posições abertas em memória (atualizado a cada fill)
        self._positions: dict = {}   # symbol → Position (importado localmente nos métodos)
        self._exec_expected_price: dict[str, float] = {}  # Phase 17: expected price at signal time
        self._cash: float = 82_515.77  # atualizado pelo ExchangeSync no boot

        # ── ML Inference ──────────────────────────────────────
        self._ml = MLInferenceEngine(models_dir=MODELS_DIR)

        # ── Meta Layer ────────────────────────────────────────
        self._meta = MetaStrategyLayer()

        # ── Futures Flow Collector (Phase 10) — M6 data source ───────────────
        self._futures_flow = FuturesFlowCollector(
            exchange=self._okx,
            cache=self._cache,
            symbols=SYMBOLS,
        )

        # ── Relative Strength Collector (Phase 11) — M7 data source ──────────
        self._rel_strength = RelativeStrengthCollector(
            market=self._market,
            cache=self._cache,
            symbols=SYMBOLS,
        )

        # ── Volatility State Collector (Phase 12) — M8 data source ───────────
        self._vol_state = VolatilityStateCollector(
            market=self._market,
            cache=self._cache,
            symbols=SYMBOLS,
        )

        # ── News Sentiment Collector (Phase 4) — M9 data source ──────────────
        self._news_sentiment = NewsSentimentCollector(cache=self._cache)

        # ── Model Health Monitor (Phase 15.2) ────────────────────────────────
        from ..monitoring.signal_log import signal_audit_log as _sal
        self._model_health = ModelHealthMonitor(cache=self._cache, signal_log=_sal)
        self._model_health.set_db(self._db)   # WR real via trades fechados

        # ── Advanced Risk Manager (Phase 14) ─────────────────────────────────
        self._adv_risk = AdvancedRiskManager(
            market=self._market,
            cache=self._cache,
        )

        # ── Meta Regime Detector (Phase 13) — macro threshold modulator ───────
        self._meta_regime = MetaRegimeDetector(
            market=self._market,
            cache=self._cache,
            symbols=SYMBOLS,
        )

        # ── Strategy Runner ───────────────────────────────────
        self._runner = StrategyRunner(
            bus=self._bus,
            market=self._market,
            cache=self._cache,
            futures_flow=self._futures_flow,
            relative_strength=self._rel_strength,
            vol_state=self._vol_state,
            meta_regime=self._meta_regime,
            news_sentiment=self._news_sentiment,
        )
        v4 = MomentumStrategy(symbols=SYMBOLS)
        self._momentum_strategy = v4
        self._runner.register(v4)
        self._meta.register(v4.strategy_id)

        # ── Alerts ────────────────────────────────────────────
        self._alert_channel = create_alert_channel(
            settings.discord_webhook_url,
            bot_name=f"CCTBv5 [{settings.bot_instance}]",
        )
        self._alert_listener = AlertListener(
            bus=self._bus,
            channel=self._alert_channel,
            cache=self._cache,
        )
        # Manager de alertas de trading (regime, drawdown, WR, trades)
        self._trading_alerts = TradingAlertsManager(self._alert_channel)

        # ── Watchdogs ─────────────────────────────────────────
        self._heartbeat = HeartbeatWatchdog(
            bus=self._bus,
            kill_switch=self._kill_switch,
            timeout_seconds=60,
            interval_seconds=15,
        )
        self._ws_watchdog = WebSocketWatchdog(
            on_dead=self._on_ws_dead,
            dead_threshold=300,   # 5 min sem mensagem antes de considerar morto
            name="okx_market",
        )
        self._resource_watchdog = ResourceWatchdog(
            kill_switch=self._kill_switch,
            interval_seconds=30,
        )

        # ── Position Monitor ──────────────────────────────────
        self._position_monitor = PositionMonitor(
            bus=self._bus,
            market=self._market,
            portfolio=self._portfolio,
            oms=self._oms,
            cache=self._cache,
            interval_seconds=30,
        )

        # ── Periodic Reconciler ───────────────────────────────
        self._reconciler = PeriodicReconciler(
            bus=self._bus,
            db=self._db,
            exchange=self._okx,
            order_manager=self._oms,
            kill_switch=self._kill_switch,
            interval_seconds=120,
        )

        # ── Signal consumer (pipeline SIGNAL → RISK → OMS) ───
        self._signal_queue: asyncio.Queue | None = None
        self._signal_task: asyncio.Task | None = None

        self._market.set_on_poll_callback(self._ws_watchdog.record_message)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        logger.info("TradingLoop: starting boot sequence...")

        boot = BootSequence(
            bus=self._bus,
            db=self._db,
            cache=self._cache,
            exchange=self._okx,
            order_manager=self._oms,
        )
        ready = await boot.run()
        if not ready:
            logger.error("TradingLoop: boot failed — not starting")
            return False

        # Injeta DB no AdvancedRiskManager (disponível pós-boot)
        self._adv_risk.set_db(self._db)

        # Phase 17 — Injeta cache e carrega histórico de slippage do Redis
        _exec_intel.set_cache(self._cache)
        await _exec_intel.load_history()

        if self._app_state:
            self._app_state.ws_watchdog       = self._ws_watchdog
            self._app_state.heartbeat_watchdog = self._heartbeat
            self._app_state.resource_watchdog  = self._resource_watchdog
            self._app_state.portfolio          = self._portfolio
            self._app_state.position_monitor   = self._position_monitor

        # ── Subscreve ao pipeline de sinais (queue criada antes dos starts) ──
        # IMPORTANTE: a task só é criada DEPOIS de self._running = True
        # para evitar que o consumer veja self._running=False e saia imediatamente.
        self._signal_queue = self._bus.subscribe(Topic.SIGNAL)

        await self._alert_listener.start()
        await self._heartbeat.start()
        await self._ws_watchdog.start()
        await self._resource_watchdog.start()
        await self._market.start()
        await self._futures_flow.start()     # Phase 10: M6 data antes do runner
        await self._rel_strength.start()    # Phase 11: M7 data antes do runner
        await self._vol_state.start()       # Phase 12: M8 data antes do runner
        await self._meta_regime.start()     # Phase 13: macro regime antes do runner
        # Phase 4: M9 — warm-up síncrono + background loop
        await self._news_sentiment.warm_up()
        asyncio.create_task(
            self._news_sentiment.run_forever(), name="news_sentiment_poll"
        )
        await self._adv_risk.start()        # Phase 14: advanced risk
        await self._model_health.start()    # Phase 15.2: model health monitor

        # CrossAssetEngine — estratégia market-neutral
        from ..strategies.cross_asset.cross_asset_strategy import CrossAssetEngine
        self._cross_asset = CrossAssetEngine(cache=self._cache, okx_client=self._okx)
        await self._cross_asset.start()
        logger.info("CrossAssetEngine: market-neutral strategy started")
        # B1 fix: injeta no PositionMonitor para fechar SHORTs direcionais via swap
        self._position_monitor.set_cross_asset(self._cross_asset)

        # v5.20 Module 3 — FundingHarvest: coleta funding passiva de perpetual swaps
        from ..strategies.funding.funding_harvest import FundingHarvest
        self._funding_harvest = FundingHarvest(
            cache=self._cache,
            okx_client=self._okx,
            position_monitor=self._position_monitor,
        )
        await self._funding_harvest.start()
        logger.info("FundingHarvest: passive funding income strategy started")

        # v5.20 Module 5 — SectorPairDetector: mean-reversion pairs em CHOP
        from ..strategies.pairs.sector_pair_detector import SectorPairDetector
        self._sector_pairs = SectorPairDetector(
            cache=self._cache,
            okx_client=self._okx,
        )
        await self._sector_pairs.start()
        logger.info("SectorPairDetector: mean-reversion pairs strategy started")

        # v5.20 Module 2 — ShortSqueezeDetector: detecta e opera short squeezes
        from ..strategies.squeeze.short_squeeze_detector import ShortSqueezeDetector
        self._squeeze = ShortSqueezeDetector(
            cache=self._cache,
            okx_client=self._okx,
        )
        await self._squeeze.start()
        logger.info("ShortSqueezeDetector: squeeze detection strategy started")

        # v5.20 Module 1 — RegimeAwarePairEngine: pares adaptativos BULL/BEAR/CHOP
        from ..strategies.cross_asset.regime_aware_pair_engine import RegimeAwarePairEngine
        self._rape = RegimeAwarePairEngine(
            cache=self._cache,
            okx_client=self._okx,
        )
        await self._rape.start()
        logger.info("RegimeAwarePairEngine: adaptive pairs BULL/BEAR/CHOP started")

        await self._runner.start()
        await self._position_monitor.start()
        await self._reconciler.start()

        # P2-4: Alerta crítico se PlattCalibrator não carregou coeficientes reais
        if self._momentum_strategy.is_platt_using_defaults:
            await self._alert_channel.critical(
                "PLATT CALIBRATOR — COEFICIENTES PADRÃO",
                "PlattCalibrator usando DEFAULTS (A=2.5, B=-1.2). "
                "Coeficientes reais (A≈0.476, B≈-0.826) não foram carregados. "
                "Scoring de sinais fortemente impactado — execute calibrate.py.",
            )

        # Restaura histórico de live features do Redis (janela deslizante do DriftMonitor)
        try:
            from ..monitoring.feature_governance import governance
            restored = await governance.drift.load_from_redis(self._cache)
            if restored:
                logger.info(
                    "DriftMonitor: histórico restaurado (%d obs) — PSI disponível imediatamente",
                    governance.drift._obs_count,
                )
        except Exception as exc:
            logger.warning("DriftMonitor: falha ao restaurar histórico: %s", exc)

        # Restaura histórico de equity diário para o DrawdownEngine (janela 30d)
        try:
            import json as _json
            raw_equity = await self._cache.get("drawdown:daily_equity")
            if raw_equity:
                snapshots = _json.loads(raw_equity) if isinstance(raw_equity, str) else raw_equity
                self._risk._drawdown.restore_daily_equity(
                    [(s["date"], float(s["value"])) for s in snapshots]
                )
                logger.info(
                    "DrawdownEngine: %d snapshots de equity restaurados",
                    len(snapshots),
                )
        except Exception as exc:
            logger.debug("DrawdownEngine: falha ao restaurar equity histórico: %s", exc)

        # Sincronização completa com OKX: saldo, posições e histórico de ordens
        try:
            from ..recovery.exchange_sync import ExchangeSync
            sync = ExchangeSync(
                exchange=self._okx,
                db=self._db,
                cache=self._cache,
                portfolio=self._portfolio,
            )
            sync_result = await sync.run()

            # Atualiza tracker interno — usa USDT como base do portfolio
            usdt = sync_result["usdt_balance"]
            portfolio_usdt = self._portfolio.state.total_value  # já calculado no sync
            if usdt > 0:
                self._cash = usdt
                self._runner.update_portfolio_value(portfolio_usdt)
                logger.info(
                    "ExchangeSync: USDT=%.2f portfolio_USDT=%.2f posições=%s ordens=%d",
                    usdt, portfolio_usdt,
                    list(sync_result["crypto_positions"].keys()),
                    sync_result["orders_imported"],
                )

            # Carrega posições exchange_sync em self._positions para permitir vendas
            # Sem isso, o bot não sabe que tem BTC/ETH/SOL e ignora sinais de saída.
            # IMPORTANTE: usa net_bot_qty (comprado - vendido pelo bot) quando disponível,
            # para evitar criar ExitPlans para balances pré-existentes na exchange
            # que não foram comprados por esta instância do bot.
            from ..core.models import Position, PositionSide
            for symbol, qty in sync_result.get("crypto_positions", {}).items():
                try:
                    price_raw = await self._cache.get_price(symbol)
                    price = float(price_raw) if price_raw else 0.0
                    if qty > 0:
                        # Calcula net do que o bot comprou/vendeu neste símbolo
                        bot_bought = await self._db.fetchval(
                            "SELECT COALESCE(SUM(filled_quantity),0) FROM orders "
                            "WHERE symbol=$1 AND side='buy' AND status='filled' "
                            "AND strategy_id NOT IN ('exchange_sync','okx_import')",
                            symbol,
                        ) if self._db else 0.0
                        bot_sold = await self._db.fetchval(
                            "SELECT COALESCE(SUM(filled_quantity),0) FROM orders "
                            "WHERE symbol=$1 AND side='sell' AND status='filled' "
                            "AND strategy_id NOT IN ('exchange_sync','okx_import')",
                            symbol,
                        ) if self._db else 0.0
                        net_bot_qty = max(float(bot_bought or 0) - float(bot_sold or 0), 0.0)
                        # Se o bot tem posição própria rastreada, usa ela; senão usa exchange qty
                        effective_qty = net_bot_qty if net_bot_qty > 0.001 else qty
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.LONG,
                            strategy_id="exchange_sync",
                            quantity=effective_qty,
                            avg_entry_price=price,
                            total_fees=0.0,
                        )
                        logger.info(
                            "Posição carregada: %s exchange=%.6f bot_net=%.6f effective=%.6f @ %.2f",
                            symbol, qty, net_bot_qty, effective_qty, price,
                        )
                except Exception as pos_exc:
                    logger.debug("Erro ao carregar posição %s: %s", symbol, pos_exc)

            # Cria ExitPlans para posições sincronizadas da exchange
            # Usa a mesma lógica de effective_qty para não criar planos inflados
            for symbol, qty in sync_result.get("crypto_positions", {}).items():
                try:
                    price_raw = await self._cache.get_price(symbol)
                    price = float(price_raw) if price_raw else 0.0
                    if price > 0 and qty > 0:
                        # Reutiliza effective_qty já calculado acima (está em self._positions)
                        pos = self._positions.get(symbol)
                        effective_qty = pos.quantity if pos else qty
                        await self._position_monitor._create_plan(
                            symbol, effective_qty, price, "exchange_sync"
                        )
                        logger.info(
                            "ExitPlan criado para posição sync: %s qty=%.4f entry=%.2f",
                            symbol, effective_qty, price,
                        )
                except Exception as ep_exc:
                    logger.debug("ExitPlan sync falhou para %s: %s", symbol, ep_exc)
        except Exception as exc:
            logger.warning("ExchangeSync falhou no boot: %s", exc)

        self._running = True
        logger.info("TradingLoop: all services started — RUNNING")

        # Cria a task do consumer APÓS self._running = True
        # (se fosse antes, o while self._running: retornaria False imediatamente)
        self._signal_task = asyncio.create_task(
            self._consume_signals(), name="signal_consumer"
        )

        # Dispara notificação Discord sem bloquear o loop principal.
        # await direto antes de _run_loop() pode travar o event loop se o
        # httpx/DNS resolver usar fallback síncrono (causa do heartbeat_timeout).
        asyncio.create_task(
            self._alert_channel.info(
                title="CCTBv5 Started",
                message="Sistema iniciado com sucesso.",
                mode="paper" if settings.okx_paper_trading else "live",
                symbols=", ".join(SYMBOLS),
            ),
            name="discord_startup_notify",
        )

        await self._run_loop()
        return True

    async def stop(self) -> None:
        self._running = False
        logger.info("TradingLoop: shutting down...")

        if self._signal_task and not self._signal_task.done():
            self._signal_task.cancel()
            try:
                await self._signal_task
            except asyncio.CancelledError:
                pass

        await self._market.stop()
        await self._futures_flow.stop()
        await self._rel_strength.stop()
        await self._vol_state.stop()
        await self._meta_regime.stop()
        self._news_sentiment.stop()
        await self._adv_risk.stop()
        await self._model_health.stop()
        if hasattr(self, "_rape"):
            await self._rape.stop()
        if hasattr(self, "_squeeze"):
            await self._squeeze.stop()
        if hasattr(self, "_sector_pairs"):
            await self._sector_pairs.stop()
        if hasattr(self, "_funding_harvest"):
            await self._funding_harvest.stop()
        if hasattr(self, "_cross_asset"):
            await self._cross_asset.stop()
        await self._runner.stop()
        await self._position_monitor.stop()
        await self._reconciler.stop()
        await self._heartbeat.stop()
        await self._ws_watchdog.stop()
        await self._resource_watchdog.stop()
        await self._alert_listener.stop()

        await self._alert_channel.info(
            title="CCTBv5 Stopped",
            message="Sistema encerrado graciosamente.",
        )
        logger.info("TradingLoop: shutdown complete")

    # ── Pipeline: Signal → Risk → Sizing → OMS ───────────────────────────────

    async def _consume_signals(self) -> None:
        """
        Consumer do pipeline de sinais.
        Recebe cada SignalEvent, avalia risco, dimensiona e envia ao OMS.
        """
        logger.info("Signal consumer iniciado — aguardando sinais...")
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._signal_queue.get(), timeout=1.0
                )
                if isinstance(event, SignalEvent) and event.signal:
                    await self._process_signal(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Signal consumer error: %s", exc, exc_info=True)

    async def _process_signal(self, event: SignalEvent) -> None:
        """
        Processa um sinal aprovado pela estratégia:
          1. RiskEngine — verifica limites de drawdown, exposição, cooldown
          2. Sizing     — calcula quantidade usando kelly_fraction e preço atual
          3. OMS        — cria e submete ordem de mercado
        """
        # Modo monitor: sinais avaliados e logados, mas nenhuma ordem executada
        if settings.monitor_only:
            logger.debug(
                "[MONITOR_ONLY] Sinal recebido mas não executado: %s %s score=%.3f",
                event.signal.direction, event.signal.symbol, event.signal.calibrated_score,
            )
            return

        from ..core.models.signal import SignalDirection

        signal  = event.signal

        # ── SHORT via swap perp (BEAR_TREND com flag short_via_swap) ─────────
        # Sinais SHORT gerados pelo MomentumStrategy em BEAR_TREND são roteados
        # para o CrossAssetEngine que os executa via OKX perpetual swap.
        # Pipeline completo: sizing por kelly → swap order → DB persist → ShortSwapPlan
        if (signal.direction == SignalDirection.SHORT
                and (signal.factors or {}).get("short_via_swap")
                and hasattr(self, "_cross_asset")):
            import uuid as _uuid

            from ..oms.position_monitor import _calc_atr as _pm_calc_atr
            from ..strategies.cross_asset.cross_asset_strategy import SWAP_CONTRACT_SIZE

            swap_sym  = signal.symbol.replace("-USDT", "-USDT-SWAP")
            pv        = self._portfolio.state.total_value or 82_515.77
            price_raw = await self._cache.get_price(signal.symbol)
            price     = float(price_raw) if price_raw else 0.0

            if price > 0 and pv > 0:
                # B2 fix: usa kelly_fraction do sinal (calculado pelo SizingEngine)
                # em vez de 5% hardcoded — respeita o sizing da estratégia.
                kelly    = max(float(signal.kelly_fraction or 0.03), 0.01)
                notional = pv * kelly
                cs       = SWAP_CONTRACT_SIZE.get(swap_sym, 1.0)
                contracts = max(1, int(notional / (price * cs)))
                logger.info(
                    "BEAR SHORT %s: %d contratos via swap | kelly=%.1f%% notional=%.2f",
                    swap_sym, contracts, kelly * 100, notional,
                )

                # B3 fix: usa await para obter exchange_order_id antes de persistir
                try:
                    eid = await self._cross_asset._place_swap_short(
                        signal.symbol, contracts
                    )
                except Exception as exc:
                    logger.error(
                        "BEAR SHORT %s: falha ao colocar ordem swap: %s", swap_sym, exc
                    )
                    return

                # B3 fix: persiste ordem no banco via INSERT direto
                # (OMS não gerencia swaps; inserimos como strategy_id='momentum_short')
                if self._db:
                    try:
                        coid = f"bear_short_{_uuid.uuid4().hex[:16]}"
                        qty_base = float(contracts) * cs
                        await self._db.execute(
                            """
                            INSERT INTO orders (
                                client_order_id, exchange_order_id,
                                symbol, side, order_type, mode, status,
                                quantity, filled_quantity, avg_fill_price,
                                fees_paid, strategy_id, signal_id,
                                created_at, filled_at
                            ) VALUES (
                                $1, $2,
                                $3, 'sell', 'market', 'paper', 'filled',
                                $4, $4, $5,
                                0, 'momentum_short', 'bear_short_signal',
                                NOW(), NOW()
                            ) ON CONFLICT (client_order_id) DO NOTHING
                            """,
                            coid,
                            eid or f"swap_{signal.symbol}_{coid}",
                            signal.symbol,
                            qty_base,
                            price,
                        )
                        logger.info(
                            "BEAR SHORT %s: ordem persistida no DB (coid=%s qty=%.4f)",
                            signal.symbol, coid, qty_base,
                        )
                    except Exception as db_exc:
                        logger.warning(
                            "BEAR SHORT: falha ao persistir ordem no DB: %s", db_exc
                        )

                # B1 fix: cria ShortSwapPlan no PositionMonitor para SL/TP automático
                candles_1h = self._market.get_candles(signal.symbol, "1H", limit=20)
                atr = (
                    _pm_calc_atr(candles_1h, 14)
                    if len(candles_1h) >= 15
                    else price * 0.015
                )
                self._position_monitor.register_short_plan(
                    symbol=signal.symbol,
                    swap_sym=swap_sym,
                    contracts=contracts,
                    entry_price=price,
                    atr=atr,
                    strategy_id="momentum_short",
                )

            return

        # B7 fix: SHORT sem short_via_swap não deve cair no path de venda de long.
        # Rejeita explicitamente para evitar venda acidental de posição comprada.
        if signal.direction == SignalDirection.SHORT:
            logger.warning(
                "Sinal SHORT sem flag short_via_swap rejeitado para %s "
                "(sem rota de execução definida)",
                signal.symbol,
            )
            return

        is_exit = signal.direction in (SignalDirection.FLAT,)

        open_orders    = self._oms.get_open_orders()
        existing_plans = getattr(self._position_monitor, '_plans', {})

        # ── SAÍDA (SELL/FLAT): vende posição aberta ────────────────────────────
        if is_exit:
            if signal.symbol not in self._positions:
                logger.debug(
                    "Sinal de saída ignorado — sem posição aberta para %s", signal.symbol
                )
                return
            # Bloqueia se já há ordem de venda pendente para o símbolo
            sell_pending = any(
                o.symbol == signal.symbol
                and str(getattr(o, "side", "")).lower() in ("sell", "short")
                for o in open_orders
            )
            if sell_pending:
                logger.debug("Ordem de venda já pendente para %s", signal.symbol)
                return

            pos       = self._positions[signal.symbol]
            precision = QTY_PRECISION.get(signal.symbol, 4)
            quantity  = round(pos.quantity, precision)
            min_qty   = MIN_QTY.get(signal.symbol, 0.0001)

            if quantity < min_qty:
                logger.info(
                    "Venda ignorada — qty %.6f < mínimo %.6f para %s",
                    quantity, min_qty, signal.symbol,
                )
                return

            price_raw = await self._cache.get_price(signal.symbol)
            price = float(price_raw) if price_raw else pos.avg_entry_price
            if price <= 0:
                price = pos.avg_entry_price
            logger.info(
                "SELL %s qty=%.6f @ ~%.2f (entrada=%.2f strat=%s)",
                signal.symbol, quantity, price, pos.avg_entry_price, pos.strategy_id,
            )
            await self._oms.create_order_from_signal(event, quantity)
            return

        # ── ENTRADA (BUY/LONG): bloqueia se já há posição ou ordem aberta ─────
        already_open = any(o.symbol == signal.symbol for o in open_orders)
        if not already_open:
            already_open = signal.symbol in existing_plans

        if already_open:
            logger.debug(
                "Sinal de entrada ignorado — posição/ordem já aberta para %s", signal.symbol
            )
            return

        # 1a. Phase 14 — Circuit breakers avançados
        cb_result = await self._adv_risk.check_circuit_breakers(signal.symbol)
        if not cb_result["allowed"]:
            logger.warning(
                "CIRCUIT BREAKER ativo: %s symbol=%s — %s",
                cb_result.get("cb_type"), signal.symbol, cb_result.get("reason"),
            )
            return

        # 1b. Avaliação de risco (RiskEngine padrão)
        portfolio_value = self._portfolio.state.total_value or 82_515.77
        risk_ctx = RiskContext(
            portfolio_value=portfolio_value,
            open_positions=[],
            strategy_id=signal.strategy_id,
        )
        action = await self._risk.evaluate(risk_ctx)

        if action != RiskAction.NORMAL:
            logger.info(
                "Sinal rejeitado pelo RiskEngine action=%s symbol=%s strategy=%s",
                action, signal.symbol, signal.strategy_id,
            )
            return

        # 2. Sizing usando kelly_fraction e preço atual
        price_raw = await self._cache.get_price(signal.symbol)
        if not price_raw:
            logger.warning(
                "Sem preço em cache para %s — sinal descartado", signal.symbol
            )
            return

        price = float(price_raw)
        if price <= 0:
            return

        # Kelly já vem ajustado pelo regime_mult da estratégia
        # Phase 14: aplica multiplicador dos circuit breakers (corr/liquidez/weeklyDD)
        adv_kelly_mult = cb_result.get("kelly_mult", 1.0)
        kelly    = (signal.kelly_fraction if signal.kelly_fraction is not None else 0.05) * adv_kelly_mult
        regime   = getattr(signal, "regime", "MEAN_REVERTING_CHOP")

        # Phase 15 — Portfolio Intelligence Layer
        # Atualiza correlações e posições abertas no allocator
        try:
            adv_state = await self._adv_risk.get_snapshot()
            corr_matrix = ((adv_state or {}).get("correlation") or {}).get("matrix") or {}
            if corr_matrix:
                _portfolio_allocator.update_correlations(corr_matrix)
        except Exception:
            pass
        _portfolio_allocator.update_open_positions(set(self._positions.keys()))

        # Ranking cross-asset: ajusta kelly pelo edge relativo do portfólio
        alloc = _portfolio_allocator.allocate(
            symbol=signal.symbol,
            original_kelly=kelly,
            regime=regime,
        )
        kelly = alloc.allocated_kelly
        logger.info(
            "PortfolioAllocator: %s rank=%d/%d kelly %.1f%% → %.1f%% [%s]",
            signal.symbol, alloc.rank, alloc.total_signals,
            alloc.original_kelly * 100, alloc.allocated_kelly * 100,
            alloc.reason,
        )

        # Cap máximo de Kelly por família de regime (segurança extra)
        KELLY_CAP = {
            "TREND_EXPANSION":        0.15,
            "VOLATILITY_COMPRESSION": 0.12,
            "TREND_EXHAUSTION":       0.10,
            "MEAN_REVERTING_CHOP":    0.08,
            "HIGH_CORRELATION_RISK":  0.05,
        }
        kelly     = min(kelly, KELLY_CAP.get(regime, 0.08))

        # Phase 17 — Execution Intelligence: sizing mult por qualidade histórica de execução
        exec_quality_mult = _exec_intel.sizing_mult(signal.symbol)
        if exec_quality_mult < 1.0:
            logger.info(
                "ExecutionIntelligence: %s quality_mult=%.2f → kelly %.1f%% → %.1f%%",
                signal.symbol, exec_quality_mult,
                kelly * 100, kelly * exec_quality_mult * 100,
            )
        kelly *= exec_quality_mult

        notional  = portfolio_value * kelly
        precision = QTY_PRECISION.get(signal.symbol, 4)
        quantity  = round(notional / price, precision)
        min_qty   = MIN_QTY.get(signal.symbol, 0.0001)

        if quantity < min_qty:
            logger.info(
                "Quantidade %.6f < mínimo %.6f para %s — sinal descartado",
                quantity, min_qty, signal.symbol,
            )
            return

        logger.info(
            "BUY %s regime=%s qty=%.6f price=%.2f notional=%.2f kelly=%.1f%% (cap=%.0f%%)",
            signal.symbol, regime, quantity, price, notional,
            kelly * 100, KELLY_CAP.get(regime, 0.08) * 100,
        )

        # 3. Phase 17 — SmartOrderRouter: decide maker vs taker
        try:
            ticker = await self._okx.get_ticker(signal.symbol)
            spread_pct = ticker.spread_pct if ticker else 0.005
            mid_price  = ticker.mid         if ticker else price
        except Exception:
            spread_pct = 0.005
            mid_price  = price

        order_decision = _exec_intel.decide_order_type(
            symbol=signal.symbol,
            side=signal.direction,
            mid_price=mid_price,
            spread_pct=spread_pct,
            atr_pct=getattr(signal, "atr_pct", 0.01),
            regime=regime,
        )
        # Armazena preço esperado para slippage tracking no fill
        self._exec_expected_price[signal.symbol] = mid_price

        # 4. Criar e submeter ordem via OMS com tipo decidido pelo SmartRouter
        # B4 fix: repassa o regime ADX do sinal para o PositionMonitor ANTES do submit,
        # para que _create_plan use regime correto (ADX-based) em vez de SMA-based.
        self._position_monitor.set_pending_signal_factors(
            signal.symbol,
            {**(signal.factors or {}), "regime": signal.regime},
        )
        await self._oms.create_order_from_signal(
            event,
            quantity,
            order_type_str=order_decision.order_type,
            limit_price=order_decision.limit_price,
        )

        # Alerta Discord — trade executado (paper + live)
        regime = getattr(signal, "regime", "")
        asyncio.create_task(
            self._trading_alerts.on_order_filled(
                symbol=signal.symbol,
                side=signal.direction,
                quantity=float(quantity),
                price=float(price),
                notional=float(notional),
                fees=float(notional * OKX_TAKER_FEE),
                strategy_id="momentum_v2",
                regime=regime,
                score=float(signal.calibrated_score or 0),
            ),
            name="discord_trade_alert",
        )

    # ── Alertas de trading (regime, DD, WR) ─────────────────────────────────
    async def _check_trading_alerts(self) -> None:
        """Verificações periódicas — regime, drawdown, win rate."""
        try:
            # Regime atual (via Redis do motor de sinais)
            for sym in SYMBOLS:
                sig_raw = await self._cache.get(f"signal:{sym}")
                if sig_raw:
                    import json as _json
                    sig = _json.loads(sig_raw) if isinstance(sig_raw, str) else sig_raw
                    regime = sig.get("regime", "")
                    if regime:
                        await self._trading_alerts.on_regime_check(regime)
                    break  # checa só o primeiro símbolo disponível

            # Drawdown diário
            p = self._portfolio.state
            daily_pnl = getattr(p, "daily_pnl", 0.0) or 0.0
            daily_dd  = abs(min(daily_pnl, 0.0))  # drawdown só quando negativo
            total_val = getattr(p, "total_value", 0.0) or 1.0
            dd_pct    = daily_dd / total_val if total_val > 0 else 0.0
            await self._trading_alerts.on_drawdown_check(dd_pct, total_val)

        except Exception as exc:
            logger.debug("_check_trading_alerts error: %s", exc)

    # ── Resumo diário Discord ─────────────────────────────────────────────────

    async def _maybe_send_daily_summary(self) -> None:
        """
        Envia resumo diário enriquecido ao Discord (~00:00 UTC).

        Inclui:
          - P&L do dia + retorno acumulado
          - Benchmark 24h: BTC / ETH / SOL vs bot
          - Win rate do dia vs calibrado (37.5%)
          - Estatísticas de sinais (avaliações, taxa de sinal)
          - Regime atual
        """
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        if now.hour != 0 or now.minute > 14:
            return
        # Usa Redis para garantir envio único por dia
        key = f"daily_summary:{now.strftime('%Y-%m-%d')}"
        already_sent = await self._cache.get(key)
        if already_sent:
            return
        await self._cache.set(key, "1", ttl=86400)

        p        = self._portfolio.state
        dpnl     = self._trading_alerts._pnl_today
        total    = getattr(p, "total_value", 0.0) or 0.0
        initial  = getattr(p, "initial_capital", total) or total
        ret      = (total - initial) / initial if initial > 0 else 0.0
        dd_pct   = getattr(p, "drawdown_pct", 0.0) or 0.0
        mode_tag = "🎮 Paper" if settings.okx_paper_trading else "💰 Live"
        emoji    = "📈" if dpnl >= 0 else "📉"

        # ── Trades do dia ─────────────────────────────────────────────────────
        trades_d = self._trading_alerts._trades_today
        wins_d   = getattr(self._trading_alerts, "_wins_today", 0)
        wr_day   = wins_d / trades_d if trades_d > 0 else None

        # ── Win rate rolling (últimas 20 trades no DB) ─────────────────────
        wr_rolling = None
        n_rolling  = 0
        try:
            from ..persistence.repositories.orders import EXCLUDED_STRATEGY_IDS
            rows = await self._db.fetch(
                """
                SELECT side, filled_quantity, avg_fill_price
                FROM orders
                WHERE status='filled' AND strategy_id != ALL($1)
                ORDER BY filled_at DESC LIMIT 40
                """,
                list(EXCLUDED_STRATEGY_IDS),
            )
            # Pareia compras/vendas por símbolo para contar wins
            buys: dict[str, list] = {}
            sell_count = 0
            for r in reversed(rows):
                side = str(r["side"]).upper()
                # simplificação: conta venda com preço > média de compras anteriores
                if side in ("BUY", "LONG"):
                    sym = "?"
                    buys.setdefault(sym, []).append(float(r["avg_fill_price"] or 0))
                elif side in ("SELL", "SHORT"):
                    sell_count += 1
            # Fallback: lê do model health monitor se disponível
            if hasattr(self, "_model_health"):
                mh = self._model_health
                wr_rolling = getattr(mh, "rolling_win_rate", None)
                n_rolling  = getattr(mh, "rolling_n", 0)
        except Exception:
            pass

        # ── Benchmark: preço atual vs 24h atrás (via OKX ticker) ─────────────
        benchmarks: dict[str, float] = {}   # symbol → pct_change_24h
        bench_lines: list[str] = []
        try:
            for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
                ticker = await self._okx.get_ticker(sym)
                if ticker:
                    last   = float(getattr(ticker, "last", 0) or 0)
                    open24 = float(getattr(ticker, "open_24h", 0) or 0)
                    if open24 > 0:
                        chg = (last - open24) / open24
                        benchmarks[sym] = chg
                        sign_b  = "+" if chg >= 0 else ""
                        sym_tag = sym.replace("-USDT", "")
                        bench_lines.append(f"{sym_tag}: {sign_b}{chg*100:.2f}%")
        except Exception:
            pass

        # Bot vs benchmark (avg) — quanto o bot ganhou vs simplesmente hold
        avg_bench = sum(benchmarks.values()) / len(benchmarks) if benchmarks else None
        bot_pct   = dpnl / total if total > 0 else 0.0
        alpha_txt = ""
        if avg_bench is not None:
            alpha    = bot_pct - avg_bench
            sign_al  = "+" if alpha >= 0 else ""
            alpha_txt = f"{sign_al}{alpha*100:.2f}%"

        # ── Regime atual ──────────────────────────────────────────────────────
        regime_now = ""
        try:
            regime_raw = await self._cache.get("regime:current")
            regime_now = regime_raw or ""
        except Exception:
            pass

        # ── Sinais do dia (contadores no Redis) ───────────────────────────────
        signals_eval  = 0
        signals_fired = 0
        try:
            ev_raw = await self._cache.get("signals:daily_evals")
            fi_raw = await self._cache.get("signals:daily_fired")
            signals_eval  = int(ev_raw)  if ev_raw  else 0
            signals_fired = int(fi_raw)  if fi_raw  else 0
        except Exception:
            pass
        signal_rate = signals_fired / signals_eval if signals_eval > 0 else None

        # ── Monta mensagem ────────────────────────────────────────────────────
        sign_d   = "+" if dpnl >= 0 else ""
        bench_str = " | ".join(bench_lines) if bench_lines else "N/A"

        fields: dict[str, str] = {
            "Portfolio":    f"${total:,.2f}",
            "P&L do Dia":   f"{sign_d}${dpnl:,.2f}",
            "Retorno Total":f"{ret*100:+.2f}%",
            "Drawdown":     f"{dd_pct*100:.2f}%",
            "Trades Hoje":  f"{trades_d} operações",
        }
        if wr_day is not None:
            fields["WR Hoje"] = f"{wr_day*100:.0f}% ({wins_d}/{trades_d})"
        if wr_rolling is not None and n_rolling > 0:
            fields["WR Rolling"] = f"{wr_rolling*100:.1f}% (n={n_rolling}) | Calibrado: 37.5%"
        if bench_str:
            fields["Benchmark 24h"] = bench_str
        if alpha_txt:
            fields["Alpha vs B&H"] = alpha_txt
        if signal_rate is not None:
            fields["Sinais"] = f"{signals_fired}/{signals_eval} avaliados ({signal_rate*100:.1f}%)"
        if regime_now:
            reg_emoji = self._trading_alerts._regime_emoji(regime_now)
            fields["Regime"] = f"{reg_emoji} {regime_now.replace('_', ' ')}"

        from ..alerts.base import Alert, AlertLevel
        await self._alert_channel.send(Alert(
            level=AlertLevel.INFO,
            title=f"{emoji} Digest Diário — {now.strftime('%d/%m/%Y')} {mode_tag}",
            message=(
                f"Resultado do dia: **{sign_d}${dpnl:,.2f}** "
                f"| Total: **${total:,.2f}** | Retorno: **{ret*100:+.2f}%**"
            ),
            fields=fields,
        ))

        # Persiste snapshot diário de equity para DrawdownEngine (janela 30d)
        try:
            import json as _json
            equity = self._portfolio.state.total_value or 0.0
            if equity > 0:
                dd_state = self._risk._drawdown._state
                # Adiciona o dia que está terminando
                dd_state.daily_equity.append((now.strftime("%Y-%m-%d"), equity))
                if len(dd_state.daily_equity) > 30:
                    dd_state.daily_equity = dd_state.daily_equity[-30:]
                await self._cache.set(
                    "drawdown:daily_equity",
                    _json.dumps([{"date": d, "value": v} for d, v in dd_state.daily_equity]),
                    ttl=30 * 86400,
                )
        except Exception as exc:
            logger.debug("DrawdownEngine: falha ao persistir equity snapshot: %s", exc)

        # Reseta contadores diários
        self._trading_alerts.reset_daily_stats()
        # Reseta contadores de sinais no Redis
        try:
            await self._cache.set("signals:daily_evals", "0", ttl=90000)
            await self._cache.set("signals:daily_fired", "0", ttl=90000)
        except Exception:
            pass

    # ── Paper fill simulator ──────────────────────────────────────────────────

    async def _simulate_paper_fills(self) -> None:
        """
        Detecta fills de ordens paper trading:
          - PAPER-xxx: simulação local (sem credenciais demo) — usa preço do cache
          - OKX demo IDs: verifica status real via OKX e propaga fill
          - Fallback: ordens SUBMITTED há >10min são verificadas via histórico OKX
        """
        if not settings.okx_paper_trading or settings.monitor_only:
            return
        import datetime as _dt

        from ..core.events import OrderFilledEvent
        from .models import OrderStatus
        open_orders = self._oms.get_open_orders()
        for order in open_orders:
            eid = order.exchange_order_id or ""

            if eid.startswith("PAPER-"):
                # Local simulation path (no demo credentials)
                price = await self._cache.get_price(order.symbol)
                if not price:
                    continue
                order.status          = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price  = price
                order.filled_at       = _dt.datetime.now(_dt.UTC)
                order.fees_paid       = round(price * order.quantity * OKX_TAKER_FEE, 6)
                logger.info(
                    "[PAPER-FILL] order=%s symbol=%s qty=%s price=%s",
                    order.client_order_id, order.symbol, order.quantity, price,
                )
                await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))
                asyncio.create_task(
                    self._oms._persist(order),
                    name=f"persist_fill_{order.client_order_id[:8]}",
                )
                await self._on_fill_update_portfolio(order)

            elif eid:
                # OKX demo path — query real status (timeout 8s para não bloquear o loop)
                try:
                    remote = await asyncio.wait_for(
                        self._okx.get_order_status(eid, symbol=order.symbol),
                        timeout=8.0,
                    )
                    remote_status = remote.get("status", "")
                    if remote_status == "filled":
                        order.status          = OrderStatus.FILLED
                        order.filled_quantity = float(remote.get("filled_qty") or order.quantity)
                        order.avg_fill_price  = float(remote.get("avg_px") or 0)
                        order.filled_at       = _dt.datetime.now(_dt.UTC)
                        order.fees_paid       = round(
                            order.avg_fill_price * order.filled_quantity * OKX_TAKER_FEE, 6
                        )
                        logger.info(
                            "[DEMO-FILL] order=%s symbol=%s side=%s qty=%s price=%s",
                            order.client_order_id, order.symbol, order.side,
                            order.filled_quantity, order.avg_fill_price,
                        )
                        await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))
                        # Persiste fill no DB e atualiza portfolio
                        asyncio.create_task(
                            self._oms._persist(order),
                            name=f"persist_fill_{order.client_order_id[:8]}",
                        )
                        await self._on_fill_update_portfolio(order)
                    elif remote_status == "cancelled":
                        order.status       = OrderStatus.CANCELLED
                        order.cancelled_at = _dt.datetime.now(_dt.UTC)
                        logger.warning(
                            "[DEMO-CANCELLED] order=%s symbol=%s side=%s — cancelada na OKX",
                            order.client_order_id, order.symbol, order.side,
                        )
                        asyncio.create_task(
                            self._oms._persist(order),
                            name=f"persist_cancel_{order.client_order_id[:8]}",
                        )
                    else:
                        # Status inesperado ou ainda pendente — loga para diagnóstico
                        logger.debug(
                            "[DEMO-STATUS] order=%s symbol=%s side=%s status=%r (aguardando fill)",
                            order.client_order_id, order.symbol, order.side, remote_status,
                        )
                except TimeoutError:
                    logger.warning(
                        "[DEMO-FILL-TIMEOUT] eid=%s symbol=%s side=%s — "
                        "verificação de fill excedeu 8s; será retentada no próximo ciclo",
                        eid, order.symbol, getattr(order, "side", "?"),
                    )
                except Exception as exc:
                    logger.warning(
                        "[DEMO-FILL-ERROR] eid=%s symbol=%s: %s — retentando no próximo ciclo",
                        eid, order.symbol, exc,
                    )

        # ── Fallback: ordens SUBMITTED há >10min → busca em histórico OKX ────
        # Cobre o caso em que get_order_status falha repetidamente mas OKX executou
        stale_cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=10)
        for order in self._oms.get_open_orders():
            eid = order.exchange_order_id or ""
            if eid.startswith("PAPER-") or not eid:
                continue
            sub_at = getattr(order, "submitted_at", None)
            if not sub_at or sub_at > stale_cutoff:
                continue  # não é stale
            try:
                logger.warning(
                    "[STALE-ORDER] %s %s %s submitted_at=%s — consultando histórico OKX",
                    order.client_order_id, order.side, order.symbol, sub_at,
                )
                filled_history = await asyncio.wait_for(
                    self._okx.get_filled_orders(limit=50),
                    timeout=8.0,
                )
                for rec in filled_history:
                    if str(rec.get("ordId", "")) == str(eid):
                        order.status          = OrderStatus.FILLED
                        order.filled_quantity = float(rec.get("fillSz", 0) or order.quantity)
                        order.avg_fill_price  = float(rec.get("avgPx", 0) or 0)
                        order.filled_at       = _dt.datetime.now(_dt.UTC)
                        order.fees_paid       = round(
                            order.avg_fill_price * order.filled_quantity * OKX_TAKER_FEE, 6
                        )
                        logger.info(
                            "[STALE-FILL-RECOVERED] order=%s symbol=%s side=%s qty=%s price=%s",
                            order.client_order_id, order.symbol, order.side,
                            order.filled_quantity, order.avg_fill_price,
                        )
                        await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))
                        asyncio.create_task(
                            self._oms._persist(order),
                            name=f"persist_stale_{order.client_order_id[:8]}",
                        )
                        await self._on_fill_update_portfolio(order)
                        break
            except Exception as exc:
                logger.warning("[STALE-ORDER-CHECK-ERROR] %s: %s", eid, exc)

    # ── Main heartbeat loop ────────────────────────────────────────────────────

    async def _on_fill_update_portfolio(self, order) -> None:
        """Atualiza portfolio e posições em memória após um fill."""
        from ..core.models import Position, PositionSide, PositionStatus

        try:
            side_raw = str(getattr(order, "side", "")).lower()
            is_buy   = side_raw in ("buy", "long")
            symbol   = order.symbol

            # Phase 17 — Slippage tracking
            fill_price = float(order.avg_fill_price or 0)
            if fill_price > 0:
                expected = self._exec_expected_price.pop(symbol, fill_price)
                try:
                    _exec_intel.record_fill(
                        symbol=symbol,
                        side="buy" if is_buy else "sell",
                        expected_price=expected,
                        fill_price=fill_price,
                        quantity=float(order.filled_quantity or order.quantity or 0),
                    )
                    await _exec_intel.flush_to_redis(symbol)
                except Exception as _exc:
                    logger.debug("record_fill error: %s", _exc)
            qty      = order.filled_quantity or order.quantity
            price    = order.avg_fill_price or 0.0
            fees     = order.fees_paid or 0.0

            if price <= 0 or qty <= 0:
                logger.warning(
                    "_on_fill_update_portfolio: fill inválido %s price=%.4f qty=%.6f — ignorado",
                    symbol, price, qty,
                )
                return

            if is_buy:
                # Abre ou amplia posição
                cost = qty * price + fees
                self._cash -= cost
                if self._cash < 0:
                    logger.warning(
                        "Cash negativo após compra %s: cash=%.2f cost=%.2f "
                        "(possível desync com exchange — aguardando sync periódico)",
                        symbol, self._cash, cost,
                    )
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    # Recalcula preço médio
                    total_qty = pos.quantity + qty
                    pos.avg_entry_price = (
                        (pos.quantity * pos.avg_entry_price + qty * price) / total_qty
                    )
                    pos.quantity    = total_qty
                    pos.total_fees += fees
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        strategy_id=order.strategy_id or "momentum_v2",
                        quantity=qty,
                        avg_entry_price=price,
                        total_fees=fees,
                    )
            else:
                # Fecha ou reduz posição
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    pnl = (price - pos.avg_entry_price) * qty - fees
                    pos.realized_pnl += pnl
                    self._cash += qty * price - fees
                    pos.quantity -= qty
                    if pos.quantity <= 1e-8:
                        pos.status = PositionStatus.CLOSED
                        del self._positions[symbol]
                        # Remove imediatamente do Redis — não esperar TTL expirar
                        try:
                            await self._cache.delete_position(symbol)
                        except Exception:
                            pass
                        logger.info(
                            "Posição fechada: %s | pnl=%.2f | cash=%.2f",
                            symbol, pnl, self._cash,
                        )
                else:
                    # Venda sem posição registrada (ex: restart)
                    self._cash += qty * price - fees
                    logger.warning(
                        "SELL sem posição em memória: %s | qty=%s price=%s "
                        "(possível restart entre BUY e SELL)",
                        symbol, qty, price,
                    )

            # Atualiza PortfolioEngine com preços atuais
            prices = {}
            for sym in self._positions:
                p = await self._cache.get_price(sym)
                if p:
                    prices[sym] = float(p)

            self._portfolio.update(
                positions=list(self._positions.values()),
                current_prices=prices,
                cash=max(self._cash, 0.0),
            )

            # Persiste posição no Redis para o dashboard
            for sym, pos in self._positions.items():
                cur_price = prices.get(sym, pos.avg_entry_price)
                await self._cache.set_position(sym, {
                    "symbol":           sym,
                    "side":             pos.side.value,
                    "quantity":         pos.quantity,
                    "avg_entry":        pos.avg_entry_price,
                    "current_price":    cur_price,
                    "notional":         pos.quantity * cur_price,
                    "unrealized_pnl":   (cur_price - pos.avg_entry_price) * pos.quantity,
                    "realized_pnl":     pos.realized_pnl,
                    "strategy_id":      pos.strategy_id,
                })

            logger.info(
                "Portfolio atualizado: cash=%.2f positions=%s",
                self._cash, list(self._positions.keys()),
            )
        except Exception as exc:
            logger.error("_on_fill_update_portfolio error: %s", exc, exc_info=True)

    async def _run_loop(self) -> None:
        # Auto-reset: se o kill switch ficou SOFT por heartbeat_timeout no boot
        # anterior, reseta agora que o loop está operacional.
        ks_event = getattr(self._kill_switch, "_current_event", None)
        ks_reason = getattr(ks_event, "reason", "") or ""
        if not self._kill_switch.allows_new_entries and "heartbeat" in ks_reason:
            logger.warning(
                "Auto-resetando kill switch SOFT de boot anterior (reason=%s)", ks_reason
            )
            self._kill_switch.reset_soft(reason="auto_reset_on_boot")
            if self._oms:
                self._oms.open_gate()

        # Contador para sincronização periódica de saldos OKX (a cada 5 min)
        _sync_tick = 0
        _SYNC_EVERY = 20   # 20 × 15s = 300s = 5 minutos

        while self._running:
            try:
                self._heartbeat.beat()
                self._infra_metrics.record_ws_message()

                # Simula fills para ordens paper
                await self._simulate_paper_fills()
                # Resumo diário
                await self._maybe_send_daily_summary()
                # Alertas: regime + drawdown + win rate
                await self._check_trading_alerts()

                # Sincronização periódica de saldos OKX (a cada 5 min)
                _sync_tick += 1
                if _sync_tick >= _SYNC_EVERY:
                    _sync_tick = 0
                    asyncio.create_task(
                        self._periodic_balance_sync(),
                        name="periodic_balance_sync",
                    )

                # Atualiza portfolio value no runner
                self._runner.update_portfolio_value(
                    self._portfolio.state.total_value
                )

                if not self._kill_switch.allows_any_operation:
                    logger.critical("HARD kill switch active — stopping loop")
                    break

                await asyncio.sleep(15)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("TradingLoop error: %s", exc, exc_info=True)
                self._infra_metrics.record_error()

    async def _periodic_balance_sync(self) -> None:
        """
        Sincronização leve com OKX a cada 5 minutos.
        Atualiza saldos no Redis e portfolio state sem recriar posições.
        """
        try:
            from ..recovery.exchange_sync import ExchangeSync
            sync = ExchangeSync(
                exchange=self._okx,
                db=None,        # sem DB — só Redis
                cache=self._cache,
                portfolio=self._portfolio,
            )
            total = await sync.sync_balances()
            if total > 0:
                self._runner.update_portfolio_value(total)
                if hasattr(self, "_cross_asset"):
                    self._cross_asset.set_portfolio_value(total)
                if hasattr(self, "_funding_harvest"):
                    self._funding_harvest.set_portfolio_value(total)
                if hasattr(self, "_sector_pairs"):
                    self._sector_pairs.set_portfolio_value(total)
                if hasattr(self, "_squeeze"):
                    self._squeeze.set_portfolio_value(total)
                if hasattr(self, "_rape"):
                    self._rape.set_portfolio_value(total)
                # Sincroniza cash interno com saldo USDT real da OKX
                usdt_cash = self._portfolio.state.cash_available
                if usdt_cash > 0:
                    self._cash = usdt_cash
                logger.debug("Sync periódico OKX: portfolio=%.2f USD cash=%.2f", total, self._cash)
        except Exception as exc:
            logger.debug("Sync periódico falhou: %s", exc)

    async def _on_ws_dead(self) -> None:
        # WebSocket morto NÃO dispara kill switch — market engine tem polling REST
        # como fallback a cada 15s. Kill switch causava mais dano do que o WS morto.
        logger.warning("WS dead — aguardando reconexão (polling REST continua ativo)")
        self._infra_metrics.record_error()
        asyncio.create_task(
            self._alert_channel.warning(
                title="⚠️ WebSocket Temporariamente Indisponível",
                message=(
                    "Conexão com OKX perdida. Usando polling REST como fallback."
                    " Trading continua."
                ),
            ),
            name="discord_ws_dead",
        )
