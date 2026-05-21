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
from ..exchange.okx.client import OKXClient
from ..market.engine import MarketEngine
from ..metrics.infra_metrics import InfraMetrics
from ..oms.execution_router import ExecutionRouter
from ..oms.order_manager import OrderManager
from ..oms.position_monitor import PositionMonitor
from ..persistence import Cache, Database
from ..portfolio.engine import PortfolioEngine
from ..recovery.boot import BootSequence
from ..recovery.periodic_reconciler import PeriodicReconciler
from ..risk.engine import RiskContext, RiskEngine
from ..risk.kill_switch import KillSwitch
from ..strategies.meta_layer import MetaStrategyLayer
from ..strategies.ml.inference import MLInferenceEngine
from ..strategies.trend.trend_strategy import TrendStrategy
from ..strategies.runner import StrategyRunner
from ..watchdog.heartbeat import HeartbeatWatchdog
from ..watchdog.resource_watchdog import ResourceWatchdog
from ..watchdog.websocket_watchdog import WebSocketWatchdog
from .bus import EventBus
from .config import settings
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
            secret_key=settings.okx_demo_secret_key if (_paper and _has_demo) else settings.okx_secret_key,
            passphrase=settings.okx_demo_passphrase if (_paper and _has_demo) else settings.okx_passphrase,
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
            initial_capital=10000.0,
        )
        # Rastreia posições abertas em memória (atualizado a cada fill)
        self._positions: dict = {}   # symbol → Position (importado localmente nos métodos)
        self._cash: float = 10000.0

        # ── ML Inference ──────────────────────────────────────
        self._ml = MLInferenceEngine(models_dir=MODELS_DIR)

        # ── Meta Layer ────────────────────────────────────────
        self._meta = MetaStrategyLayer()

        # ── Strategy Runner ───────────────────────────────────
        self._runner = StrategyRunner(
            bus=self._bus,
            market=self._market,
            cache=self._cache,
        )
        v56 = TrendStrategy(symbols=SYMBOLS)
        self._runner.register(v56)
        self._meta.register(v56.strategy_id)

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
        await self._runner.start()
        await self._position_monitor.start()
        await self._reconciler.start()

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
            from ..core.models import Position, PositionSide
            for symbol, qty in sync_result.get("crypto_positions", {}).items():
                try:
                    price_raw = await self._cache.get_price(symbol)
                    price = float(price_raw) if price_raw else 0.0
                    if qty > 0:
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.LONG,
                            strategy_id="exchange_sync",
                            quantity=qty,
                            avg_entry_price=price,
                            total_fees=0.0,
                        )
                        logger.info(
                            "Posição carregada: %s qty=%.6f @ %.2f (disponível para venda)",
                            symbol, qty, price,
                        )
                except Exception as pos_exc:
                    logger.debug("Erro ao carregar posição %s: %s", symbol, pos_exc)

            # Cria ExitPlans para posições sincronizadas da exchange
            for symbol, qty in sync_result.get("crypto_positions", {}).items():
                try:
                    price_raw = await self._cache.get_price(symbol)
                    price = float(price_raw) if price_raw else 0.0
                    if price > 0 and qty > 0:
                        await self._position_monitor._create_plan(
                            symbol, qty, price, "exchange_sync"
                        )
                        logger.info(
                            "ExitPlan criado para posição sync: %s qty=%.4f entry=%.2f",
                            symbol, qty, price,
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
        Processa sinal da TrendStrategy v5.6 (Long + Flat — Brasil):
          LONG  → compra spot quando BTC > EMA50D
          FLAT  → vende spot quando BTC < EMA50D ou vol spike

          Pipeline: RiskEngine → Sizing (vol-target) → OMS
        """
        if settings.monitor_only:
            logger.debug(
                "[MONITOR_ONLY] %s %s score=%.3f regime=%s",
                event.signal.direction, event.signal.symbol,
                event.signal.calibrated_score, event.signal.regime,
            )
            return

        from ..core.models.signal import SignalDirection

        signal = event.signal

        # ── FLAT: fecha posição spot se existir ──────────────────────────────
        if signal.direction == SignalDirection.FLAT:
            await self._close_spot_position(signal.symbol, reason=signal.regime)
            return

        # ── LONG via spot ─────────────────────────────────────────────────────
        is_exit = False
        open_orders    = self._oms.get_open_orders()
        existing_plans = getattr(self._position_monitor, '_plans', {})

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
                o.symbol == signal.symbol and str(getattr(o, "side", "")).lower() in ("sell", "short")
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

        # 1. Avaliação de risco
        portfolio_value = self._portfolio.state.total_value or 85_000.0
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
        kelly    = signal.kelly_fraction or 0.05
        regime   = getattr(signal, "regime", "MEAN_REVERTING_CHOP")

        # Cap máximo de Kelly por regime
        KELLY_CAP = {
            "TREND_UP":               0.33,   # v5.6: vol-target sizing (já calibrado pela estratégia)
            "BTC_FLAT":               0.00,   # BTC não confirma → não entra
            "VOL_SPIKE":              0.00,   # vol alta → não entra
            "TREND_DOWN":             0.00,   # sinal de saída → não entra long
            "TREND_EXPANSION":        0.15,
            "VOLATILITY_COMPRESSION": 0.12,
            "TREND_EXHAUSTION":       0.10,
            "MEAN_REVERTING_CHOP":    0.08,
            "HIGH_CORRELATION_RISK":  0.05,
            "REVERSAL_1H":            0.08,
        }
        kelly     = min(kelly, KELLY_CAP.get(regime, 0.08))
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

        # Registra fatores do sinal no PositionMonitor ANTES do fill chegar
        # → ExitPlan usará sl_pct/tp_pct relativos ao fill_price real (reversão 1.5:1)
        signal_factors = getattr(signal, "factors", {}) or {}
        if signal_factors.get("sl_pct") and signal_factors.get("tp_pct"):
            self._position_monitor.set_pending_signal_factors(signal.symbol, signal_factors)

        # 3. Criar e submeter ordem via OMS
        await self._oms.create_order_from_signal(event, quantity)

        # Alerta Discord para sinais reais
        if not settings.okx_paper_trading:
            await self._alert_channel.info(
                title=f"Sinal: {signal.symbol}",
                message=(
                    f"Direção: {signal.direction} | "
                    f"Score: {signal.calibrated_score:.3f} | "
                    f"Qty: {quantity} | Notional: ${notional:.0f}"
                ),
            )

    # ── v5.6: Fechar posição spot (Long + Flat — sem derivativos) ────────────

    async def _close_spot_position(self, sym: str, reason: str = "FLAT") -> None:
        """
        Fecha posição spot para um símbolo quando sinal FLAT é emitido.
        Triggered quando: BTC < EMA50D | vol spike | BTC não confirma.
        """
        if sym not in self._positions:
            return   # sem posição, nada a fazer

        open_orders = self._oms.get_open_orders()
        sell_pending = any(
            o.symbol == sym and str(getattr(o, "side", "")).lower() in ("sell", "short")
            for o in open_orders
        )
        if sell_pending:
            logger.debug("Venda já pendente para %s — FLAT ignorado", sym)
            return

        pos = self._positions[sym]
        qty = round(pos.quantity, QTY_PRECISION.get(sym, 4))
        if qty < MIN_QTY.get(sym, 0.0001):
            return

        # Cria evento de saída
        from ..core.models.signal import SignalDirection
        from ..core.models import Signal
        from datetime import UTC, datetime
        flat_sig = Signal(
            strategy_id="trend_v56", symbol=sym,
            direction=SignalDirection.FLAT,
            timestamp=datetime.now(UTC),
            score=0.0, calibrated_score=0.0, confidence=0.0,
            expected_value=0.0, kelly_fraction=0.0,
            regime=reason, timeframe="1D", factors={"reason": reason},
        )
        fake_event = SignalEvent(signal=flat_sig)
        await self._oms.create_order_from_signal(fake_event, qty)
        logger.info("FLAT %s qty=%.6f reason=%s (EMA50D cross / vol spike)", sym, qty, reason)

    # ── Resumo diário Discord ─────────────────────────────────────────────────

    async def _maybe_send_daily_summary(self) -> None:
        """Envia resumo diário do portfolio ao Discord uma vez por dia (~00:00 UTC)."""
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

        p = self._portfolio.state
        pnl     = getattr(p, "realized_pnl", 0.0) or 0.0
        dpnl    = getattr(p, "daily_pnl", 0.0) or 0.0
        total   = getattr(p, "total_value", 0.0) or 0.0
        dd      = getattr(p, "drawdown_pct", 0.0) or 0.0
        ret     = getattr(p, "total_return_pct", 0.0) or 0.0
        sign    = "+" if dpnl >= 0 else ""
        emoji   = "📈" if dpnl >= 0 else "📉"
        await self._alert_channel.info(
            title=f"{emoji} Resumo Diário — {now.strftime('%d/%m/%Y')}",
            message=(
                f"P&L do dia: **{sign}${dpnl:,.2f}**\n"
                f"P&L total: ${pnl:,.2f} | Retorno: {ret*100:+.2f}%\n"
                f"Portfolio: ${total:,.2f} | Drawdown: {dd*100:.2f}%"
            ),
        )

    # ── Paper fill simulator ──────────────────────────────────────────────────

    async def _simulate_paper_fills(self) -> None:
        """
        Detecta fills de ordens paper trading:
          - PAPER-xxx: simulação local (sem credenciais demo) — usa preço do cache
          - OKX demo IDs: verifica status real via OKX e propaga fill
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
                order.fees_paid       = round(price * order.quantity * 0.001, 6)
                logger.info(
                    "[PAPER-FILL] order=%s symbol=%s qty=%s price=%s",
                    order.client_order_id, order.symbol, order.quantity, price,
                )
                await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))
                asyncio.create_task(self._oms._persist(order), name=f"persist_fill_{order.client_order_id[:8]}")
                await self._on_fill_update_portfolio(order)

            elif eid:
                # OKX demo path — query real status (timeout 5s para não bloquear o loop)
                try:
                    remote = await asyncio.wait_for(
                        self._okx.get_order_status(eid, symbol=order.symbol),
                        timeout=5.0,
                    )
                    if remote.get("status") == "filled":
                        order.status          = OrderStatus.FILLED
                        order.filled_quantity = float(remote.get("filled_qty") or order.quantity)
                        order.avg_fill_price  = float(remote.get("avg_px") or 0)
                        order.filled_at       = _dt.datetime.now(_dt.UTC)
                        order.fees_paid       = round(
                            order.avg_fill_price * order.filled_quantity * 0.001, 6
                        )
                        logger.info(
                            "[DEMO-FILL] order=%s symbol=%s qty=%s price=%s",
                            order.client_order_id, order.symbol,
                            order.filled_quantity, order.avg_fill_price,
                        )
                        await self._bus.publish(Topic.FILL, OrderFilledEvent(order=order))
                        # Persiste fill no DB e atualiza portfolio
                        asyncio.create_task(self._oms._persist(order), name=f"persist_fill_{order.client_order_id[:8]}")
                        await self._on_fill_update_portfolio(order)
                except TimeoutError:
                    logger.debug("Fill check timeout eid=%s", eid)
                except Exception as exc:
                    logger.debug("Fill check failed eid=%s: %s", eid, exc)

    # ── Main heartbeat loop ────────────────────────────────────────────────────

    async def _on_fill_update_portfolio(self, order) -> None:
        """Atualiza portfolio e posições em memória após um fill."""
        from ..core.models import Position, PositionSide, PositionStatus

        try:
            side_raw = str(getattr(order, "side", "")).lower()
            is_buy   = side_raw in ("buy", "long")
            symbol   = order.symbol
            qty      = order.filled_quantity or order.quantity
            price    = order.avg_fill_price or 0.0
            fees     = order.fees_paid or 0.0

            if is_buy:
                # Abre ou amplia posição
                cost = qty * price + fees
                self._cash -= cost
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
                else:
                    # Venda sem posição registrada (ex: restart)
                    self._cash += qty * price - fees

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
                logger.debug("Sync periódico OKX: portfolio=%.2f USD", total)
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
                message="Conexão com OKX perdida. Usando polling REST como fallback. Trading continua.",
            ),
            name="discord_ws_dead",
        )
