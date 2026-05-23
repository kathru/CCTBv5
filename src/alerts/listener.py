"""
Alert Listener — subscribes to bus events and sends alerts automatically.

Listens to RISK, SYSTEM, ORDER and FILL topics.
Translates events into human-readable Discord alerts.

Events that trigger alerts:
  - KillSwitchEvent       → CRITICAL
  - RiskEvaluatedEvent    → WARNING (if action != NORMAL)
  - ReconciliationEvent   → WARNING (if divergences found)
  - SystemStatusEvent     → INFO/WARNING (on state changes)
  - OrderFilledEvent      → INFO (trade executado)
  - OrderRejectedEvent    → WARNING (ordem rejeitada)
"""

import asyncio
import logging

from ..core.bus import EventBus
from ..core.events import (
    KillSwitchEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    ReconciliationEvent,
    RiskEvaluatedEvent,
    SystemStatusEvent,
    Topic,
)
from ..core.events.risk_events import RiskAction
from ..core.events.system_events import SystemStatus
from .base import AlertChannel

logger = logging.getLogger(__name__)

# Lados que representam abertura vs fechamento de posição
_OPEN_SIDES  = {"buy",  "BUY",  "long",  "LONG"}
_CLOSE_SIDES = {"sell", "SELL", "short", "SHORT"}


class AlertListener:
    """
    Background task that converts bus events into Discord alerts.
    """

    def __init__(self, bus: EventBus, channel: AlertChannel, cache=None) -> None:
        self._bus = bus
        self._channel = channel
        self._cache = cache       # opcional — enriquece notificações com P&L e regime
        self._tasks: list[asyncio.Task] = []
        self._running = False
        # Guarda dados de entrada para calcular P&L no fechamento
        self._entry_data: dict[str, dict] = {}   # symbol → {price, qty, notional}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        risk_q   = self._bus.subscribe(Topic.RISK)
        system_q = self._bus.subscribe(Topic.SYSTEM)
        fill_q   = self._bus.subscribe(Topic.FILL)

        self._tasks = [
            asyncio.create_task(self._consume_risk(risk_q),     name="alert_risk"),
            asyncio.create_task(self._consume_system(system_q), name="alert_system"),
            asyncio.create_task(self._consume_fills(fill_q),    name="alert_fills"),
        ]
        logger.info("AlertListener started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _consume_fills(self, queue: asyncio.Queue) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)

                if isinstance(event, OrderFilledEvent) and event.order:
                    await self._handle_fill(event.order)

                elif isinstance(event, OrderRejectedEvent) and event.order:
                    o = event.order
                    await self._channel.warning(
                        title=f"⛔ Ordem Rejeitada — {o.symbol}",
                        message=f"Motivo: {event.reason or 'desconhecido'}",
                        symbol=o.symbol,
                        order_id=o.client_order_id[:8] + "…",
                    )

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("AlertListener fill error: %s", exc)

    async def _handle_fill(self, o) -> None:
        """Envia notificação rica distinguindo entrada e saída, com P&L."""
        qty      = o.filled_quantity or 0.0
        price    = o.avg_fill_price  or 0.0
        notional = qty * price
        fee      = o.fees_paid or 0.0
        side_raw = (o.side.value if hasattr(o.side, "value") else str(o.side))
        is_open  = side_raw in _OPEN_SIDES

        if is_open:
            # ── Posição Aberta ───────────────────────────────────
            self._entry_data[o.symbol] = {
                "price": price, "qty": qty, "notional": notional,
            }

            # Tenta obter regime do cache (enriquece a mensagem)
            regime = "–"
            if self._cache:
                try:
                    pos = await self._cache.get_position(o.symbol)
                    regime = (pos or {}).get("regime", "–")
                except Exception:
                    pass

            fields = {
                "Símbolo":   o.symbol,
                "Quantidade": f"{qty:.6f}",
                "Preço":     f"${price:,.2f}",
                "Notional":  f"${notional:,.2f}",
                "Fee":       f"${fee:.4f}",
                "Regime":    regime,
                "Estratégia": o.strategy_id or "–",
            }
            await self._channel.info(
                title=f"🟢 Posição Aberta — {o.symbol}",
                message=f"**COMPRA** de {qty:.6f} unidades @ ${price:,.2f}",
                **fields,
            )

        else:
            # ── Posição Fechada ──────────────────────────────────
            entry = self._entry_data.pop(o.symbol, None)

            pnl_usdt = 0.0
            pnl_pct  = 0.0
            if entry and entry["price"] > 0:
                pnl_usdt = (price - entry["price"]) * qty - fee
                pnl_pct  = pnl_usdt / entry["notional"] * 100 if entry["notional"] else 0.0

            pnl_emoji = "🟢" if pnl_usdt >= 0 else "🔴"
            pnl_str   = f"{pnl_emoji} ${pnl_usdt:+,.2f} ({pnl_pct:+.2f}%)"

            fields = {
                "Símbolo":   o.symbol,
                "Quantidade": f"{qty:.6f}",
                "Preço saída": f"${price:,.2f}",
                "Preço entrada": f"${entry['price']:,.2f}" if entry else "–",
                "P&L":       pnl_str,
                "Fee":       f"${fee:.4f}",
                "Estratégia": o.strategy_id or "–",
            }
            await self._channel.info(
                title=f"🔔 Posição Fechada — {o.symbol}",
                message=f"**VENDA** de {qty:.6f} @ ${price:,.2f}\nResultado: {pnl_str}",
                **fields,
            )

    async def _consume_risk(self, queue: asyncio.Queue) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)

                if isinstance(event, KillSwitchEvent):
                    await self._channel.critical(
                        title="🚨 Kill Switch Ativado",
                        message=f"Modo: **{event.mode.upper()}**\nMotivo: {event.reason}",
                        mode=event.mode,
                        reason=event.reason,
                    )

                elif isinstance(event, RiskEvaluatedEvent):
                    if event.action != RiskAction.NORMAL:
                        await self._channel.warning(
                            title="⚠️ Ação de Risco",
                            message=f"Ação: **{event.action.upper()}**\n{event.reason}",
                            action=event.action,
                            drawdown=f"{event.drawdown_pct:.2%}",
                        )

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("AlertListener risk error: %s", exc)

    async def _consume_system(self, queue: asyncio.Queue) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)

                if isinstance(event, ReconciliationEvent):
                    if event.divergences_found > 0:
                        await self._channel.warning(
                            title="⚠️ Divergências na Reconciliação",
                            message=(
                                event.detail
                                or "Divergências encontradas entre estado local e exchange."
                            ),
                            divergences=str(event.divergences_found),
                            resolved=str(event.resolved),
                        )

                elif isinstance(event, SystemStatusEvent):
                    if event.status == SystemStatus.SUSPENDED:
                        await self._channel.critical(
                            title="🚨 Sistema Suspenso",
                            message=f"Motivo: {event.reason}",
                            status=event.status,
                        )
                    elif event.status == SystemStatus.RUNNING:
                        await self._channel.info(
                            title="✅ Sistema Operacional",
                            message="Boot concluído com sucesso.",
                            status=event.status,
                        )

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("AlertListener system error: %s", exc)
