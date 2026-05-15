"""
Alert Listener — subscribes to bus events and sends alerts automatically.

Listens to RISK and SYSTEM topics.
Translates events into human-readable Discord alerts.

Events that trigger alerts:
  - KillSwitchEvent       → CRITICAL
  - RiskEvaluatedEvent    → WARNING (if action != NORMAL)
  - ReconciliationEvent   → WARNING (if divergences found)
  - SystemStatusEvent     → INFO/WARNING (on state changes)
"""

import asyncio
import logging

from ..core.bus import EventBus
from ..core.events import (
    KillSwitchEvent,
    ReconciliationEvent,
    RiskEvaluatedEvent,
    SystemStatusEvent,
    Topic,
)
from ..core.events.risk_events import RiskAction
from ..core.events.system_events import SystemStatus
from .base import AlertChannel

logger = logging.getLogger(__name__)


class AlertListener:
    """
    Background task that converts bus events into Discord alerts.
    """

    def __init__(self, bus: EventBus, channel: AlertChannel) -> None:
        self._bus = bus
        self._channel = channel
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Subscribe to RISK and SYSTEM topics
        risk_q = self._bus.subscribe(Topic.RISK)
        system_q = self._bus.subscribe(Topic.SYSTEM)

        self._tasks = [
            asyncio.create_task(self._consume_risk(risk_q), name="alert_risk"),
            asyncio.create_task(self._consume_system(system_q), name="alert_system"),
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
                            message=event.detail or "Divergências encontradas entre estado local e exchange.",
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
