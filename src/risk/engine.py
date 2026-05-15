"""
Risk Engine — orchestrates all risk checks and emits RiskEvaluatedEvent.

Called every cycle BEFORE any new order is allowed.
Strategies and OMS do NOT call risk checks directly — they
subscribe to RiskEvaluatedEvent from the bus.

Decision flow:
  1. Kill switch (overrides everything)
  2. Daily drawdown limit
  3. Drawdown acceleration
  4. Total exposure limit
  5. Strategy cooldown (per strategy)
  → Emit RiskEvaluatedEvent with action
"""

import logging
from dataclasses import dataclass

from ..core.bus import EventBus
from ..core.events import KillSwitchEvent, RiskAction, RiskEvaluatedEvent, Topic
from ..core.events.risk_events import KillSwitchMode
from ..core.models import Position
from .cooldown import CooldownEngine
from .drawdown import DrawdownEngine
from .exposure import ExposureEngine
from .kill_switch import KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class RiskContext:
    """Input snapshot for a risk evaluation cycle."""
    portfolio_value: float
    open_positions: list[Position]
    strategy_id: str | None = None   # if checking for a specific strategy


class RiskEngine:
    """
    Central risk orchestrator.
    Evaluate once per cycle, emit result to bus.
    """

    def __init__(
        self,
        bus: EventBus,
        kill_switch: KillSwitch,
        exposure: ExposureEngine | None = None,
        drawdown: DrawdownEngine | None = None,
        cooldown: CooldownEngine | None = None,
    ) -> None:
        self._bus = bus
        self._kill_switch = kill_switch
        self._exposure = exposure or ExposureEngine()
        self._drawdown = drawdown or DrawdownEngine()
        self._cooldown = cooldown or CooldownEngine()

    async def evaluate(self, ctx: RiskContext) -> RiskAction:
        """
        Run all risk checks. Emit event. Return action.
        Callers should await this and respect the returned action.
        """
        # Update drawdown state with current portfolio value
        self._drawdown.update(ctx.portfolio_value)

        action, reason = await self._run_checks(ctx)

        event = RiskEvaluatedEvent(
            action=action,
            reason=reason,
            drawdown_pct=self._drawdown.daily_drawdown,
        )
        await self._bus.publish(Topic.RISK, event)

        if action in {RiskAction.CLOSE, RiskAction.SUSPEND}:
            logger.warning(
                "Risk action=%s reason=%s drawdown=%.2f%%",
                action, reason, self._drawdown.daily_drawdown * 100,
            )

        return action

    async def _run_checks(
        self, ctx: RiskContext
    ) -> tuple[RiskAction, str]:

        # 1. Kill switch — highest priority
        if not self._kill_switch.allows_any_operation:
            return RiskAction.SUSPEND, "kill_switch_hard"

        if not self._kill_switch.allows_new_entries:
            return RiskAction.CLOSE, "kill_switch_soft"

        # 2. Daily drawdown limit
        breached, reason = self._drawdown.is_daily_limit_breached()
        if breached:
            self._kill_switch.trigger_soft(reason=reason)
            await self._bus.publish(
                Topic.RISK,
                KillSwitchEvent(mode=KillSwitchMode.SOFT, reason=reason),
            )
            return RiskAction.CLOSE, reason

        # 3. Drawdown acceleration
        accelerating, reason = self._drawdown.is_accelerating()
        if accelerating:
            return RiskAction.REDUCE, reason

        # 4. Peak drawdown
        peak_breached, reason = self._drawdown.is_peak_limit_breached()
        if peak_breached:
            self._kill_switch.trigger_soft(reason=reason)
            return RiskAction.CLOSE, reason

        # 5. Strategy cooldown
        if ctx.strategy_id:
            allowed, reason = self._cooldown.is_allowed(ctx.strategy_id)
            if not allowed:
                return RiskAction.SUSPEND, reason

        return RiskAction.NORMAL, ""

    def record_trade_result(
        self, strategy_id: str, is_win: bool
    ) -> None:
        """Called by OMS after each fill to update cooldown state."""
        if is_win:
            self._cooldown.record_win(strategy_id)
        else:
            activated = self._cooldown.record_loss(strategy_id)
            if activated:
                logger.warning(
                    "Cooldown activated strategy=%s", strategy_id
                )

    def trigger_kill_switch(
        self, mode: str, reason: str
    ) -> None:
        """External trigger — websocket dead, latency spike, etc."""
        if mode == "hard":
            self._kill_switch.trigger_hard(reason)
        else:
            self._kill_switch.trigger_soft(reason)

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch
