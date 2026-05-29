"""
GlobalRiskGuard — guarda de risco compartilhada para estratégias v5.20.

As estratégias bypass-OMS (FundingHarvest, ShortSqueezeDetector,
SectorPairDetector, RegimeAwarePairEngine) não passam pelo OMS e por isso
não herdavam automaticamente os controles globais de risco.

Este módulo implementa um guard único consultado por todas as estratégias
antes de abrir qualquer nova posição:

  Verificações (em ordem de prioridade):
    1. KillSwitch HARD/SOFT  →  bloqueia imediatamente (kelly=0)
    2. CB all_assets_crash   →  bloqueia imediatamente (kelly=0)
    3. CB weekly_drawdown    →  permite mas reduz kelly×0.5
    4. CB consec_loss_strat  →  bloqueia a estratégia específica por 24h

Uso nas estratégias:
    result = await self._risk_guard.allows_new_entry(strategy_id="funding_harvest")
    if not result.allowed:
        logger.info("GlobalRiskGuard bloqueou: %s", result.reason)
        return
    # ajusta sizing pelo kelly_mult
    notional = base_notional * result.kelly_mult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskGuardResult:
    allowed:    bool
    kelly_mult: float          # 0.0 = bloqueado, 0.5 = defensivo, 1.0 = normal
    reason:     str = ""


class GlobalRiskGuard:
    """
    Guard injetado via trading_loop em todas as estratégias v5.20.
    Precisa de referência ao KillSwitch e ao cache Redis para ler
    o snapshot do AdvancedRiskManager.
    """

    def __init__(self, kill_switch, cache) -> None:
        self._ks    = kill_switch   # src.risk.kill_switch.KillSwitch
        self._cache = cache         # Redis cache

    async def allows_new_entry(
        self,
        strategy_id: str = "",
        symbol: str = "",
    ) -> RiskGuardResult:
        """
        Retorna RiskGuardResult com allowed=True/False e kelly_mult.

        Fail-open: se não há dados de risco, permite entrada com kelly normal
        (evita bloquear operação por indisponibilidade temporária do Redis).
        """
        # ── 1. KillSwitch — prioridade máxima ─────────────────────────────────
        if self._ks is not None:
            from .kill_switch import KillSwitchState
            state = self._ks.state
            if state == KillSwitchState.HARD:
                return RiskGuardResult(
                    allowed=False,
                    kelly_mult=0.0,
                    reason="KillSwitch HARD — operação completamente suspensa",
                )
            if state == KillSwitchState.SOFT:
                return RiskGuardResult(
                    allowed=False,
                    kelly_mult=0.0,
                    reason="KillSwitch SOFT — fechando posições, sem novas entradas",
                )

        # ── 2. Lê snapshot do AdvancedRiskManager ────────────────────────────
        snap: dict = {}
        try:
            raw = await self._cache.get("advanced_risk")
            if raw:
                import json
                snap = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception as exc:
            logger.debug("GlobalRiskGuard: falha ao ler advanced_risk: %s", exc)
            # fail-open
            return RiskGuardResult(allowed=True, kelly_mult=1.0, reason="no_snapshot")

        cb        = snap.get("circuit_breakers", {})
        checklist = cb.get("checklist", {})

        # ── 3. CB all_assets_crash — halt imediato ────────────────────────────
        all_crash = checklist.get("all_assets_crash", {})
        if all_crash.get("triggered"):
            return RiskGuardResult(
                allowed=False,
                kelly_mult=0.0,
                reason="ALL_CRASH — todos os ativos caindo >3% em 1H",
            )

        # ── 4. CB estratégia específica (consec_loss) ─────────────────────────
        if strategy_id:
            cb_strat = checklist.get(f"consec_loss_strat_{strategy_id}", {})
            if cb_strat.get("triggered"):
                return RiskGuardResult(
                    allowed=False,
                    kelly_mult=0.0,
                    reason=(
                        f"CONSEC_LOSS_STRAT — {cb_strat.get('consec_losses')} losses "
                        f"consecutivos em {strategy_id}"
                    ),
                )

        # ── 5. CB símbolo específico (consec_loss por símbolo) ────────────────
        if symbol:
            cb_sym = checklist.get(f"consec_loss_{symbol}", {})
            if cb_sym.get("triggered"):
                h = cb_sym.get("pause_remaining_h", 0)
                return RiskGuardResult(
                    allowed=False,
                    kelly_mult=0.0,
                    reason=(
                        f"CONSEC_LOSS — {cb_sym.get('consec_losses')} losses em "
                        f"{symbol} (pausa {h:.1f}h)"
                    ),
                )

        # ── 6. Weekly drawdown → modo defensivo (kelly × 0.5) ────────────────
        weekly_dd = checklist.get("weekly_drawdown", {})
        kelly_mult = 1.0
        if weekly_dd.get("triggered"):
            kelly_mult = float(weekly_dd.get("kelly_mult", 0.5))
            logger.debug(
                "GlobalRiskGuard [%s]: modo defensivo kelly×%.1f (weekly_dd=%.1f%%)",
                strategy_id or symbol,
                kelly_mult,
                weekly_dd.get("current_dd_pct", 0),
            )

        return RiskGuardResult(allowed=True, kelly_mult=kelly_mult, reason="ok")


# ── Singleton lazy — inicializado pelo trading_loop ──────────────────────────

class _LazyGuard:
    """Proxy seguro: retorna allowed=True se ainda não inicializado."""

    def __init__(self) -> None:
        self._inner: GlobalRiskGuard | None = None

    def init(self, kill_switch, cache) -> None:
        self._inner = GlobalRiskGuard(kill_switch, cache)
        logger.info("GlobalRiskGuard: inicializado")

    async def allows_new_entry(
        self,
        strategy_id: str = "",
        symbol: str = "",
    ) -> RiskGuardResult:
        if self._inner is None:
            return RiskGuardResult(allowed=True, kelly_mult=1.0, reason="not_init")
        return await self._inner.allows_new_entry(
            strategy_id=strategy_id,
            symbol=symbol,
        )


global_risk_guard = _LazyGuard()
