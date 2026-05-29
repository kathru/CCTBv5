"""
StrategyRouter — Fase 1: Consultor de aptidão por regime/macro  (v5.21.0)

Cada estratégia consulta o roteador ANTES de abrir uma posição:

    advice = await strategy_router.consult(
        strategy_id="funding_harvest",
        regime="BEAR_TREND",
        macro_state="RISK_OFF",
    )
    if not advice.allowed:
        return
    notional *= advice.kelly_mult
    threshold *= advice.thr_mult
    score_min *= advice.score_mult

O roteador não substitui nenhuma lógica existente — apenas devolve
multiplicadores que cada estratégia aplica livremente.

Matriz de aptidão (regime × macro → por estratégia):

    Regimes (position_monitor):
        TREND_EXPANSION, VOLATILITY_COMPRESSION, MEAN_REVERTING_CHOP,
        TREND_EXHAUSTION, HIGH_CORRELATION_RISK, BEAR_TREND, PANIC_LIQUIDATION

    Macro states (meta_regime):
        RISK_ON, ALTSEASON, SIDEWAYS, TRANSITION, RISK_OFF

Valores de aptidão:
    0.0 = estratégia não deve atuar neste regime/macro
    0.5 = aptidão reduzida — parâmetros conservadores
    1.0 = aptidão normal
    1.2 = aptidão elevada — condições ideais

Endpoint para dashboard: GET /api/analytics/strategy_router
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Regime e macro labels ─────────────────────────────────────────────────────

KNOWN_REGIMES = {
    "TREND_EXPANSION",
    "VOLATILITY_COMPRESSION",
    "MEAN_REVERTING_CHOP",
    "TREND_EXHAUSTION",
    "HIGH_CORRELATION_RISK",
    "BEAR_TREND",
    "PANIC_LIQUIDATION",
}

KNOWN_MACROS = {
    "RISK_ON",
    "ALTSEASON",
    "SIDEWAYS",
    "TRANSITION",
    "RISK_OFF",
}

# ── Tabela de aptidão ─────────────────────────────────────────────────────────
#
# Estrutura: APTITUDE[strategy_id][regime] → base_aptitude (float)
# A aptitude base é depois modulada pelo macro_state.
#
# Lógica por estratégia:
#   funding_harvest  — funciona melhor em regimes não-direcional e bull;
#                      ruim em BEAR/PANIC (funding negativo → shorts pagam)
#   short_squeeze    — precisa de BEAR + momentum (squeeze após capitulação);
#                      ótimo em PANIC_LIQUIDATION e MEAN_REVERTING_CHOP
#   sector_pair      — mercado neutro; melhor em CHOP/COMPRESSION, ruim em PANIC
#   regime_pair      — cross-asset; bom em TREND e EXPANSION, ruim em PANIC
#   momentum         — trend-following; ótimo em TREND_EXPANSION, ruim em CHOP

APTITUDE: dict[str, dict[str, float]] = {
    "funding_harvest": {
        "TREND_EXPANSION":       1.2,   # bull — funding alto, ideal
        "VOLATILITY_COMPRESSION": 1.0,  # acumulação, funding moderado
        "MEAN_REVERTING_CHOP":   0.8,   # neutro
        "TREND_EXHAUSTION":      0.5,   # bull a perder força, funding caindo
        "HIGH_CORRELATION_RISK": 0.5,   # correlações altas = risco sistêmico
        "BEAR_TREND":            0.0,   # funding negativo → posição short paga
        "PANIC_LIQUIDATION":     0.0,   # funding negativo extremo, não entrar
    },
    "short_squeeze": {
        "TREND_EXPANSION":       0.5,   # tendência alta, poucos shorts p/ squeeze
        "VOLATILITY_COMPRESSION": 0.5,  # aguardando catalisador
        "MEAN_REVERTING_CHOP":   1.0,   # alto OI + shorts = squeeze frequente
        "TREND_EXHAUSTION":      1.2,   # esgotamento + short interest alto
        "HIGH_CORRELATION_RISK": 0.8,   # correlação alta pode ampliar squeeze
        "BEAR_TREND":            1.2,   # capitulação cria shorts que viram alvo
        "PANIC_LIQUIDATION":     1.0,   # pós-pânico: squeeze de recuperação
    },
    "sector_pair": {
        "TREND_EXPANSION":       0.8,   # pares divergem no trend
        "VOLATILITY_COMPRESSION": 1.2,  # compressão → mean-reversion entre pares
        "MEAN_REVERTING_CHOP":   1.2,   # ideal para market-neutral
        "TREND_EXHAUSTION":      1.0,   # reversão de pares comum
        "HIGH_CORRELATION_RISK": 0.5,   # alta correlação anula edge de par
        "BEAR_TREND":            0.8,   # pares ainda funcionam em bear
        "PANIC_LIQUIDATION":     0.0,   # correlação extrema, par desaparece
    },
    "regime_pair": {
        "TREND_EXPANSION":       1.2,   # cross-asset diverge em trend forte
        "VOLATILITY_COMPRESSION": 0.8,  # pouca divergência cross-asset
        "MEAN_REVERTING_CHOP":   0.5,   # chop → par sem sinal claro
        "TREND_EXHAUSTION":      1.0,   # reversão cross-asset
        "HIGH_CORRELATION_RISK": 0.5,   # ativos correlacionados = sem edge
        "BEAR_TREND":            1.0,   # divergência BTC vs alts funciona
        "PANIC_LIQUIDATION":     0.0,   # tudo cai junto, sem edge de par
    },
    "momentum": {
        "TREND_EXPANSION":       1.2,   # regime ideal para trend-following
        "VOLATILITY_COMPRESSION": 0.8,  # breakout iminente mas não confirmado
        "MEAN_REVERTING_CHOP":   0.0,   # momentum em chop = whipsaw
        "TREND_EXHAUSTION":      0.5,   # trend perdendo força
        "HIGH_CORRELATION_RISK": 0.8,   # correlação alta pode ampliar sinal
        "BEAR_TREND":            1.0,   # SHORT momentum em bear
        "PANIC_LIQUIDATION":     0.5,   # pânico = stop-hunt, cuidado
    },
}

# ── Modificador macro por estratégia ─────────────────────────────────────────
#
# MACRO_MOD[strategy_id][macro_state] → multiplicador sobre aptitude base
# Resultado final: aptitude = APTITUDE[sid][regime] * MACRO_MOD[sid][macro]

MACRO_MOD: dict[str, dict[str, float]] = {
    "funding_harvest": {
        "RISK_ON":    1.15,   # bull market = funding alto
        "ALTSEASON":  1.20,   # alts com funding alto
        "SIDEWAYS":   1.00,
        "TRANSITION": 0.85,   # incerteza, funding instável
        "RISK_OFF":   0.60,   # fuga para USDT, funding cai
    },
    "short_squeeze": {
        "RISK_ON":    0.80,   # poucos shorts, menos squeeze
        "ALTSEASON":  0.90,
        "SIDEWAYS":   1.00,
        "TRANSITION": 1.10,
        "RISK_OFF":   1.20,   # muitos shorts = maior potencial de squeeze
    },
    "sector_pair": {
        "RISK_ON":    1.00,
        "ALTSEASON":  1.10,   # pares entre alts
        "SIDEWAYS":   1.10,
        "TRANSITION": 0.90,
        "RISK_OFF":   0.80,
    },
    "regime_pair": {
        "RISK_ON":    1.10,
        "ALTSEASON":  0.90,   # correlação alta entre alts
        "SIDEWAYS":   0.80,
        "TRANSITION": 1.00,
        "RISK_OFF":   1.20,   # divergência BTC vs alts aumenta
    },
    "momentum": {
        "RISK_ON":    1.15,
        "ALTSEASON":  1.20,
        "SIDEWAYS":   0.70,
        "TRANSITION": 0.85,
        "RISK_OFF":   0.80,   # momentum curto em bear
    },
}

# ── Limiares de aptidão para parâmetros ──────────────────────────────────────
# aptitude >= APT_FULL  → parâmetros normais
# APT_REDUCED <= aptitude < APT_FULL → parâmetros conservadores
# aptitude < APT_REDUCED → entrada bloqueada

APT_FULL    = 0.70   # aptidão mínima para parâmetros normais
APT_REDUCED = 0.35   # aptidão mínima para entrada (conservadora)


# ── Dataclass de resposta ─────────────────────────────────────────────────────

@dataclass
class RouterAdvice:
    """Conselho do StrategyRouter para uma estratégia antes de entrar."""
    strategy_id: str
    regime:      str
    macro_state: str
    aptitude:    float       # valor final calculado (0.0–1.4)
    allowed:     bool        # False = não entrar (aptidão < limiar)
    score_mult:  float = 1.0 # multiplica o score mínimo de entrada
    kelly_mult:  float = 1.0 # multiplica o Kelly sizing
    thr_mult:    float = 1.0 # multiplica o threshold de entrada
    gate_mult:   float = 1.0 # multiplica gates de filtros
    ev_mult:     float = 1.0 # multiplica EV mínimo
    reason:      str   = ""
    params:      dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "regime":      self.regime,
            "macro_state": self.macro_state,
            "aptitude":    round(self.aptitude, 3),
            "allowed":     self.allowed,
            "score_mult":  round(self.score_mult, 3),
            "kelly_mult":  round(self.kelly_mult, 3),
            "thr_mult":    round(self.thr_mult, 3),
            "gate_mult":   round(self.gate_mult, 3),
            "ev_mult":     round(self.ev_mult, 3),
            "reason":      self.reason,
        }


# ── StrategyRouter ────────────────────────────────────────────────────────────

class StrategyRouter:
    """
    Consultor de regime/macro para estratégias v5.20.

    Não substitui nenhuma estratégia — apenas calcula multiplicadores
    adaptativos baseados no regime atual do mercado.
    """

    def __init__(self, cache=None) -> None:
        self._cache = cache

    def init(self, cache) -> None:
        """Inicializado pelo trading_loop após criar o cache Redis."""
        self._cache = cache
        logger.info("StrategyRouter: inicializado")

    async def _get_current_regime_macro(self) -> tuple[str, str, float]:
        """
        Lê regime e macro state do Redis e do signal_audit_log.
        Retorna (regime_micro, macro_state, threshold_mult).

        Fontes:
          - macro_state: Redis "meta_regime" → campo "regime" (RISK_ON, RISK_OFF, etc.)
          - regime_micro: última entrada do signal_audit_log com regime válido
        """
        regime_name  = "UNKNOWN"
        macro_name   = "UNKNOWN"
        thr_mult_raw = 1.0

        if self._cache is None:
            return regime_name, macro_name, thr_mult_raw

        try:
            # Macro state via meta_regime Redis
            meta_raw = await self._cache.get("meta_regime")
            if meta_raw:
                meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                macro_name   = meta.get("regime", "UNKNOWN")
                thr_mult_raw = float(meta.get("threshold_mult", 1.0))
        except Exception as exc:
            logger.debug("StrategyRouter: falha ao ler meta_regime: %s", exc)

        # Regime micro via signal_audit_log (regime mais recente avaliado)
        try:
            from ..monitoring.signal_log import signal_audit_log
            entries = list(signal_audit_log._entries)
            # Itera do mais recente para o mais antigo
            for e in reversed(entries):
                r = getattr(e, "regime", None) or ""
                if r and r in KNOWN_REGIMES:
                    regime_name = r
                    break
        except Exception as exc:
            logger.debug("StrategyRouter: falha ao ler regime do signal_audit_log: %s", exc)

        return regime_name, macro_name, thr_mult_raw

    def _compute_advice(
        self,
        strategy_id: str,
        regime: str,
        macro_state: str,
        thr_mult_raw: float = 1.0,
    ) -> RouterAdvice:
        """Calcula RouterAdvice a partir da tabela de aptidão."""

        # Aptidão base por regime
        base_apt = APTITUDE.get(strategy_id, {}).get(regime, 0.7)
        # Modificador macro
        macro_mod = MACRO_MOD.get(strategy_id, {}).get(macro_state, 1.0)
        aptitude  = round(base_apt * macro_mod, 4)

        # Bloqueado — aptidão insuficiente
        if aptitude < APT_REDUCED:
            return RouterAdvice(
                strategy_id=strategy_id,
                regime=regime,
                macro_state=macro_state,
                aptitude=aptitude,
                allowed=False,
                score_mult=1.0,
                kelly_mult=0.0,
                thr_mult=1.0,
                reason=(
                    f"aptidão={aptitude:.2f} < {APT_REDUCED} "
                    f"[{regime}/{macro_state}]"
                ),
            )

        # Modo conservador — aptidão reduzida mas acima do mínimo
        if aptitude < APT_FULL:
            # Parâmetros conservadores: threshold mais alto, Kelly menor
            ratio = aptitude / APT_FULL  # 0.5 → 0.71, 0.6 → 0.86
            return RouterAdvice(
                strategy_id=strategy_id,
                regime=regime,
                macro_state=macro_state,
                aptitude=aptitude,
                allowed=True,
                score_mult=1.0 + (1.0 - ratio) * 0.15,  # até +15% no score mínimo
                kelly_mult=0.5 + ratio * 0.4,             # 0.5 a 0.9
                thr_mult=thr_mult_raw * (1.0 + (1.0 - ratio) * 0.10),
                gate_mult=1.0 + (1.0 - ratio) * 0.10,
                ev_mult=1.0 + (1.0 - ratio) * 0.10,
                reason=f"conservador aptidão={aptitude:.2f} [{regime}/{macro_state}]",
            )

        # Modo normal ou elevado
        # aptitude >= 1.0 → parâmetros mais permissivos (maior Kelly, menor thr)
        if aptitude >= 1.0:
            boost = min(aptitude - 1.0, 0.25)  # até 0.25 de bônus
            return RouterAdvice(
                strategy_id=strategy_id,
                regime=regime,
                macro_state=macro_state,
                aptitude=aptitude,
                allowed=True,
                score_mult=max(0.85, 1.0 - boost * 0.4),  # score mínimo ligeiramente menor
                kelly_mult=min(1.3, 1.0 + boost * 1.2),    # Kelly até 130%
                thr_mult=thr_mult_raw * max(0.88, 1.0 - boost * 0.5),
                gate_mult=max(0.90, 1.0 - boost * 0.3),
                ev_mult=max(0.90, 1.0 - boost * 0.3),
                reason=f"elevado aptidão={aptitude:.2f} [{regime}/{macro_state}]",
            )

        # Normal (APT_FULL <= aptitude < 1.0)
        return RouterAdvice(
            strategy_id=strategy_id,
            regime=regime,
            macro_state=macro_state,
            aptitude=aptitude,
            allowed=True,
            score_mult=1.0,
            kelly_mult=1.0,
            thr_mult=thr_mult_raw,
            gate_mult=1.0,
            ev_mult=1.0,
            reason=f"normal aptidão={aptitude:.2f} [{regime}/{macro_state}]",
        )

    async def consult(
        self,
        strategy_id: str,
        regime: str | None = None,
        macro_state: str | None = None,
    ) -> RouterAdvice:
        """
        Consulta o roteador. Se regime/macro_state não fornecidos, lê do Redis.

        Fail-open: se cache indisponível, retorna RouterAdvice neutro (allowed=True,
        todos os multiplicadores = 1.0) para não bloquear operação por falta de dados.
        """
        if self._cache is None and regime is None:
            return RouterAdvice(
                strategy_id=strategy_id,
                regime="UNKNOWN",
                macro_state="UNKNOWN",
                aptitude=1.0,
                allowed=True,
                reason="not_init",
            )

        # Lê do Redis se não fornecido
        thr_mult_raw = 1.0
        if regime is None or macro_state is None:
            r, m, thr_mult_raw = await self._get_current_regime_macro()
            regime      = regime      or r
            macro_state = macro_state or m

        # Normaliza
        regime_norm = regime.upper()     if regime      else "UNKNOWN"
        macro_norm  = macro_state.upper() if macro_state else "UNKNOWN"

        # Se regime/macro desconhecidos → fail-open conservador
        if regime_norm not in KNOWN_REGIMES or macro_norm not in KNOWN_MACROS:
            return RouterAdvice(
                strategy_id=strategy_id,
                regime=regime_norm,
                macro_state=macro_norm,
                aptitude=0.7,
                allowed=True,
                score_mult=1.0,
                kelly_mult=0.8,
                thr_mult=thr_mult_raw,
                reason=f"regime/macro desconhecido [{regime_norm}/{macro_norm}]",
            )

        advice = self._compute_advice(
            strategy_id=strategy_id,
            regime=regime_norm,
            macro_state=macro_norm,
            thr_mult_raw=thr_mult_raw,
        )

        logger.debug(
            "StrategyRouter [%s]: regime=%s macro=%s apt=%.2f allowed=%s "
            "kelly=%.2f thr=%.2f — %s",
            strategy_id, regime_norm, macro_norm,
            advice.aptitude, advice.allowed,
            advice.kelly_mult, advice.thr_mult,
            advice.reason,
        )
        return advice

    async def snapshot(self) -> dict:
        """
        Retorna snapshot completo do estado atual do roteador para o dashboard.
        """
        regime, macro_state, thr_raw = await self._get_current_regime_macro()

        all_strategies = list(APTITUDE.keys())
        advices = {}
        for sid in all_strategies:
            adv = await self.consult(strategy_id=sid, regime=regime, macro_state=macro_state)
            advices[sid] = adv.to_dict()

        return {
            "regime":       regime,
            "macro_state":  macro_state,
            "thr_mult_raw": round(thr_raw, 4),
            "strategies":   advices,
            "matrix_summary": {
                sid: {
                    rg: round(
                        APTITUDE.get(sid, {}).get(rg, 0.7)
                        * MACRO_MOD.get(sid, {}).get(macro_state, 1.0),
                        3,
                    )
                    for rg in KNOWN_REGIMES
                }
                for sid in all_strategies
            },
        }


# ── Singleton lazy ────────────────────────────────────────────────────────────

class _LazyRouter:
    """Proxy seguro: retorna advice neutro se ainda não inicializado."""

    def __init__(self) -> None:
        self._inner: StrategyRouter | None = None

    def init(self, cache) -> None:
        self._inner = StrategyRouter(cache=cache)
        logger.info("StrategyRouter singleton: inicializado")

    async def consult(
        self,
        strategy_id: str,
        regime: str | None = None,
        macro_state: str | None = None,
    ) -> RouterAdvice:
        if self._inner is None:
            return RouterAdvice(
                strategy_id=strategy_id,
                regime="UNKNOWN",
                macro_state="UNKNOWN",
                aptitude=1.0,
                allowed=True,
                reason="not_init",
            )
        return await self._inner.consult(
            strategy_id=strategy_id,
            regime=regime,
            macro_state=macro_state,
        )

    async def snapshot(self) -> dict:
        if self._inner is None:
            return {"error": "not_init"}
        return await self._inner.snapshot()


strategy_router = _LazyRouter()
