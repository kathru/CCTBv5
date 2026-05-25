"""
PortfolioAllocator — Portfolio Intelligence Layer (Phase 15).

Problema resolvido:
  Antes: cada sinal era avaliado e dimensionado INDEPENDENTEMENTE.
  Agora:  todos os sinais ativos são RANKEADOS por edge relativo e o capital
          é distribuído proporcionalmente — o melhor edge recebe mais capital.

Pipeline:
  1. Sinal chega → registrado no buffer (por símbolo)
  2. Antes do sizing → consulta allocate(signal, portfolio_state)
  3. Allocator calcula edge_score ponderado por:
       a. calibrated_score      (força do sinal)
       b. regime_quality        (qualidade do regime para trading)
       c. corr_penalty          (penaliza correlação marginal alta)
       d. opportunity_cost      (penaliza signal se capital já alocado em melhor edge)
  4. Retorna kelly ajustado + breakdown para diagnóstico

Capital budget por regime (total alocável ao portfólio):
  TREND_EXPANSION:        25% (mercado favorável — sobe budget total)
  VOLATILITY_COMPRESSION: 18%
  TREND_EXHAUSTION:       12%
  MEAN_REVERTING_CHOP:    10%
  HIGH_CORRELATION_RISK:  8%

Exemplo prático:
  Budget EXPANSION = 25%, 3 sinais: BTC edge=0.72, ETH edge=0.61, SOL edge=0.45
  Total edge = 1.78
  BTC kelly  = 0.25 × (0.72/1.78) = 10.1%
  ETH kelly  = 0.25 × (0.61/1.78) = 8.6%
  SOL kelly  = 0.25 × (0.45/1.78) = 6.3%
  (vs hoje: todos recebem kelly independente, podendo somar 35%+ de exposição)
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

# ── Budget total de Kelly por regime ─────────────────────────────────────────
# Representa o máximo que o PORTFÓLIO pode ter alocado simultaneamente.
# Distribui entre os sinais ativos proporcionalmente ao edge relativo.

PORTFOLIO_KELLY_BUDGET: dict[str, float] = {
    "TREND_EXPANSION":        0.25,   # ambiente favorável → budget maior
    "VOLATILITY_COMPRESSION": 0.18,   # aguardando rompimento → moderado
    "TREND_EXHAUSTION":       0.12,   # defensivo → conservador
    "MEAN_REVERTING_CHOP":    0.10,   # lateral → mínimo
    "HIGH_CORRELATION_RISK":  0.08,   # correlação alta → mínimo
    "BEAR_TREND":             0.00,   # bloqueado
    "PANIC_LIQUIDATION":      0.00,   # bloqueado
}

# Kelly máximo por posição individual (cap de concentração)
POSITION_KELLY_CAP: dict[str, float] = {
    "TREND_EXPANSION":        0.15,
    "VOLATILITY_COMPRESSION": 0.12,
    "TREND_EXHAUSTION":       0.10,
    "MEAN_REVERTING_CHOP":    0.08,
    "HIGH_CORRELATION_RISK":  0.05,
    "BEAR_TREND":             0.00,
    "PANIC_LIQUIDATION":      0.00,
}

# Qualidade do regime — escala edge pelo ambiente de mercado
REGIME_QUALITY: dict[str, float] = {
    "TREND_EXPANSION":        1.00,   # ideal para momentum
    "VOLATILITY_COMPRESSION": 0.85,   # aceitável
    "TREND_EXHAUSTION":       0.60,   # degradado
    "MEAN_REVERTING_CHOP":    0.45,   # ruim para momentum
    "HIGH_CORRELATION_RISK":  0.40,   # alto risco sistêmico
    "BEAR_TREND":             0.00,
    "PANIC_LIQUIDATION":      0.00,
}

# Correlação entre pares (fallback estático — atualizado pelo AdvancedRiskManager)
DEFAULT_CORR: dict[tuple[str, str], float] = {
    ("BTC-USDT", "ETH-USDT"): 0.92,
    ("BTC-USDT", "SOL-USDT"): 0.87,
    ("ETH-USDT", "SOL-USDT"): 0.90,
}

# Sinal válido por no máximo 2 horas (evita entradas atrasadas)
SIGNAL_TTL = timedelta(hours=2)


@dataclass
class SignalSlot:
    """Sinal registrado no buffer do allocator."""
    symbol:           str
    calibrated_score: float
    regime:           str
    strategy_id:      str
    registered_at:    datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        return (datetime.now(UTC) - self.registered_at) > SIGNAL_TTL


@dataclass
class AllocationResult:
    """Resultado da alocação para um sinal específico."""
    symbol:           str
    original_kelly:   float
    allocated_kelly:  float
    edge_score:       float
    regime_quality:   float
    corr_penalty:     float
    rank:             int        # 1 = melhor edge do ciclo
    total_signals:    int
    budget_used_pct:  float      # % do budget do portfólio usado
    reason:           str        # descrição da decisão


class PortfolioAllocator:
    """
    Rankeia sinais por edge relativo e distribui capital proporcionalmente.

    Thread-safe: usa asyncio (single-threaded event loop), sem locks necessários.
    """

    def __init__(self) -> None:
        # Buffer de sinais ativos (um por símbolo)
        self._slots: dict[str, SignalSlot] = {}
        # Correlação live (atualizada pelo AdvancedRiskManager via update_correlations)
        self._corr: dict[tuple[str, str], float] = dict(DEFAULT_CORR)
        # Símbolos com posição aberta (não competem por capital novo)
        self._open_symbols: set[str] = set()

    # ── API pública ───────────────────────────────────────────────────────────

    def register_signal(
        self,
        symbol:           str,
        calibrated_score: float,
        regime:           str,
        strategy_id:      str,
    ) -> None:
        """Registra sinal no buffer — sobrescreve se já houver para o símbolo."""
        self._slots[symbol] = SignalSlot(
            symbol=symbol,
            calibrated_score=calibrated_score,
            regime=regime,
            strategy_id=strategy_id,
        )
        # Limpa slots expirados a cada registro
        self._purge_expired()

    def update_correlations(self, corr_matrix: dict[str, dict[str, float]]) -> None:
        """Atualiza matriz de correlação com dados ao vivo do AdvancedRiskManager."""
        for sym1, row in corr_matrix.items():
            for sym2, val in row.items():
                if sym1 != sym2:
                    key = tuple(sorted([sym1, sym2]))
                    self._corr[key] = val  # type: ignore[index]

    def update_open_positions(self, open_symbols: set[str]) -> None:
        """Atualiza conjunto de símbolos com posição aberta."""
        self._open_symbols = set(open_symbols)

    def allocate(
        self,
        symbol:         str,
        original_kelly: float,
        regime:         str,
    ) -> AllocationResult:
        """
        Calcula kelly alocado para o símbolo com base no ranking de edge.

        Chamado em _process_signal antes do sizing.
        """
        budget = PORTFOLIO_KELLY_BUDGET.get(regime, 0.10)
        cap    = POSITION_KELLY_CAP.get(regime, 0.08)

        if budget == 0.0:
            return AllocationResult(
                symbol=symbol, original_kelly=original_kelly,
                allocated_kelly=0.0, edge_score=0.0,
                regime_quality=0.0, corr_penalty=1.0,
                rank=1, total_signals=1,
                budget_used_pct=0.0,
                reason="regime bloqueado",
            )

        # Coleta sinais ativos (não expirados, sem posição aberta)
        active = {
            s: slot for s, slot in self._slots.items()
            if not slot.is_expired and s not in self._open_symbols
        }

        if not active:
            # Nenhum concorrente — usa kelly original (com cap)
            allocated = min(original_kelly, cap)
            return AllocationResult(
                symbol=symbol, original_kelly=original_kelly,
                allocated_kelly=allocated, edge_score=original_kelly,
                regime_quality=REGIME_QUALITY.get(regime, 0.7),
                corr_penalty=1.0, rank=1, total_signals=1,
                budget_used_pct=allocated / budget if budget > 0 else 0,
                reason="sinal único — kelly original aplicado",
            )

        # Computa edge_score para cada sinal ativo
        scores = {}
        for sym, slot in active.items():
            rq   = REGIME_QUALITY.get(slot.regime, 0.7)
            corr = self._marginal_corr_penalty(sym, set(active.keys()) - {sym})
            scores[sym] = slot.calibrated_score * rq * corr

        total_edge = sum(scores.values())
        if total_edge <= 0:
            allocated = min(original_kelly, cap)
            return AllocationResult(
                symbol=symbol, original_kelly=original_kelly,
                allocated_kelly=allocated, edge_score=0.0,
                regime_quality=REGIME_QUALITY.get(regime, 0.7),
                corr_penalty=1.0, rank=1, total_signals=len(active),
                budget_used_pct=allocated / budget,
                reason="edge total zero — kelly original aplicado",
            )

        # Ranking: posição do símbolo por edge decrescente
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank   = next((i + 1 for i, (s, _) in enumerate(ranked) if s == symbol), 1)

        # Alocação proporcional ao edge relativo
        my_edge    = scores.get(symbol, 0.0)
        proportion = my_edge / total_edge
        allocated  = min(budget * proportion, cap)

        # Penalização extra para 3º colocado ou abaixo (opportunity cost)
        if rank >= 3 and len(active) >= 3:
            allocated *= 0.70
            reason = f"rank #{rank}/{len(active)} — opportunity cost -30%"
        elif rank == 2:
            reason = f"rank #{rank}/{len(active)} — alocação proporcional"
        else:
            reason = f"rank #1/{len(active)} — melhor edge do ciclo"

        my_slot   = active.get(symbol)
        rq        = REGIME_QUALITY.get(regime, 0.7)
        corr_pen  = self._marginal_corr_penalty(symbol, set(active.keys()) - {symbol})

        logger.info(
            "PortfolioAllocator: %s rank=%d/%d edge=%.3f (score=%.3f rq=%.2f corr=%.2f) "
            "kelly %.1f%% → %.1f%% (budget=%.0f%% prop=%.0f%%)",
            symbol, rank, len(active), my_edge,
            my_slot.calibrated_score if my_slot else 0,
            rq, corr_pen,
            original_kelly * 100, allocated * 100,
            budget * 100, proportion * 100,
        )

        return AllocationResult(
            symbol=symbol,
            original_kelly=original_kelly,
            allocated_kelly=round(allocated, 4),
            edge_score=round(my_edge, 4),
            regime_quality=round(rq, 3),
            corr_penalty=round(corr_pen, 3),
            rank=rank,
            total_signals=len(active),
            budget_used_pct=round(allocated / budget, 3) if budget > 0 else 0,
            reason=reason,
        )

    def status(self) -> dict:
        """Estado atual do allocator — para diagnóstico e dashboard."""
        self._purge_expired()
        active = {
            s: slot for s, slot in self._slots.items()
            if not slot.is_expired
        }
        return {
            "active_signals": {
                sym: {
                    "calibrated_score": slot.calibrated_score,
                    "regime":           slot.regime,
                    "age_s":            int((datetime.now(UTC) - slot.registered_at).total_seconds()),
                }
                for sym, slot in active.items()
            },
            "open_positions":  list(self._open_symbols),
            "correlation_map": {
                f"{k[0]}|{k[1]}": round(v, 4)
                for k, v in self._corr.items()
            },
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _marginal_corr_penalty(self, symbol: str, other_symbols: set[str]) -> float:
        """
        Penaliza edge se o símbolo está altamente correlacionado com outros sinais ativos.

        Lógica:
          avg_corr = média das correlações com os outros símbolos ativos
          penalty  = 1 - 0.5 × max(0, avg_corr - 0.7)
          Correlação 0.70 → penalty 1.00 (sem penalização)
          Correlação 0.90 → penalty 0.90 (10% de penalização)
          Correlação 1.00 → penalty 0.85 (15% de penalização)
        """
        if not other_symbols:
            return 1.0

        corrs = []
        for other in other_symbols:
            key = tuple(sorted([symbol, other]))
            c   = self._corr.get(key, 0.85)  # type: ignore[arg-type]
            corrs.append(c)

        avg_corr = sum(corrs) / len(corrs)
        penalty  = 1.0 - 0.5 * max(0.0, avg_corr - 0.70)
        return max(0.60, round(penalty, 3))  # mínimo 60% para não zerar edge

    def _purge_expired(self) -> None:
        expired = [s for s, slot in self._slots.items() if slot.is_expired]
        for s in expired:
            del self._slots[s]
            logger.debug("PortfolioAllocator: slot expirado removido — %s", s)


# ── Singleton ─────────────────────────────────────────────────────────────────
portfolio_allocator = PortfolioAllocator()
