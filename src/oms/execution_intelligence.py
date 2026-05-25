"""
ExecutionIntelligence — Phase 17: Smart Order Routing + Slippage Feedback.

Problemas resolvidos:
  1. Sempre market order → paga taker fee + sofre spread em condições ilíquidas
  2. Sem feedback de execução → sistema cego à qualidade real das entradas
  3. Sizing não considera se execução está degradando

Componentes:

  A. SmartOrderRouter
     Decide maker (limit) vs taker (market) por spread + volatilidade:
       spread < SPREAD_MAKER_THR  → limit post_only (maker, -0.02% fee OKX)
       spread >= SPREAD_TAKER_THR → market (taker, +0.08% fee OKX)
       Entre os dois → depende do regime: expansão → taker, compressão → maker

     Bônus maker: evita pagar 0.10% de spread médio + economiza taker fee.
     Risco maker: non-fill se mercado move antes do fill.
     Timeout: se limit não preencher em LIMIT_TIMEOUT_S → converte para market.

  B. SlippageTracker
     Mede slippage por trade: (fill_price - expected_price) / expected_price
     Mantém janela rolling de 20 trades por símbolo.
     Calcula:
       avg_slippage    : média dos últimos 20 trades (bps)
       worst_slippage  : pior trade na janela
       quality_score   : 1.0 = execução perfeita, 0.0 = slippage catastrófico

  C. ExecutionQualityMult (integração com SizingEngine)
     quality_score → multiplicador de sizing:
       quality > 0.85 → mult 1.0  (execução boa, sizing normal)
       quality 0.70–0.85 → mult 0.85
       quality 0.50–0.70 → mult 0.70  (slippage moderado, reduz 30%)
       quality < 0.50 → mult 0.50  (slippage alto, half-size)

Fee OKX Spot (tier VIP0):
  Maker: -0.02% (rebate)
  Taker: +0.08%
  Spread médio BTC: ~0.01–0.05%; ETH/SOL: ~0.02–0.10%
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Thresholds de decisão maker/taker ────────────────────────────────────────

SPREAD_MAKER_THR  = 0.0003   # 3 bps → abaixo disso usa maker (spread justo)
SPREAD_TAKER_THR  = 0.0010   # 10 bps → acima disso sempre taker (mercado aberto)
LIMIT_TIMEOUT_S   = 30       # segundos antes de converter limit → market

# ── Slippage thresholds ───────────────────────────────────────────────────────

SLIPPAGE_WARN_BPS   = 10.0   # 10 bps = 0.10% — começa a reduzir sizing
SLIPPAGE_SEVERE_BPS = 25.0   # 25 bps = 0.25% — half-size
SLIPPAGE_WINDOW     = 20     # trades para rolling average

# ── Quality → sizing mult ─────────────────────────────────────────────────────

QUALITY_BANDS: list[tuple[float, float]] = [
    (0.85, 1.00),   # excellent → sizing normal
    (0.70, 0.85),   # good → -15%
    (0.50, 0.70),   # moderate → -30%
    (0.00, 0.50),   # poor → -50%
]


@dataclass
class OrderDecision:
    """Resultado do SmartOrderRouter."""
    order_type:    str           # "market" | "limit" | "post_only"
    limit_price:   float | None  # preço para limit orders (None = market)
    fee_est_pct:   float         # fee estimado (negativo = rebate maker)
    reason:        str
    spread_pct:    float


@dataclass
class SlippageRecord:
    """Registro de execução de um trade."""
    symbol:         str
    side:           str          # "buy" | "sell"
    expected_price: float
    fill_price:     float
    quantity:       float
    executed_at:    datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def slippage_bps(self) -> float:
        """Slippage em basis points. Positivo = pagou mais (compra) ou recebeu menos (venda)."""
        if self.expected_price <= 0:
            return 0.0
        raw = (self.fill_price - self.expected_price) / self.expected_price
        # Para venda: slippage invertido (fill abaixo do esperado = ruim)
        if self.side == "sell":
            raw = -raw
        return round(raw * 10_000, 2)   # em bps


class SlippageTracker:
    """Rastreia qualidade de execução por símbolo (janela rolling)."""

    def __init__(self, window: int = SLIPPAGE_WINDOW) -> None:
        self._window  = window
        self._records: dict[str, deque[SlippageRecord]] = {}

    def record(
        self,
        symbol:         str,
        side:           str,
        expected_price: float,
        fill_price:     float,
        quantity:       float,
    ) -> SlippageRecord:
        rec = SlippageRecord(
            symbol=symbol, side=side,
            expected_price=expected_price,
            fill_price=fill_price,
            quantity=quantity,
        )
        if symbol not in self._records:
            self._records[symbol] = deque(maxlen=self._window)
        self._records[symbol].append(rec)
        logger.info(
            "Slippage %s %s: expected=%.4f fill=%.4f slippage=%.2f bps",
            side, symbol, expected_price, fill_price, rec.slippage_bps,
        )
        return rec

    def stats(self, symbol: str) -> dict:
        """Estatísticas de execução para um símbolo."""
        recs = list(self._records.get(symbol, []))
        if not recs:
            return {
                "n": 0, "avg_slippage_bps": 0.0,
                "worst_slippage_bps": 0.0, "quality_score": 1.0,
                "sizing_mult": 1.0,
            }
        slippages = [r.slippage_bps for r in recs]
        avg  = sum(slippages) / len(slippages)
        worst = max(slippages)
        quality = self._quality_from_avg(avg)
        sizing  = self._sizing_mult(quality)
        return {
            "n":                   len(recs),
            "avg_slippage_bps":    round(avg, 2),
            "worst_slippage_bps":  round(worst, 2),
            "quality_score":       round(quality, 3),
            "sizing_mult":         round(sizing, 3),
        }

    def all_stats(self) -> dict:
        return {sym: self.stats(sym) for sym in self._records}

    def quality_sizing_mult(self, symbol: str) -> float:
        """Retorna sizing mult direto — usado pelo SizingEngine."""
        return self.stats(symbol)["sizing_mult"]

    @staticmethod
    def _quality_from_avg(avg_bps: float) -> float:
        """Converte slippage médio em quality score [0,1]."""
        if avg_bps <= 0:
            return 1.0   # slippage negativo = melhor que esperado
        if avg_bps <= SLIPPAGE_WARN_BPS:
            return 1.0 - (avg_bps / SLIPPAGE_WARN_BPS) * 0.15   # degrada suave
        if avg_bps <= SLIPPAGE_SEVERE_BPS:
            return 0.85 - ((avg_bps - SLIPPAGE_WARN_BPS) / (SLIPPAGE_SEVERE_BPS - SLIPPAGE_WARN_BPS)) * 0.35
        return max(0.0, 0.50 - (avg_bps - SLIPPAGE_SEVERE_BPS) * 0.01)

    @staticmethod
    def _sizing_mult(quality: float) -> float:
        for threshold, mult in QUALITY_BANDS:
            if quality >= threshold:
                return mult
        return 0.50


class SmartOrderRouter:
    """
    Decide tipo de ordem (maker/taker) com base em spread e volatilidade.

    Integração: chamado em _process_signal antes de criar a ordem,
    retorna OrderDecision com order_type e limit_price para passar ao OMS.
    """

    def decide(
        self,
        symbol:        str,
        side:          str,           # "buy" | "sell"
        mid_price:     float,
        spread_pct:    float,         # bid-ask spread / mid
        atr_pct:       float = 0.01,  # ATR% atual (proxy de volatilidade)
        regime:        str = "",
    ) -> OrderDecision:
        """
        Regra de decisão:
          spread < 3 bps  → maker (limit post_only) — mercado justo
          spread > 10 bps → taker (market) — spread muito largo
          3–10 bps        → depende:
            TREND_EXPANSION / PANIC → market (velocidade importa)
            outros → maker (economiza fee)
        """
        aggressive_regimes = {"TREND_EXPANSION", "PANIC_LIQUIDATION", "BEAR_TREND"}
        fee_maker = -0.0002   # -2 bps (rebate OKX maker)
        fee_taker =  0.0008   # +8 bps (taker OKX)

        if spread_pct < SPREAD_MAKER_THR:
            # Spread justo → maker seguro
            limit_px = self._limit_price(side, mid_price, spread_pct)
            return OrderDecision(
                order_type="post_only", limit_price=limit_px,
                fee_est_pct=fee_maker,
                reason=f"spread={spread_pct*10000:.1f}bps < {SPREAD_MAKER_THR*10000:.0f}bps → maker",
                spread_pct=spread_pct,
            )

        if spread_pct >= SPREAD_TAKER_THR or regime in aggressive_regimes:
            # Spread largo OU regime agressivo → market
            return OrderDecision(
                order_type="market", limit_price=None,
                fee_est_pct=fee_taker,
                reason=f"spread={spread_pct*10000:.1f}bps≥{SPREAD_TAKER_THR*10000:.0f}bps or regime={regime} → taker",
                spread_pct=spread_pct,
            )

        # Zona intermediária (3–10 bps)
        if atr_pct > 0.015:
            # Alta volatilidade → velocidade > economia de fee
            return OrderDecision(
                order_type="market", limit_price=None,
                fee_est_pct=fee_taker,
                reason=f"atr={atr_pct*100:.2f}% alto → taker",
                spread_pct=spread_pct,
            )

        # Baixa volatilidade, spread moderado → tenta maker
        limit_px = self._limit_price(side, mid_price, spread_pct)
        return OrderDecision(
            order_type="post_only", limit_price=limit_px,
            fee_est_pct=fee_maker,
            reason=f"spread={spread_pct*10000:.1f}bps moderado, atr={atr_pct*100:.2f}% baixo → maker",
            spread_pct=spread_pct,
        )

    @staticmethod
    def _limit_price(side: str, mid: float, spread_pct: float) -> float:
        """
        Preço limit levemente dentro do spread para maximizar chance de fill.
          Compra: mid - 25% do spread (mais próximo do ask, mas como maker)
          Venda:  mid + 25% do spread
        """
        offset = mid * spread_pct * 0.25
        if side == "buy":
            return round(mid - offset, 8)
        return round(mid + offset, 8)


class ExecutionIntelligence:
    """
    Façade — agrega SmartOrderRouter + SlippageTracker.
    Instância única usada pelo TradingLoop.
    """

    def __init__(self) -> None:
        self.router  = SmartOrderRouter()
        self.tracker = SlippageTracker()

    def decide_order_type(
        self,
        symbol:     str,
        side:       str,
        mid_price:  float,
        spread_pct: float,
        atr_pct:    float = 0.01,
        regime:     str   = "",
    ) -> OrderDecision:
        decision = self.router.decide(
            symbol=symbol, side=side, mid_price=mid_price,
            spread_pct=spread_pct, atr_pct=atr_pct, regime=regime,
        )
        logger.info(
            "SmartRouter %s %s: %s price=%s | %s",
            side, symbol, decision.order_type,
            f"{decision.limit_price:.4f}" if decision.limit_price else "market",
            decision.reason,
        )
        return decision

    def record_fill(
        self,
        symbol:         str,
        side:           str,
        expected_price: float,
        fill_price:     float,
        quantity:       float,
    ) -> float:
        """Registra fill e retorna slippage em bps."""
        rec = self.tracker.record(symbol, side, expected_price, fill_price, quantity)
        return rec.slippage_bps

    def sizing_mult(self, symbol: str) -> float:
        """Multiplicador de sizing baseado em qualidade de execução histórica."""
        return self.tracker.quality_sizing_mult(symbol)

    def status(self) -> dict:
        return {
            "slippage_by_symbol": self.tracker.all_stats(),
            "router_config": {
                "spread_maker_thr_bps": SPREAD_MAKER_THR * 10_000,
                "spread_taker_thr_bps": SPREAD_TAKER_THR * 10_000,
                "limit_timeout_s":      LIMIT_TIMEOUT_S,
            },
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
execution_intelligence = ExecutionIntelligence()
