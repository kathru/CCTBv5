"""
HoldEngine — Conviction Score por trade aberto.

Avalia a cada ciclo se uma posição aberta deve ser MANTIDA ou FECHADA,
baseado em 6 componentes de saúde do mercado — não em tempo decorrido.

Componentes (pesos):
  1. Trend Persistence  (25%) — SMA structure + Higher Highs/Lows
  2. Relative Strength  (20%) — M7 vs BTC/peers
  3. Volatility Health  (20%) — M8 state machine
  4. Market Breadth     (15%) — MetaRegime breadth feature
  5. Volume Behavior    (10%) — volume confirmando direção do preço
  6. Distribution       (10%) — ausência de distribuição (vol alto + price stalling)

Estados por Conviction Score (0–100):
  HOLD forte (≥ 85): trailing 2.5× ATR — winners respiram
  HOLD       (70–84): trailing 2.0× ATR — espaço confortável
  WATCH      (50–69): trailing 1.0× ATR — tighten
  ALERT      (30–49): trailing 0.5× ATR — muito apertado
  EXIT       (< 30):  sair imediatamente

  Running mode (TP convertido em TREND_EXPANSION):
  Todos os estados ganham +0.5× ATR de folga adicional.

Proteções anti-bag-holding (triggers de EXIT independente do score):
  P1: preço < entrada AND conviction < 50%  → sair (trade negativo sem convicção)
  P2: conviction < 30% por 3+ ciclos seguidos → sair (sem recuperação)
  P3: preço < entrada × 0.92 AND conviction < 60% → sair (colapso crypto)

Filosofia:
  NÃO é "vender só no lucro".
  É "vender apenas quando o risco estrutural superar a expectativa futura".
  Enquanto existir persistência, breadth saudável, força relativa e
  estrutura de tendência → MANTER independente do PnL momentâneo.
"""

import json
import logging
from dataclasses import dataclass

from ..core.models import Candle

logger = logging.getLogger(__name__)

# ── Pesos dos componentes ─────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "trend_persistence": 0.25,
    "relative_strength": 0.20,
    "vol_health":        0.20,
    "breadth":           0.15,
    "volume_behavior":   0.10,
    "distribution":      0.10,
}

# ── Thresholds de estado ──────────────────────────────────────────────────────

CONVICTION_HOLD  = 70.0
CONVICTION_WATCH = 50.0
CONVICTION_ALERT = 30.0

# ── Trailing ATR multiplier — função contínua por conviction score ────────────
#
# Filosofia de convexidade:
#   HOLD forte → trailing LARGO  (winners respiram, trends correm)
#   Deterioração → trailing APERTA progressivamente
#   Running mode (TP convertido) → ganha +0.5× de folga extra
#
# Normal mode:
#   score ≥ 85  → 2.5× ATR  (HOLD forte)
#   score 70–84 → 2.0× ATR  (HOLD normal)
#   score 50–69 → 1.0× ATR  (WATCH)
#   score 30–49 → 0.5× ATR  (ALERT)
#   score < 30  → 0.3× ATR  (EXIT iminente)
#
# Running mode (adiciona +0.5×):
#   score ≥ 85  → 3.0× ATR
#   score 70–84 → 2.5× ATR
#   score 50–69 → 1.5× ATR
#   score 30–49 → 1.0× ATR

def trail_atr_mult(conviction_score: float, running_mode: bool = False) -> float:
    """
    ATR multiplier para o trailing stop baseado no conviction score.
    Quanto mais alta a convicção, mais largo o trailing → winners respiram.
    running_mode adiciona +0.5× em todos os níveis (posição já convertida).
    """
    if conviction_score >= 85:
        base = 2.5
    elif conviction_score >= 70:
        base = 2.0
    elif conviction_score >= 50:
        base = 1.0
    elif conviction_score >= 30:
        base = 0.5
    else:
        base = 0.3
    return base + (0.5 if running_mode else 0.0)

# ── Proteções anti-bag-holding ────────────────────────────────────────────────

BAG_P1_CONVICTION  = 50.0   # P1: trade negativo AND conviction < 50%
BAG_P2_STREAK      = 3      # P2: conviction < 30% por N ciclos → EXIT
BAG_P3_DRAWDOWN    = 0.92   # P3: queda > 8% vs entrada
BAG_P3_CONVICTION  = 60.0   # P3: AND conviction < 60% → EXIT (colapso crypto)

# ── Volatility state → health score ──────────────────────────────────────────

VOL_STATE_HEALTH: dict[str, float] = {
    "EXPANDING":      0.90,   # breakout ativo — ótimo para holders
    "TREND":          0.80,   # tendência direcional — bom para holders
    "COMPRESSED":     0.50,   # aguardando breakout — neutro
    "MEAN_REVERTING": 0.30,   # lateralização — ruim para momentum
    "CHAOTIC":        0.05,   # caótico — péssimo, sair
    "UNKNOWN":        0.50,
}


# ── Resultado da avaliação ────────────────────────────────────────────────────

@dataclass
class ConvictionResult:
    """Resultado completo da avaliação Hold Engine para um trade."""
    score:       float           # 0–100
    state:       str             # HOLD / WATCH / ALERT / EXIT
    components:  dict            # breakdown dos 6 componentes (0–1 cada)
    exit_reason: str | None = None   # razão se state == EXIT


# ── HoldEngine ────────────────────────────────────────────────────────────────

class HoldEngine:
    """
    Avalia o Conviction Score de um trade aberto.
    Instância global — avalia qualquer símbolo via market + cache.
    """

    def __init__(self, market, cache) -> None:
        self._market = market
        self._cache  = cache

    # ── API pública ───────────────────────────────────────────────────────────

    async def evaluate(
        self,
        symbol:                str,
        entry_price:           float,
        low_conviction_streak: int,
    ) -> ConvictionResult:
        """
        Avalia o conviction score do trade em `symbol`.
        `low_conviction_streak`: quantos ciclos consecutivos conviction < 30%.
        """
        components = await self._compute_components(symbol, entry_price)

        # Score final ponderado (0–100)
        raw = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
        score = round(max(0.0, min(100.0, raw * 100)), 1)

        # ── Proteções anti-bag-holding ────────────────────────────────────────
        current_price = await self._get_price(symbol)
        if current_price and current_price > 0 and entry_price > 0:

            # P3: queda > 8% E conviction < 60% → colapso crypto
            if current_price < entry_price * BAG_P3_DRAWDOWN:
                if score < BAG_P3_CONVICTION:
                    logger.warning(
                        "%s HoldEngine P3: queda>8%% (%.2f→%.2f) conviction=%.0f<60 → EXIT",
                        symbol, entry_price, current_price, score,
                    )
                    return ConvictionResult(
                        score=score, state="EXIT",
                        components=components,
                        exit_reason="bag_hold_p3_crypto_collapse",
                    )

            # P1: trade negativo E conviction < 50%
            if current_price < entry_price and score < BAG_P1_CONVICTION:
                logger.warning(
                    "%s HoldEngine P1: trade negativo (%.2f<%.2f) conviction=%.0f<50 → EXIT",
                    symbol, current_price, entry_price, score,
                )
                return ConvictionResult(
                    score=score, state="EXIT",
                    components=components,
                    exit_reason="bag_hold_p1_negative_low_conviction",
                )

        # P2: conviction < 30% por N ciclos consecutivos
        if score < CONVICTION_ALERT and low_conviction_streak >= BAG_P2_STREAK:
            logger.warning(
                "%s HoldEngine P2: conviction=%.0f<30 por %d ciclos → EXIT",
                symbol, score, low_conviction_streak,
            )
            return ConvictionResult(
                score=score, state="EXIT",
                components=components,
                exit_reason=f"bag_hold_p2_streak_{low_conviction_streak}_cycles",
            )

        # ── Estado por threshold ──────────────────────────────────────────────
        if score >= CONVICTION_HOLD:
            state = "HOLD"
        elif score >= CONVICTION_WATCH:
            state = "WATCH"
        elif score >= CONVICTION_ALERT:
            state = "ALERT"
        else:
            state = "EXIT"

        exit_reason = "conviction_collapse" if state == "EXIT" else None

        if state != "HOLD":
            logger.info(
                "%s HoldEngine: conviction=%.0f state=%s | "
                "trend=%.2f rs=%.2f vol=%.2f breadth=%.2f vol_beh=%.2f dist=%.2f",
                symbol, score, state,
                components["trend_persistence"],
                components["relative_strength"],
                components["vol_health"],
                components["breadth"],
                components["volume_behavior"],
                components["distribution"],
            )

        return ConvictionResult(
            score=score,
            state=state,
            components=components,
            exit_reason=exit_reason,
        )

    # ── Componentes ───────────────────────────────────────────────────────────

    async def _compute_components(
        self, symbol: str, entry_price: float
    ) -> dict[str, float]:
        candles = self._market.get_candles(symbol, "1H", limit=25)
        return {
            "trend_persistence": self._trend_persistence(candles),
            "relative_strength": await self._relative_strength(symbol),
            "vol_health":        await self._vol_health(symbol),
            "breadth":           await self._market_breadth(),
            "volume_behavior":   self._volume_behavior(candles, entry_price),
            "distribution":      self._distribution(candles),
        }

    # ── 1. Trend Persistence ──────────────────────────────────────────────────

    def _trend_persistence(self, candles: list[Candle]) -> float:
        """
        SMA5 > SMA20 (estrutura macro) + Higher Highs/Higher Lows (microestrutura).
        Score 0–1: 1.0 = tendência intacta e forte.
        """
        if len(candles) < 20:
            return 0.5

        closes = [c.close for c in candles[:20]]
        highs  = [c.high  for c in candles[:6]]
        lows   = [c.low   for c in candles[:6]]

        sma5  = sum(closes[:5]) / 5
        sma20 = sum(closes[:20]) / 20

        # Melhoria B: SMA score CONTÍNUO (não binário) — evita cliff edge.
        # Antes: 1.0 se sma5 > sma20, 0.0 caso contrário → uma vela contra-tendência
        #        derrubava conviction abruptamente de ~85% para ~45%.
        # Agora: margem normalizada em [-3%, +3%] → score gradual.
        #   -3%  → 0.0  (tendência claramente quebrada)
        #    0%  → 0.5  (neutro — SMA5 = SMA20)
        #   +3%  → 1.0  (tendência forte)
        # A tendência precisa deteriorar CONSISTENTEMENTE para impactar conviction.
        margin = (sma5 - sma20) / sma20 if sma20 > 0 else 0.0
        sma_score = min(max((margin + 0.03) / 0.06, 0.0), 1.0)

        # Higher Highs + Higher Lows (últimos 5 candles)
        hh = sum(1 for i in range(min(4, len(highs) - 1)) if highs[i] > highs[i + 1])
        hl = sum(1 for i in range(min(4, len(lows)  - 1)) if lows[i]  > lows[i + 1])
        structure_score = (hh + hl) / 8

        # Pesos: SMA score agora tem peso maior (0.60) pois é contínuo e mais informativo
        return round(sma_score * 0.60 + structure_score * 0.40, 4)

    # ── 2. Relative Strength ──────────────────────────────────────────────────

    async def _relative_strength(self, symbol: str) -> float:
        """M7 score do RelativeStrengthCollector (0–1). Fallback neutro = 0.5."""
        try:
            raw = await self._cache.get(f"relative_strength:{symbol}")
            if not raw:
                return 0.5
            data = raw if isinstance(raw, dict) else json.loads(raw)
            return round(float(data.get("scores", {}).get("m7", 0.5)), 4)
        except Exception:
            return 0.5

    # ── 3. Volatility Health ──────────────────────────────────────────────────

    async def _vol_health(self, symbol: str) -> float:
        """M8 state machine → score de saúde (EXPANDING=0.9, CHAOTIC=0.05)."""
        try:
            raw = await self._cache.get(f"vol_state:{symbol}")
            if not raw:
                return 0.5
            data  = raw if isinstance(raw, dict) else json.loads(raw)
            state = data.get("state", "UNKNOWN")
            return VOL_STATE_HEALTH.get(state, 0.5)
        except Exception:
            return 0.5

    # ── 4. Market Breadth ─────────────────────────────────────────────────────

    async def _market_breadth(self) -> float:
        """Breadth do MetaRegimeDetector (0–1). 1.0 = todos os ativos subindo."""
        try:
            raw = await self._cache.get("meta_regime")
            if not raw:
                return 0.5
            data = raw if isinstance(raw, dict) else json.loads(raw)
            breadth = data.get("features_raw", {}).get("breadth", 0.5)
            return round(float(breadth), 4)
        except Exception:
            return 0.5

    # ── 5. Volume Behavior ────────────────────────────────────────────────────

    def _volume_behavior(self, candles: list[Candle], entry_price: float) -> float:
        """
        Volume confirmando a direção do preço.
        Bullish + volume crescente → 0.85 | Bearish + volume alto → 0.15
        """
        if len(candles) < 5:
            return 0.5

        closes = [c.close  for c in candles[:5]]
        vols   = [c.volume for c in candles[:5]]

        avg_vol    = sum(vols[1:]) / max(len(vols) - 1, 1)
        recent_vol = vols[0]
        vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

        price_up = closes[0] >= entry_price

        if price_up and vol_ratio > 1.10:
            return 0.85   # bullish + volume crescente = confirmação
        if price_up and vol_ratio >= 0.80:
            return 0.65   # bullish + volume normal
        if not price_up and vol_ratio > 1.20:
            return 0.15   # bearish + volume alto = distribuição
        if not price_up:
            return 0.35   # bearish + volume normal
        return 0.5

    # ── 6. Distribution Detection ─────────────────────────────────────────────

    def _distribution(self, candles: list[Candle]) -> float:
        """
        Retorna score de AUSÊNCIA de distribuição (1.0 = saudável, sem distribuição).
        Distribuição = vela bearish com volume acima da média (smart money saindo).
        """
        if len(candles) < 4:
            return 0.5

        closes = [c.close  for c in candles[:4]]
        opens  = [c.open   for c in candles[:4]]
        vols   = [c.volume for c in candles[:4]]

        avg_vol = sum(vols[1:]) / max(len(vols) - 1, 1)

        dist_signals = 0
        for i in range(min(3, len(closes))):
            bearish  = closes[i] < opens[i]
            high_vol = vols[i] > avg_vol * 1.15
            if bearish and high_vol:
                dist_signals += 1

        # 0 sinais → 1.0 (perfeito), 3 sinais → 0.10 (distribuição clara)
        return round(1.0 - (dist_signals / 3) * 0.90, 4)

    # ── Helper ────────────────────────────────────────────────────────────────

    async def _get_price(self, symbol: str) -> float | None:
        try:
            price = await self._cache.get_price(symbol)
            return float(price) if price else None
        except Exception:
            return None
