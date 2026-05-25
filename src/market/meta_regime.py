"""
MetaRegimeDetector — classifica o regime macro cross-asset.

Observa BTC + ETH + SOL simultaneamente para detectar o estado
macroeconômico do mercado cripto usando 5 features normalizadas:

  1. BTC trend score      : retorno BTC nas últimas 20H (bull/bear direcional)
  2. Cross-asset corr     : correlação Pearson pairwise entre símbolos (1H × 24)
  3. Market breadth       : % símbolos acima da SMA20 (participation)
  4. Avg volatility state : média dos M8 scores em cache Redis
  5. BTC dominance        : RS do BTC vs alts (quem está liderando)

Classificação por distância Euclidiana a 5 arquétipos pré-definidos
(não requer treino — baseado em knowledge estrutural do mercado):

  RISK_ON    → btc+, corr alta, breadth alto, vol expandindo, balanced
  RISK_OFF   → btc-, corr muito alta, breadth baixo, vol alta, BTC dominante
  SIDEWAYS   → btc flat, corr baixa, breadth médio, vol comprimida
  ALTSEASON  → btc flat, corr média, breadth alto, alts liderando
  TRANSITION → sinais mistos, tudo médio

Threshold modifiers aplicados em MomentumStrategy.evaluate():
  RISK_ON    × 0.92  (maré alta — mais fácil entrar)
  ALTSEASON  × 0.95
  SIDEWAYS   × 1.00  (sem mudança)
  TRANSITION × 1.05  (conservador)
  RISK_OFF   × 1.20  (muito difícil entrar)

Ciclo: 15 minutos. Cache Redis: `meta_regime` TTL=1800s.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from .indicators import euclidean as _euclidean
from .indicators import pearson as _pearson
from .indicators import returns_1h as _returns_1h
from .indicators import sigmoid as _sigmoid

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900    # 15 minutos
REDIS_TTL     = 1800   # 30 minutos
SYMBOLS       = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
BTC           = "BTC-USDT"

# ── Threshold modifiers por macro regime ─────────────────────────────────────
REGIME_THRESHOLD_MULT: dict[str, float] = {
    "RISK_ON":    0.92,
    "ALTSEASON":  0.95,
    "SIDEWAYS":   1.00,
    "TRANSITION": 1.035,
    "RISK_OFF":   1.20,
    "UNKNOWN":    1.00,
}

# ── Arquétipos de regime (feature space normalizado [0,1]) ───────────────────
# Ordem: [btc_trend, cross_corr, breadth, avg_vol, btc_dominance]
ARCHETYPES: dict[str, list[float]] = {
    "RISK_ON":    [0.75, 0.80, 0.75, 0.70, 0.50],
    "RISK_OFF":   [0.15, 0.85, 0.25, 0.60, 0.70],
    "SIDEWAYS":   [0.50, 0.30, 0.55, 0.35, 0.45],
    "ALTSEASON":  [0.50, 0.50, 0.80, 0.65, 0.25],
    "TRANSITION": [0.50, 0.55, 0.50, 0.55, 0.50],
}

# Interpretações descritivas para o dashboard
REGIME_DESCRIPTION: dict[str, str] = {
    "RISK_ON":    "BTC↑ + alts↑ — maré alta favorece entradas",
    "RISK_OFF":   "BTC↓ + correlação alta — evitar novas posições",
    "SIDEWAYS":   "Mercado lateral — aguardar definição direcional",
    "ALTSEASON":  "Alts liderando BTC — momentum nas alts",
    "TRANSITION": "Regime em mudança — sinais mistos, conservador",
    "UNKNOWN":    "Dados insuficientes",
}

REGIME_COLOR: dict[str, str] = {
    "RISK_ON":    "green",
    "ALTSEASON":  "blue",
    "SIDEWAYS":   "yellow",
    "TRANSITION": "orange",
    "RISK_OFF":   "red",
    "UNKNOWN":    "muted",
}


# ── Helpers matemáticos ───────────────────────────────────────────────────────





def classify_macro_regime(features: list[float]) -> tuple[str, dict]:
    """
    Classifica o regime macro com distribuição de probabilidade softmax.

    Evolução Phase 16 (meta-learning probabilístico):
      Antes: label único = argmin(distância Euclidiana)
      Agora: P(regime) = softmax(-temperatura × distância) para cada arquétipo

    Isso resolve regime boundary instability — em vez de saltar entre
    RISK_ON e TRANSITION na fronteira, o sistema blenda gradualmente.

    Retorna:
      regime     : label dominante (argmax da distribuição)
      proba      : distribuição completa {regime: probabilidade}
      confidence : entropia invertida — 1.0=certeza total, 0.0=uniforme
      threshold_mult_blended : mult ponderado pela distribuição (não mais binário)
    """
    import math

    distances = {
        name: _euclidean(features, center)
        for name, center in ARCHETYPES.items()
    }

    # Softmax com temperatura T=8: transforma distâncias em probabilidades
    # T alto → distribuição mais concentrada no melhor; T baixo → mais difusa
    T = 8.0
    raw = {name: math.exp(-T * d) for name, d in distances.items()}
    total = sum(raw.values()) or 1.0
    proba = {name: round(v / total, 4) for name, v in raw.items()}

    # Regime dominante = maior probabilidade
    best = max(proba, key=proba.get)

    # Confiança = 1 - entropia normalizada (0=uniforme, 1=certeza)
    n = len(proba)
    entropy = -sum(p * math.log(p + 1e-9) for p in proba.values())
    max_entropy = math.log(n)
    confidence = round(1.0 - entropy / max_entropy, 3)

    # Threshold mult BLENDADO — ponderado pela distribuição inteira
    # Evita salto brusco ao cruzar fronteira de regime
    thr_blended = sum(
        proba[name] * REGIME_THRESHOLD_MULT.get(name, 1.0)
        for name in proba
    )

    return best, {
        "features":              [round(f, 4) for f in features],
        "distances":             {k: round(v, 4) for k, v in distances.items()},
        "proba":                 proba,
        "confidence":            confidence,
        "threshold_mult_blended": round(thr_blended, 4),
    }


class MetaRegimeDetector:
    """
    Detecta o regime macro cross-asset com distribuição probabilística.

    Phase 16 — Meta-learning de Regimes:
      - Classificação softmax → P(regime) para cada arquétipo
      - Persistência exponencial: suaviza distribuição com EMA entre ciclos
        (α=0.35 → novo ciclo pesa 35%, histórico pesa 65%)
      - Threshold blendado: usa distribuição inteira, não só o label dominante
      - Expõe regime_distribution para dashboard e MomentumStrategy
    """

    # EMA alpha para suavização da distribuição entre ciclos
    # 0.35 = novo ciclo tem 35% de peso, acumula estabilidade em ~8 ciclos (2h)
    EMA_ALPHA = 0.35

    def __init__(self, market, cache, symbols: list[str] = SYMBOLS) -> None:
        self._market  = market
        self._cache   = cache
        self._symbols = symbols
        self._task: asyncio.Task | None = None
        self._running = False
        # Distribuição acumulada (EMA entre ciclos)
        self._smoothed_proba: dict[str, float] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._detect()
        self._task = asyncio.create_task(self._loop(), name="meta_regime_detector")
        logger.info("MetaRegimeDetector started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._detect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("MetaRegimeDetector error: %s", exc)

    async def _detect(self) -> None:
        try:
            result = self._compute()
            await self._cache.set("meta_regime", json.dumps(result), ttl=REDIS_TTL)
            logger.info(
                "MetaRegime → %s (conf=%.2f) | btc_trend=%.3f corr=%.3f"
                " breadth=%.2f vol=%.3f dom=%.3f",
                result["regime"], result["confidence"],
                *result["features_raw"].values(),
            )
        except Exception as exc:
            logger.debug("MetaRegimeDetector._detect: %s", exc)

    def _compute(self) -> dict:
        """Calcula as 5 features e classifica o regime."""

        # ── Coleta candles de todos os símbolos ───────────────────────────────
        candle_map: dict[str, list] = {}
        for sym in self._symbols:
            c = self._market.get_candles(sym, "1H")
            if len(c) >= 25:
                candle_map[sym] = c

        if BTC not in candle_map:
            return self._unknown("BTC sem candles")

        btc_closes = [c.close for c in candle_map[BTC][:26]]

        # ── Feature 1: BTC trend score ────────────────────────────────────────
        # Retorno BTC nas últimas 20H → sigmoid para [0,1]
        btc_ret_20h = (
            (btc_closes[0] - btc_closes[20]) / btc_closes[20] if btc_closes[20] > 0 else 0.0
        )
        btc_trend = round(_sigmoid(btc_ret_20h, k=15), 4)

        # ── Feature 2: Cross-asset correlation ───────────────────────────────
        # Pearson pairwise de retornos 1H (24 períodos) → média das 3 pares
        returns_map: dict[str, list[float]] = {}
        for _sym, candles in candle_map.items():
            closes = [c.close for c in candles[:26]]
            returns_map[sym] = _returns_1h(closes, 24)

        syms = list(returns_map.keys())
        corrs = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                r = abs(_pearson(returns_map[syms[i]], returns_map[syms[j]]))
                corrs.append(r)
        cross_corr = round(sum(corrs) / len(corrs), 4) if corrs else 0.5

        # ── Feature 3: Market breadth ─────────────────────────────────────────
        # % símbolos com close > SMA20
        above_sma = 0
        total = 0
        for _sym, candles in candle_map.items():
            closes = [c.close for c in candles[:21]]
            if len(closes) >= 20:
                sma20 = sum(closes[:20]) / 20
                if closes[0] > sma20:
                    above_sma += 1
                total += 1
        breadth = round(above_sma / total, 4) if total > 0 else 0.5

        # ── Feature 4: Avg volatility state ──────────────────────────────────
        # Lê M8 scores do Redis para todos os símbolos
        # (síncrono não é possível aqui → usa fallback 0.5 calculado de candles)
        vol_scores = []
        for _sym, candles in candle_map.items():
            closes = [c.close for c in candles[:26]]
            highs  = [c.high  for c in candles[:26]]
            lows   = [c.low   for c in candles[:26]]
            vol_scores.append(self._quick_vol_score(closes, highs, lows))
        avg_vol = round(sum(vol_scores) / len(vol_scores), 4) if vol_scores else 0.5

        # ── Feature 5: BTC dominance ──────────────────────────────────────────
        # BTC return 5H vs média dos outros símbolos
        btc_ret_5h = (
            (btc_closes[0] - btc_closes[5]) / btc_closes[5]
            if len(btc_closes) > 5 and btc_closes[5] > 0 else 0.0
        )
        alt_rets = []
        for _sym, candles in candle_map.items():
            if sym == BTC:
                continue
            closes = [c.close for c in candles[:7]]
            if len(closes) > 5 and closes[5] > 0:
                alt_rets.append((closes[0] - closes[5]) / closes[5])
        avg_alt_ret = sum(alt_rets) / len(alt_rets) if alt_rets else btc_ret_5h
        # BTC dominance: 0 = alts liderando, 1 = BTC liderando
        btc_dom_raw = btc_ret_5h - avg_alt_ret  # positivo = BTC outperforming
        btc_dominance = round(_sigmoid(btc_dom_raw, k=50), 4)

        # ── Classificação probabilística ──────────────────────────────────────
        features = [btc_trend, cross_corr, breadth, avg_vol, btc_dominance]
        regime, details = classify_macro_regime(features)

        # ── Persistência exponencial (EMA da distribuição) ────────────────────
        # Suaviza oscilações rápidas de regime — evita flip a cada 15min
        raw_proba: dict[str, float] = details["proba"]
        if not self._smoothed_proba:
            self._smoothed_proba = dict(raw_proba)
        else:
            α = self.EMA_ALPHA
            self._smoothed_proba = {
                name: round(α * raw_proba.get(name, 0.0) + (1 - α) * self._smoothed_proba.get(name, 0.0), 4)
                for name in raw_proba
            }
            # Renormaliza para garantir soma = 1.0
            total = sum(self._smoothed_proba.values()) or 1.0
            self._smoothed_proba = {k: round(v / total, 4) for k, v in self._smoothed_proba.items()}

        # Regime suavizado = argmax da distribuição EMA
        smoothed_regime = max(self._smoothed_proba, key=self._smoothed_proba.get)

        # Threshold mult blendado pela distribuição SUAVIZADA (mais estável)
        thr_blended = sum(
            self._smoothed_proba.get(name, 0.0) * REGIME_THRESHOLD_MULT.get(name, 1.0)
            for name in ARCHETYPES
        )

        logger.info(
            "MetaRegime → %s (raw) → %s (smoothed) | conf=%.2f thr×%.3f | dist=%s",
            regime, smoothed_regime, details["confidence"], thr_blended,
            {k: f"{v:.2f}" for k, v in self._smoothed_proba.items()},
        )

        return {
            "regime":                  smoothed_regime,
            "regime_raw":              regime,
            "description":             REGIME_DESCRIPTION.get(smoothed_regime, ""),
            "color":                   REGIME_COLOR.get(smoothed_regime, "muted"),
            "confidence":              details["confidence"],
            "threshold_mult":          round(thr_blended, 4),   # blendado (não mais binário)
            "threshold_mult_hard":     REGIME_THRESHOLD_MULT.get(regime, 1.0),  # referência
            "regime_distribution":     self._smoothed_proba,    # P(regime) suavizado
            "regime_distribution_raw": raw_proba,               # P(regime) instantâneo
            "features_raw": {
                "btc_trend":      btc_trend,
                "cross_corr":     cross_corr,
                "breadth":        breadth,
                "avg_vol_state":  avg_vol,
                "btc_dominance":  btc_dominance,
            },
            "archetype_distances": details["distances"],
            "updated_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _quick_vol_score(closes: list[float], highs: list[float],
                          lows: list[float]) -> float:
        """Versão inline e síncrona do M8 para uso dentro do detector."""
        if len(closes) < 22:
            return 0.5
        price = closes[0]
        # ATR simples (7 períodos)
        n = min(7, len(highs) - 1)
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i+1]),
                   abs(lows[i] - closes[i+1])) for i in range(n)]
        atr = sum(trs) / len(trs) if trs else price * 0.01
        atr_pct = atr / price if price > 0 else 0.0
        # Mapeia ATR% → vol score
        if atr_pct < 0.008:
            return 0.35   # comprimido
        elif atr_pct > 0.025:
            return 0.70   # alto
        else:
            return 0.50 + (atr_pct - 0.008) / 0.017 * 0.20  # interpolado

    def _unknown(self, reason: str) -> dict:
        return {
            "regime":          "UNKNOWN",
            "description":     f"Sem dados: {reason}",
            "color":           "muted",
            "confidence":      0.0,
            "threshold_mult":  1.0,
            "features_raw":    {},
            "archetype_distances": {},
            "updated_at": datetime.now(UTC).isoformat(),
        }

    async def get_regime(self) -> dict | None:
        """Lê o regime macro atual do Redis."""
        raw = await self._cache.get("meta_regime")
        if not raw:
            return None
        return raw if isinstance(raw, dict) else json.loads(raw)
