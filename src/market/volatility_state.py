"""
VolatilityStateMachine — classifica o estado atual da volatilidade em 5 regimes.

Estados:
  COMPRESSED      : ATR baixo, BB estreito, candles pequenos → pré-breakout
  EXPANDING       : ATR crescente, BB alargando, volume subindo → breakout iniciando
  TREND           : ATR moderado e estável, direção consistente → tendência saudável
  CHAOTIC         : ATR alto, direção inconsistente, gaps → perigoso para momentum
  MEAN_REVERTING  : ATR moderado, preço oscilando na média → lateralização estrutural

Métricas usadas:
  1. ATR normalizado (ATR / preço) → nível absoluto de volatilidade
  2. ATR change rate (ATR[0] / ATR[N]) → aceleração/desaceleração
  3. Bollinger Band width (BB_width / preço) → volatilidade relativa ao range
  4. Directional consistency (% candles na mesma direção) → ruído vs tendência
  5. Mean reversion score (distância do preço à média vs BB) → overextension

M8 score por estado:
  EXPANDING       → 0.80  (melhor estado para momentum)
  TREND           → 0.70
  COMPRESSED      → 0.65  (potencial breakout, mas ainda sem confirmação)
  MEAN_REVERTING  → 0.35
  CHAOTIC         → 0.20  (evitar novas entradas)

Ciclo: 15 minutos (acoplado aos outros coletores).
Cache Redis: `vol_state:{symbol}` TTL=1800s.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from .indicators import atr as _atr_fn
from .indicators import bollinger as _bollinger_fn

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900    # 15 minutos
REDIS_TTL     = 1800   # 30 minutos

# M8 score por estado de volatilidade
STATE_M8_SCORE: dict[str, float] = {
    "EXPANDING":      0.80,
    "TREND":          0.70,
    "COMPRESSED":     0.65,
    "MEAN_REVERTING": 0.35,
    "CHAOTIC":        0.20,
}

# Descrições para o dashboard
STATE_DESCRIPTION: dict[str, str] = {
    "EXPANDING":      "Breakout iniciando — volatilidade crescente",
    "TREND":          "Tendência saudável — volatilidade estável",
    "COMPRESSED":     "Compressão — setup pré-breakout",
    "MEAN_REVERTING": "Lateralização — sem momentum direcional",
    "CHAOTIC":        "Caótico — risco elevado, evitar entradas",
}

STATE_COLOR: dict[str, str] = {
    "EXPANDING":      "green",
    "TREND":          "blue",
    "COMPRESSED":     "yellow",
    "MEAN_REVERTING": "orange",
    "CHAOTIC":        "red",
}




def _classify_state(
    atr_now: float,
    atr_prev: float,
    bb_width_pct: float,
    dir_consistency: float,
    price: float,
    bb_mid: float,
    bb_upper: float,
    bb_lower: float,
) -> tuple[str, dict]:
    """
    Classifica o estado de volatilidade com base em 4 métricas.

    Retorna (state_name, metrics_dict).
    """
    # Normaliza ATR como % do preço
    atr_pct = atr_now / price if price > 0 else 0.0

    # Taxa de mudança do ATR (>1 = crescendo, <1 = caindo)
    atr_change = atr_now / atr_prev if atr_prev > 0 else 1.0

    # Distância do preço ao centro da BB (overextension)
    bb_range  = (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 1.0
    price_pos = (price - bb_lower) / bb_range   # 0=bottom, 0.5=mid, 1=top

    # Heurísticas de classificação (por ordem de prioridade)
    #
    # CHAOTIC: ATR alto + direção inconsistente
    if atr_pct > 0.025 and dir_consistency < 0.45:
        state = "CHAOTIC"

    # EXPANDING: ATR crescendo rapidamente + direção consistente
    elif atr_change > 1.15 and dir_consistency > 0.55:
        state = "EXPANDING"

    # COMPRESSED: ATR baixo + BB estreita
    elif atr_pct < 0.008 or bb_width_pct < 0.015:
        state = "COMPRESSED"

    # TREND: ATR moderado, estável, direção consistente
    elif dir_consistency > 0.60 and 0.008 <= atr_pct <= 0.025:
        state = "TREND"

    # MEAN_REVERTING: preço oscilando perto da média, ATR moderado
    else:
        state = "MEAN_REVERTING"

    metrics = {
        "atr_pct":        round(atr_pct * 100, 4),      # ATR como % do preço
        "atr_change":     round(atr_change, 4),           # taxa de mudança do ATR
        "bb_width_pct":   round(bb_width_pct * 100, 3),  # BB width como % do preço
        "dir_consistency":round(dir_consistency, 3),       # 0=random, 1=perfeita
        "price_pos_bb":   round(price_pos, 3),             # posição na BB
    }
    return state, metrics


def compute_vol_state(
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> dict:
    """
    Computa o estado de volatilidade a partir de listas de candles 1H.
    closes[0] = mais recente.
    """
    if len(closes) < 22:
        return {
            "state": "UNKNOWN",
            "m8_score": 0.5,
            "description": "Dados insuficientes",
            "color": "muted",
            "metrics": {},
        }

    price = closes[0]

    # ATR atual (14 períodos) vs ATR anterior (14 períodos, offset 7)
    atr_now  = _atr_fn(highs, lows, closes, 14)
    atr_prev = _atr_fn(highs[7:], lows[7:], closes[7:], 14)

    # Bollinger Bands (20 períodos)
    bb_upper, bb_mid, bb_lower = _bollinger_fn(closes, 20)
    bb_width_pct = (bb_upper - bb_lower) / price if price > 0 else 0.0

    # Consistência direcional: % candles na mesma direção nos últimos 10
    n_dir = min(10, len(closes) - 1)
    ups   = sum(1 for i in range(n_dir) if closes[i] > closes[i + 1])
    dns   = n_dir - ups
    dir_consistency = max(ups, dns) / n_dir if n_dir > 0 else 0.5

    state, metrics = _classify_state(
        atr_now, atr_prev, bb_width_pct,
        dir_consistency, price,
        bb_mid, bb_upper, bb_lower,
    )

    return {
        "state":       state,
        "m8_score":    STATE_M8_SCORE.get(state, 0.5),
        "description": STATE_DESCRIPTION.get(state, ""),
        "color":       STATE_COLOR.get(state, "muted"),
        "atr_now":     round(atr_now, 6),
        "atr_prev":    round(atr_prev, 6),
        "bb_upper":    round(bb_upper, 4),
        "bb_mid":      round(bb_mid, 4),
        "bb_lower":    round(bb_lower, 4),
        "metrics":     metrics,
    }


class VolatilityStateCollector:
    """
    Calcula e cacheia estado de volatilidade para alimentar M8 do MomentumStrategy.
    Acoplado ao MarketEngine — lê candles 1H já em memória.
    """

    def __init__(self, market, cache, symbols: list[str]) -> None:
        self._market  = market
        self._cache   = cache
        self._symbols = symbols
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._compute_all()
        self._task = asyncio.create_task(self._loop(), name="vol_state_collector")
        logger.info("VolatilityStateCollector started for %d symbols", len(self._symbols))

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
                await self._compute_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("VolatilityStateCollector error: %s", exc)

    async def _compute_all(self) -> None:
        for symbol in self._symbols:
            try:
                await self._compute_one(symbol)
            except Exception as exc:
                logger.debug("VolState %s: %s", symbol, exc)

    async def _compute_one(self, symbol: str) -> None:
        candles = self._market.get_candles(symbol, "1H")
        if len(candles) < 22:
            return

        closes = [c.close  for c in candles[:25]]
        highs  = [c.high   for c in candles[:25]]
        lows   = [c.low    for c in candles[:25]]

        result = compute_vol_state(closes, highs, lows)
        result["symbol"]     = symbol
        result["updated_at"] = datetime.now(UTC).isoformat()

        await self._cache.set(
            f"vol_state:{symbol}",
            json.dumps(result),
            ttl=REDIS_TTL,
        )
        logger.debug(
            "VolState %s → %s (M8=%.2f) ATR%%=%.3f%% dir=%.2f",
            symbol, result["state"], result["m8_score"],
            result["metrics"].get("atr_pct", 0),
            result["metrics"].get("dir_consistency", 0),
        )

    async def get_state(self, symbol: str) -> dict | None:
        """Lê do Redis o estado de volatilidade mais recente."""
        raw = await self._cache.get(f"vol_state:{symbol}")
        if not raw:
            return None
        return raw if isinstance(raw, dict) else json.loads(raw)
