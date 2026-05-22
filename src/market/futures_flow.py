"""
FuturesFlowCollector — coleta sinais do mercado de futuros perpétuos.

Estratégia: mesmo que o bot opere SPOT (Brasil — sem derivativos),
os dados de futuros revelam o posicionamento institucional:

  • Funding Rate   : positivo = mercado bull-crowded; negativo = bearish
  • Open Interest  : crescente + preço subindo = nova força direcional
  • OI momentum    : aceleração do OI (derivada)

Esses dados alimentam o fator M6 do MomentumStrategy sem nunca
executar ordens em derivativos.

Ciclo de atualização: a cada 15 minutos (funding settles cada 8h na OKX).
Persistido no Redis com TTL=1800s para sobreviver a restarts.

Mapeamento SPOT → SWAP:
  BTC-USDT  → BTC-USDT-SWAP
  ETH-USDT  → ETH-USDT-SWAP
  SOL-USDT  → SOL-USDT-SWAP
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

POLL_INTERVAL = 900    # 15 minutos
REDIS_TTL     = 1800   # 30 minutos (2 ciclos de folga)


def _spot_to_swap(symbol: str) -> str:
    """Converte 'BTC-USDT' → 'BTC-USDT-SWAP'."""
    if symbol.endswith("-SWAP"):
        return symbol
    return f"{symbol}-SWAP"


def _score_funding(funding_rate: float) -> float:
    """
    Converte funding rate em score [0, 1].

    Lógica econômica:
      - Funding muito negativo (< -0.05%) : bears dominam → score baixo (0.1)
      - Funding neutro (≈ 0%)             : mercado equilibrado → 0.5
      - Funding moderado positivo (0.01%) : bulls leves → score alto (0.7)
      - Funding extremo positivo (> 0.1%) : crowded long → contrarian → 0.3

    Perfil: triangular com pico em +0.01% (sinal ideal de momentum saudável)
    """
    fr_pct = funding_rate * 100   # converte para %

    if fr_pct < -0.05:
        return 0.1                          # bearish dominante
    elif fr_pct < 0.0:
        return 0.1 + (fr_pct + 0.05) / 0.05 * 0.4   # [-0.05, 0] → [0.1, 0.5]
    elif fr_pct <= 0.01:
        return 0.5 + fr_pct / 0.01 * 0.25  # [0, 0.01] → [0.5, 0.75]
    elif fr_pct <= 0.05:
        return 0.75 - (fr_pct - 0.01) / 0.04 * 0.2  # [0.01, 0.05] → [0.75, 0.55]
    else:
        return max(0.2, 0.55 - (fr_pct - 0.05) / 0.05 * 0.35)  # > 0.05% → cai


def _score_oi_change(oi_now: float, oi_prev: float) -> float:
    """
    Score [0, 1] baseado na variação de Open Interest.

    OI crescente + contexto de preço subindo → nova demanda de longs
    OI decrescente → short covering ou saída de posições → fraqueza

    oi_change positivo → score > 0.5 (fortalecimento)
    oi_change negativo → score < 0.5 (enfraquecimento)
    """
    if oi_prev <= 0:
        return 0.5
    change_pct = (oi_now - oi_prev) / oi_prev

    # Mapeia [-5%, +5%] → [0.1, 0.9] com centro em 0
    score = 0.5 + change_pct / 0.05 * 0.4
    return round(min(max(score, 0.1), 0.9), 4)


def _score_funding_trend(history: list[float]) -> float:
    """
    Score [0, 1] baseado na tendência do funding rate (últimos 3 períodos).

    Funding subindo = demanda crescente por longs = bullish momentum
    Funding caindo = demand exhaustion ou reversão
    """
    if len(history) < 2:
        return 0.5
    # Tendência simples: último vs primeiro
    delta = history[0] - history[-1]   # history[0] = mais recente
    # Mapeia [-0.001, +0.001] → [0.2, 0.8]
    score = 0.5 + delta / 0.001 * 0.3
    return round(min(max(score, 0.2), 0.8), 4)


class FuturesFlowCollector:
    """
    Coleta e cacheia dados de futuros para alimentar M6 do MomentumStrategy.
    Roda como background task — falhas individuais não interrompem o trading.
    """

    def __init__(self, exchange, cache, symbols: list[str]) -> None:
        self._exchange = exchange
        self._cache    = cache
        self._symbols  = symbols
        self._task: asyncio.Task | None = None
        self._running  = False
        self._last_oi: dict[str, float] = {}   # OI anterior por símbolo

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Coleta imediata no boot
        await self._collect_all()
        self._task = asyncio.create_task(self._loop(), name="futures_flow_collector")
        logger.info("FuturesFlowCollector started for %d symbols", len(self._symbols))

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
                await self._collect_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("FuturesFlowCollector loop error: %s", exc)

    async def _collect_all(self) -> None:
        """Coleta dados para todos os símbolos e persiste no Redis."""
        for symbol in self._symbols:
            try:
                await self._collect_one(symbol)
            except Exception as exc:
                logger.debug("FuturesFlowCollector %s: %s", symbol, exc)

    async def _collect_one(self, symbol: str) -> None:
        swap = _spot_to_swap(symbol)

        # ── Funding rate atual ────────────────────────────────────────────────
        funding_rate = await self._exchange.get_swap_funding_rate(swap)
        funding_history = await self._exchange.get_funding_rate_history(swap, limit=3)

        # ── Open Interest ─────────────────────────────────────────────────────
        oi_data  = await self._exchange.get_open_interest(swap)
        oi_now   = oi_data["oiUsd"] if oi_data else 0.0
        oi_prev  = self._last_oi.get(symbol, oi_now)
        if oi_now > 0:
            self._last_oi[symbol] = oi_now

        # ── Scores ───────────────────────────────────────────────────────────
        funding_score      = _score_funding(funding_rate)        if funding_rate is not None else 0.5
        oi_change_score    = _score_oi_change(oi_now, oi_prev)   if oi_now > 0 else 0.5
        funding_trend_score = _score_funding_trend(funding_history) if funding_history else 0.5

        # M6 = blend dos 3 sub-scores
        m6_score = (
            funding_score       * 0.50 +   # dominante: preço que os longs pagam
            oi_change_score     * 0.30 +   # segundo: fluxo de capital novo
            funding_trend_score * 0.20     # terceiro: aceleração/desaceleração
        )

        payload = {
            "symbol":              symbol,
            "swap_symbol":         swap,
            "funding_rate":        round(funding_rate, 8) if funding_rate is not None else None,
            "funding_rate_pct":    round(funding_rate * 100, 5) if funding_rate is not None else None,
            "funding_history":     [round(f, 8) for f in funding_history],
            "oi_usd":              round(oi_now, 0) if oi_now else None,
            "oi_change_pct":       round((oi_now - oi_prev) / oi_prev * 100, 3) if oi_prev > 0 else None,
            "scores": {
                "funding":       round(funding_score, 4),
                "oi_change":     round(oi_change_score, 4),
                "funding_trend": round(funding_trend_score, 4),
                "m6":            round(m6_score, 4),
            },
            "updated_at": datetime.now(UTC).isoformat(),
        }

        await self._cache.set(
            f"futures_flow:{symbol}",
            json.dumps(payload),
            ttl=REDIS_TTL,
        )
        logger.debug(
            "FuturesFlow %s — FR=%.5f%% OI=$%.0fM M6=%.3f",
            symbol,
            (funding_rate or 0) * 100,
            (oi_now or 0) / 1_000_000,
            m6_score,
        )

    async def get_flow(self, symbol: str) -> dict | None:
        """Lê do Redis o dado de futures flow mais recente para um símbolo."""
        raw = await self._cache.get(f"futures_flow:{symbol}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None
