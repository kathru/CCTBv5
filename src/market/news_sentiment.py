"""
NewsSentimentCollector — M9 factor (Phase 4 CCTBv5)

Coleta e processa sentimento de mercado de duas fontes gratuitas:

  1. Fear & Greed Index (alternative.me)
     — Índice global 0-100: Extreme Fear → Extreme Greed
     — Atualizado diariamente, sem API key
     — Interpretação para momentum: greed > 60 = favorável, fear < 40 = cuidado

  2. CoinGecko — social/market data por símbolo
     — price_change_24h, sentiment_votes_up/down
     — Tier gratuito, sem API key (rate limit: ~30 req/min)

Lógica de M9:
  - M9 = 0.0: sentimento muito negativo (extreme fear + queda de preço)
  - M9 = 0.5: neutro (sem sinal claro)
  - M9 = 1.0: sentimento muito positivo (greed + alta confirmada)

Nota: seguimos o momentum do sentimento, NÃO o contrário.
"Comprar na extrema euforia" → M9 alto contribui para score.
"Comprar no desespero" → M9 baixo penaliza score.

Cache Redis: TTL 1h (sentiment não muda por minuto).
Poll: a cada 30 minutos no background (não bloqueia ciclo de trading).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from ..persistence.cache import Cache

logger = logging.getLogger(__name__)

# ── Mapeamento CoinGecko IDs ──────────────────────────────────────────────────
COINGECKO_IDS: dict[str, str] = {
    "BTC-USDT": "bitcoin",
    "ETH-USDT": "ethereum",
    "SOL-USDT": "solana",
}

# URLs das APIs (gratuitas, sem autenticação)
FNG_URL        = "https://api.alternative.me/fng/?limit=1"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# TTLs de cache
CACHE_TTL_FNG      = 3600   # 1h — F&G atualiza diariamente
CACHE_TTL_COIN     = 1800   # 30min — dados de moeda
CACHE_KEY_FNG      = "sentiment:fng_score"
CACHE_KEY_COIN     = "sentiment:coin:{symbol}"
CACHE_KEY_M9       = "sentiment:m9:{symbol}"

# Intervalo de polling background
POLL_INTERVAL = 1800   # 30 minutos


class NewsSentimentCollector:
    """
    Coleta sentimento de mercado e expõe score M9 por símbolo.

    Integração no TradingLoop:
        self._news_sentiment = NewsSentimentCollector(cache=self._cache)
        asyncio.create_task(self._news_sentiment.run_forever())

    Leitura no StrategyRunner:
        m9_data = await self._news_sentiment.get_scores(symbol)
        ctx.extra["news_sentiment"] = m9_data
    """

    def __init__(self, cache: Cache) -> None:
        self._cache   = cache
        self._symbols = list(COINGECKO_IDS.keys())
        self._running = False
        self._client: httpx.AsyncClient | None = None

    # ── API pública ───────────────────────────────────────────────────────────

    async def get_scores(self, symbol: str) -> dict:
        """
        Retorna scores de sentimento para o símbolo.
        Nunca levanta exceção — fallback neutro se dados indisponíveis.
        """
        try:
            raw = await self._cache.get(CACHE_KEY_M9.format(symbol=symbol))
            if raw:
                # cache.get() already parses JSON → raw may be a dict or str
                if isinstance(raw, dict):
                    return raw
                import json
                return json.loads(raw)
        except Exception:
            pass
        return {"m9_score": 0.5, "fng": 50, "coin_sentiment": 0.5, "source": "fallback"}

    async def warm_up(self) -> None:
        """Coleta dados imediatamente na inicialização (sem esperar 30min)."""
        logger.info("NewsSentimentCollector: warm-up...")
        await self._collect_all()

    async def run_forever(self) -> None:
        """Background loop — coleta a cada POLL_INTERVAL segundos."""
        self._running = True
        # Warm-up inicial com pequeno delay para não sobrecarregar boot
        await asyncio.sleep(10)
        while self._running:
            try:
                await self._collect_all()
            except Exception as exc:
                logger.warning("NewsSentimentCollector: poll error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    # ── Coleta interna ────────────────────────────────────────────────────────

    async def _collect_all(self) -> None:
        """Coleta F&G + dados por símbolo e computa M9."""
        async with httpx.AsyncClient(timeout=15) as client:
            fng_score = await self._fetch_fng(client)
            for symbol in self._symbols:
                coin_data = await self._fetch_coingecko(client, symbol)
                m9        = self._compute_m9(fng_score, coin_data)
                await self._persist(symbol, fng_score, coin_data, m9)

    async def _fetch_fng(self, client: httpx.AsyncClient) -> float:
        """
        Busca Fear & Greed Index (0-100).
        Retorna 50 (neutro) em caso de erro.
        """
        # Verifica cache primeiro
        cached = await self._cache.get(CACHE_KEY_FNG)
        if cached:
            return float(cached)

        try:
            resp = await client.get(FNG_URL)
            resp.raise_for_status()
            data  = resp.json()
            value = int(data["data"][0]["value"])
            await self._cache.set(CACHE_KEY_FNG, str(value), ttl=CACHE_TTL_FNG)
            logger.debug("NewsSentimentCollector: F&G=%d (%s)", value,
                         data["data"][0].get("value_classification", ""))
            return float(value)
        except Exception as exc:
            logger.debug("NewsSentimentCollector: F&G fetch failed: %s", exc)
            return 50.0

    async def _fetch_coingecko(
        self, client: httpx.AsyncClient, symbol: str
    ) -> dict:
        """
        Busca dados de mercado/sentimento da CoinGecko.
        Retorna dict com price_change_24h, sentiment_up_pct, etc.
        """
        coin_id = COINGECKO_IDS.get(symbol)
        if not coin_id:
            return {}

        cache_key = CACHE_KEY_COIN.format(symbol=symbol)
        cached = await self._cache.get(cache_key)
        if cached:
            # cache.get() already parses JSON → cached may be a dict or str
            if isinstance(cached, dict):
                return cached
            import json
            try:
                return json.loads(cached)
            except Exception:
                pass

        try:
            url = (
                f"{COINGECKO_BASE}/coins/{coin_id}"
                "?localization=false&tickers=false&market_data=true"
                "&community_data=true&developer_data=false"
            )
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()

            mkt = data.get("market_data", {})
            result = {
                "price_change_24h_pct": float(
                    mkt.get("price_change_percentage_24h", 0) or 0
                ),
                "price_change_7d_pct":  float(
                    mkt.get("price_change_percentage_7d", 0) or 0
                ),
                "sentiment_up_pct":     float(
                    data.get("sentiment_votes_up_percentage", 50) or 50
                ),
                "sentiment_down_pct":   float(
                    data.get("sentiment_votes_down_percentage", 50) or 50
                ),
                "market_cap_rank":      int(data.get("market_cap_rank", 99) or 99),
            }

            import json
            await self._cache.set(
                cache_key, json.dumps(result), ttl=CACHE_TTL_COIN
            )
            logger.debug(
                "NewsSentimentCollector: %s px24h=%.1f%% sentiment_up=%.0f%%",
                symbol, result["price_change_24h_pct"], result["sentiment_up_pct"],
            )
            return result

        except Exception as exc:
            logger.debug("NewsSentimentCollector: CoinGecko %s failed: %s", symbol, exc)
            return {}

    # ── Cálculo M9 ────────────────────────────────────────────────────────────

    def _compute_m9(self, fng: float, coin_data: dict) -> float:
        """
        Computa M9 [0, 1] combinando F&G global + dados específicos da moeda.

        Componentes (pesos internos):
          A. F&G normalizado       (40%): greed = mais favorável para momentum
          B. Price sentiment        (35%): % votantes positivos na CoinGecko
          C. Momentum de preço 24h  (25%): confirmação técnica de curto prazo

        Lógica:
          - F&G 0-100 → normalizado 0-1 (linearmente)
          - sentiment_up_pct 50-100 → normalizado 0-1
          - price_change_24h +3% = score 1.0, -3% = score 0.0
        """
        # A. Fear & Greed → [0, 1]
        fng_norm = fng / 100.0

        # B. Sentiment votes (CoinGecko) → [0, 1]
        # 50% up = neutro (0.5), 80% up = bullish (0.8), 20% up = bearish (0.2)
        sent_up = coin_data.get("sentiment_up_pct", 50.0)
        sent_norm = sent_up / 100.0

        # C. Price momentum 24h → [0, 1]
        # +3% ou mais = score 1.0 | 0% = score 0.5 | -3% = score 0.0
        px_chg = coin_data.get("price_change_24h_pct", 0.0)
        px_norm = min(max((px_chg + 3.0) / 6.0, 0.0), 1.0)

        # Composição com pesos
        if not coin_data:
            # Sem dados da CoinGecko — usa apenas F&G
            m9 = fng_norm
        else:
            m9 = fng_norm * 0.40 + sent_norm * 0.35 + px_norm * 0.25

        return round(min(max(m9, 0.0), 1.0), 4)

    async def _persist(
        self, symbol: str, fng: float, coin_data: dict, m9: float
    ) -> None:
        """Salva resultado final no Redis para leitura pelo StrategyRunner."""
        import json
        payload = {
            "m9_score":        m9,
            "fng":             fng,
            "fng_class":       self._fng_class(fng),
            "coin_sentiment":  coin_data.get("sentiment_up_pct", 50.0) / 100.0,
            "price_chg_24h":   coin_data.get("price_change_24h_pct", 0.0),
            "price_chg_7d":    coin_data.get("price_change_7d_pct",  0.0),
            "source":          "fng+coingecko",
            "updated_at":      datetime.now(UTC).isoformat(),
        }
        await self._cache.set(
            CACHE_KEY_M9.format(symbol=symbol),
            json.dumps(payload),
            ttl=CACHE_TTL_FNG,
        )
        logger.info(
            "NewsSentimentCollector: %s M9=%.3f (F&G=%d [%s] sent=%.0f%% px24h=%.1f%%)",
            symbol, m9, fng, self._fng_class(fng),
            coin_data.get("sentiment_up_pct", 50.0),
            coin_data.get("price_change_24h_pct", 0.0),
        )

    @staticmethod
    def _fng_class(value: float) -> str:
        v = int(value)
        if v <= 24:
            return "Extreme Fear"
        if v <= 44:
            return "Fear"
        if v <= 55:
            return "Neutral"
        if v <= 74:
            return "Greed"
        return "Extreme Greed"
