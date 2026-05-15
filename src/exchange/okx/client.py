"""
OKX REST Client — public and private market data.

Strategies NEVER import this directly.
Only the Market Engine uses this client.

Rate limiting: conservative (50% of OKX max).
Retries: handled by caller (RetryPolicy).
"""

import json
import logging

import httpx

from ...core.models import Candle, Ticker
from .auth import build_headers
from .normalizer import candle_from_okx, ticker_from_okx

logger = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com"


class OKXClient:
    """
    Async OKX REST client.
    Public endpoints: no auth needed.
    Private endpoints: requires API key/secret/passphrase.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        paper_trading: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._passphrase = passphrase
        self._paper = paper_trading
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=OKX_BASE,
            timeout=10.0,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use OKXClient as async context manager")
        return self._client

    # ── Public endpoints ──────────────────────────────────────

    async def get_candles(
        self,
        symbol: str,
        granularity: str = "1H",
        limit: int = 100,
    ) -> list[Candle]:
        """Fetch historical candles (confirmed + forming)."""
        path = "/api/v5/market/candles"
        resp = await self._http().get(path, params={
            "instId": symbol,
            "bar": granularity,
            "limit": str(limit),
        })
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        return [candle_from_okx(symbol, granularity, r) for r in rows]

    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker for a symbol."""
        path = "/api/v5/market/ticker"
        resp = await self._http().get(path, params={"instId": symbol})
        resp.raise_for_status()
        data = resp.json()
        return ticker_from_okx(data["data"][0])

    async def get_tickers(self, symbols: list[str]) -> list[Ticker]:
        """Fetch tickers for multiple symbols."""
        tickers = []
        for symbol in symbols:
            try:
                tickers.append(await self.get_ticker(symbol))
            except Exception as exc:
                logger.warning("Failed to get ticker symbol=%s error=%s", symbol, exc)
        return tickers

    # ── Private endpoints ─────────────────────────────────────

    async def get_balance(self) -> dict:
        """Fetch account balance (private)."""
        path = "/api/v5/account/balance"
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", path,
        )
        resp = await self._http().get(path, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        client_order_id: str,
    ) -> str:
        """Place an order. Returns exchange_order_id."""
        path = "/api/v5/trade/order"
        body_dict = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": order_type,
            "sz": str(quantity),
            "clOrdId": client_order_id,
        }
        if price is not None:
            body_dict["px"] = str(price)

        body = json.dumps(body_dict)
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "POST", path, body,
        )
        if self._paper:
            logger.info(
                "[PAPER] Would place order: %s", body_dict
            )
            return f"PAPER-{client_order_id}"

        resp = await self._http().post(path, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["ordId"]

    async def cancel_order(
        self,
        symbol: str,
        exchange_order_id: str,
    ) -> bool:
        """Cancel an order. Returns True if successful."""
        path = "/api/v5/trade/cancel-order"
        body = json.dumps({"instId": symbol, "ordId": exchange_order_id})
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "POST", path, body,
        )
        if self._paper:
            logger.info("[PAPER] Would cancel order: %s", exchange_order_id)
            return True

        resp = await self._http().post(path, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json().get("code") == "0"

    async def get_order_status(self, exchange_order_id: str) -> dict:
        """Get order status from exchange (used by reconciler)."""
        path = "/api/v5/trade/order"
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", path,
        )
        resp = await self._http().get(
            path,
            params={"ordId": exchange_order_id},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return {
            "status": data.get("state", ""),
            "filled_qty": float(data.get("fillSz", 0)),
            "avg_px": float(data.get("avgPx", 0) or 0),
        }

    async def get_open_positions(self) -> list[dict]:
        """Get open positions from exchange (used by reconciler)."""
        path = "/api/v5/account/positions"
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", path,
        )
        resp = await self._http().get(path, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [{"symbol": p["instId"]} for p in data if float(p.get("pos", 0)) != 0]
