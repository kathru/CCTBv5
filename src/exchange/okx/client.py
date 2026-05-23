"""
OKX REST Client — public and private market data.

Strategies NEVER import this directly.
Only the Market Engine uses this client.

Rate limiting: conservative (50% of OKX max).
Retries: handled by caller (RetryPolicy).
"""

import json
import logging
from datetime import UTC, datetime

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
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=OKX_BASE,
                timeout=httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0),
                headers={"Content-Type": "application/json"},
            )
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
            "GET", path, paper=self._paper,
        )
        resp = await self._http().get(path, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def get_usdt_balance(self) -> float:
        """Retorna o saldo USDT disponível para trading. 0.0 em caso de erro."""
        try:
            data = await self.get_balance()
            details = data.get("data", [{}])[0].get("details", [])
            usdt = next((d for d in details if d.get("ccy") == "USDT"), {})
            return float(usdt.get("availBal", 0.0))
        except Exception as exc:
            logger.warning("get_usdt_balance failed: %s", exc)
            return 0.0

    async def get_total_equity_usd(self) -> float:
        """Retorna o equity total da conta em USD. 0.0 em caso de erro."""
        try:
            data = await self.get_balance()
            eq = data.get("data", [{}])[0].get("totalEq", "0")
            return float(eq)
        except Exception as exc:
            logger.warning("get_total_equity_usd failed: %s", exc)
            return 0.0

    async def get_account_details(self) -> list[dict]:
        """
        Retorna detalhes de TODOS os ativos da conta OKX (saldo > 0).
        Não filtra por ativo — a filtragem de ativos válidos é feita no ExchangeSync.
        """
        try:
            data = await self.get_balance()
            details = data.get("data", [{}])[0].get("details", [])
            all_assets = [
                {
                    "ccy":       d.get("ccy", ""),
                    "availBal":  float(d.get("availBal", 0)),
                    "cashBal":   float(d.get("cashBal", 0)),
                    "frozenBal": float(d.get("frozenBal", 0)),
                    "usdValue":  float(d.get("eqUsd", 0) or 0),
                }
                for d in details
                if float(d.get("cashBal", 0)) > 0 or float(d.get("availBal", 0)) > 0
            ]
            logger.info(
                "OKX account assets found: %s",
                [f"{a['ccy']}={a['cashBal']:.6f}" for a in all_assets],
            )
            return all_assets
        except Exception as exc:
            logger.warning("get_account_details failed: %s", exc)
            return []

    async def get_filled_orders(
        self,
        inst_type: str = "SPOT",
        limit: int = 100,
        after: str = "",
    ) -> list[dict]:
        """Retorna ordens preenchidas recentes do OKX."""
        path = "/api/v5/trade/orders-history"
        params: dict = {"instType": inst_type, "state": "filled", "limit": str(limit)}
        if after:
            params["after"] = after
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signed_path = f"{path}?{query}"
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", signed_path, paper=self._paper,
        )
        resp = await self._http().get(
            path,
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        orders = []
        for o in data.get("data", []):
            orders.append({
                "ordId":    o.get("ordId", ""),
                "clOrdId":  o.get("clOrdId", ""),
                "symbol":   o.get("instId", ""),
                "side":     o.get("side", ""),
                "ordType":  o.get("ordType", ""),
                "sz":       float(o.get("sz", 0)),
                "fillSz":   float(o.get("fillSz", 0)),
                "avgPx":    float(o.get("avgPx", 0) or 0),
                "fee":      float(o.get("fee", 0) or 0),
                "feeCcy":   o.get("feeCcy", ""),
                "state":    o.get("state", ""),
                "uTime":    int(o.get("uTime", 0)),   # update timestamp ms
                "cTime":    int(o.get("cTime", 0)),   # create timestamp ms
            })
        return orders

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        client_order_id: str,
    ) -> str:
        """Place an order via OKX (demo em paper_trading=True, real em live).
        Retorna exchange_order_id.
        """
        path = "/api/v5/trade/order"
        # OKX clOrdId: alphanumeric only, max 32 chars — strip hyphens from UUID
        cl_ord_id = client_order_id.replace("-", "")[:32]
        body_dict = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": order_type,
            "sz": str(quantity),
            "clOrdId": cl_ord_id,
        }
        # For market buy orders, sz means quote currency (USDT) by default.
        # Set tgtCcy=base_ccy so sz is interpreted as base currency (BTC/ETH/SOL).
        if order_type == "market" and side == "buy":
            body_dict["tgtCcy"] = "base_ccy"
        if price is not None:
            body_dict["px"] = str(price)

        body = json.dumps(body_dict)
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "POST", path, body, paper=self._paper,
        )
        resp = await self._http().post(path, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise ValueError(f"OKX order rejected: {data.get('msg')} data={data}")
        order_id = data["data"][0]["ordId"]
        if self._paper:
            logger.info("[PAPER] Order placed on OKX simulated env ordId=%s", order_id)
        return order_id

    async def place_swap_order(
        self,
        symbol: str,           # ex: "BTC-USDT-SWAP"
        side: str,             # "buy" (close short / open long) | "sell" (open short / close long)
        pos_side: str,         # "long" | "short"
        quantity: float,       # contratos (1 contrato BTC-SWAP = 0.01 BTC)
        order_type: str = "market",
        client_order_id: str = "",
    ) -> str:
        """
        Coloca ordem em perpetual swap (futuros perpétuos) na OKX.
        Usa cross-margin com hedge mode (long+short simultâneos permitidos).

        Para SHORT:  side="sell", pos_side="short"
        Para fechar: side="buy",  pos_side="short"

        OKX demo suporta SWAP com x-simulated-trading: 1.
        """
        path      = "/api/v5/trade/order"
        cl_ord_id = (
            (client_order_id or "").replace("-", "")[:32]
            or "cctb" + str(int(datetime.now(UTC).timestamp()))[-8:]
        )
        body_dict = {
            "instId":  symbol,
            "tdMode":  "cross",     # cross-margin (não isolated)
            "side":    side,
            "posSide": pos_side,    # hedge mode obrigatório para distinguir long/short
            "ordType": order_type,
            "sz":      str(int(quantity)),  # OKX SWAP: sz em número de contratos (inteiro)
            "clOrdId": cl_ord_id,
        }
        body    = json.dumps(body_dict)
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "POST", path, body, paper=self._paper,
        )
        resp = await self._http().post(path, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise ValueError(f"OKX SWAP order rejected: {data.get('msg')} data={data}")
        order_id = data["data"][0]["ordId"]
        logger.info(
            "[%s] SWAP Order %s %s %s qty=%d ordId=%s",
            "PAPER" if self._paper else "LIVE",
            side.upper(), pos_side.upper(), symbol, int(quantity), order_id,
        )
        return order_id

    async def get_swap_positions(self) -> list[dict]:
        """Retorna posições abertas em perpetual swaps."""
        path    = "/api/v5/account/positions"
        params  = {"instType": "SWAP"}
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", path, "", paper=self._paper,
        )
        resp = await self._http().get(path, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            return []
        return data.get("data", [])

    async def get_swap_funding_rate(self, symbol: str) -> float | None:
        """Retorna o funding rate atual de um instrumento SWAP (ex: BTC-USDT-SWAP)."""
        path    = "/api/v5/public/funding-rate"
        params  = {"instId": symbol}
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", path, "", paper=self._paper,
        )
        try:
            resp = await self._http().get(path, params=params, headers=headers)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                return float(data["data"][0].get("fundingRate", 0))
        except Exception:
            pass
        return None

    async def get_open_interest(self, symbol: str) -> dict | None:
        """
        Retorna Open Interest de um instrumento SWAP (endpoint público).

        symbol: ex. 'BTC-USDT-SWAP'
        Retorna dict com:
          oi      : OI em contratos
          oiCcy   : OI em moeda base
          oiUsd   : OI em USD
          ts      : timestamp
        """
        path   = "/api/v5/public/open-interest"
        params = {"instType": "SWAP", "instId": symbol}
        try:
            resp = await self._http().get(path, params=params)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                row = data["data"][0]
                return {
                    "oi":     float(row.get("oi",    0)),
                    "oiCcy":  float(row.get("oiCcy", 0)),
                    "oiUsd":  float(row.get("oiUsd", 0) or row.get("oiCcy", 0)),
                    "ts":     int(row.get("ts", 0)),
                }
        except Exception:
            pass
        return None

    async def get_funding_rate_history(self, symbol: str, limit: int = 3) -> list[float]:
        """
        Retorna os últimos N funding rates de um SWAP para calcular tendência.
        Permite detectar se o funding está subindo (crescente demanda de longs)
        ou caindo (bearish ou desinteresse).
        """
        path   = "/api/v5/public/funding-rate-history"
        params = {"instId": symbol, "limit": str(limit)}
        try:
            resp = await self._http().get(path, params=params)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                return [float(r.get("fundingRate", 0)) for r in data["data"]]
        except Exception:
            pass
        return []

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
            "POST", path, body, paper=self._paper,
        )
        resp = await self._http().post(path, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json().get("code") == "0"

    async def get_order_status(self, exchange_order_id: str, symbol: str | None = None) -> dict:
        """Get order status from exchange (used by reconciler).

        OKX requires instId alongside ordId — pass symbol when available.
        OKX GET auth: query string must be included in the signed path.
        """
        base_path = "/api/v5/trade/order"
        params: dict = {"ordId": exchange_order_id}
        if symbol:
            params["instId"] = symbol
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signed_path = f"{base_path}?{query}"
        headers = build_headers(
            self._api_key, self._secret_key, self._passphrase,
            "GET", signed_path, paper=self._paper,
        )
        resp = await self._http().get(
            base_path,
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            # Order not found on exchange — treat as cancelled
            return {"status": "cancelled", "filled_qty": 0.0, "avg_px": 0.0}
        data = rows[0]
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
            "GET", path, paper=self._paper,
        )
        resp = await self._http().get(path, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [{"symbol": p["instId"]} for p in data if float(p.get("pos", 0)) != 0]
