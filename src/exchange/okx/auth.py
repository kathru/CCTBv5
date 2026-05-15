"""
OKX API Authentication — HMAC-SHA256 signing.

OKX requires:
  - OK-ACCESS-KEY    : API key
  - OK-ACCESS-SIGN   : base64(HMAC-SHA256(timestamp+method+path+body))
  - OK-ACCESS-TIMESTAMP : ISO8601 UTC
  - OK-ACCESS-PASSPHRASE: passphrase set when creating the API key
"""

import base64
import hashlib
import hmac
from datetime import datetime, timezone


def _utc_now() -> str:
    """Return current UTC time in OKX format: 2024-01-01T00:00:00.000Z"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sign(
    timestamp: str,
    method: str,        # "GET" or "POST"
    path: str,          # e.g. "/api/v5/trade/order"
    body: str,          # "" for GET, JSON string for POST
    secret_key: str,
) -> str:
    """Return base64-encoded HMAC-SHA256 signature."""
    message = timestamp + method.upper() + path + body
    mac = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


def build_headers(
    api_key: str,
    secret_key: str,
    passphrase: str,
    method: str,
    path: str,
    body: str = "",
) -> dict[str, str]:
    """Build authenticated headers for an OKX API request."""
    timestamp = _utc_now()
    signature = sign(timestamp, method, path, body, secret_key)
    return {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }
