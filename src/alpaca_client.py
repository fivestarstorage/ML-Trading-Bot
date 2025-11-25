import os
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from .utils import get_logger

load_dotenv()


class AlpacaClient:
    """
    Minimal client for Alpaca's REST and market data endpoints.
    Only implements the calls we need for historical bars.
    """

    DEFAULT_TRADING_URL = "https://paper-api.alpaca.markets"
    DEFAULT_DATA_URL = "https://data.alpaca.markets/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        trading_url: Optional[str] = None,
        data_url: Optional[str] = None,
    ):
        self.logger = get_logger("alpaca_client")
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.trading_url = trading_url or os.getenv("ALPACA_BASE_URL") or self.DEFAULT_TRADING_URL
        self.data_url = data_url or os.getenv("ALPACA_DATA_URL") or self.DEFAULT_DATA_URL

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing ALPACA_API_KEY / ALPACA_API_SECRET environment variables.")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------ #
    # Market data helpers
    # ------------------------------------------------------------------ #
    def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Pull historical bars for `symbol` using the Data API v2.
        Automatically handles pagination via `next_page_token`.
        """
        url = f"{self.data_url}/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "limit": limit,
        }
        if start:
            params["start"] = self._to_iso8601(start)
        if end:
            params["end"] = self._to_iso8601(end)

        rows: List[dict] = []
        next_token: Optional[str] = None

        while True:
            if next_token:
                params["page_token"] = next_token
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 422:
                raise RuntimeError(f"Alpaca rejected request parameters: {response.text}")
            response.raise_for_status()
            payload = response.json()
            bars = payload.get("bars", [])
            for bar in bars:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(bar["t"], utc=True),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )
            next_token = payload.get("next_page_token")
            if not next_token:
                break

        if not rows:
            raise RuntimeError(f"Alpaca returned no bars for {symbol}.")

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    def fetch_recent_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 50) -> pd.DataFrame:
        """
        Convenience helper that just pulls the latest `limit` bars without pagination.
        """
        url = f"{self.data_url}/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "limit": limit,
        }
        response = self.session.get(url, params=params, timeout=15)
        if response.status_code == 422:
            raise RuntimeError(f"Alpaca rejected request parameters: {response.text}")
        response.raise_for_status()
        payload = response.json()
        rows = [
            {
                "timestamp": pd.to_datetime(bar["t"], utc=True),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
            }
            for bar in payload.get("bars", [])
        ]
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).set_index("timestamp").sort_index()

    # ------------------------------------------------------------------ #
    # Trading helpers
    # ------------------------------------------------------------------ #
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """
        Submit a simple market/limit order to the Alpaca trading REST API.
        """
        url = f"{self.trading_url}/v2/orders"
        payload: Dict[str, Optional[str]] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if limit_price is not None:
            payload["limit_price"] = float(limit_price)
        if stop_price is not None:
            payload["stop_price"] = float(stop_price)

        response = self.session.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        self.logger.info("Alpaca order submitted: %s %s x%s", side.upper(), symbol, qty)
        return data

    @staticmethod
    def _to_iso8601(ts: pd.Timestamp) -> str:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()

