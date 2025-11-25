import os
from typing import Any, Dict, Optional, List

import pandas as pd
import requests
from dotenv import load_dotenv

from .utils import get_logger

load_dotenv()


class IGClient:
    """Lightweight REST client for IG's trading API (demo or live)."""

    DEMO_BASE_URL = "https://demo-api.ig.com/gateway/deal"
    LIVE_BASE_URL = "https://api.ig.com/gateway/deal"

    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        account_id: Optional[str] = None,
        use_demo: bool = True,
    ):
        self.logger = get_logger("ig_client")

        env_prefix = "IG_" + ("API_KEY_DEMO" if use_demo else "API_KEY_LIVE")
        self.api_key = api_key or os.getenv(env_prefix)
        self.username = username or os.getenv("IG_USERNAME_DEMO" if use_demo else "IG_USERNAME_LIVE")
        self.password = password or os.getenv("IG_PASSWORD_DEMO" if use_demo else "IG_PASSWORD_LIVE")
        self.account_id = account_id or os.getenv("IG_ACCOUNT_ID_DEMO" if use_demo else "IG_ACCOUNT_ID_LIVE")

        if not all([self.api_key, self.username, self.password, self.account_id]):
            raise ValueError(
                "Missing IG API credentials. Ensure IG_API_KEY_DEMO, IG_USERNAME_DEMO, "
                "IG_PASSWORD_DEMO and IG_ACCOUNT_ID_DEMO are defined in your environment."
            )

        self.base_url = self.DEMO_BASE_URL if use_demo else self.LIVE_BASE_URL
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-IG-API-KEY": self.api_key,
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json; charset=UTF-8",
            }
        )

        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self.oauth_access_token: Optional[str] = None
        self.oauth_refresh_token: Optional[str] = None
        self.oauth_expires_in: Optional[int] = None
        self.logged_in = False

    # --------------------------------------------------------------------- #
    # Authentication helpers
    # --------------------------------------------------------------------- #
    def login(self) -> Dict[str, Any]:
        """Authenticate against IG and store security tokens."""
        payload = {"identifier": self.username, "password": self.password}
        headers = {"Version": "3"}
        session_url = f"{self.base_url}/session?fetchSessionTokens=true"
        response = self.session.post(session_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()

        self.cst = response.headers.get("CST")
        self.security_token = response.headers.get("X-SECURITY-TOKEN")
        body = response.json()
        self.account_id = self.account_id or body.get("currentAccountId")

        oauth = body.get("oauthToken") or {}
        self.oauth_access_token = oauth.get("access_token")
        self.oauth_refresh_token = oauth.get("refresh_token")
        try:
            self.oauth_expires_in = int(oauth.get("expires_in", 0)) if oauth.get("expires_in") else None
        except ValueError:
            self.oauth_expires_in = None

        if self.cst and self.security_token:
            self.session.headers.update({"CST": self.cst, "X-SECURITY-TOKEN": self.security_token})
        elif self.oauth_access_token:
            self.session.headers.update({"Authorization": f"Bearer {self.oauth_access_token}"})
        else:
            raise RuntimeError(
                "IG authentication succeeded but no session or OAuth tokens were returned. "
                "Verify API key permissions."
            )
        self.logged_in = True
        self.logger.info("Authenticated with IG API (account %s)", self.account_id)
        return body

    def ensure_session(self):
        if not self.logged_in:
            self.login()

    def _request(self, method: str, path: str, version: str = "2", **kwargs):
        """Internal helper that injects auth headers and (re)authenticates on 401s."""
        self.ensure_session()
        headers = kwargs.pop("headers", {})
        headers.setdefault("Version", version)
        headers.setdefault("IG-ACCOUNT-ID", self.account_id)
        if self.oauth_access_token and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.oauth_access_token}"

        response = self.session.request(method, f"{self.base_url}{path}", headers=headers, timeout=20, **kwargs)

        if response.status_code == 401:
            self.logger.warning("IG session expired. Attempting to re-authenticate...")
            self.logged_in = False
            self.ensure_session()
            headers.setdefault("Version", version)
            headers.setdefault("IG-ACCOUNT-ID", self.account_id)
            if self.oauth_access_token:
                headers["Authorization"] = f"Bearer {self.oauth_access_token}"
            response = self.session.request(method, f"{self.base_url}{path}", headers=headers, timeout=20, **kwargs)

        response.raise_for_status()
        return response

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def get_market_details(self, epic: str) -> Dict[str, Any]:
        response = self._request("GET", f"/markets/{epic}", version="3")
        return response.json()

    def create_position(
        self,
        epic: str,
        direction: str,
        size: float,
        order_type: str = "MARKET",
        level: Optional[float] = None,
        limit_level: Optional[float] = None,
        stop_level: Optional[float] = None,
        currency: str = "USD",
        expiry: str = "DFB",
        force_open: bool = True,
        guaranteed_stop: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epic": epic,
            "direction": direction.upper(),
            "size": round(float(size), 2),
            "orderType": order_type.upper(),
            "currencyCode": currency,
            "forceOpen": force_open,
            "guaranteedStop": guaranteed_stop,
            "expiry": expiry,
        }

        if level is not None:
            payload["level"] = float(level)
        if limit_level is not None:
            payload["limitLevel"] = float(limit_level)
        if stop_level is not None:
            payload["stopLevel"] = float(stop_level)

        response = self._request("POST", "/positions/otc", version="2", json=payload)
        data = response.json()
        self.logger.info("IG order accepted. Deal reference: %s", data.get("dealReference"))
        return data

    def close_position(self, deal_id: str, direction: str, size: float) -> Dict[str, Any]:
        payload = {
            "dealId": deal_id,
            "direction": direction.upper(),
            "size": round(float(size), 2),
            "orderType": "MARKET",
        }
        response = self._request("POST", "/positions/otc/close", version="1", json=payload)
        data = response.json()
        self.logger.info("IG close order accepted. Deal reference: %s", data.get("dealReference"))
        return data

    # ------------------------------------------------------------------ #
    # Market data helpers
    # ------------------------------------------------------------------ #
    def fetch_prices(self, epic: str, resolution: str = "5MINUTE", max_points: int = 200) -> pd.DataFrame:
        """
        Retrieve recent OHLC data via REST (useful to seed the streaming buffer).
        """
        params = {"resolution": resolution, "pageSize": max_points}
        response = self._request("GET", f"/prices/{epic}", version="3", params=params)
        rows: List[Dict[str, Any]] = []
        for entry in response.json().get("prices", []):
            timestamp = entry.get("snapshotTimeUTC") or entry.get("snapshotTime")
            if timestamp is None:
                continue
            row = {
                "timestamp": pd.to_datetime(timestamp, utc=True),
                "open": self._pick_price(entry.get("openPrice")),
                "high": self._pick_price(entry.get("highPrice")),
                "low": self._pick_price(entry.get("lowPrice")),
                "close": self._pick_price(entry.get("closePrice")),
                "volume": entry.get("lastTradedVolume") or 0,
            }
            rows.append(row)

        if not rows:
            raise RuntimeError("IG price history request returned no rows.")

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df

    def streaming_password(self) -> str:
        """
        Build Lightstreamer password using session or OAuth tokens.
        """
        if self.cst and self.security_token:
            return f"CST-{self.cst}|XST-{self.security_token}"
        if self.oauth_access_token and self.oauth_refresh_token:
            return f"CST-{self.oauth_access_token}|XST-{self.oauth_refresh_token}"
        raise RuntimeError("No IG session tokens available for Lightstreamer connection.")

    @staticmethod
    def _pick_price(price_payload: Optional[Dict[str, Any]]) -> Optional[float]:
        if not price_payload:
            return None
        for key in ("bid", "ask", "mid", "BID", "ASK", "MID"):
            if key in price_payload and price_payload[key] is not None:
                return float(price_payload[key])
        # fallback to first value
        try:
            return float(next(iter(price_payload.values())))
        except StopIteration:
            return None

