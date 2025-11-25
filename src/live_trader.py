from __future__ import annotations

from typing import Optional

import pandas as pd

from .ig_client import IGClient
from .utils import get_logger


class IGLiveTrader:
    """Selects the latest high-probability signal and sends it to IG's demo API."""

    def __init__(self, config: dict, live_config: Optional[dict] = None, ig_client: Optional["IGClient"] = None):
        self.config = config
        self.live_config = live_config or config.get("live_trading", {})
        self.logger = get_logger("live_trader")

        self.threshold = self.live_config.get(
            "probability_threshold",
            self.config["strategy"].get("model_threshold", 0.5),
        )

        use_demo = self.live_config.get("use_demo", True)
        self.client = ig_client or IGClient(
            api_key=self.live_config.get("api_key"),
            username=self.live_config.get("username"),
            password=self.live_config.get("password"),
            account_id=self.live_config.get("account_id"),
            use_demo=use_demo,
        )

    # ------------------------------------------------------------------ #
    def run(self, candidates: pd.DataFrame, probabilities) -> Optional[dict]:
        trade_row = self._select_candidate(candidates, probabilities)
        if trade_row is None:
            return None

        return self._execute_trade(trade_row)

    # ------------------------------------------------------------------ #
    def _select_candidate(self, candidates: pd.DataFrame, probabilities) -> Optional[pd.Series]:
        if "entry_time" not in candidates.columns:
            self.logger.error("Candidates dataframe is missing 'entry_time'. Cannot pick a live signal.")
            return None

        df = candidates.copy()
        df["prob"] = probabilities
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df = df.sort_values("entry_time")

        latest_time = df["entry_time"].max()
        latest_bucket = df[df["entry_time"] == latest_time]
        if latest_bucket.empty:
            self.logger.warning("No recent candidates available for live trading.")
            return None

        top_candidate = latest_bucket.sort_values("prob", ascending=False).iloc[0]
        if top_candidate["prob"] < self.threshold:
            self.logger.info(
                "Best candidate prob %.3f is below threshold %.3f. Skipping live trade.",
                top_candidate["prob"],
                self.threshold,
            )
            return None

        return top_candidate

    # ------------------------------------------------------------------ #
    def _execute_trade(self, row: pd.Series) -> dict:
        epic = self.live_config.get("epic")
        if not epic:
            raise ValueError("`live_trading.epic` must be set in config.yml for IG orders.")

        direction = "BUY" if str(row.get("bias", "")).lower() in ("bull", "long") else "SELL"
        order_type = self.live_config.get("order_type", "MARKET")
        size = self.live_config.get("size", 1.0)
        currency = self.live_config.get("currency_code", "USD")
        expiry = self.live_config.get("expiry", "DFB")

        level = float(row["entry_price"]) if order_type.upper() != "MARKET" else None
        limit_level = row.get("tp")
        stop_level = row.get("sl")

        limit_level = float(limit_level) if limit_level is not None and pd.notna(limit_level) else None
        stop_level = float(stop_level) if stop_level is not None and pd.notna(stop_level) else None

        payload = self.client.create_position(
            epic=epic,
            direction=direction,
            size=size,
            order_type=order_type,
            level=level,
            limit_level=limit_level,
            stop_level=stop_level,
            currency=currency,
            expiry=expiry,
            force_open=self.live_config.get("force_open", True),
            guaranteed_stop=self.live_config.get("guaranteed_stop", False),
        )

        self.logger.info(
            "Live trade sent ➜ %s %s (size %.2f, prob=%.3f). Deal reference: %s",
            direction,
            epic,
            size,
            row["prob"],
            payload.get("dealReference"),
        )
        return payload

