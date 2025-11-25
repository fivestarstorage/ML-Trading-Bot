from __future__ import annotations

from typing import Optional

import pandas as pd

from .alpaca_client import AlpacaClient
from .utils import get_logger


class AlpacaLiveTrader:
    """Executes signals on Alpaca (paper) accounts."""

    def __init__(
        self,
        config: dict,
        live_config: Optional[dict] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        dry_run: bool = False,
    ):
        self.config = config
        self.live_config = live_config or config.get("live_trading", {})
        self.logger = get_logger("alpaca_live_trader")
        self.threshold = self.live_config.get(
            "probability_threshold",
            self.config["strategy"].get("model_threshold", 0.55),
        )
        self.client = alpaca_client or AlpacaClient()
        self.symbol = self.live_config.get("symbol", self.config["data"]["symbol"])
        self.qty = float(self.live_config.get("qty", 1))
        self.order_type = self.live_config.get("order_type", "market")
        self.time_in_force = self.live_config.get("time_in_force", "day")
        self.dry_run = dry_run

    def run(self, candidates: pd.DataFrame, probabilities):
        trade_row = self._select_candidate(candidates, probabilities)
        if trade_row is None:
            return None
        return self._execute_trade(trade_row)

    def _select_candidate(self, candidates: pd.DataFrame, probabilities) -> Optional[pd.Series]:
        if "entry_time" not in candidates.columns:
            self.logger.error("Candidates missing 'entry_time'; cannot select a signal.")
            return None

        df = candidates.copy()
        df["prob"] = probabilities
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        latest_time = df["entry_time"].max()
        latest_bucket = df[df["entry_time"] == latest_time]
        if latest_bucket.empty:
            return None

        top_candidate = latest_bucket.sort_values("prob", ascending=False).iloc[0]
        if top_candidate["prob"] < self.threshold:
            self.logger.info(
                "Highest probability %.3f is below threshold %.3f. No trade taken.",
                top_candidate["prob"],
                self.threshold,
            )
            return None
        return top_candidate

    def _execute_trade(self, row: pd.Series):
        direction = "buy" if str(row.get("bias", "")).lower() in ("bull", "long") else "sell"
        limit_price = None
        stop_price = None
        if self.order_type.lower() == "limit":
            limit_price = float(row.get("entry_price", row.get("close", 0)))
        if self.order_type.lower() == "stop":
            stop_price = float(row.get("entry_price", row.get("close", 0)))

        if self.dry_run:
            self.logger.info(
                "[DRY-RUN] %s %s qty=%.2f (prob=%.3f)",
                direction.upper(),
                self.symbol,
                self.qty,
                row["prob"],
            )
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "direction": direction,
                "qty": self.qty,
                "probability": row["prob"],
                "entry_time": row["entry_time"],
            }

        payload = self.client.submit_order(
            symbol=self.symbol,
            qty=self.qty,
            side=direction,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        self.logger.info(
            "Alpaca order sent ➜ %s %s qty=%.2f (prob=%.3f) id=%s",
            direction.upper(),
            self.symbol,
            self.qty,
            row["prob"],
            payload.get("id"),
        )
        return payload

