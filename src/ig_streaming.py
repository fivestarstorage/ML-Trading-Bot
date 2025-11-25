import queue
from typing import Callable, Dict, Optional

import pandas as pd

try:
    from lightstreamer_client import LightstreamerClient, LightstreamerSubscription
except ImportError:
    LightstreamerClient = None  # type: ignore
    LightstreamerSubscription = None  # type: ignore

from .utils import get_logger
from .ig_client import IGClient


class ChartListener:
    def __init__(self, on_bar: Callable[[Dict[str, str]], None]):
        self.on_bar = on_bar

    def onItemUpdate(self, item_update):
        self.on_bar(item_update.getFields())


class IGStreamingClient:
    CHART_FIELDS = [
        "UTM",
        "BID_OPEN",
        "BID_HIGH",
        "BID_LOW",
        "BID_CLOSE",
        "LTP_OPEN",
        "LTP_HIGH",
        "LTP_LOW",
        "LTP_CLOSE",
        "CONS_END",
        "BID_TICK_VOLUME",
    ]

    def __init__(
        self,
        ig_client: IGClient,
        epic: str,
        resolution: str,
        stream_url: str,
        adapter_set: str = "DEFAULT",
    ):
        if LightstreamerClient is None or LightstreamerSubscription is None:
            raise ImportError(
                "Lightstreamer client library is not installed. Install with 'pip install lightstreamer-client-lib'."
            )
        self.logger = get_logger("ig_stream")
        self.ig_client = ig_client
        self.epic = epic
        self.resolution = resolution
        self.stream_url = stream_url
        self.adapter_set = adapter_set

        self.queue: queue.Queue = queue.Queue()
        self.client: Optional[LightstreamerClient] = None
        self.subscription: Optional[LightstreamerSubscription] = None

    def start(self):
        self.ig_client.ensure_session()
        session_password = self.ig_client.streaming_password()

        self.client = LightstreamerClient(self.stream_url, self.adapter_set)
        self.client.connection_details.set_user(self.ig_client.account_id)
        self.client.connection_details.set_password(session_password)
        self.client.connection_options.set_http_extra_headers(
            {"X-IG-API-KEY": self.ig_client.api_key}
        )

        self.logger.info("Connecting to IG Lightstreamer at %s...", self.stream_url)
        self.client.connect()

        item = f"CHART:{self.epic}:{self.resolution}"
        self.subscription = LightstreamerSubscription(mode="MERGE", items=[item], fields=self.CHART_FIELDS)
        self.subscription.addlistener(ChartListener(self._handle_update))
        self.client.subscribe(self.subscription)
        self.logger.info("Subscribed to %s", item)

    def stop(self):
        if self.subscription and self.client:
            try:
                self.client.unsubscribe(self.subscription)
            except Exception:
                pass
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass

    def _handle_update(self, values: Dict[str, str]):
        # Only push completed candles (CONS_END == "1")
        if values.get("CONS_END") == "1":
            self.queue.put(values)

    def get_next_bar(self, timeout: int = 120) -> pd.Series:
        values = self.queue.get(timeout=timeout)
        timestamp = pd.to_datetime(int(values["UTM"]), unit="ms", utc=True)
        bar = pd.Series(
            {
                "timestamp": timestamp,
                "open": float(values.get("BID_OPEN") or values.get("LTP_OPEN") or 0),
                "high": float(values.get("BID_HIGH") or values.get("LTP_HIGH") or 0),
                "low": float(values.get("BID_LOW") or values.get("LTP_LOW") or 0),
                "close": float(values.get("BID_CLOSE") or values.get("LTP_CLOSE") or 0),
                "volume": float(values.get("BID_TICK_VOLUME") or 0),
            }
        )
        return bar

