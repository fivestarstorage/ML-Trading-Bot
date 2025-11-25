import time
from typing import Optional

import pandas as pd

from .data_adapter import DataAdapter
from .entries import EntryGenerator
from .features import FeatureEngineer
from .ml_model import MLModel
from .structure import StructureAnalyzer
from .alpaca_client import AlpacaClient
from .utils import get_logger


class AlpacaPollingRunner:
    """Polls Alpaca for fresh bars and pipes them through the strategy in near-real time."""

    def __init__(
        self,
        config: dict,
        adapter: DataAdapter,
        base_df: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        model: MLModel,
        live_trader,
        alpaca_client: AlpacaClient,
        dry_run: bool = False,
    ):
        self.config = config
        self.adapter = adapter
        self.base_df = base_df.sort_index()
        self.feature_engineer = feature_engineer
        self.model = model
        self.live_trader = live_trader
        self.alpaca_client = alpaca_client
        self.logger = get_logger("alpaca_live_runner")
        self.dry_run = dry_run

        lt_cfg = config.get("live_trading", {})
        self.poll_interval = lt_cfg.get("poll_interval_seconds", 60)
        self.catchup_limit = lt_cfg.get("catchup_bars", 50)
        self.max_history = lt_cfg.get("max_history_bars", 2000)
        self.min_history = lt_cfg.get("min_history_bars", 300)

        if len(self.base_df) > self.max_history:
            self.base_df = self.base_df.iloc[-self.max_history :]

        self.symbol = lt_cfg.get("symbol", config["data"]["symbol"])
        self.timeframe_label = config["data"]["timeframe_base"]
        self.alpaca_timeframe = self._format_timeframe(self.timeframe_label)

    def run(self):
        self.logger.info(
            "Starting Alpaca polling loop (symbol=%s, interval=%ss, dry_run=%s)...",
            self.symbol,
            self.poll_interval,
            self.dry_run,
        )
        try:
            while True:
                try:
                    self._poll_once()
                except Exception as exc:
                    self.logger.exception("Live loop error: %s", exc)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.logger.info("Alpaca live runner halted (Ctrl+C).")

    def _poll_once(self):
        new_bars = self._fetch_new_bars()
        if new_bars.empty:
            return

        for timestamp, row in new_bars.iterrows():
            self._append_bar(timestamp, row)
            if len(self.base_df) < self.min_history:
                continue
            self._process_latest_bar(timestamp)

    def _fetch_new_bars(self) -> pd.DataFrame:
        df = self.alpaca_client.fetch_recent_bars(
            self.symbol,
            timeframe=self.alpaca_timeframe,
            limit=self.catchup_limit,
        )
        if df.empty:
            return df
        last_ts: Optional[pd.Timestamp] = self.base_df.index.max() if not self.base_df.empty else None
        if last_ts is not None:
            df = df[df.index > last_ts]
        return df

    def _append_bar(self, timestamp: pd.Timestamp, row: pd.Series):
        row_df = pd.DataFrame([row]).set_index(pd.Index([timestamp], name="timestamp"))
        self.base_df = pd.concat([self.base_df, row_df])
        self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
        if len(self.base_df) > self.max_history:
            self.base_df = self.base_df.iloc[-self.max_history :]

    def _process_latest_bar(self, bar_timestamp: pd.Timestamp):
        df = self.base_df.copy()
        df_features = self.feature_engineer.calculate_technical_features(df)

        # Refresh higher timeframe structure every bar to keep context aligned.
        df_h4 = self.adapter.resample_data(df, self.config["timeframes"]["h4"])
        df_d1 = self.adapter.resample_data(df, 1440)
        structure = StructureAnalyzer(df_h4, self.config, daily_df=df_d1)
        entry_generator = EntryGenerator(self.config, structure, self.feature_engineer)

        fvgs = self.feature_engineer.detect_fvgs(df_features)
        obs = self.feature_engineer.detect_obs(df_features)
        candidates = entry_generator.generate_candidates(df_features, df_h4, fvgs, obs)
        if candidates.empty or "entry_time" not in candidates.columns:
            return

        latest_candidates = candidates[candidates["entry_time"] == bar_timestamp]
        if latest_candidates.empty:
            return

        probabilities = self.model.predict_proba(latest_candidates)
        self.live_trader.run(latest_candidates, probabilities)

    @staticmethod
    def _format_timeframe(label: str) -> str:
        label = label.lower()
        if label.endswith("m"):
            return f"{int(label[:-1])}Min"
        if label.endswith("h"):
            return f"{int(label[:-1])}Hour"
        if label.endswith("d"):
            return f"{int(label[:-1])}Day"
        raise ValueError(f"Unsupported Alpaca timeframe label '{label}' for live polling.")

