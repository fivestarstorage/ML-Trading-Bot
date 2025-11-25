import time
from typing import Optional

import pandas as pd

from .features import FeatureEngineer
from .entries import EntryGenerator
from .structure import StructureAnalyzer
from .ml_model import MLModel
from .live_trader import IGLiveTrader
from .ig_streaming import IGStreamingClient
from .utils import get_logger


class LiveStreamingRunner:
    def __init__(
        self,
        config: dict,
        base_df: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_d1: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        entry_generator: EntryGenerator,
        model: MLModel,
        live_trader: IGLiveTrader,
        streamer: IGStreamingClient,
    ):
        self.config = config
        self.base_df = base_df.sort_index()
        self.df_h4 = df_h4
        self.df_d1 = df_d1
        self.feature_engineer = feature_engineer
        self.entry_generator = entry_generator
        self.model = model
        self.live_trader = live_trader
        self.streamer = streamer
        self.logger = get_logger("live_runner")

        lt_cfg = config.get("live_trading", {})
        self.max_history = lt_cfg.get("max_history_bars", 2000)
        self.min_history = lt_cfg.get("min_history_bars", 300)

        if len(self.base_df) > self.max_history:
            self.base_df = self.base_df.iloc[-self.max_history :]

    def run(self):
        self.logger.info("Starting IG streaming loop...")
        self.streamer.start()

        try:
            while True:
                bar_series = self.streamer.get_next_bar()
                self._append_bar(bar_series)

                if len(self.base_df) < self.min_history:
                    continue

                try:
                    self._process_latest_bar(bar_series["timestamp"])
                except Exception as exc:
                    self.logger.exception("Error processing live bar: %s", exc)
                    time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping live stream (Ctrl+C)...")
        finally:
            self.streamer.stop()

    def _append_bar(self, bar_series: pd.Series):
        bar_df = pd.DataFrame([bar_series]).set_index("timestamp")
        self.base_df = pd.concat([self.base_df, bar_df])
        self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
        if len(self.base_df) > self.max_history:
            self.base_df = self.base_df.iloc[-self.max_history :]

    def _process_latest_bar(self, bar_timestamp):
        df = self.base_df.copy()

        df_features = self.feature_engineer.calculate_technical_features(df)
        fvgs = self.feature_engineer.detect_fvgs(df_features)
        obs = self.feature_engineer.detect_obs(df_features)

        candidates = self.entry_generator.generate_candidates(df_features, self.df_h4, fvgs, obs)
        latest_candidates = candidates[candidates["entry_time"] == bar_timestamp]
        if latest_candidates.empty:
            return

        probabilities = self.model.predict_proba(latest_candidates)
        self.live_trader.run(latest_candidates, probabilities)

