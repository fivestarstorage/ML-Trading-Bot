import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator
from .utils import get_logger

logger = get_logger()

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.fvg_lookback = config['strategy'].get('fvg_lookback', 10)
        self.ob_lookback = config['strategy'].get('ob_lookback', 20)
        self.atr_period = config['strategy']['atr_period']
        self.killzones = config['strategy'].get('killzones', {})
        self._killzone_windows = self._build_killzone_windows(self.killzones)

    def detect_fvgs(self, df):
        """
        Detect Bullish and Bearish FVGs.
        Bullish FVG:
        1. Large Green Candle (Body > Avg body of prev 10)
        2. Left.High < Right.Low (Gap)

        Returns a list of FVG dicts or a DataFrame mask.
        """
        fvgs = []

        # Pre-calculate body sizes
        body_sizes = (df['close'] - df['open']).abs()

        highs = df['high']
        lows = df['low']
        opens = df['open']
        closes = df['close']

        for i in range(1, len(df) - 1):
            # Calculate avg body of previous candles (up to 10, or fewer if not available)
            start_idx = max(0, i - 10)
            prev_bodies = body_sizes.iloc[start_idx:i]
            avg_prev_body = prev_bodies.mean() if not prev_bodies.empty else 0.0

            # Bullish FVG
            if closes.iloc[i] > opens.iloc[i] and body_sizes.iloc[i] > avg_prev_body:
                left_high = highs.iloc[i-1]
                right_low = lows.iloc[i+1]

                if left_high < right_low:
                    fvgs.append({
                        'type': 'bull',
                        'start_idx': i+1, # Index when FVG is confirmed (close of right candle)
                        'top': right_low,
                        'bottom': left_high,
                        'created_at': df.index[i+1],
                        'center_idx': i
                    })

            # Bearish FVG
            elif closes.iloc[i] < opens.iloc[i] and body_sizes.iloc[i] > avg_prev_body:
                left_low = lows.iloc[i-1]
                right_high = highs.iloc[i+1]

                if left_low > right_high:
                    fvgs.append({
                        'type': 'bear',
                        'start_idx': i+1,
                        'top': left_low,
                        'bottom': right_high,
                        'created_at': df.index[i+1],
                        'center_idx': i
                    })

        return fvgs

    def detect_obs(self, df):
        """
        Detect Order Blocks (OB).
        Bullish OB: Last Bearish candle before a Bullish move that caused BOS.
        Bearish OB: Last Bullish candle before a Bearish move that caused BOS.
        
        Simplified heuristic per prompt:
        "Last bearish/bullish engulfing candle before a directional move"
        We'll look for Engulfing patterns followed by continuation.
        """
        obs = []
        # Simple Engulfing detection
        # Bullish Engulfing: Red candle (i-1), Green candle (i)
        # Open[i] < Close[i-1], Close[i] > Open[i-1] (envelops body)
        
        opens = df['open']
        closes = df['close']
        highs = df['high']
        lows = df['low']
        
        for i in range(1, len(df)):
            # Bullish Engulfing (Potential Bull OB is the Red candle at i-1)
            if (closes.iloc[i-1] < opens.iloc[i-1]) and \
               (closes.iloc[i] > opens.iloc[i]) and \
               (closes.iloc[i] > opens.iloc[i-1]) and \
               (opens.iloc[i] < closes.iloc[i-1]):
               
               # Check for follow through (next 1-2 bars are green or price goes up)
               # For features, we just mark it.
               obs.append({
                   'type': 'bull',
                   'created_at': df.index[i],
                   'top': highs.iloc[i-1],
                   'bottom': lows.iloc[i-1],
                   'idx': i
               })

            # Bearish Engulfing (Potential Bear OB is the Green candle at i-1)
            if (closes.iloc[i-1] > opens.iloc[i-1]) and \
               (closes.iloc[i] < opens.iloc[i]) and \
               (closes.iloc[i] < opens.iloc[i-1]) and \
               (opens.iloc[i] > closes.iloc[i-1]):
               
               obs.append({
                   'type': 'bear',
                   'created_at': df.index[i],
                   'top': highs.iloc[i-1],
                   'bottom': lows.iloc[i-1],
                   'idx': i
               })
               
        return obs

    def calculate_technical_features(self, df):
        """
        Calculate RSI, ATR, etc.
        """
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period)
        df['atr'] = atr.average_true_range()
        
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        
        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx.adx()
        
        # Rolling volatility (std of returns)
        df['returns'] = df['close'].pct_change()
        df['rolling_vol'] = df['returns'].rolling(window=20).std()
        
        # Volume Spike (Volume / Rolling Avg Volume)
        df['vol_spike'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # ATR regime
        atr_long_window = self.config['strategy'].get('atr_regime_window', 100)
        df['atr_long'] = df['atr'].rolling(window=atr_long_window, min_periods=10).mean()
        df['atr_ratio'] = df['atr'] / df['atr_long']
        df['atr_ratio'] = df['atr_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # Volume z-score
        vol_window = self.config['strategy'].get('volume_regime_window', 50)
        vol_mean = df['volume'].rolling(window=vol_window, min_periods=5).mean()
        vol_std = df['volume'].rolling(window=vol_window, min_periods=5).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
        df['volume_zscore'] = df['volume_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Candle impulse metrics
        price_range = df['high'] - df['low']
        body_range = (df['close'] - df['open']).abs()
        df['impulse_factor'] = price_range / df['atr'].replace(0, np.nan)
        df['body_factor'] = body_range / df['atr'].replace(0, np.nan)
        df['impulse_factor'] = df['impulse_factor'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['body_factor'] = df['body_factor'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Session tagging
        session_labels, session_codes, killzone_flags = self._assign_sessions(df.index)
        df['session_label'] = session_labels
        df['session_code'] = session_codes
        df['in_killzone'] = killzone_flags
        
        return df

    def make_features(self, df, structure_info=None, fvgs=None, obs=None):
        """
        Assemble the full feature vector for a given DataFrame index/row.
        This is called for each candidate entry.
        """
        # Assuming df already has technicals calculated
        
        # We need to map the list of FVGs/OBs to the current row
        # For a given row 'i', we check if price is inside any *active* FVG/OB
        # Active means created before 'i' and not yet invalidated (price broke through).
        
        # This function might be better suited to run on the full dataframe to generate
        # a features dataframe, rather than per-row, for training speed.
        pass

    def _build_killzone_windows(self, killzones):
        windows = []
        for idx, (label, window) in enumerate(killzones.items(), start=1):
            start = self._time_to_minutes(window.get('start', "00:00"))
            end = self._time_to_minutes(window.get('end', "23:59"))
            windows.append({
                'label': label.lower().replace(" ", "_"),
                'start': start,
                'end': end,
                'code': idx
            })
        return windows
    
    def _assign_sessions(self, index):
        if not self._killzone_windows:
            return ['off'] * len(index), [0] * len(index), [False] * len(index)
        
        labels = []
        codes = []
        kill_flags = []
        for ts in index:
            minute_val = ts.hour * 60 + ts.minute
            label = 'off'
            code = 0
            is_killzone = False
            for window in self._killzone_windows:
                if self._in_window(minute_val, window['start'], window['end']):
                    label = window['label']
                    code = window['code']
                    is_killzone = True
                    break
            labels.append(label)
            codes.append(code)
            kill_flags.append(is_killzone)
        return labels, codes, kill_flags
    
    def _time_to_minutes(self, time_str):
        hour, minute = [int(x) for x in time_str.split(":")]
        return hour * 60 + minute
    
    def _in_window(self, minute_val, start, end):
        if start <= end:
            return start <= minute_val <= end
        # Overnight window
        return minute_val >= start or minute_val <= end


