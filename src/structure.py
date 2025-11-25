import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class StructureAnalyzer:
    def __init__(self, df, config, daily_df=None):
        self.df = df
        self.lookback = config['strategy']['h4_swing_lookback']
        self.inducement_lookback = config['strategy']['inducement_lookback']
        self.daily_df = daily_df
        self.daily_fast = config['strategy'].get('daily_trend_fast', 50)
        self.daily_slow = config['strategy'].get('daily_trend_slow', 200)
        self._identify_swings()
        self._prepare_daily_context()
        
    def _identify_swings(self, order=3):
        """
        Identify swing highs and lows using a rolling window (fractal) approach.
        Order 3 means 3 bars left and 3 bars right (7 bar window).
        """
        df = self.df
        
        # Vectorized rolling max/min for swings
        # Note: This uses future data if we shift -order. 
        # For strict backtesting without lookahead, a swing high at time T is only confirmed at T+order.
        # However, scanning "last N bars" implies we are at T_current and looking back.
        # So at T_current, we can see a swing high that happened at T_current - order.
        
        # We'll create boolean masks for swings confirmed by T_current
        # But for simplicity in "scan last N", we just calculate peaks on the whole series
        # and when querying "at index i", we only look at swings strictly before i (or i-order).
        
        # Using argrelextrema or similar is faster but let's use rolling for pandas compatibility
        # A swing high at index i is max of [i-order, i+order]
        
        # In a live/step-by-step context:
        # At time t, we check if t-order was a high.
        
        self.swing_highs = pd.Series(np.nan, index=df.index)
        self.swing_lows = pd.Series(np.nan, index=df.index)
        
        # We need to implement this so it can be queried efficiently
        # For now, let's compute simple fractals
        
        # Rolling window centered approach requires shifting
        # window size = 2*order + 1
        
        # To avoid lookahead bias in the stored series, we can mark the swing 
        # at the time it is CONFIRMED.
        # Swing High at T is confirmed at T+order.
        
        # But typically we want to know "where was the last swing high".
        # So we mark the swing at its actual time T, but we only "see" it from T+order onwards.
        
        for i in range(order, len(df) - order):
            # Check High
            current_high = df['high'].iloc[i]
            left_highs = df['high'].iloc[i-order:i]
            right_highs = df['high'].iloc[i+1:i+order+1]
            
            if (current_high > left_highs).all() and (current_high > right_highs).all():
                self.swing_highs.iloc[i] = current_high
                
            # Check Low
            current_low = df['low'].iloc[i]
            left_lows = df['low'].iloc[i-order:i]
            right_lows = df['low'].iloc[i+1:i+order+1]
            
            if (current_low < left_lows).all() and (current_low < right_lows).all():
                self.swing_lows.iloc[i] = current_low

    def get_bias(self, current_idx):
        """
        Determine bias (Bullish/Bearish/Neutral) based on BoS/CHoCH in the last N bars 
        relative to current_idx.
        """
        # Slice the dataframe up to current_idx
        # We only 'know' swings that are confirmed. 
        # If swing detection order is 3, we can only see swings up to current_idx - 3.
        
        # Lookback window
        if isinstance(current_idx, pd.Timestamp):
             # Find integer location
             try:
                 idx = self.df.index.get_loc(current_idx)
             except KeyError:
                 # Approximate or return neutral
                 return 'neutral', {}
        else:
             idx = current_idx
             
        start_idx = max(0, idx - self.lookback)
        
        # Get swings in this window [start_idx, idx]
        # Filter out unconfirmed swings (those within last 'order' bars)
        # Assuming order=3 hardcoded for now or passed in config
        order = 3 
        confirmed_end_idx = idx - order
        
        if confirmed_end_idx < start_idx:
            return 'neutral', {}

        # Get swings
        relevant_highs = self.swing_highs.iloc[start_idx:confirmed_end_idx+1].dropna()
        relevant_lows = self.swing_lows.iloc[start_idx:confirmed_end_idx+1].dropna()
        
        # Simple logic for BoS:
        # Bullish: Most recent significant structure break was a High.
        # Bearish: Most recent significant structure break was a Low.
        
        # We need to trace the sequence.
        # This is complex to do perfectly efficiently in python loop every step.
        # Simplified heuristic:
        # Find the most recent major swing high and low.
        # Check if price broke them.
        
        # Let's look at the sequence of highs and lows in the window
        swings = []
        for t, price in relevant_highs.items():
            swings.append({'type': 'H', 'price': price, 'time': t})
        for t, price in relevant_lows.items():
            swings.append({'type': 'L', 'price': price, 'time': t})
            
        swings.sort(key=lambda x: x['time'])
        
        if not swings:
            return 'neutral', {}
            
        # Determine trend from sequence
        # Bullish: HL -> HH
        # Bearish: LH -> LL
        
        bias = 'neutral'
        last_swing = swings[-1]
        
        # Iterate to find latest valid BoS
        # This is a simplified state machine
        state = 'neutral' # 'bull', 'bear'
        
        # We need to track the "last valid high" and "last valid low" that define the range
        # If price closes above last valid high -> Bullish BOS
        # If price closes below last valid low -> Bearish BOS
        
        # For specific CHoCH logic:
        # CHoCH is usually the first break against the trend.
        
        # Let's try to determine the state at the END of the window based on price action
        # relative to the last confirmed swings.
        
        # 1. Find the highest high and lowest low in the recent window? No, that's range.
        # 2. Use the sequence.
        
        if len(swings) < 2:
            return 'neutral', {}

        # Let's look at the last 4 swings to determine pattern
        recent_swings = swings[-4:] 
        
        # Heuristic: Compare last two highs and last two lows
        # (This is an approximation of the complex BoS logic)
        
        # Better approach:
        # Look at the most recent confirmed High (H_last) and Low (L_last).
        # If current price > H_last -> potentially bullish.
        # If current price < L_last -> potentially bearish.
        
        # User requirement: "Scan last N=60 H4 bars... Identify sequences for BOS... If BOS occurs, set bias"
        
        # Let's identify the most recent break.
        # We iterate through the window bars to see breaks of the identified swings.
        
        last_bos_type = 'neutral'
        last_bos_idx = -1
        
        # We only care about the outcome at 'idx'
        # optimization: Check if the *current* price structure supports bullish/bearish
        
        # Let's stick to the definition:
        # Bullish: Price made a Higher High (broke previous swing high)
        # Bearish: Price made a Lower Low (broke previous swing low)
        
        # Find the last swing high and last swing low relative to the current moment
        # (excluding the current developing swing)
        
        last_h = relevant_highs.iloc[-1] if not relevant_highs.empty else None
        last_l = relevant_lows.iloc[-1] if not relevant_lows.empty else None
        
        prev_h = relevant_highs.iloc[-2] if len(relevant_highs) > 1 else None
        prev_l = relevant_lows.iloc[-2] if len(relevant_lows) > 1 else None
        
        bias = 'neutral'
        
        # Check sequences
        if last_h and prev_h and last_l and prev_l:
            if last_h > prev_h and last_l > prev_l:
                bias = 'bull'
            elif last_h < prev_h and last_l < prev_l:
                bias = 'bear'
            elif last_h > prev_h: # HH but not necessarily HL yet, or HL missing
                 bias = 'bull' # Lean bull on break of high
            elif last_l < prev_l:
                 bias = 'bear' # Lean bear on break of low
                 
        # Refine with price action relative to LAST swing
        # If we were bull, but price broke below Last HL -> CHoCH -> Bear or Neutral
        
        # Implementation of strict "right zone"
        # Range High/Low
        range_high = self.df['high'].iloc[start_idx:idx+1].max()
        range_low = self.df['low'].iloc[start_idx:idx+1].min()
        mid_point = (range_high + range_low) / 2
        
        current_close = self.df['close'].iloc[idx]
        
        in_discount = current_close < mid_point
        in_premium = current_close > mid_point
        
        info = {
            'range_high': range_high,
            'range_low': range_low,
            'mid_point': mid_point,
            'in_discount': in_discount,
            'in_premium': in_premium,
            'last_high': last_h,
            'last_low': last_l
        }
        price_range = max(range_high - range_low, 1e-6)
        info['premium_pct'] = (range_high - current_close) / price_range
        info['discount_pct'] = (current_close - range_low) / price_range
        return bias, info

    def check_inducement(self, current_idx):
        """
        Check for inducement in the last N bars (inducement_lookback).
        Inducement: Price took out a minor swing point but closed back inside/rejected.
        """
        if isinstance(current_idx, pd.Timestamp):
             try:
                 idx = self.df.index.get_loc(current_idx)
             except KeyError:
                 return False
        else:
             idx = current_idx
             
        start_idx = max(0, idx - self.inducement_lookback)
        
        # Look for minor swings (e.g. fractal order 1 or 2) being breached
        # For simplicity, we check if any bar in lookback breached a recent minor swing 
        # and closed opposite.
        
        # This is a bit complex to detect perfectly without lower timeframe data, 
        # but we can approximate with H4 wicks.
        # "Candle that pierced previous minor swing and then closed back inside"
        
        # Let's look at the previous completed candle (idx-1) or current candle?
        # "within last 6 H4 bars"
        
        recent_window = self.df.iloc[start_idx:idx+1]
        
        # We need 'minor' swings - let's assume they are the same swings as identified or slightly more sensitive
        # Let's use the swings we already identified (order 3)
        
        # Find the nearest swing High/Low before this window
        # (This logic is getting complicated, will simplify for robustness)
        
        # Inducement heuristic:
        # Check if any candle in recent window has a wick that goes beyond a previous local extremum 
        # but the body closes inside.
        
        has_inducement = False
        
        # Iterate recent bars
        for i in range(len(recent_window)):
            bar = recent_window.iloc[i]
            # This needs context of previous bars to know if it pierced something
            # For now, we will return a placeholder True/False based on wick size vs body
            # If wick is large relative to body, it's a rejection which is inducement-like
            
            body = abs(bar['close'] - bar['open'])
            upper_wick = bar['high'] - max(bar['open'], bar['close'])
            lower_wick = min(bar['open'], bar['close']) - bar['low']
            
            if upper_wick > body * 2 or lower_wick > body * 2:
                has_inducement = True # Simplified "Liquidity sweep" signature
                
        return has_inducement

    def get_enriched_context(self, timestamp):
        idx = self._get_index(self.df, timestamp)
        if idx is None:
            return 'neutral', {}, 'neutral', {}
        bias, struct = self.get_bias(idx)
        daily_trend, daily_info = self.get_daily_trend(timestamp)
        struct['daily_trend'] = daily_trend
        struct['daily_context'] = daily_info
        return bias, struct, daily_trend, daily_info

    def get_daily_trend(self, timestamp):
        if self.daily_df is None or self.daily_df.empty:
            return 'neutral', {}
        idx = self._get_index(self.daily_df, timestamp)
        if idx is None:
            return 'neutral', {}
        row = self.daily_df.iloc[idx]
        fast = row.get('ema_fast', np.nan)
        slow = row.get('ema_slow', np.nan)
        close = row.get('close', np.nan)
        if pd.isna(fast) or pd.isna(slow) or pd.isna(close):
            return 'neutral', {'ema_fast': fast, 'ema_slow': slow, 'close': close}
        if fast > slow and close > slow:
            trend = 'bull'
        elif fast < slow and close < slow:
            trend = 'bear'
        else:
            trend = 'range'
        return trend, {'ema_fast': fast, 'ema_slow': slow, 'close': close}

    def _prepare_daily_context(self):
        if self.daily_df is None or self.daily_df.empty:
            return
        df = self.daily_df.copy().sort_index()
        df['ema_fast'] = df['close'].ewm(span=self.daily_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.daily_slow, adjust=False).mean()
        self.daily_df = df

    def _get_index(self, df, timestamp):
        if df is None or df.empty:
            return None
        try:
            idx = df.index.get_indexer([timestamp], method='pad')[0]
        except (KeyError, IndexError):
            return None
        if idx < 0:
            return None
        return idx

