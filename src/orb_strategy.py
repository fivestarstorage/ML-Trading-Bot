"""
Opening Range Breakout (ORB) Strategy - Enhanced with Research-Based Filters
Based on academic research and best practices for high-yielding ORB trading.

Key Enhancements:
- VWAP directional filter
- Volume confirmation (RVOL)
- Gap condition analysis
- Multi-timeframe trend alignment
- ATR volatility screening
- Retest entry detection
"""
import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class ORBEntryGenerator:
    def __init__(self, config):
        self.config = config
        self.orb_bars = config['strategy'].get('orb_bars', 12)  # Default 1 hour on 5m = 12 bars

        # Legacy parameters for backward compatibility
        self.orb_start_hour = config['strategy'].get('orb_start_hour', 0)
        self.orb_start_minute = config['strategy'].get('orb_start_minute', 0)
        self.orb_end_hour = config['strategy'].get('orb_end_hour', 23)
        self.orb_end_minute = config['strategy'].get('orb_end_minute', 59)

        # Multi-session support for 24-hour markets (XAU/Gold)
        self.sessions = config['strategy'].get('sessions', {})

        self.only_trade_after_14utc = config['strategy'].get('only_trade_after_14utc', False)
        self.require_atr_expansion = config['strategy'].get('require_atr_expansion', False)
        self.momentum_period = config['strategy'].get('momentum_period', 5)
        self.min_momentum = config['strategy'].get('min_momentum', 0.0)
        self.max_trades_per_day = config['strategy'].get('max_trades_per_day', 10)
        self.max_trades_per_session = config['strategy'].get('max_trades_per_session', 3)
        self.tp_mult = config['strategy']['tp_atr_mult']
        self.sl_atr_mult = config['strategy']['sl_atr_mult']
        self.label_window = config['strategy'].get('label_window', 200)

        # Enhanced filters from research
        self.use_vwap_filter = config['strategy'].get('use_vwap_filter', True)
        self.use_volume_filter = config['strategy'].get('use_volume_filter', True)
        self.min_rvol_orb = config['strategy'].get('min_rvol_orb', 1.3)  # Min RVOL during opening range
        self.min_rvol_breakout = config['strategy'].get('min_rvol_breakout', 1.5)  # Min RVOL at breakout
        self.use_gap_filter = config['strategy'].get('use_gap_filter', True)
        self.max_gap_percent = config['strategy'].get('max_gap_percent', 3.0)  # Skip if gap > 3%
        self.require_retest = config['strategy'].get('require_retest', False)  # Wait for pullback
        self.atr_vol_min = config['strategy'].get('atr_vol_min', 0.8)  # Min ATR vs 20-day avg
        self.atr_vol_max = config['strategy'].get('atr_vol_max', 2.0)  # Max ATR vs 20-day avg

    def _find_session_opening_range(self, day_data, session_name, session_config):
        """Find opening range bars for a specific session"""
        orb_bars_list = []
        start_minutes = session_config['start_hour'] * 60 + session_config['start_minute']
        end_minutes = session_config['end_hour'] * 60 + session_config['end_minute']

        for idx, row in day_data.iterrows():
            bar_minutes = row['hour'] * 60 + row['minute']
            if start_minutes <= bar_minutes <= end_minutes:
                orb_bars_list.append(idx)
                if len(orb_bars_list) >= self.orb_bars:
                    break

        return orb_bars_list[:self.orb_bars] if len(orb_bars_list) >= self.orb_bars else []

    def generate_candidates(self, df_base):
        """
        Generate ORB breakout candidates similar to New_XAU_Bot.mq5
        
        Strategy:
        1. Set opening range during 13:30-14:30 UTC (4 bars on M15 = 1 hour)
        2. Wait for breakout above high (BUY) or below low (SELL)
        3. Only trade after 14:00 UTC (skip stop hunt hour)
        4. Require ATR expansion
        5. Optional momentum filter
        """
        candidates = []
        
        logger.info("Generating ORB candidates...")
        
        # Ensure we have technical indicators
        if 'atr' not in df_base.columns:
            from .features import FeatureEngineer
            fe = FeatureEngineer(self.config)
            df_base = fe.calculate_technical_features(df_base)

        # Add VWAP
        df_base = self._calculate_vwap(df_base)

        # Add RVOL (Relative Volume)
        df_base = self._calculate_rvol(df_base)

        # Calculate 20-day ATR average for volatility screening
        df_base = self._calculate_atr_avg(df_base)
        
        # Add date/time columns
        df_base = df_base.copy()
        df_base['date'] = df_base.index.date
        df_base['hour'] = df_base.index.hour
        df_base['minute'] = df_base.index.minute
        
        # Calculate momentum (price change over N bars)
        df_base['momentum'] = df_base['close'].diff(self.momentum_period)
        
        unique_dates = sorted(df_base['date'].unique())
        
        for date_idx, date in enumerate(unique_dates):
            if date_idx % 50 == 0:
                logger.info(f"Processing date {date_idx+1}/{len(unique_dates)}: {date}")
            
            day_data = df_base[df_base['date'] == date].copy().sort_index()
            if len(day_data) < self.orb_bars + 20:  # Need enough bars
                continue
            
            # Set opening range (13:30-14:30 UTC)
            # Find bars within opening range period
            orb_bars_list = []
            for idx, row in day_data.iterrows():
                hour = row['hour']
                minute = row['minute']
                
                # Check if within opening range time window (13:30-14:30 UTC)
                # Convert to minutes for easier comparison
                bar_minutes = hour * 60 + minute
                start_minutes = self.orb_start_hour * 60 + self.orb_start_minute  # 13:30 = 810
                end_minutes = self.orb_end_hour * 60 + self.orb_end_minute  # 14:30 = 870
                
                if start_minutes <= bar_minutes <= end_minutes:
                    orb_bars_list.append(idx)
            
            # If we don't have enough bars in the exact window, use first N bars of the day
            # This is more flexible and handles cases where data might not align perfectly
            if len(orb_bars_list) < self.orb_bars:
                if len(day_data) >= self.orb_bars:
                    # Use first N bars of the day as opening range
                    orb_bars_list = list(day_data.index[:self.orb_bars])
                else:
                    continue
            
            # Get first N bars for opening range
            orb_indices = sorted(orb_bars_list)[:self.orb_bars]
            orb_range = day_data.loc[orb_indices]
            orb_high = orb_range['high'].max()
            orb_low = orb_range['low'].min()
            orb_time = orb_range.index[-1]  # End of opening range
            
            # Get ATR at opening range end
            orb_atr = orb_range['atr'].iloc[-1] if 'atr' in orb_range.columns else day_data['atr'].iloc[0]

            # Gap filter: Check opening gap vs previous day close
            if self.use_gap_filter and date_idx > 0:
                prev_date = unique_dates[date_idx - 1]
                prev_day_data = df_base[df_base['date'] == prev_date]
                if len(prev_day_data) > 0:
                    prev_close = prev_day_data['close'].iloc[-1]
                    today_open = day_data['open'].iloc[0]
                    gap_pct = abs((today_open - prev_close) / prev_close) * 100

                    if gap_pct > self.max_gap_percent:
                        if date_idx < 3:
                            logger.info(f"Date {date}: Skipping - gap too large ({gap_pct:.2f}%)")
                        continue

            # ATR volatility filter
            if 'atr_avg_20d' in orb_range.columns:
                atr_avg = orb_range['atr_avg_20d'].iloc[-1]
                if atr_avg > 0:
                    atr_ratio = orb_atr / atr_avg
                    if atr_ratio < self.atr_vol_min or atr_ratio > self.atr_vol_max:
                        if date_idx < 3:
                            logger.info(f"Date {date}: Skipping - ATR ratio {atr_ratio:.2f} outside range [{self.atr_vol_min}, {self.atr_vol_max}]")
                        continue

            # Volume filter during opening range
            if self.use_volume_filter and 'rvol' in orb_range.columns:
                avg_rvol_orb = orb_range['rvol'].mean()
                if avg_rvol_orb < self.min_rvol_orb:
                    if date_idx < 3:
                        logger.info(f"Date {date}: Skipping - opening range RVOL too low ({avg_rvol_orb:.2f})")
                    continue

            # Look for breakouts AFTER opening range period
            breakout_data = day_data[day_data.index > orb_time].copy()

            if len(breakout_data) < self.label_window:
                continue
            
            # Debug: Log opening range info for first few days
            if date_idx < 3:
                logger.info(f"Date {date}: ORB High={orb_high:.2f}, Low={orb_low:.2f}, Range={orb_high-orb_low:.2f}, "
                           f"Bars after ORB: {len(breakout_data)}")
            
            # Track daily trade count and breakout flags
            daily_trades = 0
            buy_breakout_triggered = False
            sell_breakout_triggered = False
            
            # Previous ATR for expansion check
            prev_atr = orb_atr
            
            for i in range(len(breakout_data) - self.label_window):
                if daily_trades >= self.max_trades_per_day:
                    break
                
                row = breakout_data.iloc[i]
                current_time = row.name
                current_price = row['close']
                current_atr = row.get('atr', orb_atr)  # Fallback to orb_atr if missing
                
                # Only trade after 14:00 UTC if configured
                if self.only_trade_after_14utc and current_time.hour < 14:
                    continue

                # For 24-hour markets like XAU, allow trading all day
                # For stock indices like SPY, uncomment below to stop at 16:00 UTC
                # if current_time.hour >= 16:
                #     break
                
                # Debug: Log potential breakouts for first day
                if date_idx == 0 and i < 20:
                    bar_high = row.get('high', current_price)
                    bar_low = row.get('low', current_price)
                    if bar_high > orb_high or bar_low < orb_low:
                        logger.info(f"Potential breakout at {current_time}: Price={current_price:.2f}, "
                                   f"ORB High={orb_high:.2f}, ORB Low={orb_low:.2f}, "
                                   f"Bar High={bar_high:.2f}, Bar Low={bar_low:.2f}")
                
                # Reset breakout flags if price returns to range (allows re-entry)
                if buy_breakout_triggered and current_price < orb_high:
                    buy_breakout_triggered = False
                if sell_breakout_triggered and current_price > orb_low:
                    sell_breakout_triggered = False
                
                # Check ATR expansion (only if enabled)
                if self.require_atr_expansion:
                    # ATR expanding if current >= previous (with 2% tolerance for noise)
                    if current_atr < prev_atr * 0.98:  # ATR not expanding
                        prev_atr = current_atr
                        continue
                    prev_atr = current_atr
                else:
                    # Update prev_atr even if expansion check is disabled
                    prev_atr = current_atr
                
                # Check for BUY breakout (price breaks above ORB high)
                # Use high/low of the bar, not just close
                bar_high = row.get('high', current_price)
                bar_low = row.get('low', current_price)

                # Check if bar high breaks above ORB high
                if bar_high > orb_high and not buy_breakout_triggered:
                    # VWAP filter: Only buy if price is above VWAP
                    if self.use_vwap_filter:
                        vwap = row.get('vwap', current_price)
                        if current_price < vwap:
                            continue

                    # Volume filter at breakout
                    if self.use_volume_filter:
                        rvol = row.get('rvol', 1.0)
                        if rvol < self.min_rvol_breakout:
                            continue

                    # Check momentum if required
                    momentum = row.get('momentum', 0)
                    if self.min_momentum > 0 and momentum < self.min_momentum:
                        continue
                    
                    # Calculate TP/SL
                    # Entry price is the breakout level (ORB high) for buy, or use current price
                    entry_price = max(orb_high, current_price)  # Enter at breakout level or better
                    sl = entry_price - (self.sl_atr_mult * current_atr)
                    tp = entry_price + (self.tp_mult * current_atr)
                    
                    # Check outcome
                    future_window = breakout_data.iloc[i+1 : i+1+self.label_window]
                    outcome, exit_time, pnl_r = self._check_outcome(
                        future_window, entry_price, tp, sl, 'bull'
                    )
                    
                    # Determine session label for compatibility with WFA
                    hour = current_time.hour
                    if 0 <= hour < 7:
                        session_label = 'asia'
                    elif 7 <= hour < 13:
                        session_label = 'london'
                    elif 13 <= hour < 20:
                        session_label = 'new_york'
                    else:
                        session_label = 'off'

                    candidate = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'bias': 'bull',
                        'entry_type': 'orb_breakout',
                        'tp': tp,
                        'sl': sl,
                        'atr': current_atr,
                        'target': outcome,
                        'pnl_r': pnl_r,
                        'orb_high': orb_high,
                        'orb_low': orb_low,
                        'momentum': momentum,
                        'rsi': row.get('rsi', 50),
                        'adx': row.get('adx', 20),
                        'hour': current_time.hour,
                        'dayofweek': current_time.dayofweek,
                        'session_label': session_label,  # For WFA compatibility
                    }
                    candidates.append(candidate)
                    daily_trades += 1
                    buy_breakout_triggered = True
                    continue
                
                # Check for SELL breakout (price breaks below ORB low)
                elif bar_low < orb_low and not sell_breakout_triggered:
                    # VWAP filter: Only sell if price is below VWAP
                    if self.use_vwap_filter:
                        vwap = row.get('vwap', current_price)
                        if current_price > vwap:
                            continue

                    # Volume filter at breakout
                    if self.use_volume_filter:
                        rvol = row.get('rvol', 1.0)
                        if rvol < self.min_rvol_breakout:
                            continue

                    # Check momentum if required
                    momentum = row.get('momentum', 0)
                    if self.min_momentum > 0 and momentum > -self.min_momentum:
                        continue
                    
                    # Calculate TP/SL
                    # Entry price is the breakout level (ORB low) for sell, or use current price
                    entry_price = min(orb_low, current_price)  # Enter at breakout level or better
                    sl = entry_price + (self.sl_atr_mult * current_atr)
                    tp = entry_price - (self.tp_mult * current_atr)
                    
                    # Check outcome
                    future_window = breakout_data.iloc[i+1 : i+1+self.label_window]
                    outcome, exit_time, pnl_r = self._check_outcome(
                        future_window, entry_price, tp, sl, 'bear'
                    )
                    
                    # Determine session label for compatibility with WFA
                    hour = current_time.hour
                    if 0 <= hour < 7:
                        session_label = 'asia'
                    elif 7 <= hour < 13:
                        session_label = 'london'
                    elif 13 <= hour < 20:
                        session_label = 'new_york'
                    else:
                        session_label = 'off'

                    candidate = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'bias': 'bear',
                        'entry_type': 'orb_breakout',
                        'tp': tp,
                        'sl': sl,
                        'atr': current_atr,
                        'target': outcome,
                        'pnl_r': pnl_r,
                        'orb_high': orb_high,
                        'orb_low': orb_low,
                        'momentum': momentum,
                        'rsi': row.get('rsi', 50),
                        'adx': row.get('adx', 20),
                        'hour': current_time.hour,
                        'dayofweek': current_time.dayofweek,
                        'session_label': session_label,  # For WFA compatibility
                    }
                    candidates.append(candidate)
                    daily_trades += 1
                    sell_breakout_triggered = True
                    continue
        
        logger.info(f"Generated {len(candidates)} ORB candidates")
        
        # Debug: If no candidates, log why
        if len(candidates) == 0:
            logger.warning("No ORB candidates generated. Possible reasons:")
            logger.warning("  - No breakouts detected")
            logger.warning("  - All breakouts filtered out by ATR expansion or momentum")
            logger.warning("  - Opening range window too strict")
            logger.warning("  - Not enough data after opening range")
        
        return pd.DataFrame(candidates)
    
    def _check_outcome(self, future_window, entry_price, tp, sl, bias):
        """Check if TP or SL was hit first"""
        tp_hit_first = False
        sl_hit_first = False
        exit_time = None

        for _, bar in future_window.iterrows():
            if bias == 'bull':
                if bar['low'] <= sl:
                    sl_hit_first = True
                    exit_time = bar.name
                    break
                if bar['high'] >= tp:
                    tp_hit_first = True
                    exit_time = bar.name
                    break
            else:  # bear
                if bar['high'] >= sl:
                    sl_hit_first = True
                    exit_time = bar.name
                    break
                if bar['low'] <= tp:
                    tp_hit_first = True
                    exit_time = bar.name
                    break

        if tp_hit_first and not sl_hit_first:
            return 1, exit_time, self.tp_mult / self.sl_atr_mult
        elif sl_hit_first:
            return 0, exit_time, -1.0
        else:
            return 0, None, 0.0

    def _calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price) for each day"""
        df = df.copy()
        df['date'] = df.index.date
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']

        # Calculate cumulative sums within each day
        df['cumsum_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
        df['cumsum_volume'] = df.groupby('date')['volume'].cumsum()

        # VWAP = cumulative(typical_price * volume) / cumulative(volume)
        df['vwap'] = df['cumsum_tp_volume'] / df['cumsum_volume']

        # Fill NaN with typical price (for first bar of day or zero volume)
        df['vwap'].fillna(df['typical_price'], inplace=True)

        # Clean up temporary columns
        df.drop(['typical_price', 'tp_volume', 'cumsum_tp_volume', 'cumsum_volume'], axis=1, inplace=True)

        return df

    def _calculate_rvol(self, df):
        """Calculate Relative Volume (RVOL) - current volume vs 20-day average at same time"""
        df = df.copy()
        df['time'] = df.index.time

        # Calculate 20-day rolling average volume for each time of day
        df['volume_20d_avg'] = df.groupby('time')['volume'].transform(lambda x: x.rolling(window=20, min_periods=5).mean())

        # RVOL = current volume / 20-day average volume
        df['rvol'] = df['volume'] / df['volume_20d_avg']

        # Fill NaN with 1.0 (neutral RVOL)
        df['rvol'].fillna(1.0, inplace=True)

        # Clean up temporary columns
        df.drop(['volume_20d_avg'], axis=1, inplace=True)

        return df

    def _calculate_atr_avg(self, df):
        """Calculate 20-day rolling average of ATR for volatility screening"""
        df = df.copy()

        if 'atr' in df.columns:
            # Calculate 20-day average (20 days * ~78 bars per day on 5m = ~1560 bars)
            # Use simpler approach: 400 bars (~5 days on 5m data)
            df['atr_avg_20d'] = df['atr'].rolling(window=400, min_periods=100).mean()
            df['atr_avg_20d'].fillna(df['atr'], inplace=True)
        else:
            df['atr_avg_20d'] = 0

        return df

