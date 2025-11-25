"""
Opening Range Breakout (ORB) Strategy - Similar to New_XAU_Bot.mq5
Replaces SMC strategy with ORB breakout logic.
"""
import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class ORBEntryGenerator:
    def __init__(self, config):
        self.config = config
        self.orb_bars = config['strategy'].get('orb_bars', 4)
        self.orb_start_hour = config['strategy'].get('orb_start_hour', 13)
        self.orb_start_minute = config['strategy'].get('orb_start_minute', 30)
        self.orb_end_hour = config['strategy'].get('orb_end_hour', 14)
        self.orb_end_minute = config['strategy'].get('orb_end_minute', 30)
        self.only_trade_after_14utc = config['strategy'].get('only_trade_after_14utc', True)
        self.require_atr_expansion = config['strategy'].get('require_atr_expansion', True)
        self.momentum_period = config['strategy'].get('momentum_period', 5)
        self.min_momentum = config['strategy'].get('min_momentum', 0.0)
        self.max_trades_per_day = config['strategy'].get('max_trades_per_day', 5)
        self.tp_mult = config['strategy']['tp_atr_mult']
        self.sl_atr_mult = config['strategy']['sl_atr_mult']
        self.label_window = config['strategy'].get('label_window', 200)
        
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
                
                # Stop trading after 16:00 UTC (end of NY session)
                if current_time.hour >= 16:
                    break
                
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
                    }
                    candidates.append(candidate)
                    daily_trades += 1
                    buy_breakout_triggered = True
                    continue
                
                # Check for SELL breakout (price breaks below ORB low)
                elif bar_low < orb_low and not sell_breakout_triggered:
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

