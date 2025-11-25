import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class EntryGenerator:
    def __init__(self, config, structure_analyzer, feature_engineer):
        self.config = config
        self.structure = structure_analyzer
        self.features = feature_engineer
        self.tp_mult = config['strategy']['tp_atr_mult']
        self.sl_atr_mult = config['strategy']['sl_atr_mult']
        self.label_window = config['strategy'].get('label_window', 50)
        strategy_cfg = config['strategy']
        self.require_killzone = strategy_cfg.get('require_killzone_filter', False)
        sessions = strategy_cfg.get('session_filter', [])
        self.session_filter = {s.lower().replace(" ", "_") for s in sessions}
        self.require_liquidity_sweep = strategy_cfg.get('require_liquidity_sweep', False)
        self.sweep_lookback = strategy_cfg.get('liquidity_sweep_lookback', 36)
        self.sweep_tolerance = strategy_cfg.get('liquidity_sweep_tolerance', 0.001)
        self.require_displacement = strategy_cfg.get('require_displacement', False)
        self.displacement_min = strategy_cfg.get('displacement_min_atr', 1.0)
        self.use_daily_filter = strategy_cfg.get('use_daily_trend_filter', False)
        self.allow_daily_range = strategy_cfg.get('allow_daily_range_trades', False)
        self.daily_range_threshold = strategy_cfg.get('daily_trend_range_threshold', 0.1)
        self.low_vol_ratio = strategy_cfg.get('low_vol_ratio', 0.9)
        self.min_volume_zscore = strategy_cfg.get('min_volume_zscore', -1.0)
        
    def generate_candidates(self, df_base, df_h4, fvgs, obs):
        """
        Scan Base Timeframe data, check H4 structure, and generate candidate entries.
        df_base: Base Timeframe DataFrame with technicals (e.g., 5m)
        df_h4: H4 DataFrame
        fvgs: List of detected FVGs on Base TF
        obs: List of detected OBs on Base TF
        """
        candidates = []
        
        logger.info("Generating candidates...")
        
        # Iterate Base TF bars
        start_idx = 100 
        
        for i in range(start_idx, len(df_base) - self.label_window):
            # Progress Log every 5000 bars
            if i % 5000 == 0:
                 pct = (i / len(df_base)) * 100
                 logger.info(f"Scanning... {pct:.1f}% ({i}/{len(df_base)})")
            
            curr_time = df_base.index[i]
            
            # Map to H4 index for inducement checks
            h4_pos = df_h4.index.get_indexer([curr_time], method='pad')
            if h4_pos.size == 0:
                continue
            h4_idx = h4_pos[0]
            if h4_idx < 0:
                continue
            
            # Skip rows without ATR/technical context
            if pd.isna(df_base['atr'].iloc[i]) or pd.isna(df_base['session_code'].iloc[i]):
                continue

            # Get enriched structure context
            bias, struct_info, daily_trend, daily_info = self.structure.get_enriched_context(curr_time)
            
            if bias == 'neutral':
                continue
            if self.use_daily_filter:
                allow_daily = True
                if daily_trend in ('neutral', 'range'):
                    if not self.allow_daily_range:
                        allow_daily = False
                    else:
                        ema_fast = daily_info.get('ema_fast')
                        ema_slow = daily_info.get('ema_slow')
                        if ema_fast is not None and ema_slow is not None:
                            diff = abs(ema_fast - ema_slow)
                            denom = abs(ema_slow) if abs(ema_slow) > 1e-6 else 1.0
                            if (diff / denom) > self.daily_range_threshold:
                                allow_daily = False
                else:
                    if bias == 'bull' and daily_trend != 'bull':
                        allow_daily = False
                    if bias == 'bear' and daily_trend != 'bear':
                        allow_daily = False
                if not allow_daily:
                    continue
                
            # Zone check (can be disabled via config)
            current_price = df_base['close'].iloc[i]
            
            require_zone = self.config['strategy'].get('require_discount_premium', True)
            if require_zone:
                valid_zone = False
                if bias == 'bull' and struct_info.get('in_discount', False):
                    valid_zone = True
                elif bias == 'bear' and struct_info.get('in_premium', False):
                    valid_zone = True
                    
                if not valid_zone:
                    continue
                
            # Session filter
            session_label = df_base['session_label'].iloc[i] if 'session_label' in df_base.columns else 'off'
            if self.require_killzone and self.session_filter:
                if session_label not in self.session_filter:
                    continue
            
            # Inducement Check (can be disabled via config)
            require_inducement = self.config['strategy'].get('require_inducement', True)
            if require_inducement:
                has_inducement = self.structure.check_inducement(h4_idx)
                if not has_inducement:
                    continue
                
            # Liquidity sweep filter
            sweep_flag, sweep_dir = self._detect_liquidity_sweep(df_base, i, bias)
            if self.require_liquidity_sweep and not sweep_flag:
                continue
            
            # Displacement filter
            displacement_factor = df_base['body_factor'].iloc[i]
            if self.require_displacement and displacement_factor < self.displacement_min:
                continue
            
            # Additional volume/volatility filters
            if df_base['atr_ratio'].iloc[i] < 0.2:  # ensure ATR valid
                continue
            if df_base['volume_zscore'].iloc[i] < self.min_volume_zscore:
                continue
            
            # Check for Entry Triggers (FVG/OB interaction)
            triggered = False
            entry_type = None
            
            relevant_fvgs = [f for f in fvgs if f['created_at'] < curr_time and f['type'] == bias]
            
            for fvg in relevant_fvgs:
                if bias == 'bull':
                    if current_price <= fvg['top'] and current_price >= fvg['bottom']:
                        triggered = True
                        entry_type = 'fvg'
                        break
                else: # bear
                    if current_price >= fvg['bottom'] and current_price <= fvg['top']:
                        triggered = True
                        entry_type = 'fvg'
                        break
            
            if not triggered:
                relevant_obs = [o for o in obs if o['created_at'] < curr_time and o['type'] == bias]
                for ob in relevant_obs:
                    if bias == 'bull':
                         if current_price <= ob['top'] and current_price >= ob['bottom']:
                            triggered = True
                            entry_type = 'ob'
                            break
                    else:
                        if current_price >= ob['bottom'] and current_price <= ob['top']:
                            triggered = True
                            entry_type = 'ob'
                            break
                            
            if triggered:
                # Additional technical filters to improve win rate
                rsi = df_base['rsi'].iloc[i]
                adx = df_base['adx'].iloc[i]
                
                # RSI filter: Based on analysis, higher RSI (60-70) performs better
                require_rsi_filter = self.config['strategy'].get('require_rsi_filter', False)
                if require_rsi_filter:
                    if bias == 'bull':
                        # Analysis shows RSI 60-70 has 33% win rate vs 21-24% for lower
                        min_rsi = self.config['strategy'].get('min_rsi_bull', 50)
                        max_rsi = self.config['strategy'].get('max_rsi_bull', 70)
                        if rsi < min_rsi or rsi > max_rsi:
                            continue
                    if bias == 'bear':
                        # Bear trades have very low win rate (16%), consider filtering them out
                        min_rsi = self.config['strategy'].get('min_rsi_bear', 30)
                        max_rsi = self.config['strategy'].get('max_rsi_bear', 50)
                        if rsi < min_rsi or rsi > max_rsi:
                            continue
                
                # ADX filter: Analysis shows ADX >40 has 32.5% win rate
                require_adx_filter = self.config['strategy'].get('require_adx_filter', False)
                min_adx = self.config['strategy'].get('min_adx', 25)  # Increased from 20
                if require_adx_filter and adx < min_adx:
                    continue
                
                # Bias filter: Analysis shows bull has 27.9% vs bear 16%
                require_bull_only = self.config['strategy'].get('require_bull_only', False)
                if require_bull_only and bias != 'bull':
                    continue
                
                atr = df_base['atr'].iloc[i]
                entry_price = df_base['close'].iloc[i]
                
                if bias == 'bull':
                    tp = entry_price + (self.tp_mult * atr)
                    sl = entry_price - (self.sl_atr_mult * atr)
                else:
                    tp = entry_price - (self.tp_mult * atr)
                    sl = entry_price + (self.sl_atr_mult * atr)
                
                outcome = 0
                exit_time = None
                pnl = 0
                
                future_window = df_base.iloc[i+1 : i+1+self.label_window]
                
                # Track if we hit TP or SL
                sl_hit_first = False
                tp_hit_first = False
                
                for j in range(len(future_window)):
                    bar = future_window.iloc[j]
                    
                    sl_hit = False
                    tp_hit = False
                    
                    if bias == 'bull':
                        if bar['low'] <= sl: sl_hit = True
                        if bar['high'] >= tp: tp_hit = True
                    else:
                        if bar['high'] >= sl: sl_hit = True
                        if bar['low'] <= tp: tp_hit = True
                    
                    # Check which hits first (SL takes priority for risk management)
                    if sl_hit and not tp_hit_first:
                        outcome = 0
                        exit_time = bar.name
                        pnl = -1.0
                        sl_hit_first = True
                        break
                    
                    if tp_hit and not sl_hit_first:
                        outcome = 1
                        exit_time = bar.name
                        pnl = self.tp_mult / self.sl_atr_mult
                        tp_hit_first = True
                        break
                
                # If neither TP nor SL hit within label window, label as "no outcome" (0 with pnl=0)
                # This is better than labeling as loss, but we'll still exclude from training
                # by filtering out pnl_r == 0 trades, or we can keep them as "incomplete" signals
                if exit_time is None:
                    outcome = 0
                    pnl = 0  # Mark as incomplete/no outcome
                
                range_low = struct_info.get('range_low', current_price)
                range_high = struct_info.get('range_high', current_price)
                safe_low = range_low if abs(range_low) > 1e-6 else current_price
                safe_high = range_high if abs(range_high) > 1e-6 else current_price
                
                candidate = {
                    'entry_time': curr_time,
                    'entry_price': entry_price,
                    'bias': bias,
                    'entry_type': entry_type,
                    'tp': tp,
                    'sl': sl,
                    'atr': atr,
                    'target': outcome,
                    'pnl_r': pnl,
                    'rsi': df_base['rsi'].iloc[i],
                    'adx': df_base['adx'].iloc[i],
                    'vol_spike': df_base['vol_spike'].iloc[i],
                    'rolling_vol': df_base['rolling_vol'].iloc[i],
                    'hour': curr_time.hour,
                    'dayofweek': curr_time.dayofweek,
                    'dist_to_range_low': (current_price - safe_low) / max(abs(safe_low), 1e-6),
                    'dist_to_range_high': (safe_high - current_price) / max(abs(safe_high), 1e-6),
                    'session_label': session_label,
                    'session_code': df_base['session_code'].iloc[i],
                    'in_killzone': bool(df_base['in_killzone'].iloc[i]) if 'in_killzone' in df_base.columns else False,
                    'daily_trend_label': daily_trend,
                    'daily_trend': self._encode_trend(daily_trend),
                    'displacement_factor': displacement_factor,
                    'liquidity_sweep': int(sweep_flag),
                    'liquidity_sweep_dir': sweep_dir,
                    'atr_ratio': df_base['atr_ratio'].iloc[i],
                    'volume_zscore': df_base['volume_zscore'].iloc[i],
                    'vol_regime': self._vol_regime(df_base['atr_ratio'].iloc[i]),
                    'vol_regime_code': 1 if df_base['atr_ratio'].iloc[i] < self.low_vol_ratio else 0,
                    'premium_pct': struct_info.get('premium_pct'),
                    'discount_pct': struct_info.get('discount_pct'),
                    'body_factor': df_base['body_factor'].iloc[i],
                    'impulse_factor': df_base['impulse_factor'].iloc[i],
                    'exit_time': exit_time,
                    'holding_minutes': (exit_time - curr_time).total_seconds() / 60 if exit_time else None
                }
                candidates.append(candidate)

                
        return pd.DataFrame(candidates)

    def _detect_liquidity_sweep(self, df, idx, bias):
        if idx < self.sweep_lookback:
            return False, 0
        window = df.iloc[idx - self.sweep_lookback:idx]
        current_low = df['low'].iloc[idx]
        current_high = df['high'].iloc[idx]
        close = df['close'].iloc[idx]
        if bias == 'bull':
            prior_low = window['low'].min()
            if prior_low <= 0:
                return False, 0
            swept = current_low < prior_low * (1 - self.sweep_tolerance) and close > prior_low
            return swept, 1 if swept else 0
        else:
            prior_high = window['high'].max()
            swept = current_high > prior_high * (1 + self.sweep_tolerance) and close < prior_high
            return swept, -1 if swept else 0

    def _vol_regime(self, atr_ratio):
        if atr_ratio < self.low_vol_ratio:
            return 'low'
        return 'normal'

    def _encode_trend(self, trend):
        if trend == 'bull':
            return 1
        if trend == 'bear':
            return -1
        return 0


