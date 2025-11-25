import copy
import pandas as pd
import numpy as np
import itertools
from .utils import get_logger
from .data_adapter import DataAdapter
from .structure import StructureAnalyzer
from .features import FeatureEngineer
from .entries import EntryGenerator
from .backtester import Backtester
from .ml_model import MLModel

logger = get_logger()

class VariableOptimizer:
    def __init__(self, config, max_drawdown_limit=None):
        self.config = config
        self.max_drawdown_limit = max_drawdown_limit if max_drawdown_limit is not None else config['backtest']['propfirm']['max_drawdown_pct']
        
    def optimize(self, start_date, end_date):
        """
        Test different variable combinations and return top 5 best configurations.
        """
        logger.info("=" * 70)
        logger.info("VARIABLE OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Testing period: {start_date} to {end_date}")
        logger.info(f"Max drawdown limit: {self.max_drawdown_limit*100:.1f}%")
        logger.info(f"Initial capital: ${self.config['backtest']['propfirm']['initial_capital']:.2f}")
        
        # Load data once
        adapter = DataAdapter(self.config)
        base_tf_str = self.config['data']['timeframe_base']
        df_base = adapter.load_data(timeframe_suffix=base_tf_str)
        df_base = df_base[df_base.index >= pd.Timestamp(start_date)]
        df_base = df_base[df_base.index <= pd.Timestamp(end_date)]
        df_base_raw = df_base.copy()
        self.df_base = df_base  # Store for re-checking TP/SL outcomes
        
        # Load H4 + Daily data directly
        df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
        df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)
        
        # Generate candidates - will regenerate if structure params change
        # For now, generate once with default config (structure params tested separately)
        features = FeatureEngineer(self.config)
        structure = StructureAnalyzer(df_h4, self.config, daily_df=df_d1)
        df_base = features.calculate_technical_features(df_base)
        fvgs = features.detect_fvgs(df_base)
        obs = features.detect_obs(df_base)
        
        entry_gen = EntryGenerator(self.config, structure, features)
        candidates_base = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)
        
        logger.info(f"Generated {len(candidates_base)} candidates with default config")
        min_candidates = self.config.get('optimizer', {}).get('min_candidates', 50)
        if len(candidates_base) < min_candidates:
            logger.warning("⚠️  Candidate count is extremely low for optimisation. Relaxing filters (killzone, daily trend, sweeps, displacement, volume) to continue.")
            relaxed = self._regenerate_with_relaxed_filters(df_base_raw, df_h4, df_d1)
            if relaxed is None:
                logger.error("No candidates were generated even after relaxing filters. Optimisation aborted — broaden dates or relax filters manually.")
                return None
            features, structure, df_base, fvgs, obs, entry_gen, candidates_base = relaxed
            logger.info(f"Relaxed filters produced {len(candidates_base)} candidates for optimisation.")
        
        # Load model and get predictions for all candidates
        model = MLModel(self.config)
        try:
            model.load(self.config['ml']['model_path'])
            probs_base = model.predict_proba(candidates_base)
        except FileNotFoundError:
            logger.warning("Model not found. Training on first 50% of data...")
            split = int(len(candidates_base) * 0.5)
            train_df = candidates_base.iloc[:split]
            model.train(train_df)
            probs_base = model.predict_proba(candidates_base)
        
        # Define variable ranges to test - TARGET: >$500 profit, >80% win rate, >50 trades
        # Need to balance: high threshold (for win rate), lower threshold (for trade count), high risk (for profit)
        variables = {
            'tp_atr_mult': [2.0, 2.5, 3.0, 3.5, 4.0],  # Moderate TP
            'sl_atr_mult': [1.0, 1.2, 1.5, 2.0],  # Moderate SL
            'model_threshold': [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70],  # Moderate-high thresholds
            'entry_type_filter': ['ob_only'],  # Focus on OB only
            'cooldown_minutes': [15, 30, 45, 60],  # Shorter cooldowns = more trades
            'per_trade_risk_pct': [0.010, 0.015, 0.020, 0.025, 0.030],  # Higher risk = more profit
        }
        
        # Generate test combinations - TARGET: >$500 profit, >80% win rate, >50 trades
        test_configs = []
        
        filter_sets = [
            {
                'filter_profile': 'strict_institutional',
                'require_killzone_filter': True,
                'use_daily_trend_filter': True,
                'require_liquidity_sweep': True,
                'require_displacement': True,
                'require_inducement': True,
                'entry_type_filter': 'ob_only'
            },
            {
                'filter_profile': 'session_focus',
                'require_killzone_filter': True,
                'use_daily_trend_filter': True,
                'require_liquidity_sweep': True,
                'require_displacement': False,
                'require_inducement': False,
                'entry_type_filter': 'ob_only'
            },
            {
                'filter_profile': 'liquidity_only',
                'require_killzone_filter': False,
                'use_daily_trend_filter': False,
                'require_liquidity_sweep': True,
                'require_displacement': True,
                'require_inducement': False,
                'entry_type_filter': 'both'
            },
            {
                'filter_profile': 'free_fire',
                'require_killzone_filter': False,
                'use_daily_trend_filter': False,
                'require_liquidity_sweep': False,
                'require_displacement': False,
                'require_inducement': False,
                'entry_type_filter': 'both'
            }
        ]
        
        def append_with_filters(base_cfg):
            for fset in filter_sets:
                cfg = base_cfg.copy()
                profile = fset.get('filter_profile', 'baseline')
                for k, v in fset.items():
                    if k == 'filter_profile':
                        continue
                    cfg[k] = v
                cfg['filter_profile'] = profile
                test_configs.append(cfg)
        
        # Strategy 1: VERY LOW threshold (0.35-0.42) for MAXIMUM TRADES, Very high risk (4-7%)
        # This is aggressive but needed to get >50 trades
        for tp in [3.0, 3.5, 4.0]:
            for sl in [1.0, 1.2]:
                for thresh in [0.35, 0.38, 0.40, 0.42]:
                    for risk in [0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070]:
                        base_cfg = {
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'cooldown_minutes': 15,
                            'per_trade_risk_pct': risk,
                        }
                        append_with_filters(base_cfg)
        
        # Strategy 2: Low threshold (0.38-0.45) for MORE TRADES, High risk (3.5-5.5%)
        for tp in [2.5, 3.0, 3.5]:
            for sl in [1.0, 1.2]:
                for thresh in [0.38, 0.40, 0.42, 0.45]:
                    for risk in [0.035, 0.040, 0.045, 0.050, 0.055]:
                        base_cfg = {
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'cooldown_minutes': 15,
                            'per_trade_risk_pct': risk,
                        }
                        append_with_filters(base_cfg)
        
        # Strategy 3: Minimum threshold (0.32-0.40) for ABSOLUTE MAX trades, Maximum risk (5-8%)
        for tp in [3.5, 4.0, 4.5]:
            for sl in [1.0]:
                for thresh in [0.32, 0.35, 0.38, 0.40]:
                    for risk in [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080]:
                        base_cfg = {
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'cooldown_minutes': 15,
                            'per_trade_risk_pct': risk,
                        }
                        append_with_filters(base_cfg)
        
        # Strategy 4: Balanced low threshold (0.40-0.48), High risk (4-6%)
        for tp in [3.0, 3.5, 4.0]:
            for sl in [1.0, 1.2, 1.5]:
                for thresh in [0.40, 0.42, 0.45, 0.48]:
                    for risk in [0.040, 0.045, 0.050, 0.055, 0.060]:
                        for cooldown in [15, 30]:
                            base_cfg = {
                                'tp_atr_mult': tp,
                                'sl_atr_mult': sl,
                                'model_threshold': thresh,
                                'cooldown_minutes': cooldown,
                                'per_trade_risk_pct': risk,
                            }
                            append_with_filters(base_cfg)
        
        # Strategy 5: High TP (4.0-6.0), Tight SL (1.0), Very low threshold (0.35-0.42), Very high risk
        for tp in [4.0, 4.5, 5.0, 5.5, 6.0]:
            for sl in [1.0]:
                for thresh in [0.35, 0.38, 0.40, 0.42]:
                    for risk in [0.045, 0.050, 0.055, 0.060]:
                        base_cfg = {
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'cooldown_minutes': 15,
                            'per_trade_risk_pct': risk,
                        }
                        append_with_filters(base_cfg)
        
        # Remove duplicates
        seen = set()
        unique_configs = []
        for cfg in test_configs:
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)
        
        logger.info(f"Testing {len(unique_configs)} variable combinations...")
        
        results = []
        
        for i, var_config in enumerate(unique_configs):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(unique_configs)} configurations tested")
            
            try:
                # Temporarily override config
                original_values = {}
                for key, value in var_config.items():
                    if key == 'filter_profile':
                        continue
                    if key == 'per_trade_risk_pct':
                        original_values[key] = self.config['backtest']['propfirm'][key]
                        self.config['backtest']['propfirm'][key] = value
                    elif key == 'cooldown_minutes':
                        original_values[key] = self.config['backtest'].get(key, 60)
                        self.config['backtest'][key] = value
                    else:
                        original_values[key] = self.config['strategy'][key]
                        self.config['strategy'][key] = value
                
                # Use base candidates and probabilities
                candidates = candidates_base.copy()
                probs = probs_base.copy()
                
                # Apply entry type filter BEFORE backtesting
                entry_type_filter = var_config['entry_type_filter']
                if entry_type_filter == 'ob_only':
                    mask = candidates['entry_type'] == 'ob'
                    candidates = candidates[mask].copy()
                    probs = probs[mask]
                elif entry_type_filter == 'fvg_only':
                    mask = candidates['entry_type'] == 'fvg'
                    candidates = candidates[mask].copy()
                    probs = probs[mask]
                
                # Apply threshold filter BEFORE backtesting
                threshold = var_config['model_threshold']
                mask = probs >= threshold
                candidates = candidates[mask].copy()
                probs = probs[mask]
                
                if len(candidates) == 0:
                    continue  # No candidates pass filters
                
                # ALWAYS recalculate TP/SL and outcomes for each configuration
                # This ensures different TP/SL multipliers produce different results
                logger.debug(f"Recalculating TP/SL for {len(candidates)} candidates with TP={var_config['tp_atr_mult']}, SL={var_config['sl_atr_mult']}")
                
                for idx in candidates.index:
                    entry_price = candidates.loc[idx, 'entry_price']
                    atr = candidates.loc[idx, 'atr']
                    bias = candidates.loc[idx, 'bias']
                    entry_time = candidates.loc[idx, 'entry_time']
                    
                    # Calculate new TP/SL based on current multipliers
                    if bias == 'bull':
                        new_tp = entry_price + (var_config['tp_atr_mult'] * atr)
                        new_sl = entry_price - (var_config['sl_atr_mult'] * atr)
                    else:  # bear
                        new_tp = entry_price - (var_config['tp_atr_mult'] * atr)
                        new_sl = entry_price + (var_config['sl_atr_mult'] * atr)
                    
                    candidates.loc[idx, 'tp'] = new_tp
                    candidates.loc[idx, 'sl'] = new_sl
                    
                    # Re-check outcome: find entry_time in self.df_base and check future bars
                    entry_idx = self.df_base.index.get_indexer([entry_time], method='nearest')[0]
                    if entry_idx >= 0 and entry_idx < len(self.df_base) - 1:
                        future_window = self.df_base.iloc[entry_idx+1 : min(entry_idx+1+200, len(self.df_base))]
                        
                        tp_hit_first = False
                        sl_hit_first = False
                        
                        for _, bar in future_window.iterrows():
                            if bias == 'bull':
                                if bar['low'] <= new_sl and not tp_hit_first:
                                    sl_hit_first = True
                                    break
                                if bar['high'] >= new_tp and not sl_hit_first:
                                    tp_hit_first = True
                                    break
                            else:  # bear
                                if bar['high'] >= new_sl and not tp_hit_first:
                                    sl_hit_first = True
                                    break
                                if bar['low'] <= new_tp and not sl_hit_first:
                                    tp_hit_first = True
                                    break
                        
                        # Update target and pnl_r based on new TP/SL outcome
                        if tp_hit_first:
                            candidates.loc[idx, 'target'] = 1
                            candidates.loc[idx, 'pnl_r'] = var_config['tp_atr_mult'] / var_config['sl_atr_mult']
                        elif sl_hit_first:
                            candidates.loc[idx, 'target'] = 0
                            candidates.loc[idx, 'pnl_r'] = -1.0
                        else:
                            # Neither hit within window - mark as incomplete/loss
                            candidates.loc[idx, 'target'] = 0
                            candidates.loc[idx, 'pnl_r'] = 0.0
                    else:
                        # Entry time not found - keep original values
                        pass
                
                # Add probabilities to candidates dataframe for backtester
                candidates['prob'] = probs
                
                # Run backtest with these variables
                backtester = Backtester(self.config)
                history, trades = backtester.run(candidates, probs)
                
                if len(trades) == 0:
                    continue
                
                # Calculate metrics
                wins = trades[trades['net_pnl'] > 0]
                losses = trades[trades['net_pnl'] <= 0]
                win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
                net_profit = trades['net_pnl'].sum()
                gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
                gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                history['equity'] = pd.to_numeric(history['equity'], errors='coerce')
                
                # Calculate max drawdown properly (peak-to-trough)
                initial_capital = self.config['backtest']['propfirm']['initial_capital']
                equity_series = history['equity'].values
                running_max = np.maximum.accumulate(equity_series)
                drawdowns = ((running_max - equity_series) / running_max) * 100
                max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
                
                # Filter out configurations that exceed max drawdown limit
                if max_dd > (self.max_drawdown_limit * 100):
                    continue
                
                # Score: Must meet ALL criteria: >$500 profit, >80% win rate, >50 trades
                # Only score configurations that meet all requirements
                if win_rate >= 80 and net_profit > 500 and len(trades) > 50:
                    # All criteria met - high score
                    score = net_profit * (win_rate / 100) * profit_factor * len(trades) / 50 if profit_factor != float('inf') else net_profit * win_rate * len(trades)
                elif win_rate >= 80 and net_profit > 500:
                    # Missing trade count - partial score
                    score = net_profit * (win_rate / 100) * profit_factor * (len(trades) / 50) if profit_factor != float('inf') else net_profit * win_rate
                elif win_rate >= 80 and len(trades) > 50:
                    # Missing profit - partial score
                    score = net_profit * (win_rate / 100) * profit_factor if profit_factor != float('inf') else win_rate * len(trades)
                else:
                    # Doesn't meet criteria - low score
                    score = net_profit * (win_rate / 100) * profit_factor if profit_factor != float('inf') else 0
                
                result = var_config.copy()
                result.update({
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'net_profit': net_profit,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_dd,
                    'score': score,
                    'history': history,  # Store history for plotting
                    'trades_df': trades,  # Store trades for reference
                })
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Config {i+1} failed: {e}")
            finally:
                # Restore original values
                for key, value in original_values.items():
                    if key == 'per_trade_risk_pct':
                        self.config['backtest']['propfirm'][key] = value
                    elif key == 'cooldown_minutes':
                        self.config['backtest'][key] = value
                    else:
                        self.config['strategy'][key] = value
        
        if not results:
            logger.error("No successful tests!")
            return None
        
        # Sort by score (best first)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        # Log configurations that meet criteria
        meets_all = results_df[(results_df['win_rate'] >= 80) & (results_df['net_profit'] > 500) & (results_df['trades'] > 50)]
        if len(meets_all) > 0:
            logger.info(f"\n🎯 FOUND {len(meets_all)} CONFIGURATIONS MEETING ALL CRITERIA!")
            for idx, row in meets_all.head(10).iterrows():
                logger.info(f"  TP={row['tp_atr_mult']}, SL={row['sl_atr_mult']}, Thresh={row['model_threshold']}, Risk={row['per_trade_risk_pct']*100:.1f}%")
                logger.info(f"    Profile: {row.get('filter_profile', 'baseline')}")
                logger.info(f"    → Trades: {int(row['trades'])}, Win Rate: {row['win_rate']:.1f}%, Profit: ${row['net_profit']:.2f}")
        else:
            # Log configurations that meet 2 out of 3 criteria
            meets_2 = results_df[
                ((results_df['win_rate'] >= 80) & (results_df['net_profit'] > 500)) |
                ((results_df['win_rate'] >= 80) & (results_df['trades'] > 50)) |
                ((results_df['net_profit'] > 500) & (results_df['trades'] > 50))
            ]
            if len(meets_2) > 0:
                logger.info(f"\n⚠️  Found {len(meets_2)} configurations meeting 2/3 criteria (best candidates):")
                for idx, row in meets_2.head(5).iterrows():
                    logger.info(f"  TP={row['tp_atr_mult']}, SL={row['sl_atr_mult']}, Thresh={row['model_threshold']}, Risk={row['per_trade_risk_pct']*100:.1f}%")
                    logger.info(f"    Profile: {row.get('filter_profile', 'baseline')}")
                    logger.info(f"    → Trades: {int(row['trades'])}, Win Rate: {row['win_rate']:.1f}%, Profit: ${row['net_profit']:.2f}")
        
        # Return top 5 (or all that meet criteria if found)
        if len(meets_all) > 0:
            top_5 = meets_all.head(5)
        else:
            top_5 = results_df.head(5)
        
        logger.info("\n" + "=" * 70)
        logger.info("TOP 5 OPTIMIZED CONFIGURATIONS")
        logger.info("=" * 70)
        
        # Save results and generate equity graphs
        import datetime
        import matplotlib.pyplot as plt
        import os
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join("reports", f"optimization_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            logger.info(f"\n#{idx} Configuration:")
            logger.info(f"  TP Multiplier: {row['tp_atr_mult']}")
            logger.info(f"  SL Multiplier: {row['sl_atr_mult']}")
            logger.info(f"  Model Threshold: {row['model_threshold']}")
            logger.info(f"  Entry Type Filter: {row['entry_type_filter']}")
            logger.info(f"  Cooldown Minutes: {row['cooldown_minutes']}")
            logger.info(f"  Risk Per Trade: {row['per_trade_risk_pct']*100:.2f}%")
            logger.info(f"  Filter Profile: {row.get('filter_profile', 'baseline')}")
            logger.info(f"\n  Results:")
            logger.info(f"    Trades: {int(row['trades'])}")
            logger.info(f"    Win Rate: {row['win_rate']:.2f}%")
            logger.info(f"    Net Profit: ${row['net_profit']:.2f}")
            logger.info(f"    Profit Factor: {row['profit_factor']:.2f}")
            logger.info(f"    Max Drawdown: {row['max_drawdown']:.2f}%")
            logger.info(f"    Score: {row['score']:.2f}")
            
            # Generate equity graph for this configuration
            if 'history' in row and row['history'] is not None and not row['history'].empty:
                history = row['history']
                plt.figure(figsize=(12, 6))
                plt.plot(history['time'], history['equity'], label='Equity', linewidth=2, color='#2E86AB')
                
                # Add drawdown limit line
                initial_capital = self.config['backtest']['propfirm']['initial_capital']
                dd_limit = initial_capital * (1 - self.max_drawdown_limit)
                plt.axhline(y=dd_limit, color='r', linestyle='--', linewidth=1.5, 
                           label=f'Max DD Limit ({self.max_drawdown_limit*100:.1f}%)', alpha=0.7)
                
                # Add initial capital line
                plt.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Capital')
                
                plt.title(f"Configuration #{idx}: TP={row['tp_atr_mult']}, SL={row['sl_atr_mult']}, Thresh={row['model_threshold']:.2f}, Entry={row['entry_type_filter']}\n"
                         f"Trades: {int(row['trades'])}, Win Rate: {row['win_rate']:.1f}%, Profit: ${row['net_profit']:.2f}, PF: {row['profit_factor']:.2f}",
                         fontsize=11, pad=10)
                plt.xlabel("Date", fontsize=10)
                plt.ylabel("Equity ($)", fontsize=10)
                plt.legend(loc='best', fontsize=9)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                graph_path = os.path.join(report_dir, f"config_{idx}_equity.png")
                plt.savefig(graph_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"    Equity graph saved: config_{idx}_equity.png")
        
        # Save CSV (without history/trades_df columns)
        top_5_clean = top_5.drop(columns=['history', 'trades_df'], errors='ignore')
        csv_path = os.path.join(report_dir, "top_5_configurations.csv")
        top_5_clean.to_csv(csv_path, index=False)
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Results saved to: {report_dir}")
        logger.info(f"  - CSV: top_5_configurations.csv")
        logger.info(f"  - Equity graphs: config_1_equity.png through config_5_equity.png")
        logger.info(f"{'=' * 70}")
        
        return top_5_clean

    def _regenerate_with_relaxed_filters(self, df_base_raw, df_h4, df_d1):
        relaxed_config = copy.deepcopy(self.config)
        strat = relaxed_config['strategy']
        strat['require_killzone_filter'] = False
        strat['use_daily_trend_filter'] = False
        strat['require_liquidity_sweep'] = False
        strat['require_displacement'] = False
        strat['require_inducement'] = False
        strat['min_volume_zscore'] = min(strat.get('min_volume_zscore', -0.5), -2.0)
        strat['low_vol_ratio'] = min(strat.get('low_vol_ratio', 0.85), 0.6)
        
        self.config = relaxed_config
        features = FeatureEngineer(self.config)
        df_base_relaxed = features.calculate_technical_features(df_base_raw.copy())
        fvgs = features.detect_fvgs(df_base_relaxed)
        obs = features.detect_obs(df_base_relaxed)
        structure = StructureAnalyzer(df_h4, self.config, daily_df=df_d1)
        entry_gen = EntryGenerator(self.config, structure, features)
        candidates = entry_gen.generate_candidates(df_base_relaxed, df_h4, fvgs, obs)
        if candidates.empty:
            return None
        self.df_base = df_base_relaxed
        return features, structure, df_base_relaxed, fvgs, obs, entry_gen, candidates



