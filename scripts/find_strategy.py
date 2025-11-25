#!/usr/bin/env python
"""
Find trading strategy configuration that meets specific criteria.
Tests all variable combinations until finding one that matches requirements.

Usage:
    python scripts/find_strategy.py --win-rate 80 --profit-target 500 --min-trades 50 --from 2025-01-01 --to 2025-10-31 --initial-capital 1000
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import itertools
import signal
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging, get_logger
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

logger = get_logger()

# Removed global best_results - using local variable instead

def show_best_results(best_results, win_rate_target, profit_target, min_trades):
    """Display best results found so far."""
    if not best_results:
        logger.info("\nNo results to display.")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("BEST RESULTS FOUND (Top 10)")
    logger.info("=" * 80)
    
    # Sort by composite score (closeness to all targets)
    for idx, result in enumerate(best_results[:10], 1):
        score = (
            (result['win_rate'] / win_rate_target) * 0.4 +
            (result['net_profit'] / profit_target) * 0.4 +
            (result['trades'] / min_trades) * 0.2
        )
        logger.info(f"\n#{idx} Configuration:")
        logger.info(f"  TP Multiplier: {result['config']['tp_atr_mult']}")
        logger.info(f"  SL Multiplier: {result['config']['sl_atr_mult']}")
        logger.info(f"  Model Threshold: {result['config']['model_threshold']}")
        logger.info(f"  Entry Type Filter: {result['config']['entry_type_filter']}")
        logger.info(f"  Cooldown Minutes: {result['config']['cooldown_minutes']}")
        logger.info(f"  Risk Per Trade: {result['config']['per_trade_risk_pct']*100:.2f}%")
        logger.info(f"\n  Results:")
        logger.info(f"    Trades: {result['trades']} (target: {min_trades})")
        logger.info(f"    Win Rate: {result['win_rate']:.2f}% (target: {win_rate_target}%)")
        logger.info(f"    Net Profit: ${result['net_profit']:.2f} (target: ${profit_target})")
        logger.info(f"    Score: {score:.3f}")

def find_strategy(win_rate_target, profit_target, min_trades, start_date, end_date, initial_capital, max_drawdown=None):
    """
    Find strategy configuration meeting all criteria.
    
    Args:
        win_rate_target: Minimum win rate percentage (e.g., 80)
        profit_target: Minimum profit in dollars (e.g., 500)
        min_trades: Minimum number of trades (e.g., 50)
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        initial_capital: Starting capital
        max_drawdown: Maximum drawdown percentage (optional)
    """
    logger.info("=" * 80)
    logger.info("STRATEGY FINDER")
    logger.info("=" * 80)
    logger.info(f"Target Win Rate: {win_rate_target}%")
    logger.info(f"Target Profit: ${profit_target}")
    logger.info(f"Minimum Trades: {min_trades}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital}")
    if max_drawdown:
        logger.info(f"Max Drawdown Limit: {max_drawdown}%")
    logger.info("=" * 80)
    
    # Load config
    config = load_config('config.yml')
    config['backtest']['propfirm']['initial_capital'] = initial_capital
    if max_drawdown:
        config['backtest']['propfirm']['max_drawdown_pct'] = max_drawdown / 100.0
    
    # Load data
    logger.info("\n[1/6] Loading data...")
    adapter = DataAdapter(config)
    base_tf_str = config['data']['timeframe_base']
    df_base = adapter.load_data(timeframe_suffix=base_tf_str)
    df_base = df_base[df_base.index >= pd.Timestamp(start_date)]
    df_base = df_base[df_base.index <= pd.Timestamp(end_date)]
    
    if df_base.empty:
        logger.error(f"No data found for {start_date} to {end_date}")
        return None
    
    logger.info(f"✓ Loaded {len(df_base)} bars")
    
    # Load H4 data
    df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
    if df_h4 is None:
        h4_min = config['timeframes']['h4']
        df_h4 = adapter.resample_data(df_base, h4_min)
        logger.info(f"✓ Resampled to H4: {len(df_h4)} bars")
    else:
        logger.info(f"✓ Loaded H4 data: {len(df_h4)} bars")
    
    # Generate features and candidates
    logger.info("\n[2/6] Generating features and candidates...")
    features = FeatureEngineer(config)
    structure = StructureAnalyzer(df_h4, config)
    
    df_base = features.calculate_technical_features(df_base)
    fvgs = features.detect_fvgs(df_base)
    obs = features.detect_obs(df_base)
    logger.info(f"✓ Detected {len(fvgs)} FVGs and {len(obs)} OBs")
    
    entry_gen = EntryGenerator(config, structure, features)
    candidates_base = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)
    logger.info(f"✓ Generated {len(candidates_base)} candidates")
    
    if candidates_base.empty:
        logger.error("No candidates generated!")
        return None
    
    # Load ML model
    logger.info("\n[3/6] Loading ML model...")
    model = MLModel(config)
    try:
        model.load(config['ml']['model_path'])
        probs_base = model.predict_proba(candidates_base)
        logger.info(f"✓ Model loaded, predictions generated")
    except FileNotFoundError:
        logger.warning("Model not found. Training on first 50% of data...")
        split = int(len(candidates_base) * 0.5)
        train_df = candidates_base.iloc[:split]
        model.train(train_df)
        probs_base = model.predict_proba(candidates_base)
        logger.info(f"✓ Model trained, predictions generated")
    
            # Define variable ranges to test - OPTIMIZED for better results
    logger.info("\n[4/6] Defining variable ranges...")
    
    # Strategy: Lower thresholds for more trades, but test wider ranges
    # Focus on configurations that can achieve >50 trades
    test_configs = []
    
    # Strategy 1: RELAXED filters to get 50+ trades with 40-50% win rate
    # Lower thresholds, relaxed RSI/ADX to maximize trade count
    for tp in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for sl in [1.0, 1.2]:
            for thresh in [0.30, 0.35, 0.40, 0.45]:  # Lower thresholds for more trades
                for risk in [0.040, 0.045, 0.050, 0.055, 0.060]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                        'require_discount_premium': False,
                        'require_inducement': False,
                        'require_rsi_filter': True,
                        'min_rsi_bull': 40,  # Relaxed to get more trades
                        'max_rsi_bull': 80,  # Relaxed
                        'require_adx_filter': True,
                        'min_adx': 20,  # Relaxed to get more trades
                        'require_bull_only': True,
                    })
    
    # Strategy 2: Very relaxed - Maximum trades (for 50+ requirement)
    for tp in [3.5, 4.0, 4.5, 5.0]:
        for sl in [1.0, 1.2]:
            for thresh in [0.25, 0.30, 0.35]:
                for risk in [0.040, 0.045, 0.050, 0.055, 0.060]:
                    for cooldown in [10, 15]:  # Reduced cooldown for more trades
                        test_configs.append({
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'entry_type_filter': 'ob_only',
                            'cooldown_minutes': cooldown,
                            'per_trade_risk_pct': risk,
                            'require_discount_premium': False,
                            'require_inducement': False,
                            'require_rsi_filter': False,  # Disable RSI filter to maximize trades
                            'require_adx_filter': True,
                            'min_adx': 18,  # Very relaxed
                            'require_bull_only': True,
                        })
    
    # Strategy 3: Ultra relaxed - Remove ADX filter too, very low threshold
    for tp in [3.5, 4.0, 4.5]:
        for sl in [1.0, 1.2]:
            for thresh in [0.22, 0.25, 0.28]:  # Even lower threshold
                for risk in [0.045, 0.050, 0.055, 0.060]:
                    for cooldown in [5, 10]:  # Very short cooldown
                        test_configs.append({
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'model_threshold': thresh,
                            'entry_type_filter': 'ob_only',
                            'cooldown_minutes': cooldown,
                            'per_trade_risk_pct': risk,
                            'require_discount_premium': False,
                            'require_inducement': False,
                            'require_rsi_filter': False,
                            'require_adx_filter': False,  # Disable ADX too
                            'require_bull_only': True,  # Keep bull-only (much better)
                        })
    
    # Strategy 4: Minimum filters - Just bull-only and ML threshold
    for tp in [3.5, 4.0, 4.5]:
        for sl in [1.0, 1.2]:
            for thresh in [0.20, 0.22, 0.25]:
                for risk in [0.050, 0.055, 0.060]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 5,  # Minimum cooldown
                        'per_trade_risk_pct': risk,
                        'require_discount_premium': False,
                        'require_inducement': False,
                        'require_rsi_filter': False,
                        'require_adx_filter': False,
                        'require_bull_only': True,
                    })
    
    # Strategy 3: Moderate filters - Balance quality and quantity
    for tp in [4.0, 4.5, 5.0]:
        for sl in [1.0]:
            for thresh in [0.35, 0.40, 0.45]:
                for risk in [0.045, 0.050, 0.055]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                        'require_discount_premium': False,
                        'require_inducement': False,
                        'require_rsi_filter': True,
                        'min_rsi_bull': 45,
                        'max_rsi_bull': 75,
                        'require_adx_filter': True,
                        'min_adx': 22,
                        'require_bull_only': True,
                    })
    
    # Strategy 2: Low threshold (0.30-0.42), High TP (4-6), Tight SL (1.0), Very high risk
    for tp in [4.0, 4.5, 5.0, 5.5, 6.0]:
        for sl in [1.0]:
            for thresh in [0.30, 0.32, 0.35, 0.38, 0.40, 0.42]:
                for risk in [0.045, 0.050, 0.055, 0.060, 0.065, 0.070]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                    })
    
    # Strategy 3: Minimum threshold (0.22-0.32) for absolute max trades, Maximum risk
    for tp in [4.0, 4.5, 5.0]:
        for sl in [1.0]:
            for thresh in [0.22, 0.25, 0.28, 0.30, 0.32]:
                for risk in [0.055, 0.060, 0.065, 0.070, 0.075, 0.080]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                    })
    
    # Strategy 4: Balanced - Medium threshold (0.35-0.45), High TP (3.5-5), High risk
    for tp in [3.5, 4.0, 4.5, 5.0]:
        for sl in [1.0, 1.2]:
            for thresh in [0.35, 0.38, 0.40, 0.42, 0.45]:
                for risk in [0.040, 0.045, 0.050, 0.055, 0.060]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                    })
    
    # Strategy 5: Ultra aggressive - Lowest threshold (0.20-0.30), Maximum risk (6-10%)
    for tp in [4.5, 5.0, 5.5, 6.0]:
        for sl in [1.0]:
            for thresh in [0.20, 0.22, 0.25, 0.28, 0.30]:
                for risk in [0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.100]:
                    test_configs.append({
                        'tp_atr_mult': tp,
                        'sl_atr_mult': sl,
                        'model_threshold': thresh,
                        'entry_type_filter': 'ob_only',
                        'cooldown_minutes': 15,
                        'per_trade_risk_pct': risk,
                    })
    
    total_combinations = len(test_configs)
    logger.info(f"✓ Generated {total_combinations} combinations to test")
    
    # Test combinations
    logger.info("\n[5/6] Testing combinations...")
    logger.info("=" * 80)
    logger.info(f"Starting search through {total_combinations} combinations...")
    logger.info("Live progress will be shown below:\n")
    sys.stdout.flush()
    
    found_config = None
    tested = 0
    last_progress_time = datetime.now()
    best_results_local = []  # Initialize BEFORE loop
    
    # Signal handler to show best results on interrupt
    def signal_handler(sig, frame):
        logger.info("\n\n⚠️  Script interrupted!")
        if best_results_local:
            show_best_results(best_results_local, win_rate_target, profit_target, min_trades)
            # Save best result
            if best_results_local:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = os.path.join("reports", f"strategy_interrupted_{timestamp}")
                os.makedirs(report_dir, exist_ok=True)
                best = best_results_local[0]
                best['trades_df'].to_csv(os.path.join(report_dir, "trades.csv"), index=False)
                best['history'].to_csv(os.path.join(report_dir, "history.csv"), index=False)
                config_df = pd.DataFrame([best['config']])
                config_df.to_csv(os.path.join(report_dir, "configuration.csv"), index=False)
                logger.info(f"\nBest result saved to: {report_dir}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    for i, var_config in enumerate(test_configs):
        tested += 1
        
        # Progress update every 50 tests
        if tested % 50 == 0:
            pct = (tested / total_combinations) * 100
            logger.info(f"Progress: {tested}/{total_combinations} ({pct:.1f}%) | Still searching...")
        
        try:
            # Temporarily override config
            original_values = {}
            for key, value in var_config.items():
                if key == 'per_trade_risk_pct':
                    original_values[key] = config['backtest']['propfirm'][key]
                    config['backtest']['propfirm'][key] = value
                elif key == 'cooldown_minutes':
                    original_values[key] = config['backtest'].get(key, 60)
                    config['backtest'][key] = value
                elif key in ['require_discount_premium', 'require_inducement', 'require_rsi_filter', 'require_adx_filter', 
                             'min_adx', 'min_rsi_bull', 'max_rsi_bull', 'min_rsi_bear', 'max_rsi_bear', 'require_bull_only']:
                    # Handle new filter flags
                    if key == 'min_adx':
                        original_values[key] = config['strategy'].get(key, 20)
                    elif 'rsi' in key:
                        original_values[key] = config['strategy'].get(key, 50 if 'bull' in key else 30)
                    elif key == 'require_bull_only':
                        original_values[key] = config['strategy'].get(key, False)
                    else:
                        original_values[key] = config['strategy'].get(key, True if 'require' in key else 20)
                    config['strategy'][key] = value
                else:
                    original_values[key] = config['strategy'][key]
                    config['strategy'][key] = value
            
            # Filter candidates
            candidates = candidates_base.copy()
            probs = probs_base.copy()
            
            # Apply entry type filter
            if var_config['entry_type_filter'] == 'ob_only':
                mask = candidates['entry_type'] == 'ob'
                candidates = candidates[mask].copy()
                probs = probs[mask]
            elif var_config['entry_type_filter'] == 'fvg_only':
                mask = candidates['entry_type'] == 'fvg'
                candidates = candidates[mask].copy()
                probs = probs[mask]
            
            # Apply threshold filter
            threshold = var_config['model_threshold']
            mask = probs >= threshold
            candidates = candidates[mask].copy()
            probs = probs[mask]
            
            if len(candidates) == 0:
                continue
            
            # Recalculate TP/SL and outcomes
            for idx in candidates.index:
                entry_price = candidates.loc[idx, 'entry_price']
                atr = candidates.loc[idx, 'atr']
                bias = candidates.loc[idx, 'bias']
                entry_time = candidates.loc[idx, 'entry_time']
                
                if bias == 'bull':
                    new_tp = entry_price + (var_config['tp_atr_mult'] * atr)
                    new_sl = entry_price - (var_config['sl_atr_mult'] * atr)
                else:
                    new_tp = entry_price - (var_config['tp_atr_mult'] * atr)
                    new_sl = entry_price + (var_config['sl_atr_mult'] * atr)
                
                candidates.loc[idx, 'tp'] = new_tp
                candidates.loc[idx, 'sl'] = new_sl
                
                # Re-check outcome
                entry_idx = df_base.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx >= 0 and entry_idx < len(df_base) - 1:
                    future_window = df_base.iloc[entry_idx+1 : min(entry_idx+1+200, len(df_base))]
                    
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
                        else:
                            if bar['high'] >= new_sl and not tp_hit_first:
                                sl_hit_first = True
                                break
                            if bar['low'] <= new_tp and not sl_hit_first:
                                tp_hit_first = True
                                break
                    
                    if tp_hit_first:
                        candidates.loc[idx, 'target'] = 1
                        candidates.loc[idx, 'pnl_r'] = var_config['tp_atr_mult'] / var_config['sl_atr_mult']
                    elif sl_hit_first:
                        candidates.loc[idx, 'target'] = 0
                        candidates.loc[idx, 'pnl_r'] = -1.0
                    else:
                        candidates.loc[idx, 'target'] = 0
                        candidates.loc[idx, 'pnl_r'] = 0.0
            
            candidates['prob'] = probs
            
            # Run backtest
            backtester = Backtester(config)
            history, trades = backtester.run(candidates, probs)
            
            if len(trades) == 0:
                continue
            
            # Calculate metrics
            wins = trades[trades['net_pnl'] > 0]
            losses = trades[trades['net_pnl'] <= 0]
            win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
            net_profit = trades['net_pnl'].sum()
            
            # Store result for comparison
            result = {
                'config': var_config.copy(),
                'trades': len(trades),
                'win_rate': win_rate,
                'net_profit': net_profit,
                'profit_factor': abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf'),
                'trades_df': trades,
                'history': history
            }
            
            # Add to best results (keep top 50)
            best_results_local.append(result)
            best_results_local.sort(key=lambda x: (
                (x['win_rate'] / win_rate_target) * 0.4 +
                (x['net_profit'] / profit_target) * 0.4 +
                (x['trades'] / min_trades) * 0.2
            ), reverse=True)
            best_results_local = best_results_local[:50]
            
            # Check if meets criteria
            meets_criteria = (
                win_rate >= win_rate_target and
                net_profit >= profit_target and
                len(trades) >= min_trades
            )
            
            if meets_criteria:
                logger.info("\n" + "=" * 80)
                logger.info("🎯 FOUND CONFIGURATION MEETING ALL CRITERIA!")
                logger.info("=" * 80)
                logger.info(f"Configuration #{tested}:")
                logger.info(f"  TP Multiplier: {var_config['tp_atr_mult']}")
                logger.info(f"  SL Multiplier: {var_config['sl_atr_mult']}")
                logger.info(f"  Model Threshold: {var_config['model_threshold']}")
                logger.info(f"  Entry Type Filter: {var_config['entry_type_filter']}")
                logger.info(f"  Cooldown Minutes: {var_config['cooldown_minutes']}")
                logger.info(f"  Risk Per Trade: {var_config['per_trade_risk_pct']*100:.2f}%")
                logger.info(f"\nResults:")
                logger.info(f"  Trades: {len(trades)}")
                logger.info(f"  Win Rate: {win_rate:.2f}%")
                logger.info(f"  Net Profit: ${net_profit:.2f}")
                logger.info(f"  Profit Factor: {abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf'):.2f}")
                
                found_config = {
                    'config': var_config,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'net_profit': net_profit,
                    'trades_df': trades,
                    'history': history
                }
                break
            
            # Log promising matches
            if win_rate >= win_rate_target * 0.7 or net_profit >= profit_target * 0.7 or len(trades) >= min_trades * 0.7:
                logger.info(f"  💡 Promising #{tested}: Trades={len(trades)}, WR={win_rate:.1f}%, Profit=${net_profit:.2f}")
                sys.stdout.flush()
        
        except Exception as e:
            logger.warning(f"Config {tested} failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue  # Skip to next config
        finally:
            # Restore original values
            for key, value in original_values.items():
                if key == 'per_trade_risk_pct':
                    config['backtest']['propfirm'][key] = value
                elif key == 'cooldown_minutes':
                    config['backtest'][key] = value
                else:
                    config['strategy'][key] = value
    
    # Final results
    logger.info("\n" + "=" * 80)
    logger.info("[6/6] FINAL RESULTS")
    logger.info("=" * 80)
    
    if found_config:
        logger.info("✅ SUCCESS! Found configuration meeting all criteria.")
        logger.info(f"\nBest Configuration:")
        logger.info(f"  TP Multiplier: {found_config['config']['tp_atr_mult']}")
        logger.info(f"  SL Multiplier: {found_config['config']['sl_atr_mult']}")
        logger.info(f"  Model Threshold: {found_config['config']['model_threshold']}")
        logger.info(f"  Entry Type Filter: {found_config['config']['entry_type_filter']}")
        logger.info(f"  Cooldown Minutes: {found_config['config']['cooldown_minutes']}")
        logger.info(f"  Risk Per Trade: {found_config['config']['per_trade_risk_pct']*100:.2f}%")
        logger.info(f"\nPerformance:")
        logger.info(f"  Trades: {found_config['trades']}")
        logger.info(f"  Win Rate: {found_config['win_rate']:.2f}%")
        logger.info(f"  Net Profit: ${found_config['net_profit']:.2f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join("reports", f"strategy_found_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        found_config['trades_df'].to_csv(os.path.join(report_dir, "trades.csv"), index=False)
        found_config['history'].to_csv(os.path.join(report_dir, "history.csv"), index=False)
        
        # Save config
        config_df = pd.DataFrame([found_config['config']])
        config_df.to_csv(os.path.join(report_dir, "configuration.csv"), index=False)
        
        logger.info(f"\nResults saved to: {report_dir}")
        
        return found_config
    else:
        logger.warning(f"❌ No configuration found meeting all criteria after testing {tested} combinations.")
        logger.info("\nShowing best results found:")
        show_best_results(best_results_local, win_rate_target, profit_target, min_trades)
        
        # Return best result even if doesn't meet all criteria
        if best_results_local:
            logger.info(f"\nReturning best configuration found (may not meet all criteria)")
            return best_results_local[0]
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Find trading strategy configuration")
    parser.add_argument("--win-rate", type=float, required=True, help="Target win rate percentage (e.g., 80)")
    parser.add_argument("--profit-target", type=float, required=True, help="Target profit in dollars (e.g., 500)")
    parser.add_argument("--min-trades", type=int, required=True, help="Minimum number of trades (e.g., 50)")
    parser.add_argument("--from", dest="start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--initial-capital", type=float, default=1000, help="Initial capital (default: 1000)")
    parser.add_argument("--max-drawdown", type=float, help="Max drawdown percentage (optional)")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    
    args = parser.parse_args()
    
    setup_logging()
    
    result = find_strategy(
        win_rate_target=args.win_rate,
        profit_target=args.profit_target,
        min_trades=args.min_trades,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        max_drawdown=args.max_drawdown
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

