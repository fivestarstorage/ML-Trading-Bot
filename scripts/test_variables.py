#!/usr/bin/env python
"""
Comprehensive variable testing script.
Tests different combinations of:
- Model thresholds
- Entry types (FVG, OB, or both)
- TP/SL multipliers
- Hour filters
- Cooldown periods
- Risk percentages
"""
import sys
import os
import pandas as pd
import numpy as np
import itertools
import datetime
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

def run_single_test(config, candidates_test, probs, test_params):
    """Run a single backtest with given parameters."""
    # Filter candidates based on test parameters
    filtered = candidates_test.copy()
    filtered['prob'] = probs
    
    # Apply threshold
    filtered = filtered[filtered['prob'] >= test_params['threshold']]
    
    # Filter entry type
    if test_params['entry_type'] == 'ob_only':
        filtered = filtered[filtered['entry_type'] == 'ob']
    elif test_params['entry_type'] == 'fvg_only':
        filtered = filtered[filtered['entry_type'] == 'fvg']
    # else: 'both' - no filter
    
    # Filter hours
    if test_params['filter_hours']:
        filtered['hour'] = pd.to_datetime(filtered['entry_time']).dt.hour
        filtered = filtered[~filtered['hour'].isin(test_params['filter_hours'])]
    
    if len(filtered) == 0:
        return None
    
    # Temporarily override config
    original_threshold = config['strategy']['model_threshold']
    original_tp_mult = config['strategy']['tp_atr_mult']
    original_sl_mult = config['strategy']['sl_atr_mult']
    original_cooldown = config['backtest'].get('cooldown_minutes', 60)
    original_risk = config['backtest']['propfirm']['per_trade_risk_pct']
    
    config['strategy']['model_threshold'] = test_params['threshold']
    config['strategy']['tp_atr_mult'] = test_params['tp_mult']
    config['strategy']['sl_atr_mult'] = test_params['sl_mult']
    if 'cooldown_minutes' in config['backtest']:
        config['backtest']['cooldown_minutes'] = test_params['cooldown_minutes']
    config['backtest']['propfirm']['per_trade_risk_pct'] = test_params['risk_pct']
    
    try:
        # Run backtest
        filtered_probs = filtered['prob'].values
        filtered_candidates = filtered.drop(columns=['prob'])
        
        backtester = Backtester(config)
        history, trades = backtester.run(filtered_candidates, filtered_probs)
        
        if len(trades) == 0:
            return None
        
        # Calculate metrics
        wins = trades[trades['net_pnl'] > 0]
        losses = trades[trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
        net_profit = trades['net_pnl'].sum()
        gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        history['equity'] = pd.to_numeric(history['equity'], errors='coerce')
        max_equity = history['equity'].max()
        min_equity = history['equity'].min()
        max_dd = ((max_equity - min_equity) / max_equity) * 100 if max_equity > 0 else 0
        
        result = {
            'threshold': test_params['threshold'],
            'entry_type': test_params['entry_type'],
            'tp_mult': test_params['tp_mult'],
            'sl_mult': test_params['sl_mult'],
            'cooldown_minutes': test_params['cooldown_minutes'],
            'risk_pct': test_params['risk_pct'],
            'filter_hours': str(test_params['filter_hours']) if test_params['filter_hours'] else 'None',
            'trades': len(trades),
            'win_rate': win_rate,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'avg_win': wins['net_pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['net_pnl'].mean() if len(losses) > 0 else 0,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
        return result
        
    finally:
        # Restore original config
        config['strategy']['model_threshold'] = original_threshold
        config['strategy']['tp_atr_mult'] = original_tp_mult
        config['strategy']['sl_atr_mult'] = original_sl_mult
        if 'cooldown_minutes' in config['backtest']:
            config['backtest']['cooldown_minutes'] = original_cooldown
        config['backtest']['propfirm']['per_trade_risk_pct'] = original_risk

def main():
    config = load_config('config.yml')
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE VARIABLE TESTING")
    logger.info("=" * 70)
    
    # Load backtest data
    logger.info("Loading data...")
    adapter = DataAdapter(config)
    base_tf_str = config['data']['timeframe_base']
    
    df_test = adapter.load_data(timeframe_suffix=base_tf_str)
    df_test = df_test[df_test.index >= pd.Timestamp('2025-01-01')]
    df_test = df_test[df_test.index <= pd.Timestamp('2025-10-31')]
    
    logger.info(f"Backtest data loaded: {len(df_test)} rows")
    
    # Generate candidates (only once)
    logger.info("Generating candidates...")
    h4_min = config['timeframes']['h4']
    df_h4_test = adapter.resample_data(df_test, h4_min)
    
    features_test = FeatureEngineer(config)
    structure_test = StructureAnalyzer(df_h4_test, config)
    
    df_test = features_test.calculate_technical_features(df_test)
    fvgs_test = features_test.detect_fvgs(df_test)
    obs_test = features_test.detect_obs(df_test)
    
    entry_gen_test = EntryGenerator(config, structure_test, features_test)
    candidates_test = entry_gen_test.generate_candidates(df_test, df_h4_test, fvgs_test, obs_test)
    
    logger.info(f"Generated {len(candidates_test)} candidates")
    
    # Load model and get predictions
    logger.info("Loading model and generating predictions...")
    model = MLModel(config)
    model.load(config['ml']['model_path'])
    probs = model.predict_proba(candidates_test)
    
    # Define parameter ranges to test
    test_params = {
        'threshold': [0.50, 0.55, 0.60, 0.65, 0.70],
        'entry_type': ['both', 'ob_only', 'fvg_only'],
        'tp_mult': [2.5, 3.0, 3.5, 4.0],
        'sl_mult': [1.0, 1.5, 2.0],
        'cooldown_minutes': [30, 60, 120],
        'risk_pct': [0.003, 0.005, 0.007],  # 0.3%, 0.5%, 0.7%
        'filter_hours': [None, [18, 19, 20], [11, 14, 15, 20]]  # None, evening, or worst hours
    }
    
    # Generate all combinations (but limit to reasonable subset)
    # For full grid search, this would be 5 * 3 * 4 * 3 * 3 * 3 * 3 = 4860 tests
    # Let's do a smarter approach: test key combinations
    
    logger.info("\nTesting key parameter combinations...")
    
    # Priority tests based on previous results
    priority_tests = [
        # Best from previous test
        {'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.0, 
         'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None},
        
        # Variations around best
        {'threshold': 0.65, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.0,
         'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None},
        {'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.5, 'sl_mult': 1.0,
         'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None},
        {'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.5,
         'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None},
        {'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.0,
         'cooldown_minutes': 120, 'risk_pct': 0.005, 'filter_hours': None},
        {'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.0,
         'cooldown_minutes': 60, 'risk_pct': 0.007, 'filter_hours': None},
    ]
    
    # Add systematic tests for key variables
    systematic_tests = []
    
    # Test different thresholds with OB only
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        systematic_tests.append({
            'threshold': thresh, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': 1.0,
            'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None
        })
    
    # Test different TP multipliers
    for tp in [2.5, 3.0, 3.5, 4.0]:
        systematic_tests.append({
            'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': tp, 'sl_mult': 1.0,
            'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None
        })
    
    # Test different SL multipliers
    for sl in [1.0, 1.5, 2.0]:
        systematic_tests.append({
            'threshold': 0.60, 'entry_type': 'ob_only', 'tp_mult': 3.0, 'sl_mult': sl,
            'cooldown_minutes': 60, 'risk_pct': 0.005, 'filter_hours': None
        })
    
    # Combine priority and systematic
    all_tests = priority_tests + systematic_tests
    
    # Remove duplicates
    seen = set()
    unique_tests = []
    for test in all_tests:
        key = tuple(sorted(test.items()))
        if key not in seen:
            seen.add(key)
            unique_tests.append(test)
    
    logger.info(f"Running {len(unique_tests)} test configurations...")
    
    results = []
    failed_tests = []
    
    for i, test_params in enumerate(tqdm(unique_tests, desc="Testing")):
        try:
            result = run_single_test(config, candidates_test, probs, test_params)
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Test {i+1} failed: {e}")
            failed_tests.append((test_params, str(e)))
    
    if not results:
        logger.error("No successful tests!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by net profit
    results_df = results_df.sort_values('net_profit', ascending=False)
    
    # Display top 20
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 CONFIGURATIONS (by Net Profit)")
    logger.info("=" * 70)
    print("\n" + results_df.head(20).to_string(index=False))
    
    # Display best by different metrics
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATIONS BY METRIC")
    logger.info("=" * 70)
    
    print("\nBest Net Profit:")
    best_profit = results_df.loc[results_df['net_profit'].idxmax()]
    print(best_profit.to_string())
    
    print("\nBest Win Rate (min 10 trades):")
    min_trades = results_df[results_df['trades'] >= 10]
    if len(min_trades) > 0:
        best_wr = min_trades.loc[min_trades['win_rate'].idxmax()]
        print(best_wr.to_string())
    
    print("\nBest Profit Factor (min 10 trades):")
    if len(min_trades) > 0:
        best_pf = min_trades.loc[min_trades['profit_factor'].idxmax()]
        print(best_pf.to_string())
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"variable_test_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(report_dir, "all_results.csv"), index=False)
    results_df.head(50).to_csv(os.path.join(report_dir, "top_50_results.csv"), index=False)
    
    logger.info(f"\nResults saved to {report_dir}")
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        logger.warning(f"\nFailed tests:")
        for test, error in failed_tests[:5]:  # Show first 5
            logger.warning(f"  {test}: {error}")

if __name__ == "__main__":
    main()


