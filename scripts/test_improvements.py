#!/usr/bin/env python
"""
Test the recommended improvements:
1. Increase threshold to 0.60
2. Filter out FVG entries (focus on OB only)
3. Filter out evening hours (18-20)
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

def main():
    config = load_config('config.yml')
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("TESTING IMPROVEMENTS")
    logger.info("=" * 70)
    
    # Load backtest data
    adapter = DataAdapter(config)
    base_tf_str = config['data']['timeframe_base']
    
    df_test = adapter.load_data(timeframe_suffix=base_tf_str)
    df_test = df_test[df_test.index >= pd.Timestamp('2025-01-01')]
    df_test = df_test[df_test.index <= pd.Timestamp('2025-10-31')]
    
    logger.info(f"Backtest data loaded: {len(df_test)} rows")
    
    # Resample for structure
    h4_min = config['timeframes']['h4']
    h1_min = config['timeframes']['h1']
    df_h4_test = adapter.resample_data(df_test, h4_min)
    df_h1_test = adapter.resample_data(df_test, h1_min)
    
    # Generate features and candidates
    features_test = FeatureEngineer(config)
    structure_test = StructureAnalyzer(df_h4_test, config)
    
    df_test = features_test.calculate_technical_features(df_test)
    fvgs_test = features_test.detect_fvgs(df_test)
    obs_test = features_test.detect_obs(df_test)
    
    entry_gen_test = EntryGenerator(config, structure_test, features_test)
    candidates_test = entry_gen_test.generate_candidates(df_test, df_h4_test, fvgs_test, obs_test)
    
    logger.info(f"Generated {len(candidates_test)} candidates")
    
    # Load model
    model = MLModel(config)
    model.load(config['ml']['model_path'])
    
    # Get predictions
    probs = model.predict_proba(candidates_test)
    candidates_test['prob'] = probs
    
    # Test different configurations
    tests = [
        {
            'name': 'Baseline (Current)',
            'threshold': 0.50,
            'filter_fvg': False,
            'filter_hours': None
        },
        {
            'name': 'Higher Threshold (0.60)',
            'threshold': 0.60,
            'filter_fvg': False,
            'filter_hours': None
        },
        {
            'name': 'OB Only',
            'threshold': 0.50,
            'filter_fvg': True,
            'filter_hours': None
        },
        {
            'name': 'OB Only + Higher Threshold',
            'threshold': 0.60,
            'filter_fvg': True,
            'filter_hours': None
        },
        {
            'name': 'OB Only + Higher Threshold + Filter Hours',
            'threshold': 0.60,
            'filter_fvg': True,
            'filter_hours': [18, 19, 20]  # Filter out evening hours
        }
    ]
    
    results = []
    
    for test_config in tests:
        logger.info("\n" + "=" * 70)
        logger.info(f"TEST: {test_config['name']}")
        logger.info("=" * 70)
        
        # Filter candidates
        filtered = candidates_test.copy()
        
        # Temporarily override config threshold for this test
        original_threshold = config['strategy']['model_threshold']
        config['strategy']['model_threshold'] = test_config['threshold']
        
        # Filter by threshold (backtester will also apply it, but we filter here for logging)
        filtered = filtered[filtered['prob'] >= test_config['threshold']]
        logger.info(f"After threshold ({test_config['threshold']}): {len(filtered)} candidates")
        
        # Filter FVG if requested
        if test_config['filter_fvg']:
            before_fvg = len(filtered)
            filtered = filtered[filtered['entry_type'] == 'ob']
            logger.info(f"After FVG filter: {len(filtered)} candidates (removed {before_fvg - len(filtered)} FVG)")
        
        # Filter hours if requested
        if test_config['filter_hours']:
            before_hours = len(filtered)
            filtered['hour'] = pd.to_datetime(filtered['entry_time']).dt.hour
            filtered = filtered[~filtered['hour'].isin(test_config['filter_hours'])]
            logger.info(f"After hour filter (excluding {test_config['filter_hours']}): {len(filtered)} candidates (removed {before_hours - len(filtered)})")
        
        if len(filtered) == 0:
            logger.warning("No candidates after filtering. Skipping test.")
            continue
        
        # Run backtest
        # Need to pass candidates and probabilities separately
        filtered_probs = filtered['prob'].values
        filtered_candidates = filtered.drop(columns=['prob']) if 'prob' in filtered.columns else filtered
        
        backtester = Backtester(config)
        history, trades = backtester.run(filtered_candidates, filtered_probs)
        
        if len(trades) == 0:
            logger.warning("No trades executed. Skipping test.")
            continue
        
        # Calculate metrics
        wins = trades[trades['net_pnl'] > 0]
        losses = trades[trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
        net_profit = trades['net_pnl'].sum()
        gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        results.append({
            'test': test_config['name'],
            'trades': len(trades),
            'win_rate': win_rate,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_win': wins['net_pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['net_pnl'].mean() if len(losses) > 0 else 0
        })
        
        logger.info(f"\nResults:")
        logger.info(f"  Trades: {len(trades)}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Net Profit: ${net_profit:.2f}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        
        # Restore original threshold
        config['strategy']['model_threshold'] = original_threshold
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY OF ALL TESTS")
    logger.info("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"improvements_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(report_dir, "test_results.csv"), index=False)
    logger.info(f"\nResults saved to {report_dir}")

if __name__ == "__main__":
    main()

