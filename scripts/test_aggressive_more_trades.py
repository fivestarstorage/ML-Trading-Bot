#!/usr/bin/env python
"""
Test even more aggressive configurations to maximize trade count.
Tests:
- Lower thresholds (0.45, 0.48)
- Reduced cooldowns (15 min, 30 min)
- Quality filters (RSI, ADX thresholds)
- Different combinations
"""
import sys
import os
import pandas as pd
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
    filtered = candidates_test.copy()
    filtered['prob'] = probs
    
    # Apply threshold
    filtered = filtered[filtered['prob'] >= test_params['threshold']]
    
    # Filter entry type
    if test_params['entry_type'] == 'ob_only':
        filtered = filtered[filtered['entry_type'] == 'ob']
    elif test_params['entry_type'] == 'fvg_only':
        filtered = filtered[filtered['entry_type'] == 'fvg']
    elif test_params['entry_type'] == 'ob_preferred':
        ob_candidates = filtered[filtered['entry_type'] == 'ob']
        fvg_candidates = filtered[filtered['entry_type'] == 'fvg']
        fvg_candidates = fvg_candidates[fvg_candidates['prob'] >= test_params.get('fvg_threshold', 0.65)]
        filtered = pd.concat([ob_candidates, fvg_candidates])
    
    # Apply quality filters if specified
    if 'min_rsi' in test_params and test_params['min_rsi'] is not None:
        filtered = filtered[filtered['rsi'] >= test_params['min_rsi']]
    if 'max_rsi' in test_params and test_params['max_rsi'] is not None:
        filtered = filtered[filtered['rsi'] <= test_params['max_rsi']]
    if 'min_adx' in test_params and test_params['min_adx'] is not None:
        filtered = filtered[filtered['adx'] >= test_params['min_adx']]
    
    if len(filtered) == 0:
        return None
    
    # Temporarily override config
    original_threshold = config['strategy']['model_threshold']
    original_cooldown = config['backtest'].get('cooldown_minutes', 60)
    original_entry_filter = config['strategy'].get('entry_type_filter', 'both')
    
    config['strategy']['model_threshold'] = test_params['threshold']
    if 'cooldown_minutes' in config['backtest']:
        config['backtest']['cooldown_minutes'] = test_params['cooldown_minutes']
    config['strategy']['entry_type_filter'] = test_params.get('entry_type_filter', 'ob_only')
    
    try:
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
        
        history['equity'] = pd.to_numeric(history['equity'], errors='coerce')
        max_equity = history['equity'].max()
        min_equity = history['equity'].min()
        max_dd = ((max_equity - min_equity) / max_equity) * 100 if max_equity > 0 else 0
        
        expected_value = net_profit / len(trades) if len(trades) > 0 else 0
        
        result = {
            'threshold': test_params['threshold'],
            'entry_type': test_params['entry_type'],
            'fvg_threshold': test_params.get('fvg_threshold', test_params['threshold']),
            'cooldown_minutes': test_params['cooldown_minutes'],
            'min_rsi': test_params.get('min_rsi', None),
            'min_adx': test_params.get('min_adx', None),
            'trades': len(trades),
            'win_rate': win_rate,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'expected_value_per_trade': expected_value,
            'avg_win': wins['net_pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['net_pnl'].mean() if len(losses) > 0 else 0,
        }
        
        return result
        
    finally:
        config['strategy']['model_threshold'] = original_threshold
        if 'cooldown_minutes' in config['backtest']:
            config['backtest']['cooldown_minutes'] = original_cooldown
        config['strategy']['entry_type_filter'] = original_entry_filter

def main():
    config = load_config('config.yml')
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("AGGRESSIVE TESTING FOR MAXIMUM TRADES")
    logger.info("=" * 70)
    
    # Load backtest data
    logger.info("Loading data...")
    adapter = DataAdapter(config)
    base_tf_str = config['data']['timeframe_base']
    
    df_test = adapter.load_data(timeframe_suffix=base_tf_str)
    df_test = df_test[df_test.index >= pd.Timestamp('2025-01-01')]
    df_test = df_test[df_test.index <= pd.Timestamp('2025-10-31')]
    
    logger.info(f"Backtest data loaded: {len(df_test)} rows")
    
    # Generate candidates
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
    
    # Load model
    logger.info("Loading model...")
    model = MLModel(config)
    model.load(config['ml']['model_path'])
    probs = model.predict_proba(candidates_test)
    
    # Test aggressive configurations
    test_configs = [
        # Current best
        {'threshold': 0.50, 'entry_type': 'ob_only', 'cooldown_minutes': 60, 'entry_type_filter': 'ob_only'},
        
        # Lower thresholds
        {'threshold': 0.48, 'entry_type': 'ob_only', 'cooldown_minutes': 60, 'entry_type_filter': 'ob_only'},
        {'threshold': 0.45, 'entry_type': 'ob_only', 'cooldown_minutes': 60, 'entry_type_filter': 'ob_only'},
        
        # Reduced cooldown
        {'threshold': 0.50, 'entry_type': 'ob_only', 'cooldown_minutes': 30, 'entry_type_filter': 'ob_only'},
        {'threshold': 0.50, 'entry_type': 'ob_only', 'cooldown_minutes': 15, 'entry_type_filter': 'ob_only'},
        {'threshold': 0.48, 'entry_type': 'ob_only', 'cooldown_minutes': 30, 'entry_type_filter': 'ob_only'},
        
        # Include high-confidence FVG
        {'threshold': 0.50, 'entry_type': 'ob_preferred', 'fvg_threshold': 0.70, 'cooldown_minutes': 60, 'entry_type_filter': 'both'},
        {'threshold': 0.48, 'entry_type': 'ob_preferred', 'fvg_threshold': 0.70, 'cooldown_minutes': 60, 'entry_type_filter': 'both'},
        
        # Quality filters to maintain win rate
        {'threshold': 0.48, 'entry_type': 'ob_only', 'cooldown_minutes': 60, 'min_rsi': 30, 'min_adx': 15, 'entry_type_filter': 'ob_only'},
        {'threshold': 0.45, 'entry_type': 'ob_only', 'cooldown_minutes': 60, 'min_rsi': 30, 'min_adx': 15, 'entry_type_filter': 'ob_only'},
    ]
    
    logger.info(f"\nRunning {len(test_configs)} test configurations...")
    
    results = []
    
    for i, test_params in enumerate(tqdm(test_configs, desc="Testing")):
        try:
            result = run_single_test(config, candidates_test, probs, test_params)
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Test {i+1} failed: {e}")
    
    if not results:
        logger.error("No successful tests!")
        return
    
    results_df = pd.DataFrame(results)
    
    # Calculate score: trades * win_rate * profit_factor
    results_df['score'] = results_df['trades'] * (results_df['win_rate'] / 100) * results_df['profit_factor']
    
    # Sort by score
    results_df = results_df.sort_values('score', ascending=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL RESULTS (sorted by score)")
    logger.info("=" * 70)
    print("\n" + results_df.to_string(index=False))
    
    # Display best
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATIONS")
    logger.info("=" * 70)
    
    print("\n1. Best Score (trades * win_rate * profit_factor):")
    best_score = results_df.iloc[0]
    print(best_score.to_string())
    
    print("\n2. Most Trades (min 30% win rate, min $50 profit):")
    min_quality = results_df[(results_df['win_rate'] >= 30) & (results_df['net_profit'] >= 50)]
    if len(min_quality) > 0:
        most_trades = min_quality.loc[min_quality['trades'].idxmax()]
        print(most_trades.to_string())
    
    print("\n3. Best Win Rate (min 25 trades):")
    min_trades = results_df[results_df['trades'] >= 25]
    if len(min_trades) > 0:
        best_wr = min_trades.loc[min_trades['win_rate'].idxmax()]
        print(best_wr.to_string())
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"aggressive_test_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(report_dir, "all_results.csv"), index=False)
    logger.info(f"\nResults saved to {report_dir}")

if __name__ == "__main__":
    main()


