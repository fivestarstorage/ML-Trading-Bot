#!/usr/bin/env python
"""
Test hybrid approach: OB entries at lower threshold + FVG entries at higher threshold.
This should give us MORE trades while maintaining quality.
"""
import sys
import os
import pandas as pd
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
    logger.info("TESTING HYBRID APPROACH")
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
    candidates_test['prob'] = probs
    
    # Test different hybrid configurations
    test_configs = [
        {
            'name': 'OB 0.55 + FVG 0.65',
            'ob_threshold': 0.55,
            'fvg_threshold': 0.65,
            'cooldown': 60
        },
        {
            'name': 'OB 0.55 + FVG 0.70',
            'ob_threshold': 0.55,
            'fvg_threshold': 0.70,
            'cooldown': 60
        },
        {
            'name': 'OB 0.50 + FVG 0.65',
            'ob_threshold': 0.50,
            'fvg_threshold': 0.65,
            'cooldown': 60
        },
        {
            'name': 'OB 0.50 + FVG 0.70',
            'ob_threshold': 0.50,
            'fvg_threshold': 0.70,
            'cooldown': 60
        },
        {
            'name': 'OB 0.55 + FVG 0.65 (30min cooldown)',
            'ob_threshold': 0.55,
            'fvg_threshold': 0.65,
            'cooldown': 30
        },
    ]
    
    results = []
    
    for test_config in test_configs:
        logger.info("\n" + "=" * 70)
        logger.info(f"TEST: {test_config['name']}")
        logger.info("=" * 70)
        
        # Filter candidates
        ob_candidates = candidates_test[
            (candidates_test['entry_type'] == 'ob') & 
            (candidates_test['prob'] >= test_config['ob_threshold'])
        ]
        fvg_candidates = candidates_test[
            (candidates_test['entry_type'] == 'fvg') & 
            (candidates_test['prob'] >= test_config['fvg_threshold'])
        ]
        
        combined = pd.concat([ob_candidates, fvg_candidates])
        combined = combined.sort_values('entry_time')
        
        logger.info(f"OB candidates (>= {test_config['ob_threshold']}): {len(ob_candidates)}")
        logger.info(f"FVG candidates (>= {test_config['fvg_threshold']}): {len(fvg_candidates)}")
        logger.info(f"Combined candidates: {len(combined)}")
        
        # Apply cooldown
        combined['entry_time'] = pd.to_datetime(combined['entry_time'])
        filtered_trades = []
        last_trade_time = None
        
        for idx, row in combined.iterrows():
            if last_trade_time is None:
                filtered_trades.append(row)
                last_trade_time = row['entry_time']
            else:
                time_diff = (row['entry_time'] - last_trade_time).total_seconds() / 60
                if time_diff >= test_config['cooldown']:
                    filtered_trades.append(row)
                    last_trade_time = row['entry_time']
        
        logger.info(f"After {test_config['cooldown']} min cooldown: {len(filtered_trades)} trades")
        
        if len(filtered_trades) == 0:
            logger.warning("No trades after filtering. Skipping.")
            continue
        
        # Run backtest
        filtered_df = pd.DataFrame(filtered_trades)
        filtered_probs = filtered_df['prob'].values
        filtered_candidates = filtered_df.drop(columns=['prob'])
        
        # Temporarily override config
        original_threshold = config['strategy']['model_threshold']
        original_cooldown = config['backtest'].get('cooldown_minutes', 60)
        original_entry_filter = config['strategy'].get('entry_type_filter', 'both')
        
        config['strategy']['model_threshold'] = min(test_config['ob_threshold'], test_config['fvg_threshold'])
        config['backtest']['cooldown_minutes'] = test_config['cooldown']
        config['strategy']['entry_type_filter'] = 'both'  # Allow both types since we already filtered
        
        try:
            backtester = Backtester(config)
            history, trades = backtester.run(filtered_candidates, filtered_probs)
            
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
            
            # Count OB vs FVG trades
            ob_trades = len(trades[trades['entry_type'] == 'ob'])
            fvg_trades = len(trades[trades['entry_type'] == 'fvg'])
            
            results.append({
                'name': test_config['name'],
                'ob_threshold': test_config['ob_threshold'],
                'fvg_threshold': test_config['fvg_threshold'],
                'cooldown': test_config['cooldown'],
                'trades': len(trades),
                'ob_trades': ob_trades,
                'fvg_trades': fvg_trades,
                'win_rate': win_rate,
                'net_profit': net_profit,
                'profit_factor': profit_factor,
                'avg_win': wins['net_pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['net_pnl'].mean() if len(losses) > 0 else 0,
            })
            
            logger.info(f"\nResults:")
            logger.info(f"  Total Trades: {len(trades)} (OB: {ob_trades}, FVG: {fvg_trades})")
            logger.info(f"  Win Rate: {win_rate:.1f}%")
            logger.info(f"  Net Profit: ${net_profit:.2f}")
            logger.info(f"  Profit Factor: {profit_factor:.2f}")
            
        finally:
            config['strategy']['model_threshold'] = original_threshold
            config['backtest']['cooldown_minutes'] = original_cooldown
            config['strategy']['entry_type_filter'] = original_entry_filter
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('trades', ascending=False)
    
    print("\n" + results_df.to_string(index=False))
    
    # Find best
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATIONS")
    logger.info("=" * 70)
    
    print("\nMost Trades (min 30% win rate, min $100 profit):")
    min_quality = results_df[(results_df['win_rate'] >= 30) & (results_df['net_profit'] >= 100)]
    if len(min_quality) > 0:
        best = min_quality.iloc[0]
        print(f"  {best['name']}")
        print(f"  Trades: {best['trades']} (OB: {best['ob_trades']}, FVG: {best['fvg_trades']})")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Net Profit: ${best['net_profit']:.2f}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
    
    print("\nBest Net Profit:")
    best_profit = results_df.loc[results_df['net_profit'].idxmax()]
    print(f"  {best_profit['name']}")
    print(f"  Trades: {best_profit['trades']} (OB: {best_profit['ob_trades']}, FVG: {best_profit['fvg_trades']})")
    print(f"  Win Rate: {best_profit['win_rate']:.1f}%")
    print(f"  Net Profit: ${best_profit['net_profit']:.2f}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"hybrid_test_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(report_dir, "results.csv"), index=False)
    logger.info(f"\nResults saved to {report_dir}")

if __name__ == "__main__":
    main()

