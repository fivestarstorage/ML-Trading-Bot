#!/usr/bin/env python
"""
Analyze strategy performance and suggest improvements.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

def analyze_strategy():
    config = load_config('config.yml')
    logger = setup_logging()
    
    # Use best config from results
    config['strategy']['tp_atr_mult'] = 6.0
    config['strategy']['sl_atr_mult'] = 1.0
    config['strategy']['model_threshold'] = 0.35
    config['strategy']['entry_type_filter'] = 'ob_only'
    config['backtest']['cooldown_minutes'] = 15
    config['backtest']['propfirm']['per_trade_risk_pct'] = 0.07
    config['backtest']['propfirm']['initial_capital'] = 1000
    
    logger.info("Loading data...")
    adapter = DataAdapter(config)
    df_base = adapter.load_data()
    df_base = df_base[(df_base.index >= pd.Timestamp('2025-01-01')) & (df_base.index <= pd.Timestamp('2025-10-31'))]
    
    df_h4 = adapter.load_h4_data(start_date='2025-01-01', end_date='2025-10-31')
    if df_h4 is None:
        df_h4 = adapter.resample_data(df_base, config['timeframes']['h4'])
    
    features = FeatureEngineer(config)
    structure = StructureAnalyzer(df_h4, config)
    df_base = features.calculate_technical_features(df_base)
    fvgs = features.detect_fvgs(df_base)
    obs = features.detect_obs(df_base)
    
    entry_gen = EntryGenerator(config, structure, features)
    candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)
    
    model = MLModel(config)
    model.load(config['ml']['model_path'])
    probs = model.predict_proba(candidates)
    
    # Filter
    mask = (candidates['entry_type'] == 'ob') & (probs >= 0.35)
    candidates_filtered = candidates[mask].copy()
    probs_filtered = probs[mask]
    
    backtester = Backtester(config)
    history, trades = backtester.run(candidates_filtered, probs_filtered)
    
    logger.info("\n" + "="*80)
    logger.info("STRATEGY ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nTotal candidates generated: {len(candidates)}")
    logger.info(f"Candidates after ML filter (>=0.35): {len(candidates_filtered)}")
    logger.info(f"Final trades: {len(trades)}")
    
    if len(trades) == 0:
        logger.warning("No trades executed!")
        return
    
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100
    
    logger.info(f"\nWin Rate: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L)")
    logger.info(f"Net Profit: ${trades['net_pnl'].sum():.2f}")
    
    logger.info(f"\n=== BY BIAS ===")
    bias_stats = trades.groupby('bias').agg({
        'net_pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
    })
    logger.info(bias_stats)
    
    logger.info(f"\n=== BY HOUR ===")
    trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
    hour_stats = trades.groupby('hour').agg({
        'net_pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
    })
    logger.info(hour_stats.sort_values(('net_pnl', 'sum'), ascending=False).head(10))
    
    logger.info(f"\n=== TECHNICAL INDICATORS (Winners vs Losers) ===")
    logger.info(f"Winners - Mean RSI: {wins['rsi'].mean():.1f}, Mean ADX: {wins['adx'].mean():.1f}")
    logger.info(f"Losers - Mean RSI: {losses['rsi'].mean():.1f}, Mean ADX: {losses['adx'].mean():.1f}")
    
    logger.info(f"\n=== RECOMMENDATIONS ===")
    logger.info("1. Strategy is too restrictive - only 26 trades in 10 months")
    logger.info("2. Win rate is low (26-43%) - need better entry filters")
    logger.info("3. Consider:")
    logger.info("   - Lowering ML threshold to get more trades")
    logger.info("   - Adding RSI/ADX filters to improve win rate")
    logger.info("   - Removing some SMC filters (inducement, discount/premium)")
    logger.info("   - Testing different TP/SL ratios")

if __name__ == "__main__":
    analyze_strategy()


