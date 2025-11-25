#!/usr/bin/env python
"""
Analyze why win rate is so low and identify patterns in winning vs losing trades.
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

def analyze_win_rate():
    config = load_config('config.yml')
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("WIN RATE ANALYSIS")
    logger.info("="*80)
    
    # Use relaxed filters to get more trades
    config['strategy']['require_discount_premium'] = False
    config['strategy']['require_inducement'] = False
    config['strategy']['require_rsi_filter'] = True
    config['strategy']['require_adx_filter'] = True
    config['strategy']['min_adx'] = 20
    config['strategy']['model_threshold'] = 0.35
    config['strategy']['entry_type_filter'] = 'ob_only'
    config['backtest']['cooldown_minutes'] = 15
    config['backtest']['propfirm']['per_trade_risk_pct'] = 0.07
    
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
    
    logger.info(f"Generated {len(candidates)} candidates")
    
    model = MLModel(config)
    model.load(config['ml']['model_path'])
    probs = model.predict_proba(candidates)
    
    # Filter
    mask = (candidates['entry_type'] == 'ob') & (probs >= 0.35)
    candidates_filtered = candidates[mask].copy()
    probs_filtered = probs[mask]
    
    logger.info(f"After filtering: {len(candidates_filtered)} candidates")
    
    # Analyze candidates BEFORE backtesting
    logger.info("\n" + "="*80)
    logger.info("CANDIDATE ANALYSIS (Before Backtest)")
    logger.info("="*80)
    
    logger.info(f"Total candidates: {len(candidates_filtered)}")
    logger.info(f"Win rate (from labels): {candidates_filtered['target'].mean()*100:.1f}%")
    logger.info(f"Mean ML probability: {probs_filtered.mean():.3f}")
    logger.info(f"Median ML probability: {np.median(probs_filtered):.3f}")
    
    # Analyze by features
    logger.info("\n=== BY RSI ===")
    candidates_filtered['rsi_bucket'] = pd.cut(candidates_filtered['rsi'], bins=[0, 30, 40, 50, 60, 70, 100], labels=['<30', '30-40', '40-50', '50-60', '60-70', '>70'])
    rsi_stats = candidates_filtered.groupby('rsi_bucket').agg({
        'target': ['count', 'mean']
    })
    logger.info(rsi_stats)
    
    logger.info("\n=== BY ADX ===")
    candidates_filtered['adx_bucket'] = pd.cut(candidates_filtered['adx'], bins=[0, 20, 25, 30, 40, 100], labels=['<20', '20-25', '25-30', '30-40', '>40'])
    adx_stats = candidates_filtered.groupby('adx_bucket').agg({
        'target': ['count', 'mean']
    })
    logger.info(adx_stats)
    
    logger.info("\n=== BY BIAS ===")
    bias_stats = candidates_filtered.groupby('bias').agg({
        'target': ['count', 'mean']
    })
    logger.info(bias_stats)
    
    logger.info("\n=== BY ML PROBABILITY ===")
    candidates_filtered['prob_bucket'] = pd.cut(probs_filtered, bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0], labels=['<0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '>0.7'])
    prob_stats = candidates_filtered.groupby('prob_bucket').agg({
        'target': ['count', 'mean']
    })
    logger.info(prob_stats)
    
    # Run backtest
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTEST")
    logger.info("="*80)
    
    backtester = Backtester(config)
    history, trades = backtester.run(candidates_filtered, probs_filtered)
    
    if len(trades) == 0:
        logger.warning("No trades executed!")
        return
    
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100
    
    logger.info(f"\n=== BACKTEST RESULTS ===")
    logger.info(f"Total trades: {len(trades)}")
    logger.info(f"Win rate: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L)")
    logger.info(f"Net profit: ${trades['net_pnl'].sum():.2f}")
    
    # Analyze winning vs losing trades
    logger.info("\n=== WINNING TRADES CHARACTERISTICS ===")
    logger.info(f"Mean RSI: {wins['rsi'].mean():.1f}")
    logger.info(f"Mean ADX: {wins['adx'].mean():.1f}")
    logger.info(f"Mean ML Prob: {wins.get('prob', pd.Series([0])).mean():.3f}")
    
    logger.info("\n=== LOSING TRADES CHARACTERISTICS ===")
    logger.info(f"Mean RSI: {losses['rsi'].mean():.1f}")
    logger.info(f"Mean ADX: {losses['adx'].mean():.1f}")
    logger.info(f"Mean ML Prob: {losses.get('prob', pd.Series([0])).mean():.3f}")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    
    if wins['rsi'].mean() < losses['rsi'].mean():
        logger.info("✓ Winners have lower RSI - consider filtering for RSI < 50 for longs")
    if wins['adx'].mean() > losses['adx'].mean():
        logger.info(f"✓ Winners have higher ADX - consider increasing min ADX to {int(wins['adx'].mean())}")
    
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_win_rate()


