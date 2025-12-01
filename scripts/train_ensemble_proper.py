#!/usr/bin/env python3
"""
Train ensemble using REAL trading candidates from crypto strategy.
This uses actual trade outcomes, not arbitrary price predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from src.ensemble_model import HighPerformanceEnsemble
from src.features import FeatureEngineer
from src.data_adapter import DataAdapter
from src.crypto_strategy import CryptoMomentumEntryGenerator
from src.utils import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = get_logger()


def main():
    """Train ensemble using real trade candidates."""
    print("="*60)
    print("ENSEMBLE TRAINING WITH REAL TRADING STRATEGY")
    print("="*60)

    # Load config
    config_path = project_root / 'config.yml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\nLoading data...")

    # Load data
    start_date = '2023-01-01'
    end_date = '2024-11-29'
    data_adapter = DataAdapter(config, start_date=start_date, end_date=end_date)

    df = data_adapter.load_data()
    print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # Create features
    print("\nCreating features...")
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.calculate_technical_features(df)

    print(f"Features created: {len(df.columns)} columns")

    # Generate REAL trade candidates using crypto strategy
    print("\n" + "="*60)
    print("GENERATING REAL TRADE CANDIDATES")
    print("="*60)

    crypto_generator = CryptoMomentumEntryGenerator(config)
    candidates_df = crypto_generator.generate_candidates(df)

    if len(candidates_df) == 0:
        print("❌ No candidates generated! Check your strategy configuration.")
        return

    print(f"\n✅ Generated {len(candidates_df)} trade candidates")
    print(f"\nTarget distribution:")
    print(candidates_df['target'].value_counts())
    print(f"Win rate: {candidates_df['target'].mean():.2%}")

    # Prepare for ML training
    # The candidates already have all features we need
    exclude_cols = [
        'entry_time', 'entry_price', 'tp', 'sl', 'target', 'pnl_r',
        'bias', 'entry_type', 'session_label', 'exit_time', 'regime',
        'tp_mult_used', 'sl_mult_used'
    ]

    feature_cols = [c for c in candidates_df.columns if c not in exclude_cols]
    print(f"\nML features: {len(feature_cols)}")

    # Split data
    split_idx = int(len(candidates_df) * 0.8)
    df_train = candidates_df.iloc[:split_idx].copy()
    df_test = candidates_df.iloc[split_idx:].copy()

    print(f"\nTrain: {len(df_train)} candidates")
    print(f"Test:  {len(df_test)} candidates")
    print(f"Train win rate: {df_train['target'].mean():.2%}")
    print(f"Test win rate:  {df_test['target'].mean():.2%}")

    if len(df_train) < 1000:
        print(f"\n⚠️  Warning: Only {len(df_train)} training samples.")
        print("Consider using a longer date range for better training.")

    # Initialize and train ensemble
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE ON REAL TRADES")
    print("="*60)

    ensemble = HighPerformanceEnsemble(config)
    training_results = ensemble.train(df_train)

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    test_proba = ensemble.predict_proba(df_test)
    test_pred = (test_proba > 0.5).astype(int)
    test_actual = df_test['target'].values

    # Metrics
    accuracy = accuracy_score(test_actual, test_pred)
    precision = precision_score(test_actual, test_pred, zero_division=0)
    recall = recall_score(test_actual, test_pred, zero_division=0)
    f1 = f1_score(test_actual, test_pred, zero_division=0)
    auc = roc_auc_score(test_actual, test_proba)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    # Trading simulation with different thresholds
    print("\n" + "="*60)
    print("TRADING SIMULATION (PROBABILITY FILTERING)")
    print("="*60)

    for threshold in [0.50, 0.55, 0.60, 0.65]:
        test_df = df_test.copy()
        test_df['pred_proba'] = test_proba

        # Filter trades by probability threshold
        filtered_trades = test_df[test_df['pred_proba'] > threshold].copy()

        if len(filtered_trades) == 0:
            print(f"\nThreshold {threshold}: No trades")
            continue

        # Calculate metrics
        actual_win_rate = filtered_trades['target'].mean()
        avg_pnl_r = filtered_trades['pnl_r'].mean()
        total_return_r = filtered_trades['pnl_r'].sum()

        # Sharpe-like metric (R-based)
        sharpe_r = avg_pnl_r / filtered_trades['pnl_r'].std() * np.sqrt(len(filtered_trades)) if filtered_trades['pnl_r'].std() > 0 else 0

        print(f"\nThreshold {threshold}:")
        print(f"  Trades:        {len(filtered_trades)}")
        print(f"  Win Rate:      {actual_win_rate:.2%}")
        print(f"  Avg PnL (R):   {avg_pnl_r:.3f}R")
        print(f"  Total (R):     {total_return_r:.2f}R")
        print(f"  Sharpe (R):    {sharpe_r:.2f}")

    # Compare to strategy without ML filter
    print("\n" + "="*60)
    print("COMPARISON: ENSEMBLE vs NO FILTER")
    print("="*60)

    no_filter_win_rate = df_test['target'].mean()
    no_filter_avg_pnl = df_test['pnl_r'].mean()
    no_filter_total = df_test['pnl_r'].sum()

    print(f"\nNo Filter (All Test Trades):")
    print(f"  Trades:        {len(df_test)}")
    print(f"  Win Rate:      {no_filter_win_rate:.2%}")
    print(f"  Avg PnL (R):   {no_filter_avg_pnl:.3f}R")
    print(f"  Total (R):     {no_filter_total:.2f}R")

    # Best threshold
    best_threshold = 0.60
    best_filtered = df_test[test_proba > best_threshold]
    if len(best_filtered) > 0:
        improvement = (best_filtered['target'].mean() - no_filter_win_rate) / no_filter_win_rate * 100
        print(f"\nEnsemble @ {best_threshold} threshold:")
        print(f"  Win Rate Improvement: {improvement:+.1f}%")
        print(f"  PnL Improvement:      {(best_filtered['pnl_r'].sum() / no_filter_total - 1)*100:+.1f}%")

    # Save model
    print("\n" + "="*60)
    model_path = project_root / 'models' / 'ensemble_model.pkl'
    ensemble.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # Save candidates for analysis
    candidates_path = project_root / 'models' / 'training_candidates.parquet'
    candidates_df.to_parquet(candidates_path)
    print(f"Candidates saved to: {candidates_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if accuracy >= 0.60:
        print("🎉 EXCELLENT! >60% accuracy on real trades!")
    elif accuracy >= 0.55:
        print("👍 GOOD! >55% accuracy - better than random!")
    elif accuracy >= 0.50:
        print("📊 MODERATE - Slightly better than random")
    else:
        print("⚠️  Below 50% - ensemble may not be helping")

    print(f"\nKey Insight:")
    print(f"The ensemble is {'helping' if accuracy > no_filter_win_rate else 'not helping'} vs baseline strategy.")
    print(f"Baseline WR: {no_filter_win_rate:.1%}, Ensemble WR: {accuracy:.1%}")

    print("\n✅ Training complete!")
    print(f"\nNext steps:")
    print(f"1. Open Jupyter Lab: jupyter lab notebooks/ensemble_analysis.ipynb")
    print(f"2. Analyze feature importance and model performance")
    print(f"3. Fine-tune probability thresholds")
    print(f"4. Run full walk-forward backtest")


if __name__ == '__main__':
    main()
