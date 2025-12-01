#!/usr/bin/env python3
"""
Quick test script for the new ensemble system.
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
from src.utils import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = get_logger()


def main():
    """Test the ensemble model."""
    print("="*60)
    print("HIGH-PERFORMANCE ENSEMBLE TEST")
    print("="*60)

    # Load config
    config_path = project_root / 'config.yml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\nLoading data...")

    # Load data
    start_date = '2024-01-01'
    end_date = '2024-11-29'
    data_adapter = DataAdapter(config, start_date=start_date, end_date=end_date)

    df = data_adapter.load_data()
    print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # Create features
    print("\nCreating features...")
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.calculate_technical_features(df)

    # Create target (predict if price goes up in next 1h)
    df['target'] = (df['close'].shift(-12) > df['close']).astype(int)  # 12 bars = 1h on 5m
    df = df.dropna()

    print(f"Features created: {len(df.columns)} columns")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # Split data
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"\nTrain: {len(df_train)} samples")
    print(f"Test:  {len(df_test)} samples")

    # Initialize and train ensemble
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE")
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

    # Trading simulation
    print("\n" + "="*60)
    print("TRADING SIMULATION")
    print("="*60)

    test_df = df_test.copy()
    test_df['pred_proba'] = test_proba
    test_df['forward_return'] = test_df['close'].pct_change().shift(-1)

    # Only trade when confident (probability > 0.6 or < 0.4)
    confidence_threshold = 0.6
    test_df['signal'] = 0
    test_df.loc[test_df['pred_proba'] > confidence_threshold, 'signal'] = 1
    test_df.loc[test_df['pred_proba'] < (1 - confidence_threshold), 'signal'] = -1

    test_df['strategy_return'] = test_df['signal'] * test_df['forward_return']

    # Calculate metrics
    strategy_returns = test_df['strategy_return'].dropna()
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 12) if strategy_returns.std() > 0 else 0

    n_trades = (test_df['signal'].diff().abs() > 0).sum()
    winning_trades = (strategy_returns > 0).sum()
    win_rate = winning_trades / len(strategy_returns) if len(strategy_returns) > 0 else 0

    print(f"\nConfidence Threshold: {confidence_threshold}")
    print(f"Total Return:         {total_return:.2%}")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print(f"Win Rate:             {win_rate:.2%}")
    print(f"Number of Trades:     {n_trades}")

    # Save model
    print("\n" + "="*60)
    model_path = project_root / 'models' / 'ensemble_model.pkl'
    ensemble.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if accuracy >= 0.70:
        print("🎉 EXCELLENT! >70% accuracy achieved!")
    elif accuracy >= 0.60:
        print("👍 GOOD! >60% accuracy achieved!")
    elif accuracy >= 0.55:
        print("📊 MODERATE - Above random (50%)")
    else:
        print("⚠️  NEEDS IMPROVEMENT - Below 55%")

    if sharpe >= 2.0:
        print("🎉 EXCELLENT Sharpe ratio (>2.0)!")
    elif sharpe >= 1.5:
        print("👍 GOOD Sharpe ratio (>1.5)")
    elif sharpe >= 1.0:
        print("📊 MODERATE Sharpe ratio (>1.0)")
    else:
        print("⚠️  LOW Sharpe ratio (<1.0)")

    print("\n✅ Test complete!")
    print(f"\nNext steps:")
    print(f"1. Review the Jupyter notebook: notebooks/ensemble_testing.ipynb")
    print(f"2. Run full backtest: trading-bot --action backtest --wfa")
    print(f"3. Fine-tune ensemble parameters if needed")


if __name__ == '__main__':
    main()
