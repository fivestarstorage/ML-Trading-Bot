#!/usr/bin/env python
"""
Backtest only script - uses existing trained model.
Usage:
    python scripts/backtest_only.py --from 2025-01-01 --to 2025-10-31
"""
import sys
import os
import pandas as pd
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

def main():
    parser = argparse.ArgumentParser(description="Backtest Only Script")
    parser.add_argument("--from", dest="start_date", default="2025-01-01", help="Backtest start date")
    parser.add_argument("--to", dest="end_date", default="2025-10-31", help="Backtest end date")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("BACKTEST ONLY (Using Existing Model)")
    logger.info("=" * 60)
    
    model_path = config['ml']['model_path']
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train a model first.")
        logger.error(f"Run: python scripts/train_and_backtest.py")
        return
    
    logger.info(f"Using model: {model_path}")
    
    # Load backtest data
    logger.info(f"\nLoading backtest data ({args.start_date} to {args.end_date})...")
    adapter = DataAdapter(config)
    base_tf_str = config['data']['timeframe_base']
    
    df_test = adapter.load_data(timeframe_suffix=base_tf_str)
    df_test = df_test[df_test.index >= pd.Timestamp(args.start_date)]
    df_test = df_test[df_test.index <= pd.Timestamp(args.end_date)]
    
    if df_test.empty:
        logger.error(f"No backtest data found for {args.start_date} to {args.end_date}. Exiting.")
        return
    
    logger.info(f"Backtest data loaded: {len(df_test)} rows")
    logger.info(f"  Date range: {df_test.index.min()} to {df_test.index.max()}")
    logger.info(f"  Price range: ${df_test['close'].min():.2f} - ${df_test['close'].max():.2f}")
    
    # Resample for structure
    h4_min = config['timeframes']['h4']
    h1_min = config['timeframes']['h1']
    logger.info(f"Resampling to H4 ({h4_min}min) and H1 ({h1_min}min)...")
    df_h4_test = adapter.resample_data(df_test, h4_min)
    df_h1_test = adapter.resample_data(df_test, h1_min)
    
    logger.info(f"Resampled: H4={len(df_h4_test)} bars, H1={len(df_h1_test)} bars")
    
    # Generate features and candidates
    logger.info("Analyzing market structure...")
    structure_test = StructureAnalyzer(df_h4_test, config)
    features = FeatureEngineer(config)
    
    logger.info("Calculating technical indicators...")
    df_test = features.calculate_technical_features(df_test)
    logger.info(f"  RSI range: {df_test['rsi'].min():.1f} - {df_test['rsi'].max():.1f}")
    logger.info(f"  ADX range: {df_test['adx'].min():.1f} - {df_test['adx'].max():.1f}")
    
    logger.info("Detecting FVGs and Order Blocks...")
    fvgs_test = features.detect_fvgs(df_test)
    obs_test = features.detect_obs(df_test)
    logger.info(f"  Detected {len(fvgs_test)} FVGs and {len(obs_test)} OBs")
    
    logger.info("Generating backtest candidates...")
    entry_gen_test = EntryGenerator(config, structure_test, features)
    candidates_test = entry_gen_test.generate_candidates(df_test, df_h4_test, fvgs_test, obs_test)
    
    logger.info(f"Generated {len(candidates_test)} backtest candidates")
    
    if candidates_test.empty:
        logger.warning("No backtest candidates generated. Exiting.")
        return
    
    # Analyze candidate quality
    win_rate_test = candidates_test['target'].mean() * 100
    logger.info(f"Candidate statistics:")
    logger.info(f"  Win rate: {win_rate_test:.1f}%")
    logger.info(f"  Wins: {(candidates_test['target'] == 1).sum()}, Losses: {(candidates_test['target'] == 0).sum()}")
    logger.info(f"  Average R-multiple: {candidates_test['pnl_r'].mean():.2f}")
    
    # Load model and predict
    logger.info(f"\nLoading model from {model_path}...")
    model = MLModel(config)
    model.load(model_path)
    logger.info(f"✓ Model loaded successfully")
    
    logger.info(f"Generating predictions (threshold: {config['strategy']['model_threshold']})...")
    probs = model.predict_proba(candidates_test)
    
    # Analyze predictions
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean probability: {probs.mean():.3f}")
    logger.info(f"  Min probability: {probs.min():.3f}")
    logger.info(f"  Max probability: {probs.max():.3f}")
    above_threshold = (probs >= config['strategy']['model_threshold']).sum()
    logger.info(f"  Candidates above threshold: {above_threshold} / {len(probs)} ({above_threshold/len(probs)*100:.1f}%)")
    
    # Run backtest
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BACKTEST SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Initial capital: ${config['backtest']['propfirm']['initial_capital']:.2f}")
    logger.info(f"Risk per trade: {config['backtest']['propfirm']['per_trade_risk_pct']*100:.2f}%")
    logger.info(f"Commission: ${config['backtest']['commission']:.2f} per oz")
    logger.info(f"Slippage: ${config['backtest']['slippage']:.2f} per oz")
    
    backtester = Backtester(config)
    history, trades = backtester.run(candidates_test, probs)
    
    # Save reports
    import matplotlib.pyplot as plt
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", timestamp)
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    trades.to_csv(os.path.join(report_dir, "backtest_trades.csv"))
    history.to_csv(os.path.join(report_dir, "backtest_history.csv"))
    
    # Plot equity
    if not history.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(history['time'], history['equity'], label='Equity', linewidth=2)
        
        if config['backtest']['mode'] == 'propfirm':
            initial = config['backtest']['propfirm']['initial_capital']
            dd_limit = initial * (1 - config['backtest']['propfirm']['max_drawdown_pct'])
            plt.axhline(y=dd_limit, color='r', linestyle='--', label='Hard Breach Level')
        
        plt.title(f"Equity Curve ({args.start_date} to {args.end_date})")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, "backtest_equity.png"))
        plt.close()
    
    # Summary metrics
    if not trades.empty:
        wins = trades[trades['net_pnl'] > 0]
        losses = trades[trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
        total_profit = trades['net_pnl'].sum()
        profit_factor = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if losses['net_pnl'].sum() != 0 else float('inf')
        
        summary = {
            'total_trades': len(trades),
            'net_profit': total_profit,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        pd.Series(summary).to_csv(os.path.join(report_dir, "backtest_summary.csv"))
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"  Wins: {len(wins)}, Losses: {len(losses)}")
        logger.info(f"Net Profit: ${total_profit:.2f}")
        logger.info(f"  Return: {(total_profit / config['backtest']['propfirm']['initial_capital']) * 100:.2f}%")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        # Additional metrics
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        
        # Drawdown analysis
        if not history.empty:
            peak = history['equity'].expanding().max()
            drawdown = (history['equity'] - peak) / peak * 100
            max_dd = drawdown.min()
            logger.info(f"Max Drawdown: {max_dd:.2f}%")
        
        # Cost analysis
        total_commission = (trades['units'] * config['backtest']['commission'] * 2).sum()
        total_slippage = (trades['units'] * config['backtest']['slippage'] * 2).sum()
        logger.info(f"Total Costs: ${total_commission + total_slippage:.2f} (Comm: ${total_commission:.2f}, Slippage: ${total_slippage:.2f})")
        
        logger.info(f"\nReports saved to: {report_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()


