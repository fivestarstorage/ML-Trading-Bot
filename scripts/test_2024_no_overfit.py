#!/usr/bin/env python
"""
Test the found configuration on 2024 data with NO overfitting.
Trains model on 2020-2023, tests on 2024 only.
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester
from src.ml_model import MLModel

def test_2024_no_overfit():
    config = load_config('config.yml')
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("TESTING ON 2024 DATA (NO OVERFITTING)")
    logger.info("="*80)
    logger.info("Training: 2020-01-01 to 2023-12-31")
    logger.info("Testing: 2024-01-01 to 2024-12-31")
    logger.info("="*80)
    
    # Load the best configuration
    config_path = "reports/strategy_found_20251122_174839/configuration.csv"
    if os.path.exists(config_path):
        best_config_df = pd.read_csv(config_path)
        best_config = best_config_df.iloc[0].to_dict()
        logger.info(f"\nLoaded configuration from {config_path}")
    else:
        # Use the found configuration manually
        best_config = {
            'tp_atr_mult': 3.5,
            'sl_atr_mult': 1.2,
            'model_threshold': 0.22,
            'entry_type_filter': 'ob_only',
            'cooldown_minutes': 5,
            'per_trade_risk_pct': 0.045,
            'require_discount_premium': False,
            'require_inducement': False,
            'require_rsi_filter': False,
            'require_adx_filter': False,
            'require_bull_only': True,
        }
        logger.info("\nUsing hardcoded best configuration")
    
    # Apply configuration
    config['strategy']['tp_atr_mult'] = best_config['tp_atr_mult']
    config['strategy']['sl_atr_mult'] = best_config['sl_atr_mult']
    config['strategy']['model_threshold'] = best_config['model_threshold']
    config['strategy']['entry_type_filter'] = best_config['entry_type_filter']
    config['backtest']['cooldown_minutes'] = int(best_config['cooldown_minutes'])
    config['backtest']['propfirm']['per_trade_risk_pct'] = best_config['per_trade_risk_pct']
    config['backtest']['propfirm']['initial_capital'] = 1000
    config['backtest']['propfirm']['max_drawdown_pct'] = 0.15
    
    # Apply filter settings
    for key in ['require_discount_premium', 'require_inducement', 'require_rsi_filter', 
                'require_adx_filter', 'require_bull_only']:
        if key in best_config:
            config['strategy'][key] = best_config[key]
    
    adapter = DataAdapter(config)
    
    # STEP 1: Train model on 2020-2023 (NO 2024 data)
    logger.info("\n[1/4] Training model on 2020-2023 data...")
    df_train = adapter.load_data()
    df_train = df_train[(df_train.index >= pd.Timestamp('2020-01-01')) & 
                        (df_train.index <= pd.Timestamp('2023-12-31'))]
    
    logger.info(f"Training data: {len(df_train)} bars ({df_train.index.min()} to {df_train.index.max()})")
    
    df_h4_train = adapter.load_h4_data(start_date='2020-01-01', end_date='2023-12-31')
    if df_h4_train is None:
        df_h4_train = adapter.resample_data(df_train, config['timeframes']['h4'])
    
    features = FeatureEngineer(config)
    structure_train = StructureAnalyzer(df_h4_train, config)
    df_train = features.calculate_technical_features(df_train)
    fvgs_train = features.detect_fvgs(df_train)
    obs_train = features.detect_obs(df_train)
    
    entry_gen_train = EntryGenerator(config, structure_train, features)
    candidates_train = entry_gen_train.generate_candidates(df_train, df_h4_train, fvgs_train, obs_train)
    
    logger.info(f"Generated {len(candidates_train)} training candidates")
    
    model = MLModel(config)
    model.train(candidates_train)
    model_path_2024 = "models/lgbm_model_2024_test.pkl"
    model.save(model_path_2024)
    logger.info(f"✓ Model trained and saved to {model_path_2024}")
    
    # STEP 2: Test on 2024 data ONLY
    logger.info("\n[2/4] Loading 2024 test data...")
    df_test = adapter.load_data()
    df_test = df_test[(df_test.index >= pd.Timestamp('2024-01-01')) & 
                      (df_test.index <= pd.Timestamp('2024-12-31'))]
    
    logger.info(f"Test data: {len(df_test)} bars ({df_test.index.min()} to {df_test.index.max()})")
    
    df_h4_test = adapter.load_h4_data(start_date='2024-01-01', end_date='2024-12-31')
    if df_h4_test is None:
        df_h4_test = adapter.resample_data(df_test, config['timeframes']['h4'])
    
    features_test = FeatureEngineer(config)
    structure_test = StructureAnalyzer(df_h4_test, config)
    df_test = features_test.calculate_technical_features(df_test)
    fvgs_test = features_test.detect_fvgs(df_test)
    obs_test = features_test.detect_obs(df_test)
    
    entry_gen_test = EntryGenerator(config, structure_test, features_test)
    candidates_test = entry_gen_test.generate_candidates(df_test, df_h4_test, fvgs_test, obs_test)
    
    logger.info(f"Generated {len(candidates_test)} test candidates")
    
    # STEP 3: Predict with trained model
    logger.info("\n[3/4] Generating predictions with trained model...")
    model_test = MLModel(config)
    model_test.load(model_path_2024)
    probs_test = model_test.predict_proba(candidates_test)
    
    logger.info(f"Mean probability: {probs_test.mean():.3f}")
    logger.info(f"Candidates above threshold ({best_config['model_threshold']}): {(probs_test >= best_config['model_threshold']).sum()}")
    
    # STEP 4: Run backtest
    logger.info("\n[4/4] Running backtest on 2024 data...")
    backtester = Backtester(config)
    history, trades = backtester.run(candidates_test, probs_test)
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("2024 BACKTEST RESULTS (NO OVERFITTING)")
    logger.info("="*80)
    
    if len(trades) == 0:
        logger.warning("No trades executed!")
        return
    
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100
    net_profit = trades['net_pnl'].sum()
    profit_factor = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf')
    
    logger.info(f"Total Trades: {len(trades)}")
    logger.info(f"Wins: {len(wins)}, Losses: {len(losses)}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Net Profit: ${net_profit:.2f}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    
    # Calculate drawdown
    equity_series = history['equity'].values
    running_max = pd.Series(equity_series).expanding().max()
    drawdowns = ((running_max - equity_series) / running_max) * 100
    max_dd = drawdowns.max()
    logger.info(f"Max Drawdown: {max_dd:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("reports", f"2024_test_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    trades.to_csv(os.path.join(report_dir, "trades.csv"), index=False)
    history.to_csv(os.path.join(report_dir, "history.csv"), index=False)
    
    # Generate equity curve
    logger.info("\nGenerating equity curve visualization...")
    plt.figure(figsize=(14, 8))
    
    # Main equity curve
    plt.subplot(2, 1, 1)
    plt.plot(history['time'], history['equity'], label='Equity', linewidth=2, color='#2E86AB')
    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Drawdown limit
    dd_limit = 1000 * (1 - 0.15)
    plt.axhline(y=dd_limit, color='r', linestyle='--', alpha=0.7, label='Max Drawdown Limit (15%)')
    
    plt.title(f'Equity Curve - 2024 Test (Trained on 2020-2023)\nWin Rate: {win_rate:.1f}% | Profit: ${net_profit:.2f} | Trades: {len(trades)}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Drawdown chart
    plt.subplot(2, 1, 2)
    plt.fill_between(history['time'], 0, drawdowns, color='red', alpha=0.3, label='Drawdown')
    plt.plot(history['time'], drawdowns, color='darkred', linewidth=1.5)
    plt.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7, label=f'Max DD: {max_dd:.2f}%')
    plt.title('Drawdown Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    equity_path = os.path.join(report_dir, "equity_curve_2024.png")
    plt.savefig(equity_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Equity curve saved to {equity_path}")
    logger.info(f"✓ Full results saved to {report_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    test_2024_no_overfit()


