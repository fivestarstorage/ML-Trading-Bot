#!/usr/bin/env python3
"""
Optimize and Test RSI Mean Reversion Strategy

Find the best parameters for RSI mean reversion trading.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from itertools import product
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rsi_mean_reversion_strategy import RSIMeanReversionStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class SimpleBacktester:
    """Simple backtester."""

    def __init__(self, commission=0.0005, slippage=0.0002, take_profit=0.02, stop_loss=0.01, max_hold_bars=30):
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest."""
        df = data.copy()
        trades = []

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_signal = signals.iloc[i]

            # Entry
            if not in_position and current_signal != 0:
                position_side = current_signal
                entry_price = current_price * (1 + self.slippage * position_side)
                entry_idx = i
                in_position = True
                bars_held = 0
                continue

            # Position management
            if in_position:
                bars_held += 1

                # Calculate P&L
                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                exit_signal = False
                exit_reason = None

                # TP/SL/Time
                if pnl_pct >= self.take_profit:
                    exit_signal = True
                    exit_reason = 'TP'
                elif pnl_pct <= -self.stop_loss:
                    exit_signal = True
                    exit_reason = 'SL'
                elif bars_held >= self.max_hold_bars:
                    exit_signal = True
                    exit_reason = 'Time'

                if exit_signal:
                    exit_price = current_price * (1 - self.slippage * position_side)

                    if position_side == 1:
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    pnl_pct -= (self.commission * 2)

                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': position_side,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })

                    in_position = False

        if len(trades) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return': 0, 'sharpe': 0}

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        win_rate = len(winners) / len(trades_df)
        gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        total_return = trades_df['pnl_pct'].sum()
        sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if len(trades_df) > 1 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe': sharpe,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0
        }


def main():
    print("="*80)
    print("RSI MEAN REVERSION STRATEGY - PARAMETER OPTIMIZATION")
    print("="*80)

    # Fetch data
    SYMBOL = "BTC/USD"
    DAYS_BACK = 90

    print(f"\nFetching {DAYS_BACK} days of data for {SYMBOL}...")
    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=DAYS_BACK)

    try:
        data = client.fetch_bars(SYMBOL, timeframe="5Min", start=start, end=end, limit=10000)
        print(f"Fetched {len(data)} bars from {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Parameter grid
    print("\nOptimizing parameters...")
    param_grid = {
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'volume_threshold': [1.0, 1.2, 1.5],
        'use_volume_filter': [True, False]
    }

    best_result = None
    best_params = None
    best_sharpe = -999

    results = []

    total_tests = len(list(product(*param_grid.values())))
    test_num = 0

    for oversold, overbought, vol_thresh, use_vol in product(*param_grid.values()):
        test_num += 1

        # Test configuration
        strategy = RSIMeanReversionStrategy(
            rsi_period=14,
            rsi_oversold=oversold,
            rsi_overbought=overbought,
            volume_ma_period=20,
            volume_threshold=vol_thresh,
            use_volume_filter=use_vol
        )

        signals = strategy.generate_signals(data)

        # Skip if no signals
        if (signals != 0).sum() == 0:
            continue

        # Backtest
        backtester = SimpleBacktester(
            commission=0.0005,
            slippage=0.0002,
            take_profit=0.02,  # 2%
            stop_loss=0.01,     # 1%
            max_hold_bars=30
        )

        result = backtester.run(data, signals)

        # Store result
        result_entry = {
            'rsi_oversold': oversold,
            'rsi_overbought': overbought,
            'volume_threshold': vol_thresh,
            'use_volume_filter': use_vol,
            **result
        }
        results.append(result_entry)

        # Track best
        if result['sharpe'] > best_sharpe and result['total_trades'] > 30:
            best_sharpe = result['sharpe']
            best_result = result
            best_params = result_entry

        if test_num % 5 == 0:
            print(f"  Tested {test_num}/{total_tests} configurations...")

    # Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False)

    print("\n" + "="*80)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("="*80)
    print(results_df.head(10).to_string(index=False))

    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    if best_params:
        print(f"RSI Oversold: {best_params['rsi_oversold']}")
        print(f"RSI Overbought: {best_params['rsi_overbought']}")
        print(f"Volume Threshold: {best_params['volume_threshold']}")
        print(f"Use Volume Filter: {best_params['use_volume_filter']}")
        print(f"\nPerformance:")
        print(f"  Total Trades: {best_result['total_trades']}")
        print(f"  Win Rate: {best_result['win_rate']:.2%}")
        print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"  Total Return: {best_result['total_return']:.2%}")
        print(f"  Sharpe Ratio: {best_result['sharpe']:.2f}")
        print(f"  Avg Win: {best_result['avg_win']:.3%}")
        print(f"  Avg Loss: {best_result['avg_loss']:.3%}")

        # Verdict
        print("\n" + "="*80)
        if best_result['win_rate'] >= 0.55 and best_result['profit_factor'] > 1.3:
            print("✅ WINNING STRATEGY FOUND!")
            print("This configuration shows a profitable edge.")
        elif best_result['win_rate'] >= 0.50 and best_result['profit_factor'] > 1.0:
            print("⚠️  MARGINAL EDGE")
            print("Shows potential but needs further refinement.")
        else:
            print("❌ NO STRONG EDGE FOUND")
            print("RSI mean reversion may not work well on this asset/timeframe.")

        # Save results
        output_dir = Path("alpha_research/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"rsi_optimization_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nFull results saved to: {results_file}")


if __name__ == '__main__':
    main()
