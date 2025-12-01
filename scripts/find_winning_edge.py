#!/usr/bin/env python3
"""
Find Winning Edge - Comprehensive Strategy Testing

Test multiple strategies across different timeframes and parameters
until we find a winning combination.
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

from src.trend_breakout_strategy import TrendBreakoutStrategy
from src.rsi_mean_reversion_strategy import RSIMeanReversionStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class AdvancedBacktester:
    """Advanced backtester with trailing stops and better position management."""

    def __init__(
        self,
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        take_profit=0.03,  # 3%
        stop_loss=0.015,   # 1.5%
        trailing_stop=0.02,  # 2% trailing
        max_hold_bars=50,
        use_trailing=True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_bars = max_hold_bars
        self.use_trailing = use_trailing

    def run(self, data, signals):
        """Run backtest with trailing stops."""
        df = data.copy()
        trades = []
        capital = self.initial_capital

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0
        max_profit = 0  # Track max profit for trailing stop

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
                max_profit = 0
                continue

            # Position management
            if in_position:
                bars_held += 1

                # Calculate current P&L
                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Update max profit for trailing stop
                if pnl_pct > max_profit:
                    max_profit = pnl_pct

                exit_signal = False
                exit_reason = None

                # Take profit
                if pnl_pct >= self.take_profit:
                    exit_signal = True
                    exit_reason = 'TP'

                # Stop loss
                elif pnl_pct <= -self.stop_loss:
                    exit_signal = True
                    exit_reason = 'SL'

                # Trailing stop (if enabled and in profit)
                elif self.use_trailing and max_profit > 0.01:  # Only trail if up 1%+
                    drawdown_from_peak = max_profit - pnl_pct
                    if drawdown_from_peak >= self.trailing_stop:
                        exit_signal = True
                        exit_reason = 'Trail'

                # Time exit
                elif bars_held >= self.max_hold_bars:
                    exit_signal = True
                    exit_reason = 'Time'

                # Execute exit
                if exit_signal:
                    exit_price = current_price * (1 - self.slippage * position_side)

                    if position_side == 1:
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    pnl_pct -= (self.commission * 2)

                    pnl_dollars = capital * pnl_pct
                    capital += pnl_dollars

                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'LONG' if position_side == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'max_profit_reached': max_profit
                    })

                    in_position = False

        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_bars_held': 0
            }

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        win_rate = len(winners) / len(trades_df)
        gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        total_return_pct = (capital - self.initial_capital) / self.initial_capital
        sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if len(trades_df) > 1 and trades_df['pnl_pct'].std() > 0 else 0

        # Max drawdown
        capital_curve = trades_df['capital'].values
        running_max = np.maximum.accumulate(capital_curve)
        drawdown = (capital_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_return_pct,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
            'trades_df': trades_df
        }


def fetch_data(symbol, timeframe, days_back):
    """Fetch data from Alpaca."""
    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=days_back)

    try:
        data = client.fetch_bars(symbol, timeframe=timeframe, start=start, end=end, limit=10000)
        logger.info(f"Fetched {len(data)} {timeframe} bars for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch {timeframe} data: {e}")
        return None


def test_configuration(data, strategy_name, strategy_params, backtest_params, config_name):
    """Test a single configuration."""

    # Initialize strategy
    if strategy_name == "TrendBreakout":
        strategy = TrendBreakoutStrategy(**strategy_params)
    elif strategy_name == "RSIReversion":
        strategy = RSIMeanReversionStrategy(**strategy_params)
    else:
        return None

    # Generate signals
    signals = strategy.generate_signals(data)

    if (signals != 0).sum() == 0:
        return None

    # Backtest
    backtester = AdvancedBacktester(**backtest_params)
    results = backtester.run(data, signals)

    # Add metadata
    results['strategy'] = strategy_name
    results['config_name'] = config_name
    results['strategy_params'] = strategy_params
    results['backtest_params'] = backtest_params

    return results


def main():
    print("="*80)
    print("FIND WINNING EDGE - COMPREHENSIVE STRATEGY TESTING")
    print("="*80)

    SYMBOL = "BTC/USD"
    TIMEFRAMES = ["1Hour", "4Hour"]  # Test longer timeframes
    DAYS_BACK = 180  # 6 months of data

    all_results = []

    for timeframe in TIMEFRAMES:
        print(f"\n{'='*80}")
        print(f"TESTING TIMEFRAME: {timeframe}")
        print('='*80)

        # Fetch data
        print(f"\nFetching {timeframe} data for {SYMBOL}...")
        data = fetch_data(SYMBOL, timeframe, DAYS_BACK)

        if data is None or len(data) < 100:
            print(f"Insufficient data for {timeframe}, skipping...")
            continue

        print(f"Data: {len(data)} bars from {data.index.min()} to {data.index.max()}")

        # Test configurations
        configs = []

        # 1. Trend Breakout Configurations
        configs.append({
            'name': 'TrendBreakout_Conservative',
            'strategy': 'TrendBreakout',
            'strategy_params': {
                'fast_ema': 9,
                'slow_ema': 21,
                'breakout_period': 20,
                'volume_threshold': 1.5,
                'use_volume_filter': True
            },
            'backtest_params': {
                'commission': 0.0005,
                'slippage': 0.0002,
                'take_profit': 0.04,  # 4%
                'stop_loss': 0.02,    # 2%
                'trailing_stop': 0.025,  # 2.5%
                'max_hold_bars': 50,
                'use_trailing': True
            }
        })

        configs.append({
            'name': 'TrendBreakout_Aggressive',
            'strategy': 'TrendBreakout',
            'strategy_params': {
                'fast_ema': 5,
                'slow_ema': 13,
                'breakout_period': 10,
                'volume_threshold': 1.2,
                'use_volume_filter': True
            },
            'backtest_params': {
                'commission': 0.0005,
                'slippage': 0.0002,
                'take_profit': 0.05,  # 5%
                'stop_loss': 0.015,   # 1.5%
                'trailing_stop': 0.03,  # 3%
                'max_hold_bars': 30,
                'use_trailing': True
            }
        })

        configs.append({
            'name': 'TrendBreakout_NoVolFilter',
            'strategy': 'TrendBreakout',
            'strategy_params': {
                'fast_ema': 9,
                'slow_ema': 21,
                'breakout_period': 15,
                'use_volume_filter': False
            },
            'backtest_params': {
                'commission': 0.0005,
                'slippage': 0.0002,
                'take_profit': 0.06,  # 6%
                'stop_loss': 0.02,    # 2%
                'trailing_stop': 0.03,
                'max_hold_bars': 40,
                'use_trailing': True
            }
        })

        # 2. RSI Mean Reversion (might work on longer timeframes)
        configs.append({
            'name': 'RSI_LongerTF',
            'strategy': 'RSIReversion',
            'strategy_params': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_threshold': 1.3,
                'use_volume_filter': True
            },
            'backtest_params': {
                'commission': 0.0005,
                'slippage': 0.0002,
                'take_profit': 0.03,  # 3%
                'stop_loss': 0.015,   # 1.5%
                'trailing_stop': 0.02,
                'max_hold_bars': 20,
                'use_trailing': False
            }
        })

        # Test all configurations
        print(f"\nTesting {len(configs)} configurations on {timeframe}...")

        for i, config in enumerate(configs, 1):
            print(f"  [{i}/{len(configs)}] Testing {config['name']}...", end=' ')

            result = test_configuration(
                data,
                config['strategy'],
                config['strategy_params'],
                config['backtest_params'],
                config['name']
            )

            if result and result['total_trades'] > 0:
                result['timeframe'] = timeframe
                all_results.append(result)
                print(f"✓ {result['total_trades']} trades, {result['win_rate']:.1%} WR, {result['total_return_pct']:.1%} return")
            else:
                print("✗ No signals")

    # Analyze results
    if len(all_results) == 0:
        print("\n❌ No valid results generated!")
        return

    print("\n" + "="*80)
    print("RESULTS SUMMARY - ALL CONFIGURATIONS")
    print("="*80)

    # Sort by Sharpe ratio
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n{'Rank':<6}{'Config':<35}{'TF':<8}{'Trades':<8}{'WinRate':<10}{'PF':<8}{'Return':<10}{'Sharpe':<8}{'MaxDD':<10}")
    print("-"*110)

    for rank, r in enumerate(all_results[:15], 1):  # Top 15
        print(f"{rank:<6}{r['config_name']:<35}{r['timeframe']:<8}{r['total_trades']:<8}"
              f"{r['win_rate']:<10.2%}{r['profit_factor']:<8.2f}{r['total_return_pct']:<10.2%}"
              f"{r['sharpe']:<8.2f}{r['max_drawdown']:<10.2%}")

    # Find best strategy
    best = all_results[0]

    print("\n" + "="*80)
    print("🏆 BEST STRATEGY FOUND")
    print("="*80)
    print(f"\nStrategy: {best['strategy']}")
    print(f"Configuration: {best['config_name']}")
    print(f"Timeframe: {best['timeframe']}")
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Win Rate: {best['win_rate']:.2%}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total Return: {best['total_return_pct']:.2%}")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"  Max Drawdown: {best['max_drawdown']:.2%}")
    print(f"  Avg Win: {best['avg_win']:.3%}")
    print(f"  Avg Loss: {best['avg_loss']:.3%}")
    print(f"  Avg Bars Held: {best['avg_bars_held']:.1f}")

    if 'exit_reasons' in best:
        print(f"\n  Exit Reasons:")
        for reason, count in best['exit_reasons'].items():
            pct = count / best['total_trades'] * 100
            print(f"    {reason}: {count} ({pct:.1f}%)")

    print(f"\nStrategy Parameters:")
    for k, v in best['strategy_params'].items():
        print(f"  {k}: {v}")

    print(f"\nBacktest Parameters:")
    for k, v in best['backtest_params'].items():
        print(f"  {k}: {v}")

    # Verdict
    print("\n" + "="*80)
    if best['win_rate'] >= 0.50 and best['profit_factor'] > 1.5 and best['total_return_pct'] > 0.10:
        print("✅ EXCELLENT! WINNING STRATEGY FOUND!")
        print("This strategy shows strong edge and is ready for live trading (paper trading first).")
    elif best['win_rate'] >= 0.45 and best['profit_factor'] > 1.2 and best['total_return_pct'] > 0:
        print("✅ GOOD! PROFITABLE STRATEGY FOUND!")
        print("This strategy is profitable and shows promise.")
    elif best['total_return_pct'] > 0:
        print("⚠️  MARGINAL - Strategy is profitable but needs optimization")
    else:
        print("❌ No winning strategy found in this batch.")
        print("Recommendation: Test on different assets or continue optimization.")

    # Save results
    output_dir = Path("alpha_research/comprehensive_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary
    summary_data = []
    for r in all_results:
        summary_data.append({
            'strategy': r['strategy'],
            'config': r['config_name'],
            'timeframe': r['timeframe'],
            'trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'profit_factor': r['profit_factor'],
            'return_pct': r['total_return_pct'],
            'sharpe': r['sharpe'],
            'max_dd': r['max_drawdown']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    # Save best strategy trades
    if 'trades_df' in best and len(best['trades_df']) > 0:
        trades_file = output_dir / f"best_strategy_trades_{timestamp}.csv"
        best['trades_df'].to_csv(trades_file, index=False)
        print(f"Best strategy trades saved to: {trades_file}")


if __name__ == '__main__':
    main()
