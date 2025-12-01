#!/usr/bin/env python3
"""
Optimize Strategy Across All Years

Goal: Find parameters that work consistently across 2021-2025
Approach: Minimize variance while maximizing average profitability
Avoid overfitting by validating on ALL years simultaneously
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trend_breakout_strategy import TrendBreakoutStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class RobustBacktester:
    """Backtester for multi-year validation."""

    def __init__(self, take_profit=0.04, stop_loss=0.015, trailing_stop=0.025, max_hold_bars=40):
        self.commission = 0.0005
        self.slippage = 0.0002
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest."""
        df = data.copy()
        trades = []
        capital = 10000

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0
        max_profit = 0

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_signal = signals.iloc[i]

            if not in_position and current_signal != 0:
                position_side = current_signal
                entry_price = current_price * (1 + self.slippage * position_side)
                entry_idx = i
                in_position = True
                bars_held = 0
                max_profit = 0
                continue

            if in_position:
                bars_held += 1

                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                if pnl_pct > max_profit:
                    max_profit = pnl_pct

                exit_signal = False
                exit_reason = None

                if pnl_pct >= self.take_profit:
                    exit_signal = True
                    exit_reason = 'TP'
                elif pnl_pct <= -self.stop_loss:
                    exit_signal = True
                    exit_reason = 'SL'
                elif max_profit > 0.01 and (max_profit - pnl_pct) >= self.trailing_stop:
                    exit_signal = True
                    exit_reason = 'Trail'
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
                    pnl_dollars = capital * pnl_pct
                    capital += pnl_dollars

                    trades.append({
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })

                    in_position = False

        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        return {
            'total_trades': len(trades_df),
            'win_rate': len(winners) / len(trades_df) if len(trades_df) > 0 else 0,
            'total_return': (capital - 10000) / 10000,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl_pct'].mean() if len(losers) > 0 else 0,
            'profit_factor': (winners['pnl_pct'].sum() / abs(losers['pnl_pct'].sum())) if len(losers) > 0 else 999,
            'sharpe': (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
        }


def test_config_all_years(symbol, timeframe, strategy_params, backtest_params, years):
    """Test a configuration across all years and calculate aggregate metrics."""
    client = AlpacaClient()

    year_results = {}
    all_data = {}

    # Fetch data for all years
    for year in years:
        start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        end = pd.Timestamp(f'{year}-12-31', tz='UTC')

        try:
            data = client.fetch_bars(symbol, timeframe=timeframe, start=start, end=end, limit=10000)

            if len(data) < 500:
                continue

            all_data[year] = data

        except Exception as e:
            continue

    if len(all_data) < 4:  # Need at least 4 years
        return None

    # Test on each year
    for year, data in all_data.items():
        try:
            strategy = TrendBreakoutStrategy(**strategy_params)
            signals = strategy.generate_signals(data)

            if (signals != 0).sum() < 10:
                continue

            backtester = RobustBacktester(**backtest_params)
            result = backtester.run(data, signals)

            if result and result['total_trades'] >= 20:
                year_results[year] = result

        except Exception as e:
            continue

    if len(year_results) < 4:
        return None

    # Calculate aggregate metrics
    returns = [r['total_return'] for r in year_results.values()]
    win_rates = [r['win_rate'] for r in year_results.values()]
    sharpes = [r['sharpe'] for r in year_results.values()]
    profit_factors = [r['profit_factor'] for r in year_results.values() if r['profit_factor'] < 10]

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    avg_win_rate = np.mean(win_rates)
    min_return = np.min(returns)
    positive_years = sum(1 for r in returns if r > 0)

    # Consistency score: penalize variance, reward positive years
    # We want: positive average return, low variance, most years positive
    consistency_score = (
        avg_return * 2.0 +  # Weight average return highly
        (positive_years / len(year_results)) * 0.5 -  # Reward positive years
        std_return * 1.0 -  # Penalize variance
        max(0, -min_return) * 0.5  # Penalize worst year
    )

    return {
        'avg_return': avg_return,
        'std_return': std_return,
        'min_return': min_return,
        'max_return': np.max(returns),
        'avg_win_rate': avg_win_rate,
        'min_win_rate': np.min(win_rates),
        'avg_sharpe': np.mean(sharpes),
        'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0,
        'positive_years': positive_years,
        'total_years': len(year_results),
        'consistency_score': consistency_score,
        'total_trades': sum([r['total_trades'] for r in year_results.values()]),
        'year_results': year_results,
        'strategy_params': strategy_params,
        'backtest_params': backtest_params
    }


def main():
    print("="*80)
    print("🔧 OPTIMIZING STRATEGY ACROSS ALL YEARS")
    print("Goal: Consistent profitability without overfitting")
    print("="*80)

    SYMBOL = "ETH/USD"
    TIMEFRAME = "1Hour"
    YEARS = [2021, 2022, 2023, 2024, 2025]

    print(f"\nSymbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Years: {YEARS}")
    print(f"\nObjective: Maximize consistency score across ALL years")
    print(f"  - Positive average return")
    print(f"  - Low variance between years")
    print(f"  - Most years profitable")
    print(f"  - Minimize worst-case loss\n")

    # Parameter grid - broader search
    # Strategy parameters
    ema_combos = [
        (5, 13), (8, 21), (10, 26), (12, 30),
        (5, 21), (8, 26), (10, 30)
    ]

    breakout_periods = [5, 8, 10, 15, 20]

    volume_settings = [
        (False, 1.0),  # No volume filter
        (True, 1.2),   # Low threshold
        (True, 1.5),   # Medium threshold
        (True, 2.0),   # High threshold
    ]

    # Risk management - test wider ranges
    risk_configs = [
        # Conservative (tight stops, small targets)
        {'take_profit': 0.025, 'stop_loss': 0.012, 'trailing_stop': 0.015, 'max_hold_bars': 30},
        {'take_profit': 0.030, 'stop_loss': 0.015, 'trailing_stop': 0.020, 'max_hold_bars': 35},

        # Moderate (balanced)
        {'take_profit': 0.035, 'stop_loss': 0.015, 'trailing_stop': 0.020, 'max_hold_bars': 40},
        {'take_profit': 0.040, 'stop_loss': 0.015, 'trailing_stop': 0.025, 'max_hold_bars': 40},
        {'take_profit': 0.045, 'stop_loss': 0.018, 'trailing_stop': 0.025, 'max_hold_bars': 45},

        # Aggressive (wider stops, bigger targets)
        {'take_profit': 0.050, 'stop_loss': 0.020, 'trailing_stop': 0.030, 'max_hold_bars': 50},
        {'take_profit': 0.060, 'stop_loss': 0.025, 'trailing_stop': 0.035, 'max_hold_bars': 60},
    ]

    total_combos = len(ema_combos) * len(breakout_periods) * len(volume_settings) * len(risk_configs)
    print(f"Testing {total_combos} configurations across {len(YEARS)} years...")
    print(f"This will take several minutes...\n")

    best_configs = []
    test_num = 0

    for (fast, slow), bp, (use_vol, vol_thresh), risk in product(
        ema_combos, breakout_periods, volume_settings, risk_configs
    ):
        test_num += 1

        if test_num % 20 == 0:
            best_so_far = best_configs[0] if best_configs else None
            if best_so_far:
                print(f"  Progress: {test_num}/{total_combos}... Best consistency: {best_so_far['consistency_score']:.3f} "
                      f"(Avg: {best_so_far['avg_return']:.1%}, {best_so_far['positive_years']}/{best_so_far['total_years']} years +)")
            else:
                print(f"  Progress: {test_num}/{total_combos}...")

        strategy_params = {
            'fast_ema': fast,
            'slow_ema': slow,
            'breakout_period': bp,
            'volume_threshold': vol_thresh,
            'use_volume_filter': use_vol,
            'volume_ma_period': 20
        }

        result = test_config_all_years(SYMBOL, TIMEFRAME, strategy_params, risk, YEARS)

        if result:
            best_configs.append(result)

    if len(best_configs) == 0:
        print("\n❌ No valid configurations found!")
        return

    # Sort by consistency score
    best_configs.sort(key=lambda x: x['consistency_score'], reverse=True)

    print(f"\n{'='*80}")
    print("📊 TOP 15 MOST CONSISTENT CONFIGURATIONS")
    print('='*80)
    print(f"\n{'#':<4}{'EMAs':<10}{'BP':<6}{'Vol':<6}{'TP/SL/Trail':<16}{'Years+':<8}{'Avg Ret':<10}{'Min Ret':<10}{'Std':<8}{'Score'}")
    print("-"*110)

    for i, config in enumerate(best_configs[:15], 1):
        sp = config['strategy_params']
        bp_params = config['backtest_params']

        emas = f"{sp['fast_ema']}/{sp['slow_ema']}"
        vol = f"{sp['volume_threshold']:.1f}" if sp['use_volume_filter'] else "N"
        risk_str = f"{bp_params['take_profit']*100:.1f}/{bp_params['stop_loss']*100:.1f}/{bp_params['trailing_stop']*100:.1f}"
        years_pos = f"{config['positive_years']}/{config['total_years']}"

        status = "✅" if config['positive_years'] >= 3 and config['avg_return'] > 0 else "⚠️ "

        print(f"{status}{i:<3}{emas:<10}{sp['breakout_period']:<6}{vol:<6}{risk_str:<16}"
              f"{years_pos:<8}{config['avg_return']:<10.1%}{config['min_return']:<10.1%}"
              f"{config['std_return']:<8.2%}{config['consistency_score']:.3f}")

    # Best config details
    best = best_configs[0]

    print(f"\n{'='*80}")
    print("🏆 MOST CONSISTENT CONFIGURATION")
    print('='*80)

    print(f"\n📋 Strategy Parameters:")
    for k, v in best['strategy_params'].items():
        print(f"  {k}: {v}")

    print(f"\n🎯 Risk Management:")
    for k, v in best['backtest_params'].items():
        print(f"  {k}: {v}")

    print(f"\n📊 Overall Performance:")
    print(f"  Consistency Score: {best['consistency_score']:.3f}")
    print(f"  Average Return: {best['avg_return']:.2%}")
    print(f"  Std Deviation: {best['std_return']:.2%}")
    print(f"  Min Return: {best['min_return']:.2%}")
    print(f"  Max Return: {best['max_return']:.2%}")
    print(f"  Average Win Rate: {best['avg_win_rate']:.2%}")
    print(f"  Average Sharpe: {best['avg_sharpe']:.2f}")
    print(f"  Average Profit Factor: {best['avg_profit_factor']:.2f}")
    print(f"  Positive Years: {best['positive_years']}/{best['total_years']}")
    print(f"  Total Trades: {best['total_trades']}")

    print(f"\n📅 Year-by-Year Breakdown:")
    print(f"{'Year':<8}{'Trades':<10}{'Win Rate':<12}{'Return':<12}{'Sharpe':<10}{'Status'}")
    print("-"*70)
    for year, yr_result in sorted(best['year_results'].items()):
        status = "✅" if yr_result['total_return'] > 0 else "❌"
        print(f"{year:<8}{yr_result['total_trades']:<10}{yr_result['win_rate']:<12.1%}"
              f"{yr_result['total_return']:<12.1%}{yr_result['sharpe']:<10.2f}{status}")

    print(f"\n{'='*80}")

    positive_pct = (best['positive_years'] / best['total_years']) * 100

    if best['positive_years'] >= 4 and best['avg_return'] > 0.10:
        print("✅ EXCELLENT! Strategy is consistently profitable!")
        print(f"   {best['positive_years']}/{best['total_years']} years profitable ({positive_pct:.0f}%)")
        print(f"   Average return: {best['avg_return']:.1%}")
    elif best['positive_years'] >= 3 and best['avg_return'] > 0:
        print("✅ GOOD! Strategy shows solid consistency")
        print(f"   {best['positive_years']}/{best['total_years']} years profitable ({positive_pct:.0f}%)")
        print(f"   Average return: {best['avg_return']:.1%}")
    elif best['avg_return'] > 0:
        print("⚠️  MODERATE. Strategy is profitable on average but inconsistent")
        print(f"   {best['positive_years']}/{best['total_years']} years profitable ({positive_pct:.0f}%)")
        print(f"   Average return: {best['avg_return']:.1%}")
    else:
        print("❌ Strategy still showing losses. May need:")
        print("   - Different asset or timeframe")
        print("   - Alternative strategy approach")
        print("   - ML-based adaptive system")

    # Save results
    output_dir = Path("alpha_research/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"consistent_optimization_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write(f"Multi-Year Optimization Results\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Best Configuration:\n")
        f.write(f"Consistency Score: {best['consistency_score']:.3f}\n")
        f.write(f"Average Return: {best['avg_return']:.2%}\n")
        f.write(f"Positive Years: {best['positive_years']}/{best['total_years']}\n\n")
        f.write(f"Strategy: {best['strategy_params']}\n\n")
        f.write(f"Risk: {best['backtest_params']}\n\n")
        f.write(f"Yearly Results:\n")
        for year, yr_result in sorted(best['year_results'].items()):
            f.write(f"{year}: {yr_result['total_return']:.2%} return, {yr_result['win_rate']:.2%} WR, "
                   f"{yr_result['total_trades']} trades, Sharpe {yr_result['sharpe']:.2f}\n")

    print(f"\n📁 Results saved to: {results_file}")

    # Save best config as JSON for easy use
    import json
    config_file = output_dir / f"best_config_{timestamp}.json"

    config_output = {
        'symbol': SYMBOL,
        'timeframe': TIMEFRAME,
        'strategy_params': best['strategy_params'],
        'backtest_params': best['backtest_params'],
        'performance': {
            'avg_return': best['avg_return'],
            'std_return': best['std_return'],
            'avg_win_rate': best['avg_win_rate'],
            'positive_years': best['positive_years'],
            'total_years': best['total_years'],
            'consistency_score': best['consistency_score']
        }
    }

    with open(config_file, 'w') as f:
        json.dump(config_output, f, indent=2)

    print(f"📁 Config saved to: {config_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
