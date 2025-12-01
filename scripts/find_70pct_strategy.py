#!/usr/bin/env python3
"""
Find 70%+ Win Rate Strategy

Aggressively search for a configuration that achieves >70% win rate
across all years (2020-2025) to avoid overfitting.

Key insight: High win rate requires SMALL, FREQUENT wins with tight SL.
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


class TightStopBacktester:
    """Backtester optimized for high win rate with tight stops and small targets."""

    def __init__(self, take_profit=0.015, stop_loss=0.008, max_hold_bars=20):
        self.commission = 0.0005
        self.slippage = 0.0002
        self.take_profit = take_profit
        self.stop_loss = stop_loss
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

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_signal = signals.iloc[i]

            if not in_position and current_signal != 0:
                position_side = current_signal
                entry_price = current_price * (1 + self.slippage * position_side)
                entry_idx = i
                in_position = True
                bars_held = 0
                continue

            if in_position:
                bars_held += 1

                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                exit_signal = False
                exit_reason = None

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

        return {
            'total_trades': len(trades_df),
            'win_rate': len(winners) / len(trades_df),
            'total_return': (capital - 10000) / 10000,
            'profit_factor': (winners['pnl_pct'].sum() / abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())) if len(trades_df[trades_df['pnl_pct'] < 0]) > 0 else 999
        }


def test_config_all_years(symbol, timeframe, strategy_params, backtest_params, years):
    """Test a configuration across all years."""
    client = AlpacaClient()

    year_results = {}

    for year in years:
        start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        end = pd.Timestamp(f'{year}-12-31', tz='UTC')

        try:
            data = client.fetch_bars(symbol, timeframe=timeframe, start=start, end=end, limit=10000)

            if len(data) < 500:
                continue

            strategy = TrendBreakoutStrategy(**strategy_params)
            signals = strategy.generate_signals(data)

            if (signals != 0).sum() < 10:
                continue

            backtester = TightStopBacktester(**backtest_params)
            result = backtester.run(data, signals)

            if result and result['total_trades'] >= 20:
                year_results[year] = result

        except:
            continue

    if len(year_results) < 3:  # Need at least 3 years
        return None

    # Calculate average win rate across years
    avg_win_rate = np.mean([r['win_rate'] for r in year_results.values()])
    min_win_rate = np.min([r['win_rate'] for r in year_results.values()])
    total_trades = sum([r['total_trades'] for r in year_results.values()])
    avg_return = np.mean([r['total_return'] for r in year_results.values()])

    return {
        'avg_win_rate': avg_win_rate,
        'min_win_rate': min_win_rate,
        'total_trades': total_trades,
        'avg_return': avg_return,
        'years_tested': len(year_results),
        'year_results': year_results,
        'strategy_params': strategy_params,
        'backtest_params': backtest_params
    }


def main():
    print("="*80)
    print("🎯 FINDING 70%+ WIN RATE STRATEGY")
    print("Testing Across All Years (2020-2025)")
    print("="*80)

    SYMBOL = "ETH/USD"
    TIMEFRAME = "1Hour"
    YEARS = [2021, 2022, 2023, 2024, 2025]  # Skip 2020 (no data)

    print(f"\nSymbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Years: {YEARS}")
    print(f"\nTarget: >70% Win Rate on ALL years")
    print(f"Searching parameter space...\n")

    # Parameter grid - focused on high win rate
    ema_combos = [(5, 13), (9, 21), (12, 26)]
    breakout_periods = [15, 20, 30]
    volume_settings = [(True, 1.3), (True, 1.5), (False, 1.0)]

    # Key: TIGHT stops and targets for high win rate
    risk_configs = [
        {'take_profit': 0.012, 'stop_loss': 0.008, 'max_hold_bars': 15},  # 1.2%/0.8%
        {'take_profit': 0.015, 'stop_loss': 0.008, 'max_hold_bars': 20},  # 1.5%/0.8%
        {'take_profit': 0.018, 'stop_loss': 0.010, 'max_hold_bars': 20},  # 1.8%/1.0%
        {'take_profit': 0.020, 'stop_loss': 0.010, 'max_hold_bars': 25},  # 2.0%/1.0%
    ]

    total_combos = len(ema_combos) * len(breakout_periods) * len(volume_settings) * len(risk_configs)
    print(f"Testing {total_combos} configurations across {len(YEARS)} years...")
    print("This may take a few minutes...\n")

    best_configs = []
    test_num = 0

    for (fast, slow), bp, (use_vol, vol_thresh), risk in product(
        ema_combos, breakout_periods, volume_settings, risk_configs
    ):
        test_num += 1

        if test_num % 10 == 0:
            print(f"  Progress: {test_num}/{total_combos}... (Best so far: {len([c for c in best_configs if c['avg_win_rate'] >= 0.70])} configs >70%)")

        strategy_params = {
            'fast_ema': fast,
            'slow_ema': slow,
            'breakout_period': bp,
            'volume_threshold': vol_thresh,
            'use_volume_filter': use_vol
        }

        result = test_config_all_years(SYMBOL, TIMEFRAME, strategy_params, risk, YEARS)

        if result and result['min_win_rate'] >= 0.65:  # At least 65% on worst year
            best_configs.append(result)

    # Sort by average win rate
    best_configs.sort(key=lambda x: x['avg_win_rate'], reverse=True)

    print(f"\n{'='*80}")
    print("📊 TOP 10 CONFIGURATIONS")
    print('='*80)
    print(f"\n{'#':<4}{'EMAs':<10}{'BP':<6}{'Vol':<6}{'TP/SL':<12}{'Years':<8}{'Avg WR':<10}{'Min WR':<10}{'Avg Ret':<12}{'Trades'}")
    print("-"*100)

    for i, config in enumerate(best_configs[:10], 1):
        sp = config['strategy_params']
        bp_params = config['backtest_params']

        emas = f"{sp['fast_ema']}/{sp['slow_ema']}"
        vol = "Y" if sp['use_volume_filter'] else "N"
        tp_sl = f"{bp_params['take_profit']*100:.1f}/{bp_params['stop_loss']*100:.1f}"

        status = "✅" if config['avg_win_rate'] >= 0.70 else "⚠️ "

        print(f"{status}{i:<3}{emas:<10}{sp['breakout_period']:<6}{vol:<6}{tp_sl:<12}"
              f"{config['years_tested']:<8}{config['avg_win_rate']:<10.1%}{config['min_win_rate']:<10.1%}"
              f"{config['avg_return']:<12.1%}{config['total_trades']}")

    if len(best_configs) == 0:
        print("\n❌ No configurations found meeting criteria!")
        print("Try relaxing parameters or testing different assets.")
        return

    # Best config details
    best = best_configs[0]

    print(f"\n{'='*80}")
    print("🏆 BEST CONFIGURATION")
    print('='*80)

    print(f"\nStrategy Parameters:")
    for k, v in best['strategy_params'].items():
        print(f"  {k}: {v}")

    print(f"\nRisk Management:")
    for k, v in best['backtest_params'].items():
        print(f"  {k}: {v}")

    print(f"\nOverall Performance:")
    print(f"  Average Win Rate: {best['avg_win_rate']:.2%}")
    print(f"  Minimum Win Rate: {best['min_win_rate']:.2%}")
    print(f"  Average Return: {best['avg_return']:.2%}")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Years Tested: {best['years_tested']}")

    print(f"\nYear-by-Year Breakdown:")
    for year, yr_result in sorted(best['year_results'].items()):
        status = "✅" if yr_result['win_rate'] >= 0.70 else "⚠️ "
        print(f"  {year}: {status} {yr_result['total_trades']} trades, {yr_result['win_rate']:.1%} WR, {yr_result['total_return']:+.1%} return")

    years_above_70 = sum(1 for yr in best['year_results'].values() if yr['win_rate'] >= 0.70)

    print(f"\n{'='*80}")
    if best['avg_win_rate'] >= 0.70 and years_above_70 >= 3:
        print("✅ EXCELLENT! 70%+ Win Rate Achieved!")
        print(f"Strategy is robust - {years_above_70}/{len(best['year_results'])} years above 70%")
    elif best['avg_win_rate'] >= 0.65:
        print("⚠️  CLOSE! Strategy approaches 70% win rate")
        print(f"{years_above_70}/{len(best['year_results'])} years above 70%")
        print("Consider further optimization")
    else:
        print("❌ Target not met. Best result: {:.1%}".format(best['avg_win_rate']))
        print("70%+ win rate is very challenging - current best shows promise")

    # Save results
    output_dir = Path("alpha_research/high_winrate_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"search_results_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write(f"High Win Rate Strategy Search Results\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Best Configuration:\n")
        f.write(f"Average Win Rate: {best['avg_win_rate']:.2%}\n")
        f.write(f"Strategy: {best['strategy_params']}\n")
        f.write(f"Risk: {best['backtest_params']}\n")
        f.write(f"\nYearly Results:\n")
        for year, yr_result in sorted(best['year_results'].items()):
            f.write(f"{year}: {yr_result['win_rate']:.2%} WR, {yr_result['total_trades']} trades\n")

    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
