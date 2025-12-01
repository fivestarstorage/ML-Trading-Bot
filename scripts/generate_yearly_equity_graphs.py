#!/usr/bin/env python3
"""
Generate Year-by-Year Equity Graphs

Creates individual equity curves for each year (2020-2025) for the winning strategy.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trend_breakout_strategy import TrendBreakoutStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class DetailedBacktester:
    """Backtester with detailed equity tracking."""

    def __init__(self, take_profit=0.06, stop_loss=0.02, trailing_stop=0.035, max_hold_bars=60):
        self.commission = 0.0005
        self.slippage = 0.0002
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest and track equity."""
        df = data.copy()
        trades = []
        capital = 10000

        # Track equity at every bar (not just trades)
        equity_curve = []
        equity_dates = []

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0
        max_profit = 0

        for i in range(len(df)):
            current_time = df.index[i]
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

            # Track equity every bar
            if in_position:
                bars_held += 1

                # Calculate unrealized P&L
                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                if pnl_pct > max_profit:
                    max_profit = pnl_pct

                # Current equity (including unrealized)
                unrealized_pnl = capital * pnl_pct
                current_equity = capital + unrealized_pnl

                exit_signal = False
                exit_reason = None

                # Exit conditions
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
                        'entry_time': df.index[entry_idx],
                        'exit_time': current_time,
                        'side': 'LONG' if position_side == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
                        'exit_reason': exit_reason
                    })

                    in_position = False
                    current_equity = capital

            else:
                current_equity = capital

            equity_curve.append(current_equity)
            equity_dates.append(current_time)

        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]

        # Drawdown analysis
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max

        return {
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'equity_dates': equity_dates,
            'drawdown_curve': drawdown,
            'total_trades': len(trades_df),
            'win_rate': len(winners) / len(trades_df),
            'total_return': (capital - 10000) / 10000,
            'final_capital': capital,
            'max_drawdown': abs(drawdown.min()),
            'profit_factor': (winners['pnl_pct'].sum() / abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())) if len(trades_df[trades_df['pnl_pct'] < 0]) > 0 else 999,
            'sharpe': (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
        }


def test_year(symbol, timeframe, year, strategy_params):
    """Test strategy on a specific year."""
    print(f"\nProcessing {year}...", end=' ')

    client = AlpacaClient()

    # Define year boundaries
    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year}-12-31', tz='UTC')

    try:
        # Fetch data
        data = client.fetch_bars(symbol, timeframe=timeframe, start=start, end=end, limit=10000)

        if len(data) < 100:
            print(f"❌ Insufficient data ({len(data)} bars)")
            return None

        # Initialize strategy
        strategy = TrendBreakoutStrategy(**strategy_params)

        # Generate signals
        signals = strategy.generate_signals(data)

        if (signals != 0).sum() < 5:
            print(f"❌ Too few signals ({(signals != 0).sum()})")
            return None

        # Backtest
        backtester = DetailedBacktester(
            take_profit=0.06,
            stop_loss=0.02,
            trailing_stop=0.035,
            max_hold_bars=60
        )

        results = backtester.run(data, signals)

        if results is None:
            print("❌ No trades")
            return None

        # Print result
        print(f"✅ {results['total_trades']} trades, {results['win_rate']:.1%} WR, {results['total_return']:+.1%} return")

        results['year'] = year

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def plot_yearly_equity_graphs(all_results, output_dir):
    """Generate individual equity graphs for each year."""

    # Create figure with subplots for each year
    years_with_data = [year for year, result in sorted(all_results.items()) if result is not None]
    n_years = len(years_with_data)

    if n_years == 0:
        print("No data to plot!")
        return None

    # Calculate grid dimensions
    n_cols = 2
    n_rows = (n_years + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.suptitle('ETH Trend Breakout Strategy - Year-by-Year Equity Curves',
                 fontsize=16, fontweight='bold', y=0.995)

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, year in enumerate(years_with_data):
        result = all_results[year]
        ax = axes[idx]

        # Plot equity curve
        dates = result['equity_dates']
        equity = result['equity_curve']

        ax.plot(dates, equity, linewidth=2, color='#2E86AB', label='Equity')
        ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Start Capital')

        # Fill area under curve
        ax.fill_between(dates, 10000, equity, where=(np.array(equity) >= 10000),
                        alpha=0.3, color='green', interpolate=True)
        ax.fill_between(dates, 10000, equity, where=(np.array(equity) < 10000),
                        alpha=0.3, color='red', interpolate=True)

        # Formatting
        ax.set_title(f'{year} - {result["total_trades"]} Trades | {result["win_rate"]:.1%} WR | {result["total_return"]:+.1%} Return',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Capital ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add statistics box
        stats_text = f'Sharpe: {result["sharpe"]:.2f}\nProfit Factor: {result["profit_factor"]:.2f}\nMax DD: {result["max_drawdown"]:.1%}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for idx in range(n_years, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_file = output_dir / f"yearly_equity_curves_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Equity curves saved to: {chart_file}")

    return chart_file


def plot_combined_overlay(all_results, output_dir):
    """Create a single graph with all years overlaid."""

    fig, ax = plt.subplots(figsize=(16, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (year, result) in enumerate(sorted(all_results.items())):
        if result is None:
            continue

        dates = result['equity_dates']
        equity = result['equity_curve']

        label = f"{year} ({result['win_rate']:.1%} WR, {result['total_return']:+.1%})"
        ax.plot(dates, equity, linewidth=2, label=label, color=colors[idx], alpha=0.8)

    ax.axhline(y=10000, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Starting Capital')

    ax.set_title('ETH Trend Breakout Strategy - All Years Overlay',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Capital ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_file = output_dir / f"yearly_equity_overlay_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"✅ Overlay chart saved to: {chart_file}")

    return chart_file


def main():
    print("="*80)
    print("📊 GENERATING YEAR-BY-YEAR EQUITY GRAPHS")
    print("ETH Trend Breakout Strategy (Winning Configuration)")
    print("="*80)

    # Configuration
    SYMBOL = "ETH/USD"
    TIMEFRAME = "1Hour"
    YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

    # Winning strategy parameters
    strategy_params = {
        'fast_ema': 5,
        'slow_ema': 13,
        'breakout_period': 10,
        'volume_ma_period': 20,
        'volume_threshold': 1.5,
        'use_volume_filter': True
    }

    print(f"\nSymbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Strategy: ETH Trend Breakout (Validated Winner)")
    print(f"\nTesting years: {YEARS}")

    # Test each year
    all_results = {}

    for year in YEARS:
        result = test_year(SYMBOL, TIMEFRAME, year, strategy_params)
        all_results[year] = result

    # Summary table
    print(f"\n{'='*80}")
    print("YEAR-BY-YEAR SUMMARY")
    print('='*80)
    print(f"\n{'Year':<6}{'Trades':<10}{'Win Rate':<12}{'Return':<12}{'Sharpe':<10}{'Max DD':<10}{'Status'}")
    print("-"*80)

    valid_years = 0
    for year in YEARS:
        result = all_results[year]
        if result:
            valid_years += 1
            status = "✅"
            print(f"{year:<6}{result['total_trades']:<10}{result['win_rate']:<12.1%}"
                  f"{result['total_return']:<12.1%}{result['sharpe']:<10.2f}"
                  f"{result['max_drawdown']:<10.1%}{status}")
        else:
            print(f"{year:<6}{'N/A':<10}{'N/A':<12}{'N/A':<12}{'N/A':<10}{'N/A':<10}❌")

    # Generate graphs
    output_dir = Path("alpha_research/yearly_equity_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENERATING EQUITY GRAPHS")
    print('='*80)

    plot_yearly_equity_graphs(all_results, output_dir)
    plot_combined_overlay(all_results, output_dir)

    # Overall statistics
    valid_results = [r for r in all_results.values() if r is not None]

    if valid_results:
        avg_win_rate = np.mean([r['win_rate'] for r in valid_results])
        avg_return = np.mean([r['total_return'] for r in valid_results])
        avg_sharpe = np.mean([r['sharpe'] for r in valid_results])
        total_trades = sum([r['total_trades'] for r in valid_results])

        print(f"\n{'='*80}")
        print("OVERALL STATISTICS (ALL YEARS)")
        print('='*80)
        print(f"Years with Data: {valid_years}/{len(YEARS)}")
        print(f"Average Win Rate: {avg_win_rate:.2%}")
        print(f"Average Return per Year: {avg_return:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Total Trades (All Years): {total_trades}")

        if avg_win_rate >= 0.60:
            print("\n✅ EXCELLENT! Strategy performs consistently well across years!")
        elif avg_win_rate >= 0.50:
            print("\n✅ GOOD! Strategy shows solid performance across different market conditions!")
        elif avg_win_rate >= 0.45:
            print("\n⚠️  MODERATE. Strategy is profitable but could be improved.")
        else:
            print("\n⚠️  Performance varies significantly by year.")

    print("\n" + "="*80)
    print("✅ EQUITY GRAPHS GENERATED!")
    print("="*80)
    print(f"\nCheck the following directory for your graphs:")
    print(f"  {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - yearly_equity_curves_*.png (individual year graphs)")
    print(f"  - yearly_equity_overlay_*.png (all years combined)")


if __name__ == '__main__':
    main()
