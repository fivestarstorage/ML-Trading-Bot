#!/usr/bin/env python3
"""
Generate Detailed Yearly Reports

Creates comprehensive validation reports for each year (2020-2025)
matching the format from /alpha_research/final_validation/
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
    """Backtester with full trade tracking."""

    def __init__(self, take_profit=0.06, stop_loss=0.02, trailing_stop=0.035, max_hold_bars=60):
        self.commission = 0.0005
        self.slippage = 0.0002
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest with full tracking."""
        df = data.copy()
        trades = []
        capital = 10000
        equity_curve = [capital]
        equity_dates = [df.index[0]]

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
                continue

            # Position management
            if in_position:
                bars_held += 1

                # Calculate P&L
                if position_side == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Update max profit
                if pnl_pct > max_profit:
                    max_profit = pnl_pct

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
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
                        'bars_held': bars_held,
                        'max_profit': max_profit,
                        'exit_reason': exit_reason
                    })

                    equity_curve.append(capital)
                    equity_dates.append(current_time)

                    in_position = False

        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        # Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_dd = abs(drawdown.min())

        # Consecutive wins/losses
        is_winner = trades_df['pnl_pct'] > 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for winner in is_winner:
            if winner:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'equity_dates': equity_dates,
            'total_trades': len(trades_df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades_df),
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0,
            'largest_win': winners['pnl_pct'].max() if len(winners) > 0 else 0,
            'largest_loss': abs(losers['pnl_pct'].min()) if len(losers) > 0 else 0,
            'gross_profit': winners['pnl_dollars'].sum() if len(winners) > 0 else 0,
            'gross_loss': abs(losers['pnl_dollars'].sum()) if len(losers) > 0 else 0,
            'profit_factor': (winners['pnl_dollars'].sum() / abs(losers['pnl_dollars'].sum())) if len(losers) > 0 else 999,
            'total_return_pct': (capital - 10000) / 10000,
            'total_return_dollars': capital - 10000,
            'final_capital': capital,
            'sharpe': (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0,
            'max_drawdown': max_dd,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }


def generate_year_report(symbol, timeframe, year, strategy_params, output_base_dir):
    """Generate full report for a single year."""

    print(f"\n{'='*80}")
    print(f"GENERATING REPORT FOR {year}")
    print('='*80)

    client = AlpacaClient()
    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year}-12-31', tz='UTC')

    try:
        # Fetch data
        print(f"Fetching {timeframe} data for {symbol} in {year}...")
        data = client.fetch_bars(symbol, timeframe=timeframe, start=start, end=end, limit=10000)

        if len(data) < 100:
            print(f"❌ Insufficient data for {year}")
            return None

        print(f"Data loaded: {len(data)} bars from {data.index.min()} to {data.index.max()}")

        # Initialize strategy
        strategy = TrendBreakoutStrategy(**strategy_params)

        # Generate signals
        print("Generating signals...")
        signals = strategy.generate_signals(data)

        total_signals = (signals != 0).sum()
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()

        print(f"Signals: {total_signals} total ({long_signals} long, {short_signals} short)")

        if total_signals == 0:
            print(f"❌ No signals for {year}")
            return None

        # Run backtest
        print("Running backtest...")
        backtester = DetailedBacktester()
        results = backtester.run(data, signals)

        if results is None:
            print(f"❌ No trades for {year}")
            return None

        # Create year directory
        year_dir = output_base_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Print results to console
        print_results(year, results)

        # Save trades CSV
        trades_file = year_dir / f"trades_{year}.csv"
        results['trades_df'].to_csv(trades_file, index=False)
        print(f"✅ Trades saved: {trades_file}")

        # Generate equity curve chart
        chart_file = year_dir / f"equity_curve_{year}.png"
        plot_equity_curve(year, results, chart_file)
        print(f"✅ Chart saved: {chart_file}")

        # Generate text report
        report_file = year_dir / f"report_{year}.txt"
        save_text_report(year, symbol, timeframe, strategy_params, results, report_file)
        print(f"✅ Report saved: {report_file}")

        return results

    except Exception as e:
        print(f"❌ Error processing {year}: {e}")
        return None


def print_results(year, results):
    """Print results to console."""

    print(f"\n{'='*80}")
    print(f"📊 PERFORMANCE RESULTS - {year}")
    print('='*80)

    print(f"\n💰 RETURNS:")
    print(f"  Starting Capital: $10,000.00")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return_pct']:.2%} (${results['total_return_dollars']:,.2f})")
    print(f"  Gross Profit: ${results['gross_profit']:,.2f}")
    print(f"  Gross Loss: ${results['gross_loss']:,.2f}")

    print(f"\n📈 TRADE STATISTICS:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winners']} ({results['win_rate']:.2%})")
    print(f"  Losing Trades: {results['losers']} ({(1-results['win_rate']):.2%})")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")

    print(f"\n🎯 WIN/LOSS ANALYSIS:")
    print(f"  Average Win: {results['avg_win']:.3%}")
    print(f"  Average Loss: {results['avg_loss']:.3%}")
    print(f"  Win/Loss Ratio: {(results['avg_win']/results['avg_loss'] if results['avg_loss'] > 0 else 999):.2f}:1")
    print(f"  Largest Win: {results['largest_win']:.2%}")
    print(f"  Largest Loss: {results['largest_loss']:.2%}")

    print(f"\n📊 RISK METRICS:")
    print(f"  Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Avg Holding Period: {results['avg_bars_held']:.1f} hours")
    print(f"  Max Consecutive Wins: {results['max_consecutive_wins']}")
    print(f"  Max Consecutive Losses: {results['max_consecutive_losses']}")

    print(f"\n🚪 EXIT REASONS:")
    for reason, count in results['exit_reasons'].items():
        pct = count / results['total_trades'] * 100
        print(f"  {reason}: {count} trades ({pct:.1f}%)")


def plot_equity_curve(year, results, output_file):
    """Generate equity curve chart."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Equity curve
    ax1 = axes[0]
    ax1.plot(results['equity_dates'], results['equity_curve'], linewidth=2, color='green')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.set_title(f'Equity Curve - {year}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capital ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Drawdown
    ax2 = axes[1]
    equity_array = np.array(results['equity_curve'])
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100

    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown (%)', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def save_text_report(year, symbol, timeframe, strategy_params, results, output_file):
    """Save detailed text report."""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"STRATEGY VALIDATION REPORT - {year}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Symbol: {symbol}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Year: {year}\n")
        f.write(f"Strategy: ETH Trend Breakout\n\n")

        f.write("STRATEGY PARAMETERS:\n")
        f.write("-"*80 + "\n")
        for k, v in strategy_params.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("RETURNS:\n")
        f.write(f"  Starting Capital: $10,000.00\n")
        f.write(f"  Final Capital: ${results['final_capital']:,.2f}\n")
        f.write(f"  Total Return: {results['total_return_pct']:.2%} (${results['total_return_dollars']:,.2f})\n")
        f.write(f"  Gross Profit: ${results['gross_profit']:,.2f}\n")
        f.write(f"  Gross Loss: ${results['gross_loss']:,.2f}\n\n")

        f.write("TRADE STATISTICS:\n")
        f.write(f"  Total Trades: {results['total_trades']}\n")
        f.write(f"  Winning Trades: {results['winners']} ({results['win_rate']:.2%})\n")
        f.write(f"  Losing Trades: {results['losers']} ({(1-results['win_rate']):.2%})\n")
        f.write(f"  Profit Factor: {results['profit_factor']:.2f}\n\n")

        f.write("WIN/LOSS ANALYSIS:\n")
        f.write(f"  Average Win: {results['avg_win']:.3%}\n")
        f.write(f"  Average Loss: {results['avg_loss']:.3%}\n")
        f.write(f"  Win/Loss Ratio: {(results['avg_win']/results['avg_loss'] if results['avg_loss'] > 0 else 999):.2f}:1\n")
        f.write(f"  Largest Win: {results['largest_win']:.2%}\n")
        f.write(f"  Largest Loss: {results['largest_loss']:.2%}\n\n")

        f.write("RISK METRICS:\n")
        f.write(f"  Sharpe Ratio: {results['sharpe']:.2f}\n")
        f.write(f"  Max Drawdown: {results['max_drawdown']:.2%}\n")
        f.write(f"  Avg Holding Period: {results['avg_bars_held']:.1f} hours\n")
        f.write(f"  Max Consecutive Wins: {results['max_consecutive_wins']}\n")
        f.write(f"  Max Consecutive Losses: {results['max_consecutive_losses']}\n\n")

        f.write("EXIT REASONS:\n")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            f.write(f"  {reason}: {count} trades ({pct:.1f}%)\n")

        # Top trades
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 WINNING TRADES\n")
        f.write("="*80 + "\n")
        trades_df = results['trades_df']
        top_winners = trades_df.nlargest(10, 'pnl_pct')[['entry_time', 'side', 'pnl_pct', 'pnl_dollars', 'exit_reason']]
        for idx, trade in top_winners.iterrows():
            f.write(f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | {trade['side']:<5} | "
                   f"{trade['pnl_pct']:>7.2%} | ${trade['pnl_dollars']:>8.2f} | {trade['exit_reason']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 LOSING TRADES\n")
        f.write("="*80 + "\n")
        top_losers = trades_df.nsmallest(10, 'pnl_pct')[['entry_time', 'side', 'pnl_pct', 'pnl_dollars', 'exit_reason']]
        for idx, trade in top_losers.iterrows():
            f.write(f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | {trade['side']:<5} | "
                   f"{trade['pnl_pct']:>7.2%} | ${trade['pnl_dollars']:>8.2f} | {trade['exit_reason']}\n")


def main():
    print("="*80)
    print("📊 GENERATING DETAILED YEARLY REPORTS")
    print("Creating validation reports for each year (2020-2025)")
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
    print(f"Strategy: ETH Trend Breakout (Winning Configuration)")
    print(f"\nGenerating reports for years: {YEARS}")

    # Output directory
    output_base_dir = Path("alpha_research/yearly_reports")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Process each year
    all_results = {}

    for year in YEARS:
        result = generate_year_report(SYMBOL, TIMEFRAME, year, strategy_params, output_base_dir)
        all_results[year] = result

    # Summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY - ALL YEARS")
    print('='*80)

    valid_years = sum(1 for r in all_results.values() if r is not None)

    print(f"\nReports generated: {valid_years}/{len(YEARS)} years")
    print(f"\nOutput location: {output_base_dir}/")
    print(f"\nEach year has:")
    print(f"  - trades_YYYY.csv (detailed trade log)")
    print(f"  - equity_curve_YYYY.png (equity chart)")
    print(f"  - report_YYYY.txt (full text report)")

    print("\n" + "="*80)
    print("✅ ALL YEARLY REPORTS GENERATED!")
    print("="*80)
    print(f"\nCheck directory: {output_base_dir}/")


if __name__ == '__main__':
    main()
