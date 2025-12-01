#!/usr/bin/env python3
"""
Yearly Validation - Test Strategy Across All Years (2020-2025)

Validates strategy performance year-by-year to avoid overfitting.
Generates WFA-style graphs and metrics.
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

from src.high_winrate_strategy import HighWinRateStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class ConservativeBacktester:
    """Backtester optimized for high win rate."""

    def __init__(
        self,
        commission=0.0005,
        slippage=0.0002,
        take_profit=0.025,  # 2.5% (conservative)
        stop_loss=0.015,    # 1.5%
        max_hold_bars=30
    ):
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest."""
        df = data.copy()
        trades = []
        capital = 10000
        equity_curve = []
        equity_dates = []

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0

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

                # Exit conditions
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
                        'entry_time': df.index[entry_idx],
                        'exit_time': current_time,
                        'side': 'LONG' if position_side == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
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

        return {
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'equity_dates': equity_dates,
            'total_trades': len(trades_df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades_df),
            'profit_factor': (winners['pnl_pct'].sum() / abs(losers['pnl_pct'].sum())) if len(losers) > 0 else 999,
            'total_return': (capital - 10000) / 10000,
            'final_capital': capital,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0,
            'sharpe': (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
        }


def test_year(symbol, timeframe, year, strategy_params, backtest_params):
    """Test strategy on a specific year."""
    print(f"\n  Testing {year}...", end=' ')

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
        strategy = HighWinRateStrategy(**strategy_params)

        # Generate signals
        signals = strategy.generate_signals(data)

        if (signals != 0).sum() < 5:
            print(f"❌ Too few signals ({(signals != 0).sum()})")
            return None

        # Backtest
        backtester = ConservativeBacktester(**backtest_params)
        results = backtester.run(data, signals)

        if results is None:
            print("❌ No trades")
            return None

        # Print result
        wr = results['win_rate']
        ret = results['total_return']
        trades = results['total_trades']

        status = "✅" if wr >= 0.70 else "⚠️ " if wr >= 0.60 else "❌"
        print(f"{status} {trades} trades, {wr:.1%} WR, {ret:+.1%} return")

        results['year'] = year
        results['data_points'] = len(data)

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def plot_wfa_style_charts(all_results, output_dir):
    """Generate WFA-style charts."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. Equity Curve by Year
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (year, result) in enumerate(all_results.items()):
        if result and 'equity_curve' in result:
            dates = result['equity_dates']
            equity = result['equity_curve']
            ax1.plot(dates, equity, label=f'{year}', color=colors[idx], linewidth=2)

    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.set_title('Equity Curve by Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capital ($)', fontsize=12)
    ax1.legend(loc='best', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Win Rate by Year
    ax2 = fig.add_subplot(gs[1, 0])
    years = []
    win_rates = []

    for year, result in sorted(all_results.items()):
        if result:
            years.append(year)
            win_rates.append(result['win_rate'] * 100)

    bars = ax2.bar(years, win_rates, color=['green' if wr >= 70 else 'orange' if wr >= 60 else 'red' for wr in win_rates])
    ax2.axhline(y=70, color='green', linestyle='--', label='70% Target', linewidth=2)
    ax2.axhline(y=60, color='orange', linestyle='--', label='60% Threshold', alpha=0.5)
    ax2.set_title('Win Rate by Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Total Return by Year
    ax3 = fig.add_subplot(gs[1, 1])
    returns = []

    for year, result in sorted(all_results.items()):
        if result:
            returns.append(result['total_return'] * 100)

    bars = ax3.bar(years, returns, color=['green' if r > 0 else 'red' for r in returns])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Total Return by Year', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Return (%)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        y_pos = height + 2 if height > 0 else height - 5
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

    # 4. Trade Count by Year
    ax4 = fig.add_subplot(gs[2, 0])
    trade_counts = []

    for year, result in sorted(all_results.items()):
        if result:
            trade_counts.append(result['total_trades'])

    ax4.bar(years, trade_counts, color='steelblue')
    ax4.set_title('Number of Trades by Year', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Trades', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    for i, (year, count) in enumerate(zip(years, trade_counts)):
        ax4.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')

    # 5. Profit Factor by Year
    ax5 = fig.add_subplot(gs[2, 1])
    profit_factors = []

    for year, result in sorted(all_results.items()):
        if result:
            pf = min(result['profit_factor'], 10)  # Cap at 10 for visualization
            profit_factors.append(pf)

    bars = ax5.bar(years, profit_factors, color=['green' if pf >= 2 else 'orange' if pf >= 1.5 else 'red' for pf in profit_factors])
    ax5.axhline(y=2.0, color='green', linestyle='--', label='2.0 Target', linewidth=2)
    ax5.axhline(y=1.0, color='red', linestyle='--', label='Breakeven', alpha=0.5)
    ax5.set_title('Profit Factor by Year', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Profit Factor', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, pf in zip(bars, profit_factors):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{pf:.2f}', ha='center', va='bottom', fontweight='bold')

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    table_data.append(['Year', 'Trades', 'Win Rate', 'Profit Factor', 'Return', 'Sharpe', 'Avg Win', 'Avg Loss'])

    for year, result in sorted(all_results.items()):
        if result:
            row = [
                str(year),
                str(result['total_trades']),
                f"{result['win_rate']:.1%}",
                f"{min(result['profit_factor'], 99):.2f}",
                f"{result['total_return']:+.1%}",
                f"{result['sharpe']:.2f}",
                f"{result['avg_win']:.2%}",
                f"{result['avg_loss']:.2%}"
            ]
            table_data.append(row)

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.08, 0.12, 0.14, 0.10, 0.10, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code win rates
    for row_idx in range(1, len(table_data)):
        wr_val = float(table_data[row_idx][2].strip('%'))
        if wr_val >= 70:
            table[(row_idx, 2)].set_facecolor('#90EE90')
        elif wr_val >= 60:
            table[(row_idx, 2)].set_facecolor('#FFD700')
        else:
            table[(row_idx, 2)].set_facecolor('#FFB6C1')

    plt.suptitle('Strategy Performance: Year-by-Year Analysis (2020-2025)',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_file = output_dir / f"yearly_validation_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"\nCharts saved to: {chart_file}")

    return chart_file


def main():
    print("="*80)
    print("📊 YEARLY VALIDATION - AVOIDING OVERFITTING")
    print("Testing Strategy Across All Years (2020-2025)")
    print("="*80)

    # Configuration
    SYMBOL = "ETH/USD"
    TIMEFRAME = "1Hour"
    YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

    # Strategy parameters (relaxed for signals)
    strategy_params = {
        'rsi_period': 14,
        'rsi_extreme_oversold': 30,  # Relaxed from 25
        'rsi_extreme_overbought': 70,  # Relaxed from 75
        'bb_period': 20,
        'bb_std': 2.0,  # Relaxed from 2.5
        'volume_spike_threshold': 1.5,  # Relaxed from 2.0
        'wick_ratio_threshold': 0.5,  # Relaxed from 0.6
        'support_resistance_lookback': 50,
        'support_resistance_tolerance': 0.005  # Relaxed from 0.002
    }

    # Backtest parameters
    backtest_params = {
        'commission': 0.0005,
        'slippage': 0.0002,
        'take_profit': 0.025,  # 2.5%
        'stop_loss': 0.015,     # 1.5%
        'max_hold_bars': 30
    }

    print(f"\nSymbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"\nStrategy: High Win Rate (Ultra-Selective Mean Reversion)")
    print(f"Target: >70% Win Rate")

    # Test each year
    print(f"\n{'='*80}")
    print("TESTING EACH YEAR INDIVIDUALLY")
    print('='*80)

    all_results = {}

    for year in YEARS:
        result = test_year(SYMBOL, TIMEFRAME, year, strategy_params, backtest_params)
        all_results[year] = result

    # Summary
    print(f"\n{'='*80}")
    print("📈 SUMMARY - YEAR-BY-YEAR PERFORMANCE")
    print('='*80)

    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if len(valid_results) == 0:
        print("\n❌ No valid results generated!")
        print("Try adjusting parameters or using different data.")
        return

    # Table
    print(f"\n{'Year':<6}{'Trades':<10}{'Win Rate':<12}{'P.Factor':<12}{'Return':<12}{'Sharpe':<10}{'Status'}")
    print("-"*80)

    years_above_70 = 0

    for year in YEARS:
        result = all_results[year]
        if result:
            wr = result['win_rate']
            status = "✅ PASS" if wr >= 0.70 else "⚠️  CLOSE" if wr >= 0.60 else "❌ FAIL"

            if wr >= 0.70:
                years_above_70 += 1

            print(f"{year:<6}{result['total_trades']:<10}{result['win_rate']:<12.1%}"
                  f"{result['profit_factor']:<12.2f}{result['total_return']:<12.1%}"
                  f"{result['sharpe']:<10.2f}{status}")
        else:
            print(f"{year:<6}{'N/A':<10}{'N/A':<12}{'N/A':<12}{'N/A':<12}{'N/A':<10}❌ NO DATA")

    # Overall assessment
    print(f"\n{'='*80}")
    print("🎯 OVERALL ASSESSMENT")
    print('='*80)

    print(f"\nYears with >70% Win Rate: {years_above_70}/{len(YEARS)}")
    print(f"Validation Rate: {years_above_70/len(YEARS)*100:.1f}%")

    if years_above_70 >= 5:
        print("\n✅ EXCELLENT! Strategy is robust across years!")
        print("No overfitting detected - consistent performance.")
    elif years_above_70 >= 4:
        print("\n✅ GOOD! Strategy performs well most years.")
        print("Minor variance acceptable.")
    elif years_above_70 >= 3:
        print("\n⚠️  MODERATE. Strategy works but inconsistent.")
        print("May need further optimization.")
    else:
        print("\n❌ NEEDS IMPROVEMENT.")
        print("Strategy not meeting 70% win rate target consistently.")

    # Generate charts
    print(f"\n{'='*80}")
    print("📊 GENERATING WFA-STYLE CHARTS")
    print('='*80)

    output_dir = Path("alpha_research/yearly_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_wfa_style_charts(valid_results, output_dir)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_dir / f"yearly_summary_{timestamp}.csv"

    summary_data = []
    for year, result in all_results.items():
        if result:
            summary_data.append({
                'year': year,
                'trades': result['total_trades'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'total_return': result['total_return'],
                'sharpe': result['sharpe'],
                'avg_win': result['avg_win'],
                'avg_loss': result['avg_loss']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
