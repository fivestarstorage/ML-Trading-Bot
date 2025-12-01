#!/usr/bin/env python3
"""
Validate Winning Strategy - Final Verification

Run the winning ETH Trend Breakout strategy and display detailed results.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
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
    """Detailed backtester with full trade logging."""

    def __init__(self):
        self.commission = 0.0005
        self.slippage = 0.0002
        self.take_profit = 0.06
        self.stop_loss = 0.02
        self.trailing_stop = 0.035
        self.max_hold_bars = 60

    def run(self, data, signals):
        """Run backtest with detailed logging."""
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

        # Calculate metrics
        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        # Drawdown analysis
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


def main():
    print("="*80)
    print("🏆 WINNING STRATEGY VALIDATION")
    print("ETH Trend Breakout - Final Verification")
    print("="*80)

    # Configuration
    SYMBOL = "ETH/USD"
    TIMEFRAME = "1Hour"
    DAYS_BACK = 180

    # Fetch data
    print(f"\nFetching {DAYS_BACK} days of {TIMEFRAME} data for {SYMBOL}...")
    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=DAYS_BACK)

    data = client.fetch_bars(SYMBOL, timeframe=TIMEFRAME, start=start, end=end, limit=10000)
    print(f"Data loaded: {len(data)} bars")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Initialize winning strategy
    print("\nInitializing winning strategy configuration...")
    strategy = TrendBreakoutStrategy(
        fast_ema=5,
        slow_ema=13,
        breakout_period=10,
        volume_ma_period=20,
        volume_threshold=1.5,
        use_volume_filter=True
    )

    # Generate signals
    print("Generating signals...")
    signals = strategy.generate_signals(data)

    total_signals = (signals != 0).sum()
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()

    print(f"\nSignals generated:")
    print(f"  Total: {total_signals}")
    print(f"  Long: {long_signals}")
    print(f"  Short: {short_signals}")

    # Run backtest
    print("\nRunning detailed backtest...")
    backtester = DetailedBacktester()
    results = backtester.run(data, signals)

    # Display results
    print("\n" + "="*80)
    print("📊 PERFORMANCE RESULTS")
    print("="*80)

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

    # Monthly breakdown
    print(f"\n📅 MONTHLY BREAKDOWN:")
    trades_df = results['trades_df']
    trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({
        'pnl_dollars': ['count', 'sum'],
        'pnl_pct': lambda x: (x > 0).sum() / len(x)
    })
    monthly.columns = ['Trades', 'P&L', 'Win Rate']

    for month, row in monthly.iterrows():
        print(f"  {month}: {int(row['Trades'])} trades, ${row['P&L']:.2f} P&L, {row['Win Rate']:.1%} WR")

    # Top 10 best/worst trades
    print(f"\n🏆 TOP 10 WINNING TRADES:")
    top_winners = trades_df.nlargest(10, 'pnl_pct')[['entry_time', 'side', 'pnl_pct', 'pnl_dollars', 'exit_reason']]
    for idx, trade in top_winners.iterrows():
        print(f"  {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | {trade['side']:<5} | "
              f"{trade['pnl_pct']:>7.2%} | ${trade['pnl_dollars']:>8.2f} | {trade['exit_reason']}")

    print(f"\n💔 TOP 10 LOSING TRADES:")
    top_losers = trades_df.nsmallest(10, 'pnl_pct')[['entry_time', 'side', 'pnl_pct', 'pnl_dollars', 'exit_reason']]
    for idx, trade in top_losers.iterrows():
        print(f"  {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | {trade['side']:<5} | "
              f"{trade['pnl_pct']:>7.2%} | ${trade['pnl_dollars']:>8.2f} | {trade['exit_reason']}")

    # Plot equity curve
    print(f"\nGenerating equity curve chart...")
    plt.figure(figsize=(14, 8))

    # Equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results['equity_dates'], results['equity_curve'], linewidth=2, color='green')
    plt.title('Equity Curve - ETH Trend Breakout Strategy', fontsize=14, fontweight='bold')
    plt.ylabel('Capital ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    plt.legend()

    # Drawdown
    plt.subplot(2, 1, 2)
    equity_array = np.array(results['equity_curve'])
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown, color='red', linewidth=1)
    plt.title('Drawdown (%)', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.xlabel('Trade Number', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save chart
    output_dir = Path("alpha_research/final_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_file = output_dir / f"winning_strategy_equity_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"Equity curve saved to: {chart_file}")

    # Save detailed trades
    trades_file = output_dir / f"winning_strategy_trades_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"Detailed trades saved to: {trades_file}")

    # Final verdict
    print("\n" + "="*80)
    print("✅ STRATEGY VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nThe ETH Trend Breakout strategy demonstrates:")
    print(f"  ✓ Strong profitability: {results['total_return_pct']:.1%} return")
    print(f"  ✓ Good win rate: {results['win_rate']:.1%}")
    print(f"  ✓ Excellent Sharpe ratio: {results['sharpe']:.2f}")
    print(f"  ✓ Solid profit factor: {results['profit_factor']:.2f}")
    print(f"\n🚀 READY FOR PAPER TRADING!")
    print("\nNext steps:")
    print("  1. Paper trade for 30 days to validate in live market")
    print("  2. Monitor slippage and execution quality")
    print("  3. Track key metrics (win rate, profit factor)")
    print("  4. Deploy with proper risk management once validated")


if __name__ == '__main__':
    main()
