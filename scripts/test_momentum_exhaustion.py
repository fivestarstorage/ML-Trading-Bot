#!/usr/bin/env python3
"""
Test Momentum Exhaustion Strategy

Backtest the momentum exhaustion strategy to validate the edge
before integrating into the main WFA framework.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vwap_reversion_strategy import VWAPReversionStrategy
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class SimpleBacktester:
    """Simple backtester with realistic position management."""

    def __init__(
        self,
        initial_capital=10000,
        position_size_pct=1.0,  # 100% of capital per trade
        commission=0.0005,  # 0.05%
        slippage=0.0002,  # 0.02%
        take_profit=0.015,  # 1.5%
        stop_loss=0.01,  # 1%
        max_hold_bars=20  # Max 20 bars (~100 minutes on 5min)
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_bars = max_hold_bars

    def run(self, data, signals):
        """Run backtest with position management."""
        df = data.copy()
        df['signal'] = signals

        trades = []
        capital = self.initial_capital
        equity_curve = [capital]

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        entry_capital = 0
        bars_held = 0

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_signal = df['signal'].iloc[i]

            # Entry logic
            if not in_position and current_signal != 0:
                position_side = current_signal
                # Apply adverse slippage on entry
                entry_price = current_price * (1 + self.slippage * position_side)
                entry_idx = i
                entry_capital = capital
                in_position = True
                bars_held = 0
                continue

            # Position management
            if in_position:
                bars_held += 1

                # Calculate unrealized P&L
                if position_side == 1:  # Long
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Short
                    pnl_pct = (entry_price - current_price) / entry_price

                # Exit conditions
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

                # Time-based exit
                elif bars_held >= self.max_hold_bars:
                    exit_signal = True
                    exit_reason = 'Time'

                # Execute exit
                if exit_signal:
                    # Apply slippage on exit
                    exit_price = current_price * (1 - self.slippage * position_side)

                    # Calculate realized P&L
                    if position_side == 1:
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    # Apply costs
                    pnl_pct -= (self.commission * 2)  # Round-trip commission

                    # Update capital
                    pnl_dollars = entry_capital * pnl_pct
                    capital += pnl_dollars

                    # Record trade
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'side': 'LONG' if position_side == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason
                    })

                    # Reset position
                    in_position = False
                    position_side = 0

            # Update equity curve
            equity_curve.append(capital)

        # Calculate metrics
        if len(trades) == 0:
            return {
                'trades': [],
                'total_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'total_return_dollars': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'equity_curve': equity_curve
            }

        trades_df = pd.DataFrame(trades)

        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]

        win_rate = len(winners) / len(trades_df)
        total_return_pct = (capital - self.initial_capital) / self.initial_capital
        total_return_dollars = capital - self.initial_capital

        avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0

        gross_profit = winners['pnl_dollars'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_dollars'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio
        returns = trades_df['pnl_pct'].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        return {
            'trades': trades_df,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'total_return_dollars': total_return_dollars,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }


def fetch_data(symbol, days_back=90):
    """Fetch data from Alpaca."""
    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=days_back)

    try:
        # Try 5Min first
        data = client.fetch_bars(symbol, timeframe="5Min", start=start, end=end, limit=10000)
        logger.info(f"Fetched {len(data)} 5-minute bars for {symbol}")
        return data, "5Min"
    except:
        try:
            # Fallback to 1Hour
            data = client.fetch_bars(symbol, timeframe="1Hour", start=start, end=end, limit=5000)
            logger.info(f"Fetched {len(data)} 1-hour bars for {symbol}")
            return data, "1Hour"
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise


def main():
    print("="*80)
    print("VWAP MEAN REVERSION STRATEGY BACKTEST")
    print("Testing VWAP Deviation + Volume Confirmation Edge")
    print("="*80)

    # Choose symbol (crypto is free on Alpaca)
    SYMBOL = "BTC/USD"  # Can also try: ETH/USD, SPY (if you have subscription)
    DAYS_BACK = 90

    # Fetch data
    print(f"\nFetching {DAYS_BACK} days of data for {SYMBOL}...")
    try:
        data, timeframe = fetch_data(SYMBOL, DAYS_BACK)
        print(f"Data range: {data.index.min()} to {data.index.max()}")
        print(f"Timeframe: {timeframe}")
        print(f"Total bars: {len(data)}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Requires Alpaca API keys in .env file")
        return

    # Initialize strategy
    print("\nInitializing VWAP Reversion Strategy...")
    strategy = VWAPReversionStrategy(
        vwap_threshold=0.008,  # 0.8% deviation
        volume_threshold=1.5,   # 1.5x average volume
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=40,
        rsi_overbought=60
    )

    # Generate signals
    print("Generating trading signals...")
    signals = strategy.generate_signals(data)

    total_signals = (signals != 0).sum()
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()

    print(f"\nSignals Generated:")
    print(f"  Total: {total_signals}")
    print(f"  Long: {long_signals}")
    print(f"  Short: {short_signals}")

    if total_signals == 0:
        print("\n⚠️  No signals generated. Try adjusting parameters.")
        return

    # Run backtest
    print("\nRunning backtest...")
    backtester = SimpleBacktester(
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        take_profit=0.015,
        stop_loss=0.01,
        max_hold_bars=20
    )

    results = backtester.run(data, signals)

    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Average Win: {results['avg_win']:.3%}")
    print(f"Average Loss: {results['avg_loss']:.3%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2%} (${results['total_return_dollars']:.2f})")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

    if 'exit_reasons' in results:
        print("\nExit Breakdown:")
        for reason, count in results['exit_reasons'].items():
            print(f"  {reason}: {count} ({count/results['total_trades']*100:.1f}%)")

    # Analysis
    print("\n" + "="*80)
    if results['win_rate'] >= 0.60 and results['profit_factor'] > 1.5:
        print("✅ STRONG EDGE DETECTED!")
        print("This strategy shows a profitable edge suitable for trading.")
    elif results['win_rate'] >= 0.55 and results['profit_factor'] > 1.2:
        print("⚠️  MODERATE EDGE")
        print("Strategy shows promise but may need optimization.")
    else:
        print("❌ WEAK EDGE")
        print("Strategy needs significant improvement.")

    # Save results
    if results['total_trades'] > 0:
        output_dir = Path("alpha_research/backtests")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"vwap_reversion_{SYMBOL.replace('/', '_')}_{timestamp}.csv"

        results['trades'].to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

    # Plot equity curve
    if len(results['equity_curve']) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f'VWAP Reversion Strategy - {SYMBOL}')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)

        chart_file = output_dir / f"equity_curve_{timestamp}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        print(f"Equity curve saved to: {chart_file}")


if __name__ == '__main__':
    main()
