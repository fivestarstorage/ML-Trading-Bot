#!/usr/bin/env python3
"""
LIQUIDITY GAP REVERSAL STRATEGY

Unique Edge Discovery: Exploiting Intraday Liquidity Gaps

HYPOTHESIS:
When a stock experiences a sudden liquidity gap (abnormal spread widening or
volume collapse) followed by a sharp price move, there's a high probability
of mean reversion within the next few bars as liquidity returns.

This edge exploits:
1. Temporary liquidity vacuums that cause overextended moves
2. Market maker behavior rushing to fill the gap
3. Algorithmic reversion trading kicking in
4. Stop-loss cascades followed by value buyers

UNIQUE ASPECTS:
- Focuses on microstructure patterns most traders ignore
- Uses volume delta and spread proxy to detect liquidity
- Combines momentum exhaustion with liquidity analysis
- Time-based filters for optimal execution windows

Expected Performance:
- Win Rate: 65-75% (mean reversion edge)
- Risk/Reward: 1:1.5 to 1:2
- Trade Frequency: 5-15 per day on liquid stocks
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class LiquidityGapStrategy:
    """
    Liquidity Gap Reversal Strategy

    Detects temporary liquidity gaps and trades the reversion.
    """

    def __init__(
        self,
        volume_threshold=2.0,  # Z-score for abnormal volume
        price_move_threshold=0.003,  # 0.3% minimum move
        rsi_oversold=30,
        rsi_overbought=70,
        lookback_period=20,
        hold_periods=10  # Hold for 10 bars (50 minutes on 5min data)
    ):
        self.volume_threshold = volume_threshold
        self.price_move_threshold = price_move_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.lookback_period = lookback_period
        self.hold_periods = hold_periods

    def calculate_features(self, data):
        """Calculate all required features for the strategy."""
        df = data.copy()

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(self.lookback_period).mean()
        df['volume_std'] = df['volume'].rolling(self.lookback_period).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (df['volume_std'] + 1e-8)

        # Liquidity proxy: Estimate spread using high-low range
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_ma'] = df['spread_proxy'].rolling(self.lookback_period).mean()
        df['spread_ratio'] = df['spread_proxy'] / (df['spread_ma'] + 1e-8)

        # Volume delta (buy/sell pressure proxy)
        df['price_change'] = df['close'].diff()
        df['volume_delta'] = df['volume'] * np.sign(df['price_change'])
        df['cumulative_delta'] = df['volume_delta'].rolling(self.lookback_period).sum()
        df['delta_normalized'] = df['cumulative_delta'] / (df['volume_ma'] * self.lookback_period + 1e-8)

        # RSI for overbought/oversold
        df['rsi'] = self._calculate_rsi(df['close'], period=14)

        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Liquidity gap detection
        # Gap = Low volume + Wide spread + Big move
        df['liquidity_gap'] = (
            (df['volume_zscore'] < -1.0) &  # Volume collapse
            (df['spread_ratio'] > 1.2) &     # Spread widening
            (abs(df['returns']) > self.price_move_threshold)  # Significant move
        ).astype(int)

        # Debug: Print stats
        logger.info(f"Liquidity gaps detected: {df['liquidity_gap'].sum()}")
        logger.info(f"RSI range: {df['rsi'].min():.1f} to {df['rsi'].max():.1f}")
        logger.info(f"Volume z-score range: {df['volume_zscore'].min():.1f} to {df['volume_zscore'].max():.1f}")

        return df

    def generate_signals(self, data):
        """
        Generate trading signals based on liquidity gap reversals.

        Signal Logic:
        LONG:
        - Liquidity gap detected
        - Price dropped sharply (oversold)
        - RSI < 30 or price momentum very negative
        - Volume delta negative (selling exhaustion)

        SHORT:
        - Liquidity gap detected
        - Price rose sharply (overbought)
        - RSI > 70 or price momentum very positive
        - Volume delta positive (buying exhaustion)
        """
        df = self.calculate_features(data)

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Long signals (buy the dip on liquidity gap)
        long_conditions = (
            (df['liquidity_gap'] == 1) &
            (df['returns'] < -self.price_move_threshold) &  # Price dropped
            (
                (df['rsi'] < self.rsi_oversold) |  # Oversold
                (df['momentum_5'] < -0.01)  # Strong negative momentum
            ) &
            (df['delta_normalized'] < -0.5)  # Selling pressure exhausted
        )

        # Short signals (sell the rip on liquidity gap)
        short_conditions = (
            (df['liquidity_gap'] == 1) &
            (df['returns'] > self.price_move_threshold) &  # Price rose
            (
                (df['rsi'] > self.rsi_overbought) |  # Overbought
                (df['momentum_5'] > 0.01)  # Strong positive momentum
            ) &
            (df['delta_normalized'] > 0.5)  # Buying pressure exhausted
        )

        # Apply time-based filters (avoid first/last hour of trading)
        df['hour'] = df.index.hour
        tradeable_hours = ((df['hour'] >= 10) & (df['hour'] <= 15))

        long_conditions = long_conditions & tradeable_hours
        short_conditions = short_conditions & tradeable_hours

        # Set signals
        signals[long_conditions] = 1
        signals[short_conditions] = -1

        # Store for analysis
        self.signal_data = df
        self.raw_signals = signals.copy()

        return signals

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi


class LiquidityGapBacktester:
    """Backtester for Liquidity Gap Strategy with position management."""

    def __init__(
        self,
        commission=0.0005,  # 0.05% per trade
        slippage=0.0002,    # 0.02% slippage
        take_profit=0.015,  # 1.5% TP
        stop_loss=0.01,     # 1% SL
        hold_periods=10
    ):
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.hold_periods = hold_periods

    def run(self, data, signals):
        """Run backtest with proper position management."""
        df = data.copy()
        trades = []

        in_position = False
        position_side = 0
        entry_price = 0
        entry_idx = None
        bars_held = 0

        for i in range(len(df)):
            current_idx = df.index[i]
            current_price = df['close'].iloc[i]
            signal = signals.iloc[i]

            # Check if we should enter
            if not in_position and signal != 0:
                in_position = True
                position_side = signal
                entry_price = current_price * (1 + self.slippage * signal)  # Adverse slippage
                entry_idx = i
                bars_held = 0
                continue

            # Manage open position
            if in_position:
                bars_held += 1

                # Calculate P&L
                if position_side == 1:  # Long
                    pnl_pct = (current_price - entry_price) / entry_price
                elif position_side == -1:  # Short
                    pnl_pct = (entry_price - current_price) / entry_price
                else:
                    pnl_pct = 0

                # Exit conditions
                exit_reason = None
                exit_price = current_price

                # Take profit hit
                if pnl_pct >= self.take_profit:
                    exit_reason = 'take_profit'
                    exit_price = current_price * (1 - self.slippage * position_side)

                # Stop loss hit
                elif pnl_pct <= -self.stop_loss:
                    exit_reason = 'stop_loss'
                    exit_price = current_price * (1 - self.slippage * position_side)

                # Time-based exit
                elif bars_held >= self.hold_periods:
                    exit_reason = 'time_exit'
                    exit_price = current_price * (1 - self.slippage * position_side)

                # Exit if conditions met
                if exit_reason:
                    # Calculate final P&L
                    if position_side == 1:
                        final_pnl = (exit_price - entry_price) / entry_price
                    else:
                        final_pnl = (entry_price - exit_price) / entry_price

                    # Subtract commission
                    final_pnl -= (self.commission * 2)  # Entry + exit

                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': current_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': position_side,
                        'pnl_pct': final_pnl,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held
                    })

                    # Reset position
                    in_position = False
                    position_side = 0
                    entry_price = 0
                    entry_idx = None
                    bars_held = 0

        # Calculate performance metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'trades': []
            }

        trades_df = pd.DataFrame(trades)

        winning_trades = trades_df[trades_df['pnl_pct'] > 0]
        losing_trades = trades_df[trades_df['pnl_pct'] < 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl_pct'].mean()) if len(losing_trades) > 0 else 0

        gross_profit = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        total_return = trades_df['pnl_pct'].sum()

        # Sharpe ratio (annualized)
        if len(trades_df) > 1:
            returns_std = trades_df['pnl_pct'].std()
            sharpe = (trades_df['pnl_pct'].mean() / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe = 0

        results = {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'trades': trades_df,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }

        return results


def fetch_stock_data(symbol, days_back=60):
    """Fetch stock data using Alpaca (daily if 5min not available)."""
    logger.info(f"Fetching data for {symbol}")

    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=days_back)

    try:
        # Try 5Min data first
        data = client.fetch_bars(
            symbol=symbol,
            timeframe="5Min",
            start=start,
            end=end,
            limit=10000
        )
        logger.info(f"Fetched {len(data)} 5-minute bars")
        return data, "5Min"
    except Exception as e:
        logger.warning(f"5Min data failed: {e}. Trying 1Hour...")
        try:
            data = client.fetch_bars(
                symbol=symbol,
                timeframe="1Hour",
                start=start,
                end=end,
                limit=10000
            )
            logger.info(f"Fetched {len(data)} 1-hour bars")
            return data, "1Hour"
        except Exception as e2:
            logger.warning(f"1Hour failed: {e2}. Trying 1Day...")
            # Fall back to daily data (usually free)
            data = client.fetch_bars(
                symbol=symbol,
                timeframe="1Day",
                start=start,
                end=end,
                limit=500
            )
            logger.info(f"Fetched {len(data)} daily bars")
            return data, "1Day"


def main():
    """Run Liquidity Gap Strategy backtest."""
    print("="*80)
    print("LIQUIDITY GAP REVERSAL STRATEGY")
    print("Exploiting Temporary Liquidity Gaps for Mean Reversion Edge")
    print("="*80)

    # Configuration
    # Try crypto first (free on Alpaca), then fallback to stocks
    SYMBOL = "BTC/USD"  # Can also try: ETH/USD, SPY, QQQ, AAPL, TSLA
    DAYS_BACK = 90

    # Fetch data
    print(f"\nFetching data for {SYMBOL}...")
    try:
        data, timeframe = fetch_stock_data(SYMBOL, DAYS_BACK)
        print(f"Data range: {data.index.min()} to {data.index.max()}")
        print(f"Timeframe: {timeframe}")
        print(f"Total bars: {len(data)}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("\nNote: This strategy requires Alpaca market data subscription.")
        print("Alternatively, use crypto (BTC/USD, ETH/USD) which is free.")
        return

    # Initialize strategy
    print("\nInitializing Liquidity Gap Strategy...")
    # More sensitive parameters for crypto
    strategy = LiquidityGapStrategy(
        volume_threshold=1.5,  # Lower threshold for crypto
        price_move_threshold=0.002,  # 0.2% for crypto volatility
        rsi_oversold=35,  # Less extreme
        rsi_overbought=65,  # Less extreme
        lookback_period=20,
        hold_periods=10
    )

    # Generate signals
    print("Generating signals...")
    signals = strategy.generate_signals(data)
    print(f"Total signals generated: {(signals != 0).sum()}")
    print(f"Long signals: {(signals == 1).sum()}")
    print(f"Short signals: {(signals == -1).sum()}")

    # Backtest
    print("\nRunning backtest...")
    backtester = LiquidityGapBacktester(
        commission=0.0005,
        slippage=0.0002,
        take_profit=0.015,
        stop_loss=0.01,
        hold_periods=10
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
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    if 'avg_bars_held' in results:
        print(f"Avg Bars Held: {results['avg_bars_held']:.1f}")

    if 'exit_reasons' in results and results['exit_reasons']:
        print("\nExit Reasons:")
        for reason, count in results['exit_reasons'].items():
            print(f"  {reason}: {count}")

    # Verdict
    print("\n" + "="*80)
    if results['win_rate'] >= 0.6 and results['profit_factor'] > 1.3:
        print("✅ STRATEGY SHOWS STRONG EDGE!")
        print("This strategy has a profitable edge and can be deployed.")
    elif results['win_rate'] >= 0.5 and results['profit_factor'] > 1.0:
        print("⚠️  STRATEGY SHOWS MODERATE EDGE")
        print("Needs optimization but shows promise.")
    else:
        print("❌ STRATEGY NEEDS IMPROVEMENT")
        print("Edge not strong enough for live trading.")

    # Save results
    if results['total_trades'] > 0:
        output_dir = Path("alpha_research/backtests")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"liquidity_gap_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades'].to_csv(results_file, index=False)
        print(f"\nTrades saved to: {results_file}")


if __name__ == '__main__':
    main()
