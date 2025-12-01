#!/usr/bin/env python3
"""
Optimize Trend Strategy - Fine-tune parameters to find winning edge

Aggressively optimize parameters until we find a profitable configuration.
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
from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()


class AdvancedBacktester:
    """Advanced backtester."""

    def __init__(self, commission=0.0005, slippage=0.0002, take_profit=0.03,
                 stop_loss=0.015, trailing_stop=0.02, max_hold_bars=50, use_trailing=True):
        self.commission = commission
        self.slippage = slippage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_bars = max_hold_bars
        self.use_trailing = use_trailing

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
                elif self.use_trailing and max_profit > 0.01:
                    if (max_profit - pnl_pct) >= self.trailing_stop:
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
                        'pnl_dollars': pnl_dollars,
                        'capital': capital,
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
            'win_rate': len(winners) / len(trades_df),
            'profit_factor': (winners['pnl_pct'].sum() / abs(losers['pnl_pct'].sum())) if len(losers) > 0 else 999,
            'total_return': (capital - 10000) / 10000,
            'sharpe': (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': abs(losers['pnl_pct'].mean()) if len(losers) > 0 else 0
        }


def main():
    print("="*80)
    print("TREND STRATEGY OPTIMIZATION - AGGRESSIVE PARAMETER SEARCH")
    print("="*80)

    # Test both BTC and ETH
    SYMBOLS = ["BTC/USD", "ETH/USD"]
    TIMEFRAME = "1Hour"
    DAYS_BACK = 180

    client = AlpacaClient()
    end = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=DAYS_BACK)

    all_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"OPTIMIZING ON: {symbol}")
        print('='*80)

        # Fetch data
        try:
            data = client.fetch_bars(symbol, timeframe=TIMEFRAME, start=start, end=end, limit=10000)
            print(f"Fetched {len(data)} bars from {data.index.min()} to {data.index.max()}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

        # Parameter grid - more aggressive
        param_combinations = []

        # EMA combinations
        ema_combos = [(5, 13), (9, 21), (12, 26), (20, 50)]

        # Breakout periods
        breakout_periods = [10, 15, 20, 30]

        # Volume settings
        volume_settings = [
            (False, 1.0),
            (True, 1.2),
            (True, 1.5)
        ]

        # Risk management
        risk_configs = [
            {'tp': 0.05, 'sl': 0.02, 'trail': 0.03, 'use_trail': True},   # 5%/2% R:R 2.5:1
            {'tp': 0.06, 'sl': 0.02, 'trail': 0.035, 'use_trail': True},  # 6%/2% R:R 3:1
            {'tp': 0.04, 'sl': 0.015, 'trail': 0.025, 'use_trail': True}, # 4%/1.5% R:R 2.67:1
            {'tp': 0.08, 'sl': 0.025, 'trail': 0.04, 'use_trail': True},  # 8%/2.5% R:R 3.2:1
        ]

        total_combos = len(ema_combos) * len(breakout_periods) * len(volume_settings) * len(risk_configs)
        print(f"\nTesting {total_combos} parameter combinations...")

        test_num = 0

        for (fast_ema, slow_ema), breakout_period, (use_vol, vol_thresh), risk in product(
            ema_combos, breakout_periods, volume_settings, risk_configs
        ):
            test_num += 1

            if test_num % 20 == 0:
                print(f"  Progress: {test_num}/{total_combos}...")

            # Strategy
            strategy = TrendBreakoutStrategy(
                fast_ema=fast_ema,
                slow_ema=slow_ema,
                breakout_period=breakout_period,
                volume_threshold=vol_thresh,
                use_volume_filter=use_vol
            )

            signals = strategy.generate_signals(data)

            if (signals != 0).sum() < 20:  # Need at least 20 signals
                continue

            # Backtest
            backtester = AdvancedBacktester(
                take_profit=risk['tp'],
                stop_loss=risk['sl'],
                trailing_stop=risk['trail'],
                use_trailing=risk['use_trail'],
                max_hold_bars=60
            )

            result = backtester.run(data, signals)

            if result is None:
                continue

            # Store
            result['symbol'] = symbol
            result['fast_ema'] = fast_ema
            result['slow_ema'] = slow_ema
            result['breakout_period'] = breakout_period
            result['use_volume'] = use_vol
            result['volume_threshold'] = vol_thresh
            result['take_profit'] = risk['tp']
            result['stop_loss'] = risk['sl']
            result['trailing_stop'] = risk['trail']

            all_results.append(result)

    if len(all_results) == 0:
        print("\n❌ No results generated!")
        return

    # Sort by total return
    all_results.sort(key=lambda x: x['total_return'], reverse=True)

    print("\n" + "="*80)
    print("TOP 20 CONFIGURATIONS BY TOTAL RETURN")
    print("="*80)
    print(f"\n{'#':<4}{'Symbol':<10}{'EMAs':<10}{'BP':<6}{'Vol':<6}{'TP/SL':<12}{'Trades':<8}{'WR':<8}{'PF':<8}{'Return':<10}{'Sharpe':<8}")
    print("-"*110)

    for i, r in enumerate(all_results[:20], 1):
        emas = f"{r['fast_ema']}/{r['slow_ema']}"
        tp_sl = f"{r['take_profit']*100:.0f}/{r['stop_loss']*100:.0f}"
        vol = "Y" if r['use_volume'] else "N"

        print(f"{i:<4}{r['symbol']:<10}{emas:<10}{r['breakout_period']:<6}{vol:<6}{tp_sl:<12}"
              f"{r['total_trades']:<8}{r['win_rate']:<8.2%}{r['profit_factor']:<8.2f}"
              f"{r['total_return']:<10.2%}{r['sharpe']:<8.2f}")

    # Best strategy
    best = all_results[0]

    print("\n" + "="*80)
    print("🏆 BEST STRATEGY")
    print("="*80)
    print(f"\nSymbol: {best['symbol']}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"\nStrategy Parameters:")
    print(f"  Fast EMA: {best['fast_ema']}")
    print(f"  Slow EMA: {best['slow_ema']}")
    print(f"  Breakout Period: {best['breakout_period']}")
    print(f"  Use Volume Filter: {best['use_volume']}")
    if best['use_volume']:
        print(f"  Volume Threshold: {best['volume_threshold']}")

    print(f"\nRisk Management:")
    print(f"  Take Profit: {best['take_profit']:.1%}")
    print(f"  Stop Loss: {best['stop_loss']:.1%}")
    print(f"  Trailing Stop: {best['trailing_stop']:.1%}")
    print(f"  Risk/Reward Ratio: {best['take_profit']/best['stop_loss']:.2f}:1")

    print(f"\nPerformance:")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Win Rate: {best['win_rate']:.2%}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total Return: {best['total_return']:.2%}")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"  Avg Win: {best['avg_win']:.3%}")
    print(f"  Avg Loss: {best['avg_loss']:.3%}")

    # Verdict
    print("\n" + "="*80)
    if best['total_return'] > 0.15 and best['win_rate'] > 0.45 and best['profit_factor'] > 1.5:
        print("✅ EXCELLENT STRATEGY FOUND!")
        print("Strong edge with good win rate and returns!")
    elif best['total_return'] > 0.10 and best['profit_factor'] > 1.3:
        print("✅ WINNING STRATEGY FOUND!")
        print("Profitable with solid metrics!")
    elif best['total_return'] > 0.05:
        print("⚠️  MARGINALLY PROFITABLE")
        print("Shows promise but could use more optimization.")
    else:
        print("❌ Still searching for profitability...")
        print("Best result: {:.2%} return".format(best['total_return']))

    # Save results
    output_dir = Path("alpha_research/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f"trend_optimization_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nFull results saved to: {results_file}")


if __name__ == '__main__':
    main()
