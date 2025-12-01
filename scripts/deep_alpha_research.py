#!/usr/bin/env python3
"""
Deep Alpha Research - Finding Unique Edge in Stock Markets

This script performs intensive analysis to discover unique, exploitable patterns
that most traders miss. Focus on microstructure, behavioral anomalies, and
statistical edges.
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.alpaca_client import AlpacaClient
from src.utils import get_logger

logger = get_logger()

class DeepAlphaResearcher:
    """
    Deep dive researcher looking for unique edges in market data.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = AlpacaClient()
        self.data = None

    def fetch_data(self, timeframe="5Min", days_back=60):
        """Fetch comprehensive market data."""
        logger.info(f"Fetching {days_back} days of {timeframe} data for {self.symbol}")

        end = pd.Timestamp.now(tz='UTC')
        start = end - pd.Timedelta(days=days_back)

        try:
            self.data = self.client.fetch_bars(
                symbol=self.symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=10000
            )

            logger.info(f"Fetched {len(self.data)} bars from {self.data.index.min()} to {self.data.index.max()}")
            return self.data

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def analyze_volume_microstructure(self):
        """
        Research Edge #1: Volume Microstructure Anomalies

        Most traders look at volume as a single metric. We'll analyze:
        - Volume clustering patterns
        - Volume delta (buy vs sell pressure estimation)
        - Volume acceleration patterns
        - Abnormal volume spikes and their aftermath
        """
        logger.info("Analyzing volume microstructure...")

        df = self.data.copy()

        # Calculate volume metrics
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (df['volume_std'] + 1e-8)

        # Volume delta proxy (using price change and volume)
        df['price_change'] = df['close'].diff()
        df['volume_delta'] = df['volume'] * np.sign(df['price_change'])
        df['cumulative_volume_delta'] = df['volume_delta'].rolling(20).sum()

        # Volume acceleration
        df['volume_accel'] = df['volume'].diff().diff()

        # Identify volume spikes (Z-score > 2)
        volume_spikes = df[df['volume_zscore'] > 2.0].copy()

        results = {
            'volume_spikes_count': len(volume_spikes),
            'avg_volume': df['volume'].mean(),
            'volume_spike_df': volume_spikes
        }

        # Analyze what happens AFTER volume spikes
        if len(volume_spikes) > 0:
            forward_returns = []
            for idx in volume_spikes.index:
                try:
                    idx_loc = df.index.get_loc(idx)
                    if idx_loc < len(df) - 20:  # Need 20 bars forward
                        future_price = df['close'].iloc[idx_loc + 20]
                        current_price = df['close'].iloc[idx_loc]
                        fwd_ret = (future_price - current_price) / current_price
                        forward_returns.append(fwd_ret)
                except:
                    pass

            if forward_returns:
                results['avg_return_after_spike'] = np.mean(forward_returns)
                results['win_rate_after_spike'] = sum(1 for r in forward_returns if r > 0) / len(forward_returns)

        logger.info(f"Volume analysis: {results['volume_spikes_count']} spikes found")
        return results, df

    def analyze_price_rejection_patterns(self, df):
        """
        Research Edge #2: Price Rejection Patterns

        Look for candle patterns that indicate strong rejection:
        - Long wicks relative to body
        - Failed breakouts
        - Immediate reversals after touching levels
        """
        logger.info("Analyzing price rejection patterns...")

        # Calculate candle metrics
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        # Rejection ratios
        df['upper_wick_ratio'] = df['upper_wick'] / (df['total_range'] + 1e-8)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['total_range'] + 1e-8)

        # Find strong rejections (wick > 60% of total range)
        upper_rejections = df[df['upper_wick_ratio'] > 0.6].copy()
        lower_rejections = df[df['lower_wick_ratio'] > 0.6].copy()

        # What happens after rejections?
        results = {
            'upper_rejections': len(upper_rejections),
            'lower_rejections': len(lower_rejections)
        }

        # Analyze forward returns after upper rejections (should be bearish)
        if len(upper_rejections) > 10:
            returns_after_upper = self._calculate_forward_returns(df, upper_rejections.index, periods=10)
            results['upper_rejection_fwd_return'] = np.mean(returns_after_upper)
            results['upper_rejection_win_rate'] = sum(1 for r in returns_after_upper if r < 0) / len(returns_after_upper)

        # Analyze forward returns after lower rejections (should be bullish)
        if len(lower_rejections) > 10:
            returns_after_lower = self._calculate_forward_returns(df, lower_rejections.index, periods=10)
            results['lower_rejection_fwd_return'] = np.mean(returns_after_lower)
            results['lower_rejection_win_rate'] = sum(1 for r in returns_after_lower if r > 0) / len(returns_after_lower)

        logger.info(f"Rejection analysis: {results['upper_rejections']} upper, {results['lower_rejections']} lower")
        return results, df

    def analyze_time_of_day_edge(self, df):
        """
        Research Edge #3: Intraday Time-Based Patterns

        Different times of day have different characteristics:
        - Market open volatility
        - Lunch hour doldrums
        - Power hour trends
        - Overnight gaps
        """
        logger.info("Analyzing time-of-day patterns...")

        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['time_bucket'] = df['hour'] * 60 + df['minute']

        # Calculate returns by time bucket
        df['returns'] = df['close'].pct_change()

        time_stats = df.groupby('time_bucket').agg({
            'returns': ['mean', 'std', 'count'],
            'volume': 'mean'
        }).round(6)

        # Find best performing time buckets
        time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
        time_stats['sharpe'] = time_stats['returns_mean'] / (time_stats['returns_std'] + 1e-8)

        best_times = time_stats.nlargest(5, 'sharpe')
        worst_times = time_stats.nsmallest(5, 'sharpe')

        results = {
            'best_time_buckets': best_times,
            'worst_time_buckets': worst_times,
            'time_stats': time_stats
        }

        logger.info(f"Time analysis: Best times identified")
        return results, df

    def analyze_momentum_exhaustion(self, df):
        """
        Research Edge #4: Momentum Exhaustion Detection

        Identify when strong moves are running out of steam:
        - Decreasing volume on continued price movement
        - RSI divergence patterns
        - Slowing rate of change
        """
        logger.info("Analyzing momentum exhaustion...")

        # Calculate momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        df['roc'] = df['close'].pct_change(periods=10)
        df['roc_ma'] = df['roc'].rolling(5).mean()

        # Volume trends during price moves
        df['price_direction'] = np.sign(df['close'].diff(5))
        df['volume_trend'] = df['volume'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)

        # Exhaustion signal: price up but volume decreasing (or vice versa)
        df['bullish_exhaustion'] = ((df['price_direction'] > 0) & (df['volume_trend'] < 0)).astype(int)
        df['bearish_exhaustion'] = ((df['price_direction'] < 0) & (df['volume_trend'] < 0)).astype(int)

        # Find exhaustion points
        bull_exhaustion_points = df[df['bullish_exhaustion'] == 1]
        bear_exhaustion_points = df[df['bearish_exhaustion'] == 1]

        results = {
            'bull_exhaustion_count': len(bull_exhaustion_points),
            'bear_exhaustion_count': len(bear_exhaustion_points)
        }

        # Analyze what happens after exhaustion
        if len(bull_exhaustion_points) > 10:
            returns = self._calculate_forward_returns(df, bull_exhaustion_points.index, periods=5)
            results['bull_exhaustion_fwd_return'] = np.mean(returns)
            results['bull_exhaustion_reversal_rate'] = sum(1 for r in returns if r < 0) / len(returns)

        if len(bear_exhaustion_points) > 10:
            returns = self._calculate_forward_returns(df, bear_exhaustion_points.index, periods=5)
            results['bear_exhaustion_fwd_return'] = np.mean(returns)
            results['bear_exhaustion_reversal_rate'] = sum(1 for r in returns if r > 0) / len(returns)

        logger.info(f"Exhaustion analysis: {results['bull_exhaustion_count']} bull, {results['bear_exhaustion_count']} bear")
        return results, df

    def analyze_gap_behavior(self, df):
        """
        Research Edge #5: Gap Fill Probability

        Analyze overnight gaps and their tendency to fill:
        - Gap size
        - Gap direction
        - Volume on gap
        - Fill rate and timing
        """
        logger.info("Analyzing gap behavior...")

        # Detect gaps (comparing today's open to yesterday's close)
        df['prev_close'] = df['close'].shift(1)
        df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['gap_size'] = abs(df['gap'])

        # Only consider significant gaps (> 0.2%)
        significant_gaps = df[df['gap_size'] > 0.002].copy()

        results = {
            'total_gaps': len(significant_gaps),
            'avg_gap_size': significant_gaps['gap'].mean() if len(significant_gaps) > 0 else 0
        }

        # Analyze gap fill behavior
        if len(significant_gaps) > 10:
            fill_results = []
            for idx in significant_gaps.index:
                try:
                    idx_loc = df.index.get_loc(idx)
                    gap_value = df['gap'].iloc[idx_loc]
                    prev_close = df['prev_close'].iloc[idx_loc]

                    # Look forward to see if gap fills within next 20 bars
                    filled = False
                    bars_to_fill = None

                    for i in range(1, min(21, len(df) - idx_loc)):
                        future_low = df['low'].iloc[idx_loc + i]
                        future_high = df['high'].iloc[idx_loc + i]

                        # Check if gap filled
                        if gap_value > 0:  # Gap up
                            if future_low <= prev_close:
                                filled = True
                                bars_to_fill = i
                                break
                        else:  # Gap down
                            if future_high >= prev_close:
                                filled = True
                                bars_to_fill = i
                                break

                    fill_results.append({
                        'filled': filled,
                        'bars_to_fill': bars_to_fill,
                        'gap_size': gap_value
                    })
                except:
                    pass

            if fill_results:
                fill_rate = sum(1 for r in fill_results if r['filled']) / len(fill_results)
                avg_bars_to_fill = np.mean([r['bars_to_fill'] for r in fill_results if r['filled']])

                results['gap_fill_rate'] = fill_rate
                results['avg_bars_to_fill'] = avg_bars_to_fill

        logger.info(f"Gap analysis: {results['total_gaps']} gaps, {results.get('gap_fill_rate', 0):.1%} fill rate")
        return results, df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_forward_returns(self, df, indices, periods=10):
        """Calculate forward returns from given indices."""
        returns = []
        for idx in indices:
            try:
                idx_loc = df.index.get_loc(idx)
                if idx_loc < len(df) - periods:
                    current_price = df['close'].iloc[idx_loc]
                    future_price = df['close'].iloc[idx_loc + periods]
                    ret = (future_price - current_price) / current_price
                    returns.append(ret)
            except:
                pass
        return returns

    def run_full_research(self):
        """Run all research analyses."""
        logger.info(f"Starting deep research on {self.symbol}")

        all_results = {}

        # Run all analyses
        vol_results, df = self.analyze_volume_microstructure()
        all_results['volume'] = vol_results

        rej_results, df = self.analyze_price_rejection_patterns(df)
        all_results['rejections'] = rej_results

        time_results, df = self.analyze_time_of_day_edge(df)
        all_results['time_patterns'] = time_results

        mom_results, df = self.analyze_momentum_exhaustion(df)
        all_results['momentum'] = mom_results

        gap_results, df = self.analyze_gap_behavior(df)
        all_results['gaps'] = gap_results

        # Store enriched dataframe
        self.enriched_data = df

        return all_results


def print_research_report(symbol, results):
    """Print comprehensive research report."""
    print("\n" + "="*80)
    print(f"DEEP ALPHA RESEARCH REPORT: {symbol}")
    print("="*80)

    # Volume Analysis
    print("\n1. VOLUME MICROSTRUCTURE ANALYSIS")
    print("-" * 80)
    vol = results['volume']
    print(f"   Volume Spikes Detected: {vol['volume_spikes_count']}")
    if 'avg_return_after_spike' in vol:
        print(f"   Avg Return After Spike (20 bars): {vol['avg_return_after_spike']:.4%}")
        print(f"   Win Rate After Spike: {vol['win_rate_after_spike']:.2%}")

    # Rejection Patterns
    print("\n2. PRICE REJECTION PATTERNS")
    print("-" * 80)
    rej = results['rejections']
    print(f"   Upper Rejections (bearish): {rej['upper_rejections']}")
    if 'upper_rejection_fwd_return' in rej:
        print(f"   Avg Return After Upper Rejection: {rej['upper_rejection_fwd_return']:.4%}")
        print(f"   Win Rate (short): {rej['upper_rejection_win_rate']:.2%}")
    print(f"   Lower Rejections (bullish): {rej['lower_rejections']}")
    if 'lower_rejection_fwd_return' in rej:
        print(f"   Avg Return After Lower Rejection: {rej['lower_rejection_fwd_return']:.4%}")
        print(f"   Win Rate (long): {rej['lower_rejection_win_rate']:.2%}")

    # Time Patterns
    print("\n3. TIME-OF-DAY EDGE ANALYSIS")
    print("-" * 80)
    time_p = results['time_patterns']
    print("   Best Performing Times:")
    print(time_p['best_time_buckets'][['returns_mean', 'sharpe']].head(3))

    # Momentum Exhaustion
    print("\n4. MOMENTUM EXHAUSTION DETECTION")
    print("-" * 80)
    mom = results['momentum']
    print(f"   Bullish Exhaustion Signals: {mom['bull_exhaustion_count']}")
    if 'bull_exhaustion_reversal_rate' in mom:
        print(f"   Reversal Rate: {mom['bull_exhaustion_reversal_rate']:.2%}")
    print(f"   Bearish Exhaustion Signals: {mom['bear_exhaustion_count']}")
    if 'bear_exhaustion_reversal_rate' in mom:
        print(f"   Reversal Rate: {mom['bear_exhaustion_reversal_rate']:.2%}")

    # Gap Analysis
    print("\n5. GAP FILL PROBABILITY")
    print("-" * 80)
    gaps = results['gaps']
    print(f"   Total Significant Gaps: {gaps['total_gaps']}")
    if 'gap_fill_rate' in gaps:
        print(f"   Gap Fill Rate: {gaps['gap_fill_rate']:.2%}")
        print(f"   Avg Bars to Fill: {gaps['avg_bars_to_fill']:.1f}")

    print("\n" + "="*80)


def main():
    """Main research execution."""

    # Choose a high-volume stock
    SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']

    print("DEEP ALPHA RESEARCH - Finding Unique Edges")
    print("="*80)
    print("\nAnalyzing major stocks for unique patterns...")

    # Research multiple symbols
    all_symbol_results = {}

    for symbol in SYMBOLS[:1]:  # Start with SPY
        print(f"\n\nResearching {symbol}...")
        print("-"*80)

        researcher = DeepAlphaResearcher(symbol)
        researcher.fetch_data(timeframe="5Min", days_back=60)

        results = researcher.run_full_research()
        all_symbol_results[symbol] = results

        print_research_report(symbol, results)

        # Save enriched data
        output_dir = Path("alpha_research/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched_file = output_dir / f"{symbol}_enriched_{datetime.now().strftime('%Y%m%d')}.parquet"
        researcher.enriched_data.to_parquet(enriched_file)
        print(f"\nEnriched data saved to: {enriched_file}")

    print("\n\nResearch complete! Analyzing results for strategy development...")


if __name__ == '__main__':
    main()
