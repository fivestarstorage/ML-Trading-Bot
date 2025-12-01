"""
Fetch BTC-USD data from Yahoo Finance for 2020-2025
Uses 5-minute intervals to match existing data structure
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

def fetch_btc_data(start_date: str = "2020-01-01", end_date: str = "2025-11-30", interval: str = "5m"):
    """
    Fetch BTC-USD data from Yahoo Finance

    Note: Yahoo Finance 5m data has limitations:
    - Only available for last ~60 days
    - For longer periods, we'll use 1h or 1d and resample if needed

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching BTC-USD data from {start_date} to {end_date}...")
    print(f"Interval: {interval}")

    # For 5-year period, we need to use daily data
    # Yahoo Finance limitations:
    # - 5m data: only 60 days
    # - 1h data: only 730 days
    # - 1d data: unlimited history
    if interval in ["5m", "1h"]:
        print(f"⚠️  Yahoo Finance {interval} data has time limitations")
        print("Using 1d (daily) interval instead for 5-year period")
        interval = "1d"

    # Download data
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    # Rename columns to match our format
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Ensure index is timezone-aware datetime
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    df.index.name = 'timestamp'

    print(f"✅ Fetched {len(df):,} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")

    return df


def main():
    # Fetch data - use daily data since yfinance limits hourly to 730 days
    df = fetch_btc_data(
        start_date="2020-01-01",
        end_date="2025-11-30",
        interval="1d"  # Use daily data for 5-year period
    )

    # Save to parquet
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'Data' / 'BTCUSD_1d_yfinance.parquet'
    output_path.parent.mkdir(exist_ok=True)

    df.to_parquet(output_path)
    print(f"\n✅ Saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Duration: {(df.index.max() - df.index.min()).days} days")
    print(f"\nPrice statistics:")
    print(f"  Min:  ${df['close'].min():,.2f}")
    print(f"  Max:  ${df['close'].max():,.2f}")
    print(f"  Mean: ${df['close'].mean():,.2f}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print("="*60)

    # Split into train/test periods
    train_df = df['2020-01-01':'2024-12-31']
    test_df = df['2025-01-01':'2025-11-30']

    print(f"\n📊 TRAIN PERIOD (2020-2024):")
    print(f"   Rows: {len(train_df):,}")
    print(f"   Date: {train_df.index.min().date()} to {train_df.index.max().date()}")

    print(f"\n📊 TEST PERIOD (2025):")
    print(f"   Rows: {len(test_df):,}")
    print(f"   Date: {test_df.index.min().date()} to {test_df.index.max().date()}")

    return df


if __name__ == "__main__":
    main()
