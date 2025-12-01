"""
Fetch BTC-USD hourly data from Yahoo Finance for 2023-2025
Uses 1-hour intervals (Yahoo Finance allows 730 days of hourly data)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

def fetch_btc_hourly():
    """
    Fetch BTC-USD hourly data

    Yahoo Finance limitations:
    - 1h data: 730 days max (we'll get ~2 years)
    - This gives us 2023-2024 for training, 2025 for testing
    """
    print("Fetching BTC-USD hourly data (last 730 days)...")

    ticker = yf.Ticker("BTC-USD")

    # Get maximum available hourly data (730 days)
    df = ticker.history(period="730d", interval="1h")

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    # Rename columns
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Ensure UTC timezone
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    df.index.name = 'timestamp'

    print(f"✅ Fetched {len(df):,} hourly bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def main():
    # Fetch hourly data
    df = fetch_btc_hourly()

    # Save to parquet
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'Data' / 'BTCUSD_1h_yfinance.parquet'
    output_path.parent.mkdir(exist_ok=True)

    df.to_parquet(output_path)
    print(f"\n✅ Saved to: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Duration: {(df.index.max() - df.index.min()).days} days")
    print(f"\nPrice statistics:")
    print(f"  Min:  ${df['close'].min():,.2f}")
    print(f"  Max:  ${df['close'].max():,.2f}")
    print(f"  Mean: ${df['close'].mean():,.2f}")
    print(f"\nMissing values:")
    print(df.isnull().sum())

    # Split suggestions
    # Use last 11 months for testing (2025)
    cutoff_date = '2024-12-31'
    train_df = df[df.index < cutoff_date]
    test_df = df[df.index >= '2025-01-01']

    print("\n" + "="*60)
    print("SUGGESTED TRAIN/TEST SPLIT")
    print("="*60)
    print(f"\n📊 TRAIN PERIOD (pre-2025):")
    if len(train_df) > 0:
        print(f"   Bars: {len(train_df):,}")
        print(f"   Date: {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"   Duration: {(train_df.index.max() - train_df.index.min()).days} days")
    else:
        print("   No data before 2025")

    print(f"\n📊 TEST PERIOD (2025):")
    if len(test_df) > 0:
        print(f"   Bars: {len(test_df):,}")
        print(f"   Date: {test_df.index.min().date()} to {test_df.index.max().date()}")
        print(f"   Duration: {(test_df.index.max() - test_df.index.min()).days} days")
    else:
        print("   No 2025 data yet")

    print("="*60)

    return df


if __name__ == "__main__":
    main()
