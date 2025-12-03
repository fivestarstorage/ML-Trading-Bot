"""
Download SPY data from Yahoo Finance for ORB strategy backtesting
Note: Yahoo Finance limits 5m data to 60 days, so we download 1h data and resample to 5m
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_spy_data():
    """Download SPY daily data from 2020 to present"""
    print("Downloading SPY daily data (full historical)...")
    print("Note: Yahoo Finance limits intraday data (5m, 1h) to recent periods only")

    # Create Data folder if it doesn't exist
    data_folder = Path("Data")
    data_folder.mkdir(exist_ok=True)

    # Download 1-hour data (available for longer periods)
    all_data = []

    start_date = "2020-01-01"
    end_date = "2025-12-31"

    try:
        print(f"Downloading {start_date} to {end_date}...")
        ticker = yf.Ticker("SPY")
        df = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)

        if len(df) > 0:
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Convert index to datetime with UTC timezone
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize('UTC')
            else:
                df.index = pd.to_datetime(df.index).tz_convert('UTC')

            all_data.append(df)
            print(f"  Downloaded {len(df)} bars")
        else:
            print(f"  No data for this period")

    except Exception as e:
        print(f"  Error: {e}")
        return

    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()

        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Save daily data
        output_file = data_folder / "SPY_1d_data.csv"
        combined_df.to_csv(output_file)

        print(f"\nTotal daily bars downloaded: {len(combined_df)}")
        print(f"Date range: {combined_df.index[0]} to {combined_df.index[-1]}")
        print(f"Saved to: {output_file}")

        # Display sample
        print("\nSample data:")
        print(combined_df.head())
        print("\nData info:")
        print(combined_df.info())

        # Note about ORB strategy adaptation
        print("\n" + "="*70)
        print("NOTE: Due to Yahoo Finance limitations, daily data will be used.")
        print("The ORB strategy will be adapted to work on daily timeframe:")
        print("- Opening range = First few bars of the day")
        print("- This tests the daily breakout variation of ORB strategy")
        print("="*70)

    else:
        print("No data downloaded!")

if __name__ == "__main__":
    download_spy_data()
