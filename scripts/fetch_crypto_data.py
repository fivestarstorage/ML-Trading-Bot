import argparse
import time
from pathlib import Path

import pandas as pd
import ccxt


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, start: str, end: str | None, limit: int, throttle: float):
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({'enableRateLimit': True})
    exchange.load_markets()
    df_rows = []
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    start_ms = int(pd.Timestamp(start).tz_localize('UTC').timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).tz_localize('UTC').timestamp() * 1000) if end else None
    cursor = start_ms
    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not data:
            break
        for row in data:
            ts = row[0]
            if end_ms and ts > end_ms:
                return df_rows
            df_rows.append(row)
        cursor = data[-1][0] + tf_ms
        if end_ms and cursor > end_ms:
            break
        if len(data) < limit:
            break
        time.sleep(throttle)
    return df_rows


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data via ccxt and save to CSV.")
    parser.add_argument("--exchange", default="coinbase", help="ccxt exchange id (default: coinbase)")
    parser.add_argument("--symbol", default="BTC/USD", help="Market symbol, e.g. BTC/USD")
    parser.add_argument("--timeframe", default="5m", help="Timeframe supported by exchange")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=1500, help="Max candles per API call")
    parser.add_argument("--throttle", type=float, default=0.35, help="Seconds to sleep between requests")
    parser.add_argument("--output", default="Data/BTCUSD_5m_data.csv", help="Destination CSV/Parquet file")
    args = parser.parse_args()

    rows = fetch_ohlcv(
        args.exchange,
        args.symbol,
        args.timeframe,
        args.start,
        args.end,
        args.limit,
        args.throttle,
    )
    if not rows:
        raise RuntimeError("No data returned.")

    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path)
    else:
        df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

