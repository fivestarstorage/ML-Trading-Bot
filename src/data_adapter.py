import os
from typing import Optional

import pandas as pd

from .utils import get_logger
from .ig_client import IGClient
from .alpaca_client import AlpacaClient

logger = get_logger()

class DataAdapter:
    def __init__(
        self,
        config,
        ig_client: Optional[IGClient] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        data_cfg = config['data']
        self.config = config
        self.data_folder = data_cfg['folder']
        self.symbol = data_cfg['symbol']
        self.base_tf = data_cfg['timeframe_base']
        self.source = data_cfg.get('source', 'csv').lower()
        self.ig_settings = data_cfg.get('ig_stream', {})
        self._ig_client = ig_client
        self._ig_cache = None
        self._alpaca_cache: Optional[pd.DataFrame] = None
        self._alpaca_client: Optional[AlpacaClient] = None
        self.requested_start = pd.to_datetime(start_date) if start_date else None
        self.requested_end = pd.to_datetime(end_date) if end_date else None
        
    def load_data(self, timeframe_suffix=None):
        if self.source == 'ig':
            suffix = timeframe_suffix.lower() if timeframe_suffix else self.base_tf.lower()
            base_suffix = self.base_tf.lower()
            base_df = self._load_ig_data()
            if suffix in (base_suffix, None):
                return base_df
            minutes = self._tf_to_minutes(suffix)
            return self.resample_data(base_df, minutes)
        if self.source == 'alpaca':
            return self._load_alpaca_data(timeframe_suffix)

        tf = timeframe_suffix if timeframe_suffix else self.base_tf.upper()
        if tf.endswith('m'): tf = 'M' + tf[:-1]
        if tf.endswith('h'): tf = 'H' + tf[:-1]
        
        # Try to locate the file with various naming conventions
        # Handle XAU vs XAUUSD mismatch
        symbols_to_check = [self.symbol]
        if self.symbol == 'XAUUSD':
            symbols_to_check.append('XAU')
        elif self.symbol == 'XAU':
            symbols_to_check.append('XAUUSD')
            
        possible_filenames = []
        for sym in symbols_to_check:
            # Priority 1: Exact match with requested timeframe
            # e.g. XAU_5m_data.csv if tf="5m"
            # tf usually comes in as "M5" or "H1" from the logic above
            
            # If tf is 'M5', we want '5m'
            if tf.startswith('M'):
                tf_lower = tf[1:] + 'm'
            elif tf.startswith('H'):
                tf_lower = tf[1:] + 'h'
            else:
                tf_lower = tf
                
            possible_filenames.extend([
                f"{sym}_{tf_lower}_data.csv", # XAU_5m_data.csv
                f"{sym}_{tf}.csv",            # XAU_M5.csv
            ])
            
        # Add fallbacks only if not found
        for sym in symbols_to_check:
            possible_filenames.extend([
                f"{sym}_1m_data.csv",
                f"{sym}.csv"
            ])
        
        path = None
        for fname in possible_filenames:
            p = os.path.join(self.data_folder, fname)
            if os.path.exists(p):
                path = p
                break
            # Check parent dir
            parent_p = os.path.join(os.path.dirname(os.path.dirname(__file__)), p)
            if os.path.exists(parent_p):
                path = parent_p
                break
        
        if not path:
             raise FileNotFoundError(f"Data file not found. Checked: {possible_filenames}")

        logger.info(f"Loading data from {path}")
        
        try:
            # Pre-inspect file to handle header/column mismatch
            with open(path, 'r') as f:
                header_line = f.readline().strip()
                first_data = f.readline().strip()
                
            # Simple separator detection
            if '\t' in header_line:
                sep = '\t'
            elif ',' in header_line:
                sep = ','
            elif ';' in header_line:
                sep = ';'
            else:
                sep = None # let python engine decide or fail
                
            # Check column counts
            if sep:
                h_cols = header_line.split(sep)
                d_cols = first_data.split(sep)
                
                if len(d_cols) > len(h_cols):
                    diff = len(d_cols) - len(h_cols)
                    logger.info(f"Detected {len(d_cols)} data columns but only {len(h_cols)} headers. Patching headers.")
                    names = [c.strip().lower() for c in h_cols] + [f"extra_{i}" for i in range(diff)]
                    df = pd.read_csv(path, sep=sep, names=names, header=0, engine='python')
                else:
                    df = pd.read_csv(path, sep=sep, engine='python')
            else:
                 df = pd.read_csv(path, sep=None, engine='python')

            # Normalize columns
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Identify Timestamp column
            time_col = None
            for col in ['time', 'date', 'datetime', 'timestamp']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                df.rename(columns={time_col: 'timestamp'}, inplace=True)
            else:
                # Use first column
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Normalize OHLCV
            req_cols = ['open', 'high', 'low', 'close']
            
            # Handle Volume variations
            if 'volume' not in df.columns:
                if 'tick_volume' in df.columns:
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                elif 'tickvol' in df.columns:
                    df.rename(columns={'tickvol': 'volume'}, inplace=True)
                else:
                     # Check extra cols?
                     pass
            
            for col in req_cols:
                if col not in df.columns:
                     raise ValueError(f"Missing required column: {col}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _load_ig_data(self):
        if self._ig_cache is not None:
            return self._ig_cache.copy()

        if not self.ig_settings:
            raise ValueError("IG streaming settings missing in config['data']['ig_stream'].")
        epic = self.ig_settings.get('epic')
        resolution = self.ig_settings.get('resolution', '5MINUTE')
        max_bars = self.ig_settings.get('max_bars', 2000)
        if not epic:
            raise ValueError("`data.ig_stream.epic` must be set when data.source = 'ig'.")
        client = self._get_ig_client()
        df = client.fetch_prices(epic, resolution=resolution, max_points=max_bars)
        df = df.rename(columns=str.lower)
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        self._ig_cache = df
        return df.copy()

    def _get_ig_client(self):
        if self._ig_client is None:
            live_cfg = self.config.get('live_trading', {})
            self._ig_client = IGClient(
                api_key=live_cfg.get('api_key'),
                username=live_cfg.get('username'),
                password=live_cfg.get('password'),
                account_id=live_cfg.get('account_id'),
                use_demo=live_cfg.get('use_demo', True),
            )
        return self._ig_client

    def _load_alpaca_data(self, timeframe_suffix: Optional[str]) -> pd.DataFrame:
        base_df = self._get_alpaca_base()
        target_tf = (timeframe_suffix or self.base_tf).lower()
        base_tf = self.base_tf.lower()
        if target_tf == base_tf:
            return base_df.copy()
        minutes = self._tf_to_minutes(target_tf)
        return self.resample_data(base_df, minutes)

    def _get_alpaca_base(self) -> pd.DataFrame:
        if self._alpaca_cache is not None:
            return self._alpaca_cache

        client = self._get_alpaca_client()
        alpaca_cfg = self.config['data'].get('alpaca', {})
        start = self.requested_start or self._coerce_timestamp(alpaca_cfg.get('start'))
        end = self.requested_end or self._coerce_timestamp(alpaca_cfg.get('end'))
        tf_label = self._alpaca_timeframe(self.base_tf)
        symbol = self.symbol.upper()
        logger.info(f"Fetching {symbol} bars from Alpaca ({tf_label})...")
        df = client.fetch_bars(symbol, timeframe=tf_label, start=start, end=end)
        # Ensure UTC tz
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        self._alpaca_cache = df[['open', 'high', 'low', 'close', 'volume']]
        return self._alpaca_cache

    def _get_alpaca_client(self) -> AlpacaClient:
        if self._alpaca_client is None:
            self._alpaca_client = AlpacaClient()
        return self._alpaca_client

    def _alpaca_timeframe(self, tf: str) -> str:
        tf = tf.lower()
        if tf.endswith('m'):
            return f"{int(tf[:-1])}Min"
        if tf.endswith('h'):
            minutes = int(tf[:-1]) * 60
            return f"{minutes}Min"
        if tf.endswith('d'):
            return f"{int(tf[:-1])}Day"
        raise ValueError(f"Unsupported timeframe '{tf}' for Alpaca.")

    @staticmethod
    def _coerce_timestamp(value: Optional[str]):
        if not value:
            return None
        ts = pd.to_datetime(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts

    def load_h4_data(self, start_date=None, end_date=None):
        """
        Load H4 data directly from file, fallback to resampling if not found.
        """
        if self.source == 'ig':
            df_base = self.load_data(timeframe_suffix=self.base_tf)
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_base = df_base[df_base.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_base = df_base[df_base.index <= ts]
            h4_min = 240
            return self.resample_data(df_base, h4_min)

        try:
            df_h4 = self.load_data(timeframe_suffix='4h')
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_h4 = df_h4[df_h4.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_h4 = df_h4[df_h4.index <= ts]
            logger.info(f"Loaded H4 data directly: {len(df_h4)} bars")
            return df_h4
        except FileNotFoundError:
            logger.info("4h data file not found, resampling from base timeframe...")
            df_base = self.load_data(timeframe_suffix=self.base_tf)
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_base = df_base[df_base.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_base = df_base[df_base.index <= ts]
            h4_min = 240  # 4 hours
            return self.resample_data(df_base, h4_min)
    
    def load_daily_data(self, start_date=None, end_date=None):
        """
        Load Daily data (1D) directly if available, otherwise resample base timeframe.
        """
        if self.source == 'ig':
            df_base = self.load_data(timeframe_suffix=self.base_tf)
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_base = df_base[df_base.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_base = df_base[df_base.index <= ts]
            return self.resample_data(df_base, 1440)

        try:
            df_daily = self.load_data(timeframe_suffix='1d')
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_daily = df_daily[df_daily.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_daily = df_daily[df_daily.index <= ts]
            logger.info(f"Loaded Daily data directly: {len(df_daily)} bars")
            return df_daily
        except FileNotFoundError:
            logger.info("Daily data file not found, resampling from base timeframe...")
            df_base = self.load_data(timeframe_suffix=self.base_tf)
            if start_date:
                ts = self._to_utc_timestamp(start_date)
                df_base = df_base[df_base.index >= ts]
            if end_date:
                ts = self._to_utc_timestamp(end_date)
                df_base = df_base[df_base.index <= ts]
            return self.resample_data(df_base, 1440)
    
    def resample_data(self, df, timeframe_minutes):
        if timeframe_minutes == 1:
            return df
        rule = f'{timeframe_minutes}min'
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        resampled = df.resample(rule).agg(agg_dict)
        return resampled.dropna()

    @staticmethod
    def _tf_to_minutes(label: str) -> int:
        label = label.lower()
        if label.endswith('m'):
            return int(label[:-1])
        if label.endswith('h'):
            return int(label[:-1]) * 60
        if label.endswith('d'):
            return int(label[:-1]) * 1440
        raise ValueError(f"Unsupported timeframe suffix '{label}' for IG resampling.")

    @staticmethod
    def _to_utc_timestamp(value):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts


