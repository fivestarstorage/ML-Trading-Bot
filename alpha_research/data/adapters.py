"""
Data Adapters

Modular adapters for loading data from various sources.
Supports crypto, stocks, commodities across different data providers.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import warnings

from ..utils.logging import get_logger

logger = get_logger()


class DataAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    All adapters return data in a standardized format:
    - DatetimeIndex (UTC timezone)
    - Columns: open, high, low, close, volume (minimum)
    - Optional columns: funding_rate, open_interest, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """
        Load data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            timeframe: Data timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data format.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
                df = df.drop(columns=['timestamp'])
            elif 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
                df = df.drop(columns=['date'])
            elif 'time' in df.columns:
                df.index = pd.to_datetime(df['time'])
                df = df.drop(columns=['time'])
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure UTC timezone
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        return df


class CSVAdapter(DataAdapter):
    """Adapter for loading CSV files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_dir = Path(config.get('data_dir', 'Data'))
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        
        # Try different file naming patterns
        patterns = [
            f"{symbol}_{timeframe}_data.csv",
            f"{symbol}_{timeframe}.csv",
            f"{symbol}.csv",
            f"{symbol}_data.csv",
        ]
        
        csv_path = None
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                f"No CSV file found for {symbol} in {self.data_dir}"
            )
        
        logger.info(f"Loading data from {csv_path}")
        
        # Try to detect datetime column
        df = pd.read_csv(csv_path)
        
        # Detect date column
        date_cols = [c for c in df.columns if any(
            x in c.lower() for x in ['date', 'time', 'timestamp']
        )]
        
        if date_cols:
            df = pd.read_csv(csv_path, parse_dates=[date_cols[0]], index_col=date_cols[0])
        else:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        df = self.preprocess(df)
        
        # Apply date filter
        if start_date:
            start_ts = pd.Timestamp(start_date, tz='UTC')
            df = df[df.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date, tz='UTC')
            df = df[df.index <= end_ts]
        
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        return df


class ParquetAdapter(DataAdapter):
    """Adapter for loading Parquet files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_dir = Path(config.get('data_dir', 'Data'))
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load data from Parquet file."""
        
        patterns = [
            f"{symbol}_{timeframe}_*.parquet",
            f"{symbol}_{timeframe}.parquet",
            f"{symbol}.parquet",
            f"*{symbol}*.parquet",
        ]
        
        parquet_path = None
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                parquet_path = matches[0]
                break
        
        if parquet_path is None:
            raise FileNotFoundError(
                f"No Parquet file found for {symbol} in {self.data_dir}"
            )
        
        logger.info(f"Loading data from {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        df = self.preprocess(df)
        
        if start_date:
            start_ts = pd.Timestamp(start_date, tz='UTC')
            df = df[df.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date, tz='UTC')
            df = df[df.index <= end_ts]
        
        logger.info(f"Loaded {len(df)} rows from {parquet_path}")
        
        return df


class CCXTAdapter(DataAdapter):
    """Adapter for loading crypto data via CCXT."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange = config.get('exchange', 'binance')
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load crypto data via CCXT."""
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt is required for CCXTAdapter. Install with: pip install ccxt")
        
        # Initialize exchange
        exchange_class = getattr(ccxt, self.exchange)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        # Normalize symbol format
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"
        
        logger.info(f"Fetching {symbol} from {self.exchange}")
        
        # Calculate timestamps
        if start_date:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
        else:
            since = int((pd.Timestamp.now() - pd.Timedelta(days=365)).timestamp() * 1000)
        
        if end_date:
            until = int(pd.Timestamp(end_date).timestamp() * 1000)
        else:
            until = int(pd.Timestamp.now().timestamp() * 1000)
        
        # Fetch OHLCV data
        all_candles = []
        current_since = since
        
        while current_since < until:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            current_since = candles[-1][0] + 1
            
            # Respect rate limits
            import time
            time.sleep(exchange.rateLimit / 1000)
        
        if not all_candles:
            raise ValueError(f"No data fetched for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = self.preprocess(df)
        
        # Filter by end date
        if end_date:
            end_ts = pd.Timestamp(end_date, tz='UTC')
            df = df[df.index <= end_ts]
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        
        return df


class AlpacaAdapter(DataAdapter):
    """Adapter for loading stock/crypto data via Alpaca."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.asset_class = config.get('asset_class', 'stocks')
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load data via Alpaca API."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError(
                "alpaca-py is required for AlpacaAdapter. "
                "Install with: pip install alpaca-py"
            )
        
        import os
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required"
            )
        
        # Parse timeframe
        tf_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, 'Min'),
            '15m': TimeFrame(15, 'Min'),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day,
        }
        
        tf = tf_map.get(timeframe.lower())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Set dates
        start = pd.Timestamp(start_date) if start_date else pd.Timestamp.now() - pd.Timedelta(days=365)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        
        logger.info(f"Fetching {symbol} from Alpaca ({self.asset_class})")
        
        if self.asset_class == 'crypto':
            client = CryptoHistoricalDataClient()
            
            if '/' not in symbol:
                symbol = f"{symbol}/USD"
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            bars = client.get_crypto_bars(request)
        else:
            client = StockHistoricalDataClient(api_key, secret_key)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(request)
        
        # Convert to DataFrame
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        
        df = self.preprocess(df)
        
        logger.info(f"Fetched {len(df)} bars for {symbol}")
        
        return df


class UniversalAdapter(DataAdapter):
    """
    Universal adapter that automatically selects the right source.
    
    Tries adapters in order:
    1. Local CSV
    2. Local Parquet
    3. CCXT (for crypto)
    4. Alpaca (for stocks/crypto)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapters = [
            ('csv', CSVAdapter(config)),
            ('parquet', ParquetAdapter(config)),
        ]
        
        # Add network adapters if configured
        if config.get('enable_ccxt', True):
            self.adapters.append(('ccxt', CCXTAdapter(config)))
        if config.get('enable_alpaca', False):
            self.adapters.append(('alpaca', AlpacaAdapter(config)))
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load data using the first successful adapter."""
        
        errors = []
        
        for adapter_name, adapter in self.adapters:
            try:
                logger.debug(f"Trying {adapter_name} adapter for {symbol}")
                return adapter.load(symbol, start_date, end_date, timeframe)
            except Exception as e:
                errors.append(f"{adapter_name}: {str(e)}")
                continue
        
        raise ValueError(
            f"Failed to load data for {symbol}. Errors: {'; '.join(errors)}"
        )


class EnhancedCryptoAdapter(DataAdapter):
    """
    Enhanced adapter for crypto data with additional features:
    - Funding rates
    - Open interest
    - Orderbook data (if available)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange = config.get('exchange', 'binance')
        self.include_funding = config.get('include_funding', True)
        self.include_oi = config.get('include_oi', True)
    
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m'
    ) -> pd.DataFrame:
        """Load crypto data with enhanced features."""
        
        # Load base OHLCV data
        base_adapter = CCXTAdapter(self.config)
        df = base_adapter.load(symbol, start_date, end_date, timeframe)
        
        # Try to add funding rates
        if self.include_funding:
            try:
                funding = self._fetch_funding_rates(symbol, df.index.min(), df.index.max())
                if funding is not None:
                    df = df.merge(funding, left_index=True, right_index=True, how='left')
                    df['funding_rate'] = df['funding_rate'].fillna(0)
            except Exception as e:
                logger.warning(f"Failed to fetch funding rates: {e}")
        
        # Try to add open interest
        if self.include_oi:
            try:
                oi = self._fetch_open_interest(symbol, df.index.min(), df.index.max())
                if oi is not None:
                    df = df.merge(oi, left_index=True, right_index=True, how='left')
                    df['open_interest'] = df['open_interest'].fillna(method='ffill')
            except Exception as e:
                logger.warning(f"Failed to fetch open interest: {e}")
        
        return df
    
    def _fetch_funding_rates(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Fetch historical funding rates."""
        try:
            import ccxt
            
            exchange = getattr(ccxt, self.exchange)({
                'enableRateLimit': True,
            })
            
            if not hasattr(exchange, 'fetch_funding_rate_history'):
                return None
            
            # Normalize symbol for futures
            if '/' not in symbol:
                symbol = f"{symbol}/USDT:USDT"
            
            rates = []
            since = int(start.timestamp() * 1000)
            until = int(end.timestamp() * 1000)
            
            while since < until:
                history = exchange.fetch_funding_rate_history(
                    symbol,
                    since=since,
                    limit=500
                )
                if not history:
                    break
                rates.extend(history)
                since = history[-1]['timestamp'] + 1
            
            if not rates:
                return None
            
            funding_df = pd.DataFrame(rates)
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms', utc=True)
            funding_df = funding_df.set_index('timestamp')
            funding_df = funding_df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
            
            return funding_df
            
        except Exception as e:
            logger.debug(f"Error fetching funding rates: {e}")
            return None
    
    def _fetch_open_interest(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Fetch historical open interest."""
        try:
            import ccxt
            
            exchange = getattr(ccxt, self.exchange)({
                'enableRateLimit': True,
            })
            
            if not hasattr(exchange, 'fetch_open_interest_history'):
                return None
            
            # Normalize symbol for futures
            if '/' not in symbol:
                symbol = f"{symbol}/USDT:USDT"
            
            oi_data = []
            since = int(start.timestamp() * 1000)
            until = int(end.timestamp() * 1000)
            
            while since < until:
                history = exchange.fetch_open_interest_history(
                    symbol,
                    since=since,
                    limit=500
                )
                if not history:
                    break
                oi_data.extend(history)
                since = history[-1]['timestamp'] + 1
            
            if not oi_data:
                return None
            
            oi_df = pd.DataFrame(oi_data)
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms', utc=True)
            oi_df = oi_df.set_index('timestamp')
            oi_df = oi_df[['openInterestValue']].rename(columns={'openInterestValue': 'open_interest'})
            
            return oi_df
            
        except Exception as e:
            logger.debug(f"Error fetching open interest: {e}")
            return None

