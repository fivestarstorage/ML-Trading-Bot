"""
Data utility functions for the Alpha Research Engine.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    fill_value: float = 0.0
) -> Union[pd.Series, np.ndarray, float]:
    """
    Safely divide two arrays, handling division by zero.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division with zeros handled
    """
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        return fill_value if denominator == 0 else numerator / denominator
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        
    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    else:
        result = np.where(np.isfinite(result), result, fill_value)
        
    return result


def winsorize(
    data: Union[pd.Series, np.ndarray],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> Union[pd.Series, np.ndarray]:
    """
    Winsorize data by clipping extreme values.
    
    Args:
        data: Input data
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        
    Returns:
        Winsorized data
    """
    if isinstance(data, pd.Series):
        lower = data.quantile(lower_percentile / 100)
        upper = data.quantile(upper_percentile / 100)
        return data.clip(lower=lower, upper=upper)
    else:
        lower = np.percentile(data[~np.isnan(data)], lower_percentile)
        upper = np.percentile(data[~np.isnan(data)], upper_percentile)
        return np.clip(data, lower, upper)


def zscore(
    data: Union[pd.Series, np.ndarray],
    ddof: int = 1
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate z-score of data.
    
    Args:
        data: Input data
        ddof: Degrees of freedom for std calculation
        
    Returns:
        Z-scored data
    """
    if isinstance(data, pd.Series):
        mean = data.mean()
        std = data.std(ddof=ddof)
    else:
        mean = np.nanmean(data)
        std = np.nanstd(data, ddof=ddof)
    
    return safe_divide(data - mean, std, fill_value=0.0)


def rolling_zscore(
    data: Union[pd.Series, np.ndarray],
    window: int = 50,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling z-score.
    
    Args:
        data: Input data
        window: Rolling window size
        min_periods: Minimum periods for calculation
        
    Returns:
        Rolling z-score Series
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    if min_periods is None:
        min_periods = window // 2
    
    rolling_mean = data.rolling(window, min_periods=min_periods).mean()
    rolling_std = data.rolling(window, min_periods=min_periods).std()
    
    return safe_divide(data - rolling_mean, rolling_std, fill_value=0.0)


def lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    prefix: str = "lag"
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag values
        prefix: Prefix for lag column names
        
    Returns:
        DataFrame with added lag columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            result[f"{prefix}_{col}_{lag}"] = df[col].shift(lag)
    
    return result


def create_return_labels(
    prices: pd.Series,
    forward_window: int = 1,
    threshold: float = 0.0,
    label_type: str = "binary"
) -> pd.Series:
    """
    Create return-based labels for supervised learning.
    
    Args:
        prices: Price series
        forward_window: Number of periods to look forward
        threshold: Threshold for positive/negative classification
        label_type: "binary", "ternary", or "continuous"
        
    Returns:
        Label series
    """
    forward_returns = prices.pct_change(forward_window).shift(-forward_window)
    
    if label_type == "binary":
        return (forward_returns > threshold).astype(int)
    elif label_type == "ternary":
        labels = pd.Series(0, index=prices.index)
        labels[forward_returns > threshold] = 1
        labels[forward_returns < -threshold] = -1
        return labels
    elif label_type == "continuous":
        return forward_returns
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


def calculate_returns(
    prices: pd.Series,
    return_type: str = "simple"
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        return_type: "simple" or "log"
        
    Returns:
        Returns series
    """
    if return_type == "simple":
        return prices.pct_change()
    elif return_type == "log":
        return np.log(prices).diff()
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.
    
    Args:
        df: DataFrame with OHLCV columns and datetime index
        target_freq: Target frequency (e.g., '1H', '4H', '1D')
        
    Returns:
        Resampled DataFrame
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    return df.resample(target_freq).agg(agg_dict).dropna()


def detect_outliers(
    data: Union[pd.Series, np.ndarray],
    method: str = "zscore",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.
    
    Args:
        data: Input data
        method: "zscore" or "iqr"
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if method == "zscore":
        z = np.abs((data - np.nanmean(data)) / np.nanstd(data))
        return z > threshold
    elif method == "iqr":
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    else:
        raise ValueError(f"Unknown method: {method}")


def align_dataframes(
    dfs: List[pd.DataFrame],
    how: str = "inner"
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames on their index.
    
    Args:
        dfs: List of DataFrames to align
        how: Alignment method ("inner" or "outer")
        
    Returns:
        List of aligned DataFrames
    """
    if len(dfs) < 2:
        return dfs
    
    common_index = dfs[0].index
    for df in dfs[1:]:
        if how == "inner":
            common_index = common_index.intersection(df.index)
        else:
            common_index = common_index.union(df.index)
    
    return [df.reindex(common_index) for df in dfs]


def split_train_test_by_time(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    gap_periods: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/test sets by time, avoiding lookahead bias.
    
    Args:
        df: Input DataFrame with datetime index
        train_pct: Percentage for training data
        gap_periods: Number of periods to leave as gap between train/test
        
    Returns:
        Tuple of (train_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_pct)
    test_start = train_end + gap_periods
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[test_start:]
    
    return train_df, test_df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added time features
    """
    result = df.copy()
    
    # Hour of day
    result['hour'] = df.index.hour
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    
    # Day of week
    result['day_of_week'] = df.index.dayofweek
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    
    # Month
    result['month'] = df.index.month
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    # Week of year
    result['week_of_year'] = df.index.isocalendar().week.astype(int)
    
    # Is weekend
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    
    # Session indicators
    hour = result['hour']
    result['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    result['is_london_session'] = ((hour >= 8) & (hour < 16)).astype(int)
    result['is_ny_session'] = ((hour >= 13) & (hour < 21)).astype(int)
    
    return result

