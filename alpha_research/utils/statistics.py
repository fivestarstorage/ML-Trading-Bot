"""
Statistical utilities for the Alpha Research Engine.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sortino ratio (using downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    
    if downside_std == 0:
        return 0.0
    
    return mean_excess / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    equity_curve: Union[pd.Series, np.ndarray]
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    if isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values
    
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    
    max_dd = np.max(drawdowns)
    trough_idx = np.argmax(drawdowns)
    peak_idx = np.argmax(equity_curve[:trough_idx + 1]) if trough_idx > 0 else 0
    
    return float(max_dd), int(peak_idx), int(trough_idx)


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate CAGR
    cumulative = np.cumprod(1 + returns)
    total_return = cumulative[-1] - 1
    years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate max drawdown
    max_dd, _, _ = calculate_max_drawdown(cumulative)
    
    if max_dd == 0:
        return 0.0
    
    return cagr / max_dd


def bootstrap_confidence_interval(
    returns: Union[pd.Series, np.ndarray],
    statistic_func: callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        returns: Series of returns
        statistic_func: Function to calculate the statistic
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    
    np.random.seed(random_seed)
    
    point_estimate = statistic_func(returns)
    
    bootstrap_stats = []
    n = len(returns)
    
    for _ in range(n_iterations):
        sample_idx = np.random.randint(0, n, size=n)
        sample = returns[sample_idx]
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return float(point_estimate), float(lower), float(upper)


def monte_carlo_pvalue(
    observed_returns: Union[pd.Series, np.ndarray],
    n_simulations: int = 1000,
    statistic: str = "sharpe",
    random_seed: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Calculate Monte Carlo p-value by comparing observed statistic to random permutations.
    
    Args:
        observed_returns: Series of actual returns
        n_simulations: Number of Monte Carlo simulations
        statistic: Which statistic to use ("sharpe", "mean", "total_return")
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (p_value, null_distribution)
    """
    if isinstance(observed_returns, pd.Series):
        observed_returns = observed_returns.values
    
    observed_returns = observed_returns[~np.isnan(observed_returns)]
    
    np.random.seed(random_seed)
    
    if statistic == "sharpe":
        observed_stat = calculate_sharpe_ratio(observed_returns, periods_per_year=len(observed_returns))
    elif statistic == "mean":
        observed_stat = np.mean(observed_returns)
    elif statistic == "total_return":
        observed_stat = np.sum(observed_returns)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    null_distribution = []
    
    for _ in range(n_simulations):
        # Randomly shuffle signs to create null hypothesis
        random_signs = np.random.choice([-1, 1], size=len(observed_returns))
        shuffled = observed_returns * random_signs
        
        if statistic == "sharpe":
            sim_stat = calculate_sharpe_ratio(shuffled, periods_per_year=len(shuffled))
        elif statistic == "mean":
            sim_stat = np.mean(shuffled)
        elif statistic == "total_return":
            sim_stat = np.sum(shuffled)
        
        null_distribution.append(sim_stat)
    
    null_distribution = np.array(null_distribution)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))
    
    return float(p_value), null_distribution


def information_ratio(
    strategy_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Information Ratio (excess return over benchmark / tracking error).
    
    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized Information Ratio
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_returns = strategy_returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    excess_returns = strategy_returns - benchmark_returns
    
    mask = ~(np.isnan(excess_returns))
    excess_returns = excess_returns[mask]
    
    if len(excess_returns) < 2:
        return 0.0
    
    tracking_error = np.std(excess_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error * np.sqrt(periods_per_year)


def rolling_statistics(
    returns: Union[pd.Series, np.ndarray],
    window: int = 50,
    min_periods: Optional[int] = None
) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics for a return series.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        min_periods: Minimum periods for calculation
        
    Returns:
        Dictionary of rolling statistics Series
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if min_periods is None:
        min_periods = window // 2
    
    results = {}
    
    results['rolling_mean'] = returns.rolling(window, min_periods=min_periods).mean()
    results['rolling_std'] = returns.rolling(window, min_periods=min_periods).std()
    results['rolling_sharpe'] = results['rolling_mean'] / results['rolling_std'].replace(0, np.nan)
    results['rolling_skew'] = returns.rolling(window, min_periods=min_periods).skew()
    results['rolling_kurtosis'] = returns.rolling(window, min_periods=min_periods).kurt()
    
    # Rolling cumulative return
    results['rolling_cumret'] = (1 + returns).rolling(window, min_periods=min_periods).apply(
        lambda x: np.prod(x) - 1, raw=True
    )
    
    # Rolling max drawdown
    def calc_dd(x):
        cum = np.cumprod(1 + x)
        running_max = np.maximum.accumulate(cum)
        dd = (running_max - cum) / running_max
        return np.max(dd)
    
    results['rolling_maxdd'] = returns.rolling(window, min_periods=min_periods).apply(
        calc_dd, raw=True
    )
    
    return results


def calculate_t_stat(
    returns: Union[pd.Series, np.ndarray],
    null_hypothesis: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate t-statistic and p-value for mean return.
    
    Args:
        returns: Series of returns
        null_hypothesis: Null hypothesis mean value
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_1samp(returns, null_hypothesis)
    
    return float(t_stat), float(p_value)


def calculate_autocorrelation(
    returns: Union[pd.Series, np.ndarray],
    max_lags: int = 20
) -> pd.Series:
    """
    Calculate autocorrelation for multiple lags.
    
    Args:
        returns: Series of returns
        max_lags: Maximum number of lags to calculate
        
    Returns:
        Series of autocorrelation values
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    autocorr = {}
    for lag in range(1, max_lags + 1):
        autocorr[lag] = returns.autocorr(lag=lag)
    
    return pd.Series(autocorr)

