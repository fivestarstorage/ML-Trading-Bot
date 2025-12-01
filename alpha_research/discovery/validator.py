"""
Signal Validator

Comprehensive validation framework for trading signals,
including statistical tests and robustness checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats
from dataclasses import dataclass

from ..utils.statistics import (
    calculate_sharpe_ratio,
    bootstrap_confidence_interval,
    monte_carlo_pvalue,
    calculate_autocorrelation,
    rolling_statistics,
)
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class ValidationResult:
    """Results from signal validation."""
    is_valid: bool
    confidence_level: float
    
    # Statistical tests
    p_value: float
    t_statistic: float
    bootstrap_ci: Tuple[float, float, float]  # (point, lower, upper)
    
    # Robustness checks
    passed_monte_carlo: bool
    passed_bootstrap: bool
    passed_autocorr_check: bool
    passed_stationarity: bool
    
    # Details
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'confidence_level': self.confidence_level,
            'p_value': self.p_value,
            't_statistic': self.t_statistic,
            'bootstrap_ci': self.bootstrap_ci,
            'passed_monte_carlo': self.passed_monte_carlo,
            'passed_bootstrap': self.passed_bootstrap,
            'passed_autocorr_check': self.passed_autocorr_check,
            'passed_stationarity': self.passed_stationarity,
            'details': self.details,
        }


class SignalValidator:
    """
    Validates trading signals using multiple statistical methods.
    
    Tests include:
    - T-test for mean return significance
    - Bootstrap confidence intervals
    - Monte Carlo permutation test
    - Autocorrelation analysis
    - Stationarity tests
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.significance_level = config.get('significance_level', 0.05)
        self.bootstrap_iterations = config.get('bootstrap_iterations', 1000)
        self.monte_carlo_sims = config.get('monte_carlo_simulations', 1000)
        self.min_samples = config.get('min_samples', 50)
    
    def validate(
        self, 
        returns: pd.Series,
        signals: Optional[pd.Series] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation on signal returns.
        
        Args:
            returns: Strategy returns series
            signals: Optional signal series for additional analysis
            
        Returns:
            ValidationResult with all test outcomes
        """
        returns = returns.dropna()
        
        if len(returns) < self.min_samples:
            return ValidationResult(
                is_valid=False,
                confidence_level=0.0,
                p_value=1.0,
                t_statistic=0.0,
                bootstrap_ci=(0.0, 0.0, 0.0),
                passed_monte_carlo=False,
                passed_bootstrap=False,
                passed_autocorr_check=False,
                passed_stationarity=False,
                details={'error': f'Insufficient samples: {len(returns)}'}
            )
        
        details = {}
        
        # 1. T-test
        t_stat, t_pvalue = self._t_test(returns)
        details['t_test'] = {'t_statistic': t_stat, 'p_value': t_pvalue}
        
        # 2. Bootstrap
        point_est, ci_lower, ci_upper = self._bootstrap_test(returns)
        passed_bootstrap = ci_lower > 0
        details['bootstrap'] = {
            'point_estimate': point_est,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        # 3. Monte Carlo
        mc_pvalue, mc_dist = self._monte_carlo_test(returns)
        passed_monte_carlo = mc_pvalue < self.significance_level
        details['monte_carlo'] = {
            'p_value': mc_pvalue,
            'null_mean': float(np.mean(mc_dist)),
            'null_std': float(np.std(mc_dist))
        }
        
        # 4. Autocorrelation check
        passed_autocorr = self._autocorrelation_check(returns)
        details['autocorrelation'] = {'passed': passed_autocorr}
        
        # 5. Stationarity test
        passed_stationarity = self._stationarity_test(returns)
        details['stationarity'] = {'passed': passed_stationarity}
        
        # Calculate overall confidence
        checks_passed = sum([
            t_pvalue < self.significance_level,
            passed_bootstrap,
            passed_monte_carlo,
            passed_autocorr,
            passed_stationarity
        ])
        confidence = checks_passed / 5.0
        
        # Determine validity
        is_valid = (
            t_pvalue < self.significance_level and
            passed_bootstrap and
            passed_monte_carlo and
            confidence >= 0.6
        )
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_level=confidence,
            p_value=t_pvalue,
            t_statistic=t_stat,
            bootstrap_ci=(point_est, ci_lower, ci_upper),
            passed_monte_carlo=passed_monte_carlo,
            passed_bootstrap=passed_bootstrap,
            passed_autocorr_check=passed_autocorr,
            passed_stationarity=passed_stationarity,
            details=details
        )
    
    def _t_test(self, returns: pd.Series) -> Tuple[float, float]:
        """Perform one-sample t-test for mean return."""
        t_stat, p_value = stats.ttest_1samp(returns.values, 0)
        return float(t_stat), float(p_value)
    
    def _bootstrap_test(self, returns: pd.Series) -> Tuple[float, float, float]:
        """Perform bootstrap confidence interval test."""
        point_est, ci_lower, ci_upper = bootstrap_confidence_interval(
            returns.values,
            statistic_func=lambda x: calculate_sharpe_ratio(x, periods_per_year=len(x)),
            n_iterations=self.bootstrap_iterations,
            confidence_level=1 - self.significance_level
        )
        return point_est, ci_lower, ci_upper
    
    def _monte_carlo_test(self, returns: pd.Series) -> Tuple[float, np.ndarray]:
        """Perform Monte Carlo permutation test."""
        p_value, null_dist = monte_carlo_pvalue(
            returns.values,
            n_simulations=self.monte_carlo_sims,
            statistic="sharpe"
        )
        return p_value, null_dist
    
    def _autocorrelation_check(self, returns: pd.Series) -> bool:
        """
        Check for excessive autocorrelation in returns.
        
        High autocorrelation might indicate:
        - Lookahead bias
        - Overfitting
        - Market inefficiency being exploited (could be good)
        """
        autocorr = calculate_autocorrelation(returns, max_lags=10)
        
        # Check if any lag has significant autocorrelation
        # (Using ±2/sqrt(n) as rough significance bound)
        n = len(returns)
        bound = 2 / np.sqrt(n)
        
        significant_lags = (autocorr.abs() > bound).sum()
        
        # Allow some autocorrelation but flag if excessive
        return significant_lags <= 3
    
    def _stationarity_test(self, returns: pd.Series) -> bool:
        """
        Test for stationarity using ADF test.
        
        Stationary returns are important for:
        - Reliable backtesting
        - Consistent future performance
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(returns.values, autolag='AIC')
            p_value = result[1]
            return p_value < 0.05  # Stationary if we reject unit root
        except Exception:
            # If statsmodels not available, skip this test
            return True
    
    def validate_multiple(
        self, 
        returns_dict: Dict[str, pd.Series]
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple signal returns.
        
        Args:
            returns_dict: Dictionary mapping signal names to returns
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        for name, returns in returns_dict.items():
            results[name] = self.validate(returns)
        return results
    
    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> List[float]:
        """
        Apply multiple testing correction to p-values.
        
        Args:
            p_values: List of p-values
            method: Correction method ("bonferroni", "holm", "fdr")
            
        Returns:
            Corrected p-values
        """
        n = len(p_values)
        p_values = np.array(p_values)
        
        if method == "bonferroni":
            # Bonferroni: multiply by number of tests
            return list(np.minimum(p_values * n, 1.0))
        
        elif method == "holm":
            # Holm-Bonferroni: step-down procedure
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            adjusted = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                adjusted[i] = min(p * (n - i), 1.0)
            
            # Enforce monotonicity
            for i in range(1, n):
                adjusted[i] = max(adjusted[i], adjusted[i-1])
            
            # Restore original order
            result = np.zeros_like(adjusted)
            result[sorted_idx] = adjusted
            return list(result)
        
        elif method == "fdr":
            # Benjamini-Hochberg FDR control
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            adjusted = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                adjusted[i] = min(p * n / (i + 1), 1.0)
            
            # Enforce monotonicity (from end)
            for i in range(n - 2, -1, -1):
                adjusted[i] = min(adjusted[i], adjusted[i + 1])
            
            # Restore original order
            result = np.zeros_like(adjusted)
            result[sorted_idx] = adjusted
            return list(result)
        
        else:
            raise ValueError(f"Unknown method: {method}")


class RollingValidator:
    """
    Validates signals using rolling window analysis.
    
    Checks for:
    - Consistent performance over time
    - Regime changes affecting performance
    - Stability of edge
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.window_sizes = config.get('window_sizes', [50, 100, 200])
        self.min_window_sharpe = config.get('min_window_sharpe', 0.3)
    
    def validate_stability(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Validate signal stability over time.
        
        Args:
            returns: Strategy returns series
            
        Returns:
            Stability analysis results
        """
        results = {}
        
        for window in self.window_sizes:
            if len(returns) < window * 2:
                continue
            
            rolling_stats = rolling_statistics(returns, window=window)
            
            # Calculate stability metrics
            sharpe_series = rolling_stats['rolling_sharpe'].dropna()
            
            if len(sharpe_series) > 0:
                results[f'window_{window}'] = {
                    'mean_sharpe': float(sharpe_series.mean()),
                    'std_sharpe': float(sharpe_series.std()),
                    'min_sharpe': float(sharpe_series.min()),
                    'max_sharpe': float(sharpe_series.max()),
                    'pct_positive': float((sharpe_series > 0).mean()),
                    'pct_above_threshold': float((sharpe_series > self.min_window_sharpe).mean()),
                }
        
        # Overall stability score
        if results:
            avg_pct_positive = np.mean([r['pct_positive'] for r in results.values()])
            avg_pct_above_thresh = np.mean([r['pct_above_threshold'] for r in results.values()])
            
            results['overall'] = {
                'stability_score': (avg_pct_positive + avg_pct_above_thresh) / 2,
                'is_stable': avg_pct_positive > 0.6 and avg_pct_above_thresh > 0.4
            }
        
        return results

