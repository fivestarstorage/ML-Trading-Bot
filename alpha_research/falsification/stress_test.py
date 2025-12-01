"""
Stress Testing Module

Applies rigorous stress tests to edge candidates to
identify and eliminate spurious edges.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats

from ..utils.statistics import calculate_sharpe_ratio, calculate_max_drawdown
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class StressTestResult:
    """Results from a single stress test."""
    test_name: str
    passed: bool
    baseline_metric: float
    stressed_metric: float
    degradation_pct: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FalsificationResult:
    """Complete falsification results for an edge."""
    edge_id: str
    edge_name: str
    
    # Overall result
    passed_falsification: bool
    falsification_score: float  # 0-1, higher is more robust
    
    # Individual test results
    test_results: List[StressTestResult] = field(default_factory=list)
    
    # Summary statistics
    tests_passed: int = 0
    tests_failed: int = 0
    
    # Key findings
    vulnerabilities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


class StressTester:
    """
    Comprehensive stress testing for edge candidates.
    
    Tests include:
    1. Slippage stress test
    2. Entry timing randomization
    3. Label shifting
    4. Feature noise injection
    5. Rolling window stability
    6. Regime robustness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stress tester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        falsification_config = config.get('falsification', {})
        
        # Slippage settings
        self.slippage_multipliers = falsification_config.get(
            'slippage_multipliers', [1.0, 1.5, 2.0, 3.0, 5.0]
        )
        
        # Entry timing
        self.entry_noise_bars = falsification_config.get(
            'entry_noise_bars', [0, 1, 2, 3]
        )
        self.entry_noise_iterations = falsification_config.get(
            'entry_noise_iterations', 100
        )
        
        # Label shifting
        self.label_shift_bars = falsification_config.get(
            'label_shift_bars', [-3, -2, -1, 1, 2, 3]
        )
        
        # Noise injection
        self.noise_levels = falsification_config.get(
            'noise_levels', [0.01, 0.02, 0.05, 0.1]
        )
        self.noise_iterations = falsification_config.get(
            'noise_iterations', 50
        )
        
        # Rolling window
        self.window_sizes = falsification_config.get(
            'window_sizes', [100, 200, 500, 1000]
        )
        self.min_window_sharpe = falsification_config.get(
            'min_window_sharpe', 0.3
        )
        self.max_sharpe_variance = falsification_config.get(
            'max_sharpe_variance', 1.0
        )
        
        # Regime settings
        self.min_regime_count = falsification_config.get(
            'min_regime_count', 2
        )
        self.max_regime_sharpe_diff = falsification_config.get(
            'max_regime_sharpe_diff', 1.5
        )
        
        # Thresholds
        self.max_degradation_pct = 0.5  # Max 50% performance degradation
        self.min_passing_tests = 0.7   # 70% of tests must pass
    
    def run_full_falsification(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        hypothesis_name: str,
        hypothesis_id: str
    ) -> FalsificationResult:
        """
        Run all falsification tests on an edge.
        
        Args:
            data: OHLCV data
            signals: Trading signals
            hypothesis_name: Name of the hypothesis
            hypothesis_id: ID of the hypothesis
            
        Returns:
            Complete falsification result
        """
        logger.info(f"Running falsification tests on: {hypothesis_name}")
        
        # Calculate baseline performance
        returns = signals * data['close'].pct_change().shift(-1)
        baseline_sharpe = calculate_sharpe_ratio(
            returns.dropna().values,
            periods_per_year=252
        )
        
        test_results = []
        
        # 1. Slippage stress test
        slippage_result = self._test_slippage(data, signals, baseline_sharpe)
        test_results.append(slippage_result)
        
        # 2. Entry timing randomization
        timing_result = self._test_entry_timing(data, signals, baseline_sharpe)
        test_results.append(timing_result)
        
        # 3. Label shifting
        label_result = self._test_label_shifting(data, signals, baseline_sharpe)
        test_results.append(label_result)
        
        # 4. Feature noise injection
        noise_result = self._test_noise_injection(data, signals, baseline_sharpe)
        test_results.append(noise_result)
        
        # 5. Rolling window stability
        stability_result = self._test_rolling_stability(data, signals)
        test_results.append(stability_result)
        
        # 6. Regime robustness
        regime_result = self._test_regime_robustness(data, signals)
        test_results.append(regime_result)
        
        # Calculate overall result
        tests_passed = sum(1 for r in test_results if r.passed)
        tests_failed = len(test_results) - tests_passed
        
        falsification_score = tests_passed / len(test_results)
        passed_falsification = (
            falsification_score >= self.min_passing_tests and
            baseline_sharpe > 0
        )
        
        # Identify vulnerabilities and strengths
        vulnerabilities = [r.test_name for r in test_results if not r.passed]
        strengths = [r.test_name for r in test_results if r.passed and r.degradation_pct < 0.1]
        
        return FalsificationResult(
            edge_id=hypothesis_id,
            edge_name=hypothesis_name,
            passed_falsification=passed_falsification,
            falsification_score=falsification_score,
            test_results=test_results,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            vulnerabilities=vulnerabilities,
            strengths=strengths
        )
    
    def _test_slippage(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        baseline_sharpe: float
    ) -> StressTestResult:
        """Test sensitivity to slippage assumptions."""
        
        forward_returns = data['close'].pct_change().shift(-1)
        signal_changes = signals.diff().abs()
        
        stressed_sharpes = []
        
        for multiplier in self.slippage_multipliers:
            # Apply increased slippage
            slippage_cost = signal_changes * 0.0002 * multiplier  # 2bp base slippage
            net_returns = signals * forward_returns - slippage_cost
            
            sharpe = calculate_sharpe_ratio(
                net_returns.dropna().values,
                periods_per_year=252
            )
            stressed_sharpes.append(sharpe)
        
        # Check worst case
        worst_sharpe = min(stressed_sharpes)
        degradation = (baseline_sharpe - worst_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 1.0
        
        passed = worst_sharpe > 0 and degradation < self.max_degradation_pct
        
        return StressTestResult(
            test_name="Slippage Stress Test",
            passed=passed,
            baseline_metric=baseline_sharpe,
            stressed_metric=worst_sharpe,
            degradation_pct=degradation,
            details={
                'multipliers_tested': self.slippage_multipliers,
                'sharpes': stressed_sharpes,
            }
        )
    
    def _test_entry_timing(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        baseline_sharpe: float
    ) -> StressTestResult:
        """Test sensitivity to entry timing."""
        
        forward_returns = data['close'].pct_change().shift(-1)
        
        timing_sharpes = []
        
        for _ in range(self.entry_noise_iterations):
            # Randomly shift signal entries
            noise_bars = np.random.choice(self.entry_noise_bars, size=len(signals))
            
            # Create noisy signals
            noisy_signals = signals.copy()
            for i, shift in enumerate(noise_bars):
                if i + shift < len(signals) and i + shift >= 0:
                    noisy_signals.iloc[i] = signals.iloc[i + shift]
            
            returns = noisy_signals * forward_returns
            sharpe = calculate_sharpe_ratio(
                returns.dropna().values,
                periods_per_year=252
            )
            timing_sharpes.append(sharpe)
        
        mean_sharpe = np.mean(timing_sharpes)
        std_sharpe = np.std(timing_sharpes)
        worst_sharpe = np.percentile(timing_sharpes, 5)  # 5th percentile
        
        degradation = (baseline_sharpe - worst_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 1.0
        
        passed = worst_sharpe > 0 and degradation < self.max_degradation_pct
        
        return StressTestResult(
            test_name="Entry Timing Randomization",
            passed=passed,
            baseline_metric=baseline_sharpe,
            stressed_metric=worst_sharpe,
            degradation_pct=degradation,
            details={
                'mean_sharpe': mean_sharpe,
                'std_sharpe': std_sharpe,
                'iterations': self.entry_noise_iterations,
            }
        )
    
    def _test_label_shifting(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        baseline_sharpe: float
    ) -> StressTestResult:
        """
        Test if edge depends on exact label timing.
        
        If shifting labels by a few bars destroys the edge,
        it may be due to lookahead bias or overfitting.
        """
        
        shift_sharpes = {}
        
        for shift in self.label_shift_bars:
            # Shift the forward returns (simulating label timing error)
            shifted_returns = data['close'].pct_change().shift(-1 + shift)
            returns = signals * shifted_returns
            
            sharpe = calculate_sharpe_ratio(
                returns.dropna().values,
                periods_per_year=252
            )
            shift_sharpes[shift] = sharpe
        
        # The edge should be somewhat robust to small shifts
        # If it completely breaks, that's suspicious
        positive_shifts = sum(1 for s in shift_sharpes.values() if s > 0)
        avg_sharpe = np.mean(list(shift_sharpes.values()))
        
        degradation = (baseline_sharpe - avg_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 1.0
        
        # Passed if at least half the shifts are positive and avg is positive
        passed = positive_shifts >= len(self.label_shift_bars) // 2 and avg_sharpe > 0
        
        return StressTestResult(
            test_name="Label Shifting Test",
            passed=passed,
            baseline_metric=baseline_sharpe,
            stressed_metric=avg_sharpe,
            degradation_pct=degradation,
            details={
                'shift_sharpes': shift_sharpes,
                'positive_shifts': positive_shifts,
            }
        )
    
    def _test_noise_injection(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        baseline_sharpe: float
    ) -> StressTestResult:
        """Test sensitivity to feature noise."""
        
        forward_returns = data['close'].pct_change().shift(-1)
        
        noise_results = {level: [] for level in self.noise_levels}
        
        for level in self.noise_levels:
            for _ in range(self.noise_iterations):
                # Add random noise to signals
                noise = np.random.normal(0, level, len(signals))
                noisy_signals = np.clip(signals + noise, -1, 1)
                
                returns = noisy_signals * forward_returns.values
                sharpe = calculate_sharpe_ratio(
                    returns[~np.isnan(returns)],
                    periods_per_year=252
                )
                noise_results[level].append(sharpe)
        
        # Check performance at highest noise level
        highest_noise = max(self.noise_levels)
        mean_at_highest = np.mean(noise_results[highest_noise])
        worst_at_highest = np.percentile(noise_results[highest_noise], 5)
        
        degradation = (baseline_sharpe - worst_at_highest) / abs(baseline_sharpe) if baseline_sharpe != 0 else 1.0
        
        passed = mean_at_highest > 0 and degradation < 0.7  # Allow more degradation for noise
        
        return StressTestResult(
            test_name="Feature Noise Injection",
            passed=passed,
            baseline_metric=baseline_sharpe,
            stressed_metric=mean_at_highest,
            degradation_pct=degradation,
            details={
                'noise_levels': self.noise_levels,
                'mean_sharpes': {k: np.mean(v) for k, v in noise_results.items()},
            }
        )
    
    def _test_rolling_stability(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> StressTestResult:
        """Test stability across rolling windows."""
        
        forward_returns = data['close'].pct_change().shift(-1)
        strategy_returns = signals * forward_returns
        
        window_results = {}
        
        for window in self.window_sizes:
            if len(strategy_returns) < window * 2:
                continue
            
            rolling_sharpes = []
            
            for start in range(0, len(strategy_returns) - window, window // 2):
                end = start + window
                window_returns = strategy_returns.iloc[start:end].dropna()
                
                if len(window_returns) > 20:
                    sharpe = calculate_sharpe_ratio(
                        window_returns.values,
                        periods_per_year=252
                    )
                    rolling_sharpes.append(sharpe)
            
            if rolling_sharpes:
                window_results[window] = {
                    'mean': np.mean(rolling_sharpes),
                    'std': np.std(rolling_sharpes),
                    'min': min(rolling_sharpes),
                    'pct_positive': sum(1 for s in rolling_sharpes if s > 0) / len(rolling_sharpes)
                }
        
        if not window_results:
            return StressTestResult(
                test_name="Rolling Window Stability",
                passed=False,
                baseline_metric=0,
                stressed_metric=0,
                degradation_pct=1.0,
                details={'error': 'Insufficient data'}
            )
        
        # Check stability criteria
        avg_pct_positive = np.mean([r['pct_positive'] for r in window_results.values()])
        avg_sharpe_std = np.mean([r['std'] for r in window_results.values()])
        
        passed = (
            avg_pct_positive >= 0.6 and  # 60% of windows positive
            avg_sharpe_std < self.max_sharpe_variance  # Not too variable
        )
        
        return StressTestResult(
            test_name="Rolling Window Stability",
            passed=passed,
            baseline_metric=avg_pct_positive,
            stressed_metric=avg_sharpe_std,
            degradation_pct=1 - avg_pct_positive,
            details={'window_results': window_results}
        )
    
    def _test_regime_robustness(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> StressTestResult:
        """Test performance across different market regimes."""
        
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()
        
        forward_returns = data['close'].pct_change().shift(-1)
        
        # Define regimes
        vol = data['returns'].rolling(20).std()
        vol_median = vol.median()
        
        trend = data['close'].rolling(50).mean()
        price_above_trend = data['close'] > trend
        
        # Create regime labels
        regimes = pd.Series('neutral', index=data.index)
        regimes[(vol > vol_median) & price_above_trend] = 'high_vol_uptrend'
        regimes[(vol > vol_median) & ~price_above_trend] = 'high_vol_downtrend'
        regimes[(vol <= vol_median) & price_above_trend] = 'low_vol_uptrend'
        regimes[(vol <= vol_median) & ~price_above_trend] = 'low_vol_downtrend'
        
        # Calculate performance per regime
        regime_sharpes = {}
        strategy_returns = signals * forward_returns
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_returns = strategy_returns[regime_mask].dropna()
            
            if len(regime_returns) >= 30:
                sharpe = calculate_sharpe_ratio(
                    regime_returns.values,
                    periods_per_year=252
                )
                regime_sharpes[regime] = sharpe
        
        if len(regime_sharpes) < self.min_regime_count:
            return StressTestResult(
                test_name="Regime Robustness",
                passed=False,
                baseline_metric=0,
                stressed_metric=0,
                degradation_pct=1.0,
                details={'error': f'Only {len(regime_sharpes)} regimes had sufficient data'}
            )
        
        # Check regime consistency
        sharpe_values = list(regime_sharpes.values())
        min_sharpe = min(sharpe_values)
        max_sharpe = max(sharpe_values)
        sharpe_range = max_sharpe - min_sharpe
        
        positive_regimes = sum(1 for s in sharpe_values if s > 0)
        
        passed = (
            positive_regimes >= len(sharpe_values) * 0.5 and  # At least half positive
            sharpe_range < self.max_regime_sharpe_diff  # Not too variable
        )
        
        return StressTestResult(
            test_name="Regime Robustness",
            passed=passed,
            baseline_metric=np.mean(sharpe_values),
            stressed_metric=min_sharpe,
            degradation_pct=sharpe_range / max(abs(max_sharpe), 0.01),
            details={
                'regime_sharpes': regime_sharpes,
                'positive_regimes': positive_regimes,
                'total_regimes': len(regime_sharpes),
            }
        )

