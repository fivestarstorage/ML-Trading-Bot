"""
Robustness Analyzer

Provides detailed robustness analysis for edge candidates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from ..utils.statistics import calculate_sharpe_ratio, rolling_statistics
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class RobustnessMetrics:
    """Comprehensive robustness metrics for an edge."""
    
    # Stability metrics
    sharpe_stability: float  # 1 / (1 + std of rolling Sharpe)
    drawdown_consistency: float  # How consistent are drawdowns
    recovery_time_avg: float  # Average time to recover from drawdowns
    
    # Regime metrics
    best_regime: str
    worst_regime: str
    regime_consistency: float  # Variance across regimes
    
    # Parameter sensitivity
    parameter_sensitivity: float  # How sensitive to parameter changes
    
    # Overall score
    robustness_score: float


class RobustnessAnalyzer:
    """
    Analyzes robustness of trading edges from multiple angles.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rolling_windows = [50, 100, 200, 500]
    
    def analyze(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        hypothesis_params: Dict[str, Any] = None
    ) -> RobustnessMetrics:
        """
        Perform comprehensive robustness analysis.
        
        Args:
            data: OHLCV data
            signals: Trading signals
            hypothesis_params: Optional parameters for sensitivity analysis
            
        Returns:
            RobustnessMetrics
        """
        # Calculate strategy returns
        forward_returns = data['close'].pct_change().shift(-1)
        strategy_returns = signals * forward_returns
        
        # Stability analysis
        sharpe_stability = self._calculate_sharpe_stability(strategy_returns)
        dd_consistency = self._calculate_drawdown_consistency(strategy_returns)
        recovery_avg = self._calculate_recovery_time(strategy_returns)
        
        # Regime analysis
        best_regime, worst_regime, regime_consistency = self._analyze_regimes(
            data, signals, forward_returns
        )
        
        # Parameter sensitivity
        param_sensitivity = self._analyze_parameter_sensitivity(
            data, signals, forward_returns, hypothesis_params
        )
        
        # Calculate overall score
        robustness_score = self._calculate_overall_score(
            sharpe_stability, dd_consistency, regime_consistency, param_sensitivity
        )
        
        return RobustnessMetrics(
            sharpe_stability=sharpe_stability,
            drawdown_consistency=dd_consistency,
            recovery_time_avg=recovery_avg,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_consistency=regime_consistency,
            parameter_sensitivity=param_sensitivity,
            robustness_score=robustness_score
        )
    
    def _calculate_sharpe_stability(self, returns: pd.Series) -> float:
        """Calculate stability of rolling Sharpe ratio."""
        returns = returns.dropna()
        
        if len(returns) < 100:
            return 0.0
        
        rolling_sharpes = []
        
        for window in self.rolling_windows:
            if len(returns) >= window * 2:
                stats = rolling_statistics(returns, window=window)
                sharpe_series = stats['rolling_sharpe'].dropna()
                if len(sharpe_series) > 0:
                    rolling_sharpes.extend(sharpe_series.values)
        
        if not rolling_sharpes:
            return 0.0
        
        # Stability = inverse of variance
        std = np.std(rolling_sharpes)
        stability = 1 / (1 + std)
        
        return float(stability)
    
    def _calculate_drawdown_consistency(self, returns: pd.Series) -> float:
        """Calculate consistency of drawdowns."""
        returns = returns.dropna()
        
        if len(returns) < 50:
            return 0.0
        
        # Calculate equity curve
        equity = (1 + returns).cumprod()
        
        # Calculate drawdown series
        running_max = equity.expanding().max()
        drawdown = (running_max - equity) / running_max
        
        # Find drawdown episodes
        dd_starts = []
        dd_magnitudes = []
        
        in_drawdown = False
        dd_start = None
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] > 0.01:  # 1% threshold
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
            else:
                if in_drawdown:
                    dd_starts.append(dd_start)
                    dd_magnitudes.append(drawdown.iloc[dd_start:i].max())
                    in_drawdown = False
        
        if not dd_magnitudes:
            return 1.0  # No significant drawdowns
        
        # Consistency = inverse of coefficient of variation
        cv = np.std(dd_magnitudes) / np.mean(dd_magnitudes) if np.mean(dd_magnitudes) > 0 else 0
        consistency = 1 / (1 + cv)
        
        return float(consistency)
    
    def _calculate_recovery_time(self, returns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        returns = returns.dropna()
        
        if len(returns) < 50:
            return float('inf')
        
        equity = (1 + returns).cumprod()
        running_max = equity.expanding().max()
        
        recovery_times = []
        
        in_drawdown = False
        dd_start = None
        
        for i in range(len(equity)):
            if equity.iloc[i] < running_max.iloc[i] * 0.99:  # 1% threshold
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
            else:
                if in_drawdown:
                    recovery_times.append(i - dd_start)
                    in_drawdown = False
        
        if not recovery_times:
            return 0.0  # No drawdowns
        
        return float(np.mean(recovery_times))
    
    def _analyze_regimes(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        forward_returns: pd.Series
    ) -> Tuple[str, str, float]:
        """Analyze performance across regimes."""
        
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()
        
        # Define regimes
        vol = data['returns'].rolling(20).std()
        vol_median = vol.median()
        
        trend = data['returns'].rolling(50).mean()
        
        regime_sharpes = {}
        
        # High/Low volatility
        for vol_label, vol_mask in [('high_vol', vol > vol_median), ('low_vol', vol <= vol_median)]:
            regime_returns = (signals * forward_returns)[vol_mask].dropna()
            if len(regime_returns) >= 30:
                sharpe = calculate_sharpe_ratio(regime_returns.values, periods_per_year=252)
                regime_sharpes[vol_label] = sharpe
        
        # Trending/Mean-reverting
        for trend_label, trend_mask in [
            ('uptrend', trend > 0.0001),
            ('downtrend', trend < -0.0001),
            ('sideways', (trend >= -0.0001) & (trend <= 0.0001))
        ]:
            regime_returns = (signals * forward_returns)[trend_mask].dropna()
            if len(regime_returns) >= 30:
                sharpe = calculate_sharpe_ratio(regime_returns.values, periods_per_year=252)
                regime_sharpes[trend_label] = sharpe
        
        if not regime_sharpes:
            return 'unknown', 'unknown', 0.0
        
        best_regime = max(regime_sharpes, key=regime_sharpes.get)
        worst_regime = min(regime_sharpes, key=regime_sharpes.get)
        
        # Consistency = inverse of coefficient of variation
        sharpe_values = list(regime_sharpes.values())
        if np.mean(sharpe_values) != 0:
            cv = np.std(sharpe_values) / abs(np.mean(sharpe_values))
        else:
            cv = float('inf')
        consistency = 1 / (1 + cv)
        
        return best_regime, worst_regime, float(consistency)
    
    def _analyze_parameter_sensitivity(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        forward_returns: pd.Series,
        params: Dict[str, Any] = None
    ) -> float:
        """
        Analyze sensitivity to parameter changes.
        
        Since we don't have direct access to hypothesis parameters,
        we test sensitivity by perturbing the signals themselves.
        """
        base_sharpe = calculate_sharpe_ratio(
            (signals * forward_returns).dropna().values,
            periods_per_year=252
        )
        
        # Perturb signals
        perturbation_levels = [0.05, 0.10, 0.15, 0.20]
        perturbed_sharpes = []
        
        for level in perturbation_levels:
            for _ in range(10):
                # Add random perturbation
                noise = np.random.uniform(-level, level, len(signals))
                perturbed = np.clip(signals + noise, -1, 1)
                
                returns = perturbed * forward_returns.values
                sharpe = calculate_sharpe_ratio(
                    returns[~np.isnan(returns)],
                    periods_per_year=252
                )
                perturbed_sharpes.append(sharpe)
        
        # Sensitivity = how much performance degrades with perturbation
        mean_perturbed = np.mean(perturbed_sharpes)
        
        if base_sharpe != 0:
            sensitivity = abs(base_sharpe - mean_perturbed) / abs(base_sharpe)
        else:
            sensitivity = 1.0
        
        # Invert to get robustness score (lower sensitivity = more robust)
        return float(1 - min(sensitivity, 1))
    
    def _calculate_overall_score(
        self,
        sharpe_stability: float,
        dd_consistency: float,
        regime_consistency: float,
        param_robustness: float
    ) -> float:
        """Calculate overall robustness score."""
        
        weights = {
            'sharpe_stability': 0.3,
            'dd_consistency': 0.2,
            'regime_consistency': 0.3,
            'param_robustness': 0.2
        }
        
        score = (
            weights['sharpe_stability'] * sharpe_stability +
            weights['dd_consistency'] * dd_consistency +
            weights['regime_consistency'] * regime_consistency +
            weights['param_robustness'] * param_robustness
        )
        
        return float(score)


class OutOfSampleValidator:
    """
    Validates edges on out-of-sample data.
    """
    
    def __init__(self, holdout_pct: float = 0.2):
        self.holdout_pct = holdout_pct
    
    def validate(
        self,
        data: pd.DataFrame,
        signals_func: callable,
        min_sharpe: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate on held-out data.
        
        Args:
            data: Full dataset
            signals_func: Function that generates signals from data
            min_sharpe: Minimum acceptable OOS Sharpe
            
        Returns:
            Validation results
        """
        n = len(data)
        train_end = int(n * (1 - self.holdout_pct))
        
        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:]
        
        # Generate signals on test data
        test_signals = signals_func(test_data)
        
        # Calculate performance
        forward_returns = test_data['close'].pct_change().shift(-1)
        test_returns = test_signals * forward_returns
        
        oos_sharpe = calculate_sharpe_ratio(
            test_returns.dropna().values,
            periods_per_year=252
        )
        
        equity = (1 + test_returns.dropna()).cumprod()
        total_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0
        
        passed = oos_sharpe >= min_sharpe and total_return > 0
        
        return {
            'passed': passed,
            'oos_sharpe': oos_sharpe,
            'total_return': total_return,
            'n_test_samples': len(test_data),
            'holdout_pct': self.holdout_pct,
        }

