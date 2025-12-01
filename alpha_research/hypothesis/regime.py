"""
Regime-based hypothesis generation.

Focuses on detecting regime changes and volatility clusters
that may indicate exploitable edges.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .base import (
    HypothesisGenerator, 
    Hypothesis, 
    HypothesisType, 
    HypothesisMechanism
)


class RegimeHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses based on market regime detection.
    
    Key patterns searched:
    - Volatility regimes (low/medium/high vol)
    - Trend regimes (trending vs mean-reverting)
    - Momentum regimes
    - Volatility clustering and mean reversion
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback = config.get('regime_lookback', 100)
        self.volatility_threshold = config.get('volatility_cluster_threshold', 1.5)
        self.n_regimes = config.get('n_regimes', 3)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.REGIME
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate regime-based hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Volatility regime hypotheses
        vol_hypotheses = self._generate_volatility_hypotheses(data)
        hypotheses.extend(vol_hypotheses)
        
        # 2. Trend regime hypotheses
        trend_hypotheses = self._generate_trend_hypotheses(data)
        hypotheses.extend(trend_hypotheses)
        
        # 3. Volatility mean reversion hypothesis
        vol_mr = self._generate_volatility_mean_reversion(data)
        if vol_mr:
            hypotheses.append(vol_mr)
        
        # 4. Regime transition hypotheses
        transition_hypotheses = self._generate_transition_hypotheses(data)
        hypotheses.extend(transition_hypotheses)
        
        # 5. Hidden Markov regime hypotheses
        hmm_hypotheses = self._generate_hmm_regimes(data)
        hypotheses.extend(hmm_hypotheses)
        
        return hypotheses
    
    def _generate_volatility_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on volatility regimes."""
        hypotheses = []
        
        # Calculate volatility measures
        returns = data['returns'].dropna()
        rolling_vol = returns.rolling(self.lookback).std()
        vol_zscore = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()
        
        # Store in data for signal generation
        data['rolling_vol'] = rolling_vol
        data['vol_zscore'] = vol_zscore
        
        lookback = self.lookback  # Capture for closure
        
        # Hypothesis 1: Low volatility breakout
        # After periods of low volatility, expect increased momentum
        def low_vol_breakout_signal(df, vol_threshold=-1.0, momentum_lookback=5, lookback_window=100):
            signals = pd.Series(0, index=df.index)
            
            # Calculate vol_zscore internally
            returns = df['close'].pct_change()
            rolling_vol = returns.rolling(lookback_window).std()
            vol_zscore = (rolling_vol - rolling_vol.rolling(lookback_window).mean()) / rolling_vol.rolling(lookback_window).std()
            
            low_vol = vol_zscore < vol_threshold
            momentum = df['close'].pct_change(momentum_lookback)
            
            # Long on positive momentum after low vol
            signals[(low_vol.shift(1)) & (momentum > 0)] = 1
            # Short on negative momentum after low vol
            signals[(low_vol.shift(1)) & (momentum < 0)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("low_vol_breakout"),
            name="Low Volatility Breakout",
            description="After periods of unusually low volatility, momentum signals "
                       "tend to have higher predictive power as compression resolves.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=low_vol_breakout_signal,
            parameters={'vol_threshold': -1.0, 'momentum_lookback': 5, 'lookback_window': lookback},
            required_features=['close'],
            economic_rationale="Low volatility periods often precede significant moves. "
                              "Market participants become complacent, reducing hedging. "
                              "When vol expands, the move tends to have follow-through.",
            source="regime_generator",
            priority=7
        ))
        
        # Hypothesis 2: High volatility mean reversion
        def high_vol_reversion_signal(df, vol_threshold=1.5, reversion_lookback=3, lookback_window=100):
            signals = pd.Series(0, index=df.index)
            
            # Calculate vol_zscore internally
            returns = df['close'].pct_change()
            rolling_vol = returns.rolling(lookback_window).std()
            vol_zscore = (rolling_vol - rolling_vol.rolling(lookback_window).mean()) / rolling_vol.rolling(lookback_window).std()
            
            high_vol = vol_zscore > vol_threshold
            short_return = df['close'].pct_change(reversion_lookback)
            
            # Fade extreme moves in high volatility
            signals[(high_vol) & (short_return > 0.02)] = -1
            signals[(high_vol) & (short_return < -0.02)] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("high_vol_reversion"),
            name="High Volatility Mean Reversion",
            description="During high volatility regimes, short-term extremes "
                       "tend to mean revert as panic subsides.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=high_vol_reversion_signal,
            parameters={'vol_threshold': 1.5, 'reversion_lookback': 3, 'lookback_window': lookback},
            required_features=['close'],
            economic_rationale="High volatility often reflects panic or euphoria. "
                              "These emotional extremes tend to overshoot fair value "
                              "and subsequently correct.",
            source="regime_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_trend_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on trend regimes."""
        hypotheses = []
        
        # Hypothesis: Trend continuation in strong trends
        def trend_continuation_signal(df, trend_threshold=0.02, fast_ma=20, slow_ma=50):
            signals = pd.Series(0, index=df.index)
            
            # Calculate trend strength internally
            fast = df['close'].rolling(fast_ma).mean()
            slow = df['close'].rolling(slow_ma).mean()
            trend_strength = (fast - slow) / df['close']
            
            strong_uptrend = trend_strength > trend_threshold
            strong_downtrend = trend_strength < -trend_threshold
            
            # Pullback entry in uptrend
            pullback_up = df['close'].pct_change(3) < 0
            signals[(strong_uptrend) & (pullback_up)] = 1
            
            # Pullback entry in downtrend
            pullback_down = df['close'].pct_change(3) > 0
            signals[(strong_downtrend) & (pullback_down)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("trend_continuation"),
            name="Trend Pullback Continuation",
            description="In strong trending regimes, pullbacks provide "
                       "opportunities to enter in the direction of the trend.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=trend_continuation_signal,
            parameters={'trend_threshold': 0.02, 'fast_ma': 20, 'slow_ma': 50},
            required_features=['close'],
            economic_rationale="Trends persist due to gradual information incorporation "
                              "and behavioral biases. Temporary retracements offer "
                              "better risk/reward for trend-following.",
            source="regime_generator",
            priority=7
        ))
        
        # Hypothesis: Mean reversion in range-bound markets
        def range_reversion_signal(df, range_threshold=0.005, lookback=20, fast_ma=20, slow_ma=50):
            signals = pd.Series(0, index=df.index)
            
            # Calculate trend strength internally
            fast = df['close'].rolling(fast_ma).mean()
            slow = df['close'].rolling(slow_ma).mean()
            trend_strength = (fast - slow) / df['close']
            
            # Calculate ATR internally
            high_low = df['high'] - df['low']
            atr = high_low.rolling(14).mean()
            
            # Identify range-bound regime
            weak_trend = trend_strength.abs() < range_threshold
            
            # Price distance from rolling mean
            rolling_mean = df['close'].rolling(lookback).mean()
            distance = (df['close'] - rolling_mean) / atr.replace(0, np.nan)
            
            # Mean revert in ranges
            signals[(weak_trend) & (distance > 1.5)] = -1
            signals[(weak_trend) & (distance < -1.5)] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("range_reversion"),
            name="Range-Bound Mean Reversion",
            description="In sideways markets with no clear trend, price tends to "
                       "revert to the mean from extreme positions.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=range_reversion_signal,
            parameters={'range_threshold': 0.005, 'lookback': 20, 'fast_ma': 20, 'slow_ma': 50},
            required_features=['close', 'high', 'low'],
            economic_rationale="Range-bound markets reflect equilibrium between "
                              "buyers and sellers. Deviations from this equilibrium "
                              "tend to be temporary.",
            source="regime_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_volatility_mean_reversion(self, data: pd.DataFrame) -> Hypothesis:
        """Generate volatility mean reversion hypothesis."""
        
        def vol_mean_reversion_signal(df, vol_ma_fast=10, vol_ma_slow=50, threshold=1.3):
            signals = pd.Series(0, index=df.index)
            
            # Calculate returns if not present
            returns = df['returns'] if 'returns' in df.columns else df['close'].pct_change()
            
            vol = returns.rolling(10).std()
            vol_fast = vol.rolling(vol_ma_fast).mean()
            vol_slow = vol.rolling(vol_ma_slow).mean()
            vol_ratio = vol_fast / vol_slow.replace(0, np.nan)
            
            # When vol is high, follow momentum; when low, reduce exposure
            momentum = df['close'].pct_change(5)
            signals[(vol_ratio > threshold) & (momentum > 0)] = 1
            signals[(vol_ratio > threshold) & (momentum < 0)] = -1
            signals[(vol_ratio < 1/threshold)] = 0  # Low vol, stay flat
            
            return signals.fillna(0)
        
        return Hypothesis(
            id=self._create_hypothesis_id("vol_mean_reversion"),
            name="Volatility Mean Reversion",
            description="Volatility tends to mean revert. High volatility periods "
                       "are followed by contraction, and vice versa.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=vol_mean_reversion_signal,
            parameters={'vol_ma_fast': 10, 'vol_ma_slow': 50, 'threshold': 1.3},
            required_features=['close'],
            economic_rationale="Volatility clustering and mean reversion is one of "
                              "the most robust empirical findings in finance. "
                              "Extreme volatility is typically temporary.",
            source="regime_generator",
            priority=8
        )
    
    def _generate_transition_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses about regime transitions."""
        hypotheses = []
        
        def regime_transition_signal(df, mean_thresh=0.5, lookback=50):
            signals = pd.Series(0, index=df.index)
            
            # Calculate returns internally
            returns = df['returns'] if 'returns' in df.columns else df['close'].pct_change()
            
            # Calculate rolling regime indicators
            rolling_mean = returns.rolling(lookback).mean()
            rolling_std = returns.rolling(lookback).std()
            
            # Detect regime shifts using change point analysis
            mean_change = rolling_mean.diff(10).abs() / rolling_std.replace(0, np.nan)
            
            # Detect significant regime change
            mean_shift = mean_change > mean_thresh
            
            # New bullish regime (mean shift up)
            new_bull = (rolling_mean > 0) & (rolling_mean.diff(5) > 0) & mean_shift
            signals[new_bull] = 1
            
            # New bearish regime
            new_bear = (rolling_mean < 0) & (rolling_mean.diff(5) < 0) & mean_shift
            signals[new_bear] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("regime_transition"),
            name="Regime Transition Momentum",
            description="When markets transition between regimes, early detection "
                       "can capture the initial move before it becomes consensus.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=regime_transition_signal,
            parameters={'mean_thresh': 0.5, 'lookback': 50},
            required_features=['close'],
            economic_rationale="Regime changes often occur due to fundamental shifts "
                              "that take time to be fully recognized. Early detection "
                              "provides an information advantage.",
            source="regime_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_hmm_regimes(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses using Gaussian Mixture for regime detection."""
        hypotheses = []
        n_regimes = self.n_regimes

        # Original GMM strategy
        def gmm_regime_signal(df, n_components=3):
            signals = pd.Series(0, index=df.index)

            # Calculate features
            returns = df['close'].pct_change()
            vol = returns.rolling(20).std()

            features = pd.DataFrame({
                'returns': returns,
                'volatility': vol
            }).dropna()

            if len(features) < 200:
                return signals

            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42
                )
                regimes = gmm.fit_predict(features[['returns', 'volatility']])

                # Find bullish/bearish regimes
                features['regime'] = regimes
                regime_means = features.groupby('regime')['returns'].mean()
                bull_regime = regime_means.idxmax()
                bear_regime = regime_means.idxmin()

                # Create signals
                for i, idx in enumerate(features.index):
                    if regimes[i] == bull_regime:
                        signals[idx] = 1
                    elif regimes[i] == bear_regime:
                        signals[idx] = -1

            except Exception:
                pass

            return signals.fillna(0)

        # Enhanced GMM with more features and confidence
        def enhanced_gmm_signal(df, n_components=3, confidence_threshold=0.6):
            signals = pd.Series(0, index=df.index)

            # Enhanced feature set
            returns = df['close'].pct_change()

            # Multiple volatility measures
            vol_10 = returns.rolling(10).std()
            vol_20 = returns.rolling(20).std()
            vol_50 = returns.rolling(50).std()

            # Momentum features
            momentum_5 = returns.rolling(5).mean()
            momentum_20 = returns.rolling(20).mean()

            # Trend strength
            if 'ema_9' in df.columns and 'ema_21' in df.columns:
                trend_strength = (df['ema_9'] - df['ema_21']) / df['close']
            else:
                # Calculate if not present
                ema9 = df['close'].ewm(span=9, adjust=False).mean()
                ema21 = df['close'].ewm(span=21, adjust=False).mean()
                trend_strength = (ema9 - ema21) / df['close']

            # Volume-based features if available
            if 'volume_ratio' in df.columns:
                vol_ratio = df['volume_ratio']
            else:
                vol_ma = df['volume'].rolling(20).mean()
                vol_ratio = df['volume'] / vol_ma.replace(0, np.nan)

            features = pd.DataFrame({
                'returns': returns,
                'vol_10': vol_10,
                'vol_20': vol_20,
                'momentum_5': momentum_5,
                'trend_strength': trend_strength,
                'vol_ratio': vol_ratio
            }).dropna()

            if len(features) < 200:
                return signals

            try:
                from sklearn.mixture import GaussianMixture
                from sklearn.preprocessing import StandardScaler

                # Scale features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42
                )

                regimes = gmm.fit_predict(scaled_features)
                regime_probs = gmm.predict_proba(scaled_features)

                # Find regimes by clustering performance
                features['regime'] = regimes
                regime_stats = features.groupby('regime').agg({
                    'returns': ['mean', 'std', 'count'],
                    'trend_strength': 'mean'
                })

                # Identify regimes based on performance
                regime_returns = regime_stats[('returns', 'mean')]
                bull_regime = regime_returns.idxmax()
                bear_regime = regime_returns.idxmin()

                # Neutral regime (if exists)
                neutral_regime = None
                if len(regime_returns) > 2:
                    neutral_regime = regime_returns.abs().idxmin()

                # Generate signals with confidence threshold
                for i, idx in enumerate(features.index):
                    max_prob = regime_probs[i].max()
                    predicted_regime = regimes[i]

                    # Only trade if confident
                    if max_prob >= confidence_threshold:
                        if predicted_regime == bull_regime:
                            signals[idx] = 1
                        elif predicted_regime == bear_regime:
                            signals[idx] = -1
                        elif predicted_regime == neutral_regime:
                            signals[idx] = 0  # Stay flat

            except Exception as e:
                print(f"Enhanced GMM failed: {e}")
                pass

            return signals.fillna(0)

        # Rolling GMM - retrains periodically
        def rolling_gmm_signal(df, n_components=3, retrain_period=1000):
            signals = pd.Series(0, index=df.index)

            returns = df['close'].pct_change()
            vol = returns.rolling(20).std()

            features = pd.DataFrame({
                'returns': returns,
                'volatility': vol
            }).dropna()

            if len(features) < 200:
                return signals

            try:
                from sklearn.mixture import GaussianMixture

                # Process in chunks
                chunk_size = retrain_period
                for start_idx in range(0, len(features), chunk_size):
                    end_idx = min(start_idx + chunk_size + 200, len(features))
                    chunk = features.iloc[start_idx:end_idx]

                    if len(chunk) < 200:
                        continue

                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type='full',
                        random_state=42
                    )
                    regimes = gmm.fit_predict(chunk[['returns', 'volatility']])

                    # Find regime performance for this chunk
                    chunk_with_regime = chunk.copy()
                    chunk_with_regime['regime'] = regimes
                    regime_means = chunk_with_regime.groupby('regime')['returns'].mean()

                    if len(regime_means) >= 2:
                        bull_regime = regime_means.idxmax()
                        bear_regime = regime_means.idxmin()

                        # Apply to chunk indices
                        for i, idx in enumerate(chunk.index):
                            if regimes[i] == bull_regime:
                                signals[idx] = 1
                            elif regimes[i] == bear_regime:
                                signals[idx] = -1

            except Exception:
                pass

            return signals.fillna(0)

        # Add all GMM variants
        hypotheses.extend([
            Hypothesis(
                id=self._create_hypothesis_id("gmm_regime"),
                name="GMM Regime Following",
                description="Use Gaussian Mixture Model to identify market regimes "
                           "and position accordingly.",
                hypothesis_type=HypothesisType.REGIME,
                mechanism=HypothesisMechanism.MOMENTUM,
                signal_generator=gmm_regime_signal,
                parameters={'n_components': n_regimes},
                required_features=['close'],
                economic_rationale="Market regimes reflect underlying structural conditions. "
                                  "Probabilistic regime detection can identify favorable "
                                  "trading conditions before they become obvious.",
                source="regime_generator",
                priority=6
            ),
            Hypothesis(
                id=self._create_hypothesis_id("enhanced_gmm"),
                name="Enhanced GMM with Confidence",
                description="Advanced GMM using multiple features with confidence thresholds "
                           "to improve accuracy.",
                hypothesis_type=HypothesisType.REGIME,
                mechanism=HypothesisMechanism.MOMENTUM,
                signal_generator=enhanced_gmm_signal,
                parameters={'n_components': n_regimes, 'confidence_threshold': 0.6},
                required_features=['close', 'volume'],
                economic_rationale="Enhanced feature set and confidence filtering reduce false "
                                  "signals while maintaining edge detection capability.",
                source="regime_generator",
                priority=8
            ),
            Hypothesis(
                id=self._create_hypothesis_id("rolling_gmm"),
                name="Rolling GMM Adaptation",
                description="GMM that retrains periodically to adapt to changing market conditions.",
                hypothesis_type=HypothesisType.REGIME,
                mechanism=HypothesisMechanism.MOMENTUM,
                signal_generator=rolling_gmm_signal,
                parameters={'n_components': n_regimes, 'retrain_period': 1000},
                required_features=['close'],
                economic_rationale="Markets change over time. Periodic retraining maintains "
                                  "edge relevance across different market conditions.",
                source="regime_generator",
                priority=7
            )
        ])

        return hypotheses

