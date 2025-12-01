"""
Cross-asset hypothesis generation.

Focuses on relationships between different assets:
- Spot vs perpetual basis
- Futures term structure
- Correlated assets and lead-lag relationships
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats

from .base import (
    HypothesisGenerator, 
    Hypothesis, 
    HypothesisType, 
    HypothesisMechanism
)


class CrossAssetHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses based on cross-asset relationships.
    
    Key patterns searched:
    - Spot/perp basis trades
    - Futures curve analysis
    - Correlation breakdowns
    - Lead-lag relationships
    - Relative value opportunities
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.basis_threshold = config.get('basis_significance_threshold', 0.5)
        self.term_structure_lookback = config.get('term_structure_lookback', 30)
        self.correlation_window = config.get('correlation_window', 50)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.CROSS_ASSET
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate cross-asset based hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. If we have perp data, generate basis hypotheses
        if 'perp_close' in data.columns or 'perpetual_price' in data.columns:
            basis_hypotheses = self._generate_basis_hypotheses(data)
            hypotheses.extend(basis_hypotheses)
        
        # 2. If we have futures data, generate term structure hypotheses
        if 'futures_near' in data.columns and 'futures_far' in data.columns:
            term_hypotheses = self._generate_term_structure_hypotheses(data)
            hypotheses.extend(term_hypotheses)
        
        # 3. Generic correlation-based hypotheses using OHLCV
        corr_hypotheses = self._generate_correlation_hypotheses(data)
        hypotheses.extend(corr_hypotheses)
        
        # 4. Self-referential cross-timeframe hypotheses
        tf_hypotheses = self._generate_timeframe_hypotheses(data)
        hypotheses.extend(tf_hypotheses)
        
        return hypotheses
    
    def _generate_basis_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on spot-perp basis."""
        hypotheses = []
        
        perp_col = 'perp_close' if 'perp_close' in data.columns else 'perpetual_price'
        
        # Calculate basis
        data['basis'] = (data[perp_col] - data['close']) / data['close']
        data['basis_zscore'] = (
            (data['basis'] - data['basis'].rolling(100).mean()) / 
            data['basis'].rolling(100).std()
        )
        
        # Hypothesis 1: Basis mean reversion
        def basis_reversion_signal(df, threshold=2.0):
            signals = pd.Series(0, index=df.index)
            
            # Extreme contango (perp > spot) - expect basis to contract
            # This means perp should underperform spot
            signals[df['basis_zscore'] > threshold] = -1  # Short perp or short the asset
            
            # Extreme backwardation (perp < spot)
            signals[df['basis_zscore'] < -threshold] = 1  # Long the asset
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("basis_reversion"),
            name="Spot-Perp Basis Mean Reversion",
            description="Extreme basis between spot and perpetual tends to "
                       "mean revert as arbitrageurs step in.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.ARBITRAGE,
            signal_generator=basis_reversion_signal,
            parameters={'threshold': 2.0},
            required_features=['basis_zscore'],
            economic_rationale="The basis represents the cost of carry and sentiment. "
                              "Extreme deviations create arbitrage opportunities that "
                              "market makers will exploit, normalizing the basis.",
            source="cross_asset_generator",
            priority=8
        ))
        
        # Hypothesis 2: Basis momentum (trend following on basis)
        def basis_momentum_signal(df, lookback=24, threshold=0.001):
            signals = pd.Series(0, index=df.index)
            
            basis_change = df['basis'].diff(lookback)
            
            # If basis is expanding (more contango), sentiment is getting more bullish
            # This often predicts spot price increase
            signals[basis_change > threshold] = 1
            signals[basis_change < -threshold] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("basis_momentum"),
            name="Basis Momentum",
            description="Changes in the spot-perp basis can predict directional "
                       "price movement as it reflects changing market sentiment.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.INFORMATION_ASYMMETRY,
            signal_generator=basis_momentum_signal,
            parameters={'lookback': 24, 'threshold': 0.001},
            required_features=['basis'],
            economic_rationale="The perpetual market often leads spot as it's where "
                              "leveraged speculators express views. Basis changes "
                              "can signal impending spot moves.",
            source="cross_asset_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_term_structure_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on futures term structure."""
        hypotheses = []
        
        # Calculate term structure slope
        data['curve_slope'] = (data['futures_far'] - data['futures_near']) / data['close']
        data['curve_slope_zscore'] = (
            (data['curve_slope'] - data['curve_slope'].rolling(50).mean()) / 
            data['curve_slope'].rolling(50).std()
        )
        
        # Hypothesis: Term structure steepening/flattening
        def curve_signal(df, steep_threshold=1.5, flat_threshold=-1.0):
            signals = pd.Series(0, index=df.index)
            
            # Extreme contango (steep curve) often precedes corrections
            signals[df['curve_slope_zscore'] > steep_threshold] = -1
            
            # Extreme backwardation often precedes rallies
            signals[df['curve_slope_zscore'] < flat_threshold] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("term_structure"),
            name="Term Structure Extremes",
            description="Extreme contango or backwardation in the futures curve "
                       "often precedes mean-reverting price moves.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.RISK_PREMIUM,
            signal_generator=curve_signal,
            parameters={'steep_threshold': 1.5, 'flat_threshold': -1.0},
            required_features=['curve_slope_zscore'],
            economic_rationale="The term structure reflects storage costs, convenience "
                              "yield, and market expectations. Extremes indicate "
                              "unsustainable conditions.",
            source="cross_asset_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_correlation_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on correlation patterns."""
        hypotheses = []
        
        # Use high-low correlation as a proxy for market structure
        data['hl_correlation'] = data['high'].rolling(20).corr(data['low'])
        
        # Calculate return autocorrelation
        data['return_autocorr'] = data['returns'].rolling(50).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 10 else 0
        )
        
        # Hypothesis: Autocorrelation regime
        def autocorr_signal(df, pos_threshold=0.1, neg_threshold=-0.1):
            signals = pd.Series(0, index=df.index)
            
            # Positive autocorrelation = momentum works
            # Negative autocorrelation = mean reversion works
            
            pos_autocorr = df['return_autocorr'] > pos_threshold
            neg_autocorr = df['return_autocorr'] < neg_threshold
            
            recent_return = df['returns'].rolling(3).mean()
            
            # In momentum regime, follow recent returns
            signals[(pos_autocorr) & (recent_return > 0)] = 1
            signals[(pos_autocorr) & (recent_return < 0)] = -1
            
            # In mean reversion regime, fade recent returns
            signals[(neg_autocorr) & (recent_return > 0.01)] = -1
            signals[(neg_autocorr) & (recent_return < -0.01)] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("autocorr_regime"),
            name="Autocorrelation Regime Adaptation",
            description="Adapt strategy based on current autocorrelation regime - "
                       "momentum when positive, mean reversion when negative.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=autocorr_signal,
            parameters={'pos_threshold': 0.1, 'neg_threshold': -0.1},
            required_features=['return_autocorr', 'returns'],
            economic_rationale="Markets oscillate between trending and mean-reverting "
                              "regimes. Detecting the current regime allows adapting "
                              "the appropriate strategy.",
            source="cross_asset_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_timeframe_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on multi-timeframe analysis."""
        hypotheses = []
        
        # Calculate multi-timeframe metrics
        data['ma_fast'] = data['close'].rolling(10).mean()
        data['ma_medium'] = data['close'].rolling(50).mean()
        data['ma_slow'] = data['close'].rolling(200).mean()
        
        # Timeframe alignment
        data['tf_alignment'] = (
            np.sign(data['close'] - data['ma_fast']) +
            np.sign(data['close'] - data['ma_medium']) +
            np.sign(data['close'] - data['ma_slow'])
        )
        
        # Hypothesis: Multi-timeframe alignment
        def tf_alignment_signal(df, strong_threshold=2):
            signals = pd.Series(0, index=df.index)
            
            # Strong bullish alignment
            signals[df['tf_alignment'] >= strong_threshold] = 1
            
            # Strong bearish alignment
            signals[df['tf_alignment'] <= -strong_threshold] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("tf_alignment"),
            name="Multi-Timeframe Alignment",
            description="When price is aligned across multiple timeframes "
                       "(above/below key MAs), trends tend to persist.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=tf_alignment_signal,
            parameters={'strong_threshold': 2},
            required_features=['tf_alignment'],
            economic_rationale="Multi-timeframe alignment represents consensus "
                              "across different participant time horizons. This "
                              "creates self-reinforcing price action.",
            source="cross_asset_generator",
            priority=6
        ))
        
        # Hypothesis: Timeframe divergence reversal
        def tf_divergence_signal(df, lookback=20):
            signals = pd.Series(0, index=df.index)
            
            # Price making new highs but medium-term MA not confirming
            price_high = df['close'] == df['close'].rolling(lookback).max()
            ma_not_high = df['ma_medium'] < df['ma_medium'].rolling(lookback).max() * 0.99
            
            bearish_divergence = price_high & ma_not_high
            
            # Price making new lows but medium-term MA not confirming
            price_low = df['close'] == df['close'].rolling(lookback).min()
            ma_not_low = df['ma_medium'] > df['ma_medium'].rolling(lookback).min() * 1.01
            
            bullish_divergence = price_low & ma_not_low
            
            signals[bearish_divergence] = -1
            signals[bullish_divergence] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("tf_divergence"),
            name="Timeframe Divergence Reversal",
            description="When price makes new extremes but underlying momentum "
                       "doesn't confirm, a reversal is more likely.",
            hypothesis_type=HypothesisType.CROSS_ASSET,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=tf_divergence_signal,
            parameters={'lookback': 20},
            required_features=['close', 'ma_medium'],
            economic_rationale="Price-momentum divergences indicate exhaustion. "
                              "The final push to new extremes lacks conviction "
                              "and often marks reversal points.",
            source="cross_asset_generator",
            priority=6
        ))
        
        return hypotheses

