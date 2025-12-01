"""
Microstructure-based hypothesis generation.

Focuses on orderbook dynamics, liquidity patterns, and 
large player footprints that may indicate exploitable edges.
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats

# Suppress pandas future warnings about fillna downcasting
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass  # Option not available in this pandas version

from .base import (
    HypothesisGenerator, 
    Hypothesis, 
    HypothesisType, 
    HypothesisMechanism
)


class MicrostructureHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses based on market microstructure patterns.
    
    Key patterns searched:
    - Orderbook imbalance signals
    - Liquidity shifts and drying
    - Large player footprints via volume analysis
    - Bid-ask spread dynamics
    - Trade flow toxicity
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.imbalance_window = config.get('orderbook_imbalance_window', 20)
        self.liquidity_threshold = config.get('liquidity_shift_threshold', 0.3)
        self.large_player_zscore = config.get('large_player_volume_zscore', 2.5)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.MICROSTRUCTURE
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate microstructure-based hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Volume-based hypotheses
        vol_hypotheses = self._generate_volume_hypotheses(data)
        hypotheses.extend(vol_hypotheses)
        
        # 2. Price action hypotheses
        price_hypotheses = self._generate_price_action_hypotheses(data)
        hypotheses.extend(price_hypotheses)
        
        # 3. Liquidity-based hypotheses
        liquidity_hypotheses = self._generate_liquidity_hypotheses(data)
        hypotheses.extend(liquidity_hypotheses)
        
        return hypotheses
    
    def _generate_volume_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on volume patterns."""
        hypotheses = []
        
        # Hypothesis 1: Large player accumulation
        def accumulation_signal(df, vol_threshold=2.0, price_change_thresh=0.001):
            signals = pd.Series(0, index=df.index)
            
            # Calculate volume metrics internally
            volume_ma = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            volume_zscore = (df['volume'] - volume_ma) / volume_std.replace(0, np.nan)
            
            # Calculate volume delta
            up_volume = np.where(df['close'] > df['open'], df['volume'], 0)
            down_volume = np.where(df['close'] < df['open'], df['volume'], 0)
            up_vol_sum = pd.Series(up_volume, index=df.index).rolling(10).sum()
            down_vol_sum = pd.Series(down_volume, index=df.index).rolling(10).sum()
            total_vol = df['volume'].rolling(10).sum()
            volume_delta = (up_vol_sum - down_vol_sum) / total_vol.replace(0, np.nan)
            
            # Calculate returns
            returns = df['close'].pct_change()
            
            # High volume with small price change = accumulation/distribution
            high_vol = volume_zscore > vol_threshold
            small_move = returns.abs() < price_change_thresh
            
            # Direction determined by volume delta
            bullish_accum = high_vol & small_move & (volume_delta > 0.3)
            bearish_dist = high_vol & small_move & (volume_delta < -0.3)
            
            signals[bullish_accum] = 1
            signals[bearish_dist] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("accumulation"),
            name="Large Player Accumulation Detection",
            description="Detect institutional accumulation/distribution through "
                       "high volume with small price movement.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.INFORMATION_ASYMMETRY,
            signal_generator=accumulation_signal,
            parameters={'vol_threshold': 2.0, 'price_change_thresh': 0.001},
            required_features=['open', 'close', 'volume'],
            economic_rationale="Large players must accumulate/distribute slowly to "
                              "avoid moving the market. This creates detectable "
                              "footprints in volume-price relationships.",
            source="microstructure_generator",
            priority=8
        ))
        
        # Hypothesis 2: Volume climax reversal
        def volume_climax_signal(df, vol_threshold=3.0, lookback=5):
            signals = pd.Series(0, index=df.index)
            
            # Calculate volume z-score
            volume_ma = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            volume_zscore = (df['volume'] - volume_ma) / volume_std.replace(0, np.nan)
            
            # Extreme volume spike
            climax = volume_zscore > vol_threshold
            
            # Price direction into climax
            price_change = df['close'].pct_change(lookback)
            
            # Reversal signal after climax
            bullish_climax = climax & (price_change < -0.02)  # Sharp down into climax
            bearish_climax = climax & (price_change > 0.02)   # Sharp up into climax
            
            # Fade the panic/euphoria
            bull_mask = bullish_climax.shift(1).fillna(False)
            bear_mask = bearish_climax.shift(1).fillna(False)
            signals.loc[bull_mask.values] = 1
            signals.loc[bear_mask.values] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("volume_climax"),
            name="Volume Climax Reversal",
            description="Extreme volume spikes often mark exhaustion points "
                       "where the current move has run out of steam.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.BEHAVIORAL,
            signal_generator=volume_climax_signal,
            parameters={'vol_threshold': 3.0, 'lookback': 5},
            required_features=['close', 'volume'],
            economic_rationale="Volume climaxes represent capitulation or euphoric peaks. "
                              "The last marginal buyers/sellers have acted, leaving "
                              "the market positioned for reversal.",
            source="microstructure_generator",
            priority=7
        ))
        
        # Hypothesis 3: Volume dry-up breakout
        def volume_dryup_breakout(df, vol_threshold=-1.0, consolidation_bars=10):
            signals = pd.Series(0, index=df.index)
            
            # Calculate volume z-score
            volume_ma = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            volume_zscore = (df['volume'] - volume_ma) / volume_std.replace(0, np.nan)
            
            # Low volume period
            low_vol = volume_zscore.rolling(consolidation_bars).mean() < vol_threshold
            
            # Calculate consolidation range
            range_high = df['high'].rolling(consolidation_bars).max()
            range_low = df['low'].rolling(consolidation_bars).min()
            
            # Breakout detection
            low_vol_prev = low_vol.shift(1).fillna(False)
            breakout_up = (df['close'] > range_high.shift(1)) & low_vol_prev.values
            breakout_down = (df['close'] < range_low.shift(1)) & low_vol_prev.values
            
            signals[breakout_up] = 1
            signals[breakout_down] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("volume_dryup"),
            name="Volume Dry-Up Breakout",
            description="Consolidations with declining volume often precede "
                       "significant breakouts with directional follow-through.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=volume_dryup_breakout,
            parameters={'vol_threshold': -1.0, 'consolidation_bars': 10},
            required_features=['high', 'low', 'close', 'volume'],
            economic_rationale="Volume dry-up during consolidation indicates "
                              "equilibrium. When this balance breaks, the move "
                              "tends to be sustained as new participants enter.",
            source="microstructure_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_price_action_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on price action patterns."""
        hypotheses = []
        
        # Hypothesis: Rejection wicks
        def rejection_wick_signal(df, wick_threshold=0.6, body_max=0.3):
            signals = pd.Series(0, index=df.index)
            
            # Calculate candle metrics
            body = (df['close'] - df['open']).abs()
            range_hl = df['high'] - df['low']
            
            body_ratio = body / range_hl.replace(0, np.nan)
            upper_wick = (df['high'] - df[['open', 'close']].max(axis=1)) / range_hl.replace(0, np.nan)
            lower_wick = (df[['open', 'close']].min(axis=1) - df['low']) / range_hl.replace(0, np.nan)
            
            # Strong upper rejection (bearish)
            upper_rejection = (upper_wick > wick_threshold) & (body_ratio < body_max)
            
            # Strong lower rejection (bullish)
            lower_rejection = (lower_wick > wick_threshold) & (body_ratio < body_max)
            
            signals[upper_rejection] = -1
            signals[lower_rejection] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("rejection_wick"),
            name="Rejection Wick Reversal",
            description="Long wicks with small bodies indicate price rejection "
                       "at extreme levels, signaling potential reversal.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=rejection_wick_signal,
            parameters={'wick_threshold': 0.6, 'body_max': 0.3},
            required_features=['open', 'high', 'low', 'close'],
            economic_rationale="Rejection wicks show that price was pushed to an "
                              "extreme but couldn't hold. Aggressive orders were "
                              "absorbed and the market rejected that level.",
            source="microstructure_generator",
            priority=6
        ))
        
        # Hypothesis: Engulfing patterns with volume confirmation
        def engulfing_volume_signal(df, vol_threshold=1.2):
            signals = pd.Series(0, index=df.index)
            
            # Relative volume
            volume_ma = df['volume'].rolling(20).mean()
            rel_volume = df['volume'] / volume_ma.replace(0, np.nan)
            
            # Bullish engulfing
            bullish_engulf = (
                (df['close'] > df['open']) &  # Current is green
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous was red
                (df['close'] > df['open'].shift(1)) &  # Body covers previous
                (df['open'] < df['close'].shift(1)) &
                (rel_volume > vol_threshold)  # High volume confirmation
            )
            
            # Bearish engulfing
            bearish_engulf = (
                (df['close'] < df['open']) &  # Current is red
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous was green
                (df['close'] < df['open'].shift(1)) &  # Body covers previous
                (df['open'] > df['close'].shift(1)) &
                (rel_volume > vol_threshold)
            )
            
            signals[bullish_engulf] = 1
            signals[bearish_engulf] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("engulfing_volume"),
            name="Volume-Confirmed Engulfing",
            description="Engulfing patterns with above-average volume have "
                       "higher reliability as reversal signals.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.BEHAVIORAL,
            signal_generator=engulfing_volume_signal,
            parameters={'vol_threshold': 1.2},
            required_features=['open', 'close', 'volume'],
            economic_rationale="Engulfing patterns represent a shift in control. "
                              "When confirmed by volume, it indicates genuine "
                              "participation in the reversal, not just noise.",
            source="microstructure_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_liquidity_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on liquidity patterns."""
        hypotheses = []
        
        # Hypothesis: Liquidity vacuum
        def liquidity_vacuum_signal(df, liq_threshold=-1.5, momentum_bars=3):
            signals = pd.Series(0, index=df.index)
            
            # Approximate liquidity using volume and volatility
            range_hl = df['high'] - df['low']
            liquidity_proxy = df['volume'] / range_hl.replace(0, np.nan)
            liq_mean = liquidity_proxy.rolling(50).mean()
            liq_std = liquidity_proxy.rolling(50).std()
            liquidity_zscore = (liquidity_proxy - liq_mean) / liq_std.replace(0, np.nan)
            
            # Low liquidity environment
            low_liq = liquidity_zscore < liq_threshold
            
            # Strong directional move in low liquidity
            momentum = df['close'].pct_change(momentum_bars)
            
            # These moves often overshoot and revert
            overshoot_up = low_liq & (momentum > 0.01)
            overshoot_down = low_liq & (momentum < -0.01)
            
            # Fade the move
            signals[overshoot_up] = -1
            signals[overshoot_down] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("liquidity_vacuum"),
            name="Liquidity Vacuum Fade",
            description="Moves in low-liquidity environments tend to overshoot "
                       "and subsequently revert as liquidity returns.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.LIQUIDITY_PREMIUM,
            signal_generator=liquidity_vacuum_signal,
            parameters={'liq_threshold': -1.5, 'momentum_bars': 3},
            required_features=['high', 'low', 'close', 'volume'],
            economic_rationale="Low liquidity allows prices to move further on "
                              "small order flow. These moves often reverse as "
                              "normal liquidity returns and fair value is restored.",
            source="microstructure_generator",
            priority=6
        ))
        
        return hypotheses
