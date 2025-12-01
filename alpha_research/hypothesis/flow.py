"""
Flow-based hypothesis generation.

Focuses on asymmetries in market flows:
- Funding rates
- Net inflows/outflows
- Open interest dynamics
- Options flow signals
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


class FlowHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses based on flow asymmetries.
    
    Key patterns searched:
    - Funding rate extremes and mean reversion
    - Open interest dynamics
    - Volume flow analysis
    - Accumulation/distribution patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.funding_threshold = config.get('funding_rate_threshold', 0.0001)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.FLOW
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate flow-based hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Funding rate hypotheses (if available)
        if 'funding_rate' in data.columns:
            funding_hypotheses = self._generate_funding_hypotheses(data)
            hypotheses.extend(funding_hypotheses)
        
        # 2. Open interest hypotheses (if available)
        if 'open_interest' in data.columns:
            oi_hypotheses = self._generate_oi_hypotheses(data)
            hypotheses.extend(oi_hypotheses)
        
        # 3. Volume flow hypotheses
        vf_hypotheses = self._generate_volume_flow_hypotheses(data)
        hypotheses.extend(vf_hypotheses)
        
        # 4. Money flow hypotheses
        mf_hypotheses = self._generate_money_flow_hypotheses(data)
        hypotheses.extend(mf_hypotheses)
        
        return hypotheses
    
    def _generate_funding_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on funding rates."""
        hypotheses = []
        
        # Calculate funding rate metrics
        data['funding_zscore'] = (
            (data['funding_rate'] - data['funding_rate'].rolling(100).mean()) / 
            data['funding_rate'].rolling(100).std()
        )
        data['funding_cumsum'] = data['funding_rate'].rolling(24).sum()  # Daily cumulative
        
        # Hypothesis 1: Extreme funding mean reversion
        def funding_reversion_signal(df, threshold=2.0):
            signals = pd.Series(0, index=df.index)
            
            # Extremely positive funding = overleveraged longs
            # Expect shorts to profit as longs get squeezed out
            signals[df['funding_zscore'] > threshold] = -1
            
            # Extremely negative funding = overleveraged shorts
            signals[df['funding_zscore'] < -threshold] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("funding_reversion"),
            name="Funding Rate Mean Reversion",
            description="Extreme funding rates indicate over-leveraged positioning "
                       "that tends to unwind, often violently.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=funding_reversion_signal,
            parameters={'threshold': 2.0},
            required_features=['funding_zscore'],
            economic_rationale="High positive funding means longs are paying a premium "
                              "to maintain positions. This creates negative carry and "
                              "encourages position reduction, pressuring price down.",
            source="flow_generator",
            priority=8
        ))
        
        # Hypothesis 2: Funding rate momentum
        def funding_momentum_signal(df, lookback=8, threshold=0.0005):
            signals = pd.Series(0, index=df.index)
            
            funding_change = df['funding_cumsum'].diff(lookback)
            
            # Rising funding = increasing bullish leverage = follow
            signals[funding_change > threshold] = 1
            signals[funding_change < -threshold] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("funding_momentum"),
            name="Funding Rate Momentum",
            description="Changes in funding rate momentum can predict "
                       "near-term price direction.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=funding_momentum_signal,
            parameters={'lookback': 8, 'threshold': 0.0005},
            required_features=['funding_cumsum'],
            economic_rationale="Rising funding rates indicate increasing demand "
                              "for long exposure. Before reaching extremes, this "
                              "momentum often persists.",
            source="flow_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_oi_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on open interest."""
        hypotheses = []
        
        # Calculate OI metrics
        data['oi_change'] = data['open_interest'].pct_change()
        data['oi_zscore'] = (
            (data['open_interest'] - data['open_interest'].rolling(50).mean()) / 
            data['open_interest'].rolling(50).std()
        )
        
        # Hypothesis 1: OI-Price divergence
        def oi_divergence_signal(df, lookback=10):
            signals = pd.Series(0, index=df.index)
            
            price_change = df['close'].pct_change(lookback)
            oi_change = df['open_interest'].pct_change(lookback)
            
            # Price up, OI down = weak rally (longs closing, not new buyers)
            weak_rally = (price_change > 0.02) & (oi_change < -0.05)
            
            # Price down, OI down = weak selloff (shorts closing)
            weak_selloff = (price_change < -0.02) & (oi_change < -0.05)
            
            signals[weak_rally] = -1
            signals[weak_selloff] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("oi_divergence"),
            name="Open Interest Divergence",
            description="When price moves aren't confirmed by OI changes, "
                       "the move is likely to reverse.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=oi_divergence_signal,
            parameters={'lookback': 10},
            required_features=['close', 'open_interest'],
            economic_rationale="OI confirms conviction. Price moves with declining OI "
                              "suggest position closing rather than new interest, "
                              "indicating exhaustion.",
            source="flow_generator",
            priority=7
        ))
        
        # Hypothesis 2: OI accumulation breakout
        def oi_accumulation_signal(df, oi_threshold=0.1, price_range_bars=20):
            signals = pd.Series(0, index=df.index)
            
            # OI increasing while price ranges = accumulation
            oi_increasing = df['open_interest'].pct_change(10) > oi_threshold
            
            price_range = (
                df['high'].rolling(price_range_bars).max() - 
                df['low'].rolling(price_range_bars).min()
            ) / df['close']
            tight_range = price_range < 0.05
            
            accumulation = oi_increasing & tight_range
            
            # Breakout direction
            breakout_up = df['close'] > df['high'].rolling(price_range_bars).max().shift(1)
            breakout_down = df['close'] < df['low'].rolling(price_range_bars).min().shift(1)
            
            signals[(accumulation.shift(1)) & (breakout_up)] = 1
            signals[(accumulation.shift(1)) & (breakout_down)] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("oi_accumulation"),
            name="OI Accumulation Breakout",
            description="Rising OI during range-bound price action suggests "
                       "accumulation; breakouts tend to have follow-through.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=oi_accumulation_signal,
            parameters={'oi_threshold': 0.1, 'price_range_bars': 20},
            required_features=['open_interest', 'high', 'low', 'close'],
            economic_rationale="OI growth during consolidation means positions "
                              "are being established. When price breaks, these "
                              "positions drive continuation.",
            source="flow_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_volume_flow_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on volume flow analysis."""
        hypotheses = []
        
        # Calculate On-Balance Volume
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        data['obv_ma'] = data['obv'].rolling(20).mean()
        data['obv_signal'] = data['obv'] - data['obv_ma']
        
        # Calculate Volume-Weighted Average Price deviation
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['close']
        
        # Hypothesis 1: OBV divergence
        def obv_divergence_signal(df, lookback=20):
            signals = pd.Series(0, index=df.index)
            
            # Price making new highs but OBV not
            price_high = df['close'] == df['close'].rolling(lookback).max()
            obv_not_high = df['obv'] < df['obv'].rolling(lookback).max() * 0.95
            
            bearish_div = price_high & obv_not_high
            
            # Price making new lows but OBV not
            price_low = df['close'] == df['close'].rolling(lookback).min()
            obv_not_low = df['obv'] > df['obv'].rolling(lookback).min() * 1.05
            
            bullish_div = price_low & obv_not_low
            
            signals[bearish_div] = -1
            signals[bullish_div] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("obv_divergence"),
            name="On-Balance Volume Divergence",
            description="When price extremes aren't confirmed by OBV, "
                       "the move lacks volume conviction and may reverse.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=obv_divergence_signal,
            parameters={'lookback': 20},
            required_features=['close', 'obv'],
            economic_rationale="OBV tracks cumulative buying/selling pressure. "
                              "Divergences indicate that the marginal trader is "
                              "not participating in the price move.",
            source="flow_generator",
            priority=6
        ))
        
        # Hypothesis 2: VWAP mean reversion
        def vwap_reversion_signal(df, threshold=0.02):
            signals = pd.Series(0, index=df.index)
            
            # Price far above VWAP = overbought
            signals[df['vwap_distance'] > threshold] = -1
            
            # Price far below VWAP = oversold
            signals[df['vwap_distance'] < -threshold] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("vwap_reversion"),
            name="VWAP Mean Reversion",
            description="Price tends to revert to VWAP as it represents "
                       "the fair average price weighted by activity.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=vwap_reversion_signal,
            parameters={'threshold': 0.02},
            required_features=['vwap_distance'],
            economic_rationale="VWAP is widely used by institutional traders as "
                              "a benchmark. Extreme deviations create mean reversion "
                              "pressure as traders seek better fills.",
            source="flow_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_money_flow_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on money flow indicators."""
        hypotheses = []
        
        # Calculate Money Flow Index (MFI)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        raw_money_flow = typical_price * data['volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        
        positive_mf = pd.Series(positive_flow, index=data.index).rolling(14).sum()
        negative_mf = pd.Series(negative_flow, index=data.index).rolling(14).sum()
        
        mf_ratio = positive_mf / negative_mf.replace(0, np.nan)
        data['mfi'] = 100 - (100 / (1 + mf_ratio))
        
        # Chaikin Money Flow
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
        data['cmf'] = (clv * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Hypothesis 1: MFI extremes
        def mfi_extreme_signal(df, overbought=80, oversold=20):
            signals = pd.Series(0, index=df.index)
            
            signals[df['mfi'] > overbought] = -1
            signals[df['mfi'] < oversold] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("mfi_extreme"),
            name="Money Flow Index Extremes",
            description="MFI combines price and volume to identify overbought "
                       "and oversold conditions with higher reliability.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=mfi_extreme_signal,
            parameters={'overbought': 80, 'oversold': 20},
            required_features=['mfi'],
            economic_rationale="MFI extremes indicate exhaustion of buying/selling "
                              "pressure. Volume-weighted overbought/oversold levels "
                              "are more reliable than price-only indicators.",
            source="flow_generator",
            priority=6
        ))
        
        # Hypothesis 2: CMF trend confirmation
        def cmf_trend_signal(df, threshold=0.1, trend_lookback=20):
            signals = pd.Series(0, index=df.index)
            
            price_trend = np.sign(df['close'] - df['close'].shift(trend_lookback))
            
            # Trend with CMF confirmation
            bullish = (price_trend == 1) & (df['cmf'] > threshold)
            bearish = (price_trend == -1) & (df['cmf'] < -threshold)
            
            signals[bullish] = 1
            signals[bearish] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("cmf_confirmation"),
            name="Chaikin Money Flow Trend Confirmation",
            description="Trends confirmed by positive/negative money flow "
                       "are more likely to continue.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=cmf_trend_signal,
            parameters={'threshold': 0.1, 'trend_lookback': 20},
            required_features=['close', 'cmf'],
            economic_rationale="CMF measures accumulation/distribution within the "
                              "trading range. Trends backed by appropriate money "
                              "flow have institutional support.",
            source="flow_generator",
            priority=6
        ))
        
        return hypotheses

