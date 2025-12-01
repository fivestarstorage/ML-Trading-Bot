"""
Crypto-specific hypothesis generation.

Specialized hypotheses for cryptocurrency markets including:
- Funding rate arbitrage signals
- Open interest dynamics
- Liquidation cascade patterns
- Session-based patterns (Asia/Europe/US)
- Weekend effects
- Whale activity detection
- Correlation regime patterns
- Volatility premium patterns
"""

import warnings
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

# Suppress pandas warnings
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass  # Option not available in this pandas version


class CryptoHypothesisGenerator(HypothesisGenerator):
    """
    Generates crypto-specific hypotheses.
    
    Works with:
    - Basic OHLCV data (always available)
    - Optional: funding_rate, open_interest, liquidations, etc.
    
    Key patterns:
    - 24/7 market dynamics
    - High volatility regimes
    - Leverage and derivatives signals
    - On-chain metrics (if available)
    - Session-based patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.funding_threshold = config.get('funding_threshold', 0.01)  # 1% funding
        self.oi_lookback = config.get('oi_lookback', 24)  # 24 bars
        self.whale_threshold = config.get('whale_volume_zscore', 3.0)
        self.weekend_lookback = config.get('weekend_lookback', 48)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.MICROSTRUCTURE
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate crypto-specific hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Session-based patterns (always available)
        session_hypotheses = self._generate_session_hypotheses(data)
        hypotheses.extend(session_hypotheses)
        
        # 2. Weekend effect patterns
        weekend_hypotheses = self._generate_weekend_hypotheses(data)
        hypotheses.extend(weekend_hypotheses)
        
        # 3. Volatility regime patterns
        vol_hypotheses = self._generate_volatility_hypotheses(data)
        hypotheses.extend(vol_hypotheses)
        
        # 4. Whale detection patterns
        whale_hypotheses = self._generate_whale_hypotheses(data)
        hypotheses.extend(whale_hypotheses)
        
        # 5. Momentum exhaustion patterns
        exhaustion_hypotheses = self._generate_exhaustion_hypotheses(data)
        hypotheses.extend(exhaustion_hypotheses)
        
        # 6. Funding rate patterns (if available)
        if 'funding_rate' in data.columns:
            funding_hypotheses = self._generate_funding_hypotheses(data)
            hypotheses.extend(funding_hypotheses)
        
        # 7. Open interest patterns (if available)
        if 'open_interest' in data.columns:
            oi_hypotheses = self._generate_oi_hypotheses(data)
            hypotheses.extend(oi_hypotheses)
        
        # 8. Liquidation patterns (if available)
        if 'liquidations' in data.columns or 'long_liquidations' in data.columns:
            liq_hypotheses = self._generate_liquidation_hypotheses(data)
            hypotheses.extend(liq_hypotheses)
        
        # 9. Range breakout patterns
        breakout_hypotheses = self._generate_breakout_hypotheses(data)
        hypotheses.extend(breakout_hypotheses)
        
        # 10. VWAP deviation patterns
        vwap_hypotheses = self._generate_vwap_hypotheses(data)
        hypotheses.extend(vwap_hypotheses)
        
        return hypotheses
    
    def _generate_session_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate session-based trading hypotheses."""
        hypotheses = []
        
        # Asia session momentum
        def asia_session_signal(df, start_hour=0, end_hour=8):
            """Trade in direction of Asian session momentum."""
            signals = pd.Series(0, index=df.index)
            
            hour = df.index.hour
            
            # Asia session hours (UTC)
            asia_mask = (hour >= start_hour) & (hour < end_hour)
            
            # Calculate session returns
            session_returns = df['close'].pct_change(4)  # 4-bar momentum
            
            # Go long during Asia if momentum positive
            signals[(asia_mask) & (session_returns > 0.001)] = 1
            signals[(asia_mask) & (session_returns < -0.001)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("asia_session"),
            name="Asia Session Momentum",
            description="Asian trading hours often set the direction for the day. "
                       "Follow momentum during Asia session (00:00-08:00 UTC).",
            hypothesis_type=HypothesisType.SEASONAL,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=asia_session_signal,
            parameters={'start_hour': 0, 'end_hour': 8},
            required_features=['close'],
            economic_rationale="Asian markets represent significant crypto trading volume. "
                              "Early session direction often persists due to follow-through "
                              "from other regions.",
            source="crypto_generator",
            priority=6
        ))
        
        # US session reversal
        def us_session_reversal(df, start_hour=14, end_hour=20):
            """Fade extremes during US session."""
            signals = pd.Series(0, index=df.index)
            
            hour = df.index.hour
            us_mask = (hour >= start_hour) & (hour < end_hour)
            
            # Daily range position
            rolling_high = df['high'].rolling(48).max()
            rolling_low = df['low'].rolling(48).min()
            range_pos = (df['close'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
            
            # Fade extremes during US session
            signals[(us_mask) & (range_pos > 0.9)] = -1
            signals[(us_mask) & (range_pos < 0.1)] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("us_session_fade"),
            name="US Session Extreme Fade",
            description="US session often sees profit-taking and mean reversion "
                       "from moves established earlier in the day.",
            hypothesis_type=HypothesisType.SEASONAL,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=us_session_reversal,
            parameters={'start_hour': 14, 'end_hour': 20},
            required_features=['high', 'low', 'close'],
            economic_rationale="US traders often take profits on moves established "
                              "during Asian/European sessions, causing mean reversion "
                              "at daily extremes.",
            source="crypto_generator",
            priority=5
        ))
        
        return hypotheses
    
    def _generate_weekend_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate weekend effect hypotheses."""
        hypotheses = []
        
        def weekend_low_vol_fade(df, vol_ratio_thresh=0.6):
            """Fade weekend moves due to low liquidity."""
            signals = pd.Series(0, index=df.index)
            
            # Detect weekend
            dayofweek = df.index.dayofweek
            weekend_mask = dayofweek >= 5  # Saturday=5, Sunday=6
            
            # Calculate relative volume
            vol_ma = df['volume'].rolling(100).mean()
            rel_vol = df['volume'] / vol_ma.replace(0, np.nan)
            
            # Low volume weekend
            low_vol_weekend = weekend_mask & (rel_vol < vol_ratio_thresh)
            
            # Weekend momentum
            weekend_momentum = df['close'].pct_change(6)  # 30min * 6 = 3 hours
            
            # Fade weekend moves (they often reverse Monday)
            signals[(low_vol_weekend) & (weekend_momentum > 0.01)] = -1
            signals[(low_vol_weekend) & (weekend_momentum < -0.01)] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("weekend_fade"),
            name="Weekend Low Liquidity Fade",
            description="Weekend moves in low liquidity often reverse as "
                       "institutional traders return Monday.",
            hypothesis_type=HypothesisType.SEASONAL,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=weekend_low_vol_fade,
            parameters={'vol_ratio_thresh': 0.6},
            required_features=['close', 'volume'],
            economic_rationale="Weekend liquidity is typically 30-50% of weekday levels. "
                              "Moves on low liquidity are often driven by retail and "
                              "tend to reverse when institutional flow returns.",
            source="crypto_generator",
            priority=5
        ))
        
        # Sunday night accumulation
        def sunday_accumulation(df):
            """Detect Sunday night smart money accumulation."""
            signals = pd.Series(0, index=df.index)
            
            dayofweek = df.index.dayofweek
            hour = df.index.hour
            
            # Sunday evening (18:00-23:59 UTC)
            sunday_evening = (dayofweek == 6) & (hour >= 18)
            
            # Volume spike on Sunday evening
            vol_ma = df['volume'].rolling(20).mean()
            vol_std = df['volume'].rolling(20).std()
            vol_zscore = (df['volume'] - vol_ma) / vol_std.replace(0, np.nan)
            
            # High volume Sunday evening = accumulation
            high_vol_sunday = sunday_evening & (vol_zscore > 1.5)
            
            # Direction based on close vs open
            bullish = df['close'] > df['open']
            
            signals[(high_vol_sunday) & (bullish)] = 1
            signals[(high_vol_sunday) & (~bullish)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("sunday_accumulation"),
            name="Sunday Night Accumulation",
            description="Smart money often accumulates on Sunday evenings before "
                       "the weekly momentum begins Monday.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.INFORMATION_ASYMMETRY,
            signal_generator=sunday_accumulation,
            parameters={},
            required_features=['open', 'close', 'volume'],
            economic_rationale="Institutional traders often position before the "
                              "Monday open when retail is inactive, creating "
                              "detectable volume patterns.",
            source="crypto_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_volatility_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate crypto volatility-based hypotheses."""
        hypotheses = []
        
        # Volatility expansion trade
        def vol_expansion_momentum(df, vol_expansion_thresh=1.8, lookback=20):
            """Trade momentum during volatility expansion."""
            signals = pd.Series(0, index=df.index)
            
            # Calculate realized volatility
            returns = df['close'].pct_change()
            vol = returns.rolling(lookback).std()
            vol_ma = vol.rolling(lookback * 2).mean()
            vol_ratio = vol / vol_ma.replace(0, np.nan)
            
            # Volatility expansion
            expanding = vol_ratio > vol_expansion_thresh
            
            # Trade momentum during expansion
            momentum = returns.rolling(5).mean()
            
            signals[(expanding) & (momentum > 0)] = 1
            signals[(expanding) & (momentum < 0)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("vol_expansion"),
            name="Volatility Expansion Momentum",
            description="During volatility expansion, follow the trend as "
                       "breakouts tend to have follow-through.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=vol_expansion_momentum,
            parameters={'vol_expansion_thresh': 1.8, 'lookback': 20},
            required_features=['close'],
            economic_rationale="Volatility expansion often marks the start of "
                              "trending moves. The initial direction tends to "
                              "persist as new participants enter.",
            source="crypto_generator",
            priority=7
        ))
        
        # Volatility compression breakout
        def vol_compression_breakout(df, compression_thresh=0.5, breakout_mult=2.0):
            """Trade breakouts from volatility compression."""
            signals = pd.Series(0, index=df.index)
            
            # ATR-based volatility
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(14).mean()
            atr_ma = atr.rolling(50).mean()
            atr_ratio = atr / atr_ma.replace(0, np.nan)
            
            # Compression phase
            compressed = atr_ratio < compression_thresh
            
            # Breakout move (large candle after compression)
            candle_range = df['high'] - df['low']
            large_candle = candle_range > atr * breakout_mult
            
            # Direction of breakout
            bullish_break = large_candle & (df['close'] > df['open'])
            bearish_break = large_candle & (df['close'] < df['open'])
            
            # Signal on breakout from compression
            signals[(compressed.shift(1).fillna(False)) & bullish_break] = 1
            signals[(compressed.shift(1).fillna(False)) & bearish_break] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("compression_breakout"),
            name="Compression Breakout",
            description="Volatility compression followed by range expansion "
                       "often leads to sustained directional moves.",
            hypothesis_type=HypothesisType.REGIME,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=vol_compression_breakout,
            parameters={'compression_thresh': 0.5, 'breakout_mult': 2.0},
            required_features=['open', 'high', 'low', 'close'],
            economic_rationale="Compression represents equilibrium. When broken, "
                              "it triggers stops and forced positioning, leading "
                              "to sustained moves.",
            source="crypto_generator",
            priority=8
        ))
        
        return hypotheses
    
    def _generate_whale_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate whale activity detection hypotheses."""
        hypotheses = []
        
        def whale_volume_detection(df, vol_zscore_thresh=3.0):
            """Detect and follow whale volume spikes."""
            signals = pd.Series(0, index=df.index)
            
            # Volume z-score
            vol_ma = df['volume'].rolling(50).mean()
            vol_std = df['volume'].rolling(50).std()
            vol_zscore = (df['volume'] - vol_ma) / vol_std.replace(0, np.nan)
            
            # Whale volume (>3 std devs)
            whale_vol = vol_zscore > vol_zscore_thresh
            
            # Direction from price action
            bullish = df['close'] > df['open']
            
            # Follow whale direction
            signals[(whale_vol) & (bullish)] = 1
            signals[(whale_vol) & (~bullish)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("whale_volume"),
            name="Whale Volume Detection",
            description="Extreme volume spikes often indicate whale activity. "
                       "Follow the direction of whale trades.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.INFORMATION_ASYMMETRY,
            signal_generator=whale_volume_detection,
            parameters={'vol_zscore_thresh': 3.0},
            required_features=['open', 'close', 'volume'],
            economic_rationale="Large players have information advantages and must "
                              "trade in size. Extreme volume spikes reveal their "
                              "positioning and often predict direction.",
            source="crypto_generator",
            priority=7
        ))
        
        # Whale divergence (volume vs price)
        def whale_divergence(df, lookback=10):
            """Detect volume-price divergence indicating smart money."""
            signals = pd.Series(0, index=df.index)
            
            # Price momentum
            price_momentum = df['close'].pct_change(lookback)
            
            # Volume momentum
            vol_momentum = df['volume'].pct_change(lookback)
            
            # Divergence: high volume, low price movement
            vol_ma = df['volume'].rolling(20).mean()
            vol_std = df['volume'].rolling(20).std()
            vol_zscore = (df['volume'] - vol_ma) / vol_std.replace(0, np.nan)
            
            # High volume but small price change = accumulation/distribution
            high_vol = vol_zscore > 1.5
            small_move = price_momentum.abs() < 0.005
            
            # Direction from recent candles
            recent_up = df['close'].rolling(3).mean() > df['close'].shift(3).rolling(3).mean()
            
            signals[(high_vol) & (small_move) & (recent_up)] = 1
            signals[(high_vol) & (small_move) & (~recent_up)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("whale_divergence"),
            name="Whale Accumulation Divergence",
            description="High volume with small price movement indicates "
                       "smart money accumulation or distribution.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.INFORMATION_ASYMMETRY,
            signal_generator=whale_divergence,
            parameters={'lookback': 10},
            required_features=['close', 'volume'],
            economic_rationale="Whales must accumulate slowly to avoid moving price. "
                              "High volume without price impact suggests absorption "
                              "of counter-party flow.",
            source="crypto_generator",
            priority=8
        ))
        
        return hypotheses
    
    def _generate_exhaustion_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate momentum exhaustion hypotheses."""
        hypotheses = []
        
        def momentum_exhaustion(df, momentum_bars=20, exhaust_thresh=0.95):
            """Detect momentum exhaustion for mean reversion."""
            signals = pd.Series(0, index=df.index)
            
            # Calculate momentum
            returns = df['close'].pct_change()
            momentum = returns.rolling(momentum_bars).sum()
            
            # Momentum percentile
            momentum_pct = momentum.rolling(200).rank(pct=True)
            
            # Exhaustion at extremes
            bullish_exhaust = momentum_pct > exhaust_thresh
            bearish_exhaust = momentum_pct < (1 - exhaust_thresh)
            
            # Also check RSI-like overbought/oversold
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Combined signal
            signals[(bullish_exhaust) & (rsi > 70)] = -1
            signals[(bearish_exhaust) & (rsi < 30)] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("momentum_exhaustion"),
            name="Momentum Exhaustion Reversal",
            description="Extreme momentum combined with overbought/oversold "
                       "conditions often precedes reversals.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=momentum_exhaustion,
            parameters={'momentum_bars': 20, 'exhaust_thresh': 0.95},
            required_features=['close'],
            economic_rationale="Extended momentum runs exhaust marginal buyers/sellers. "
                              "At extremes, profit-taking and contrarian flow "
                              "typically causes reversal.",
            source="crypto_generator",
            priority=7
        ))
        
        # RSI divergence
        def rsi_divergence(df, lookback=14, threshold=30):
            """Detect RSI-price divergence."""
            signals = pd.Series(0, index=df.index)
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(lookback).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Price making new lows but RSI higher (bullish divergence)
            price_lower = df['close'] < df['close'].rolling(10).min().shift(1)
            rsi_higher = rsi > rsi.rolling(10).min().shift(1)
            bullish_div = price_lower & rsi_higher & (rsi < threshold + 10)
            
            # Price making new highs but RSI lower (bearish divergence)
            price_higher = df['close'] > df['close'].rolling(10).max().shift(1)
            rsi_lower = rsi < rsi.rolling(10).max().shift(1)
            bearish_div = price_higher & rsi_lower & (rsi > 100 - threshold - 10)
            
            signals[bullish_div] = 1
            signals[bearish_div] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("rsi_divergence"),
            name="RSI Price Divergence",
            description="When price makes new extremes but RSI doesn't confirm, "
                       "it signals weakening momentum and potential reversal.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=rsi_divergence,
            parameters={'lookback': 14, 'threshold': 30},
            required_features=['close'],
            economic_rationale="Divergences indicate underlying weakness in trends. "
                              "Fewer participants are driving price to extremes, "
                              "suggesting exhaustion.",
            source="crypto_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_funding_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate funding rate-based hypotheses (requires funding_rate column)."""
        hypotheses = []
        
        def funding_extreme_fade(df, funding_thresh=0.01):
            """Fade extreme funding rates."""
            signals = pd.Series(0, index=df.index)
            
            if 'funding_rate' not in df.columns:
                return signals
            
            # Extreme positive funding (longs pay shorts)
            extreme_positive = df['funding_rate'] > funding_thresh
            
            # Extreme negative funding (shorts pay longs)
            extreme_negative = df['funding_rate'] < -funding_thresh
            
            # Fade extremes
            signals[extreme_positive] = -1  # Too many longs, expect correction
            signals[extreme_negative] = 1   # Too many shorts, expect squeeze
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("funding_fade"),
            name="Funding Rate Extreme Fade",
            description="Extreme funding rates indicate crowded positioning. "
                       "Fade extreme positive funding, buy extreme negative.",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=funding_extreme_fade,
            parameters={'funding_thresh': 0.01},
            required_features=['funding_rate'],
            economic_rationale="High positive funding means longs are desperate and "
                              "paying premium. This crowded trade typically reverses. "
                              "Negative funding indicates shorts are crowded.",
            source="crypto_generator",
            priority=9
        ))
        
        return hypotheses
    
    def _generate_oi_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate open interest-based hypotheses (requires open_interest column)."""
        hypotheses = []
        
        def oi_divergence(df, lookback=24):
            """Detect OI-price divergence."""
            signals = pd.Series(0, index=df.index)
            
            if 'open_interest' not in df.columns:
                return signals
            
            # OI change
            oi_change = df['open_interest'].pct_change(lookback)
            
            # Price change
            price_change = df['close'].pct_change(lookback)
            
            # Rising price, falling OI = short covering (weak rally)
            weak_rally = (price_change > 0.02) & (oi_change < -0.05)
            
            # Falling price, falling OI = long liquidation (weak selloff)
            weak_selloff = (price_change < -0.02) & (oi_change < -0.05)
            
            # Strong signals: rising OI with price
            strong_rally = (price_change > 0.02) & (oi_change > 0.05)
            strong_selloff = (price_change < -0.02) & (oi_change > 0.05)
            
            # Fade weak moves, follow strong moves
            signals[weak_rally] = -1
            signals[weak_selloff] = 1
            signals[strong_rally] = 1
            signals[strong_selloff] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("oi_divergence"),
            name="Open Interest Divergence",
            description="OI divergence from price reveals whether moves are "
                       "driven by new positions (strong) or liquidations (weak).",
            hypothesis_type=HypothesisType.FLOW,
            mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
            signal_generator=oi_divergence,
            parameters={'lookback': 24},
            required_features=['open_interest', 'close'],
            economic_rationale="Rising OI with price means new longs entering (conviction). "
                              "Falling OI with rising price means shorts covering (weak). "
                              "This distinction predicts continuation vs reversal.",
            source="crypto_generator",
            priority=9
        ))
        
        return hypotheses
    
    def _generate_liquidation_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate liquidation-based hypotheses."""
        hypotheses = []
        
        def liquidation_cascade_fade(df, liq_zscore_thresh=2.5):
            """Fade liquidation cascades."""
            signals = pd.Series(0, index=df.index)
            
            liq_col = None
            if 'liquidations' in df.columns:
                liq_col = 'liquidations'
            elif 'long_liquidations' in df.columns:
                liq_col = 'long_liquidations'
            
            if liq_col is None:
                return signals
            
            # Liquidation z-score
            liq = df[liq_col]
            liq_ma = liq.rolling(100).mean()
            liq_std = liq.rolling(100).std()
            liq_zscore = (liq - liq_ma) / liq_std.replace(0, np.nan)
            
            # Extreme liquidations
            extreme_liq = liq_zscore > liq_zscore_thresh
            
            # Price direction
            price_down = df['close'].pct_change(3) < -0.01
            price_up = df['close'].pct_change(3) > 0.01
            
            # Fade after long liquidation cascades
            if 'long_liquidations' in df.columns:
                signals[(extreme_liq) & (price_down)] = 1
            elif 'short_liquidations' in df.columns:
                short_liq_zscore = (df['short_liquidations'] - df['short_liquidations'].rolling(100).mean()) / df['short_liquidations'].rolling(100).std()
                signals[(short_liq_zscore > liq_zscore_thresh) & (price_up)] = -1
            else:
                # Generic: fade big liquidation events
                signals[(extreme_liq) & (price_down)] = 1
                signals[(extreme_liq) & (price_up)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("liquidation_fade"),
            name="Liquidation Cascade Fade",
            description="Extreme liquidation events often mark local tops/bottoms "
                       "as forced selling exhausts one side of the market.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=liquidation_cascade_fade,
            parameters={'liq_zscore_thresh': 2.5},
            required_features=['liquidations'],
            economic_rationale="Liquidation cascades represent forced selling that "
                              "exhausts weak hands. Once leverage is flushed, price "
                              "often rebounds as selling pressure subsides.",
            source="crypto_generator",
            priority=9
        ))
        
        return hypotheses
    
    def _generate_breakout_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate range breakout hypotheses."""
        hypotheses = []
        
        def range_breakout(df, lookback=48, breakout_atr=1.5):
            """Trade breakouts from consolidation ranges."""
            signals = pd.Series(0, index=df.index)
            
            # Calculate range
            range_high = df['high'].rolling(lookback).max()
            range_low = df['low'].rolling(lookback).min()
            range_size = range_high - range_low
            
            # ATR
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            # Narrow range (consolidation)
            narrow = range_size < atr * 3
            
            # Breakout
            close_above = df['close'] > range_high.shift(1)
            close_below = df['close'] < range_low.shift(1)
            
            # Signal on breakout from narrow range
            signals[(narrow.shift(1).fillna(False)) & (close_above)] = 1
            signals[(narrow.shift(1).fillna(False)) & (close_below)] = -1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("range_breakout"),
            name="Consolidation Range Breakout",
            description="Breakouts from tight consolidation ranges often lead "
                       "to sustained directional moves.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=range_breakout,
            parameters={'lookback': 48, 'breakout_atr': 1.5},
            required_features=['high', 'low', 'close'],
            economic_rationale="Tight ranges represent equilibrium and building pressure. "
                              "Breakouts trigger stop orders and new entries, creating "
                              "momentum follow-through.",
            source="crypto_generator",
            priority=7
        ))
        
        return hypotheses
    
    def _generate_vwap_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate VWAP-based hypotheses."""
        hypotheses = []
        
        def vwap_deviation(df, dev_thresh=2.0):
            """Trade mean reversion to VWAP."""
            signals = pd.Series(0, index=df.index)
            
            # Calculate VWAP (rolling approximation)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cum_vol_price = (typical_price * df['volume']).rolling(48).sum()
            cum_vol = df['volume'].rolling(48).sum()
            vwap = cum_vol_price / cum_vol.replace(0, np.nan)
            
            # Standard deviation around VWAP
            deviation = (df['close'] - vwap).abs()
            dev_ma = deviation.rolling(48).mean()
            
            # Normalized deviation
            norm_dev = (df['close'] - vwap) / dev_ma.replace(0, np.nan)
            
            # Mean revert from extremes
            signals[norm_dev > dev_thresh] = -1
            signals[norm_dev < -dev_thresh] = 1
            
            return signals.fillna(0)
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("vwap_reversion"),
            name="VWAP Deviation Reversion",
            description="Extreme deviations from VWAP often revert as price "
                       "returns to volume-weighted fair value.",
            hypothesis_type=HypothesisType.MICROSTRUCTURE,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=vwap_deviation,
            parameters={'dev_thresh': 2.0},
            required_features=['high', 'low', 'close', 'volume'],
            economic_rationale="VWAP represents volume-weighted average price, "
                              "a key institutional benchmark. Extreme deviations "
                              "often see institutional rebalancing flow.",
            source="crypto_generator",
            priority=6
        ))
        
        return hypotheses

