"""
Pattern-based hypothesis generation using unsupervised learning.

Focuses on discovering patterns through:
- Embeddings and dimensionality reduction
- Clustering of price/volume sequences
- Motif discovery
- Anomaly detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .base import (
    HypothesisGenerator, 
    Hypothesis, 
    HypothesisType, 
    HypothesisMechanism
)


class PatternHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses through pattern discovery.
    
    Uses unsupervised learning to find:
    - Recurring price patterns with predictive value
    - Return distribution clusters
    - Unusual sequences that precede moves
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dim = config.get('embedding_dim', 32)
        self.cluster_min_samples = config.get('cluster_min_samples', 50)
        self.motif_min_length = config.get('motif_min_length', 5)
        self.motif_max_length = config.get('motif_max_length', 50)
        self.n_clusters = config.get('n_pattern_clusters', 10)
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.PATTERN
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate pattern-based hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Return distribution clustering
        cluster_hypotheses = self._generate_return_cluster_hypotheses(data)
        hypotheses.extend(cluster_hypotheses)
        
        # 2. Candlestick pattern clustering
        candle_hypotheses = self._generate_candlestick_cluster_hypotheses(data)
        hypotheses.extend(candle_hypotheses)
        
        # 3. Sequence embedding patterns
        seq_hypotheses = self._generate_sequence_hypotheses(data)
        hypotheses.extend(seq_hypotheses)
        
        # 4. Anomaly-based patterns
        anomaly_hypotheses = self._generate_anomaly_hypotheses(data)
        hypotheses.extend(anomaly_hypotheses)
        
        return hypotheses
    
    def _create_sequence_features(self, data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """Create features from price sequences."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling return features
        for i in range(1, lookback + 1):
            features[f'ret_{i}'] = data['returns'].shift(i)
        
        # Rolling volatility features
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = data['returns'].rolling(window).std()
        
        # Normalized price position
        for window in [10, 20, 50]:
            high = data['high'].rolling(window).max()
            low = data['low'].rolling(window).min()
            features[f'pos_{window}'] = (data['close'] - low) / (high - low).replace(0, np.nan)
        
        # Volume features
        for window in [5, 10, 20]:
            features[f'vol_ratio_{window}'] = data['volume'] / data['volume'].rolling(window).mean()
        
        return features.fillna(0)
    
    def _generate_return_cluster_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Cluster returns to find regime-like patterns."""
        hypotheses = []
        
        # Create feature set for clustering
        features = self._create_sequence_features(data, lookback=5)
        features = features.dropna()
        
        if len(features) < self.cluster_min_samples * 3:
            return hypotheses
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Reduce dimensionality
        pca = PCA(n_components=min(10, scaled_features.shape[1]))
        reduced_features = pca.fit_transform(scaled_features)
        
        # Cluster
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(reduced_features)
        
        # Analyze each cluster for predictive value
        features['cluster'] = clusters
        features['future_return'] = data.loc[features.index, 'returns'].shift(-1)
        
        cluster_stats = features.groupby('cluster')['future_return'].agg(['mean', 'std', 'count'])
        cluster_stats['sharpe'] = cluster_stats['mean'] / cluster_stats['std'].replace(0, np.nan)
        
        # Find clusters with significant edges
        significant_clusters = cluster_stats[
            (cluster_stats['count'] >= self.cluster_min_samples) &
            (cluster_stats['sharpe'].abs() > 0.5)
        ]
        
        if len(significant_clusters) == 0:
            return hypotheses
        
        # Store cluster info in data for signal generation
        data.loc[features.index, 'pattern_cluster'] = clusters
        
        # Find best bullish and bearish clusters
        if len(significant_clusters[significant_clusters['sharpe'] > 0]) > 0:
            best_bull = significant_clusters['sharpe'].idxmax()
            bull_sharpe = cluster_stats.loc[best_bull, 'sharpe']
        else:
            best_bull = None
            bull_sharpe = 0
            
        if len(significant_clusters[significant_clusters['sharpe'] < 0]) > 0:
            best_bear = significant_clusters['sharpe'].idxmin()
            bear_sharpe = cluster_stats.loc[best_bear, 'sharpe']
        else:
            best_bear = None
            bear_sharpe = 0
        
        def pattern_cluster_signal(df, bull_cluster, bear_cluster):
            signals = pd.Series(0, index=df.index)
            
            if 'pattern_cluster' not in df.columns:
                return signals
            
            if bull_cluster is not None:
                signals[df['pattern_cluster'] == bull_cluster] = 1
            if bear_cluster is not None:
                signals[df['pattern_cluster'] == bear_cluster] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("pattern_cluster"),
            name="Return Pattern Clusters",
            description=f"Identified {len(significant_clusters)} clusters with predictive value. "
                       f"Best bullish: {bull_sharpe:.2f} Sharpe, Best bearish: {bear_sharpe:.2f} Sharpe.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.UNKNOWN,
            signal_generator=pattern_cluster_signal,
            parameters={'bull_cluster': int(best_bull) if best_bull is not None else None, 
                       'bear_cluster': int(best_bear) if best_bear is not None else None},
            required_features=['pattern_cluster'],
            economic_rationale="Clustering reveals hidden structure in returns. "
                              "Some configurations consistently precede specific outcomes, "
                              "potentially due to market microstructure or behavioral patterns.",
            source="pattern_generator",
            priority=5,
            preliminary_stats={
                'n_clusters': int(len(significant_clusters)),
                'best_bull_sharpe': float(bull_sharpe) if bull_sharpe else 0,
                'best_bear_sharpe': float(bear_sharpe) if bear_sharpe else 0,
            }
        ))
        
        return hypotheses
    
    def _generate_candlestick_cluster_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Cluster candlestick patterns."""
        hypotheses = []
        
        # Create candlestick features
        candle_features = pd.DataFrame(index=data.index)
        
        body = data['close'] - data['open']
        range_hl = data['high'] - data['low']
        
        candle_features['body_ratio'] = body / range_hl.replace(0, np.nan)
        candle_features['upper_wick'] = (data['high'] - data[['open', 'close']].max(axis=1)) / range_hl.replace(0, np.nan)
        candle_features['lower_wick'] = (data[['open', 'close']].min(axis=1) - data['low']) / range_hl.replace(0, np.nan)
        candle_features['range_atr'] = range_hl / range_hl.rolling(14).mean()
        candle_features['gap'] = data['open'] / data['close'].shift(1) - 1
        
        # Sequence of candle features (last 3 candles)
        for col in ['body_ratio', 'range_atr']:
            for shift in [1, 2]:
                candle_features[f'{col}_lag{shift}'] = candle_features[col].shift(shift)
        
        candle_features = candle_features.dropna()
        
        if len(candle_features) < self.cluster_min_samples * 3:
            return hypotheses
        
        # Cluster
        scaler = StandardScaler()
        scaled = scaler.fit_transform(candle_features)
        
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)
        
        # Analyze predictive value
        candle_features['cluster'] = clusters
        candle_features['future_return'] = data.loc[candle_features.index, 'returns'].shift(-1)
        
        cluster_stats = candle_features.groupby('cluster')['future_return'].agg(['mean', 'std', 'count'])
        cluster_stats['sharpe'] = cluster_stats['mean'] / cluster_stats['std'].replace(0, np.nan)
        
        # Store in data
        data.loc[candle_features.index, 'candle_cluster'] = clusters
        
        # Find extreme clusters
        top_bull = cluster_stats[cluster_stats['count'] >= 30]['sharpe'].nlargest(1)
        top_bear = cluster_stats[cluster_stats['count'] >= 30]['sharpe'].nsmallest(1)
        
        if len(top_bull) > 0 and len(top_bear) > 0:
            bull_cluster = top_bull.index[0]
            bear_cluster = top_bear.index[0]
            
            def candle_cluster_signal(df, bull, bear):
                signals = pd.Series(0, index=df.index)
                
                if 'candle_cluster' not in df.columns:
                    return signals
                
                signals[df['candle_cluster'] == bull] = 1
                signals[df['candle_cluster'] == bear] = -1
                
                return signals
            
            hypotheses.append(Hypothesis(
                id=self._create_hypothesis_id("candle_cluster"),
                name="Candlestick Pattern Clusters",
                description="Unsupervised clustering of candlestick patterns "
                           "reveals configurations with predictive value.",
                hypothesis_type=HypothesisType.PATTERN,
                mechanism=HypothesisMechanism.BEHAVIORAL,
                signal_generator=candle_cluster_signal,
                parameters={'bull': int(bull_cluster), 'bear': int(bear_cluster)},
                required_features=['candle_cluster'],
                economic_rationale="Candlestick patterns encode market psychology. "
                                  "Clustering finds robust patterns that may not match "
                                  "traditional named patterns but have statistical edge.",
                source="pattern_generator",
                priority=5
            ))
        
        return hypotheses
    
    def _generate_sequence_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses from return sequences."""
        hypotheses = []
        
        returns = data['returns'].dropna()
        
        # Look for specific return patterns
        # Pattern: 3 down days followed by reversal
        def consecutive_down_reversal(df, n_down=3, reversal_threshold=0.01):
            signals = pd.Series(0, index=df.index)
            
            consecutive_down = pd.Series(True, index=df.index)
            for i in range(1, n_down + 1):
                consecutive_down = consecutive_down & (df['returns'].shift(i) < 0)
            
            # Strong down sequence
            cumulative_down = sum(df['returns'].shift(i) for i in range(1, n_down + 1))
            strong_down = cumulative_down < -reversal_threshold
            
            # Signal reversal
            signals[consecutive_down & strong_down] = 1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("consec_down_reversal"),
            name="Consecutive Down Reversal",
            description="After N consecutive down periods with significant loss, "
                       "mean reversion becomes more likely.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.MEAN_REVERSION,
            signal_generator=consecutive_down_reversal,
            parameters={'n_down': 3, 'reversal_threshold': 0.01},
            required_features=['returns'],
            economic_rationale="Extended losing streaks create oversold conditions. "
                              "Behavioral biases cause over-reaction which then corrects.",
            source="pattern_generator",
            priority=5
        ))
        
        # Pattern: Increasing volatility sequence
        def vol_expansion_momentum(df, vol_lookback=5, expansion_thresh=1.5):
            signals = pd.Series(0, index=df.index)
            
            vol = df['returns'].rolling(vol_lookback).std()
            vol_ratio = vol / vol.rolling(20).mean()
            
            expanding = vol_ratio > expansion_thresh
            momentum = df['returns'].rolling(vol_lookback).mean()
            
            signals[(expanding) & (momentum > 0)] = 1
            signals[(expanding) & (momentum < 0)] = -1
            
            return signals
        
        hypotheses.append(Hypothesis(
            id=self._create_hypothesis_id("vol_expansion_momo"),
            name="Volatility Expansion Momentum",
            description="When volatility expands, follow the direction of "
                       "the recent move as breakouts tend to continue.",
            hypothesis_type=HypothesisType.PATTERN,
            mechanism=HypothesisMechanism.MOMENTUM,
            signal_generator=vol_expansion_momentum,
            parameters={'vol_lookback': 5, 'expansion_thresh': 1.5},
            required_features=['returns'],
            economic_rationale="Volatility expansion often signals the start of "
                              "a new trend as the market transitions regimes.",
            source="pattern_generator",
            priority=6
        ))
        
        return hypotheses
    
    def _generate_anomaly_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses based on anomaly detection."""
        hypotheses = []
        
        # Create features for anomaly detection
        features = self._create_sequence_features(data, lookback=5)
        features = features.dropna()
        
        if len(features) < 100:
            return hypotheses
        
        # Use DBSCAN for anomaly detection
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(5, scaled.shape[1]))
        reduced = pca.fit_transform(scaled)
        
        # DBSCAN clustering (outliers are labeled -1)
        dbscan = DBSCAN(eps=1.5, min_samples=10)
        labels = dbscan.fit_predict(reduced)
        
        # Store anomaly flags
        data.loc[features.index, 'is_anomaly'] = (labels == -1).astype(int)
        
        # Analyze anomaly returns
        features['is_anomaly'] = (labels == -1)
        features['future_return'] = data.loc[features.index, 'returns'].shift(-1)
        
        # Check if anomalies have predictive value
        anomaly_returns = features[features['is_anomaly']]['future_return']
        normal_returns = features[~features['is_anomaly']]['future_return']
        
        if len(anomaly_returns) >= 20:
            t_stat, p_value = stats.ttest_ind(anomaly_returns.dropna(), normal_returns.dropna())
            
            if p_value < 0.1:  # Marginally significant
                def anomaly_signal(df, direction=1):
                    signals = pd.Series(0, index=df.index)
                    
                    if 'is_anomaly' not in df.columns:
                        return signals
                    
                    # If anomalies predict positive returns, go long
                    # If anomalies predict negative returns, go short
                    signals[df['is_anomaly'] == 1] = direction
                    
                    return signals
                
                direction = 1 if anomaly_returns.mean() > 0 else -1
                
                hypotheses.append(Hypothesis(
                    id=self._create_hypothesis_id("anomaly_signal"),
                    name="Anomaly-Based Signal",
                    description=f"Anomalous market states (detected via DBSCAN) "
                               f"predict {'positive' if direction > 0 else 'negative'} returns.",
                    hypothesis_type=HypothesisType.PATTERN,
                    mechanism=HypothesisMechanism.UNKNOWN,
                    signal_generator=anomaly_signal,
                    parameters={'direction': direction},
                    required_features=['is_anomaly'],
                    economic_rationale="Anomalous market states represent unusual "
                                      "configurations that may precede specific outcomes "
                                      "due to market microstructure or behavioral factors.",
                    source="pattern_generator",
                    priority=4,
                    preliminary_stats={
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'n_anomalies': int(len(anomaly_returns)),
                    }
                ))
        
        return hypotheses

