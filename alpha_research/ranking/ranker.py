"""
Edge Ranker

Comprehensive ranking system for discovered trading edges.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..hypothesis.base import Hypothesis, HypothesisMechanism
from ..discovery.engine import EdgeCandidate
from ..falsification.stress_test import FalsificationResult
from ..utils.logging import get_logger

logger = get_logger()


class EconomicScore(Enum):
    """Economic intuition scoring levels."""
    STRONG = 1.0  # Well-documented, economically sensible mechanism
    MODERATE = 0.7  # Plausible mechanism, some theoretical backing
    WEAK = 0.4  # Unclear mechanism but statistically significant
    UNKNOWN = 0.2  # No clear economic rationale


@dataclass
class RankedEdge:
    """A ranked edge with comprehensive scoring."""
    
    # Identity
    edge_id: str
    edge_name: str
    hypothesis_type: str
    mechanism: str
    
    # Core metrics
    oos_sharpe: float
    net_sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    n_trades: int
    
    # Statistical measures
    p_value: float
    t_statistic: float
    bootstrap_ci: tuple
    
    # Robustness scores
    falsification_score: float
    regime_stability: float
    
    # Component scores (0-1 scale)
    score_sharpe: float = 0.0
    score_stability: float = 0.0
    score_economic: float = 0.0
    score_statistical: float = 0.0
    score_turnover: float = 0.0
    score_simplicity: float = 0.0
    score_regime: float = 0.0
    
    # Overall ranking
    overall_score: float = 0.0
    rank: int = 0
    
    # Qualitative assessment
    economic_rationale: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rank': self.rank,
            'edge_id': self.edge_id,
            'edge_name': self.edge_name,
            'hypothesis_type': self.hypothesis_type,
            'mechanism': self.mechanism,
            'overall_score': self.overall_score,
            'oos_sharpe': self.oos_sharpe,
            'net_sharpe': self.net_sharpe,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'n_trades': self.n_trades,
            'p_value': self.p_value,
            'falsification_score': self.falsification_score,
            'regime_stability': self.regime_stability,
            'score_sharpe': self.score_sharpe,
            'score_stability': self.score_stability,
            'score_economic': self.score_economic,
            'score_statistical': self.score_statistical,
            'score_turnover': self.score_turnover,
            'score_simplicity': self.score_simplicity,
            'score_regime': self.score_regime,
            'economic_rationale': self.economic_rationale,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'recommendation': self.recommendation,
        }


class EdgeRanker:
    """
    Ranks edges using a weighted multi-factor scoring system.
    
    Factors:
    1. Out-of-sample Sharpe (after costs)
    2. Stability across time
    3. Economic intuition
    4. Statistical significance
    5. Turnover efficiency
    6. Simplicity vs predictive power
    7. Regime robustness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        ranking_config = config.get('ranking', {})
        
        # Scoring weights
        self.weights = ranking_config.get('weights', {
            'sharpe_oos': 0.25,
            'stability': 0.20,
            'economic_intuition': 0.15,
            'statistical_significance': 0.15,
            'turnover_efficiency': 0.10,
            'simplicity': 0.10,
            'regime_robustness': 0.05
        })
        
        # Thresholds
        self.min_overall_score = ranking_config.get('min_overall_score', 0.5)
        self.min_economic_score = ranking_config.get('min_economic_score', 0.3)
        
        # Economic mechanism scores
        self.mechanism_scores = {
            HypothesisMechanism.ARBITRAGE: EconomicScore.STRONG,
            HypothesisMechanism.RISK_PREMIUM: EconomicScore.STRONG,
            HypothesisMechanism.LIQUIDITY_PREMIUM: EconomicScore.STRONG,
            HypothesisMechanism.INFORMATION_ASYMMETRY: EconomicScore.MODERATE,
            HypothesisMechanism.STRUCTURAL_IMBALANCE: EconomicScore.MODERATE,
            HypothesisMechanism.MOMENTUM: EconomicScore.MODERATE,
            HypothesisMechanism.MEAN_REVERSION: EconomicScore.MODERATE,
            HypothesisMechanism.BEHAVIORAL: EconomicScore.WEAK,
            HypothesisMechanism.MARKET_MAKING: EconomicScore.MODERATE,
            HypothesisMechanism.UNKNOWN: EconomicScore.UNKNOWN,
        }
    
    def rank_edges(
        self,
        candidates: List[EdgeCandidate],
        falsification_results: Dict[str, FalsificationResult] = None
    ) -> List[RankedEdge]:
        """
        Rank all edge candidates.
        
        Args:
            candidates: List of validated edge candidates
            falsification_results: Optional falsification results by edge ID
            
        Returns:
            List of ranked edges, sorted by overall score
        """
        logger.info(f"Ranking {len(candidates)} edge candidates...")
        
        ranked_edges = []
        
        for candidate in candidates:
            # Get falsification result if available
            falsif_result = None
            if falsification_results:
                falsif_result = falsification_results.get(candidate.hypothesis.id)
            
            ranked = self._score_edge(candidate, falsif_result)
            ranked_edges.append(ranked)
        
        # Sort by overall score
        ranked_edges.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign ranks
        for i, edge in enumerate(ranked_edges):
            edge.rank = i + 1
        
        # Generate recommendations
        for edge in ranked_edges:
            edge.recommendation = self._generate_recommendation(edge)
        
        logger.info(f"Ranked {len(ranked_edges)} edges")
        
        return ranked_edges
    
    def _score_edge(
        self,
        candidate: EdgeCandidate,
        falsif_result: Optional[FalsificationResult]
    ) -> RankedEdge:
        """Score a single edge candidate."""
        
        hypothesis = candidate.hypothesis
        
        # 1. Sharpe score (normalize to 0-1)
        # Assume Sharpe of 2.0 is excellent
        score_sharpe = min(candidate.oos_sharpe / 2.0, 1.0) if candidate.oos_sharpe > 0 else 0.0
        
        # 2. Stability score
        score_stability = candidate.regime_stability
        
        # 3. Economic intuition score
        mechanism_score = self.mechanism_scores.get(
            hypothesis.mechanism, 
            EconomicScore.UNKNOWN
        ).value
        
        # Boost if hypothesis has detailed rationale
        if len(hypothesis.economic_rationale) > 100:
            mechanism_score = min(mechanism_score * 1.2, 1.0)
        
        score_economic = mechanism_score
        
        # 4. Statistical significance score
        # p-value of 0.01 = score 1.0, p-value of 0.05 = score 0.5
        score_statistical = max(0, 1 - (candidate.p_value / 0.05)) if candidate.p_value < 0.05 else 0.0
        
        # 5. Turnover efficiency score
        # Lower turnover is better (less transaction costs)
        turnover_penalty = min(candidate.turnover * 10, 1.0)  # Penalize high turnover
        score_turnover = 1 - turnover_penalty
        
        # 6. Simplicity score
        # Fewer required features = simpler
        n_features = len(hypothesis.required_features)
        score_simplicity = max(0, 1 - (n_features / 20))  # 20 features = score 0
        
        # 7. Regime robustness score
        if falsif_result:
            score_regime = falsif_result.falsification_score
        else:
            score_regime = candidate.regime_stability
        
        # Calculate overall score
        overall_score = (
            self.weights['sharpe_oos'] * score_sharpe +
            self.weights['stability'] * score_stability +
            self.weights['economic_intuition'] * score_economic +
            self.weights['statistical_significance'] * score_statistical +
            self.weights['turnover_efficiency'] * score_turnover +
            self.weights['simplicity'] * score_simplicity +
            self.weights['regime_robustness'] * score_regime
        )
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_qualities(
            candidate, score_sharpe, score_stability, score_economic,
            score_statistical, score_turnover, score_simplicity, score_regime
        )
        
        return RankedEdge(
            edge_id=hypothesis.id,
            edge_name=hypothesis.name,
            hypothesis_type=hypothesis.hypothesis_type.value,
            mechanism=hypothesis.mechanism.value,
            oos_sharpe=candidate.oos_sharpe,
            net_sharpe=candidate.net_sharpe,
            win_rate=candidate.win_rate,
            profit_factor=candidate.profit_factor,
            max_drawdown=candidate.max_drawdown,
            n_trades=candidate.n_trades,
            p_value=candidate.p_value,
            t_statistic=candidate.t_statistic,
            bootstrap_ci=(candidate.bootstrap_ci_lower, candidate.bootstrap_ci_upper),
            falsification_score=falsif_result.falsification_score if falsif_result else 0.0,
            regime_stability=candidate.regime_stability,
            score_sharpe=score_sharpe,
            score_stability=score_stability,
            score_economic=score_economic,
            score_statistical=score_statistical,
            score_turnover=score_turnover,
            score_simplicity=score_simplicity,
            score_regime=score_regime,
            overall_score=overall_score,
            economic_rationale=hypothesis.economic_rationale,
            strengths=strengths,
            weaknesses=weaknesses,
        )
    
    def _identify_qualities(
        self,
        candidate: EdgeCandidate,
        score_sharpe: float,
        score_stability: float,
        score_economic: float,
        score_statistical: float,
        score_turnover: float,
        score_simplicity: float,
        score_regime: float
    ) -> tuple:
        """Identify strengths and weaknesses of an edge."""
        
        strengths = []
        weaknesses = []
        
        # Sharpe analysis
        if score_sharpe >= 0.8:
            strengths.append("Excellent risk-adjusted returns")
        elif score_sharpe >= 0.5:
            strengths.append("Good risk-adjusted returns")
        elif score_sharpe < 0.3:
            weaknesses.append("Low risk-adjusted returns")
        
        # Stability analysis
        if score_stability >= 0.7:
            strengths.append("Very stable across time")
        elif score_stability < 0.4:
            weaknesses.append("Performance varies significantly over time")
        
        # Economic rationale
        if score_economic >= 0.7:
            strengths.append("Strong economic rationale")
        elif score_economic < 0.4:
            weaknesses.append("Weak economic explanation")
        
        # Statistical significance
        if score_statistical >= 0.8:
            strengths.append("Highly statistically significant")
        elif score_statistical < 0.5:
            weaknesses.append("Marginal statistical significance")
        
        # Turnover
        if score_turnover >= 0.7:
            strengths.append("Low trading costs")
        elif score_turnover < 0.4:
            weaknesses.append("High turnover erodes returns")
        
        # Simplicity
        if score_simplicity >= 0.7:
            strengths.append("Simple and interpretable")
        elif score_simplicity < 0.4:
            weaknesses.append("Complex signal generation")
        
        # Regime robustness
        if score_regime >= 0.7:
            strengths.append("Works across market regimes")
        elif score_regime < 0.4:
            weaknesses.append("Regime-dependent performance")
        
        # Trade count
        if candidate.n_trades >= 100:
            strengths.append("Large sample size")
        elif candidate.n_trades < 50:
            weaknesses.append("Limited trade sample")
        
        # Win rate
        if candidate.win_rate >= 0.55:
            strengths.append(f"High win rate ({candidate.win_rate:.1%})")
        elif candidate.win_rate < 0.45:
            weaknesses.append(f"Low win rate ({candidate.win_rate:.1%})")
        
        # Max drawdown
        if candidate.max_drawdown <= 0.15:
            strengths.append("Controlled drawdowns")
        elif candidate.max_drawdown > 0.30:
            weaknesses.append(f"Large drawdowns ({candidate.max_drawdown:.1%})")
        
        return strengths, weaknesses
    
    def _generate_recommendation(self, edge: RankedEdge) -> str:
        """Generate a recommendation for the edge."""
        
        if edge.overall_score >= 0.75:
            base = "STRONG CANDIDATE: "
            action = "Consider for live testing with small size."
        elif edge.overall_score >= 0.6:
            base = "PROMISING CANDIDATE: "
            action = "Worthy of further research and extended backtesting."
        elif edge.overall_score >= 0.5:
            base = "MODERATE CANDIDATE: "
            action = "Requires additional validation before consideration."
        else:
            base = "WEAK CANDIDATE: "
            action = "Likely not a genuine edge. Recommend rejection."
        
        # Add specific advice based on weaknesses
        advice = []
        if "Low risk-adjusted returns" in edge.weaknesses:
            advice.append("Look for ways to improve signal quality.")
        if "Weak economic explanation" in edge.weaknesses:
            advice.append("Investigate underlying mechanism more thoroughly.")
        if "High turnover erodes returns" in edge.weaknesses:
            advice.append("Consider signal smoothing or longer holding periods.")
        if "Regime-dependent performance" in edge.weaknesses:
            advice.append("Add regime detection for position sizing.")
        if "Limited trade sample" in edge.weaknesses:
            advice.append("Test on longer history or additional assets.")
        
        recommendation = base + action
        if advice:
            recommendation += " " + " ".join(advice)
        
        return recommendation
    
    def get_top_edges(self, ranked: List[RankedEdge], n: int = 10) -> List[RankedEdge]:
        """Get top N edges above minimum score threshold."""
        filtered = [e for e in ranked if e.overall_score >= self.min_overall_score]
        return filtered[:n]
    
    def to_dataframe(self, ranked: List[RankedEdge]) -> pd.DataFrame:
        """Convert ranked edges to DataFrame."""
        records = [e.to_dict() for e in ranked]
        return pd.DataFrame(records)

