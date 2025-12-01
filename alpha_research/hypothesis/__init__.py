"""
Hypothesis Generation Module
============================

Automatically proposes potential trading edges based on:
- Regime changes and volatility clusters
- Microstructure patterns
- Cross-asset relationships
- Flow asymmetries
- Pattern discovery via embeddings and clustering
- Event-driven anomalies
"""

from .base import HypothesisGenerator, Hypothesis
from .regime import RegimeHypothesisGenerator
from .microstructure import MicrostructureHypothesisGenerator
from .cross_asset import CrossAssetHypothesisGenerator
from .flow import FlowHypothesisGenerator
from .pattern import PatternHypothesisGenerator
from .seasonal import SeasonalHypothesisGenerator

__all__ = [
    "HypothesisGenerator",
    "Hypothesis",
    "RegimeHypothesisGenerator",
    "MicrostructureHypothesisGenerator", 
    "CrossAssetHypothesisGenerator",
    "FlowHypothesisGenerator",
    "PatternHypothesisGenerator",
    "SeasonalHypothesisGenerator",
]

