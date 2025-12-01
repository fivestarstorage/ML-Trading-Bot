"""
Alpha Research Engine
======================

A professional-grade quant research system for discovering statistically valid,
repeatable, and economically explainable trading edges.

The system is designed to:
1. Generate and test hypotheses about market mispricing
2. Discover hidden structure in price movements and alternative data
3. Rigorously validate edges using walk-forward and statistical testing
4. Eliminate false discoveries through comprehensive falsification
5. Rank and explain potential alpha sources

Author: Alpha Research Team
"""

from .orchestrator import AlphaResearchOrchestrator
from .config import AlphaResearchConfig

__version__ = "1.0.0"
__all__ = [
    "AlphaResearchOrchestrator",
    "AlphaResearchConfig",
]

