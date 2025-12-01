"""
Alpha Discovery Engine
======================

Core engine for discovering and validating trading edges.
"""

from .engine import AlphaDiscoveryEngine
from .validator import SignalValidator
from .ml_discovery import MLAlphaDiscovery

__all__ = [
    "AlphaDiscoveryEngine",
    "SignalValidator", 
    "MLAlphaDiscovery",
]

