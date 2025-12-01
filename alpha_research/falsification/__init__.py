"""
Falsification Module
====================

Comprehensive stress testing to destroy weak edges.
Any edge that survives falsification has a higher probability
of being a genuine, exploitable inefficiency.
"""

from .stress_test import StressTester
from .robustness import RobustnessAnalyzer

__all__ = [
    "StressTester",
    "RobustnessAnalyzer",
]

