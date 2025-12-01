"""
Utility functions and classes for the Alpha Research Engine.
"""

from .logging import get_logger, setup_logging
from .statistics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    bootstrap_confidence_interval,
    monte_carlo_pvalue,
    information_ratio,
    rolling_statistics,
)
from .data_utils import (
    safe_divide,
    winsorize,
    zscore,
    rolling_zscore,
    lag_features,
    create_return_labels,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio", 
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "bootstrap_confidence_interval",
    "monte_carlo_pvalue",
    "information_ratio",
    "rolling_statistics",
    "safe_divide",
    "winsorize",
    "zscore",
    "rolling_zscore",
    "lag_features",
    "create_return_labels",
]

