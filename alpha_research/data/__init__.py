"""
Data Adapters
=============

Modular data adapters for loading different asset types
and data sources.
"""

from .adapters import (
    DataAdapter,
    CSVAdapter,
    ParquetAdapter,
    CCXTAdapter,
    AlpacaAdapter,
    UniversalAdapter,
)

__all__ = [
    "DataAdapter",
    "CSVAdapter",
    "ParquetAdapter",
    "CCXTAdapter",
    "AlpacaAdapter",
    "UniversalAdapter",
]

