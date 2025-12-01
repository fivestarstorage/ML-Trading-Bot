"""
Base classes for hypothesis generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class HypothesisType(Enum):
    """Types of trading hypotheses."""
    REGIME = "regime"
    MICROSTRUCTURE = "microstructure"
    CROSS_ASSET = "cross_asset"
    FLOW = "flow"
    PATTERN = "pattern"
    SEASONAL = "seasonal"
    EVENT = "event"
    CUSTOM = "custom"


class HypothesisMechanism(Enum):
    """Economic mechanisms that could explain the edge."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    MARKET_MAKING = "market_making"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    STRUCTURAL_IMBALANCE = "structural_imbalance"
    BEHAVIORAL = "behavioral"
    LIQUIDITY_PREMIUM = "liquidity_premium"
    RISK_PREMIUM = "risk_premium"
    ARBITRAGE = "arbitrage"
    UNKNOWN = "unknown"


@dataclass
class Hypothesis:
    """
    Represents a trading hypothesis to be tested.
    
    A hypothesis is a testable proposition about market behavior
    that could potentially represent a trading edge.
    """
    id: str
    name: str
    description: str
    hypothesis_type: HypothesisType
    mechanism: HypothesisMechanism
    
    # Signal generation function
    signal_generator: Optional[Callable] = None
    
    # Parameters for the hypothesis
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Features required for this hypothesis
    required_features: List[str] = field(default_factory=list)
    
    # Economic intuition explaining why this edge might exist
    economic_rationale: str = ""
    
    # Initial statistical properties (before full validation)
    preliminary_stats: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""
    priority: int = 5  # 1-10 scale, 10 being highest
    
    # Validation results (populated later)
    is_validated: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on this hypothesis.
        
        Args:
            data: DataFrame with required features
            
        Returns:
            Series of signals (-1, 0, 1 or probability values)
        """
        if self.signal_generator is None:
            raise NotImplementedError(
                f"Signal generator not defined for hypothesis: {self.name}"
            )
        
        # Verify required features exist
        missing = set(self.required_features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        return self.signal_generator(data, **self.parameters)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'hypothesis_type': self.hypothesis_type.value,
            'mechanism': self.mechanism.value,
            'parameters': self.parameters,
            'required_features': self.required_features,
            'economic_rationale': self.economic_rationale,
            'preliminary_stats': self.preliminary_stats,
            'created_at': self.created_at.isoformat(),
            'source': self.source,
            'priority': self.priority,
            'is_validated': self.is_validated,
            'validation_results': self.validation_results,
        }
    
    def __repr__(self) -> str:
        return (
            f"Hypothesis(id='{self.id}', name='{self.name}', "
            f"type={self.hypothesis_type.value}, mechanism={self.mechanism.value})"
        )


class HypothesisGenerator(ABC):
    """
    Abstract base class for hypothesis generators.
    
    Each generator focuses on a specific type of potential edge
    (e.g., regime-based, microstructure-based, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hypothesis generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hypotheses: List[Hypothesis] = []
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """
        Generate hypotheses from the provided data.
        
        Args:
            data: DataFrame with OHLCV and potentially other data
            
        Returns:
            List of generated hypotheses
        """
        pass
    
    @abstractmethod
    def get_hypothesis_type(self) -> HypothesisType:
        """Return the type of hypotheses this generator creates."""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for hypothesis generation.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
        """
        # Default implementation: just ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        df = data.copy()
        
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        return df
    
    def validate_preliminary(self, hypothesis: Hypothesis, data: pd.DataFrame) -> bool:
        """
        Perform preliminary validation before full testing.
        
        Args:
            hypothesis: Hypothesis to validate
            data: Data to use for validation
            
        Returns:
            True if hypothesis passes preliminary checks
        """
        try:
            signals = hypothesis.generate_signals(data)
            
            # Check for sufficient signal variation
            if signals.nunique() < 2:
                return False
            
            # Check for reasonable signal frequency
            non_zero_pct = (signals != 0).sum() / len(signals)
            if non_zero_pct < 0.01 or non_zero_pct > 0.99:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_hypothesis_id(self, name: str) -> str:
        """Create a unique hypothesis ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        unique_str = f"{name}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]


class CompositeHypothesisGenerator(HypothesisGenerator):
    """
    Combines multiple hypothesis generators.
    """
    
    def __init__(self, config: Dict[str, Any], generators: List[HypothesisGenerator] = None):
        super().__init__(config)
        self.generators = generators or []
    
    def add_generator(self, generator: HypothesisGenerator):
        """Add a hypothesis generator."""
        self.generators.append(generator)
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate hypotheses from all registered generators."""
        all_hypotheses = []
        
        for generator in self.generators:
            try:
                hypotheses = generator.generate(data)
                all_hypotheses.extend(hypotheses)
            except Exception as e:
                print(f"Warning: Generator {generator.__class__.__name__} failed: {e}")
        
        return all_hypotheses
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.CUSTOM

