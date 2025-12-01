"""
Configuration management for the Alpha Research Engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data source configuration."""
    source: str = "csv"  # csv, parquet, ccxt, alpaca
    symbol: str = "BTCUSD"
    timeframe: str = "5m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    orderbook_depth: int = 10
    include_funding: bool = True
    include_oi: bool = True


@dataclass
class HypothesisConfig:
    """Hypothesis generation settings."""
    # Regime detection
    regime_lookback: int = 100
    volatility_cluster_threshold: float = 1.5
    
    # Microstructure
    orderbook_imbalance_window: int = 20
    liquidity_shift_threshold: float = 0.3
    large_player_volume_zscore: float = 2.5
    
    # Cross-asset
    basis_significance_threshold: float = 0.5
    term_structure_lookback: int = 30
    correlation_window: int = 50
    
    # Flow asymmetries
    funding_rate_threshold: float = 0.0001
    seasonal_test_periods: List[str] = field(default_factory=lambda: ["hour", "day_of_week", "month"])
    
    # Pattern discovery
    embedding_dim: int = 32
    cluster_min_samples: int = 50
    motif_min_length: int = 5
    motif_max_length: int = 50
    
    # Event-driven
    event_lookback: int = 10
    event_lookahead: int = 20
    

@dataclass
class ValidationConfig:
    """Statistical validation settings."""
    # Walk-forward
    train_pct: float = 0.7
    n_folds: int = 5
    min_fold_samples: int = 100
    
    # Transaction costs
    commission_bps: float = 4.0  # Basis points per side
    slippage_bps: float = 2.0   # Basis points per side
    
    # Statistical tests
    bootstrap_iterations: int = 1000
    monte_carlo_simulations: int = 1000
    significance_level: float = 0.05
    
    # Minimum requirements
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.1
    min_trades: int = 30


@dataclass
class FalsificationConfig:
    """Falsification and robustness testing settings."""
    # Slippage stress test
    slippage_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Entry timing randomization
    entry_noise_bars: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    entry_noise_iterations: int = 100
    
    # Label shifting
    label_shift_bars: List[int] = field(default_factory=lambda: [-3, -2, -1, 1, 2, 3])
    
    # Feature noise injection
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1])
    noise_iterations: int = 50
    
    # Rolling window stability
    window_sizes: List[int] = field(default_factory=lambda: [100, 200, 500, 1000])
    min_window_sharpe: float = 0.3
    max_sharpe_variance: float = 1.0
    
    # Regime robustness
    min_regime_count: int = 2
    max_regime_sharpe_diff: float = 1.5


@dataclass
class RankingConfig:
    """Edge ranking and scoring settings."""
    # Weight for each ranking factor
    weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_oos": 0.25,
        "stability": 0.20,
        "economic_intuition": 0.15,
        "statistical_significance": 0.15,
        "turnover_efficiency": 0.10,
        "simplicity": 0.10,
        "regime_robustness": 0.05
    })
    
    # Thresholds for acceptance
    min_overall_score: float = 0.5
    min_economic_score: float = 0.3


@dataclass 
class MLConfig:
    """Machine learning model settings."""
    # Model types to test
    model_types: List[str] = field(default_factory=lambda: ["lgbm", "xgboost", "linear", "neural"])
    
    # Feature importance
    shap_samples: int = 500
    min_feature_importance: float = 0.01
    
    # Cross-validation
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    
    # Neural network specific
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    nn_dropout: float = 0.2
    nn_epochs: int = 100


@dataclass
class ReportConfig:
    """Report generation settings."""
    output_dir: str = "alpha_reports"
    include_visualizations: bool = True
    include_detailed_stats: bool = True
    include_trade_log: bool = True
    formats: List[str] = field(default_factory=lambda: ["html", "json", "csv"])


@dataclass
class AlphaResearchConfig:
    """Master configuration for the Alpha Research Engine."""
    data: DataConfig = field(default_factory=DataConfig)
    hypothesis: HypothesisConfig = field(default_factory=HypothesisConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    falsification: FalsificationConfig = field(default_factory=FalsificationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # General settings
    random_seed: int = 42
    n_jobs: int = -1  # Use all cores
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "AlphaResearchConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AlphaResearchConfig":
        """Create config from dictionary."""
        config = cls()
        
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'hypothesis' in data:
            config.hypothesis = HypothesisConfig(**data['hypothesis'])
        if 'validation' in data:
            config.validation = ValidationConfig(**data['validation'])
        if 'falsification' in data:
            config.falsification = FalsificationConfig(**data['falsification'])
        if 'ranking' in data:
            config.ranking = RankingConfig(**data['ranking'])
        if 'ml' in data:
            config.ml = MLConfig(**data['ml'])
        if 'report' in data:
            config.report = ReportConfig(**data['report'])
            
        for key in ['random_seed', 'n_jobs', 'verbose']:
            if key in data:
                setattr(config, key, data[key])
                
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)

