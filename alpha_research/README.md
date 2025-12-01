# Alpha Research Engine

A professional-grade quant research system for discovering statistically valid, repeatable, and economically explainable trading edges.

## Overview

The Alpha Research Engine is designed to automatically discover hidden structure in financial markets through:

1. **Hypothesis Generation** - Automatically proposes potential edges based on:
   - Regime changes and volatility clusters
   - Microstructure patterns (orderbook imbalance, liquidity shifts)
   - Cross-asset relationships (spot vs perp basis, term structure)
   - Flow asymmetries (funding rates, open interest)
   - Pattern discovery via embeddings and clustering
   - Seasonal and time-based effects

2. **Alpha Discovery** - Tests hypotheses using:
   - Walk-forward validation with realistic costs
   - Bootstrap confidence intervals
   - Monte Carlo permutation tests
   - Multiple testing correction

3. **Falsification** - Destroys weak edges through:
   - Slippage stress testing
   - Entry timing randomization
   - Label shifting tests
   - Feature noise injection
   - Rolling window stability checks
   - Regime robustness analysis

4. **Ranking** - Ranks edges by:
   - Out-of-sample Sharpe (after costs)
   - Statistical significance
   - Economic intuition
   - Stability across regimes
   - Turnover efficiency
   - Simplicity

5. **Reporting** - Generates comprehensive reports including:
   - Executive summary
   - Individual edge analysis
   - Performance metrics
   - Robustness metrics
   - Recommendations

## Installation

The alpha research engine is included in the ML-Trading-Bot project. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Additional optional dependencies for enhanced functionality:

```bash
# For crypto data via CCXT
pip install ccxt

# For SHAP explanations
pip install shap

# For stationarity tests
pip install statsmodels
```

## Quick Start

### Using Python API

```python
from alpha_research import AlphaResearchOrchestrator

# Initialize orchestrator
orchestrator = AlphaResearchOrchestrator()

# Run full discovery
report = orchestrator.run(
    symbol='BTCUSD',
    start_date='2022-01-01',
    end_date='2024-01-01',
    timeframe='5m'
)

# View top edges
for edge in report.top_edges[:5]:
    print(f"{edge.edge.edge_name}: Sharpe {edge.edge.oos_sharpe:.3f}")

# Get summary DataFrame
df = orchestrator.get_summary_dataframe()
print(df.head(10))
```

### Using CLI

```bash
# Full discovery
python -m alpha_research.cli discover BTCUSD --timeframe 5m

# Quick scan
python -m alpha_research.cli scan ETHUSD --start 2023-01-01 --end 2024-01-01

# With custom config
python -m alpha_research.cli discover BTCUSD --config alpha_config.yml
```

### Using Pre-loaded Data

```python
import pandas as pd
from alpha_research import AlphaResearchOrchestrator

# Load your own data
data = pd.read_csv('my_data.csv', index_col=0, parse_dates=True)

# Run discovery on pre-loaded data
orchestrator = AlphaResearchOrchestrator()
report = orchestrator.run(
    symbol='CUSTOM',
    timeframe='5m',
    data=data  # Use pre-loaded data
)
```

## Configuration

Create a custom configuration file `alpha_config.yml`:

```yaml
# Data settings
data:
  source: csv
  data_dir: Data
  symbol: BTCUSD
  timeframe: 5m

# Validation thresholds
validation:
  min_sharpe_ratio: 0.5
  min_win_rate: 0.45
  min_trades: 30
  commission_bps: 4.0
  slippage_bps: 2.0

# Falsification settings
falsification:
  slippage_multipliers: [1.0, 1.5, 2.0, 3.0]
  noise_levels: [0.01, 0.02, 0.05]

# Report output
report:
  output_dir: alpha_reports
  formats: [html, json, csv]
```

## Architecture

```
alpha_research/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── orchestrator.py      # Main orchestration
├── cli.py               # Command-line interface
│
├── hypothesis/          # Hypothesis generation
│   ├── base.py          # Base classes
│   ├── regime.py        # Regime-based hypotheses
│   ├── microstructure.py # Market microstructure
│   ├── cross_asset.py   # Cross-asset relationships
│   ├── flow.py          # Flow analysis
│   ├── pattern.py       # Pattern discovery
│   └── seasonal.py      # Seasonal effects
│
├── discovery/           # Alpha discovery engine
│   ├── engine.py        # Main discovery logic
│   ├── validator.py     # Statistical validation
│   └── ml_discovery.py  # ML-based discovery
│
├── falsification/       # Stress testing
│   ├── stress_test.py   # Stress tests
│   └── robustness.py    # Robustness analysis
│
├── ranking/             # Edge ranking
│   └── ranker.py        # Multi-factor ranking
│
├── reporting/           # Report generation
│   └── generator.py     # Report generator
│
├── data/                # Data adapters
│   └── adapters.py      # Multi-source adapters
│
└── utils/               # Utilities
    ├── logging.py       # Logging setup
    ├── statistics.py    # Statistical functions
    └── data_utils.py    # Data utilities
```

## Hypothesis Types

### Regime-Based
- Volatility regime detection
- Trend/mean-reversion regime
- Regime transition signals
- Hidden Markov Model regimes

### Microstructure
- Volume accumulation patterns
- Volume climax reversals
- Price rejection patterns
- Orderbook imbalance (if data available)

### Cross-Asset
- Spot-perpetual basis
- Multi-timeframe alignment
- Correlation regime shifts

### Flow
- Funding rate extremes
- Open interest dynamics
- Money flow indicators

### Pattern Discovery
- Return clustering
- Candlestick pattern clusters
- Anomaly-based signals

### Seasonal
- Hour-of-day effects
- Day-of-week patterns
- Turn-of-month effect
- Session-based patterns

## Best Practices

1. **Data Quality**: Ensure clean, gap-free data before running discovery
2. **Sufficient History**: Use at least 1-2 years of data for reliable results
3. **Realistic Costs**: Set transaction costs based on your actual execution quality
4. **Multiple Testing**: Be aware of data snooping - the engine applies corrections
5. **Economic Sense**: Prioritize edges with clear economic rationale
6. **Out-of-Sample**: Always validate on truly unseen data before trading

## Output Files

After running discovery, reports are saved to `alpha_reports/<report_id>/`:

- `report.html` - Interactive HTML report
- `report.json` - Machine-readable JSON
- `ranked_edges.csv` - CSV of all ranked edges

## Contributing

To add a new hypothesis generator:

1. Create a new file in `alpha_research/hypothesis/`
2. Inherit from `HypothesisGenerator`
3. Implement `generate()` and `get_hypothesis_type()`
4. Register in `__init__.py`

## License

MIT License - See LICENSE file for details.

