#!/usr/bin/env python3
"""
Run Alpha Discovery

Alpha Research Engine for discovering trading edges in financial data.

Usage:
    python scripts/run_alpha_discovery.py [OPTIONS]

Options:
    --data FILE         Data file to use (default: auto-detect)
    --symbol SYMBOL     Trading symbol (default: from filename)
    --rows N            Max rows to process (default: 50000)
    --strict            Use strict validation (default: relaxed demo mode)
    --funding FILE      Optional: funding rate data file
    --oi FILE           Optional: open interest data file

Examples:
    python scripts/run_alpha_discovery.py
    python scripts/run_alpha_discovery.py --data Data/BTCUSD_5m_ccxt.parquet --rows 100000
    python scripts/run_alpha_discovery.py --strict
"""

import sys
import os
import warnings
import argparse
from pathlib import Path

# Suppress pandas deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(data_file: Path) -> pd.DataFrame:
    """Load and standardize data from file."""
    # Load based on extension
    if data_file.suffix == '.parquet':
        data = pd.read_parquet(data_file)
    else:
        # Try to detect delimiter
        with open(data_file, 'r') as f:
            first_line = f.readline()
        
        if ';' in first_line:
            data = pd.read_csv(data_file, sep=';')
        else:
            data = pd.read_csv(data_file)
    
    # Standardize column names
    data.columns = data.columns.str.lower().str.strip()
    
    # Handle index
    if 'timestamp' in data.columns:
        data.index = pd.to_datetime(data['timestamp'])
        data = data.drop(columns=['timestamp'])
    elif 'date' in data.columns:
        data.index = pd.to_datetime(data['date'])
        data = data.drop(columns=['date'])
    elif 'time' in data.columns:
        data.index = pd.to_datetime(data['time'])
        data = data.drop(columns=['time'])
    elif not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except:
            print("Warning: Could not parse index as datetime")
    
    # Ensure UTC timezone
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    
    # Calculate returns if not present
    if 'returns' not in data.columns and 'close' in data.columns:
        data['returns'] = data['close'].pct_change()
    
    return data


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add additional technical indicators to data."""
    df = data.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Volume metrics
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
    
    # Momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Trend
    df['above_ema50'] = (df['close'] > df['ema_50']).astype(int)
    
    # VWAP (rolling approximation)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(48).sum() / df['volume'].rolling(48).sum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['atr'].replace(0, np.nan)
    
    # Candle patterns
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'].abs() / (df['high'] - df['low']).replace(0, np.nan)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    return df


def main():
    """Run alpha discovery."""
    parser = argparse.ArgumentParser(description='Alpha Research Engine')
    parser.add_argument('--data', type=str, help='Data file to use')
    parser.add_argument('--symbol', type=str, help='Trading symbol')
    parser.add_argument('--rows', type=int, default=50000, help='Max rows to process')
    parser.add_argument('--strict', action='store_true', help='Use strict validation')
    parser.add_argument('--funding', type=str, help='Funding rate data file')
    parser.add_argument('--oi', type=str, help='Open interest data file')
    parser.add_argument('--add-indicators', action='store_true', default=True,
                       help='Add technical indicators to data')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ALPHA RESEARCH ENGINE")
    print("="*70)
    
    # Import after path setup
    from alpha_research import AlphaResearchOrchestrator
    from alpha_research.config import AlphaResearchConfig
    
    # Find data
    data_dir = project_root / "Data"
    
    if args.data:
        data_file = Path(args.data)
        if not data_file.exists():
            data_file = data_dir / args.data
    else:
        # Auto-detect data file
        available_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
        
        if not available_files:
            print(f"\nNo data files found in {data_dir}")
            print("Please add OHLCV data files to the Data directory.")
            print("\nExpected format: CSV or Parquet with columns:")
            print("  Required: open, high, low, close, volume")
            print("  Optional: funding_rate, open_interest, liquidations")
            return 1
        
        print(f"\nFound {len(available_files)} data files:")
        for f in available_files[:10]:
            print(f"  - {f.name}")
        
        # Prefer parquet files (faster)
        parquet_files = [f for f in available_files if f.suffix == '.parquet']
        data_file = parquet_files[0] if parquet_files else available_files[0]
    
    symbol = args.symbol or data_file.stem.split('_')[0]
    
    print(f"\nUsing: {data_file.name}")
    print(f"Symbol: {symbol}")
    
    # Load main data
    data = load_data(data_file)
    
    # Add optional data sources
    if args.funding:
        funding_file = Path(args.funding)
        if funding_file.exists():
            funding_data = load_data(funding_file)
            if 'funding_rate' in funding_data.columns:
                data = data.join(funding_data[['funding_rate']], how='left')
                print(f"Added funding rate data from {args.funding}")
    
    if args.oi:
        oi_file = Path(args.oi)
        if oi_file.exists():
            oi_data = load_data(oi_file)
            if 'open_interest' in oi_data.columns:
                data = data.join(oi_data[['open_interest']], how='left')
                print(f"Added open interest data from {args.oi}")
    
    # Add technical indicators
    if args.add_indicators:
        print("Adding technical indicators...")
        data = add_technical_indicators(data)
    
    # Limit data size
    if len(data) > args.rows:
        print(f"Limiting data from {len(data)} to last {args.rows} rows")
        data = data.tail(args.rows)
    
    print(f"\nData summary:")
    print(f"  Rows: {len(data)}")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Columns: {len(data.columns)}")
    
    # Show available features
    feature_categories = {
        'Core OHLCV': ['open', 'high', 'low', 'close', 'volume'],
        'Derivatives': ['funding_rate', 'open_interest', 'liquidations', 'long_liquidations', 'short_liquidations'],
        'Technical': ['rsi', 'macd', 'bb_pct', 'atr', 'momentum_5', 'momentum_20', 'vwap'],
        'Volume': ['volume_ratio', 'volume_ma'],
        'EMAs': ['ema_9', 'ema_21', 'ema_50'],
    }
    
    print(f"\nAvailable features:")
    for cat, features in feature_categories.items():
        available = [f for f in features if f in data.columns]
        if available:
            print(f"  {cat}: {', '.join(available)}")
    
    # Configure discovery
    if args.strict:
        # Strict validation for production use
        config = {
            'data': {'data_dir': str(data_dir)},
            'validation': {
                'min_sharpe_ratio': 0.5,
                'min_win_rate': 0.50,
                'min_profit_factor': 1.2,
                'min_trades': 30,
                'bootstrap_iterations': 1000,
                'monte_carlo_simulations': 1000,
                'significance_level': 0.05,
            },
            'falsification': {
                'noise_iterations': 50,
                'entry_noise_iterations': 100,
                'slippage_multipliers': [1.0, 1.5, 2.0, 3.0],
            },
            'report': {'output_dir': str(project_root / 'alpha_reports')},
        }
        print("\nUsing STRICT validation mode")
    else:
        # Relaxed for demo/exploration
        config = {
            'data': {'data_dir': str(data_dir)},
            'validation': {
                'min_sharpe_ratio': 0.1,
                'min_win_rate': 0.40,
                'min_profit_factor': 1.0,
                'min_trades': 5,
                'bootstrap_iterations': 100,
                'monte_carlo_simulations': 100,
                'significance_level': 0.20,
            },
            'falsification': {
                'noise_iterations': 10,
                'entry_noise_iterations': 20,
                'slippage_multipliers': [1.0, 2.0],
            },
            'report': {'output_dir': str(project_root / 'alpha_reports')},
        }
        print("\nUsing RELAXED demo mode (use --strict for production)")
    
    # Initialize orchestrator
    orchestrator = AlphaResearchOrchestrator(config_dict=config)
    
    # Run discovery
    print("\nStarting alpha discovery...")
    print("(This may take a few minutes depending on data size)\n")
    
    start_time = datetime.now()
    
    try:
        report = orchestrator.run(
            symbol=symbol,
            timeframe='5m',
            data=data
        )
        
        elapsed = datetime.now() - start_time
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Total time: {elapsed}")
        print(f"Report ID: {report.report_id}")
        print(f"\nPipeline statistics:")
        print(f"  Hypotheses generated: {report.total_hypotheses_generated}")
        print(f"  Edges discovered: {report.total_edges_discovered}")
        print(f"  Passed validation: {report.edges_passed_validation}")
        print(f"  Passed falsification: {report.edges_passed_falsification}")
        
        if report.top_edges:
            print(f"\n" + "-"*70)
            print("TOP EDGE CANDIDATES")
            print("-"*70)
            
            for i, edge_report in enumerate(report.top_edges[:5]):
                edge = edge_report.edge
                print(f"\n#{edge.rank}: {edge.edge_name}")
                print(f"   Type: {edge.hypothesis_type}")
                print(f"   Mechanism: {edge.mechanism}")
                print(f"   Overall Score: {edge.overall_score:.3f}")
                print(f"   OOS Sharpe: {edge.oos_sharpe:.3f}")
                print(f"   Win Rate: {edge.win_rate:.1%}")
                print(f"   Trades: {edge.n_trades}")
                print(f"   p-value: {edge.p_value:.4f}")
                
                if edge.strengths:
                    print(f"   Strengths: {', '.join(edge.strengths[:3])}")
                if edge.weaknesses:
                    print(f"   Weaknesses: {', '.join(edge.weaknesses[:2])}")
                
                rec = edge.recommendation[:100] + "..." if len(edge.recommendation) > 100 else edge.recommendation
                print(f"   Recommendation: {rec}")
        else:
            print("\nNo strong edges discovered in current data.")
            print("\nSuggestions:")
            print("  1. Use more historical data (--rows 100000)")
            print("  2. Add derivatives data (funding rate, OI)")
            print("  3. Try different assets or timeframes")
            print("  4. Use relaxed mode for exploration")
        
        # Save summary
        summary_df = orchestrator.get_summary_dataframe()
        if not summary_df.empty:
            summary_path = project_root / 'alpha_reports' / f'{report.report_id}_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to: {summary_path}")
        
        print(f"\nFull report saved to: {project_root / 'alpha_reports' / report.report_id}")
        
    except Exception as e:
        print(f"\nError during discovery: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
