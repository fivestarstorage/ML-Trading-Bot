"""
Command Line Interface for Alpha Research Engine

Usage:
    python -m alpha_research.cli discover BTCUSD --timeframe 5m
    python -m alpha_research.cli scan ETHUSD --start 2023-01-01 --end 2024-01-01
    python -m alpha_research.cli analyze --hypothesis-id abc123
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from .orchestrator import AlphaResearchOrchestrator, discover_alpha
from .config import AlphaResearchConfig
from .utils.logging import setup_logging, get_logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Alpha Research Engine - Discover Trading Edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full alpha discovery on Bitcoin
    python -m alpha_research.cli discover BTCUSD --timeframe 5m

    # Quick scan on Ethereum with date range
    python -m alpha_research.cli scan ETHUSD --start 2023-01-01 --end 2024-01-01

    # Use custom config
    python -m alpha_research.cli discover BTCUSD --config my_config.yml

    # Load from local CSV
    python -m alpha_research.cli discover XAUUSD --data-dir ./Data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Discover command (full pipeline)
    discover_parser = subparsers.add_parser(
        'discover', 
        help='Run full alpha discovery pipeline'
    )
    discover_parser.add_argument('symbol', type=str, help='Trading symbol')
    discover_parser.add_argument(
        '--timeframe', '-t', 
        type=str, 
        default='5m',
        help='Data timeframe (1m, 5m, 15m, 1h, etc.)'
    )
    discover_parser.add_argument(
        '--start', '-s',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    discover_parser.add_argument(
        '--end', '-e',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    discover_parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    discover_parser.add_argument(
        '--data-dir',
        type=str,
        default='Data',
        help='Directory containing data files'
    )
    discover_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='alpha_reports',
        help='Output directory for reports'
    )
    discover_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Quick scan command
    scan_parser = subparsers.add_parser(
        'scan',
        help='Quick alpha scan (no full validation)'
    )
    scan_parser.add_argument('symbol', type=str, help='Trading symbol')
    scan_parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default='5m',
        help='Data timeframe'
    )
    scan_parser.add_argument(
        '--start', '-s',
        type=str,
        default=None,
        help='Start date'
    )
    scan_parser.add_argument(
        '--end', '-e',
        type=str,
        default=None,
        help='End date'
    )
    scan_parser.add_argument(
        '--data-dir',
        type=str,
        default='Data',
        help='Directory containing data files'
    )
    scan_parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top edges to display'
    )
    
    # List hypotheses command
    list_parser = subparsers.add_parser(
        'list-hypotheses',
        help='List all hypothesis types'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger()
    
    if args.command == 'discover':
        run_discover(args)
    elif args.command == 'scan':
        run_scan(args)
    elif args.command == 'list-hypotheses':
        run_list_hypotheses()
    else:
        parser.print_help()


def run_discover(args):
    """Run full discovery pipeline."""
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("ALPHA RESEARCH ENGINE - FULL DISCOVERY")
    logger.info("="*60)
    
    # Build config
    config_dict = {
        'data': {
            'data_dir': args.data_dir,
            'enable_ccxt': True,
            'enable_alpaca': False,
        },
        'report': {
            'output_dir': args.output_dir,
            'formats': ['html', 'json', 'csv'],
        },
    }
    
    if args.config:
        config_dict = AlphaResearchConfig.from_yaml(args.config)
    
    # Run discovery
    orchestrator = AlphaResearchOrchestrator(config_dict=config_dict)
    
    try:
        report = orchestrator.run(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("DISCOVERY COMPLETE")
        print("="*60)
        print(f"Report ID: {report.report_id}")
        print(f"Hypotheses Generated: {report.total_hypotheses_generated}")
        print(f"Edges Discovered: {report.total_edges_discovered}")
        print(f"Passed Validation: {report.edges_passed_validation}")
        print(f"Passed Falsification: {report.edges_passed_falsification}")
        
        if report.top_edges:
            print(f"\nTop {len(report.top_edges)} Edges:")
            for edge_report in report.top_edges[:5]:
                edge = edge_report.edge
                print(f"  #{edge.rank}: {edge.edge_name}")
                print(f"      Score: {edge.overall_score:.3f} | Sharpe: {edge.oos_sharpe:.3f}")
                print(f"      Recommendation: {edge.recommendation[:80]}...")
        
        print(f"\nFull report saved to: {args.output_dir}/{report.report_id}/")
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise


def run_scan(args):
    """Run quick scan."""
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("ALPHA RESEARCH ENGINE - QUICK SCAN")
    logger.info("="*60)
    
    config_dict = {
        'data': {
            'data_dir': args.data_dir,
            'enable_ccxt': True,
        },
    }
    
    orchestrator = AlphaResearchOrchestrator(config_dict=config_dict)
    
    try:
        candidates = orchestrator.run_quick_scan(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
        )
        
        print(f"\nFound {len(candidates)} edge candidates")
        print("\nTop candidates:")
        print("-"*80)
        
        # Sort by OOS sharpe
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.oos_sharpe, 
            reverse=True
        )
        
        for i, candidate in enumerate(sorted_candidates[:args.top_n]):
            print(f"\n{i+1}. {candidate.hypothesis.name}")
            print(f"   Type: {candidate.hypothesis.hypothesis_type.value}")
            print(f"   Mechanism: {candidate.hypothesis.mechanism.value}")
            print(f"   OOS Sharpe: {candidate.oos_sharpe:.3f}")
            print(f"   Win Rate: {candidate.win_rate:.1%}")
            print(f"   Trades: {candidate.n_trades}")
            print(f"   p-value: {candidate.p_value:.4f}")
            if candidate.hypothesis.economic_rationale:
                print(f"   Rationale: {candidate.hypothesis.economic_rationale[:100]}...")
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise


def run_list_hypotheses():
    """List all available hypothesis types."""
    from .hypothesis.base import HypothesisType, HypothesisMechanism
    
    print("\nHYPOTHESIS TYPES")
    print("="*40)
    for ht in HypothesisType:
        print(f"  - {ht.value}")
    
    print("\nECONOMIC MECHANISMS")
    print("="*40)
    for hm in HypothesisMechanism:
        print(f"  - {hm.value}")
    
    print("\nHYPOTHESIS GENERATORS")
    print("="*40)
    generators = [
        ("Regime", "Volatility clusters, trend detection, regime transitions"),
        ("Microstructure", "Volume patterns, orderbook dynamics, liquidity"),
        ("Cross-Asset", "Basis trades, correlations, term structure"),
        ("Flow", "Funding rates, OI dynamics, money flow"),
        ("Pattern", "Clustering, motifs, anomaly detection"),
        ("Seasonal", "Time-of-day, day-of-week, monthly effects"),
    ]
    
    for name, description in generators:
        print(f"  {name}:")
        print(f"    {description}")


if __name__ == '__main__':
    main()

