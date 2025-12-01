"""
Alpha Research Orchestrator

Main entry point that coordinates all components of the
alpha research engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .config import AlphaResearchConfig
from .hypothesis.base import Hypothesis
from .discovery.engine import AlphaDiscoveryEngine, EdgeCandidate
from .discovery.ml_discovery import MLAlphaDiscovery
from .falsification.stress_test import StressTester, FalsificationResult
from .falsification.robustness import RobustnessAnalyzer
from .ranking.ranker import EdgeRanker, RankedEdge
from .reporting.generator import ReportGenerator, AlphaReport
from .data.adapters import UniversalAdapter
from .utils.logging import get_logger, setup_logging

logger = get_logger()


class AlphaResearchOrchestrator:
    """
    Main orchestrator for the Alpha Research Engine.
    
    Coordinates the full research pipeline:
    1. Data loading
    2. Hypothesis generation
    3. Alpha discovery
    4. Falsification testing
    5. Edge ranking
    6. Report generation
    """
    
    def __init__(self, config: AlphaResearchConfig = None, config_dict: Dict[str, Any] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: AlphaResearchConfig object
            config_dict: Alternative config as dictionary
        """
        if config is not None:
            self.config = config
            self._config_dict = self._config_to_dict(config)
        elif config_dict is not None:
            self._config_dict = config_dict
            self.config = None
        else:
            self.config = AlphaResearchConfig()
            self._config_dict = self._config_to_dict(self.config)
        
        # Initialize components
        self.data_adapter = UniversalAdapter(self._config_dict.get('data', {}))
        self.discovery_engine = AlphaDiscoveryEngine(self._config_dict)
        self.stress_tester = StressTester(self._config_dict)
        self.robustness_analyzer = RobustnessAnalyzer(self._config_dict)
        self.ranker = EdgeRanker(self._config_dict)
        self.report_generator = ReportGenerator(self._config_dict)
        
        # Results storage
        self.data: Optional[pd.DataFrame] = None
        self.hypotheses: List[Hypothesis] = []
        self.edge_candidates: List[EdgeCandidate] = []
        self.falsification_results: Dict[str, FalsificationResult] = {}
        self.ranked_edges: List[RankedEdge] = []
        self.report: Optional[AlphaReport] = None
    
    def _config_to_dict(self, config: AlphaResearchConfig) -> Dict[str, Any]:
        """Convert config dataclass to dictionary."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        return to_dict(config)
    
    def run(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m',
        data: Optional[pd.DataFrame] = None
    ) -> AlphaReport:
        """
        Run the complete alpha research pipeline.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            data: Optional pre-loaded DataFrame
            
        Returns:
            Complete AlphaReport
        """
        setup_logging()
        
        logger.info("="*70)
        logger.info("ALPHA RESEARCH ENGINE")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Date Range: {start_date or 'earliest'} to {end_date or 'latest'}")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading data...")
        if data is not None:
            self.data = data
            logger.info(f"Using provided data: {len(self.data)} rows")
        else:
            self.data = self.data_adapter.load(symbol, start_date, end_date, timeframe)
        
        logger.info(f"Data loaded: {len(self.data)} rows")
        logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        # Step 2: Alpha discovery
        logger.info("\n[STEP 2] Discovering alpha...")
        self.edge_candidates = self.discovery_engine.discover(self.data)
        
        logger.info(f"Discovered {len(self.edge_candidates)} edge candidates")
        
        if not self.edge_candidates:
            logger.warning("No edges discovered. Check data quality or relax validation thresholds.")
            return self._generate_empty_report(symbol, timeframe)
        
        # Step 3: Falsification testing
        logger.info("\n[STEP 3] Running falsification tests...")
        self._run_falsification()
        
        passed_falsification = sum(
            1 for r in self.falsification_results.values() 
            if r.passed_falsification
        )
        logger.info(f"{passed_falsification}/{len(self.falsification_results)} edges passed falsification")
        
        # Step 4: Ranking
        logger.info("\n[STEP 4] Ranking edges...")
        self.ranked_edges = self.ranker.rank_edges(
            self.edge_candidates, 
            self.falsification_results
        )
        
        top_edges = self.ranker.get_top_edges(self.ranked_edges, n=10)
        logger.info(f"Top {len(top_edges)} edges identified")
        
        # Step 5: Generate report
        logger.info("\n[STEP 5] Generating report...")
        discovery_metadata = {
            'data_source': 'universal',
            'symbol': symbol,
            'timeframe': timeframe,
            'total_hypotheses': len(self.discovery_engine.hypotheses),
            'edges_validated': len(self.edge_candidates),
        }
        
        self.report = self.report_generator.generate_report(
            self.ranked_edges,
            self.falsification_results,
            self.data,
            discovery_metadata
        )
        
        # Summary
        elapsed = datetime.now() - start_time
        logger.info("\n" + "="*70)
        logger.info("RESEARCH COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Hypotheses generated: {len(self.discovery_engine.hypotheses)}")
        logger.info(f"Edges discovered: {len(self.edge_candidates)}")
        logger.info(f"Edges passed falsification: {passed_falsification}")
        logger.info(f"Strong candidates: {sum(1 for e in self.ranked_edges if e.overall_score >= 0.75)}")
        
        if top_edges:
            logger.info("\nTop 5 Edges:")
            for edge in top_edges[:5]:
                logger.info(
                    f"  #{edge.rank}: {edge.edge_name} "
                    f"(Score: {edge.overall_score:.3f}, Sharpe: {edge.oos_sharpe:.3f})"
                )
        
        return self.report
    
    def _run_falsification(self):
        """Run falsification tests on all edge candidates."""
        self.falsification_results = {}
        
        for candidate in self.edge_candidates:
            try:
                signals = candidate.hypothesis.generate_signals(self.data)
                
                result = self.stress_tester.run_full_falsification(
                    self.data,
                    signals,
                    candidate.hypothesis.name,
                    candidate.hypothesis.id
                )
                
                self.falsification_results[candidate.hypothesis.id] = result
                
            except Exception as e:
                logger.warning(f"Falsification failed for {candidate.hypothesis.name}: {e}")
    
    def _generate_empty_report(self, symbol: str, timeframe: str) -> AlphaReport:
        """Generate an empty report when no edges are found."""
        return AlphaReport(
            report_id=datetime.now().strftime('%Y%m%d_%H%M%S'),
            generated_at=datetime.now(),
            data_source='universal',
            symbol=symbol,
            timeframe=timeframe,
            date_range=(None, None),
            total_hypotheses_generated=len(self.discovery_engine.hypotheses),
            total_edges_discovered=0,
            edges_passed_validation=0,
            edges_passed_falsification=0,
            top_edges=[],
            all_ranked_edges=[],
            discovery_stats={},
        )
    
    def run_quick_scan(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '5m',
        data: Optional[pd.DataFrame] = None
    ) -> List[EdgeCandidate]:
        """
        Run a quick scan without full falsification (for exploration).
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            data: Optional pre-loaded data
            
        Returns:
            List of edge candidates (without full validation)
        """
        logger.info("Running quick alpha scan...")
        
        if data is not None:
            self.data = data
        else:
            self.data = self.data_adapter.load(symbol, start_date, end_date, timeframe)
        
        self.edge_candidates = self.discovery_engine.discover(self.data)
        
        return self.edge_candidates
    
    def analyze_single_hypothesis(
        self,
        hypothesis: Hypothesis,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single hypothesis in detail.
        
        Args:
            hypothesis: Hypothesis to analyze
            data: Optional data (uses stored data if not provided)
            
        Returns:
            Detailed analysis results
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available. Load data first.")
            data = self.data
        
        # Generate signals
        signals = hypothesis.generate_signals(data)
        
        # Run stress tests
        falsification = self.stress_tester.run_full_falsification(
            data, signals, hypothesis.name, hypothesis.id
        )
        
        # Robustness analysis
        robustness = self.robustness_analyzer.analyze(data, signals)
        
        # Calculate returns
        forward_returns = data['close'].pct_change().shift(-1)
        strategy_returns = signals * forward_returns
        
        # Performance metrics
        from .utils.statistics import (
            calculate_sharpe_ratio, 
            calculate_max_drawdown,
            bootstrap_confidence_interval
        )
        
        returns = strategy_returns.dropna().values
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
        max_dd, _, _ = calculate_max_drawdown((1 + strategy_returns.dropna()).cumprod().values)
        _, ci_lower, ci_upper = bootstrap_confidence_interval(
            returns, 
            lambda x: calculate_sharpe_ratio(x, periods_per_year=len(x))
        )
        
        return {
            'hypothesis': hypothesis.to_dict(),
            'performance': {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_return': float(strategy_returns.sum()),
                'win_rate': float((strategy_returns > 0).sum() / (strategy_returns != 0).sum()),
                'n_trades': int((signals.diff().abs() != 0).sum() / 2),
            },
            'statistical': {
                'bootstrap_ci': (ci_lower, ci_upper),
            },
            'falsification': {
                'passed': falsification.passed_falsification,
                'score': falsification.falsification_score,
                'vulnerabilities': falsification.vulnerabilities,
            },
            'robustness': {
                'sharpe_stability': robustness.sharpe_stability,
                'regime_consistency': robustness.regime_consistency,
                'robustness_score': robustness.robustness_score,
            },
        }
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a summary DataFrame of all ranked edges."""
        if not self.ranked_edges:
            return pd.DataFrame()
        
        return self.ranker.to_dataframe(self.ranked_edges)


# Convenience function for quick usage
def discover_alpha(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = '5m',
    config: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None
) -> AlphaReport:
    """
    Convenience function to run alpha discovery.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        timeframe: Data timeframe
        config: Optional configuration dictionary
        data: Optional pre-loaded data
        
    Returns:
        AlphaReport with discovered edges
    """
    orchestrator = AlphaResearchOrchestrator(config_dict=config or {})
    return orchestrator.run(symbol, start_date, end_date, timeframe, data)

