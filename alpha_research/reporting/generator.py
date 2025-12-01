"""
Report Generator

Generates comprehensive research reports summarizing
discovered edges, their mechanisms, and validation results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from ..ranking.ranker import RankedEdge
from ..falsification.stress_test import FalsificationResult
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class EdgeReport:
    """Report for a single edge."""
    edge: RankedEdge
    falsification: Optional[FalsificationResult]
    
    # Performance summary
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Equity curve data
    equity_curve: Optional[pd.Series] = None
    
    # Detailed analysis
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlphaReport:
    """Complete alpha research report."""
    
    # Metadata
    report_id: str
    generated_at: datetime
    data_source: str
    symbol: str
    timeframe: str
    date_range: tuple
    
    # Summary statistics
    total_hypotheses_generated: int
    total_edges_discovered: int
    edges_passed_validation: int
    edges_passed_falsification: int
    
    # Top edges
    top_edges: List[EdgeReport]
    
    # All ranked edges
    all_ranked_edges: List[RankedEdge]
    
    # Discovery statistics
    discovery_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'data_source': self.data_source,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'date_range': [str(d) for d in self.date_range] if self.date_range else None,
            'total_hypotheses_generated': self.total_hypotheses_generated,
            'total_edges_discovered': self.total_edges_discovered,
            'edges_passed_validation': self.edges_passed_validation,
            'edges_passed_falsification': self.edges_passed_falsification,
            'top_edges': [
                {
                    'edge': e.edge.to_dict(),
                    'performance_summary': e.performance_summary,
                    'analysis': e.analysis,
                }
                for e in self.top_edges
            ],
            'discovery_stats': self.discovery_stats,
        }


class ReportGenerator:
    """
    Generates comprehensive research reports.
    
    Reports include:
    1. Executive summary
    2. Individual edge analysis
    3. Performance metrics
    4. Robustness analysis
    5. Recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        report_config = config.get('report', {})
        
        self.output_dir = Path(report_config.get('output_dir', 'alpha_reports'))
        self.include_visualizations = report_config.get('include_visualizations', True)
        self.include_detailed_stats = report_config.get('include_detailed_stats', True)
        self.include_trade_log = report_config.get('include_trade_log', True)
        self.formats = report_config.get('formats', ['html', 'json', 'csv'])
    
    def generate_report(
        self,
        ranked_edges: List[RankedEdge],
        falsification_results: Dict[str, FalsificationResult],
        data: pd.DataFrame,
        discovery_metadata: Dict[str, Any]
    ) -> AlphaReport:
        """
        Generate complete alpha research report.
        
        Args:
            ranked_edges: List of ranked edges
            falsification_results: Falsification results by edge ID
            data: Original data used for analysis
            discovery_metadata: Metadata from discovery process
            
        Returns:
            Complete AlphaReport
        """
        logger.info("Generating alpha research report...")
        
        # Generate report ID
        report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        report_dir = self.output_dir / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Build individual edge reports for top edges
        top_n = min(10, len(ranked_edges))
        top_edges = []
        
        for edge in ranked_edges[:top_n]:
            falsif = falsification_results.get(edge.edge_id)
            edge_report = self._build_edge_report(edge, falsif, data)
            top_edges.append(edge_report)
        
        # Calculate summary statistics
        passed_falsification = sum(
            1 for e in ranked_edges 
            if e.edge_id in falsification_results and 
            falsification_results[e.edge_id].passed_falsification
        )
        
        # Determine date range from data
        date_range = (data.index.min(), data.index.max()) if len(data) > 0 else (None, None)
        
        # Build discovery statistics
        discovery_stats = self._build_discovery_stats(ranked_edges, falsification_results)
        
        report = AlphaReport(
            report_id=report_id,
            generated_at=datetime.now(),
            data_source=discovery_metadata.get('data_source', 'unknown'),
            symbol=discovery_metadata.get('symbol', 'unknown'),
            timeframe=discovery_metadata.get('timeframe', 'unknown'),
            date_range=date_range,
            total_hypotheses_generated=discovery_metadata.get('total_hypotheses', 0),
            total_edges_discovered=len(ranked_edges),
            edges_passed_validation=discovery_metadata.get('edges_validated', 0),
            edges_passed_falsification=passed_falsification,
            top_edges=top_edges,
            all_ranked_edges=ranked_edges,
            discovery_stats=discovery_stats,
        )
        
        # Save report in requested formats
        self._save_report(report, report_dir)
        
        logger.info(f"Report generated: {report_dir}")
        
        return report
    
    def _build_edge_report(
        self,
        edge: RankedEdge,
        falsification: Optional[FalsificationResult],
        data: pd.DataFrame
    ) -> EdgeReport:
        """Build detailed report for a single edge."""
        
        performance_summary = {
            'oos_sharpe': edge.oos_sharpe,
            'net_sharpe': edge.net_sharpe,
            'win_rate': edge.win_rate,
            'profit_factor': edge.profit_factor,
            'max_drawdown': edge.max_drawdown,
            'n_trades': edge.n_trades,
            'total_return': edge.oos_sharpe * 0.1,  # Approximate
        }
        
        analysis = {
            'mechanism': edge.mechanism,
            'economic_rationale': edge.economic_rationale,
            'strengths': edge.strengths,
            'weaknesses': edge.weaknesses,
            'recommendation': edge.recommendation,
            'scores': {
                'sharpe': edge.score_sharpe,
                'stability': edge.score_stability,
                'economic': edge.score_economic,
                'statistical': edge.score_statistical,
                'turnover': edge.score_turnover,
                'simplicity': edge.score_simplicity,
                'regime': edge.score_regime,
            },
            'statistical_tests': {
                'p_value': edge.p_value,
                't_statistic': edge.t_statistic,
                'bootstrap_ci': edge.bootstrap_ci,
            },
        }
        
        if falsification:
            analysis['falsification'] = {
                'passed': falsification.passed_falsification,
                'score': falsification.falsification_score,
                'tests_passed': falsification.tests_passed,
                'tests_failed': falsification.tests_failed,
                'vulnerabilities': falsification.vulnerabilities,
                'test_results': [
                    {
                        'name': r.test_name,
                        'passed': r.passed,
                        'degradation': r.degradation_pct,
                    }
                    for r in falsification.test_results
                ],
            }
        
        return EdgeReport(
            edge=edge,
            falsification=falsification,
            performance_summary=performance_summary,
            analysis=analysis,
        )
    
    def _build_discovery_stats(
        self,
        ranked_edges: List[RankedEdge],
        falsification_results: Dict[str, FalsificationResult]
    ) -> Dict[str, Any]:
        """Build discovery statistics summary."""
        
        if not ranked_edges:
            return {}
        
        # Score distributions
        scores = [e.overall_score for e in ranked_edges]
        sharpes = [e.oos_sharpe for e in ranked_edges]
        
        # Type breakdown
        type_counts = {}
        mechanism_counts = {}
        
        for edge in ranked_edges:
            type_counts[edge.hypothesis_type] = type_counts.get(edge.hypothesis_type, 0) + 1
            mechanism_counts[edge.mechanism] = mechanism_counts.get(edge.mechanism, 0) + 1
        
        # Falsification summary
        falsif_passed = sum(
            1 for r in falsification_results.values() 
            if r.passed_falsification
        )
        
        return {
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
            },
            'sharpe_distribution': {
                'mean': float(np.mean(sharpes)),
                'std': float(np.std(sharpes)),
                'min': float(np.min(sharpes)),
                'max': float(np.max(sharpes)),
                'median': float(np.median(sharpes)),
            },
            'type_breakdown': type_counts,
            'mechanism_breakdown': mechanism_counts,
            'falsification_rate': falsif_passed / len(falsification_results) if falsification_results else 0,
            'n_strong_candidates': sum(1 for e in ranked_edges if e.overall_score >= 0.75),
            'n_promising_candidates': sum(1 for e in ranked_edges if 0.6 <= e.overall_score < 0.75),
        }
    
    def _save_report(self, report: AlphaReport, output_dir: Path):
        """Save report in requested formats."""
        
        if 'json' in self.formats:
            json_path = output_dir / 'report.json'
            with open(json_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved JSON report: {json_path}")
        
        if 'csv' in self.formats:
            # Save ranked edges as CSV
            csv_path = output_dir / 'ranked_edges.csv'
            df = pd.DataFrame([e.to_dict() for e in report.all_ranked_edges])
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV report: {csv_path}")
        
        if 'html' in self.formats:
            html_path = output_dir / 'report.html'
            self._generate_html_report(report, html_path)
            logger.info(f"Saved HTML report: {html_path}")
    
    def _generate_html_report(self, report: AlphaReport, output_path: Path):
        """Generate HTML report."""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Alpha Research Report - {report_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 40px; }}
        h3 {{ color: #4b5563; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8fafc; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #6b7280; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f8fafc; font-weight: 600; }}
        tr:hover {{ background: #f8fafc; }}
        .score {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 500; }}
        .score-high {{ background: #dcfce7; color: #166534; }}
        .score-medium {{ background: #fef3c7; color: #92400e; }}
        .score-low {{ background: #fee2e2; color: #991b1b; }}
        .edge-card {{ background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .strengths {{ color: #166534; }}
        .weaknesses {{ color: #991b1b; }}
        .recommendation {{ background: #e0e7ff; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Alpha Research Report</h1>
        <p><strong>Report ID:</strong> {report_id} | <strong>Generated:</strong> {generated_at}</p>
        <p><strong>Symbol:</strong> {symbol} | <strong>Timeframe:</strong> {timeframe} | <strong>Data Range:</strong> {date_range}</p>
        
        <h2>Executive Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{total_hypotheses}</div>
                <div class="stat-label">Hypotheses Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_discovered}</div>
                <div class="stat-label">Edges Discovered</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{passed_validation}</div>
                <div class="stat-label">Passed Validation</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{passed_falsification}</div>
                <div class="stat-label">Passed Falsification</div>
            </div>
        </div>
        
        <h2>Top Ranked Edges</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Edge Name</th>
                    <th>Type</th>
                    <th>Overall Score</th>
                    <th>OOS Sharpe</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                    <th>p-value</th>
                </tr>
            </thead>
            <tbody>
                {edge_rows}
            </tbody>
        </table>
        
        <h2>Detailed Edge Analysis</h2>
        {edge_details}
        
        <h2>Discovery Statistics</h2>
        <h3>Score Distribution</h3>
        <p>Mean: {score_mean:.3f} | Std: {score_std:.3f} | Min: {score_min:.3f} | Max: {score_max:.3f}</p>
        
        <h3>Hypothesis Type Breakdown</h3>
        <ul>{type_breakdown}</ul>
        
        <h3>Mechanism Breakdown</h3>
        <ul>{mechanism_breakdown}</ul>
        
        <footer style="margin-top: 40px; color: #6b7280; font-size: 14px;">
            Generated by Alpha Research Engine
        </footer>
    </div>
</body>
</html>
"""
        
        # Build edge rows
        edge_rows = ""
        for edge in report.all_ranked_edges[:20]:
            score_class = "score-high" if edge.overall_score >= 0.7 else ("score-medium" if edge.overall_score >= 0.5 else "score-low")
            edge_rows += f"""
            <tr>
                <td>{edge.rank}</td>
                <td>{edge.edge_name}</td>
                <td>{edge.hypothesis_type}</td>
                <td><span class="score {score_class}">{edge.overall_score:.3f}</span></td>
                <td>{edge.oos_sharpe:.3f}</td>
                <td>{edge.win_rate:.1%}</td>
                <td>{edge.n_trades}</td>
                <td>{edge.p_value:.4f}</td>
            </tr>
            """
        
        # Build edge details
        edge_details = ""
        for i, edge_report in enumerate(report.top_edges[:5]):
            edge = edge_report.edge
            strengths = "".join(f"<li>{s}</li>" for s in edge.strengths)
            weaknesses = "".join(f"<li>{w}</li>" for w in edge.weaknesses)
            
            edge_details += f"""
            <div class="edge-card">
                <h3>#{edge.rank}: {edge.edge_name}</h3>
                <p><strong>Type:</strong> {edge.hypothesis_type} | <strong>Mechanism:</strong> {edge.mechanism}</p>
                <p><strong>Economic Rationale:</strong> {edge.economic_rationale or 'Not specified'}</p>
                
                <p><strong>Performance:</strong> OOS Sharpe {edge.oos_sharpe:.3f} | Win Rate {edge.win_rate:.1%} | 
                   Profit Factor {edge.profit_factor:.2f} | Max DD {edge.max_drawdown:.1%}</p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div class="strengths">
                        <strong>Strengths:</strong>
                        <ul>{strengths if strengths else '<li>None identified</li>'}</ul>
                    </div>
                    <div class="weaknesses">
                        <strong>Weaknesses:</strong>
                        <ul>{weaknesses if weaknesses else '<li>None identified</li>'}</ul>
                    </div>
                </div>
                
                <div class="recommendation">
                    <strong>Recommendation:</strong> {edge.recommendation}
                </div>
            </div>
            """
        
        # Build breakdowns
        type_breakdown = "".join(
            f"<li>{k}: {v}</li>" 
            for k, v in report.discovery_stats.get('type_breakdown', {}).items()
        )
        mechanism_breakdown = "".join(
            f"<li>{k}: {v}</li>" 
            for k, v in report.discovery_stats.get('mechanism_breakdown', {}).items()
        )
        
        score_dist = report.discovery_stats.get('score_distribution', {})
        
        html = html_template.format(
            report_id=report.report_id,
            generated_at=report.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            symbol=report.symbol,
            timeframe=report.timeframe,
            date_range=f"{report.date_range[0]} to {report.date_range[1]}" if report.date_range else "N/A",
            total_hypotheses=report.total_hypotheses_generated,
            total_discovered=report.total_edges_discovered,
            passed_validation=report.edges_passed_validation,
            passed_falsification=report.edges_passed_falsification,
            edge_rows=edge_rows,
            edge_details=edge_details,
            score_mean=score_dist.get('mean', 0),
            score_std=score_dist.get('std', 0),
            score_min=score_dist.get('min', 0),
            score_max=score_dist.get('max', 0),
            type_breakdown=type_breakdown or '<li>None</li>',
            mechanism_breakdown=mechanism_breakdown or '<li>None</li>',
        )
        
        with open(output_path, 'w') as f:
            f.write(html)

