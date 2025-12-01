#!/usr/bin/env python3
"""
Enhanced Walk-Forward Analysis with Comprehensive Reporting and Graphs
Generates detailed reports with equity curves, trade analysis, and performance metrics
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.wfa import WalkForwardAnalyzer
from src.backtester import Backtester
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class WFAWithReports:
    def __init__(self, ticker, strategy, config_path):
        self.ticker = ticker
        self.strategy = strategy
        self.config = load_config(config_path)

        # Update config with ticker
        self.config['data']['symbol'] = ticker

        self.logger = setup_logging()
        self.timeframe = self.config['data']['timeframe_base']

        # Create reports directory
        self.reports_dir = Path('/Users/rileymartin/ML-Trading-Bot/reports')
        self.reports_dir.mkdir(exist_ok=True)

    def run_wfa_for_years(self, years_list):
        """Run WFA for multiple years and generate comprehensive report"""
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ANALYSIS: {self.ticker} | Strategy: {self.strategy.upper()}")
        print(f"Years: {', '.join(map(str, years_list))}")
        print(f"{'='*80}\n")

        all_results = []
        all_trades = []

        for year in years_list:
            print(f"\n{'─'*80}")
            print(f"Processing Year {year}...")
            print(f"{'─'*80}")

            result = self.run_wfa_for_year(year)
            if result:
                all_results.append(result)
                if 'trades' in result and result['trades'] is not None:
                    all_trades.extend(result['trades'])

        if not all_results:
            print("\n❌ No results to report")
            return

        # Generate comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.reports_dir / f"{self.ticker}_{self.strategy}_{timestamp}"
        report_dir.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"GENERATING COMPREHENSIVE REPORTS")
        print(f"{'='*80}\n")

        self._generate_summary_report(all_results, report_dir)
        self._generate_graphs(all_results, all_trades, report_dir)
        self._generate_detailed_csv(all_results, all_trades, report_dir)

        print(f"\n{'='*80}")
        print(f"✅ REPORTS SAVED TO: {report_dir}")
        print(f"{'='*80}\n")
        print(f"Generated files:")
        print(f"  📊 summary_report.txt - Overall performance summary")
        print(f"  📈 equity_curve.png - Equity growth over time")
        print(f"  📊 yearly_performance.png - Year-by-year breakdown")
        print(f"  📉 drawdown_analysis.png - Drawdown visualization")
        print(f"  📋 trade_distribution.png - Win/loss analysis")
        print(f"  💾 detailed_results.csv - Raw data export")
        print(f"  💾 all_trades.csv - Individual trade records\n")

    def run_wfa_for_year(self, year):
        """Run WFA for a specific year"""
        # Define date range
        if year == 2025:
            end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            end_date = f"{year}-12-31"
        start_date = f"{year}-01-01"

        # WFA needs training data from prior years
        wfa_train_years = self.config['wfa']['train_years']
        wfa_start = f"{year - wfa_train_years}-01-01"

        try:
            # Load data
            print(f"📥 Loading data from Alpaca...")
            adapter = DataAdapter(self.config, start_date=wfa_start, end_date=end_date)
            df_base = adapter.load_data(timeframe_suffix=self.timeframe)

            if df_base is None or df_base.empty:
                print(f"❌ No data loaded")
                return None

            print(f"✅ Loaded {len(df_base):,} bars ({wfa_start} to {end_date})")

            # Load higher timeframes
            df_h4 = adapter.load_h4_data(start_date=wfa_start, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=wfa_start, end_date=end_date)
            print(f"✅ H4: {len(df_h4):,} bars | Daily: {len(df_d1):,} bars")

            # Calculate features
            print(f"🔧 Calculating features...")
            features = FeatureEngineer(self.config)
            df_base = features.calculate_technical_features(df_base)

            # Detect FVGs and Order Blocks
            print(f"🔍 Detecting SMC patterns...")
            fvgs = features.detect_fvgs(df_base)
            obs = features.detect_obs(df_base)
            print(f"✅ Detected {len(fvgs):,} FVGs, {len(obs):,} OBs")

            # Generate structure and candidates
            structure = StructureAnalyzer(df_h4, self.config, daily_df=df_d1)
            entry_gen = EntryGenerator(self.config, structure, features)
            candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                print(f"❌ No candidates generated")
                return None

            print(f"✅ Generated {len(candidates):,} candidates")

            # Run Walk-Forward Analysis
            print(f"🔄 Running Walk-Forward Analysis...")
            wfa = WalkForwardAnalyzer(self.config)
            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates,
                start_date,
                end_date
            )

            if scored_candidates.empty:
                print(f"❌ WFA produced no scored candidates")
                return None

            print(f"✅ WFA: {len(scored_candidates):,} scored trades across {len(fold_summaries)} folds")

            # Run backtest
            print(f"💹 Running backtest...")
            backtester = Backtester(self.config)
            history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])

            if trades.empty:
                print(f"⚠️  No trades executed")
                return {
                    'year': year,
                    'folds': len(fold_summaries),
                    'candidates': len(scored_candidates),
                    'trades_count': 0,
                    'win_rate': 0.0,
                    'net_profit': 0.0,
                    'yoy_profit_pct': 0.0,
                    'max_drawdown': 0.0,
                    'trades': []
                }

            # Calculate metrics
            wins = trades[trades['net_pnl'] > 0]
            losses = trades[trades['net_pnl'] <= 0]

            win_rate = (len(wins) / len(trades) * 100) if len(trades) > 0 else 0
            net_profit = float(trades['net_pnl'].sum())
            initial_capital = self.config['backtest']['initial_capital']
            yoy_pct = (net_profit / initial_capital * 100) if initial_capital > 0 else 0

            # Calculate max drawdown
            max_dd = 0.0
            if not history.empty and 'equity' in history.columns:
                equity = history['equity'].values
                running_max = pd.Series(equity).expanding().max()
                drawdowns = (running_max - equity) / running_max * 100
                max_dd = float(drawdowns.max())

            # Print summary
            self._print_year_summary({
                'year': year,
                'trades_count': len(trades),
                'win_rate': win_rate,
                'net_profit': net_profit,
                'yoy_profit_pct': yoy_pct,
                'max_drawdown': max_dd
            })

            # Prepare trade records
            trades_list = []
            for _, trade in trades.iterrows():
                trades_list.append({
                    'year': year,
                    'entry_time': trade.get('entry_time'),
                    'exit_time': trade.get('exit_time'),
                    'direction': trade.get('direction'),
                    'pnl': trade.get('net_pnl'),
                    'win': trade.get('net_pnl') > 0
                })

            return {
                'year': year,
                'folds': len(fold_summaries),
                'candidates': len(scored_candidates),
                'trades_count': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': win_rate,
                'net_profit': net_profit,
                'yoy_profit_pct': yoy_pct,
                'max_drawdown': max_dd,
                'initial_capital': initial_capital,
                'final_capital': initial_capital + net_profit,
                'avg_win': float(wins['net_pnl'].mean()) if len(wins) > 0 else 0,
                'avg_loss': float(losses['net_pnl'].mean()) if len(losses) > 0 else 0,
                'trades': trades_list,
                'history': history
            }

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _print_year_summary(self, result):
        """Print year summary"""
        year = result['year']
        wr_icon = "✅" if result['win_rate'] >= 70 else "⚠️"
        profit_icon = "✅" if result['yoy_profit_pct'] >= 50 else "⚠️"
        dd_icon = "✅" if result['max_drawdown'] <= 10 else "⚠️"

        print(f"\n{year} Results:")
        print(f"  {wr_icon} WR: {result['win_rate']:.1f}% | {profit_icon} YoY: {result['yoy_profit_pct']:.1f}% | {dd_icon} DD: {result['max_drawdown']:.1f}% | Trades: {result['trades_count']}")

    def _generate_summary_report(self, results, report_dir):
        """Generate text summary report"""
        report_file = report_dir / 'summary_report.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"WALK-FORWARD ANALYSIS REPORT\n")
            f.write(f"Ticker: {self.ticker} | Strategy: {self.strategy.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Overall metrics
            total_trades = sum(r['trades_count'] for r in results)
            total_profit = sum(r['net_profit'] for r in results)
            avg_wr = np.mean([r['win_rate'] for r in results])
            avg_yoy = np.mean([r['yoy_profit_pct'] for r in results])
            max_dd = max(r['max_drawdown'] for r in results)

            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Years Tested:       {len(results)}\n")
            f.write(f"Total Trades:       {total_trades}\n")
            f.write(f"Total Profit:       ${total_profit:,.2f}\n")
            f.write(f"Avg Win Rate:       {avg_wr:.2f}%\n")
            f.write(f"Avg YoY Profit:     {avg_yoy:.2f}%\n")
            f.write(f"Max Drawdown:       {max_dd:.2f}%\n\n")

            # Target evaluation
            f.write("TARGET EVALUATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'✅' if avg_wr >= 70 else '❌'} Win Rate ≥70%:       {avg_wr:.2f}%\n")
            f.write(f"{'✅' if avg_yoy >= 50 else '❌'} YoY Profit ≥50%:    {avg_yoy:.2f}%\n")
            f.write(f"{'✅' if max_dd <= 10 else '❌'} Max Drawdown ≤10%:  {max_dd:.2f}%\n\n")

            # Year-by-year breakdown
            f.write("YEAR-BY-YEAR BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for r in results:
                f.write(f"\n{r['year']}:\n")
                f.write(f"  Trades:         {r['trades_count']}\n")
                f.write(f"  Win Rate:       {r['win_rate']:.2f}%\n")
                f.write(f"  Net Profit:     ${r['net_profit']:,.2f}\n")
                f.write(f"  YoY Profit:     {r['yoy_profit_pct']:.2f}%\n")
                f.write(f"  Max Drawdown:   {r['max_drawdown']:.2f}%\n")
                if r.get('avg_win') and r.get('avg_loss'):
                    f.write(f"  Avg Win:        ${r['avg_win']:,.2f}\n")
                    f.write(f"  Avg Loss:       ${r['avg_loss']:,.2f}\n")

        print(f"✅ Summary report saved")

    def _generate_graphs(self, results, all_trades, report_dir):
        """Generate comprehensive graphs"""
        # 1. Equity Curve
        self._plot_equity_curve(results, report_dir)

        # 2. Yearly Performance
        self._plot_yearly_performance(results, report_dir)

        # 3. Drawdown Analysis
        self._plot_drawdown_analysis(results, report_dir)

        # 4. Trade Distribution
        self._plot_trade_distribution(all_trades, report_dir)

    def _plot_equity_curve(self, results, report_dir):
        """Plot cumulative equity curve"""
        fig, ax = plt.subplots(figsize=(14, 8))

        cumulative_pnl = 0
        equity_data = []

        for r in results:
            if 'history' in r and not r['history'].empty:
                history = r['history'].copy()
                history['cumulative_equity'] = history['equity'] - r['initial_capital'] + cumulative_pnl
                equity_data.append(history)
                cumulative_pnl = history['cumulative_equity'].iloc[-1]

        if equity_data:
            combined = pd.concat(equity_data)
            ax.plot(combined.index, combined['cumulative_equity'], linewidth=2, color='#2E86AB')
            ax.fill_between(combined.index, 0, combined['cumulative_equity'], alpha=0.3, color='#2E86AB')

        ax.set_title(f'{self.ticker} - Cumulative Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(report_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Equity curve saved")

    def _plot_yearly_performance(self, results, report_dir):
        """Plot yearly performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        years = [r['year'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        yoy_profits = [r['yoy_profit_pct'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        trade_counts = [r['trades_count'] for r in results]

        # Win Rate
        axes[0, 0].bar(years, win_rates, color=['#06D6A0' if x >= 70 else '#F77F00' for x in win_rates])
        axes[0, 0].axhline(y=70, color='red', linestyle='--', label='Target: 70%')
        axes[0, 0].set_title('Win Rate by Year', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Win Rate (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # YoY Profit
        axes[0, 1].bar(years, yoy_profits, color=['#06D6A0' if x >= 50 else '#F77F00' for x in yoy_profits])
        axes[0, 1].axhline(y=50, color='red', linestyle='--', label='Target: 50%')
        axes[0, 1].set_title('YoY Profit by Year', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('YoY Profit (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Max Drawdown
        axes[1, 0].bar(years, drawdowns, color=['#06D6A0' if x <= 10 else '#F77F00' for x in drawdowns])
        axes[1, 0].axhline(y=10, color='red', linestyle='--', label='Limit: 10%')
        axes[1, 0].set_title('Max Drawdown by Year', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Trade Count
        axes[1, 1].bar(years, trade_counts, color='#2E86AB')
        axes[1, 1].set_title('Trade Count by Year', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / 'yearly_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Yearly performance chart saved")

    def _plot_drawdown_analysis(self, results, report_dir):
        """Plot drawdown analysis"""
        fig, ax = plt.subplots(figsize=(14, 8))

        for r in results:
            if 'history' in r and not r['history'].empty:
                history = r['history']
                equity = history['equity'].values
                running_max = pd.Series(equity).expanding().max()
                drawdown = (running_max - equity) / running_max * 100

                ax.fill_between(history.index, 0, -drawdown, alpha=0.5, label=f"{r['year']}")

        ax.axhline(y=-10, color='red', linestyle='--', linewidth=2, label='Max DD Limit: -10%')
        ax.set_title(f'{self.ticker} - Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Drawdown analysis saved")

    def _plot_trade_distribution(self, all_trades, report_dir):
        """Plot win/loss distribution"""
        if not all_trades:
            return

        df_trades = pd.DataFrame(all_trades)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Win/Loss pie chart
        wins = len(df_trades[df_trades['win'] == True])
        losses = len(df_trades[df_trades['win'] == False])

        axes[0].pie([wins, losses], labels=['Wins', 'Losses'],
                    autopct='%1.1f%%', colors=['#06D6A0', '#F77F00'],
                    startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[0].set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')

        # P&L histogram
        axes[1].hist(df_trades['pnl'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('P&L Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('P&L ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / 'trade_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Trade distribution chart saved")

    def _generate_detailed_csv(self, results, all_trades, report_dir):
        """Export detailed results to CSV"""
        # Results summary
        df_results = pd.DataFrame(results)
        if 'history' in df_results.columns:
            df_results = df_results.drop('history', axis=1)
        if 'trades' in df_results.columns:
            df_results = df_results.drop('trades', axis=1)

        df_results.to_csv(report_dir / 'detailed_results.csv', index=False)
        print(f"✅ Results CSV saved")

        # Trade records
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.to_csv(report_dir / 'all_trades.csv', index=False)
            print(f"✅ Trades CSV saved")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WFA with Comprehensive Reports")
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    parser.add_argument('--years', type=str, required=True, help='Years comma-separated')
    parser.add_argument('--config', type=str, default='configs/config_spy_optimized.yml')

    args = parser.parse_args()

    years_list = [int(y.strip()) for y in args.years.split(',')]

    wfa = WFAWithReports(args.ticker, args.strategy, args.config)
    wfa.run_wfa_for_years(years_list)


if __name__ == '__main__':
    main()
