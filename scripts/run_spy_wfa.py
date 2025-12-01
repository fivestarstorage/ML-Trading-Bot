#!/usr/bin/env python3
"""
Simplified Walk-Forward Analysis runner for optimized SPY strategy.
Uses the pre-optimized configuration from configs/config_spy_optimized.yml
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
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


class SPYWalkForwardAnalysis:
    def __init__(self, config_path="configs/config_spy_optimized.yml"):
        self.config = load_config(config_path)
        self.logger = setup_logging()
        self.symbol = self.config['data']['symbol']
        self.timeframe = self.config['data']['timeframe_base']

    def run_wfa_for_year(self, year: int):
        """Run WFA for a specific year"""
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ANALYSIS: {self.symbol} - Year {year}")
        print(f"{'='*80}")

        # Define date range
        if year == 2025:
            end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            end_date = f"{year}-12-31"
        start_date = f"{year}-01-01"

        # WFA needs training data from prior years
        wfa_train_years = self.config['wfa']['train_years']
        wfa_start = f"{year - wfa_train_years}-01-01"

        print(f"\n📊 Configuration:")
        print(f"   Symbol: {self.symbol}")
        print(f"   Test Period: {start_date} to {end_date}")
        print(f"   Training Window: {wfa_train_years} years")
        print(f"   Full WFA Period: {wfa_start} to {end_date}")

        try:
            # Load data
            print(f"\n📥 Loading data from Alpaca...")
            adapter = DataAdapter(self.config, start_date=wfa_start, end_date=end_date)
            df_base = adapter.load_data(timeframe_suffix=self.timeframe)

            if df_base is None or df_base.empty:
                print(f"❌ No data loaded for {self.symbol}")
                return None

            print(f"✅ Loaded {len(df_base):,} bars ({wfa_start} to {end_date})")

            # Load higher timeframes
            print(f"\n📈 Loading higher timeframes...")
            df_h4 = adapter.load_h4_data(start_date=wfa_start, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=wfa_start, end_date=end_date)
            print(f"✅ H4: {len(df_h4):,} bars | Daily: {len(df_d1):,} bars")

            # Calculate features
            print(f"\n🔧 Calculating technical features...")
            features = FeatureEngineer(self.config)
            df_base = features.calculate_technical_features(df_base)
            print(f"✅ Features calculated")

            # Detect FVGs and Order Blocks
            print(f"\n🔍 Detecting FVGs and Order Blocks...")
            fvgs = features.detect_fvgs(df_base)
            obs = features.detect_obs(df_base)
            print(f"✅ Detected {len(fvgs):,} FVGs, {len(obs):,} OBs")

            # Generate structure and candidates
            print(f"\n📐 Analyzing market structure...")
            structure = StructureAnalyzer(df_h4, self.config, daily_df=df_d1)
            entry_gen = EntryGenerator(self.config, structure, features)
            candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                print(f"❌ No candidates generated")
                return None

            print(f"✅ Generated {len(candidates):,} candidates")

            # Run Walk-Forward Analysis
            print(f"\n🔄 Running Walk-Forward Analysis...")
            wfa = WalkForwardAnalyzer(self.config)
            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates,
                start_date,  # Test only the target year
                end_date
            )

            if scored_candidates.empty:
                print(f"❌ WFA produced no scored candidates")
                return None

            print(f"✅ WFA completed: {len(scored_candidates):,} scored trades across {len(fold_summaries)} folds")

            # Run backtest on WFA results
            print(f"\n💹 Running backtest on WFA results...")
            backtester = Backtester(self.config)
            history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])

            if trades.empty:
                print(f"⚠️  No trades executed in backtest")
                return {
                    'year': year,
                    'folds': len(fold_summaries),
                    'candidates': len(scored_candidates),
                    'trades': 0,
                    'win_rate': 0.0,
                    'net_profit': 0.0,
                    'yoy_profit_pct': 0.0,
                    'max_drawdown': 0.0
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

            # Print results
            self._print_summary({
                'year': year,
                'folds': len(fold_summaries),
                'candidates': len(scored_candidates),
                'trades': len(trades),
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
            })

            return {
                'year': year,
                'folds': len(fold_summaries),
                'candidates': len(scored_candidates),
                'trades': len(trades),
                'win_rate': win_rate,
                'net_profit': net_profit,
                'yoy_profit_pct': yoy_pct,
                'max_drawdown': max_dd
            }

        except Exception as e:
            print(f"\n❌ Error during WFA for year {year}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _print_summary(self, results):
        """Print summary of results"""
        year = results['year']
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY - {year}")
        print(f"{'='*80}")
        print(f"WFA Folds:         {results['folds']}")
        print(f"Candidates:        {results['candidates']:,}")
        print(f"Total Trades:      {results['trades']}")
        print(f"Wins / Losses:     {results.get('wins', 0)} / {results.get('losses', 0)}")
        print(f"Win Rate:          {results['win_rate']:.2f}%")
        print(f"Net Profit:        ${results['net_profit']:,.2f}")
        print(f"YoY Profit:        {results['yoy_profit_pct']:.2f}%")
        print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
        print(f"Initial Capital:   ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Capital:     ${results.get('final_capital', 0):,.2f}")

        if 'avg_win' in results and 'avg_loss' in results:
            print(f"Avg Win:           ${results['avg_win']:,.2f}")
            print(f"Avg Loss:          ${results['avg_loss']:,.2f}")

        # Check targets
        print(f"\n📋 Target Evaluation:")
        wr_pass = "✅" if results['win_rate'] >= 70 else "❌"
        profit_pass = "✅" if results['yoy_profit_pct'] >= 50 else "❌"
        dd_pass = "✅" if results['max_drawdown'] <= 10 else "❌"

        print(f"   {wr_pass} Win Rate ≥70%:       {results['win_rate']:.2f}%")
        print(f"   {profit_pass} YoY Profit ≥50%:    {results['yoy_profit_pct']:.2f}%")
        print(f"   {dd_pass} Max Drawdown ≤10%:  {results['max_drawdown']:.2f}%")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="SPY Walk-Forward Analysis")
    parser.add_argument('--year', type=int, help='Year to test (e.g., 2024)')
    parser.add_argument('--years', type=str, help='Multiple years comma-separated (e.g., 2020,2021,2022)')
    parser.add_argument('--config', type=str, default='configs/config_spy_optimized.yml',
                       help='Config file path')

    args = parser.parse_args()

    wfa = SPYWalkForwardAnalysis(config_path=args.config)

    # Determine which years to test
    years_to_test = []
    if args.year:
        years_to_test = [args.year]
    elif args.years:
        years_to_test = [int(y.strip()) for y in args.years.split(',')]
    else:
        # Default to current year
        years_to_test = [datetime.now().year]

    # Run WFA for each year
    all_results = []
    for year in years_to_test:
        result = wfa.run_wfa_for_year(year)
        if result:
            all_results.append(result)

    # Print overall summary if multiple years
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"MULTI-YEAR SUMMARY")
        print(f"{'='*80}")
        for r in all_results:
            wr_icon = "✅" if r['win_rate'] >= 70 else "⚠️"
            profit_icon = "✅" if r['yoy_profit_pct'] >= 50 else "⚠️"
            dd_icon = "✅" if r['max_drawdown'] <= 10 else "⚠️"
            print(f"\n{r['year']}: {wr_icon} WR={r['win_rate']:.1f}% | {profit_icon} YoY={r['yoy_profit_pct']:.1f}% | {dd_icon} DD={r['max_drawdown']:.1f}% | Trades={r['trades']}")


if __name__ == '__main__':
    main()
