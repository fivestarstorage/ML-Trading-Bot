"""
SPY SMC Strategy Walk-Forward Analysis with Iterative Optimization
Tests SPY strategy using Alpaca data for years 2020-2025
Targets: >70% win rate, >50% YoY profit
"""
import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List, Tuple
import json
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.ml_model import MLModel
from src.wfa import WalkForwardAnalyzer
from src.backtester import Backtester


class SPYWFAOptimizer:
    """Optimize SPY SMC Strategy via Walk-Forward Analysis"""

    def __init__(self):
        self.logger = setup_logging()
        self.test_years = [2020, 2021, 2022, 2023, 2024, 2025]
        self.target_win_rate = 0.70
        self.target_yoy_profit_pct = 50.0
        self.results_history = []

    def run_optimization(self):
        """Main optimization loop"""
        self.logger.info("="*80)
        self.logger.info("SPY SMC STRATEGY - WALK FORWARD ANALYSIS & OPTIMIZATION")
        self.logger.info("="*80)
        self.logger.info(f"Target Win Rate: {self.target_win_rate:.0%}")
        self.logger.info(f"Target YoY Profit: {self.target_yoy_profit_pct:.0f}%")
        self.logger.info("="*80 + "\n")

        # Test baseline configuration first
        self.logger.info("\n" + "="*80)
        self.logger.info("BASELINE TEST - Current Config")
        self.logger.info("="*80 + "\n")

        baseline_results = self.test_all_years("configs/config_spy.yml", iteration_name="baseline")
        self.results_history.append({"iteration": "baseline", "results": baseline_results})

        # Analyze baseline
        summary = self.analyze_results(baseline_results)
        self.logger.info("\n" + "="*80)
        self.logger.info("BASELINE SUMMARY")
        self.logger.info("="*80)
        self.print_summary(summary, baseline_results)

        # Check if we need optimization
        if summary['avg_win_rate'] >= self.target_win_rate and summary['avg_yoy_profit'] >= self.target_yoy_profit_pct:
            self.logger.info("\n🎉 BASELINE ALREADY MEETS TARGETS!")
            self.save_report()
            return baseline_results

        # Run optimization iterations
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING OPTIMIZATION ITERATIONS")
        self.logger.info("="*80 + "\n")

        best_results = baseline_results
        best_summary = summary

        # Define optimization strategies
        optimization_configs = self.generate_optimization_configs()

        for idx, opt_config in enumerate(optimization_configs, 1):
            self.logger.info("\n" + "-"*80)
            self.logger.info(f"OPTIMIZATION ITERATION {idx}/{len(optimization_configs)}")
            self.logger.info(f"Strategy: {opt_config['name']}")
            self.logger.info(f"Changes: {opt_config['description']}")
            self.logger.info("-"*80 + "\n")

            # Create modified config
            config_path = self.create_modified_config(opt_config)

            # Test
            results = self.test_all_years(config_path, iteration_name=opt_config['name'])
            self.results_history.append({"iteration": opt_config['name'], "results": results})

            # Analyze
            summary = self.analyze_results(results)
            self.print_summary(summary, results)

            # Check if this is better
            if self.is_better(summary, best_summary):
                self.logger.info(f"\n✅ IMPROVEMENT FOUND! New best iteration: {opt_config['name']}")
                best_results = results
                best_summary = summary

                # Save best config
                self.save_best_config(config_path, opt_config['name'])

            # Check if targets met
            if summary['avg_win_rate'] >= self.target_win_rate and summary['avg_yoy_profit'] >= self.target_yoy_profit_pct:
                self.logger.info(f"\n🎉 TARGETS MET! Win Rate: {summary['avg_win_rate']:.2%}, YoY Profit: {summary['avg_yoy_profit']:.1f}%")
                break

        # Final report
        self.logger.info("\n" + "="*80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Best Iteration: {best_summary.get('iteration_name', 'baseline')}")
        self.print_summary(best_summary, best_results)

        self.save_report()
        return best_results

    def test_all_years(self, config_path: str, iteration_name: str = "test") -> List[Dict]:
        """Test strategy across all years"""
        results = []
        config = load_config(config_path)

        # Ensure using Alpaca as data source
        config['data']['source'] = 'alpaca'
        config['data']['symbol'] = 'SPY'

        for year in self.test_years:
            # Adjust end date for 2025 (partial year)
            if year == 2025:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = f"{year}-12-31"

            start_date = f"{year}-01-01"

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing SPY-SMC for {year} ({iteration_name})")
            self.logger.info(f"Period: {start_date} to {end_date}")
            self.logger.info(f"{'='*60}")

            result = self.test_single_year(
                config=config,
                year=year,
                start_date=start_date,
                end_date=end_date,
                iteration_name=iteration_name
            )
            results.append(result)

        return results

    def test_single_year(
        self,
        config: dict,
        year: int,
        start_date: str,
        end_date: str,
        iteration_name: str
    ) -> Dict:
        """Test WFA for a single year"""
        result = {
            'iteration': iteration_name,
            'year': year,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'error': None,
            'metrics': {}
        }

        try:
            # Load data from Alpaca
            self.logger.info("1. Loading data from Alpaca...")
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            base_tf = config['data']['timeframe_base']
            df_base = adapter.load_data(timeframe_suffix=base_tf)

            if df_base.empty:
                raise ValueError(f"No data available for {year}")

            self.logger.info(f"   ✓ Loaded {len(df_base)} bars from Alpaca")
            result['metrics']['data_bars'] = len(df_base)

            # Load higher timeframes
            self.logger.info("2. Loading higher timeframes...")
            df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)
            self.logger.info(f"   ✓ H4: {len(df_h4)} bars, D1: {len(df_d1)} bars")

            # Feature engineering
            self.logger.info("3. Calculating features...")
            features = FeatureEngineer(config)
            df_base = features.calculate_technical_features(df_base)
            self.logger.info("   ✓ Features calculated")

            # Generate candidates
            self.logger.info("4. Generating SMC candidates...")
            structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
            fvgs = features.detect_fvgs(df_base)
            obs = features.detect_obs(df_base)
            self.logger.info(f"   ✓ Detected {len(fvgs)} FVGs, {len(obs)} OBs")

            entry_gen = EntryGenerator(config, structure, features)
            candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                self.logger.warning(f"   ⚠ No candidates generated for {year}")
                result['metrics']['total_candidates'] = 0
                result['success'] = True  # Mark as success but with 0 trades
                return result

            self.logger.info(f"   ✓ Generated {len(candidates)} candidates")
            result['metrics']['total_candidates'] = len(candidates)

            # Run WFA
            self.logger.info("5. Running Walk-Forward Analysis...")
            wfa = WalkForwardAnalyzer(config)

            # For WFA we need training data from prior years
            train_years = config.get('wfa', {}).get('train_years', 3)
            wfa_start = f"{year - train_years}-01-01"

            # Load extended data for WFA training
            adapter_wfa = DataAdapter(config, start_date=wfa_start, end_date=end_date)
            df_wfa = adapter_wfa.load_data(timeframe_suffix=base_tf)
            df_wfa = features.calculate_technical_features(df_wfa)

            # Generate candidates for full WFA period
            df_h4_wfa = adapter_wfa.load_h4_data(start_date=wfa_start, end_date=end_date)
            df_d1_wfa = adapter_wfa.load_daily_data(start_date=wfa_start, end_date=end_date)
            structure_wfa = StructureAnalyzer(df_h4_wfa, config, daily_df=df_d1_wfa)
            fvgs_wfa = features.detect_fvgs(df_wfa)
            obs_wfa = features.detect_obs(df_wfa)
            entry_gen_wfa = EntryGenerator(config, structure_wfa, features)
            candidates_wfa = entry_gen_wfa.generate_candidates(df_wfa, df_h4_wfa, fvgs_wfa, obs_wfa)

            if candidates_wfa.empty:
                self.logger.warning(f"   ⚠ No WFA candidates for {year}")
                result['metrics']['wfa_trades'] = 0
                result['success'] = True
                return result

            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates_wfa,
                start_date,
                end_date
            )

            if scored_candidates.empty:
                self.logger.warning(f"   ⚠ WFA produced no scored candidates for {year}")
                result['metrics']['wfa_trades'] = 0
                result['success'] = True
                return result

            self.logger.info(f"   ✓ WFA completed: {len(scored_candidates)} scored trades across {len(fold_summaries)} folds")
            result['metrics']['wfa_trades'] = len(scored_candidates)
            result['metrics']['wfa_folds'] = len(fold_summaries)

            # Run backtest on WFA results
            self.logger.info("6. Running backtest on WFA results...")
            backtester = Backtester(config)
            history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])

            if trades.empty:
                self.logger.warning("   ⚠ No trades executed in backtest")
                result['metrics']['trade_count'] = 0
                result['success'] = True
            else:
                wins = trades[trades['net_pnl'] > 0]
                losses = trades[trades['net_pnl'] <= 0]

                initial_capital = config.get('backtest', {}).get('initial_capital', 10000)
                total_pnl = float(trades['net_pnl'].sum())
                yoy_profit_pct = (total_pnl / initial_capital) * 100

                metrics = {
                    'trade_count': len(trades),
                    'win_count': len(wins),
                    'loss_count': len(losses),
                    'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0,
                    'net_profit': total_pnl,
                    'yoy_profit_pct': yoy_profit_pct,
                    'avg_win': float(wins['net_pnl'].mean()) if len(wins) > 0 else 0,
                    'avg_loss': float(losses['net_pnl'].mean()) if len(losses) > 0 else 0,
                    'profit_factor': abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf'),
                }

                if not history.empty and 'equity' in history.columns:
                    equity = history['equity'].values
                    running_max = pd.Series(equity).expanding().max()
                    drawdowns = (running_max - equity) / running_max * 100
                    metrics['max_drawdown_pct'] = float(drawdowns.max())

                result['metrics'].update(metrics)

                self.logger.info("   ✓ Backtest Results:")
                self.logger.info(f"      Trades: {metrics['trade_count']}")
                self.logger.info(f"      Win Rate: {metrics['win_rate']:.2%}")
                self.logger.info(f"      YoY Profit: {metrics['yoy_profit_pct']:.2f}%")
                self.logger.info(f"      Net Profit: ${metrics['net_profit']:.2f}")
                self.logger.info(f"      Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                if 'max_drawdown_pct' in metrics:
                    self.logger.info(f"      Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.logger.error(f"\n❌ {year} - TEST FAILED")
            self.logger.error(f"   Error: {e}")
            self.logger.debug(traceback.format_exc())

        return result

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results across all years"""
        successful_results = [r for r in results if r['success'] and r['metrics'].get('trade_count', 0) > 0]

        if not successful_results:
            return {
                'avg_win_rate': 0,
                'avg_yoy_profit': 0,
                'total_trades': 0,
                'years_tested': len(results),
                'years_successful': 0
            }

        win_rates = [r['metrics']['win_rate'] for r in successful_results]
        yoy_profits = [r['metrics'].get('yoy_profit_pct', 0) for r in successful_results]
        total_trades = sum(r['metrics'].get('trade_count', 0) for r in successful_results)

        return {
            'avg_win_rate': np.mean(win_rates),
            'avg_yoy_profit': np.mean(yoy_profits),
            'median_win_rate': np.median(win_rates),
            'median_yoy_profit': np.median(yoy_profits),
            'min_win_rate': np.min(win_rates),
            'max_win_rate': np.max(win_rates),
            'min_yoy_profit': np.min(yoy_profits),
            'max_yoy_profit': np.max(yoy_profits),
            'total_trades': total_trades,
            'years_tested': len(results),
            'years_successful': len(successful_results)
        }

    def print_summary(self, summary: Dict, results: List[Dict]):
        """Print summary statistics"""
        self.logger.info(f"Years Tested: {summary['years_tested']}")
        self.logger.info(f"Years With Trades: {summary['years_successful']}")
        self.logger.info(f"Total Trades: {summary['total_trades']}")
        self.logger.info(f"")
        self.logger.info(f"Average Win Rate: {summary['avg_win_rate']:.2%}")
        self.logger.info(f"Median Win Rate: {summary['median_win_rate']:.2%}")
        self.logger.info(f"Win Rate Range: {summary['min_win_rate']:.2%} - {summary['max_win_rate']:.2%}")
        self.logger.info(f"")
        self.logger.info(f"Average YoY Profit: {summary['avg_yoy_profit']:.2f}%")
        self.logger.info(f"Median YoY Profit: {summary['median_yoy_profit']:.2f}%")
        self.logger.info(f"YoY Profit Range: {summary['min_yoy_profit']:.2f}% - {summary['max_yoy_profit']:.2f}%")
        self.logger.info(f"")

        # Year by year breakdown
        self.logger.info("Year-by-Year Breakdown:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Year':<8} {'Trades':<10} {'Win Rate':<12} {'YoY Profit':<15}")
        self.logger.info("-" * 80)

        for r in results:
            if r['success'] and r['metrics'].get('trade_count', 0) > 0:
                wr = r['metrics']['win_rate']
                yoy = r['metrics'].get('yoy_profit_pct', 0)
                trades = r['metrics']['trade_count']
                self.logger.info(f"{r['year']:<8} {trades:<10} {wr:>10.2%}  {yoy:>12.2f}%")
            else:
                self.logger.info(f"{r['year']:<8} No trades or error")

        self.logger.info("-" * 80)

    def is_better(self, summary: Dict, best_summary: Dict) -> bool:
        """Determine if new summary is better than current best"""
        # Prioritize meeting targets
        new_meets_targets = (summary['avg_win_rate'] >= self.target_win_rate and
                            summary['avg_yoy_profit'] >= self.target_yoy_profit_pct)
        best_meets_targets = (best_summary['avg_win_rate'] >= self.target_win_rate and
                             best_summary['avg_yoy_profit'] >= self.target_yoy_profit_pct)

        if new_meets_targets and not best_meets_targets:
            return True
        elif not new_meets_targets and best_meets_targets:
            return False

        # Both meet or both don't meet - compare overall score
        # Weighted score: 60% win rate, 40% profit
        new_score = (summary['avg_win_rate'] * 0.6) + (summary['avg_yoy_profit'] / 100 * 0.4)
        best_score = (best_summary['avg_win_rate'] * 0.6) + (best_summary['avg_yoy_profit'] / 100 * 0.4)

        return new_score > best_score

    def generate_optimization_configs(self) -> List[Dict]:
        """Generate different configuration variants to test"""
        configs = []

        # Strategy 1: More selective entries
        configs.append({
            'name': 'selective_entries',
            'description': 'Stricter filters: require displacement, liquidity sweeps, and inducement',
            'changes': {
                'strategy.require_displacement': True,
                'strategy.displacement_min_atr': 1.2,
                'strategy.require_liquidity_sweep': True,
                'strategy.require_inducement': True,
                'strategy.require_discount_premium': True,
                'strategy.min_volume_zscore': 0.0,
            }
        })

        # Strategy 2: Better risk/reward
        configs.append({
            'name': 'better_rr',
            'description': 'Wider TP, tighter SL for better R:R',
            'changes': {
                'strategy.tp_atr_mult': 2.5,
                'strategy.sl_atr_mult': 1.2,
                'strategy.label_total_rr': 2.5,
            }
        })

        # Strategy 3: Session focus
        configs.append({
            'name': 'session_focus',
            'description': 'Focus on US market open and close',
            'changes': {
                'strategy.require_killzone_filter': True,
                'strategy.session_filter': ['regular'],
            }
        })

        # Strategy 4: Stricter ML threshold
        configs.append({
            'name': 'strict_ml',
            'description': 'Higher ML probability threshold',
            'changes': {
                'strategy.model_threshold': 0.70,
            }
        })

        # Strategy 5: Combine best filters
        configs.append({
            'name': 'combined_strict',
            'description': 'Combination of displacement, sessions, and higher ML threshold',
            'changes': {
                'strategy.require_displacement': True,
                'strategy.displacement_min_atr': 1.0,
                'strategy.require_killzone_filter': True,
                'strategy.session_filter': ['regular'],
                'strategy.model_threshold': 0.65,
                'strategy.tp_atr_mult': 2.2,
                'strategy.sl_atr_mult': 1.3,
                'strategy.require_inducement': True,
            }
        })

        # Strategy 6: Daily trend alignment
        configs.append({
            'name': 'daily_trend_aligned',
            'description': 'Strong daily trend filter, no range trades',
            'changes': {
                'strategy.use_daily_trend_filter': True,
                'strategy.allow_daily_range_trades': False,
            }
        })

        # Strategy 7: Conservative sizing
        configs.append({
            'name': 'conservative_sizing',
            'description': 'Lower risk per trade but higher quality',
            'changes': {
                'strategy.risk_percent': 0.3,
                'strategy.require_displacement': True,
                'strategy.displacement_min_atr': 1.1,
                'strategy.model_threshold': 0.65,
            }
        })

        return configs

    def create_modified_config(self, opt_config: Dict) -> str:
        """Create a modified config file with changes"""
        # Load base config
        with open('configs/config_spy.yml', 'r') as f:
            config = yaml.safe_load(f)

        # Apply changes
        for key, value in opt_config['changes'].items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # Save modified config
        os.makedirs('configs/optimization', exist_ok=True)
        config_path = f'configs/optimization/config_spy_{opt_config["name"]}.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def save_best_config(self, config_path: str, name: str):
        """Save the best configuration"""
        import shutil
        best_path = 'configs/config_spy_best.yml'
        shutil.copy(config_path, best_path)
        self.logger.info(f"💾 Saved best config to: {best_path}")

    def save_report(self):
        """Save detailed optimization report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"reports/spy_wfa_optimization_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

        # Save JSON results
        with open(f"{report_dir}/optimization_results.json", 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)

        # Create summary CSV
        summary_rows = []
        for item in self.results_history:
            iteration = item['iteration']
            results = item['results']
            summary = self.analyze_results(results)
            summary['iteration'] = iteration
            summary_rows.append(summary)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"{report_dir}/optimization_summary.csv", index=False)

        self.logger.info(f"\n📊 Optimization report saved to: {report_dir}")


if __name__ == "__main__":
    optimizer = SPYWFAOptimizer()
    results = optimizer.run_optimization()

    # Exit with appropriate code
    summary = optimizer.analyze_results(results)
    targets_met = (summary['avg_win_rate'] >= optimizer.target_win_rate and
                   summary['avg_yoy_profit'] >= optimizer.target_yoy_profit_pct)

    sys.exit(0 if targets_met else 1)
