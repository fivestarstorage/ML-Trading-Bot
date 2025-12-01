"""
SPY SMC Strategy - Aggressive Optimization Focused on 2024-2025 First
Then apply successful parameters to earlier years
Targets: >70% win rate, >50% YoY profit
"""
import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import yaml
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.ml_model import MLModel
from src.wfa import WalkForwardAnalyzer
from src.backtester import Backtester


class AggressiveSPYOptimizer:
    """Aggressively optimize SPY for 2024-2025, then apply to earlier years"""

    def __init__(self):
        self.logger = setup_logging()
        self.target_win_rate = 0.70
        self.target_yoy_profit_pct = 50.0
        self.results_history = []
        self.best_config_2024 = None
        self.best_metrics_2024 = None

    def run_optimization(self):
        """Main optimization - focus on 2024-2025 first"""
        self.logger.info("="*80)
        self.logger.info("AGGRESSIVE SPY OPTIMIZATION - 2024-2025 FIRST")
        self.logger.info("="*80)
        self.logger.info(f"Target Win Rate: {self.target_win_rate:.0%}")
        self.logger.info(f"Target YoY Profit: {self.target_yoy_profit_pct:.0f}%")
        self.logger.info("="*80 + "\n")

        # Phase 1: Optimize for 2024-2025
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1: OPTIMIZE FOR 2024-2025")
        self.logger.info("="*80 + "\n")

        best_2024_config = self.optimize_for_recent_years()

        # Phase 2: Test on 2024 and verify
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: VERIFY 2024 PERFORMANCE")
        self.logger.info("="*80 + "\n")

        result_2024 = self.test_single_year(best_2024_config, 2024, "2024-01-01", "2024-12-31", "best_2024")
        self.logger.info(f"\n2024 Results with Best Config:")
        self.print_result(result_2024)

        if result_2024['success']:
            metrics = result_2024['metrics']
            win_rate = metrics.get('win_rate', 0)
            yoy_profit = metrics.get('yoy_profit_pct', 0)

            if win_rate >= self.target_win_rate and yoy_profit >= self.target_yoy_profit_pct:
                self.logger.info(f"\n🎉 2024 TARGETS MET! WR: {win_rate:.2%}, YoY: {yoy_profit:.1f}%")
                self.best_config_2024 = best_2024_config
                self.best_metrics_2024 = metrics
            else:
                self.logger.warning(f"\n⚠️ 2024 targets not fully met. WR: {win_rate:.2%}, YoY: {yoy_profit:.1f}%")
                self.logger.info("Continuing with best found configuration...")
                self.best_config_2024 = best_2024_config
                self.best_metrics_2024 = metrics

        # Phase 3: Apply to all years
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 3: APPLY BEST CONFIG TO ALL YEARS (2020-2025)")
        self.logger.info("="*80 + "\n")

        all_years_results = []
        for year in [2020, 2021, 2022, 2023, 2024, 2025]:
            end_date = datetime.now().strftime("%Y-%m-%d") if year == 2025 else f"{year}-12-31"
            result = self.test_single_year(
                self.best_config_2024,
                year,
                f"{year}-01-01",
                end_date,
                f"final_{year}"
            )
            all_years_results.append(result)

        # Final summary
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL RESULTS - ALL YEARS")
        self.logger.info("="*80)
        self.print_all_years_summary(all_years_results)

        # Save best config
        self.save_best_config(self.best_config_2024, "spy_optimized_2024")
        self.save_final_report(all_years_results)

        return all_years_results

    def optimize_for_recent_years(self) -> dict:
        """Aggressively optimize for 2024-2025"""
        base_config = load_config('configs/config_spy.yml')

        # Grid search parameters optimized for MORE TRADES and HIGHER PROFITS
        param_grid = {
            'risk_percent': [0.5, 0.7, 1.0, 1.5, 2.0],  # More aggressive position sizing
            'model_threshold': [0.50, 0.55, 0.60],  # Lower threshold = more trades
            'require_displacement': [False, True],  # Test both
            'displacement_min_atr': [0.5, 0.8, 1.0],
            'require_inducement': [False, True],
            'require_liquidity_sweep': [False],  # Keep false for more trades
            'require_discount_premium': [False, True],
            'tp_atr_mult': [2.0, 2.5, 3.0, 3.5],  # Higher TPs
            'sl_atr_mult': [1.0, 1.2, 1.5],
            'require_killzone_filter': [False, True],
            'use_daily_trend_filter': [False, True],
        }

        best_config = None
        best_score = -float('inf')
        best_metrics = None
        iteration = 0
        total_iterations = 100  # Test top 100 combinations

        self.logger.info(f"Testing {total_iterations} parameter combinations for 2024...")

        # Generate combinations (prioritizing high risk, high reward configs)
        high_priority_combos = []

        # Strategy 1: High risk, high reward, minimal filters
        for risk in [1.5, 2.0]:
            for threshold in [0.50, 0.55]:
                for tp in [3.0, 3.5]:
                    for sl in [1.0, 1.2]:
                        high_priority_combos.append({
                            'risk_percent': risk,
                            'model_threshold': threshold,
                            'require_displacement': False,
                            'displacement_min_atr': 0.5,
                            'require_inducement': False,
                            'require_liquidity_sweep': False,
                            'require_discount_premium': False,
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'require_killzone_filter': False,
                            'use_daily_trend_filter': False,
                        })

        # Strategy 2: Medium risk, selective entries
        for risk in [0.7, 1.0]:
            for threshold in [0.55, 0.60]:
                for tp in [2.5, 3.0]:
                    for sl in [1.2, 1.5]:
                        high_priority_combos.append({
                            'risk_percent': risk,
                            'model_threshold': threshold,
                            'require_displacement': True,
                            'displacement_min_atr': 0.8,
                            'require_inducement': True,
                            'require_liquidity_sweep': False,
                            'require_discount_premium': True,
                            'tp_atr_mult': tp,
                            'sl_atr_mult': sl,
                            'require_killzone_filter': True,
                            'use_daily_trend_filter': True,
                        })

        # Test combinations
        for params in high_priority_combos[:total_iterations]:
            iteration += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Iteration {iteration}/{min(total_iterations, len(high_priority_combos))}")
            self.logger.info(f"Risk: {params['risk_percent']}%, Threshold: {params['model_threshold']}")
            self.logger.info(f"TP: {params['tp_atr_mult']}x, SL: {params['sl_atr_mult']}x")
            self.logger.info(f"{'='*60}")

            # Create test config
            test_config = self.create_test_config(base_config, params)

            # Test on 2024
            result = self.test_single_year(test_config, 2024, "2024-01-01", "2024-12-31", f"iter_{iteration}")

            if result['success'] and result['metrics'].get('trade_count', 0) > 0:
                metrics = result['metrics']
                win_rate = metrics['win_rate']
                yoy_profit = metrics['yoy_profit_pct']
                trade_count = metrics['trade_count']

                # Score: prioritize meeting both targets, then optimize profit
                score = 0
                if win_rate >= self.target_win_rate:
                    score += 1000
                if yoy_profit >= self.target_yoy_profit_pct:
                    score += 1000

                # Add actual metrics to score
                score += win_rate * 100
                score += yoy_profit
                score += min(trade_count / 10, 50)  # Bonus for more trades (up to 50 points)

                self.logger.info(f"✓ WR: {win_rate:.2%}, YoY: {yoy_profit:.1f}%, Trades: {trade_count}, Score: {score:.0f}")

                if score > best_score:
                    best_score = score
                    best_config = test_config
                    best_metrics = metrics
                    self.logger.info(f"🌟 NEW BEST! Score: {score:.0f}")

                    # Check if we met targets
                    if win_rate >= self.target_win_rate and yoy_profit >= self.target_yoy_profit_pct:
                        self.logger.info(f"\n🎉 TARGETS MET EARLY! WR: {win_rate:.2%}, YoY: {yoy_profit:.1f}%")
                        # Continue to see if we can do better
            else:
                self.logger.info(f"✗ Failed or no trades")

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"OPTIMIZATION COMPLETE")
        self.logger.info(f"Best Score: {best_score:.0f}")
        if best_metrics:
            self.logger.info(f"Best WR: {best_metrics['win_rate']:.2%}")
            self.logger.info(f"Best YoY: {best_metrics['yoy_profit_pct']:.1f}%")
            self.logger.info(f"Trades: {best_metrics['trade_count']}")
        self.logger.info(f"{'='*80}\n")

        return best_config if best_config else base_config

    def create_test_config(self, base_config: dict, params: dict) -> dict:
        """Create a test configuration with given parameters"""
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy

        # Update strategy params
        for key, value in params.items():
            config['strategy'][key] = value

        # Ensure data source is Alpaca
        config['data']['source'] = 'alpaca'
        config['data']['symbol'] = 'SPY'

        return config

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
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            base_tf = config['data']['timeframe_base']
            df_base = adapter.load_data(timeframe_suffix=base_tf)

            if df_base.empty:
                raise ValueError(f"No data available for {year}")

            # Load higher timeframes
            df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)

            # Feature engineering
            features = FeatureEngineer(config)
            df_base = features.calculate_technical_features(df_base)

            # Generate candidates
            structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
            fvgs = features.detect_fvgs(df_base)
            obs = features.detect_obs(df_base)

            entry_gen = EntryGenerator(config, structure, features)
            candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                result['metrics']['total_candidates'] = 0
                result['success'] = True
                return result

            result['metrics']['total_candidates'] = len(candidates)

            # Run WFA
            wfa = WalkForwardAnalyzer(config)
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
                result['metrics']['wfa_trades'] = 0
                result['success'] = True
                return result

            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates_wfa,
                start_date,
                end_date
            )

            if scored_candidates.empty:
                result['metrics']['wfa_trades'] = 0
                result['success'] = True
                return result

            result['metrics']['wfa_trades'] = len(scored_candidates)
            result['metrics']['wfa_folds'] = len(fold_summaries)

            # Run backtest on WFA results
            backtester = Backtester(config)
            history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])

            if trades.empty:
                result['metrics']['trade_count'] = 0
                result['success'] = True
            else:
                wins = trades[trades['net_pnl'] > 0]
                losses = trades[trades['net_pnl'] <= 0]

                initial_capital = config.get('backtest', {}).get('initial_capital', 6000)
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

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.logger.error(f"Error testing {year}: {e}")

        return result

    def print_result(self, result: Dict):
        """Print a single year result"""
        if result['success'] and result['metrics'].get('trade_count', 0) > 0:
            m = result['metrics']
            self.logger.info(f"  Trades: {m['trade_count']}")
            self.logger.info(f"  Win Rate: {m['win_rate']:.2%}")
            self.logger.info(f"  YoY Profit: {m['yoy_profit_pct']:.2f}%")
            self.logger.info(f"  Net Profit: ${m['net_profit']:.2f}")
            self.logger.info(f"  Profit Factor: {m.get('profit_factor', 0):.2f}")
            if 'max_drawdown_pct' in m:
                self.logger.info(f"  Max DD: {m['max_drawdown_pct']:.2f}%")
        else:
            self.logger.info(f"  No trades or error: {result.get('error', 'N/A')}")

    def print_all_years_summary(self, results: List[Dict]):
        """Print summary for all years"""
        self.logger.info("\n" + "-"*80)
        self.logger.info(f"{'Year':<8} {'Trades':<10} {'Win Rate':<12} {'YoY Profit':<15} {'Status'}")
        self.logger.info("-"*80)

        for r in results:
            if r['success'] and r['metrics'].get('trade_count', 0) > 0:
                m = r['metrics']
                wr = m['win_rate']
                yoy = m['yoy_profit_pct']
                trades = m['trade_count']

                status = "✅" if (wr >= self.target_win_rate and yoy >= self.target_yoy_profit_pct) else "⚠️"

                self.logger.info(
                    f"{r['year']:<8} {trades:<10} {wr:>10.2%}  {yoy:>12.2f}%  {status}"
                )
            else:
                self.logger.info(f"{r['year']:<8} No trades or error")

        self.logger.info("-"*80)

        # Calculate averages
        successful = [r for r in results if r['success'] and r['metrics'].get('trade_count', 0) > 0]
        if successful:
            avg_wr = np.mean([r['metrics']['win_rate'] for r in successful])
            avg_yoy = np.mean([r['metrics']['yoy_profit_pct'] for r in successful])
            total_trades = sum(r['metrics']['trade_count'] for r in successful)

            self.logger.info(f"\nAVERAGES:")
            self.logger.info(f"  Win Rate: {avg_wr:.2%}")
            self.logger.info(f"  YoY Profit: {avg_yoy:.2f}%")
            self.logger.info(f"  Total Trades: {total_trades}")

            targets_met = avg_wr >= self.target_win_rate and avg_yoy >= self.target_yoy_profit_pct
            if targets_met:
                self.logger.info(f"\n🎉 TARGETS MET ACROSS ALL YEARS!")
            else:
                self.logger.info(f"\n⚠️ Targets not fully met. Continue optimizing...")

    def save_best_config(self, config: dict, name: str):
        """Save the best configuration"""
        os.makedirs('configs/optimization', exist_ok=True)
        config_path = f'configs/optimization/config_{name}.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Also save as the main optimized config
        best_path = 'configs/config_spy_best.yml'
        with open(best_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"\n💾 Best config saved to: {config_path}")
        self.logger.info(f"💾 Also saved to: {best_path}")

    def save_final_report(self, results: List[Dict]):
        """Save final optimization report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"reports/spy_aggressive_opt_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

        # Save JSON results
        with open(f"{report_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create CSV
        rows = []
        for r in results:
            row = {
                'year': r['year'],
                'success': r['success'],
                'error': r.get('error', ''),
            }
            row.update(r.get('metrics', {}))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"{report_dir}/final_results.csv", index=False)

        self.logger.info(f"\n📊 Final report saved to: {report_dir}")


if __name__ == "__main__":
    optimizer = AggressiveSPYOptimizer()
    results = optimizer.run_optimization()

    # Exit with appropriate code
    successful = [r for r in results if r['success'] and r['metrics'].get('trade_count', 0) > 0]
    if successful:
        avg_wr = np.mean([r['metrics']['win_rate'] for r in successful])
        avg_yoy = np.mean([r['metrics']['yoy_profit_pct'] for r in successful])
        targets_met = avg_wr >= optimizer.target_win_rate and avg_yoy >= optimizer.target_yoy_profit_pct
        sys.exit(0 if targets_met else 1)
    else:
        sys.exit(1)
