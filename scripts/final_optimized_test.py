"""
Final Comprehensive Test of Optimized SPY Strategy
Tests optimized parameters across all years 2020-2025
Validates: >70% WR, >50% YoY profit, <10% max drawdown
"""
import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.wfa import WalkForwardAnalyzer
from src.backtester import Backtester


class FinalOptimizedTest:
    """Comprehensive validation of optimized SPY strategy"""

    def __init__(self):
        self.logger = setup_logging()
        self.test_years = [2020, 2021, 2022, 2023, 2024, 2025]
        self.target_win_rate = 0.70
        self.target_yoy_profit = 50.0
        self.max_drawdown_limit = 10.0

    def run_final_test(self):
        """Run comprehensive test on all years"""
        self.logger.info("="*80)
        self.logger.info("FINAL OPTIMIZED SPY STRATEGY TEST")
        self.logger.info("="*80)
        self.logger.info("Configuration: configs/config_spy_optimized.yml")
        self.logger.info(f"Target Win Rate: {self.target_win_rate:.0%}")
        self.logger.info(f"Target YoY Profit: {self.target_yoy_profit:.0f}%")
        self.logger.info(f"Max Drawdown Limit: {self.max_drawdown_limit:.0f}%")
        self.logger.info("="*80 + "\n")

        config = load_config('configs/config_spy_optimized.yml')

        # Display key parameters
        self.logger.info("KEY STRATEGY PARAMETERS:")
        self.logger.info(f"  Risk per trade: {config['strategy']['risk_percent']}%")
        self.logger.info(f"  ML Threshold: {config['strategy']['model_threshold']}")
        self.logger.info(f"  Take Profit: {config['strategy']['tp_atr_mult']}x ATR")
        self.logger.info(f"  Stop Loss: {config['strategy']['sl_atr_mult']}x ATR")
        self.logger.info(f"  Risk:Reward: {config['strategy']['tp_atr_mult']}:1")
        self.logger.info("")

        results = []
        for year in self.test_years:
            end_date = datetime.now().strftime("%Y-%m-%d") if year == 2025 else f"{year}-12-31"
            start_date = f"{year}-01-01"

            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"TESTING YEAR: {year}")
            self.logger.info(f"Period: {start_date} to {end_date}")
            self.logger.info(f"{'='*70}")

            result = self.test_single_year(config, year, start_date, end_date)
            results.append(result)

            # Print year result
            self.print_year_result(result)

        # Generate final summary
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL RESULTS SUMMARY")
        self.logger.info("="*80)
        self.generate_summary(results)

        # Save report
        self.save_report(results, config)

        return results

    def test_single_year(self, config: dict, year: int, start_date: str, end_date: str) -> Dict:
        """Test strategy for a single year"""
        result = {
            'year': year,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'error': None,
            'metrics': {}
        }

        try:
            # Load data
            self.logger.info("1. Loading data from Alpaca...")
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            base_tf = config['data']['timeframe_base']
            df_base = adapter.load_data(timeframe_suffix=base_tf)

            if df_base.empty:
                raise ValueError(f"No data available for {year}")

            self.logger.info(f"   ✓ {len(df_base)} bars loaded")

            # Load higher timeframes
            df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)

            # Feature engineering
            self.logger.info("2. Engineering features...")
            features = FeatureEngineer(config)
            df_base = features.calculate_technical_features(df_base)
            self.logger.info("   ✓ Features calculated")

            # Generate candidates
            self.logger.info("3. Generating SMC candidates...")
            structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
            fvgs = features.detect_fvgs(df_base)
            obs = features.detect_obs(df_base)
            self.logger.info(f"   ✓ {len(fvgs)} FVGs, {len(obs)} OBs detected")

            entry_gen = EntryGenerator(config, structure, features)
            candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                self.logger.warning(f"   ⚠ No candidates generated")
                result['metrics']['total_candidates'] = 0
                result['success'] = True
                return result

            self.logger.info(f"   ✓ {len(candidates)} candidates generated")
            result['metrics']['total_candidates'] = len(candidates)

            # Run WFA
            self.logger.info("4. Running Walk-Forward Analysis...")
            wfa = WalkForwardAnalyzer(config)
            train_years = config.get('wfa', {}).get('train_years', 3)
            wfa_start = f"{year - train_years}-01-01"

            # Load WFA training data
            adapter_wfa = DataAdapter(config, start_date=wfa_start, end_date=end_date)
            df_wfa = adapter_wfa.load_data(timeframe_suffix=base_tf)
            df_wfa = features.calculate_technical_features(df_wfa)

            df_h4_wfa = adapter_wfa.load_h4_data(start_date=wfa_start, end_date=end_date)
            df_d1_wfa = adapter_wfa.load_daily_data(start_date=wfa_start, end_date=end_date)
            structure_wfa = StructureAnalyzer(df_h4_wfa, config, daily_df=df_d1_wfa)
            fvgs_wfa = features.detect_fvgs(df_wfa)
            obs_wfa = features.detect_obs(df_wfa)
            entry_gen_wfa = EntryGenerator(config, structure_wfa, features)
            candidates_wfa = entry_gen_wfa.generate_candidates(df_wfa, df_h4_wfa, fvgs_wfa, obs_wfa)

            if candidates_wfa.empty:
                self.logger.warning(f"   ⚠ No WFA candidates")
                result['success'] = True
                return result

            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates_wfa,
                start_date,
                end_date
            )

            if scored_candidates.empty:
                self.logger.warning(f"   ⚠ No scored candidates from WFA")
                result['success'] = True
                return result

            self.logger.info(f"   ✓ {len(scored_candidates)} trades scored across {len(fold_summaries)} folds")

            # Backtest
            self.logger.info("5. Running backtest...")
            backtester = Backtester(config)
            history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])

            if trades.empty:
                self.logger.warning("   ⚠ No trades executed")
                result['metrics']['trade_count'] = 0
                result['success'] = True
                return result

            # Calculate metrics
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
                'largest_win': float(wins['net_pnl'].max()) if len(wins) > 0 else 0,
                'largest_loss': float(losses['net_pnl'].min()) if len(losses) > 0 else 0,
                'profit_factor': abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf'),
                'avg_trade': total_pnl / len(trades) if len(trades) > 0 else 0,
            }

            # Calculate drawdown
            if not history.empty and 'equity' in history.columns:
                equity = history['equity'].values
                running_max = pd.Series(equity).expanding().max()
                drawdowns = (running_max - equity) / running_max * 100
                metrics['max_drawdown_pct'] = float(drawdowns.max())
                metrics['avg_drawdown_pct'] = float(drawdowns[drawdowns > 0].mean()) if (drawdowns > 0).any() else 0

            result['metrics'].update(metrics)
            result['success'] = True

            self.logger.info("   ✓ Backtest complete")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"   ✗ Error: {e}")
            self.logger.debug(traceback.format_exc())

        return result

    def print_year_result(self, result: Dict):
        """Print detailed result for a single year"""
        if not result['success'] or result['metrics'].get('trade_count', 0) == 0:
            self.logger.info(f"\n❌ {result['year']}: No trades or error")
            if result.get('error'):
                self.logger.info(f"   Error: {result['error']}")
            return

        m = result['metrics']
        wr = m['win_rate']
        yoy = m['yoy_profit_pct']
        dd = m.get('max_drawdown_pct', 0)

        # Check if targets met
        wr_met = "✅" if wr >= self.target_win_rate else "⚠️"
        yoy_met = "✅" if yoy >= self.target_yoy_profit else "⚠️"
        dd_met = "✅" if dd <= self.max_drawdown_limit else "⚠️"

        self.logger.info(f"\n{'─'*70}")
        self.logger.info(f"  Trades: {m['trade_count']} ({m['win_count']}W / {m['loss_count']}L)")
        self.logger.info(f"  {wr_met} Win Rate: {wr:.2%} (target: {self.target_win_rate:.0%})")
        self.logger.info(f"  {yoy_met} YoY Profit: {yoy:.2f}% (target: {self.target_yoy_profit:.0f}%)")
        self.logger.info(f"  {dd_met} Max Drawdown: {dd:.2f}% (limit: {self.max_drawdown_limit:.0f}%)")
        self.logger.info(f"  Net Profit: ${m['net_profit']:.2f}")
        self.logger.info(f"  Profit Factor: {m['profit_factor']:.2f}")
        self.logger.info(f"  Avg Win: ${m['avg_win']:.2f} | Avg Loss: ${m['avg_loss']:.2f}")
        self.logger.info(f"  Avg per Trade: ${m['avg_trade']:.2f}")
        self.logger.info(f"{'─'*70}")

    def generate_summary(self, results: List[Dict]):
        """Generate comprehensive summary"""
        successful = [r for r in results if r['success'] and r['metrics'].get('trade_count', 0) > 0]

        if not successful:
            self.logger.info("❌ No successful test results")
            return

        # Calculate aggregates
        total_trades = sum(r['metrics']['trade_count'] for r in successful)
        total_wins = sum(r['metrics']['win_count'] for r in successful)
        total_losses = sum(r['metrics']['loss_count'] for r in successful)

        avg_wr = np.mean([r['metrics']['win_rate'] for r in successful])
        median_wr = np.median([r['metrics']['win_rate'] for r in successful])
        avg_yoy = np.mean([r['metrics']['yoy_profit_pct'] for r in successful])
        median_yoy = np.median([r['metrics']['yoy_profit_pct'] for r in successful])
        avg_dd = np.mean([r['metrics'].get('max_drawdown_pct', 0) for r in successful])
        max_dd = max([r['metrics'].get('max_drawdown_pct', 0) for r in successful])

        # Year-by-year table
        self.logger.info("\nYEAR-BY-YEAR BREAKDOWN:")
        self.logger.info("-"*90)
        self.logger.info(f"{'Year':<8} {'Trades':<10} {'Win Rate':<12} {'YoY Profit':<15} {'Max DD':<12} {'Status'}")
        self.logger.info("-"*90)

        for r in results:
            if r['success'] and r['metrics'].get('trade_count', 0) > 0:
                m = r['metrics']
                wr_check = "✅" if m['win_rate'] >= self.target_win_rate else "⚠️"
                yoy_check = "✅" if m['yoy_profit_pct'] >= self.target_yoy_profit else "⚠️"
                dd_check = "✅" if m.get('max_drawdown_pct', 0) <= self.max_drawdown_limit else "⚠️"

                status = "✅ PASS" if all([
                    m['win_rate'] >= self.target_win_rate,
                    m['yoy_profit_pct'] >= self.target_yoy_profit,
                    m.get('max_drawdown_pct', 0) <= self.max_drawdown_limit
                ]) else "⚠️ PARTIAL"

                self.logger.info(
                    f"{r['year']:<8} {m['trade_count']:<10} "
                    f"{m['win_rate']:>10.2%}  {m['yoy_profit_pct']:>12.2f}%  "
                    f"{m.get('max_drawdown_pct', 0):>9.2f}%  {status}"
                )
            else:
                self.logger.info(f"{r['year']:<8} {'No trades':<10}")

        self.logger.info("-"*90)

        # Overall statistics
        self.logger.info(f"\nOVERALL STATISTICS:")
        self.logger.info(f"  Years Tested: {len(results)}")
        self.logger.info(f"  Years with Trades: {len(successful)}")
        self.logger.info(f"  Total Trades: {total_trades} ({total_wins}W / {total_losses}L)")
        self.logger.info(f"")
        self.logger.info(f"  Average Win Rate: {avg_wr:.2%} (median: {median_wr:.2%})")
        self.logger.info(f"  Average YoY Profit: {avg_yoy:.2f}% (median: {median_yoy:.2f}%)")
        self.logger.info(f"  Average Max DD: {avg_dd:.2f}%")
        self.logger.info(f"  Worst Max DD: {max_dd:.2f}%")

        # Target assessment
        self.logger.info(f"\nTARGET ASSESSMENT:")
        wr_status = "✅ MET" if avg_wr >= self.target_win_rate else f"⚠️ MISSED ({avg_wr:.2%} vs {self.target_win_rate:.0%})"
        yoy_status = "✅ MET" if avg_yoy >= self.target_yoy_profit else f"⚠️ MISSED ({avg_yoy:.1f}% vs {self.target_yoy_profit:.0f}%)"
        dd_status = "✅ MET" if max_dd <= self.max_drawdown_limit else f"⚠️ EXCEEDED ({max_dd:.2f}% vs {self.max_drawdown_limit:.0f}%)"

        self.logger.info(f"  Win Rate Target (>70%): {wr_status}")
        self.logger.info(f"  YoY Profit Target (>50%): {yoy_status}")
        self.logger.info(f"  Max DD Limit (<10%): {dd_status}")

        if avg_wr >= self.target_win_rate and avg_yoy >= self.target_yoy_profit and max_dd <= self.max_drawdown_limit:
            self.logger.info(f"\n🎉 ALL TARGETS MET! Strategy is ready for deployment!")
        else:
            self.logger.info(f"\n⚠️ Some targets not met. Further optimization may be needed.")

    def save_report(self, results: List[Dict], config: dict):
        """Save comprehensive report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"reports/final_optimized_test_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

        # Save JSON
        with open(f"{report_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV
        rows = []
        for r in results:
            row = {'year': r['year'], 'success': r['success'], 'error': r.get('error', '')}
            row.update(r.get('metrics', {}))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"{report_dir}/results.csv", index=False)

        # Save config
        import yaml
        with open(f"{report_dir}/config_used.yml", 'w') as f:
            yaml.dump(config, f)

        self.logger.info(f"\n📊 Report saved to: {report_dir}")


if __name__ == "__main__":
    tester = FinalOptimizedTest()
    results = tester.run_final_test()

    # Exit code based on targets
    successful = [r for r in results if r['success'] and r['metrics'].get('trade_count', 0) > 0]
    if successful:
        avg_wr = np.mean([r['metrics']['win_rate'] for r in successful])
        avg_yoy = np.mean([r['metrics']['yoy_profit_pct'] for r in successful])
        max_dd = max([r['metrics'].get('max_drawdown_pct', 0) for r in successful])

        targets_met = (avg_wr >= 0.70 and avg_yoy >= 50.0 and max_dd <= 10.0)
        sys.exit(0 if targets_met else 1)
    else:
        sys.exit(1)
