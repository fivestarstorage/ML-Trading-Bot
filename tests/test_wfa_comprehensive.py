"""
Comprehensive Walk-Forward Analysis Testing Suite
Tests WFA for SPY (SMC strategy) and BTC (Crypto Momentum) for years 2020-2025
"""
import sys
import os
import pandas as pd
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.data_adapter import DataAdapter
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
try:
    from src.crypto_strategy import CryptoMomentumEntryGenerator
    HAS_CRYPTO_STRATEGY = True
except ImportError:
    HAS_CRYPTO_STRATEGY = False
    CryptoMomentumEntryGenerator = None
from src.ml_model import MLModel
from src.wfa import WalkForwardAnalyzer
from src.backtester import Backtester


class WFATestRunner:
    """Comprehensive WFA testing for multiple years and strategies"""

    def __init__(self):
        self.logger = setup_logging()
        self.test_results = []
        self.years_to_test = [2020, 2021, 2022, 2023, 2024, 2025]

    def run_all_tests(self):
        """Run all WFA tests for both SPY and BTC"""
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE WFA TESTING SUITE")
        self.logger.info("="*80)

        # Test SPY (SMC Strategy)
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING SPY WITH SMC STRATEGY")
        self.logger.info("="*80 + "\n")
        spy_results = self.test_strategy_all_years(
            config_path="configs/config_spy.yml",
            strategy_name="SPY-SMC",
            strategy_type="smc"
        )

        # Test BTC (skip if crypto strategy not available)
        if HAS_CRYPTO_STRATEGY:
            self.logger.info("\n" + "="*80)
            self.logger.info("TESTING BTC WITH CRYPTO MOMENTUM STRATEGY")
            self.logger.info("="*80 + "\n")
            btc_results = self.test_strategy_all_years(
                config_path="configs/config_btc_optimized.yml",
                strategy_name="BTC-CryptoMomo",
                strategy_type="crypto_momo"
            )
        else:
            self.logger.warning("\n⚠️  Skipping BTC WFA tests - crypto_strategy module not found")
            self.logger.warning("   Testing BTC with standard SMC strategy instead")
            btc_results = self.test_strategy_all_years(
                config_path="configs/config_btc_optimized.yml",
                strategy_name="BTC-SMC",
                strategy_type="smc"
            )

        # Generate comprehensive report
        self.generate_report(spy_results, btc_results)

        return spy_results, btc_results

    def test_strategy_all_years(
        self,
        config_path: str,
        strategy_name: str,
        strategy_type: str
    ) -> List[Dict]:
        """Test a strategy across all specified years"""
        results = []
        config = load_config(config_path)

        for year in self.years_to_test:
            # Skip 2025 if we don't have full year data
            if year == 2025:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = f"{year}-12-31"

            start_date = f"{year}-01-01"

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing {strategy_name} for year {year}")
            self.logger.info(f"Period: {start_date} to {end_date}")
            self.logger.info(f"{'='*60}")

            result = self.test_single_year(
                config=config,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                year=year,
                start_date=start_date,
                end_date=end_date
            )
            results.append(result)

        return results

    def test_single_year(
        self,
        config: dict,
        strategy_name: str,
        strategy_type: str,
        year: int,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Test WFA for a single year"""
        result = {
            'strategy': strategy_name,
            'year': year,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'error': None,
            'metrics': {}
        }

        try:
            # Load data
            self.logger.info("1. Loading data...")
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            base_tf = config['data']['timeframe_base']
            df_base = adapter.load_data(timeframe_suffix=base_tf)

            if df_base.empty:
                raise ValueError(f"No data available for {year}")

            self.logger.info(f"   ✓ Loaded {len(df_base)} bars")
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
            self.logger.info("4. Generating candidates...")
            if strategy_type == 'crypto_momo' and HAS_CRYPTO_STRATEGY:
                entry_gen = CryptoMomentumEntryGenerator(config)
                candidates = entry_gen.generate_candidates(df_base)
            else:  # SMC (or fallback for crypto if module missing)
                structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
                fvgs = features.detect_fvgs(df_base)
                obs = features.detect_obs(df_base)
                self.logger.info(f"   ✓ Detected {len(fvgs)} FVGs, {len(obs)} OBs")
                entry_gen = EntryGenerator(config, structure, features)
                candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

            if candidates.empty:
                raise ValueError(f"No candidates generated for {year}")

            self.logger.info(f"   ✓ Generated {len(candidates)} candidates")
            result['metrics']['total_candidates'] = len(candidates)

            # Run WFA
            self.logger.info("5. Running Walk-Forward Analysis...")
            wfa = WalkForwardAnalyzer(config)

            # For WFA we need training data from prior years
            wfa_start = f"{year - config['wfa']['train_years']}-01-01"

            # Load extended data for WFA training
            adapter_wfa = DataAdapter(config, start_date=wfa_start, end_date=end_date)
            df_wfa = adapter_wfa.load_data(timeframe_suffix=base_tf)
            df_wfa = features.calculate_technical_features(df_wfa)

            # Generate candidates for full WFA period
            if strategy_type == 'crypto_momo' and HAS_CRYPTO_STRATEGY:
                candidates_wfa = entry_gen.generate_candidates(df_wfa)
            else:
                df_h4_wfa = adapter_wfa.load_h4_data(start_date=wfa_start, end_date=end_date)
                df_d1_wfa = adapter_wfa.load_daily_data(start_date=wfa_start, end_date=end_date)
                structure_wfa = StructureAnalyzer(df_h4_wfa, config, daily_df=df_d1_wfa)
                fvgs_wfa = features.detect_fvgs(df_wfa)
                obs_wfa = features.detect_obs(df_wfa)
                entry_gen_wfa = EntryGenerator(config, structure_wfa, features)
                candidates_wfa = entry_gen_wfa.generate_candidates(df_wfa, df_h4_wfa, fvgs_wfa, obs_wfa)

            scored_candidates, fold_summaries = wfa.sequential_walk(
                candidates_wfa,
                start_date,
                end_date
            )

            if scored_candidates.empty:
                raise ValueError(f"WFA produced no scored candidates for {year}")

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
            else:
                wins = trades[trades['net_pnl'] > 0]
                losses = trades[trades['net_pnl'] <= 0]

                metrics = {
                    'trade_count': len(trades),
                    'win_count': len(wins),
                    'loss_count': len(losses),
                    'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0,
                    'net_profit': float(trades['net_pnl'].sum()),
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
                self.logger.info(f"      Net Profit: ${metrics['net_profit']:.2f}")
                self.logger.info(f"      Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                if 'max_drawdown_pct' in metrics:
                    self.logger.info(f"      Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

            result['success'] = True
            self.logger.info(f"\n✅ {strategy_name} {year} - TEST PASSED")

        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.logger.error(f"\n❌ {strategy_name} {year} - TEST FAILED")
            self.logger.error(f"   Error: {e}")
            self.logger.debug(traceback.format_exc())

        return result

    def generate_report(self, spy_results: List[Dict], btc_results: List[Dict]):
        """Generate comprehensive test report"""
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPREHENSIVE TEST REPORT")
        self.logger.info("="*80 + "\n")

        # Summary table
        report_lines = []
        report_lines.append("\nSPY (SMC Strategy) Results:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Year':<8} {'Status':<12} {'Trades':<10} {'Win Rate':<12} {'Net Profit':<15} {'PF':<8}")
        report_lines.append("-" * 80)

        for result in spy_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            trades = result['metrics'].get('trade_count', 0)
            win_rate = result['metrics'].get('win_rate', 0)
            net_profit = result['metrics'].get('net_profit', 0)
            pf = result['metrics'].get('profit_factor', 0)

            if result['success']:
                report_lines.append(
                    f"{result['year']:<8} {status:<12} {trades:<10} "
                    f"{win_rate:>10.2%}  ${net_profit:>12.2f}  {pf:>6.2f}"
                )
            else:
                report_lines.append(f"{result['year']:<8} {status:<12} Error: {result['error']}")

        report_lines.append("\n\nBTC (Crypto Momentum) Results:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Year':<8} {'Status':<12} {'Trades':<10} {'Win Rate':<12} {'Net Profit':<15} {'PF':<8}")
        report_lines.append("-" * 80)

        for result in btc_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            trades = result['metrics'].get('trade_count', 0)
            win_rate = result['metrics'].get('win_rate', 0)
            net_profit = result['metrics'].get('net_profit', 0)
            pf = result['metrics'].get('profit_factor', 0)

            if result['success']:
                report_lines.append(
                    f"{result['year']:<8} {status:<12} {trades:<10} "
                    f"{win_rate:>10.2%}  ${net_profit:>12.2f}  {pf:>6.2f}"
                )
            else:
                report_lines.append(f"{result['year']:<8} {status:<12} Error: {result['error']}")

        # Overall summary
        spy_passed = sum(1 for r in spy_results if r['success'])
        btc_passed = sum(1 for r in btc_results if r['success'])
        total_tests = len(spy_results) + len(btc_results)
        total_passed = spy_passed + btc_passed

        report_lines.append("\n" + "="*80)
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("="*80)
        report_lines.append(f"SPY Tests:  {spy_passed}/{len(spy_results)} passed")
        report_lines.append(f"BTC Tests:  {btc_passed}/{len(btc_results)} passed")
        report_lines.append(f"Total:      {total_passed}/{total_tests} passed")
        report_lines.append("="*80 + "\n")

        report_text = "\n".join(report_lines)

        # Print to console
        print(report_text)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"tests/wfa_test_report_{timestamp}.txt"
        os.makedirs("tests", exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)

        self.logger.info(f"Report saved to: {report_path}")

        # Save detailed results to CSV
        results_df = pd.DataFrame(spy_results + btc_results)
        csv_path = f"tests/wfa_test_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Detailed results saved to: {csv_path}")

        return total_passed == total_tests


if __name__ == "__main__":
    runner = WFATestRunner()
    spy_results, btc_results = runner.run_all_tests()

    # Exit with appropriate code
    all_passed = all(r['success'] for r in spy_results + btc_results)
    sys.exit(0 if all_passed else 1)
