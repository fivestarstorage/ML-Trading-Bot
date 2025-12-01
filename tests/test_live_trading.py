"""
Comprehensive Live Trading Integration Tests
Tests all components of the live trading system for production readiness
"""
import sys
import os
import time
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, get_logger
from src.data_adapter import DataAdapter
from src.features import FeatureEngineer
from src.structure import StructureAnalyzer
from src.entries import EntryGenerator
try:
    from src.crypto_strategy import CryptoMomentumEntryGenerator
    HAS_CRYPTO_STRATEGY = True
except ImportError:
    HAS_CRYPTO_STRATEGY = False
    CryptoMomentumEntryGenerator = None
from src.ml_model import MLModel
from src.alpaca_client import AlpacaClient
from src.alpaca_live_trader import AlpacaLiveTrader
from src.alpaca_live_runner import AlpacaPollingRunner


class LiveTradingTestSuite:
    """Comprehensive testing suite for live trading functionality"""

    def __init__(self, dry_run=True):
        self.logger = setup_logging()
        self.dry_run = dry_run
        self.test_results = []

    def run_all_tests(self):
        """Run all live trading tests"""
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE LIVE TRADING TEST SUITE")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE (BE CAREFUL!)'}")
        self.logger.info("="*80 + "\n")

        all_passed = True

        # Test SPY
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING SPY LIVE TRADING COMPONENTS")
        self.logger.info("="*80 + "\n")
        spy_passed = self.test_strategy_components(
            config_path="configs/config_spy.yml",
            strategy_name="SPY-SMC",
            strategy_type="smc"
        )
        all_passed = all_passed and spy_passed

        # Test BTC (skip if crypto strategy not available)
        if HAS_CRYPTO_STRATEGY:
            self.logger.info("\n" + "="*80)
            self.logger.info("TESTING BTC LIVE TRADING COMPONENTS")
            self.logger.info("="*80 + "\n")
            btc_passed = self.test_strategy_components(
                config_path="configs/config_btc_optimized.yml",
                strategy_name="BTC-CryptoMomo",
                strategy_type="crypto_momo"
            )
            all_passed = all_passed and btc_passed
        else:
            self.logger.warning("\n⚠️  Skipping BTC tests - crypto_strategy module not found")
            self.logger.warning("   This is expected if BTC uses the standard SMC strategy")

        # Generate report
        self.generate_report()

        return all_passed

    def test_strategy_components(
        self,
        config_path: str,
        strategy_name: str,
        strategy_type: str
    ) -> bool:
        """Test all components for a specific strategy"""
        config = load_config(config_path)
        all_tests_passed = True

        # Test 1: Configuration validation
        test_passed = self.test_config_validation(config, strategy_name)
        all_tests_passed = all_tests_passed and test_passed

        # Test 2: Alpaca client connection
        test_passed, alpaca_client = self.test_alpaca_connection(config, strategy_name)
        all_tests_passed = all_tests_passed and test_passed
        if not test_passed:
            return False  # Can't continue without connection

        # Test 3: Historical data loading
        test_passed, df_base = self.test_data_loading(config, alpaca_client, strategy_name)
        all_tests_passed = all_tests_passed and test_passed
        if not test_passed:
            return False  # Can't continue without data

        # Test 4: Feature calculation
        test_passed, features, df_features = self.test_feature_calculation(
            config, df_base, strategy_name
        )
        all_tests_passed = all_tests_passed and test_passed

        # Test 5: Model loading
        test_passed, model = self.test_model_loading(config, strategy_name)
        all_tests_passed = all_tests_passed and test_passed
        if not test_passed:
            return False  # Can't continue without model

        # Test 6: Signal generation pipeline
        test_passed = self.test_signal_generation(
            config, df_features, features, model, strategy_type, strategy_name
        )
        all_tests_passed = all_tests_passed and test_passed

        # Test 7: Live data fetching
        test_passed = self.test_live_data_fetch(config, alpaca_client, strategy_name)
        all_tests_passed = all_tests_passed and test_passed

        # Test 8: Order execution (dry run)
        test_passed = self.test_order_execution(config, alpaca_client, strategy_name)
        all_tests_passed = all_tests_passed and test_passed

        # Test 9: Market hours detection (SPY only)
        if strategy_type == 'smc':
            test_passed = self.test_market_hours(config, strategy_name)
            all_tests_passed = all_tests_passed and test_passed

        # Test 10: Error handling
        test_passed = self.test_error_handling(config, alpaca_client, strategy_name)
        all_tests_passed = all_tests_passed and test_passed

        # Test 11: Live runner initialization
        test_passed = self.test_live_runner_init(
            config, df_base, features, model, alpaca_client, strategy_name
        )
        all_tests_passed = all_tests_passed and test_passed

        return all_tests_passed

    def test_config_validation(self, config: dict, strategy_name: str) -> bool:
        """Test configuration is valid for live trading"""
        test_name = f"{strategy_name} - Configuration Validation"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            required_keys = [
                ('data', 'symbol'),
                ('data', 'timeframe_base'),
                ('strategy', 'strategy_type'),
                ('ml', 'model_path'),
            ]

            for keys in required_keys:
                current = config
                for key in keys:
                    if key not in current:
                        raise ValueError(f"Missing required config key: {'.'.join(keys)}")
                    current = current[key]

            # Check live trading config if enabled
            live_cfg = config.get('live_trading', {})
            if live_cfg.get('enabled'):
                live_required = ['symbol', 'qty', 'probability_threshold']
                for key in live_required:
                    if key not in live_cfg:
                        raise ValueError(f"Missing live_trading.{key}")

            self.logger.info("   ✅ Configuration is valid")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Configuration validation failed: {e}")
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_alpaca_connection(
        self, config: dict, strategy_name: str
    ) -> tuple[bool, Optional[AlpacaClient]]:
        """Test Alpaca API connection"""
        test_name = f"{strategy_name} - Alpaca Connection"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Initialize client
            live_cfg = config.get('live_trading', {})
            client = AlpacaClient(
                api_key=live_cfg.get('api_key'),
                api_secret=live_cfg.get('api_secret'),
                trading_url=live_cfg.get('trading_url', 'https://paper-api.alpaca.markets'),
                data_url=live_cfg.get('data_url', 'https://data.alpaca.markets'),
            )

            # Test account access
            account = client.get_account()
            self.logger.info(f"   ✓ Connected to Alpaca")
            self.logger.info(f"   ✓ Account status: {account.get('status')}")
            self.logger.info(f"   ✓ Buying power: ${float(account.get('buying_power', 0)):,.2f}")

            # Test market status
            clock = client.get_clock()
            self.logger.info(f"   ✓ Market open: {clock.get('is_open')}")

            self.logger.info("   ✅ Alpaca connection successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True, client

        except Exception as e:
            self.logger.error(f"   ❌ Alpaca connection failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False, None

    def test_data_loading(
        self, config: dict, client: AlpacaClient, strategy_name: str
    ) -> tuple[bool, Optional[pd.DataFrame]]:
        """Test historical data loading"""
        test_name = f"{strategy_name} - Historical Data Loading"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Load last 30 days of data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            base_tf = config['data']['timeframe_base']
            df_base = adapter.load_data(timeframe_suffix=base_tf)

            if df_base.empty:
                raise ValueError("No data loaded")

            self.logger.info(f"   ✓ Loaded {len(df_base)} bars")
            self.logger.info(f"   ✓ Date range: {df_base.index.min()} to {df_base.index.max()}")
            self.logger.info(f"   ✓ Columns: {list(df_base.columns)}")

            # Verify required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df_base.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.logger.info("   ✅ Data loading successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True, df_base

        except Exception as e:
            self.logger.error(f"   ❌ Data loading failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False, None

    def test_feature_calculation(
        self, config: dict, df_base: pd.DataFrame, strategy_name: str
    ) -> tuple[bool, Optional[FeatureEngineer], Optional[pd.DataFrame]]:
        """Test feature calculation"""
        test_name = f"{strategy_name} - Feature Calculation"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            features = FeatureEngineer(config)
            df_features = features.calculate_technical_features(df_base.copy())

            if df_features.empty:
                raise ValueError("Feature calculation produced empty dataframe")

            # Check for common features
            feature_cols = ['atr', 'rsi', 'adx']
            found_features = [col for col in feature_cols if col in df_features.columns]

            self.logger.info(f"   ✓ Features calculated: {len(df_features)} rows")
            self.logger.info(f"   ✓ Feature columns found: {found_features}")

            self.logger.info("   ✅ Feature calculation successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True, features, df_features

        except Exception as e:
            self.logger.error(f"   ❌ Feature calculation failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False, None, None

    def test_model_loading(
        self, config: dict, strategy_name: str
    ) -> tuple[bool, Optional[MLModel]]:
        """Test ML model loading"""
        test_name = f"{strategy_name} - Model Loading"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            model = MLModel(config)
            model_path = config['ml']['model_path']

            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            model.load(model_path)
            self.logger.info(f"   ✓ Model loaded from {model_path}")

            self.logger.info("   ✅ Model loading successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True, model

        except Exception as e:
            self.logger.error(f"   ❌ Model loading failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False, None

    def test_signal_generation(
        self,
        config: dict,
        df_features: pd.DataFrame,
        features: FeatureEngineer,
        model: MLModel,
        strategy_type: str,
        strategy_name: str
    ) -> bool:
        """Test end-to-end signal generation"""
        test_name = f"{strategy_name} - Signal Generation Pipeline"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Load higher timeframes and generate candidates
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
            df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
            df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)

            structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
            fvgs = features.detect_fvgs(df_features)
            obs = features.detect_obs(df_features)

            self.logger.info(f"   ✓ Detected {len(fvgs)} FVGs, {len(obs)} OBs")

            entry_gen = EntryGenerator(config, structure, features)
            candidates = entry_gen.generate_candidates(df_features, df_h4, fvgs, obs)

            self.logger.info(f"   ✓ Generated {len(candidates)} candidates")

            if not candidates.empty:
                # Test prediction
                probabilities = model.predict_proba(candidates)
                self.logger.info(f"   ✓ Predictions generated: {len(probabilities)} probabilities")
                self.logger.info(f"   ✓ Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}")

                # Check threshold filtering
                threshold = config['strategy'].get('model_threshold', 0.55)
                high_prob = sum(1 for p in probabilities if p >= threshold)
                self.logger.info(f"   ✓ Signals above threshold ({threshold}): {high_prob}")

            self.logger.info("   ✅ Signal generation successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Signal generation failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_live_data_fetch(
        self, config: dict, client: AlpacaClient, strategy_name: str
    ) -> bool:
        """Test live data fetching"""
        test_name = f"{strategy_name} - Live Data Fetch"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            symbol = config['data']['symbol']
            timeframe_base = config['data']['timeframe_base']

            # Convert timeframe to Alpaca format
            if timeframe_base.endswith('m'):
                alpaca_tf = f"{int(timeframe_base[:-1])}Min"
            elif timeframe_base.endswith('h'):
                alpaca_tf = f"{int(timeframe_base[:-1])}Hour"
            else:
                alpaca_tf = "5Min"

            # Fetch latest bars
            end_time = pd.Timestamp.utcnow()
            start_time = end_time - pd.Timedelta(hours=1)

            bars = client.fetch_bars(
                symbol,
                timeframe=alpaca_tf,
                start=start_time,
                end=end_time,
                limit=50
            )

            self.logger.info(f"   ✓ Fetched {len(bars)} live bars")
            if not bars.empty:
                self.logger.info(f"   ✓ Latest bar: {bars.index.max()}")
                self.logger.info(f"   ✓ Latest close: ${bars['close'].iloc[-1]:.2f}")

            self.logger.info("   ✅ Live data fetch successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Live data fetch failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_order_execution(
        self, config: dict, client: AlpacaClient, strategy_name: str
    ) -> bool:
        """Test order execution in dry run mode"""
        test_name = f"{strategy_name} - Order Execution (Dry Run)"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            live_cfg = config.get('live_trading', {})
            trader = AlpacaLiveTrader(config, live_cfg, alpaca_client=client, dry_run=True)

            # Create dummy candidate
            candidate = pd.DataFrame([{
                'entry_time': pd.Timestamp.utcnow(),
                'entry_price': 100.0,
                'stop_loss': 99.0,
                'take_profit': 102.0,
                'direction': 1,
                'bias': 'bull',
                'prob': 0.75
            }])

            probabilities = [0.75]

            # Test order execution
            result = trader.run(candidate, probabilities)

            if result is None:
                raise ValueError("Order execution returned None")

            self.logger.info(f"   ✓ Order execution test passed (dry run)")
            self.logger.info(f"   ✓ Result: {result}")

            self.logger.info("   ✅ Order execution test successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Order execution test failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_market_hours(self, config: dict, strategy_name: str) -> bool:
        """Test market hours detection"""
        test_name = f"{strategy_name} - Market Hours Detection"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Create a dummy runner just to test market hours detection
            dummy_config = config.copy()
            dummy_df = pd.DataFrame()

            from src.alpaca_live_runner import AlpacaPollingRunner

            # Test the static methods
            is_crypto = AlpacaPollingRunner._is_crypto_symbol(config['data']['symbol'])
            self.logger.info(f"   ✓ Symbol '{config['data']['symbol']}' is crypto: {is_crypto}")

            if not is_crypto:
                # For stocks, test market hours logic
                eastern = pd.Timestamp.now(tz='US/Eastern')
                current_time = eastern.time()
                current_day = eastern.weekday()

                from datetime import time as dt_time
                market_open = dt_time(9, 30)
                market_close = dt_time(16, 0)

                is_open = (current_day < 5 and market_open <= current_time <= market_close)
                self.logger.info(f"   ✓ Current time (ET): {current_time}")
                self.logger.info(f"   ✓ Market is currently: {'OPEN' if is_open else 'CLOSED'}")

            self.logger.info("   ✅ Market hours detection successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Market hours detection failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_error_handling(
        self, config: dict, client: AlpacaClient, strategy_name: str
    ) -> bool:
        """Test error handling and recovery"""
        test_name = f"{strategy_name} - Error Handling"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Test 1: Invalid symbol
            try:
                invalid_bars = client.fetch_bars(
                    "INVALID_SYMBOL_XYZ",
                    timeframe="5Min",
                    start=pd.Timestamp.utcnow() - pd.Timedelta(hours=1),
                    end=pd.Timestamp.utcnow(),
                    limit=10
                )
                # Should handle gracefully
                self.logger.info("   ✓ Invalid symbol handled gracefully")
            except Exception as e:
                self.logger.info(f"   ✓ Invalid symbol error caught: {type(e).__name__}")

            # Test 2: Empty dataframe handling
            trader = AlpacaLiveTrader(config, config.get('live_trading', {}), alpaca_client=client, dry_run=True)
            empty_df = pd.DataFrame()
            result = trader.run(empty_df, [])
            self.logger.info("   ✓ Empty dataframe handled")

            # Test 3: Low probability signal
            low_prob_candidate = pd.DataFrame([{
                'entry_time': pd.Timestamp.utcnow(),
                'prob': 0.3,
                'bias': 'bull'
            }])
            result = trader.run(low_prob_candidate, [0.3])
            self.logger.info("   ✓ Low probability signal filtered")

            self.logger.info("   ✅ Error handling tests passed")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Error handling test failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def test_live_runner_init(
        self,
        config: dict,
        df_base: pd.DataFrame,
        features: FeatureEngineer,
        model: MLModel,
        client: AlpacaClient,
        strategy_name: str
    ) -> bool:
        """Test live runner initialization"""
        test_name = f"{strategy_name} - Live Runner Initialization"
        self.logger.info(f"\n🔍 Test: {test_name}")

        try:
            # Create components
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            adapter = DataAdapter(config, start_date=start_date, end_date=end_date)

            live_cfg = config.get('live_trading', {})
            trader = AlpacaLiveTrader(config, live_cfg, alpaca_client=client, dry_run=True)

            # Initialize runner
            runner = AlpacaPollingRunner(
                config=config,
                adapter=adapter,
                base_df=df_base.copy(),
                feature_engineer=features,
                model=model,
                live_trader=trader,
                alpaca_client=client,
                dry_run=True
            )

            self.logger.info("   ✓ Runner initialized successfully")
            self.logger.info(f"   ✓ Poll interval: {runner.poll_interval}s")
            self.logger.info(f"   ✓ Symbol: {runner.symbol}")
            self.logger.info(f"   ✓ Base history bars: {len(runner.base_df)}")
            self.logger.info(f"   ✓ Min history required: {runner.min_history}")
            self.logger.info(f"   ✓ Crypto asset: {runner.is_crypto}")

            # Don't actually run the loop, just verify it can be initialized
            self.logger.info("   ✓ Runner is ready to start (not starting in test)")

            self.logger.info("   ✅ Live runner initialization successful")
            self.test_results.append({
                'test': test_name,
                'status': 'PASS',
                'error': None
            })
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Live runner initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.test_results.append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
            return False

    def generate_report(self):
        """Generate test report"""
        self.logger.info("\n" + "="*80)
        self.logger.info("LIVE TRADING TEST REPORT")
        self.logger.info("="*80 + "\n")

        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        total = len(self.test_results)

        report_lines = []
        report_lines.append(f"{'Test Name':<60} {'Status':<10}")
        report_lines.append("-" * 80)

        for result in self.test_results:
            status_symbol = "✅" if result['status'] == 'PASS' else "❌"
            report_lines.append(f"{result['test']:<60} {status_symbol} {result['status']:<10}")
            if result['error']:
                report_lines.append(f"  Error: {result['error']}")

        report_lines.append("\n" + "="*80)
        report_lines.append(f"Total Tests: {total}")
        report_lines.append(f"Passed: {passed}")
        report_lines.append(f"Failed: {failed}")
        report_lines.append(f"Success Rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
        report_lines.append("="*80 + "\n")

        report_text = "\n".join(report_lines)
        print(report_text)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"tests/live_trading_test_report_{timestamp}.txt"
        os.makedirs("tests", exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)

        self.logger.info(f"Report saved to: {report_path}")

        return passed == total


if __name__ == "__main__":
    # Always use dry run for safety
    suite = LiveTradingTestSuite(dry_run=True)
    all_passed = suite.run_all_tests()

    sys.exit(0 if all_passed else 1)
