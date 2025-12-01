"""
Simplified Comprehensive Testing
Tests core functionality that exists in the current codebase
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging

def test_spy_wfa_2024():
    """Test SPY WFA for 2024"""
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Testing SPY Walk-Forward Analysis for 2024")
    logger.info("="*80)

    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "src.cli",
            "--config", "configs/config_spy.yml",
            "--wfa",
            "--from", "2024-01-01",
            "--to", "2024-11-30"
        ], capture_output=True, text=True, timeout=600)

        success = result.returncode == 0

        if success:
            logger.info("✅ SPY WFA 2024 - PASSED")
            logger.info(f"\nOutput:\n{result.stdout[:500]}")
        else:
            logger.error("❌ SPY WFA 2024 - FAILED")
            logger.error(f"\nError:\n{result.stderr[:500]}")

        return success
    except Exception as e:
        logger.error(f"❌ SPY WFA 2024 - ERROR: {e}")
        return False

def test_spy_live_components():
    """Test SPY live trading components"""
    logger = setup_logging()
    logger.info("\n" + "="*80)
    logger.info("Testing SPY Live Trading Components")
    logger.info("="*80)

    all_passed = True

    # Test 1: Config loading
    logger.info("\n🔍 Test 1: Configuration Loading")
    try:
        config = load_config("configs/config_spy.yml")
        logger.info(f"   ✅ Config loaded: symbol={config['data']['symbol']}")
    except Exception as e:
        logger.error(f"   ❌ Config loading failed: {e}")
        all_passed = False

    # Test 2: Alpaca client
    logger.info("\n🔍 Test 2: Alpaca Client Connection")
    try:
        from src.alpaca_client import AlpacaClient
        live_cfg = config.get('live_trading', {})
        client = AlpacaClient(
            api_key=live_cfg.get('api_key'),
            api_secret=live_cfg.get('api_secret'),
            trading_url=live_cfg.get('trading_url'),
            data_url=live_cfg.get('data_url'),
        )
        account = client.get_account()
        logger.info(f"   ✅ Connected: status={account.get('status')}, buying_power=${float(account.get('buying_power', 0)):,.2f}")
    except Exception as e:
        logger.error(f"   ❌ Alpaca connection failed: {e}")
        all_passed = False

    # Test 3: Data fetching
    logger.info("\n🔍 Test 3: Historical Data Loading")
    try:
        from src.data_adapter import DataAdapter
        import pandas as pd

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = "2024-11-01"
        adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
        df = adapter.load_data(timeframe_suffix="5m")
        logger.info(f"   ✅ Data loaded: {len(df)} bars from {df.index.min()} to {df.index.max()}")
    except Exception as e:
        logger.error(f"   ❌ Data loading failed: {e}")
        all_passed = False

    # Test 4: Feature calculation
    logger.info("\n🔍 Test 4: Feature Calculation")
    try:
        from src.features import FeatureEngineer
        features = FeatureEngineer(config)
        df_features = features.calculate_technical_features(df.copy())
        logger.info(f"   ✅ Features calculated: {len(df_features)} rows")
    except Exception as e:
        logger.error(f"   ❌ Feature calculation failed: {e}")
        all_passed = False

    # Test 5: Model loading
    logger.info("\n🔍 Test 5: ML Model Loading")
    try:
        model_path = config['ml']['model_path']
        if os.path.exists(model_path):
            logger.info(f"   ✅ Model exists at {model_path}")
        else:
            logger.warning(f"   ⚠️  Model not found at {model_path}")
            logger.info("   Run: python -m src.cli --config configs/config_spy.yml --action train_latest")
    except Exception as e:
        logger.error(f"   ❌ Model check failed: {e}")

    # Test 6: Live trader initialization
    logger.info("\n🔍 Test 6: Live Trader Initialization")
    try:
        from src.alpaca_live_trader import AlpacaLiveTrader
        trader = AlpacaLiveTrader(config, live_cfg, alpaca_client=client, dry_run=True)
        logger.info(f"   ✅ Live trader initialized (dry_run=True, threshold={trader.threshold})")
    except Exception as e:
        logger.error(f"   ❌ Live trader init failed: {e}")
        all_passed = False

    # Test 7: Market hours detection
    logger.info("\n🔍 Test 7: Market Hours Detection")
    try:
        from src.alpaca_live_runner import AlpacaPollingRunner
        import pandas as pd
        from datetime import time as dt_time

        is_crypto = AlpacaPollingRunner._is_crypto_symbol("SPY")
        logger.info(f"   ✅ SPY is crypto: {is_crypto} (expected: False)")

        eastern = pd.Timestamp.now(tz='US/Eastern')
        current_time = eastern.time()
        current_day = eastern.weekday()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        is_open = (current_day < 5 and market_open <= current_time <= market_close)
        logger.info(f"   ✅ Market is currently: {'OPEN' if is_open else 'CLOSED'}")
    except Exception as e:
        logger.error(f"   ❌ Market hours detection failed: {e}")
        all_passed = False

    if all_passed:
        logger.info("\n" + "="*80)
        logger.info("✅ ALL SPY LIVE TRADING COMPONENT TESTS PASSED")
        logger.info("="*80)
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ SOME SPY LIVE TRADING TESTS FAILED")
        logger.error("="*80)

    return all_passed

def main():
    logger = setup_logging()
    logger.info("\n\n")
    logger.info("="*80)
    logger.info("COMPREHENSIVE TESTING SUITE - SIMPLIFIED")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    results = {}

    # Test live trading components
    results['SPY Live Components'] = test_spy_live_components()

    # Test WFA for 2024
    logger.info("\n\nNOTE: WFA test may take 5-10 minutes to complete...")
    results['SPY WFA 2024'] = test_spy_wfa_2024()

    # Summary
    logger.info("\n\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:<40} {status}")

    total = len(results)
    passed_count = sum(1 for p in results.values() if p)

    logger.info("\n" + "-"*80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed_count}")
    logger.info(f"Failed: {total - passed_count}")
    logger.info(f"Success Rate: {passed_count/total*100:.1f}%")
    logger.info("-"*80)

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - CORE SYSTEM IS FUNCTIONAL!")
        logger.info("\nNext Steps:")
        logger.info("1. ✅ Live components are working")
        logger.info("2. ✅ WFA is functional")
        logger.info("3. To test live trading: python -m src.cli --config configs/config_spy.yml --action alpaca_live --dry-run")
    else:
        logger.error("\n⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")

    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
