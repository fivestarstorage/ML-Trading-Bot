"""
Master Test Runner
Runs all comprehensive tests for WFA and Live Trading
"""
import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging


def run_test_suite(test_file, description):
    """Run a test suite and return success status"""
    logger = setup_logging()
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING: {description}")
    logger.info("="*80 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        success = result.returncode == 0

        if success:
            logger.info(f"\n✅ {description} - ALL TESTS PASSED")
        else:
            logger.error(f"\n❌ {description} - SOME TESTS FAILED")

        return success

    except Exception as e:
        logger.error(f"\n❌ {description} - EXECUTION ERROR: {e}")
        return False


def main():
    logger = setup_logging()

    logger.info("="*80)
    logger.info("COMPREHENSIVE TESTING SUITE")
    logger.info("ML Trading Bot - Walk-Forward Analysis & Live Trading Tests")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    test_results = {}

    # Test 1: Walk-Forward Analysis
    wfa_passed = run_test_suite(
        "tests/test_wfa_comprehensive.py",
        "Walk-Forward Analysis Tests (SPY & BTC, 2020-2025)"
    )
    test_results['WFA Tests'] = wfa_passed

    # Test 2: Live Trading Components
    live_passed = run_test_suite(
        "tests/test_live_trading.py",
        "Live Trading Integration Tests (SPY & BTC)"
    )
    test_results['Live Trading Tests'] = live_passed

    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80 + "\n")

    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:<50} {status}")

    total_suites = len(test_results)
    passed_suites = sum(1 for p in test_results.values() if p)

    logger.info("\n" + "-"*80)
    logger.info(f"Total Test Suites: {total_suites}")
    logger.info(f"Passed: {passed_suites}")
    logger.info(f"Failed: {total_suites - passed_suites}")
    logger.info(f"Success Rate: {passed_suites/total_suites*100:.1f}%")
    logger.info("-"*80)

    all_passed = all(test_results.values())

    if all_passed:
        logger.info("\n🎉 ALL TEST SUITES PASSED - SYSTEM IS PRODUCTION READY!")
    else:
        logger.error("\n⚠️  SOME TESTS FAILED - REVIEW ERRORS BEFORE PRODUCTION")

    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
