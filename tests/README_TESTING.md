# Comprehensive Testing Guide

This directory contains comprehensive test suites for the ML Trading Bot, covering:
1. Walk-Forward Analysis (WFA) for all years (2020-2025)
2. Live Trading Integration Tests

## Test Suites

### 1. Walk-Forward Analysis Tests (`test_wfa_comprehensive.py`)

Tests WFA functionality for:
- **SPY** with SMC (Smart Money Concepts) strategy
- **BTC** with Crypto Momentum strategy

For each strategy, tests are run for years: 2020, 2021, 2022, 2023, 2024, 2025

**What it tests:**
- Data loading for each year
- Feature calculation
- Candidate generation
- WFA execution with proper training windows
- Backtesting of WFA results
- Performance metrics (win rate, profit factor, drawdown, etc.)

### 2. Live Trading Tests (`test_live_trading.py`)

Tests all live trading components for production readiness:

**For both SPY and BTC:**
1. ✅ Configuration validation
2. ✅ Alpaca API connection & authentication
3. ✅ Historical data loading
4. ✅ Feature calculation
5. ✅ ML model loading
6. ✅ Signal generation pipeline (end-to-end)
7. ✅ Live data fetching
8. ✅ Order execution (dry run mode)
9. ✅ Market hours detection (SPY only)
10. ✅ Error handling & recovery
11. ✅ Live runner initialization

**Safety:** All live trading tests run in DRY RUN mode by default (no real orders)

## Running the Tests

### Option 1: Run All Tests (Recommended)

```bash
python tests/run_all_tests.py
```

This runs both WFA and Live Trading tests and generates a comprehensive report.

### Option 2: Run Individual Test Suites

**WFA Tests Only:**
```bash
python tests/test_wfa_comprehensive.py
```

**Live Trading Tests Only:**
```bash
python tests/test_live_trading.py
```

### Option 3: Run Specific Tests

You can modify the test files to run specific years or strategies by commenting out sections.

## Prerequisites

### Required Data
- SPY data from 2020-2025 (5-minute bars)
- BTC data from 2020-2025 (5-minute bars)
- Trained models:
  - `models/spy_lgbm.pkl` for SPY
  - `models/lgbm_model.pkl` for BTC

### Required Configuration
- `.env` file with Alpaca API credentials:
  ```
  ALPACA_API_KEY=your_key_here
  ALPACA_API_SECRET=your_secret_here
  ```

- Valid config files:
  - `configs/config_spy.yml`
  - `configs/config_btc_optimized.yml`

## Training Models (if needed)

If models are missing, train them first:

**For SPY:**
```bash
python -m src.cli --config configs/config_spy.yml --action train_latest
```

**For BTC:**
```bash
python -m src.cli --config configs/config_btc_optimized.yml --action train_latest
```

## Test Output

Each test suite generates:
1. **Console output** with detailed test results
2. **Test report** (`.txt` file) in `tests/` directory
3. **Detailed results** (`.csv` file) for further analysis

### Report Location
Reports are saved with timestamps:
- `tests/wfa_test_report_YYYYMMDD_HHMMSS.txt`
- `tests/wfa_test_results_YYYYMMDD_HHMMSS.csv`
- `tests/live_trading_test_report_YYYYMMDD_HHMMSS.txt`

## Understanding Test Results

### WFA Test Results

For each year tested, you'll see:
- ✅ PASS or ❌ FAIL status
- Number of trades executed
- Win rate percentage
- Net profit/loss
- Profit factor
- Maximum drawdown

**Example output:**
```
Year      Status        Trades     Win Rate      Net Profit       PF
2020      ✅ PASS       45         65.00%        $2,450.00        2.15
2021      ✅ PASS       52         58.00%        $1,890.00        1.85
```

### Live Trading Test Results

For each component tested:
- ✅ PASS or ❌ FAIL status
- Error details if failed

**Example output:**
```
Test Name                                                    Status
SPY-SMC - Configuration Validation                         ✅ PASS
SPY-SMC - Alpaca Connection                                ✅ PASS
SPY-SMC - Signal Generation Pipeline                       ✅ PASS
```

## Troubleshooting

### "No data available for year YYYY"
- Check that data files exist in `Data/` directory
- Verify date range in data files covers the requested year

### "Model not found at models/xxx.pkl"
- Train the model first using `--action train_latest`

### "Alpaca connection failed"
- Verify `.env` file has correct API credentials
- Check that API keys are valid (not expired)
- For paper trading, ensure using paper trading keys

### "No candidates generated"
- This can be normal for some periods
- Check strategy parameters in config file
- Verify data quality for that period

## Next Steps After Testing

If all tests pass (✅):
1. ✅ System is verified for production readiness
2. ✅ WFA works correctly for historical analysis
3. ✅ Live trading components are functional

**Before going live:**
1. Review test reports thoroughly
2. Test with small position sizes first
3. Monitor closely for the first few days
4. Keep dry_run=True until confident
5. Ensure you understand all risk management rules

## Safety Notes

⚠️ **IMPORTANT:**
- Live trading tests run in DRY RUN mode by default
- No real orders are placed during testing
- Always test on paper account before live trading
- Review all configuration carefully
- Understand the risks of automated trading

## Contact

For issues or questions:
- Check logs in `logs/` directory
- Review configuration files
- Ensure all dependencies are installed: `pip install -r requirements.txt`
