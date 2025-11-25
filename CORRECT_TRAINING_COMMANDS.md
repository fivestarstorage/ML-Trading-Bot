# Correct Training and Backtesting Commands

## ⚠️ CRITICAL: Avoid Overfitting

When training a model, you MUST exclude the backtest period from training data.

## Correct Workflow

### Step 1: Train Model (2020-2024 ONLY)
```bash
python scripts/run_backtest.py --action train_model --symbol XAU --config config.yml --from 2020-01-01 --to 2024-12-31
```

**Important**: The `--to 2024-12-31` ensures training stops at end of 2024, excluding all 2025 data.

### Step 2: Backtest (2025 ONLY)
```bash
python scripts/backtest_only.py --from 2025-01-01 --to 2025-10-31
```

## Why This Matters

- **Without `--to`**: Training includes ALL data from 2020-01-01 onwards (including 2025)
- **With `--to 2024-12-31`**: Training stops at end of 2024, excluding 2025
- **Result**: Model learns patterns from 2020-2024, then tested on unseen 2025 data

## Alternative: Use train_and_backtest.py

The `train_and_backtest.py` script already handles this correctly:

```bash
python scripts/train_and_backtest.py
```

This script:
1. Trains on 2020-01-01 to 2024-12-31 (hardcoded)
2. Backtests on 2025-01-01 to 2025-10-31 (hardcoded)
3. Ensures no data leakage

## Quick Fix for Your Current Command

Change this:
```bash
python scripts/run_backtest.py --action train_model --symbol XAU --config config.yml --from 2020-01-01
```

To this:
```bash
python scripts/run_backtest.py --action train_model --symbol XAU --config config.yml --from 2020-01-01 --to 2024-12-31
```

The `--to 2024-12-31` is the critical addition!
