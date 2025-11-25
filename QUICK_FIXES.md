# Quick Fixes to Test

## Problem Summary

The strategy has a **22.2% win rate** (below 25% breakeven), meaning it loses money before ML filtering. The ML model is working (higher confidence = better win rate), but can't fix a broken strategy.

## Quick Fixes to Test

### Fix 1: Increase Threshold to 0.60
**Why:** High confidence trades (>=0.6) have 37.5% win rate vs 20% for low confidence
**How:** Change `model_threshold` in `config.yml` from 0.50 to 0.60
**Expected:** Fewer trades, higher win rate

### Fix 2: Filter Out FVG Entries
**Why:** OB entries have 33.3% win rate vs 22.6% for FVG entries
**How:** Modify entry generator to skip FVG entries, or filter in backtester
**Expected:** Higher win rate, fewer trades

### Fix 3: Filter Out Evening Hours (18-20)
**Why:** Evening hours have 0-16.7% win rate vs 66-100% for morning hours
**How:** Add hour filter in backtester
**Expected:** Better overall performance

### Fix 4: Combine All Fixes
**Why:** Stack improvements for maximum effect
**How:** Apply all three fixes together
**Expected:** Best performance

## Test Script

Run the test script to compare all configurations:

```bash
python scripts/test_improvements.py
```

This will test:
1. Baseline (current)
2. Higher threshold (0.60)
3. OB only
4. OB only + higher threshold
5. OB only + higher threshold + filter hours

## Expected Results

Based on current data:
- **Baseline:** 26.1% win rate, -$171 net
- **Higher threshold:** ~35-40% win rate, fewer trades
- **OB only:** ~33% win rate baseline, better with ML
- **Combined:** Best performance, potentially profitable

## Long-Term Fixes

These quick fixes help, but the **fundamental strategy needs fixing**:

1. **Improve Entry Conditions**
   - Current: 22.2% win rate
   - Target: >25% win rate
   - Check structure detection logic

2. **Better TP/SL Placement**
   - Current: 3:1 R:R
   - May need dynamic based on volatility

3. **Regime Detection**
   - October 2025: 8.3% win rate
   - Need to detect and avoid bad periods

