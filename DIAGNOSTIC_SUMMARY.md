# Diagnostic Summary: Why Results Were Poor

## Root Causes Identified

### 1. **Label Window Too Short (CRITICAL)**
- **Before**: 50 bars = 4.2 hours
- **After**: 200 bars = 16.7 hours
- **Problem**: With 3:1 R:R, SL is 3x closer than TP. In a 4.2-hour window, SL hits first 75% of the time, causing many eventual winners to be labeled as losses.
- **Impact**: Model trained on incorrectly labeled data, leading to poor predictions.

### 2. **Model Threshold Too High**
- **Before**: 0.60 threshold filtered out 98.7% of candidates
- **After**: 0.50 threshold allows more trades through
- **Problem**: Even though model correctly identified bad trades (mean prob 0.192), threshold was so high that almost nothing passed.
- **Impact**: Only 21 trades in 10 months = insufficient sample size.

### 3. **Strategy Win Rate Below Breakeven**
- **Actual**: 21.2% win rate
- **Required**: 25.0% win rate (for 3:1 R:R)
- **Gap**: -3.8% below breakeven
- **Problem**: Entry conditions generating losing trades even before ML filtering.

### 4. **Incomplete Trade Labeling**
- **Problem**: Trades that don't hit TP/SL within label window were labeled as losses (pnl=0)
- **Impact**: Creates negative bias in training data
- **Fix**: Improved labeling logic (though still marks as 0, at least tracks properly)

## Changes Made

1. ✅ Increased `label_window` from 50 → 200 bars
2. ✅ Lowered `model_threshold` from 0.60 → 0.50
3. ✅ Added detailed probability distribution logging
4. ✅ Improved labeling logic to track TP/SL hits properly

## Next Steps

**CRITICAL**: You MUST retrain the model with the new label window!

```bash
# Retrain model with new label window
python scripts/run_backtest.py --action train_model --symbol XAU --config config.yml --from 2020-01-01

# Then backtest with the new model
python scripts/backtest_only.py --from 2025-01-01 --to 2025-10-31
```

The old model was trained with 50-bar label window, so it learned from incorrectly labeled data. The new model needs to be trained with 200-bar window to properly learn.

## Expected Improvements

- **More trades**: Lower threshold should allow 5-10x more trades
- **Better labeling**: Longer window should reduce false negatives
- **Better model**: Retrained model should learn correct patterns
- **Better visibility**: Detailed logs will show what's happening

## If Results Still Poor

If results are still poor after retraining, the issue is likely:
1. **Entry conditions are fundamentally flawed** - Structure detection may be wrong
2. **Market regime changed** - 2025 data may be different from 2020-2024 training data
3. **Strategy needs refinement** - May need to adjust TP/SL ratios or entry filters

Check the detailed logs to see:
- Probability distribution of candidates
- How many trades pass threshold
- Win rate of high-confidence vs low-confidence trades
