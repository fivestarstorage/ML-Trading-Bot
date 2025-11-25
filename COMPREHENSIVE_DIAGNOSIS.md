# Comprehensive Backtest Diagnosis

## Executive Summary

**The strategy is fundamentally broken.** The ML model is working correctly, but it cannot fix a strategy that loses money at the base level.

### Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Training Win Rate | 22.2% | 🔴 Below breakeven (25%) |
| Test Win Rate | 26.1% | 🟡 Slightly better but still losing |
| Profit Factor | 0.86 | 🔴 Losing |
| Net Profit | -$171.25 | 🔴 Negative |
| Model Correlation | 0.129 | 🟡 Weak but positive |

---

## Root Causes

### 1. **Strategy Fundamental Problem** (CRITICAL)

**The underlying strategy loses money before ML filtering.**

- Training data: 22.2% win rate (below 25% breakeven for 3:1 R:R)
- Average R-multiple: -0.11 (negative!)
- Even with perfect ML filtering, strategy would still lose

**Why this matters:**
- ML can only FILTER trades, not fix a broken strategy
- If base win rate is 22.2%, best case after filtering is still ~22-25%
- Need base win rate >25% for ML to help

### 2. **Model Performance** (MODERATE)

**Model IS learning, but calibration is poor.**

**Good:**
- High confidence (>=0.6): 37.5% win rate ✅
- Low confidence (<0.6): 20.0% win rate
- Model correctly identifies better trades

**Bad:**
- Calibration is poor (predicted 81.2% → actual 33.3%)
- Only 16 high-confidence trades (35% of total)
- Model is overconfident

### 3. **Costs Eating Profits** (MODERATE)

**Transaction costs are significant.**

- Gross Profit: $1,011.07
- Gross Loss: $1,182.32
- Total Costs: $231.25 (22.9% of gross profit!)
- Net Profit: -$171.25

**Impact:**
- Even with 26.1% win rate, costs push strategy into loss
- Need higher win rate OR lower costs

### 4. **Entry Type Quality** (MODERATE)

**OB entries perform much better than FVG entries.**

- FVG entries: 22.6% win rate (WORSE)
- OB entries: 33.3% win rate (BETTER)
- 31 FVG trades vs 15 OB trades

**Recommendation:** Focus on OB entries, filter out FVG entries

### 5. **Market Regime Change** (MODERATE)

**October 2025 was particularly bad.**

- October: 8.3% win rate (1 win out of 12 trades)
- October net loss: -$271.48
- Other months: Mixed performance

**Possible causes:**
- Market regime change
- Strategy doesn't work in certain conditions
- Need regime detection

---

## Detailed Analysis

### Training vs Test Performance

| Metric | Training | Test | Difference |
|--------|----------|------|------------|
| Win Rate | 22.2% | 26.1% | +3.9% |
| Total Trades | 82,115 | 46 | - |
| Entry Type | FVG: 22.2%, OB: 24.7% | FVG: 22.6%, OB: 33.3% | OB improved |

**Observation:** Test performance is slightly better, suggesting model learned something, but not enough to overcome strategy flaws.

### Model Calibration

| Probability Range | Predicted WR | Actual WR | Trades |
|-------------------|--------------|-----------|--------|
| 0.5 - 0.55 | 52.4% | 17.4% | 23 |
| 0.55 - 0.6 | 57.3% | 28.6% | 7 |
| 0.6 - 0.65 | 63.1% | 42.9% | 7 |
| 0.65 - 0.7 | 67.1% | 33.3% | 3 |
| 0.7+ | 81.2% | 33.3% | 6 |

**Problem:** Model is overconfident. High probabilities don't translate to high win rates.

### Time-Based Patterns

**Best Hours:**
- Hour 3: 100% win rate (1 trade)
- Hour 8: 100% win rate (2 trades)
- Hour 9: 100% win rate (1 trade)
- Hour 21: 66.7% win rate (3 trades)

**Worst Hours:**
- Hour 18: 16.7% win rate (6 trades)
- Hour 19: 16.7% win rate (6 trades)
- Hour 20: 0% win rate (3 trades)

**Observation:** Evening hours (18-20) perform poorly. Consider filtering these out.

### Monthly Performance

| Month | Trades | Net PnL | Win Rate |
|-------|--------|---------|----------|
| Jan | 6 | +$24.95 | 33.3% |
| Feb | 3 | -$107.45 | 0% |
| Mar | 3 | -$105.25 | 0% |
| Apr | 1 | +$84.25 | 100% |
| May | 8 | +$207.98 | 50% |
| Jun | 1 | +$82.96 | 100% |
| Jul | 5 | +$62.28 | 40% |
| Aug | 5 | -$77.63 | 20% |
| Sep | 2 | -$71.87 | 0% |
| Oct | 12 | -$271.48 | 8.3% |

**Observation:** October was catastrophic. Need to understand why.

---

## Recommendations

### Priority 1: Fix the Strategy (CRITICAL)

**The strategy itself needs fixing before ML can help.**

1. **Improve Entry Conditions**
   - Current: 22.2% win rate
   - Target: >25% win rate
   - Focus on OB entries (33.3% win rate)
   - Filter out FVG entries (22.6% win rate)

2. **Better Structure Detection**
   - Check if structure analyzer is working correctly
   - Verify bias detection logic
   - Ensure discount/premium zones are correct

3. **Better TP/SL Placement**
   - Current: 3:1 R:R ratio
   - May need to adjust based on market conditions
   - Consider dynamic TP/SL based on volatility

### Priority 2: Improve Model (HIGH)

1. **Increase Threshold**
   - Current: 0.50
   - Recommended: 0.60+ (only trade high confidence)
   - High confidence trades: 37.5% win rate

2. **Focus on OB Entries**
   - Filter out FVG entries
   - OB entries: 33.3% win rate vs 22.6% for FVG

3. **Fix Calibration**
   - Model is overconfident
   - Consider calibration techniques (Platt scaling, isotonic regression)
   - Or adjust threshold based on actual performance

### Priority 3: Reduce Costs (MEDIUM)

1. **Reduce Trade Frequency**
   - Current: 46 trades in 10 months
   - Focus on highest quality trades only
   - Use higher threshold (0.6+)

2. **Better Execution**
   - Current costs: $231.25 (22.9% of gross profit)
   - Consider limit orders vs market orders
   - Reduce slippage

### Priority 4: Regime Detection (MEDIUM)

1. **Avoid Bad Periods**
   - October 2025: 8.3% win rate
   - Add regime detection
   - Skip trading during unfavorable conditions

2. **Time-Based Filtering**
   - Filter out evening hours (18-20)
   - Focus on better performing hours

---

## Next Steps

1. **Immediate:** Increase threshold to 0.60, filter out FVG entries
2. **Short-term:** Investigate why strategy has 22.2% win rate
3. **Medium-term:** Improve entry conditions, better structure detection
4. **Long-term:** Add regime detection, dynamic TP/SL

---

## Testing Plan

1. **Test 1:** Increase threshold to 0.60, filter FVG entries
   - Expected: Fewer trades, higher win rate
   - Target: >30% win rate

2. **Test 2:** Focus on OB entries only
   - Expected: Higher win rate (33.3% baseline)
   - Target: >35% win rate with ML filtering

3. **Test 3:** Filter out evening hours (18-20)
   - Expected: Better overall performance
   - Target: Reduce losses from bad hours

4. **Test 4:** Investigate October 2025
   - Why was it so bad?
   - Can we detect and avoid similar periods?

---

## Conclusion

**The ML model is working, but the strategy is broken.**

- Strategy win rate: 22.2% (below breakeven)
- Model can only filter, not fix
- Need to fix strategy first, then optimize ML

**Quick wins:**
1. Increase threshold to 0.60+
2. Filter out FVG entries
3. Focus on OB entries only

**Long-term fixes:**
1. Improve entry conditions
2. Better structure detection
3. Regime detection


