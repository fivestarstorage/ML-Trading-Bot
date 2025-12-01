# Ensemble Model Integration - Complete Guide

## ✅ What Was Done

### 1. **Clean Start**
- Removed all old ensemble attempts (`crypto_predictor/`, old test scripts)
- Created fresh, research-backed implementation from scratch

### 2. **Proper Ensemble System Created**
Based on research showing 60-70% accuracy with ensemble methods:

**File:** `src/ensemble_model.py`
- **6 diverse base models**: RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting, LogisticRegression
- **Cross-validation training**: 5-fold stratified CV for robust out-of-fold predictions
- **Meta-learning layer**: Stacking with logistic regression for optimal combination
- **Dynamic weighting**: Adapts model contributions based on recent performance
- **Feature importance tracking**: Understands what each model learns

### 3. **Integration with Your Winning Strategy**

**File:** `scripts/train_ensemble_proper.py`
- Uses **REAL trade candidates** from your crypto momentum strategy
- Proper target labels (win/loss based on TP/SL hits, not arbitrary price movement)
- All your strategy's features preserved
- Trains ensemble to filter trades (improve win rate)

### 4. **Jupyter Lab Analysis Notebook**

**File:** `notebooks/ensemble_analysis.ipynb`
- Interactive analysis and visualization
- Model weight visualization
- Feature importance plots
- Probability threshold optimization
- Equity curve comparisons
- Performance metrics

---

## 🚀 How to Use

### Step 1: Train the Ensemble (Proper Way)

```bash
# This generates real trade candidates and trains ensemble
python scripts/train_ensemble_proper.py
```

**This will:**
1. Load 2023-2024 BTC data
2. Generate features
3. Create real trade candidates using your crypto strategy
4. Train 6-model ensemble with cross-validation
5. Save model to `models/ensemble_model.pkl`
6. Save candidates to `models/training_candidates.parquet`

**Expected runtime:** 15-30 minutes

---

### Step 2: Analyze Results in Jupyter Lab

```bash
# Start Jupyter Lab
jupyter lab
```

**Then:**
1. Navigate to `notebooks/ensemble_analysis.ipynb`
2. Run all cells (Cell → Run All)
3. Explore:
   - Model weights
   - Feature importance
   - Probability calibration
   - Optimal threshold finding
   - Equity curves

**Jupyter Lab vs Jupyter Notebook:**
- **Jupyter Lab** = Modern interface with file browser, terminal, multiple tabs
- **Jupyter Notebook** = Classic single-notebook interface
- **Both work fine!** Use whichever you prefer

**Keyboard shortcuts:**
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell

---

### Step 3: Integrate with Backtesting

The ensemble is now a **trade filter** that sits on top of your strategy:

**How it works:**
1. Your crypto strategy generates trade candidates
2. Ensemble predicts probability of winning for each
3. Only take trades above probability threshold (e.g., 0.60)
4. Result: Fewer trades, higher win rate, better Sharpe

**To use in backtest:**
```python
# In your backtester
ensemble = HighPerformanceEnsemble(config)
ensemble.load('models/ensemble_model.pkl')

# When evaluating candidates
candidates_df = crypto_strategy.generate_candidates(df)
probabilities = ensemble.predict_proba(candidates_df)

# Filter by threshold
threshold = 0.60  # Adjust based on Jupyter analysis
filtered_candidates = candidates_df[probabilities > threshold]

# Run backtest on filtered trades only
```

---

## 📊 Expected Results

### What Good Looks Like:
- **Accuracy: 55-65%** on test set (vs 50% random)
- **Ensemble improves win rate** by 5-15% vs baseline
- **Sharpe ratio improves** when using probability filtering
- **Trade count reduced** by 30-50% while maintaining/improving returns

### What to Watch For:
- If accuracy < 52%: Ensemble not learning useful patterns
- If best threshold is 0.50: Ensemble not confident (use baseline strategy)
- If accuracy > baseline win rate: ✅ Ensemble is helping!

---

## 🔧 Troubleshooting

### "No candidates generated"
- Check your `config.yml` - crypto strategy parameters may be too restrictive
- Try wider date range (2022-2024) for more data
- Check data quality (`data_adapter.load_data()` working?)

### "Training takes forever"
- Reduce date range to 2024 only
- Training 75k+ candidates takes 15-30 min (normal)
- Check CPU usage - should be near 100%

### "Model not improving performance"
- Try longer training period (more data)
- Check feature importance - are the right features being used?
- Try different probability thresholds (0.55, 0.60, 0.65)
- Consider feature engineering improvements

### "Jupyter Lab not working"
```bash
# Already installed, but if issues:
pip install jupyterlab
jupyter lab --version

# If port conflict:
jupyter lab --port 8889
```

---

## 📈 Next Steps

### 1. **Optimize Threshold**
Use Jupyter analysis to find optimal probability threshold:
- Balance win rate improvement vs trade count reduction
- Consider Sharpe ratio, not just win rate

### 2. **Walk-Forward Testing**
Test ensemble on out-of-sample periods:
```bash
# Full WFA backtest with ensemble
trading-bot --action backtest --wfa --from 2023-01-01 --to 2024-11-29
```

### 3. **Live Paper Trading**
Once validated:
- Deploy ensemble-filtered strategy to paper trading
- Monitor real-time performance
- Track probability calibration drift

### 4. **Continuous Improvement**
- Retrain monthly with new data
- Track model degradation
- A/B test: ensemble vs baseline
- Add new features if performance degrades

---

## 🎯 Key Files Reference

| File | Purpose |
|------|---------|
| `src/ensemble_model.py` | Core ensemble implementation |
| `src/ml_model.py` | Interface (routes to ensemble) |
| `scripts/train_ensemble_proper.py` | Training script with real trades |
| `notebooks/ensemble_analysis.ipynb` | Interactive analysis (Jupyter Lab) |
| `models/ensemble_model.pkl` | Trained ensemble (generated) |
| `models/training_candidates.parquet` | Training data (generated) |

---

## 💡 Research References

Implementation based on:
- **Focal Article**: Ensemble methods achieve 60-70% accuracy vs 50-60% single models
- **Key techniques**: Bagging + Boosting + Stacking + Dynamic Weighting
- **Real-world results**: 18% improvement (XGBoost ensemble vs LSTM alone)

---

## ✅ Quick Start Checklist

- [ ] Run: `python scripts/train_ensemble_proper.py`
- [ ] Wait 15-30 min for training
- [ ] Check output - is accuracy > 55%?
- [ ] Run: `jupyter lab`
- [ ] Open `notebooks/ensemble_analysis.ipynb`
- [ ] Run all cells
- [ ] Find optimal threshold
- [ ] Integrate threshold into backtester
- [ ] Run full walk-forward test
- [ ] Compare vs baseline strategy

---

**Questions?** Check the notebook outputs and model performance metrics first. The ensemble should clearly show whether it's improving your strategy or not.
