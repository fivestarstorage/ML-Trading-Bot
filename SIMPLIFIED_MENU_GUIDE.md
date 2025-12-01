# Simplified WFA Menu - Quick Start Guide

## How to Use

Simply run:

```bash
trading-bot
```

## Menu Flow

The menu has been **completely simplified** to focus on Walk-Forward Analysis:

### Step 1: Select Ticker
```
Select ticker symbol:
  → SPY - S&P 500 ETF
  → QQQ - Nasdaq 100 ETF
  → IWM - Russell 2000 ETF
  → DIA - Dow Jones ETF
  → Custom ticker...
  → Exit
```

**For your current setup, select: SPY**

### Step 2: Select Strategy
```
Select strategy for SPY:
  → Smart Money Concepts (SMC)
  → Back to ticker selection
```

**Currently only SMC strategy is available**

### Step 3: Select Time Period
```
Select time period for SPY WFA:
  → Single Year (e.g., 2024)
  → Multiple Years (e.g., 2023,2024,2025)
  → All Years (2020-2025)
  → Back to strategy selection
```

**Choose based on your needs:**
- **Single Year**: Quick test (5-15 minutes)
- **Multiple Years**: Specific years (20-45 minutes)
- **All Years**: Complete validation (30-90 minutes)

## What You Get

After the WFA completes, comprehensive reports are automatically saved to:

```
/Users/rileymartin/ML-Trading-Bot/reports/{TICKER}_{STRATEGY}_{TIMESTAMP}/
```

### Generated Files:

1. **📊 summary_report.txt**
   - Overall performance metrics
   - Year-by-year breakdown
   - Target evaluation (Win Rate, YoY Profit, Max DD)

2. **📈 equity_curve.png**
   - Cumulative equity growth over time
   - Visual representation of P&L trajectory

3. **📊 yearly_performance.png**
   - 4-panel chart showing:
     - Win Rate by Year
     - YoY Profit by Year
     - Max Drawdown by Year
     - Trade Count by Year

4. **📉 drawdown_analysis.png**
   - Drawdown visualization across all years
   - Shows maximum risk exposure

5. **📋 trade_distribution.png**
   - Win/Loss pie chart
   - P&L distribution histogram

6. **💾 detailed_results.csv**
   - Year-by-year metrics in CSV format
   - Easy to import into Excel/Google Sheets

7. **💾 all_trades.csv**
   - Individual trade records
   - Entry/exit times, P&L, direction, etc.

## Example Session

```
trading-bot

Select ticker: SPY
Select strategy: Smart Money Concepts (SMC)
Select time period: Single Year
Enter year: 2024

[WFA runs for 5-15 minutes]

✅ REPORTS SAVED TO: reports/SPY_smc_20251130_173045/

Generated files:
  📊 summary_report.txt
  📈 equity_curve.png
  📊 yearly_performance.png
  📉 drawdown_analysis.png
  📋 trade_distribution.png
  💾 detailed_results.csv
  💾 all_trades.csv
```

## Tips

### For Quick Testing
- Use **Single Year** with current year (2025)
- Takes ~10 minutes
- Good for validating setup

### For Strategy Validation
- Use **All Years (2020-2025)**
- Takes ~30-90 minutes
- Shows performance across different market conditions

### For Specific Analysis
- Use **Multiple Years** with specific years
- Example: "2023,2024,2025" for recent performance
- Example: "2020,2022,2024" for specific market conditions

## What the Graphs Show

### Equity Curve
- Blue line = cumulative profit
- Shows if strategy is consistently profitable
- Steep upward slope = strong performance

### Yearly Performance
- **Green bars** = met target
- **Orange bars** = below target
- Easily see which years performed best

### Drawdown Analysis
- Shows risk exposure over time
- Red dashed line = 10% max drawdown limit
- Lower is better (less risk)

### Trade Distribution
- Pie chart shows win/loss ratio
- Histogram shows P&L distribution
- Helps understand risk/reward profile

## Current Optimized Configuration

The menu uses your optimized SPY configuration:

```yaml
Ticker: SPY
Strategy: Smart Money Concepts (SMC)
Parameters:
  - Risk per trade: 1.5%
  - ML Threshold: 0.50
  - Take Profit: 3.5x ATR
  - Stop Loss: 1.0x ATR
  - Initial Capital: $6,000
```

**Expected Performance (2024 optimization results):**
- Win Rate: ~68-70%
- YoY Profit: >700%
- Max Drawdown: <10%
- Risk:Reward: 3.5:1

## Troubleshooting

### Menu doesn't start
- Ensure you're in the project directory
- Check that `trading-bot` command is available
- Try: `python -m src.menu` as alternative

### WFA takes too long
- Each year takes 5-15 minutes
- 6 years (2020-2025) = 30-90 minutes total
- This is normal for proper Walk-Forward Analysis

### No graphs generated
- Check matplotlib is installed: `pip install matplotlib seaborn`
- Reports folder will still have CSV files

### Custom ticker not working
- Ensure ticker is valid on Alpaca
- Alpaca must have historical data for that ticker
- SMC strategy works best on liquid assets (SPY, QQQ, etc.)

## Next Steps

1. **Run Initial Test**: Start with SPY, SMC, Single Year (2024 or 2025)
2. **Review Reports**: Check the generated graphs and summary
3. **Full Validation**: Run All Years (2020-2025) for complete picture
4. **Analyze Results**: Look for consistency across years
5. **Paper Trade**: If results are good, test in paper trading
6. **Go Live**: Start with reduced risk, scale up gradually

---

**Generated:** 2025-11-30
**Menu Type:** Simplified WFA-focused
**Reports Location:** `/Users/rileymartin/ML-Trading-Bot/reports/`
