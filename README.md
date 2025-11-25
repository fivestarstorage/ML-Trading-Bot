# ML Trading Bot

Automated rule-based trading strategy with ML gatekeeper, designed for prop firm simulation and walk-forward analysis.

## Features

- **Structure Analysis**: H4 Breaks of Structure (BoS), CHoCH, and Inducement detection.
- **Institutional Flow**: Daily EMA (50/200) trend filter, liquidity sweep confirmation, displacement + mitigation checks, killzone/time-of-day gating, and premium/discount zoning.
- **Entry Logic**: M30/H1 Fair Value Gaps (FVG) and Order Blocks (OB).
- **ML Gatekeeper**: LightGBM classifier to filter entries (probability >= 55%).
- **Walk-Forward Analysis**: 5-year train -> 6-month test sliding window.
- **Prop Firm Mode**: Simulation with $6,000 capital, 8% max drawdown, and bi-weekly payouts.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare Data:
   - Place your OHLCV CSV files in `Data/`.
   - Naming convention: `SYMBOL_TF.csv` (e.g., `XAUUSD_M1.csv`).
   - Columns: `timestamp, open, high, low, close, volume`.

## Usage

### Command Line Interface

#### Quick Menu (recommended)

Install the project (e.g., `pip install -e .`) and launch the unified CLI menu:

```bash
trading-bot
```

From here you can trigger normal/prop backtests, walk-forward analysis, ML retraining, or optimisation runs without memorising arguments.

#### Script Mode

You can still run the legacy script directly if you prefer raw arguments.

**1. Interactive Mode (Recommended)**
Simply run the script without arguments to see a menu:
```bash
python scripts/run_backtest.py
```

**2. Run Backtest (Prop Firm Mode)**
Run a simulation with prop firm rules (drawdown limits, profit withdrawals):
```bash
python scripts/run_backtest.py --symbol XAUUSD --mode propfirm --config config.yml
```

**3. Run Backtest for Specific Dates**
Filter the data to a specific range (ensure your CSV covers these dates):
```bash
python scripts/run_backtest.py --symbol XAUUSD --from 2024-01-01 --to 2024-06-01 --mode propfirm --config config.yml
```

**4. Train Model (on all data)**
Create a production model using all available history:
```bash
python scripts/run_backtest.py --action train_model --config config.yml
```

**5. Walk-Forward Analysis**
Run a robust validation using rolling windows (e.g., Train 4 Years -> Test 6 Months):
```bash
python scripts/run_backtest.py --wfa --config config.yml
```

### IG demo trading (optional)

Send the freshest high-confidence signal directly to IG's demo environment via their REST API ([docs](https://www.ig.com/au/trading-platforms/trading-apis/how-to-use-ig-api#:~:text=You%20can%20also%20use%20an,request%20for%20an%20access%20token.)):

1. Create a `.env` file with your demo credentials (or export the variables):
   ```bash
   IG_API_KEY_DEMO=your_api_key
   IG_USERNAME_DEMO=your_username
   IG_PASSWORD_DEMO=your_password
   IG_ACCOUNT_ID_DEMO=your_account_id
   ```
2. Enable `live_trading` at the bottom of `config.yml` and supply the IG epic you want to trade:
   ```yaml
   live_trading:
     enabled: true
     use_demo: true
     epic: "CS.D.GC.MONTH2.IP"  # update to your preferred market
     size: 1.0
     probability_threshold: 0.65
   ```
3. Run the CLI in live mode (the bot loads the trained model, scores the latest candles, and sends a market order when the probability gate is met):
   ```bash
   python scripts/run_backtest.py --action live_demo --symbol XAUUSD --config config.yml
   ```

### Continuous IG streaming + trading

If you want the bot to _stream_ IG’s gold feed and trade every time a fresh bar completes:

1. In `config.yml`, set `data.source: ig` and configure `data.ig_stream` with your epic/resolution plus `live_trading.enabled: true`.
2. Ensure a trained model exists (same as before).
3. Launch the streaming loop:
   ```bash
   python scripts/run_backtest.py --action stream_live --symbol XAUUSD --config config.yml
   ```

The command bootstraps historical bars via IG’s `/prices` REST API, opens a Lightstreamer subscription to `CHART:{epic}:{resolution}`, recomputes the feature/candidate pipeline on every completed bar, and routes qualifying trades straight to IG’s demo account through `/positions/otc`.

### Configuration

Edit `config.yml` to adjust strategy parameters, risk settings, and file paths.

```yaml
strategy:
  strategy_type: smc
  use_daily_trend_filter: true
  require_liquidity_sweep: true
  require_displacement: true
  killzones:
    london:
      start: "07:00"
      end: "11:00"
    new_york:
      start: "13:00"
      end: "17:00"
  model_threshold: 0.55
  risk_reward_min: 2.0

backtest:
  propfirm:
    initial_capital: 6000
    max_drawdown_pct: 0.08
```

## Project Structure

- `src/`: Core logic (structure, features, entries, ml, backtest).
- `scripts/`: CLI entry points.
- `tests/`: Unit tests.
- `reports/`: Generated backtest reports and plots.
- `models/`: Saved ML models.

## Testing

Run unit tests:
```bash
pytest tests/
```

