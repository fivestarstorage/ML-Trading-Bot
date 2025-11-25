import pytest
import pandas as pd
import numpy as np
from src.structure import StructureAnalyzer
from src.features import FeatureEngineer
from src.entries import EntryGenerator
from src.backtester import Backtester

@pytest.fixture
def sample_h4_data():
    # Create synthetic H4 data with clear structure
    dates = pd.date_range('2023-01-01', periods=100, freq='4h')
    data = {
        'open': [1.0] * 100,
        'high': [1.1] * 100,
        'low': [0.9] * 100,
        'close': [1.0] * 100,
        'volume': [1000] * 100
    }
    df = pd.DataFrame(data, index=dates)

    # Create a Bullish Trend: HL -> HH
    # i=10: Low at 0.85 (swing low - lower than surrounding bars)
    # i=20: High at 1.05 (swing high - higher than surrounding bars)
    # i=30: Low at 0.95 (higher low - HL)
    # i=40: High at 1.15 (higher high - HH) -> BOS

    # Make index 10 a proper swing low by setting surrounding bars higher
    df.iloc[7:10, df.columns.get_loc('low')] = 0.88  # Left side higher than 0.85
    df.iloc[10, df.columns.get_loc('low')] = 0.85     # The swing low
    df.iloc[10, df.columns.get_loc('high')] = 1.0
    df.iloc[11:14, df.columns.get_loc('low')] = 0.88  # Right side higher than 0.85

    # Make index 20 a proper swing high by setting surrounding bars lower
    df.iloc[17:20, df.columns.get_loc('high')] = 1.02  # Left side lower than 1.05
    df.iloc[20, df.columns.get_loc('high')] = 1.05     # The swing high
    df.iloc[20, df.columns.get_loc('low')] = 1.00
    df.iloc[21:24, df.columns.get_loc('high')] = 1.02  # Right side lower than 1.05

    df.iloc[30, df.columns.get_loc('low')] = 0.95
    df.iloc[30, df.columns.get_loc('high')] = 1.02

    df.iloc[40, df.columns.get_loc('high')] = 1.15

    # Need enough bars for rolling windows
    return df

def test_swing_detection(sample_h4_data):
    config = {'strategy': {'h4_swing_lookback': 60, 'inducement_lookback': 6}}
    analyzer = StructureAnalyzer(sample_h4_data, config)
    
    # Check if swings were identified at expected indices (offset by window order)
    # Order=3, so swing at 10 is confirmed at 13
    # Just check the series
    assert not pd.isna(analyzer.swing_lows.iloc[10])
    assert not pd.isna(analyzer.swing_highs.iloc[20])

def test_fvg_detection():
    config = {'strategy': {'fvg_lookback': 10, 'ob_lookback': 20, 'atr_period': 14}}
    fe = FeatureEngineer(config)

    dates = pd.date_range('2023-01-01', periods=20, freq='1h')
    df = pd.DataFrame({
        'open': [10.0, 10.0, 10.0, 10.0, 12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'high': [11.0, 11.0, 11.0, 11.0, 15.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
        'low':  [9.0,  9.0,  9.0,  9.0, 11.0, 9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0],
        'close':[10.0, 10.0, 10.0, 10.0, 14.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'volume':[100] * 20
    }, index=dates)

    # Create a clear bullish FVG:
    # Index 4: Large green candle (open=11.5, close=14.0, body=2.5)
    # Index 3: High = 11.0
    # Index 5: Low = 12.5
    # Gap: 11.0 < 12.5 ✓
    # Body 2.5 > avg of previous bodies (which are 0.0) ✓

    df.iloc[3, df.columns.get_loc('high')] = 11.0
    df.iloc[5, df.columns.get_loc('low')] = 12.5
    df.iloc[4, df.columns.get_loc('open')] = 11.5
    df.iloc[4, df.columns.get_loc('close')] = 14.0

    fvgs = fe.detect_fvgs(df)

    # Should find one bull fvg
    assert len([f for f in fvgs if f['type'] == 'bull']) > 0
    fvg = [f for f in fvgs if f['type'] == 'bull'][0]
    assert fvg['top'] == 12.5
    assert fvg['bottom'] == 11.0

def test_prop_firm_drawdown():
    config = {
        'backtest': {
            'commission': 0,
            'slippage': 0,
            'mode': 'propfirm',
            'propfirm': {
                'initial_capital': 6000,
                'max_drawdown_pct': 0.08,
                'per_trade_risk_pct': 0.01,
                'payout_cycle_days': 14
            }
        },
        'strategy': {'model_threshold': 0.5}
    }
    
    backtester = Backtester(config)
    
    # Create trades that lose money
    dates = pd.date_range('2023-01-01', periods=10, freq='1D')
    candidates = pd.DataFrame({
        'entry_time': dates,
        'entry_price': [100]*10,
        'sl': [99]*10, # 1 dollar risk
        'tp': [105]*10,
        'pnl_r': [-1]*10, # All losses
        'target': [0]*10
    })
    
    probs = [0.6] * 10
    
    # Risk 1% = $60 per trade.
    # 8% of 6000 = 480.
    # 8 losses = 480.
    # Should fail after 8th trade.
    
    history, trades = backtester.run(candidates, probs)
    
    # Check equity
    final_equity = history.iloc[-1]['equity']
    assert final_equity <= 6000 - 480
    
    # Check if it stopped trading? 
    # The implementation breaks loop on failure.
    # So we should see roughly 8 trades.
    assert len(trades) <= 9

