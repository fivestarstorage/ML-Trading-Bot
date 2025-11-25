"""
ORB Backtester with trailing stops, break-even, and partial closes
Similar to New_XAU_Bot.mq5
"""
import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class ORBBacktester:
    def __init__(self, config):
        self.config = config
        self.commission = config['backtest']['commission']
        self.slippage = config['backtest']['slippage']
        self.mode = config['backtest']['mode']
        
        # Prop firm settings
        self.initial_capital = config['backtest']['propfirm']['initial_capital']
        self.max_dd_pct = config['backtest']['propfirm']['max_drawdown_pct']
        self.risk_pct = config['backtest']['propfirm']['per_trade_risk_pct']
        self.payout_days = config['backtest']['propfirm']['payout_cycle_days']
        
        # Adaptive risk (from EA)
        self.risk_percent = config['strategy'].get('risk_percent', 1.5) / 100.0
        self.risk_percent_low_vol = config['strategy'].get('risk_percent_low_vol', 0.5) / 100.0
        
        # Trailing stop settings
        self.use_trailing_stop = config['strategy'].get('use_trailing_stop', True)
        self.trailing_stop_atr = config['strategy'].get('trailing_stop_atr', 1.0)
        self.trailing_start_profit = config['strategy'].get('trailing_start_profit', 1.0)
        
        # Break-even settings
        self.breakeven_trigger_r = config['strategy'].get('breakeven_trigger_r', 1.5)
        
        # Partial close settings
        self.partial_close_15r = config['strategy'].get('partial_close_15r', 0.2)
        self.partial_close_2r = config['strategy'].get('partial_close_2r', 0.2)
        
        self.trades = []
        self.history = []
        
    def get_adaptive_risk(self, current_atr, df_base, current_time):
        """Get adaptive risk based on volatility (like EA)"""
        # Calculate 20-day average ATR
        lookback_bars = 20 * 96  # 20 days * 96 M15 bars per day
        time_idx = df_base.index.get_indexer([current_time], method='pad')[0]
        start_idx = max(0, time_idx - lookback_bars)
        
        if start_idx >= time_idx:
            return self.risk_pct
        
        atr_window = df_base['atr'].iloc[start_idx:time_idx]
        if len(atr_window) < 100:  # Need enough data
            return self.risk_pct
        
        avg_atr = atr_window.mean()
        
        # If current ATR < average: use LOW RISK
        if current_atr < avg_atr:
            return self.risk_percent_low_vol
        else:
            return self.risk_percent
    
    def run(self, candidates_df, probabilities, df_base):
        """
        Run ORB backtest with trailing stops, break-even, and partial closes.
        df_base is needed for adaptive risk calculation and trailing stop logic.
        """
        threshold = self.config['strategy']['model_threshold']
        
        df = candidates_df.copy()
        df['prob'] = probabilities
        
        # Filter by threshold
        trades_df = df[df['prob'] >= threshold].copy()
        
        if len(trades_df) == 0:
            logger.warning("No trades passed threshold!")
            return pd.DataFrame(), pd.DataFrame()
        
        trades_df = trades_df.sort_values('entry_time')
        
        # Cooldown filter
        cooldown_minutes = self.config['backtest'].get('cooldown_minutes', 15)
        if len(trades_df) > 0:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            filtered_trades = []
            last_trade_time = None
            
            for _, row in trades_df.iterrows():
                if last_trade_time is None:
                    filtered_trades.append(row)
                    last_trade_time = row['entry_time']
                else:
                    time_diff = (row['entry_time'] - last_trade_time).total_seconds() / 60
                    if time_diff >= cooldown_minutes:
                        filtered_trades.append(row)
                        last_trade_time = row['entry_time']
            
            trades_df = pd.DataFrame(filtered_trades)
        
        if len(trades_df) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Initialize
        current_equity = self.initial_capital
        cycle_start_time = trades_df['entry_time'].iloc[0]
        total_withdrawn = 0
        failed = False
        
        # Process each trade with advanced management
        for idx, row in trades_df.iterrows():
            if failed:
                break
            
            # Prop firm cycle check
            if self.mode == 'propfirm':
                days_since_start = (row['entry_time'] - cycle_start_time).days
                if days_since_start >= self.payout_days:
                    if current_equity > self.initial_capital:
                        profit = current_equity - self.initial_capital
                        total_withdrawn += profit
                        current_equity = self.initial_capital
            
            # Adaptive risk
            current_atr = row['atr']
            adaptive_risk = self.get_adaptive_risk(current_atr, df_base, row['entry_time'])
            
            # Position sizing
            risk_amount = current_equity * adaptive_risk
            dist_sl = abs(row['entry_price'] - row['sl'])
            if dist_sl == 0:
                continue
            
            units = risk_amount / dist_sl
            
            # Simulate trade with trailing stops and partial closes
            net_pnl = self._simulate_trade_with_management(
                row, units, df_base
            )
            
            current_equity += net_pnl
            
            # Record trade
            trade_record = row.to_dict()
            trade_record['net_pnl'] = net_pnl
            trade_record['equity_after'] = current_equity
            trade_record['units'] = units
            trade_record['adaptive_risk'] = adaptive_risk
            self.trades.append(trade_record)
            
            self.history.append({
                'time': row.get('exit_time', row['entry_time']),
                'equity': current_equity,
                'withdrawn': total_withdrawn
            })
            
            # Check drawdown
            if self.mode == 'propfirm':
                dd_limit = self.initial_capital * (1 - self.max_dd_pct)
                if current_equity < dd_limit:
                    logger.warning(f"Max drawdown breached at {row['entry_time']}")
                    failed = True
        
        return pd.DataFrame(self.history), pd.DataFrame(self.trades)
    
    def _simulate_trade_with_management(self, trade_row, units, df_base):
        """
        Simulate trade with:
        - Trailing stop after break-even
        - Break-even move at 1.5R
        - Partial closes at 1.5R and 2R
        """
        entry_time = trade_row['entry_time']
        entry_price = trade_row['entry_price']
        initial_tp = trade_row['tp']
        initial_sl = trade_row['sl']
        bias = trade_row['bias']
        atr = trade_row['atr']
        
        # Find entry index in df_base
        entry_idx = df_base.index.get_indexer([entry_time], method='pad')[0]
        if entry_idx < 0 or entry_idx >= len(df_base) - 1:
            return 0.0
        
        # Look ahead for price action
        future_window = df_base.iloc[entry_idx+1 : min(entry_idx+1+self.config['strategy']['label_window'], len(df_base))]
        
        if len(future_window) == 0:
            return 0.0
        
        # Track position state
        current_sl = initial_sl
        current_tp = initial_tp
        remaining_units = units
        breakeven_moved = False
        partial_15r_closed = False
        partial_2r_closed = False
        
        # Calculate initial risk (R)
        initial_risk = abs(entry_price - initial_sl)
        
        for _, bar in future_window.iterrows():
            current_price = bar['close']
            current_atr_bar = bar.get('atr', atr)
            
            # Calculate current profit in R
            if bias == 'bull':
                profit_in_r = (current_price - entry_price) / initial_risk
            else:
                profit_in_r = (entry_price - current_price) / initial_risk
            
            # Move to break-even at 1.5R
            if not breakeven_moved and profit_in_r >= self.breakeven_trigger_r:
                current_sl = entry_price
                breakeven_moved = True
            
            # Partial close at 1.5R (if not already closed)
            if not partial_15r_closed and profit_in_r >= 1.5:
                close_units = remaining_units * self.partial_close_15r
                remaining_units -= close_units
                partial_15r_closed = True
            
            # Partial close at 2R (if not already closed)
            if not partial_2r_closed and profit_in_r >= 2.0:
                close_units = remaining_units * self.partial_close_2r
                remaining_units -= close_units
                partial_2r_closed = True
            
            # Update trailing stop after break-even
            if breakeven_moved and self.use_trailing_stop and profit_in_r >= self.trailing_start_profit:
                trailing_distance = current_atr_bar * self.trailing_stop_atr
                
                if bias == 'bull':
                    new_sl = current_price - trailing_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = current_price + trailing_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
            
            # Check if SL hit
            if bias == 'bull':
                if bar['low'] <= current_sl:
                    # SL hit - calculate PnL
                    exit_price = current_sl
                    pnl_per_unit = exit_price - entry_price
                    gross_pnl = remaining_units * pnl_per_unit
                    
                    # Add partial close profits
                    if partial_15r_closed:
                        partial_pnl_15r = (units * self.partial_close_15r) * (entry_price + 1.5 * initial_risk - entry_price)
                        gross_pnl += partial_pnl_15r
                    if partial_2r_closed:
                        partial_pnl_2r = (units * (1 - self.partial_close_15r) * self.partial_close_2r) * (entry_price + 2.0 * initial_risk - entry_price)
                        gross_pnl += partial_pnl_2r
                    
                    # Apply costs
                    total_units_traded = units  # All units were traded at some point
                    slippage_cost = total_units_traded * self.slippage * 2
                    comm_cost = total_units_traded * self.commission * 2
                    
                    return gross_pnl - slippage_cost - comm_cost
                    
            else:  # bear
                if bar['high'] >= current_sl:
                    exit_price = current_sl
                    pnl_per_unit = entry_price - exit_price
                    gross_pnl = remaining_units * pnl_per_unit
                    
                    if partial_15r_closed:
                        partial_pnl_15r = (units * self.partial_close_15r) * (entry_price - (entry_price - 1.5 * initial_risk))
                        gross_pnl += partial_pnl_15r
                    if partial_2r_closed:
                        partial_pnl_2r = (units * (1 - self.partial_close_15r) * self.partial_close_2r) * (entry_price - (entry_price - 2.0 * initial_risk))
                        gross_pnl += partial_pnl_2r
                    
                    total_units_traded = units
                    slippage_cost = total_units_traded * self.slippage * 2
                    comm_cost = total_units_traded * self.commission * 2
                    
                    return gross_pnl - slippage_cost - comm_cost
            
            # Check if TP hit
            if bias == 'bull':
                if bar['high'] >= current_tp:
                    exit_price = current_tp
                    pnl_per_unit = exit_price - entry_price
                    gross_pnl = remaining_units * pnl_per_unit
                    
                    if partial_15r_closed:
                        partial_pnl_15r = (units * self.partial_close_15r) * (entry_price + 1.5 * initial_risk - entry_price)
                        gross_pnl += partial_pnl_15r
                    if partial_2r_closed:
                        partial_pnl_2r = (units * (1 - self.partial_close_15r) * self.partial_close_2r) * (entry_price + 2.0 * initial_risk - entry_price)
                        gross_pnl += partial_pnl_2r
                    
                    total_units_traded = units
                    slippage_cost = total_units_traded * self.slippage * 2
                    comm_cost = total_units_traded * self.commission * 2
                    
                    return gross_pnl - slippage_cost - comm_cost
                    
            else:  # bear
                if bar['low'] <= current_tp:
                    exit_price = current_tp
                    pnl_per_unit = entry_price - exit_price
                    gross_pnl = remaining_units * pnl_per_unit
                    
                    if partial_15r_closed:
                        partial_pnl_15r = (units * self.partial_close_15r) * (entry_price - (entry_price - 1.5 * initial_risk))
                        gross_pnl += partial_pnl_15r
                    if partial_2r_closed:
                        partial_pnl_2r = (units * (1 - self.partial_close_15r) * self.partial_close_2r) * (entry_price - (entry_price - 2.0 * initial_risk))
                        gross_pnl += partial_pnl_2r
                    
                    total_units_traded = units
                    slippage_cost = total_units_traded * self.slippage * 2
                    comm_cost = total_units_traded * self.commission * 2
                    
                    return gross_pnl - slippage_cost - comm_cost
        
        # No TP/SL hit within window
        return 0.0


