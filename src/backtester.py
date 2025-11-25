import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

class Backtester:
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
        strategy_cfg = config.get('strategy', {})
        default_risk = strategy_cfg.get('risk_percent', self.risk_pct * 100)
        low_risk = strategy_cfg.get('risk_percent_low_vol', max(default_risk / 2, 0.1))
        self.strategy_risk_pct = default_risk / 100.0
        self.strategy_low_vol_pct = low_risk / 100.0
        
        self.equity_curve = []
        self.trades = []
        
    def run(self, candidates_df, probabilities):
        """
        Run simulation on labeled candidates with model probabilities.
        candidates_df: DataFrame with entry info and 'target' (actual outcome)
        probabilities: Model prediction probabilities
        """
        # Filter by threshold
        threshold = self.config['strategy']['model_threshold']
        
        # Add probs to df
        df = candidates_df.copy()
        df['prob'] = probabilities
        
        # Log statistics before filtering
        logger.info(f"\n{'='*60}")
        logger.info("PRE-FILTER STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total candidates: {len(df)}")
        logger.info(f"Mean probability: {df['prob'].mean():.3f}")
        logger.info(f"Median probability: {df['prob'].median():.3f}")
        logger.info(f"Std probability: {df['prob'].std():.3f}")
        logger.info(f"Min probability: {df['prob'].min():.3f}")
        logger.info(f"Max probability: {df['prob'].max():.3f}")
        logger.info(f"\nProbability distribution:")
        logger.info(f"  < 0.3: {(df['prob'] < 0.3).sum()} ({(df['prob'] < 0.3).sum()/len(df)*100:.1f}%)")
        logger.info(f"  0.3-0.4: {((df['prob'] >= 0.3) & (df['prob'] < 0.4)).sum()} ({((df['prob'] >= 0.3) & (df['prob'] < 0.4)).sum()/len(df)*100:.1f}%)")
        logger.info(f"  0.4-0.5: {((df['prob'] >= 0.4) & (df['prob'] < 0.5)).sum()} ({((df['prob'] >= 0.4) & (df['prob'] < 0.5)).sum()/len(df)*100:.1f}%)")
        logger.info(f"  0.5-0.6: {((df['prob'] >= 0.5) & (df['prob'] < 0.6)).sum()} ({((df['prob'] >= 0.5) & (df['prob'] < 0.6)).sum()/len(df)*100:.1f}%)")
        logger.info(f"  >= 0.6: {(df['prob'] >= 0.6).sum()} ({(df['prob'] >= 0.6).sum()/len(df)*100:.1f}%)")
        logger.info(f"\nUsing threshold: {threshold}")
        
        # Select trades
        trades_df = df[df['prob'] >= threshold].copy()
        logger.info(f"Candidates above threshold: {len(trades_df)} ({len(trades_df)/len(df)*100:.1f}%)")
        
        # Apply entry type filter if configured
        entry_type_filter = self.config['strategy'].get('entry_type_filter', 'both')
        if entry_type_filter == 'ob_only':
            before_filter = len(trades_df)
            trades_df = trades_df[trades_df['entry_type'] == 'ob'].copy()
            logger.info(f"After OB-only filter: {len(trades_df)} candidates (removed {before_filter - len(trades_df)} FVG)")
        elif entry_type_filter == 'fvg_only':
            before_filter = len(trades_df)
            trades_df = trades_df[trades_df['entry_type'] == 'fvg'].copy()
            logger.info(f"After FVG-only filter: {len(trades_df)} candidates (removed {before_filter - len(trades_df)} OB)")
        
        if len(trades_df) == 0:
            logger.warning("⚠️  NO TRADES PASSED THRESHOLD! Consider lowering threshold.")
            return pd.DataFrame(), pd.DataFrame()
        
        trades_df = trades_df.sort_values('entry_time')
        
        # Filter out trades too close together (cooldown period)
        # Prevent multiple trades within the same hour (or configurable period)
        cooldown_hours = 1  # Minimum hours between trades
        if len(trades_df) > 0:
            trades_df = trades_df.copy()
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            filtered_trades = []
            last_trade_time = None
            
            for idx, row in trades_df.iterrows():
                if last_trade_time is None:
                    filtered_trades.append(idx)
                    last_trade_time = row['entry_time']
                else:
                    hours_since_last = (row['entry_time'] - last_trade_time).total_seconds() / 3600
                    if hours_since_last >= cooldown_hours:
                        filtered_trades.append(idx)
                        last_trade_time = row['entry_time']
            
            trades_df = trades_df.loc[filtered_trades]
            logger.info(f"After cooldown filter: {len(trades_df)} trades (removed {len(df[df['prob'] >= threshold]) - len(trades_df)} clustered trades)")
        
        logger.info(f"Backtesting {len(trades_df)} trades out of {len(df)} candidates...")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}, Risk per trade: {self.risk_pct*100:.2f}%")
        
        current_equity = self.initial_capital
        cycle_start_time = trades_df['entry_time'].min() if not trades_df.empty else pd.Timestamp.now()
        total_withdrawn = 0
        max_dd_amount = self.initial_capital * self.max_dd_pct
        fail_level = self.initial_capital - max_dd_amount
        
        failed = False
        fail_reason = None
        
        history = []
        
        # Progress tracking (disabled for cleaner output)
        total_trades = len(trades_df)
        # log_interval = max(1, total_trades // 20)  # Log every 5% - DISABLED

        # Iterate through trades
        for trade_idx, (i, row) in enumerate(trades_df.iterrows()):
            if failed: break
            
            # Prop firm cycle check
            if self.mode == 'propfirm':
                days_since_start = (row['entry_time'] - cycle_start_time).days
                if days_since_start >= self.payout_days:
                    # Cycle end
                    # Withdraw profits
                    if current_equity > self.initial_capital:
                        profit = current_equity - self.initial_capital
                        total_withdrawn += profit
                        logger.info(f"Cycle payout: ${profit:.2f} at {row['entry_time']}")
                        current_equity = self.initial_capital # Reset
                    else:
                        # No profit, but equity carries over? 
                        # "reset trading equity back to $6,000" - Strict reset usually means topping up if negative?
                        # "reset trading equity back to $6,000 for the next period"
                        # If below 6000 but above fail level, usually you just keep trading or it's a reset.
                        # Prompt says "reset ... back to $6,000". I'll assume reset to 6000 regardless.
                        if current_equity < self.initial_capital:
                            logger.info(f"Cycle reset from ${current_equity:.2f} to ${self.initial_capital}")
                        current_equity = self.initial_capital
                        
                    cycle_start_time = row['entry_time']
            
            # Position Sizing
            # Adaptive risk per trade
            vol_regime = row.get('vol_regime', 'normal') if isinstance(row, pd.Series) else 'normal'
            risk_pct_trade = self.strategy_low_vol_pct if vol_regime == 'low' else self.strategy_risk_pct
            if self.mode == 'propfirm':
                risk_pct_trade = min(risk_pct_trade, self.risk_pct)
            risk_amount = self.initial_capital * risk_pct_trade
            
            # Calculate position size
            # risk_amount = pos_size * (entry - sl)
            dist_sl = abs(row['entry_price'] - row['sl'])
            if dist_sl == 0: continue
            
            # Lots = risk / (dist * pip_value) 
            # Simplified: units = risk / dist
            units = risk_amount / dist_sl
            
            # Execute trade
            # We already have the outcome from 'target' (1=win, 0=loss)
            # But we need precise PnL. 'pnl_r' is R-multiple.
            
            # Gross PnL
            # If target 1 (Win): PnL = units * (TP - Entry)
            # If target 0 (Loss): PnL = units * (SL - Entry)
            
            # Wait, the labeling logic set target=0 if SL hit.
            # But we need to know if it was a full loss or partial?
            # The entries module set pnl_r = -1 for loss, or RewardRatio for win.
            
            gross_pnl = units * dist_sl * row['pnl_r']
            
            # Apply commission & slippage
            # Commission: Usually per lot or per trade for gold
            # Config: commission: 0.0005 (interpreted as $ per unit, or can be % of notional)
            # For gold, typical commission is $5-10 per lot or ~$0.50 per oz
            # Slippage: 0.0003 (in price terms, e.g., $0.30 per oz)
            
            # Slippage affects entry and exit price
            # Slippage cost = units * slippage_per_unit * 2 (entry + exit)
            slippage_cost = units * self.slippage * 2
            
            # Commission: If commission is per unit, use that. Otherwise treat as % of notional.
            # For now, treat commission as per-unit cost (more realistic for gold)
            # If you want % of notional, use: (units * row['entry_price']) * self.commission
            comm_cost = units * self.commission * 2  # Entry and exit
            
            net_pnl = gross_pnl - slippage_cost - comm_cost
            
            current_equity += net_pnl
            
            # Progress logging - DISABLED for cleaner output
            # if trade_idx % log_interval == 0 or trade_idx == total_trades - 1:
            #     pct = (trade_idx + 1) / total_trades * 100
            #     wins = sum(1 for t in self.trades if t.get('net_pnl', 0) > 0)
            #     losses = len(self.trades) - wins
            #     win_rate = wins / len(self.trades) * 100 if self.trades else 0
            #     logger.info(f"Progress: {pct:.1f}% ({trade_idx+1}/{total_trades}) | Equity: ${current_equity:.2f} | Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
            
            # Record
            trade_record = row.to_dict()
            trade_record['net_pnl'] = net_pnl
            trade_record['equity_after'] = current_equity
            trade_record['units'] = units
            trade_record['risk_pct_used'] = risk_pct_trade
            self.trades.append(trade_record)
            
            history.append({
                'time': row['exit_time'] if pd.notna(row.get('exit_time')) else row['entry_time'], # Approx
                'equity': current_equity,
                'withdrawn': total_withdrawn
            })
            
            # Check Drawdown (Prop firm hard stop)
            if self.mode == 'propfirm':
                if current_equity <= fail_level:
                    failed = True
                    fail_reason = f"Max drawdown reached. Equity: {current_equity:.2f}, Limit: {fail_level:.2f}"
                    logger.warning(fail_reason)
        
        # Final summary
        final_wins = sum(1 for t in self.trades if t.get('net_pnl', 0) > 0)
        final_losses = len(self.trades) - final_wins
        final_win_rate = final_wins / len(self.trades) * 100 if self.trades else 0
        total_pnl = sum(t.get('net_pnl', 0) for t in self.trades)
        logger.info(f"Backtest complete! Final Equity: ${current_equity:.2f} | Total PnL: ${total_pnl:.2f} | Win Rate: {final_win_rate:.1f}% ({final_wins}W/{final_losses}L)")
        
        return pd.DataFrame(history), pd.DataFrame(self.trades)



