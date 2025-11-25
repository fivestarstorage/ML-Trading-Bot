import pandas as pd
import numpy as np
import copy
from .utils import get_logger
from .ml_model import MLModel
from .backtester import Backtester
from .adaptive_filters import FilterProfileManager

logger = get_logger()

class WalkForwardAnalyzer:
    def __init__(self, config):
        self.config = config
        self.train_years = config['wfa']['train_years']
        self.min_train_samples = config['wfa'].get('min_train_samples', 200)
        self.filter_manager = FilterProfileManager(config)
        adaptive_cfg = config.get('adaptive', {})
        guard_cfg = adaptive_cfg.get('session_guard', {})
        self.session_guard_enabled = guard_cfg.get('enabled', False)
        self.session_guard_threshold = guard_cfg.get('win_rate_threshold', 0.45)
        self.session_guard_min_samples = guard_cfg.get('min_samples', 25)
    
    def sequential_walk(self, candidates_df, start_date, end_date):
        """
        Train on the previous `train_years` of data, score candidates between
        start_date and end_date, and after each prediction allow that trade
        to become part of the training window for subsequent predictions.
        """
        if start_date is None or end_date is None:
            raise ValueError("Walk-forward analysis requires both start and end dates.")
        
        numeric_cols = [
            'atr','rsi','adx','vol_spike','rolling_vol','hour','dayofweek',
            'dist_to_range_low','dist_to_range_high','session_code','daily_trend',
            'displacement_factor','liquidity_sweep','liquidity_sweep_dir',
            'atr_ratio','volume_zscore','vol_regime_code','premium_pct',
            'discount_pct','body_factor','impulse_factor','holding_minutes'
        ]
        for col in numeric_cols:
            if col in candidates_df.columns:
                candidates_df[col] = pd.to_numeric(candidates_df[col], errors='coerce')
        bool_cols = ['in_killzone']
        for col in bool_cols:
            if col in candidates_df.columns:
                candidates_df[col] = candidates_df[col].astype(bool)
        df = candidates_df.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        if df['entry_time'].dt.tz is None:
            df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')
        else:
            df['entry_time'] = df['entry_time'].dt.tz_convert('UTC')
        df = df.sort_values('entry_time').reset_index(drop=True)

        def _normalize_ts(value):
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            return ts

        start_ts = _normalize_ts(start_date)
        end_ts = _normalize_ts(end_date)
        window = pd.DateOffset(years=self.train_years)
        
        test_mask = (df['entry_time'] >= start_ts) & (df['entry_time'] <= end_ts)
        test_rows = df[test_mask]
        if test_rows.empty:
            raise ValueError("No candidates found within the requested walk-forward period.")
        
        results = []
        
        adaptive_cfg = self.config.get('adaptive', {})
        min_fold_win = adaptive_cfg.get('min_fold_win_rate', 0.55)
        skip_negative = adaptive_cfg.get('skip_negative_months', True)
        month_offset = pd.DateOffset(months=1)
        current_test_start = start_ts
        fold = 0
        fold_summaries = []
        threshold_grid = adaptive_cfg.get('threshold_grid', [0.55, 0.58, 0.6, 0.62, 0.65])
        min_trades_threshold = adaptive_cfg.get('min_trades_per_fold', 30)
        while current_test_start < end_ts:
            fold += 1
            train_start = current_test_start - window
            train_mask = (df['entry_time'] >= train_start) & (df['entry_time'] < current_test_start)
            train_data = df[train_mask]
            
            if len(train_data) < self.min_train_samples:
                logger.warning(f"[WFA] Fold {fold}: insufficient training samples ({len(train_data)}) for window ending {current_test_start}. Skipping month.")
                current_test_start += month_offset
                continue
            
            test_end = min(current_test_start + month_offset, end_ts)
            test_mask = (df['entry_time'] >= current_test_start) & (df['entry_time'] < test_end)
            test_data = df[test_mask]
            if test_data.empty:
                logger.info(f"[WFA] Fold {fold}: no trades between {current_test_start.date()} and {test_end.date()}, skipping.")
                current_test_start += month_offset
                continue
            
            logger.info(f"[WFA] Fold {fold}: base train {len(train_data)} samples ({train_start.date()} -> {current_test_start.date()}), scoring {len(test_data)} trades.")

            profile_candidates = []
            entry_filters = [
                ('both', lambda df: df),
                ('ob_only', lambda df: df[df['entry_type'] == 'ob']),
                ('fvg_only', lambda df: df[df['entry_type'] == 'fvg'])
            ]
            session_filters = [
                ('all_sessions', lambda df: df),
                ('no_london', lambda df: df[df['session_label'] != 'london']),
                ('ny_only', lambda df: df[df['session_label'] == 'new_york']),
                ('ny_off', lambda df: df[df['session_label'].isin(['new_york', 'off'])]),
                ('killzone_only', lambda df: df[df['session_label'].isin(['london', 'new_york'])]),
                ('london_only', lambda df: df[df['session_label'] == 'london']),
                ('off_only', lambda df: df[df['session_label'] == 'off']),
                ('asia_only', lambda df: df[df['session_label'] == 'asia'])
            ]
            for profile_name in self.filter_manager.list_profiles():
                train_filtered, train_meta = self.filter_manager.apply(profile_name, train_data)
                if len(train_filtered) < self.min_train_samples:
                    continue
                test_filtered, test_meta = self.filter_manager.apply(profile_name, test_data)
                if test_filtered.empty:
                    continue

                if self.session_guard_enabled:
                    guard_train, guard_test = self._apply_session_guard(
                        train_filtered,
                        test_filtered
                    )
                    if len(guard_train) < self.min_train_samples or guard_test.empty:
                        continue
                    train_filtered = guard_train
                    test_filtered = guard_test

                logger.info(f"[WFA] Fold {fold}: profile '{profile_name}' -> train {len(train_filtered)}, test {len(test_filtered)}")
                model = MLModel(self.config)
                model.train(train_filtered)
                probs = model.predict_proba(test_filtered)
                scored = test_filtered.copy()
                scored['wfa_prob'] = probs

                for entry_filter_name, entry_filter_fn in entry_filters:
                    entry_filtered = entry_filter_fn(scored)
                    if entry_filtered.empty:
                        continue
                    for session_filter_name, session_filter_fn in session_filters:
                        filtered_scored = session_filter_fn(entry_filtered)
                        if filtered_scored.empty:
                            continue

                        summary = self._evaluate_thresholds(filtered_scored, threshold_grid, min_trades_threshold, min_fold_win, skip_negative)
                        summary.update({
                            'fold': fold,
                            'start': current_test_start.isoformat(),
                            'end': test_end.isoformat(),
                            'profile': profile_name,
                            'entry_filter': entry_filter_name,
                            'session_filter': session_filter_name,
                            'train_count': len(train_filtered),
                            'test_count': len(filtered_scored),
                            'raw_candidates': len(test_data)
                        })
                        profile_candidates.append((self._score_profile(summary), summary, filtered_scored))

            if not profile_candidates:
                logger.info(f"[WFA] Fold {fold}: no filter profile met minimum criteria, skipping month.")
                current_test_start = test_end
                continue

            profile_candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_summary, best_scored = profile_candidates[0]
            fold_summaries.append(best_summary)
            if best_summary.get('skip_month'):
                logger.info(f"[WFA] Fold {fold} skipped (profile '{best_summary.get('profile')}').")
            else:
                best_scored['wfa_threshold'] = best_summary['best_threshold']
                best_scored['fold_id'] = fold
                best_scored['filter_profile'] = best_summary.get('profile', 'baseline')
                best_scored['entry_filter'] = best_summary.get('entry_filter', 'both')
                best_scored['session_filter'] = best_summary.get('session_filter', 'all_sessions')
                results.append(best_scored)
            
            current_test_start = test_end
            
        if not results:
            raise ValueError("Walk-forward produced no predictions. Check date range or training window.")
        
        full = pd.concat(results).sort_values('entry_time')
        filtered = full[full['wfa_prob'] >= full['wfa_threshold']].copy()
        if filtered.empty:
            logger.warning("All trades filtered out by adaptive thresholds; falling back to highest probabilities.")
            filtered = full.sort_values('wfa_prob', ascending=False).head(50).copy()
        logger.info(f"Sequential WFA produced {len(filtered)} filtered trades across {fold} folds (from {len(full)} raw).")
        return filtered, fold_summaries
    
    def _evaluate_thresholds(self, scored_df, threshold_grid, min_trades, min_win_rate, skip_negative):
        best = None
        fallback = None
        for threshold in threshold_grid:
            temp_config = copy.deepcopy(self.config)
            temp_config['strategy']['model_threshold'] = threshold
            backtester = Backtester(temp_config)
            history, trades = backtester.run(scored_df.copy(), scored_df['wfa_prob'])
            trade_count = len(trades)
            if trade_count < min_trades:
                continue
            net_profit = trades['net_pnl'].sum()
            wins = trades[trades['net_pnl'] > 0]
            win_rate = float(len(wins) / trade_count) if trade_count else 0.0
            max_dd = self._max_drawdown(history['equity']) if not history.empty else 0.0
            summary = {
                'best_threshold': threshold,
                'net_profit': float(net_profit),
                'win_rate': win_rate,
                'max_drawdown': max_dd,
                'trade_count': trade_count
            }
            if fallback is None or win_rate > fallback['win_rate']:
                fallback = summary
            if win_rate >= min_win_rate and net_profit > 0:
                if best is None or net_profit > best['net_profit']:
                    best = summary
        if best is None and not skip_negative:
            best = fallback if fallback else {
                'best_threshold': self.config['strategy']['model_threshold'],
                'net_profit': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'trade_count': len(scored_df)
            }
        if best is None:
            return {'skip_month': True, 'trade_count': len(scored_df), 'net_profit': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0}
        if best['trade_count'] < min_trades:
            if skip_negative:
                return {'skip_month': True, 'trade_count': best['trade_count'], 'net_profit': best['net_profit'], 'win_rate': best['win_rate'], 'max_drawdown': best['max_drawdown']}
            best = {
                'best_threshold': best['best_threshold'],
                'net_profit': best['net_profit'],
                'win_rate': best['win_rate'],
                'max_drawdown': best['max_drawdown'],
                'trade_count': best['trade_count']
            }
        best['skip_month'] = False
        return best
    
    def _score_profile(self, summary):
        if summary.get('skip_month'):
            return -1.0
        net = summary.get('net_profit', 0.0)
        win = summary.get('win_rate', 0.0)
        return net * max(win, 0.5)
    
    def _max_drawdown(self, equity_series):
        if equity_series.empty:
            return 0.0
        arr = equity_series.values
        running_max = np.maximum.accumulate(arr)
        drawdowns = (running_max - arr) / running_max * 100
        return float(np.nanmax(drawdowns))

    def _apply_session_guard(self, train_df, test_df):
        if 'session_label' not in train_df.columns or 'target' not in train_df.columns:
            return train_df, test_df
        stats = train_df.groupby('session_label').agg(
            wins=('target', 'sum'),
            count=('target', 'count')
        )
        if stats.empty:
            return train_df, test_df
        weak_sessions = stats[
            (stats['count'] >= self.session_guard_min_samples) &
            ((stats['wins'] / stats['count']) < self.session_guard_threshold)
        ].index.tolist()
        if not weak_sessions:
            return train_df, test_df
        remaining = set(train_df['session_label'].unique()) - set(weak_sessions)
        if not remaining:
            return train_df, test_df
        logger.info(
            "[SessionGuard] Dropping weak sessions %s (threshold %.2f, min_samples %d)",
            weak_sessions,
            self.session_guard_threshold,
            self.session_guard_min_samples
        )
        guarded_train = train_df[~train_df['session_label'].isin(weak_sessions)]
        guarded_test = test_df[~test_df['session_label'].isin(weak_sessions)]
        return guarded_train, guarded_test

