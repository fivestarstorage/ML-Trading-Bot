import os
import json
from datetime import datetime
import numpy as np


class AdaptiveManager:
    def __init__(self, config):
        self.config = config
        adaptive_cfg = config.get('adaptive', {})
        root = os.path.dirname(os.path.dirname(__file__))
        self.state_path = adaptive_cfg.get('state_path', 'adaptive_state.json')
        if not os.path.isabs(self.state_path):
            self.state_path = os.path.join(root, self.state_path)
        self.enabled = adaptive_cfg.get('enabled', True)
        self.risk_bounds = adaptive_cfg.get('risk_bounds', {'min': 0.0035, 'max': 0.006})
        self.risk_step = adaptive_cfg.get('risk_step', 0.0005)

    def apply_to_config(self, config):
        if not self.enabled:
            return
        state = self._load_state()
        if not state:
            return
        threshold = state.get('recommended_threshold')
        risk_pct = state.get('recommended_risk_pct')
        if threshold:
            config['strategy']['model_threshold'] = threshold
        if risk_pct:
            config['backtest']['propfirm']['per_trade_risk_pct'] = risk_pct
            config['strategy']['risk_percent'] = round(risk_pct * 100, 3)
            config['strategy']['risk_percent_low_vol'] = round(max(config['strategy'].get('risk_percent_low_vol', 0.35), risk_pct * 100 * 0.7), 3)
        profile = state.get('recommended_profile')
        if profile:
            config.setdefault('adaptive', {})['active_profile'] = profile

    def update_from_folds(self, fold_summaries):
        if not self.enabled or not fold_summaries:
            return
        adaptive_cfg = self.config.get('adaptive', {})
        min_trades = adaptive_cfg.get('min_trades_per_fold', 30)
        filtered = [f for f in fold_summaries if not f.get('skip_month') and f.get('best_threshold') is not None and f['trade_count'] >= min_trades]
        if not filtered:
            filtered = [f for f in fold_summaries if not f.get('skip_month') and f.get('best_threshold') is not None]
        if not filtered:
            print("Adaptive manager: no usable folds found; skipping state update.")
            return
        weights = np.array([max(f.get('net_profit', 0.0), 0.01) for f in filtered])
        thresholds = np.array([f['best_threshold'] for f in filtered])
        win_rates = np.array([f['win_rate'] for f in filtered])
        drawdowns = np.array([f['max_drawdown'] for f in filtered])
        base_threshold = self.config['strategy']['model_threshold']
        base_risk = self.config['backtest']['propfirm']['per_trade_risk_pct']
        if weights.sum() == 0:
            recommended_threshold = base_threshold
        else:
            recommended_threshold = float(np.clip(np.average(thresholds, weights=weights), min(thresholds), max(thresholds)))

        avg_wr = win_rates.mean() if len(win_rates) else 0.0
        avg_dd = drawdowns.mean() if len(drawdowns) else 0.0
        recommended_risk = base_risk
        if avg_wr > 0.6 and avg_dd < 4.0:
            recommended_risk = min(base_risk + self.risk_step, self.risk_bounds.get('max', base_risk))
        elif avg_wr < 0.5 or avg_dd > 6.0:
            recommended_risk = max(base_risk - self.risk_step, self.risk_bounds.get('min', base_risk))

        profile_scores = {}
        for f in filtered:
            name = f.get('profile', 'baseline')
            profile_scores[name] = profile_scores.get(name, 0.0) + f.get('net_profit', 0.0)
        recommended_profile = max(profile_scores, key=profile_scores.get) if profile_scores else None

        state = {
            'last_updated': datetime.utcnow().isoformat(),
            'folds': fold_summaries,
            'recommended_threshold': round(recommended_threshold, 3),
            'recommended_risk_pct': round(recommended_risk, 6),
            'recommended_profile': recommended_profile,
            'averages': {
                'win_rate': avg_wr,
                'drawdown': avg_dd
            }
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
        profile_msg = f", profile {recommended_profile}" if recommended_profile else ""
        print(f"Adaptive state updated ➜ {self.state_path} (threshold {state['recommended_threshold']}, risk {state['recommended_risk_pct']:.4f}{profile_msg})")

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return None
        with open(self.state_path, 'r') as f:
            return json.load(f)

