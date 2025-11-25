import copy

SESSION_LABEL_TO_CODE = {
    'asia': 1,
    'london': 2,
    'new_york': 3,
    'off': 0
}


class FilterProfileManager:
    """Applies SMC filter profiles to candidate dataframes."""

    def __init__(self, config):
        adaptive_cfg = config.get('adaptive', {})
        self.profiles = adaptive_cfg.get('filter_profiles', {}) or {}
        if 'baseline' not in self.profiles:
            self.profiles['baseline'] = {'description': 'No additional filtering'}

    def list_profiles(self):
        return list(self.profiles.keys())

    def describe(self, profile_name):
        profile = self.profiles.get(profile_name, {})
        return copy.deepcopy(profile)

    def apply(self, profile_name, df):
        profile = self.profiles.get(profile_name, {})
        if not profile or profile_name == 'baseline':
            return df.copy(), {'profile': profile_name or 'baseline', 'initial': len(df), 'filtered': len(df)}

        filtered = df.copy()
        if 'session_labels' in profile:
            labels = set([s.lower() for s in profile['session_labels']])
            filtered = filtered[filtered['session_label'].str.lower().isin(labels)]

        if 'session_codes' in profile:
            filtered = filtered[filtered['session_code'].isin(profile['session_codes'])]

        if profile.get('allow_off_session') is False:
            filtered = filtered[filtered['session_label'].str.lower() != 'off']

        if profile.get('require_killzone'):
            filtered = filtered[filtered['in_killzone'] == True]  # noqa: E712

        if profile.get('require_liquidity_sweep'):
            filtered = filtered[filtered['liquidity_sweep'] == 1]

        min_disp = profile.get('min_displacement')
        if min_disp is not None and 'displacement_factor' in filtered.columns:
            filtered = filtered[filtered['displacement_factor'] >= float(min_disp)]

        max_disp = profile.get('max_displacement')
        if max_disp is not None and 'displacement_factor' in filtered.columns:
            filtered = filtered[filtered['displacement_factor'] <= float(max_disp)]

        vol_regimes = profile.get('vol_regimes')
        if vol_regimes and 'vol_regime' in filtered.columns:
            wanted = [v.lower() for v in vol_regimes]
            filtered = filtered[filtered['vol_regime'].str.lower().isin(wanted)]

        daily_trend = profile.get('daily_trend')
        if daily_trend and 'daily_trend_label' in filtered.columns:
            filtered = filtered[filtered['daily_trend_label'].str.lower() == daily_trend.lower()]

        metadata = {
            'profile': profile_name,
            'initial': len(df),
            'filtered': len(filtered)
        }
        return filtered, metadata

    def active_profile_from_config(self, config):
        return config.get('adaptive', {}).get('active_profile')

    def apply_active(self, config, df):
        profile_name = self.active_profile_from_config(config)
        if not profile_name:
            return df, None
        filtered, meta = self.apply(profile_name, df)
        return filtered, meta

