"""
Seasonal and time-based hypothesis generation.

Focuses on discovering temporal patterns:
- Hour-of-day effects
- Day-of-week effects
- Monthly patterns
- Holiday effects
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats

from .base import (
    HypothesisGenerator, 
    Hypothesis, 
    HypothesisType, 
    HypothesisMechanism
)


class SeasonalHypothesisGenerator(HypothesisGenerator):
    """
    Generates hypotheses based on seasonal and temporal patterns.
    
    Key patterns searched:
    - Time-of-day effects
    - Day-of-week effects
    - Monthly seasonality
    - Turn-of-month effects
    - Calendar anomalies
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_periods = config.get('seasonal_test_periods', ['hour', 'day_of_week', 'month'])
        self.min_samples_per_period = 30
    
    def get_hypothesis_type(self) -> HypothesisType:
        return HypothesisType.SEASONAL
    
    def generate(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Generate seasonal hypotheses."""
        data = self.preprocess_data(data)
        hypotheses = []
        
        # 1. Hour-of-day effects
        if 'hour' in self.test_periods:
            hour_hypotheses = self._generate_hourly_hypotheses(data)
            hypotheses.extend(hour_hypotheses)
        
        # 2. Day-of-week effects
        if 'day_of_week' in self.test_periods:
            dow_hypotheses = self._generate_dow_hypotheses(data)
            hypotheses.extend(dow_hypotheses)
        
        # 3. Monthly effects
        if 'month' in self.test_periods:
            month_hypotheses = self._generate_monthly_hypotheses(data)
            hypotheses.extend(month_hypotheses)
        
        # 4. Turn-of-month effect
        tom_hypotheses = self._generate_turn_of_month_hypotheses(data)
        hypotheses.extend(tom_hypotheses)
        
        # 5. Session-based hypotheses
        session_hypotheses = self._generate_session_hypotheses(data)
        hypotheses.extend(session_hypotheses)
        
        return hypotheses
    
    def _generate_hourly_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Analyze hour-of-day effects."""
        hypotheses = []
        
        if not hasattr(data.index, 'hour'):
            return hypotheses
        
        data['hour'] = data.index.hour
        
        # Calculate hourly statistics
        hourly_stats = data.groupby('hour')['returns'].agg(['mean', 'std', 'count'])
        hourly_stats['t_stat'] = hourly_stats['mean'] / (hourly_stats['std'] / np.sqrt(hourly_stats['count']))
        hourly_stats['sharpe'] = hourly_stats['mean'] / hourly_stats['std']
        
        # Find significant hours
        significant = hourly_stats[
            (hourly_stats['count'] >= self.min_samples_per_period) &
            (hourly_stats['t_stat'].abs() > 1.96)
        ]
        
        if len(significant) > 0:
            # Get best and worst hours
            best_hour = hourly_stats['sharpe'].idxmax()
            worst_hour = hourly_stats['sharpe'].idxmin()
            
            def hourly_signal(df, good_hours, bad_hours):
                signals = pd.Series(0, index=df.index)
                
                if 'hour' not in df.columns:
                    df['hour'] = df.index.hour
                
                for hour in good_hours:
                    signals[df['hour'] == hour] = 1
                for hour in bad_hours:
                    signals[df['hour'] == hour] = -1
                
                return signals
            
            # Find all significantly positive/negative hours
            good_hours = hourly_stats[hourly_stats['t_stat'] > 1.5].index.tolist()
            bad_hours = hourly_stats[hourly_stats['t_stat'] < -1.5].index.tolist()
            
            if good_hours or bad_hours:
                hypotheses.append(Hypothesis(
                    id=self._create_hypothesis_id("hourly_effect"),
                    name="Hour-of-Day Effect",
                    description=f"Detected significant hourly patterns. "
                               f"Best: hour {best_hour} ({hourly_stats.loc[best_hour, 'sharpe']:.3f} Sharpe), "
                               f"Worst: hour {worst_hour} ({hourly_stats.loc[worst_hour, 'sharpe']:.3f} Sharpe).",
                    hypothesis_type=HypothesisType.SEASONAL,
                    mechanism=HypothesisMechanism.BEHAVIORAL,
                    signal_generator=hourly_signal,
                    parameters={'good_hours': good_hours, 'bad_hours': bad_hours},
                    required_features=['hour'],
                    economic_rationale="Intraday patterns emerge from market microstructure, "
                                      "institutional trading schedules, and global market overlaps.",
                    source="seasonal_generator",
                    priority=5,
                    preliminary_stats={
                        'n_significant_hours': len(significant),
                        'best_hour': int(best_hour),
                        'best_hour_sharpe': float(hourly_stats.loc[best_hour, 'sharpe']),
                        'worst_hour': int(worst_hour),
                        'worst_hour_sharpe': float(hourly_stats.loc[worst_hour, 'sharpe']),
                    }
                ))
        
        return hypotheses
    
    def _generate_dow_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Analyze day-of-week effects."""
        hypotheses = []
        
        data['dow'] = data.index.dayofweek
        
        # Calculate daily statistics
        dow_stats = data.groupby('dow')['returns'].agg(['mean', 'std', 'count'])
        dow_stats['t_stat'] = dow_stats['mean'] / (dow_stats['std'] / np.sqrt(dow_stats['count']))
        dow_stats['sharpe'] = dow_stats['mean'] / dow_stats['std']
        
        dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        # Check for significant day-of-week effects
        significant = dow_stats[dow_stats['t_stat'].abs() > 1.5]
        
        if len(significant) > 0:
            best_day = dow_stats['sharpe'].idxmax()
            worst_day = dow_stats['sharpe'].idxmin()
            
            def dow_signal(df, good_days, bad_days):
                signals = pd.Series(0, index=df.index)
                
                if 'dow' not in df.columns:
                    df['dow'] = df.index.dayofweek
                
                for day in good_days:
                    signals[df['dow'] == day] = 1
                for day in bad_days:
                    signals[df['dow'] == day] = -1
                
                return signals
            
            good_days = dow_stats[dow_stats['t_stat'] > 1.0].index.tolist()
            bad_days = dow_stats[dow_stats['t_stat'] < -1.0].index.tolist()
            
            hypotheses.append(Hypothesis(
                id=self._create_hypothesis_id("dow_effect"),
                name="Day-of-Week Effect",
                description=f"Detected day-of-week patterns. "
                           f"Best: {dow_names.get(best_day, best_day)}, "
                           f"Worst: {dow_names.get(worst_day, worst_day)}.",
                hypothesis_type=HypothesisType.SEASONAL,
                mechanism=HypothesisMechanism.BEHAVIORAL,
                signal_generator=dow_signal,
                parameters={'good_days': good_days, 'bad_days': bad_days},
                required_features=['dow'],
                economic_rationale="Day-of-week effects can arise from institutional "
                                  "rebalancing schedules, options expiry, and "
                                  "weekend information accumulation.",
                source="seasonal_generator",
                priority=5,
                preliminary_stats={
                    'best_day': dow_names.get(best_day, str(best_day)),
                    'best_day_sharpe': float(dow_stats.loc[best_day, 'sharpe']),
                    'worst_day': dow_names.get(worst_day, str(worst_day)),
                    'worst_day_sharpe': float(dow_stats.loc[worst_day, 'sharpe']),
                }
            ))
        
        return hypotheses
    
    def _generate_monthly_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Analyze monthly seasonality."""
        hypotheses = []
        
        data['month'] = data.index.month
        
        # Need sufficient data for monthly analysis
        n_months = data.groupby(data.index.to_period('M')).ngroups
        if n_months < 24:  # At least 2 years of data
            return hypotheses
        
        # Calculate monthly statistics
        monthly_stats = data.groupby('month')['returns'].agg(['mean', 'std', 'count'])
        monthly_stats['t_stat'] = monthly_stats['mean'] / (monthly_stats['std'] / np.sqrt(monthly_stats['count']))
        monthly_stats['sharpe'] = monthly_stats['mean'] / monthly_stats['std']
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        # Check for significant monthly effects
        significant = monthly_stats[monthly_stats['t_stat'].abs() > 1.5]
        
        if len(significant) > 0:
            best_month = monthly_stats['sharpe'].idxmax()
            worst_month = monthly_stats['sharpe'].idxmin()
            
            def monthly_signal(df, good_months, bad_months):
                signals = pd.Series(0, index=df.index)
                
                if 'month' not in df.columns:
                    df['month'] = df.index.month
                
                for month in good_months:
                    signals[df['month'] == month] = 1
                for month in bad_months:
                    signals[df['month'] == month] = -1
                
                return signals
            
            good_months = monthly_stats[monthly_stats['t_stat'] > 1.0].index.tolist()
            bad_months = monthly_stats[monthly_stats['t_stat'] < -1.0].index.tolist()
            
            hypotheses.append(Hypothesis(
                id=self._create_hypothesis_id("monthly_effect"),
                name="Monthly Seasonality",
                description=f"Detected monthly patterns. "
                           f"Best: {month_names.get(best_month, best_month)}, "
                           f"Worst: {month_names.get(worst_month, worst_month)}.",
                hypothesis_type=HypothesisType.SEASONAL,
                mechanism=HypothesisMechanism.BEHAVIORAL,
                signal_generator=monthly_signal,
                parameters={'good_months': good_months, 'bad_months': bad_months},
                required_features=['month'],
                economic_rationale="Monthly seasonality can arise from tax-related "
                                  "trading, fund flows, earnings cycles, and "
                                  "seasonal business patterns.",
                source="seasonal_generator",
                priority=4,
                preliminary_stats={
                    'best_month': month_names.get(best_month, str(best_month)),
                    'best_month_sharpe': float(monthly_stats.loc[best_month, 'sharpe']),
                    'worst_month': month_names.get(worst_month, str(worst_month)),
                    'worst_month_sharpe': float(monthly_stats.loc[worst_month, 'sharpe']),
                }
            ))
        
        return hypotheses
    
    def _generate_turn_of_month_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Analyze turn-of-month effect."""
        hypotheses = []
        
        data['day'] = data.index.day
        data['days_in_month'] = data.index.to_series().apply(
            lambda x: pd.Timestamp(x).days_in_month
        )
        
        # Define turn-of-month: last 2 days and first 3 days
        data['is_tom'] = (
            (data['day'] <= 3) | 
            (data['day'] >= data['days_in_month'] - 1)
        )
        
        # Compare TOM vs non-TOM returns
        tom_returns = data[data['is_tom']]['returns']
        non_tom_returns = data[~data['is_tom']]['returns']
        
        if len(tom_returns) >= 50 and len(non_tom_returns) >= 50:
            t_stat, p_value = stats.ttest_ind(tom_returns.dropna(), non_tom_returns.dropna())
            
            tom_mean = tom_returns.mean()
            non_tom_mean = non_tom_returns.mean()
            
            if p_value < 0.1:  # Marginally significant
                def tom_signal(df):
                    signals = pd.Series(0, index=df.index)
                    
                    if 'is_tom' not in df.columns:
                        day = df.index.day
                        days_in_month = df.index.to_series().apply(
                            lambda x: pd.Timestamp(x).days_in_month
                        )
                        is_tom = (day <= 3) | (day >= days_in_month - 1)
                    else:
                        is_tom = df['is_tom']
                    
                    # Long during TOM if TOM returns are higher
                    if tom_mean > non_tom_mean:
                        signals[is_tom] = 1
                    else:
                        signals[is_tom] = -1
                    
                    return signals
                
                hypotheses.append(Hypothesis(
                    id=self._create_hypothesis_id("turn_of_month"),
                    name="Turn-of-Month Effect",
                    description=f"Turn-of-month periods show significantly "
                               f"{'higher' if tom_mean > non_tom_mean else 'lower'} returns.",
                    hypothesis_type=HypothesisType.SEASONAL,
                    mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
                    signal_generator=tom_signal,
                    parameters={},
                    required_features=['is_tom'],
                    economic_rationale="Turn-of-month effects are documented in many markets, "
                                      "potentially driven by payroll-related investing, "
                                      "fund flows, and window dressing.",
                    source="seasonal_generator",
                    priority=5,
                    preliminary_stats={
                        'tom_mean_return': float(tom_mean),
                        'non_tom_mean_return': float(non_tom_mean),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                    }
                ))
        
        return hypotheses
    
    def _generate_session_hypotheses(self, data: pd.DataFrame) -> List[Hypothesis]:
        """Analyze trading session effects."""
        hypotheses = []
        
        if not hasattr(data.index, 'hour'):
            return hypotheses
        
        hour = data.index.hour
        
        # Define sessions (UTC times, adjust as needed)
        data['session'] = 'off'
        data.loc[(hour >= 0) & (hour < 8), 'session'] = 'asia'
        data.loc[(hour >= 8) & (hour < 14), 'session'] = 'london'
        data.loc[(hour >= 14) & (hour < 21), 'session'] = 'new_york'
        
        # Calculate session statistics
        session_stats = data.groupby('session')['returns'].agg(['mean', 'std', 'count'])
        session_stats['sharpe'] = session_stats['mean'] / session_stats['std']
        
        # Check for significant differences
        if session_stats['count'].min() >= 100:
            best_session = session_stats['sharpe'].idxmax()
            worst_session = session_stats['sharpe'].idxmin()
            
            def session_signal(df, good_sessions, bad_sessions):
                signals = pd.Series(0, index=df.index)
                
                if 'session' not in df.columns:
                    hour = df.index.hour
                    session = pd.Series('off', index=df.index)
                    session[(hour >= 0) & (hour < 8)] = 'asia'
                    session[(hour >= 8) & (hour < 14)] = 'london'
                    session[(hour >= 14) & (hour < 21)] = 'new_york'
                else:
                    session = df['session']
                
                for s in good_sessions:
                    signals[session == s] = 1
                for s in bad_sessions:
                    signals[session == s] = -1
                
                return signals
            
            good_sessions = session_stats[session_stats['sharpe'] > 0.01].index.tolist()
            bad_sessions = session_stats[session_stats['sharpe'] < -0.01].index.tolist()
            
            hypotheses.append(Hypothesis(
                id=self._create_hypothesis_id("session_effect"),
                name="Trading Session Effect",
                description=f"Detected session-based patterns. "
                           f"Best: {best_session} ({session_stats.loc[best_session, 'sharpe']:.3f} Sharpe).",
                hypothesis_type=HypothesisType.SEASONAL,
                mechanism=HypothesisMechanism.STRUCTURAL_IMBALANCE,
                signal_generator=session_signal,
                parameters={'good_sessions': good_sessions, 'bad_sessions': bad_sessions},
                required_features=['session'],
                economic_rationale="Different trading sessions have different participants, "
                                  "liquidity profiles, and information flows.",
                source="seasonal_generator",
                priority=6,
                preliminary_stats={
                    'best_session': best_session,
                    'best_session_sharpe': float(session_stats.loc[best_session, 'sharpe']),
                    'asia_sharpe': float(session_stats.loc['asia', 'sharpe']) if 'asia' in session_stats.index else 0,
                    'london_sharpe': float(session_stats.loc['london', 'sharpe']) if 'london' in session_stats.index else 0,
                    'new_york_sharpe': float(session_stats.loc['new_york', 'sharpe']) if 'new_york' in session_stats.index else 0,
                }
            ))
        
        return hypotheses

