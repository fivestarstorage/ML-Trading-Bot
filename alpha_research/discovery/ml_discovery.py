"""
Machine Learning-based Alpha Discovery

Uses ML models to discover patterns with proper
validation and explainability requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

from ..utils.statistics import calculate_sharpe_ratio
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class MLEdgeResult:
    """Results from ML-based edge discovery."""
    model_type: str
    feature_importance: Dict[str, float]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    
    # Explainability
    top_features: List[str]
    feature_attributions: Dict[str, Any]
    
    # Validation
    cv_sharpes: List[float]
    regime_performance: Dict[str, float]
    noise_sensitivity: float
    
    # Status
    is_valid: bool
    rejection_reason: Optional[str] = None


class MLAlphaDiscovery:
    """
    Discovers alpha using machine learning with strict validation.
    
    Requirements for ML-discovered edges:
    1. Must identify patterns that produce positive expected returns
    2. Must provide explanations via feature attribution
    3. Must retain performance across different periods and regimes
    4. Must reject edges that fail outside original training window
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML alpha discovery.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get('ml', {})
        
        self.model_types = self.ml_config.get('model_types', ['lgbm', 'linear'])
        self.cv_folds = self.ml_config.get('cv_folds', 5)
        self.early_stopping = self.ml_config.get('early_stopping_rounds', 50)
        self.shap_samples = self.ml_config.get('shap_samples', 500)
        self.min_feature_importance = self.ml_config.get('min_feature_importance', 0.01)
        
        self.min_oos_sharpe = config.get('validation', {}).get('min_sharpe_ratio', 0.5)
    
    def discover(
        self, 
        data: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series
    ) -> List[MLEdgeResult]:
        """
        Discover edges using ML models.
        
        Args:
            data: Original OHLCV data
            features: Feature DataFrame
            target: Target variable (e.g., forward returns)
            
        Returns:
            List of ML edge results
        """
        results = []
        
        for model_type in self.model_types:
            try:
                result = self._train_and_validate(
                    data, features, target, model_type
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"ML discovery failed for {model_type}: {e}")
        
        return results
    
    def _train_and_validate(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str
    ) -> Optional[MLEdgeResult]:
        """Train and validate a single model type."""
        
        # Align data
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        if len(X) < 200:
            return None
        
        # Time-series cross-validation
        cv_results = self._time_series_cv(X, y, model_type)
        
        if cv_results is None:
            return None
        
        # Train final model
        model, feature_importance = self._train_model(X, y, model_type)
        
        if model is None:
            return None
        
        # Calculate feature attributions (SHAP-like)
        attributions = self._calculate_attributions(model, X, model_type)
        
        # Get top features
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top_features = [f for f, imp in sorted_features if imp >= self.min_feature_importance][:10]
        
        # Test noise sensitivity
        noise_sensitivity = self._test_noise_sensitivity(X, y, model, model_type)
        
        # Test regime performance
        regime_performance = self._test_regime_performance(data, X, y, model, model_type)
        
        # Validate
        is_valid = (
            cv_results['mean_oos_sharpe'] >= self.min_oos_sharpe and
            noise_sensitivity < 0.5 and  # Not too sensitive to noise
            min(regime_performance.values()) > 0  # Positive in all regimes
        )
        
        rejection_reason = None
        if not is_valid:
            reasons = []
            if cv_results['mean_oos_sharpe'] < self.min_oos_sharpe:
                reasons.append(f"OOS Sharpe {cv_results['mean_oos_sharpe']:.2f} < {self.min_oos_sharpe}")
            if noise_sensitivity >= 0.5:
                reasons.append(f"High noise sensitivity: {noise_sensitivity:.2f}")
            if min(regime_performance.values()) <= 0:
                reasons.append("Negative performance in some regimes")
            rejection_reason = "; ".join(reasons)
        
        return MLEdgeResult(
            model_type=model_type,
            feature_importance=feature_importance,
            in_sample_sharpe=cv_results['mean_is_sharpe'],
            out_of_sample_sharpe=cv_results['mean_oos_sharpe'],
            top_features=top_features,
            feature_attributions=attributions,
            cv_sharpes=cv_results['oos_sharpes'],
            regime_performance=regime_performance,
            noise_sensitivity=noise_sensitivity,
            is_valid=is_valid,
            rejection_reason=rejection_reason
        )
    
    def _time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str
    ) -> Optional[Dict[str, Any]]:
        """Perform time-series cross-validation."""
        
        n = len(X)
        fold_size = n // (self.cv_folds + 1)
        
        is_sharpes = []
        oos_sharpes = []
        
        for fold in range(self.cv_folds):
            # Expanding window: train on all data up to fold, test on fold
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n)
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_train) < 50 or len(X_test) < 20:
                continue
            
            # Train model
            model, _ = self._train_model(X_train, y_train, model_type)
            
            if model is None:
                continue
            
            # Predict
            train_pred = self._predict(model, X_train, model_type)
            test_pred = self._predict(model, X_test, model_type)
            
            # Calculate strategy returns
            train_returns = train_pred * y_train
            test_returns = test_pred * y_test
            
            is_sharpe = calculate_sharpe_ratio(
                train_returns.values, 
                periods_per_year=len(train_returns)
            )
            oos_sharpe = calculate_sharpe_ratio(
                test_returns.values,
                periods_per_year=len(test_returns)
            )
            
            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
        
        if len(oos_sharpes) < 2:
            return None
        
        return {
            'mean_is_sharpe': np.mean(is_sharpes),
            'mean_oos_sharpe': np.mean(oos_sharpes),
            'std_oos_sharpe': np.std(oos_sharpes),
            'oos_sharpes': oos_sharpes
        }
    
    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a model and return feature importance."""
        
        feature_importance = {}
        
        if model_type == 'lgbm':
            try:
                import lightgbm as lgb
                
                # Convert target to classification
                y_class = (y > 0).astype(int)
                
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=5,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbose=-1,
                    random_state=42
                )
                
                model.fit(X, y_class)
                
                # Get feature importance
                importance = model.feature_importances_
                for i, col in enumerate(X.columns):
                    feature_importance[col] = float(importance[i]) / sum(importance)
                
                return model, feature_importance
                
            except ImportError:
                pass
        
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                
                y_class = (y > 0).astype(int)
                
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbosity=0,
                    random_state=42
                )
                
                model.fit(X, y_class)
                
                importance = model.feature_importances_
                for i, col in enumerate(X.columns):
                    feature_importance[col] = float(importance[i]) / sum(importance)
                
                return model, feature_importance
                
            except ImportError:
                pass
        
        elif model_type == 'linear':
            from sklearn.linear_model import RidgeClassifier
            from sklearn.preprocessing import StandardScaler
            
            y_class = (y > 0).astype(int)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RidgeClassifier(alpha=1.0, random_state=42)
            model.fit(X_scaled, y_class)
            
            # Feature importance from coefficients
            coefs = np.abs(model.coef_[0])
            coefs = coefs / coefs.sum()
            for i, col in enumerate(X.columns):
                feature_importance[col] = float(coefs[i])
            
            # Wrap model with scaler
            class ScaledModel:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler
                
                def predict_proba(self, X):
                    X_scaled = self.scaler.transform(X)
                    decision = self.model.decision_function(X_scaled)
                    prob = 1 / (1 + np.exp(-decision))
                    return np.column_stack([1 - prob, prob])
            
            return ScaledModel(model, scaler), feature_importance
        
        elif model_type == 'neural':
            try:
                from sklearn.neural_network import MLPClassifier
                from sklearn.preprocessing import StandardScaler
                
                y_class = (y > 0).astype(int)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                hidden_layers = self.ml_config.get('nn_hidden_layers', [64, 32])
                
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation='relu',
                    alpha=0.01,
                    max_iter=200,
                    random_state=42
                )
                
                model.fit(X_scaled, y_class)
                
                # For neural networks, use permutation importance later
                for col in X.columns:
                    feature_importance[col] = 1.0 / len(X.columns)
                
                class ScaledModel:
                    def __init__(self, model, scaler):
                        self.model = model
                        self.scaler = scaler
                    
                    def predict_proba(self, X):
                        X_scaled = self.scaler.transform(X)
                        return self.model.predict_proba(X_scaled)
                
                return ScaledModel(model, scaler), feature_importance
                
            except Exception:
                pass
        
        return None, {}
    
    def _predict(self, model: Any, X: pd.DataFrame, model_type: str) -> pd.Series:
        """Generate predictions from model."""
        try:
            proba = model.predict_proba(X)
            # Convert probability to signal: 1 if prob > 0.5, -1 otherwise
            signals = np.where(proba[:, 1] > 0.5, 1, -1)
            # Scale by confidence
            confidence = np.abs(proba[:, 1] - 0.5) * 2
            return pd.Series(signals * confidence, index=X.index)
        except Exception:
            return pd.Series(0, index=X.index)
    
    def _calculate_attributions(
        self,
        model: Any,
        X: pd.DataFrame,
        model_type: str
    ) -> Dict[str, Any]:
        """Calculate feature attributions."""
        attributions = {}
        
        try:
            import shap
            
            # Sample for SHAP
            sample_size = min(self.shap_samples, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            
            if model_type in ['lgbm', 'xgboost']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                # Mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                for i, col in enumerate(X.columns):
                    attributions[col] = float(mean_shap[i])
                    
        except Exception:
            # SHAP not available or failed
            pass
        
        return attributions
    
    def _test_noise_sensitivity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        model_type: str,
        noise_level: float = 0.05,
        n_iterations: int = 20
    ) -> float:
        """Test sensitivity to feature noise."""
        
        original_signals = self._predict(model, X, model_type)
        original_sharpe = calculate_sharpe_ratio(
            (original_signals * y).values,
            periods_per_year=len(y)
        )
        
        noisy_sharpes = []
        
        for _ in range(n_iterations):
            # Add noise to features
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise * X.std().values
            
            noisy_signals = self._predict(model, X_noisy, model_type)
            noisy_sharpe = calculate_sharpe_ratio(
                (noisy_signals * y).values,
                periods_per_year=len(y)
            )
            noisy_sharpes.append(noisy_sharpe)
        
        # Sensitivity = std of performance under noise / original performance
        if original_sharpe != 0:
            sensitivity = np.std(noisy_sharpes) / abs(original_sharpe)
        else:
            sensitivity = 1.0
        
        return float(sensitivity)
    
    def _test_regime_performance(
        self,
        data: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        model_type: str
    ) -> Dict[str, float]:
        """Test performance across different regimes."""
        
        results = {}
        
        # Calculate volatility regime
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        
        vol = data['returns'].rolling(20).std()
        vol_median = vol.median()
        
        high_vol_mask = vol > vol_median
        low_vol_mask = vol <= vol_median
        
        # Align masks with X index
        high_vol_idx = X.index.intersection(high_vol_mask[high_vol_mask].index)
        low_vol_idx = X.index.intersection(low_vol_mask[low_vol_mask].index)
        
        for regime_name, idx in [('high_vol', high_vol_idx), ('low_vol', low_vol_idx)]:
            if len(idx) < 30:
                results[regime_name] = 0.0
                continue
            
            X_regime = X.loc[idx]
            y_regime = y.loc[idx]
            
            signals = self._predict(model, X_regime, model_type)
            returns = signals * y_regime
            
            sharpe = calculate_sharpe_ratio(returns.values, periods_per_year=len(returns))
            results[regime_name] = float(sharpe)
        
        return results

