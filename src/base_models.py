"""
Base model implementations with anti-overfitting measures.

Based on research showing ensemble diversity is critical:
- Bagging (Random Forest, Extra Trees) reduces variance
- Boosting (XGBoost, LightGBM) reduces bias
- Different model types capture different patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import logging

try:
    import catboost as cb
except ImportError:
    cb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModelEnsemble:
    """
    Collection of base models with anti-overfitting constraints.

    Key anti-overfitting measures:
    - Conservative hyperparameters (max_depth, min_samples)
    - Regularization (L1, L2, dropout)
    - Early stopping on validation set
    - Class balancing
    - Cross-validation for hyperparameter tuning
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base models.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_models(self) -> Dict[str, Any]:
        """
        Create base models with anti-overfitting configurations.

        Returns:
            Dictionary of model instances
        """
        models = {}

        # Random Forest - Bagging (variance reduction)
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,  # More trees = more stable
            max_depth=8,  # Limited depth prevents overfitting
            min_samples_split=50,  # Require more samples to split
            min_samples_leaf=20,  # Require more samples in leaves
            max_features='sqrt',  # Limit features per tree
            max_samples=0.8,  # Bootstrap sampling
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Handle class imbalance
            bootstrap=True
        )

        # Extra Trees - More randomization than RF
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            max_samples=0.8,
            random_state=43,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True
        )

        # XGBoost - Gradient boosting with regularization
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,  # Will use early stopping
            max_depth=4,  # Shallow trees
            learning_rate=0.01,  # Slow learning
            subsample=0.8,  # Row sampling
            colsample_bytree=0.8,  # Feature sampling
            colsample_bylevel=0.8,  # Feature sampling per level
            min_child_weight=10,  # Minimum samples in leaf
            gamma=0.1,  # Regularization parameter
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            early_stopping_rounds=50,
            scale_pos_weight=1.0  # Class balancing
        )

        # LightGBM - Fast gradient boosting
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=50,
            min_split_gain=0.01,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            is_unbalance=True
        )

        # CatBoost - If available
        if cb is not None:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=500,
                depth=4,
                learning_rate=0.01,
                l2_leaf_reg=3.0,
                random_strength=0.5,
                bagging_temperature=0.2,
                subsample=0.8,
                random_state=42,
                verbose=False,
                early_stopping_rounds=50,
                auto_class_weights='Balanced'
            )

        # Logistic Regression - Linear baseline
        models['logistic'] = LogisticRegression(
            C=0.1,  # Strong regularization
            penalty='l2',
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        logger.info(f"Created {len(models)} base models")
        return models

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit all base models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels

        Returns:
            Training metrics for each model
        """
        logger.info("Training base models...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Create models
        self.models = self.create_models()

        metrics = {}

        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")

            try:
                # Handle models with early stopping
                if name in ['xgboost', 'lightgbm', 'catboost'] and X_val is not None:
                    if name == 'xgboost':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_val_scaled, y_val)],
                            verbose=False
                        )
                    elif name == 'lightgbm':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_val_scaled, y_val)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                        )
                    elif name == 'catboost':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=(X_val_scaled, y_val),
                            verbose=False
                        )
                else:
                    # Standard fit
                    model.fit(X_train_scaled, y_train)

                # Evaluate on training set
                y_train_pred = model.predict(X_train_scaled)
                train_metrics = self._calculate_metrics(y_train, y_train_pred, model, X_train_scaled)

                # Evaluate on validation set if available
                if X_val is not None:
                    y_val_pred = model.predict(X_val_scaled)
                    val_metrics = self._calculate_metrics(y_val, y_val_pred, model, X_val_scaled)

                    metrics[name] = {
                        'train_accuracy': train_metrics['accuracy'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1'],
                        'val_auc': val_metrics['auc'],
                        'overfit_gap': train_metrics['accuracy'] - val_metrics['accuracy']
                    }

                    logger.info(f"  Train Acc: {train_metrics['accuracy']:.4f}")
                    logger.info(f"  Val Acc: {val_metrics['accuracy']:.4f}")
                    logger.info(f"  Overfit Gap: {metrics[name]['overfit_gap']:.4f}")
                else:
                    metrics[name] = train_metrics

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue

        self.is_fitted = True
        logger.info("\nBase models training complete")

        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models.

        Args:
            X: Features

        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get probability predictions from all models.

        Args:
            X: Features

        Returns:
            Dictionary mapping model names to probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        probas = {}

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probas[name] = model.predict_proba(X_scaled)[:, 1]
            else:
                probas[name] = model.predict(X_scaled).astype(float)

        return probas

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        # Add AUC if model supports probability predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['auc'] = 0.0

        return metrics

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models.

        Returns:
            Dictionary mapping model names to feature importances
        """
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_

        return importances
