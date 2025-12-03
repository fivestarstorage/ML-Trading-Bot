import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from .utils import get_logger

logger = get_logger()

try:
    from .deep_learning import GRUAttentionModel, EnsembleModel
    HAS_DL = True
except ImportError:
    HAS_DL = False
    logger.warning("Deep learning modules not available")

class MLModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config['ml']['lgbm']
        self.model = None
        self.feature_cols = None

        # Deep learning support
        self.use_deep_learning = config.get('ml', {}).get('use_deep_learning', False)
        self.use_ensemble = config.get('ml', {}).get('use_ensemble', False)
        self.gru_model = None
        self.ensemble_model = None
        
    def prepare_data(self, df):
        """Prepare data for training/inference."""
        # Drop non-feature columns
        exclude_cols = [
            'entry_time', 'entry_price', 'tp', 'sl', 'target', 'pnl_r',
            'bias', 'entry_type', 'session_label', 'daily_trend_label',
            'vol_regime', 'exit_time'
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        features = df[feature_cols].copy()
        for col in features.columns:
            if features[col].dtype == 'O':
                features[col] = pd.to_numeric(features[col], errors='coerce')
        features = features.fillna(0.0)
        return features, feature_cols

    def train(self, df_train):
        X, self.feature_cols = self.prepare_data(df_train)
        y = df_train['target']

        # Calculate scale_pos_weight for imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
             scale_pos_weight = n_neg / n_pos
        else:
             scale_pos_weight = 1.0

        params = self.model_params.copy()
        params.setdefault('verbose', -1)
        logger.info(f"Training LGBM with {len(X)} samples...")

        dtrain = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        # Train deep learning model if enabled
        if self.use_deep_learning and HAS_DL:
            logger.info("Training GRU model with attention...")

            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]

            gru_config = {
                'sequence_length': self.config.get('ml', {}).get('sequence_length', 20)
            }
            self.gru_model = GRUAttentionModel(gru_config)
            self.gru_model.train(X_train, y_train, X_val, y_val, verbose=0)

            # Create ensemble if enabled
            if self.use_ensemble:
                logger.info("Creating ensemble (LGBM + GRU)...")
                lgbm_weight = self.config.get('ml', {}).get('lgbm_weight', 0.6)
                self.ensemble_model = EnsembleModel(self.model, self.gru_model, lgbm_weight)

                # Optimize weights on validation set
                self.ensemble_model.optimize_weights(X_val, y_val, metric='roc_auc')

        return self.model

    def predict_proba(self, df):
        if self.model is None:
            raise ValueError("Model not trained")

        X, _ = self.prepare_data(df)
        # Ensure columns match
        # LGBM is sensitive to column order if not using pandas dataframes strictly
        # We restrict to saved feature_cols
        X = X[self.feature_cols]

        # Use ensemble if available
        if self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)

        # Otherwise use LGBM only
        return self.model.predict(X)

    def save(self, path):
        if self.model is None:
            return
        joblib.dump({'model': self.model, 'features': self.feature_cols}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['features']
        logger.info(f"Model loaded from {path}")


