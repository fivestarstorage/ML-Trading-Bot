import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from .utils import get_logger

logger = get_logger()

class MLModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config['ml']['lgbm']
        self.model = None
        self.feature_cols = None
        
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
        
        return self.model

    def predict_proba(self, df):
        if self.model is None:
            raise ValueError("Model not trained")
            
        X, _ = self.prepare_data(df)
        # Ensure columns match
        # LGBM is sensitive to column order if not using pandas dataframes strictly
        # We restrict to saved feature_cols
        X = X[self.feature_cols]
        
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


