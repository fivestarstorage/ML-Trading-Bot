"""
Deep Learning Models for Trading Strategy
Implements GRU with Attention mechanism for temporal pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available. Deep learning features disabled.")

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for sequence modeling
    Helps the model focus on important time steps
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        # Apply attention weights
        output = x * a
        return tf.reduce_sum(output, axis=1)


class GRUAttentionModel:
    """
    GRU with Attention for temporal pattern recognition
    Better than LSTM for trading due to simpler architecture and faster training
    """

    def __init__(self, config: Dict):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for deep learning models")

        self.config = config
        self.model = None
        self.sequence_length = config.get('sequence_length', 20)  # 20 bars lookback
        self.feature_scaler = None

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build GRU with Attention architecture

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)

        # First GRU layer with return sequences for attention
        x = layers.GRU(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='gru_1'
        )(inputs)

        # Batch normalization
        x = layers.BatchNormalization(name='bn_1')(x)

        # Second GRU layer
        x = layers.GRU(
            units=64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='gru_2'
        )(x)

        # Batch normalization
        x = layers.BatchNormalization(name='bn_2')(x)

        # Attention mechanism
        x = AttentionLayer(name='attention')(x)

        # Dense layers
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(16, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='GRU_Attention')

        # Compile with adaptive learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        return model

    def prepare_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert tabular data to sequences for RNN input

        Args:
            X: Features dataframe
            y: Labels (optional for prediction)

        Returns:
            X_seq: (n_samples, sequence_length, n_features)
            y_seq: (n_samples,) if y provided, else None
        """
        from sklearn.preprocessing import RobustScaler

        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = RobustScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = self.feature_scaler.transform(X)

        # Create sequences
        n_samples = len(X_scaled) - self.sequence_length + 1
        n_features = X_scaled.shape[1]

        X_seq = np.zeros((n_samples, self.sequence_length, n_features))

        for i in range(n_samples):
            X_seq[i] = X_scaled[i:i + self.sequence_length]

        if y is not None:
            # Align labels with sequences (use last label in sequence)
            y_seq = y.iloc[self.sequence_length - 1:].values
            return X_seq, y_seq

        return X_seq, None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 0
    ):
        """
        Train the GRU model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
        """
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)

        # Build model if not exists
        if self.model is None:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.model = self.build_model(input_shape)

        # Calculate class weights for imbalanced data
        pos_weight = (y_train_seq == 0).sum() / (y_train_seq == 1).sum()
        class_weight = {0: 1.0, 1: pos_weight}

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=verbose
            )
        ]

        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)

        # Train
        logger.info(f"Training GRU model on {len(X_train_seq)} sequences...")
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Features dataframe

        Returns:
            Probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_seq, _ = self.prepare_sequences(X)
        probas = self.model.predict(X_seq, verbose=0).flatten()

        # Pad beginning with 0.5 (neutral) for sequences we can't predict
        padding = np.full(self.sequence_length - 1, 0.5)
        return np.concatenate([padding, probas])

    def save(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"GRU model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        self.model = keras.models.load_model(
            path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"GRU model loaded from {path}")


class EnsembleModel:
    """
    Ensemble combining LightGBM and GRU predictions
    Uses weighted averaging based on validation performance
    """

    def __init__(self, lgbm_model, gru_model, lgbm_weight: float = 0.6):
        """
        Args:
            lgbm_model: Trained LightGBM model
            gru_model: Trained GRU model
            lgbm_weight: Weight for LGBM predictions (0-1), GRU gets (1 - lgbm_weight)
        """
        self.lgbm_model = lgbm_model
        self.gru_model = gru_model
        self.lgbm_weight = lgbm_weight
        self.gru_weight = 1.0 - lgbm_weight

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble

        Args:
            X: Features dataframe

        Returns:
            Ensemble probabilities
        """
        # Get predictions from both models
        lgbm_proba = self.lgbm_model.predict_proba(X)[:, 1]
        gru_proba = self.gru_model.predict_proba(X)

        # Weighted average
        ensemble_proba = (
            self.lgbm_weight * lgbm_proba +
            self.gru_weight * gru_proba
        )

        return ensemble_proba

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'roc_auc'
    ):
        """
        Optimize ensemble weights on validation data

        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('roc_auc', 'accuracy', 'f1')
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        best_score = 0
        best_weight = 0.5

        # Try different weights
        for lgbm_w in np.arange(0.3, 0.8, 0.05):
            self.lgbm_weight = lgbm_w
            self.gru_weight = 1.0 - lgbm_w

            ensemble_proba = self.predict_proba(X_val)

            if metric == 'roc_auc':
                score = roc_auc_score(y_val, ensemble_proba)
            elif metric == 'accuracy':
                score = accuracy_score(y_val, (ensemble_proba > 0.5).astype(int))
            elif metric == 'f1':
                score = f1_score(y_val, (ensemble_proba > 0.5).astype(int))

            if score > best_score:
                best_score = score
                best_weight = lgbm_w

        self.lgbm_weight = best_weight
        self.gru_weight = 1.0 - best_weight

        logger.info(f"Optimized ensemble weights: LGBM={best_weight:.2f}, GRU={1-best_weight:.2f}")
        logger.info(f"Validation {metric}: {best_score:.4f}")
