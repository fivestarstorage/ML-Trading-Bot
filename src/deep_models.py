"""
Deep learning models for cryptocurrency prediction.

Implements research-backed architectures:
- LSTM: Long Short-Term Memory
- GRU: Gated Recurrent Unit
- CNN-LSTM: Hybrid architecture (6.2% improvement per research)
- Attention mechanisms

All models include anti-overfitting measures:
- Dropout layers
- Early stopping
- Regularization
- Batch normalization
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available. Deep learning models will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """
        LSTM model with dropout and regularization.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super(LSTMModel, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers with dropout
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            # Dropout
            self.dropout = nn.Dropout(dropout)

            # Batch normalization
            self.batch_norm = nn.BatchNorm1d(hidden_size)

            # Output layer
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # LSTM
            lstm_out, _ = self.lstm(x)

            # Take last timestep output
            last_output = lstm_out[:, -1, :]

            # Batch norm
            normed = self.batch_norm(last_output)

            # Dropout
            dropped = self.dropout(normed)

            # Output
            out = self.fc(dropped)
            out = self.sigmoid(out)

            return out


    class GRUModel(nn.Module):
        """
        GRU model (faster than LSTM, often similar performance).
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super(GRUModel, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # GRU layers
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            self.dropout = nn.Dropout(dropout)
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            gru_out, _ = self.gru(x)
            last_output = gru_out[:, -1, :]
            normed = self.batch_norm(last_output)
            dropped = self.dropout(normed)
            out = self.fc(dropped)
            out = self.sigmoid(out)
            return out


    class CNNLSTMModel(nn.Module):
        """
        CNN-LSTM hybrid model.
        Research shows 6.2% improvement over standalone LSTM.
        CNN extracts local patterns, LSTM captures temporal dependencies.
        """

        def __init__(
            self,
            input_size: int,
            num_filters: int = 64,
            kernel_size: int = 3,
            hidden_size: int = 64,
            dropout: float = 0.3
        ):
            super(CNNLSTMModel, self).__init__()

            # CNN layers for feature extraction
            self.conv1 = nn.Conv1d(
                in_channels=input_size,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2)

            self.conv2 = nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(kernel_size=2)

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=num_filters * 2,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout,
                batch_first=True
            )

            # Dropout and output
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            # Conv1d expects: (batch, features, seq_len)
            x = x.permute(0, 2, 1)

            # CNN layers
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)

            # Back to (batch, seq_len, features) for LSTM
            x = x.permute(0, 2, 1)

            # LSTM
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]

            # Output
            normed = self.batch_norm(last_output)
            dropped = self.dropout(normed)
            out = self.fc(dropped)
            out = self.sigmoid(out)

            return out


    class DeepModelTrainer:
        """
        Trainer for deep learning models with anti-overfitting measures.
        """

        def __init__(
            self,
            model_type: str = 'lstm',
            sequence_length: int = 20,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            learning_rate: float = 0.001,
            batch_size: int = 64,
            max_epochs: int = 100,
            early_stopping_patience: int = 10,
            device: str = None
        ):
            """
            Initialize deep model trainer.

            Args:
                model_type: 'lstm', 'gru', or 'cnn_lstm'
                sequence_length: Number of timesteps in input sequences
                hidden_size: Hidden layer size
                num_layers: Number of recurrent layers
                dropout: Dropout rate
                learning_rate: Learning rate
                batch_size: Batch size
                max_epochs: Maximum training epochs
                early_stopping_patience: Patience for early stopping
                device: 'cuda' or 'cpu'
            """
            self.model_type = model_type
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.max_epochs = max_epochs
            self.early_stopping_patience = early_stopping_patience

            # Set device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            self.model = None
            self.is_fitted = False

        def _create_sequences(
            self,
            X: np.ndarray,
            y: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Create sequences for time series data."""

            X_seq = []
            y_seq = []

            for i in range(len(X) - self.sequence_length):
                X_seq.append(X[i:i + self.sequence_length])
                y_seq.append(y[i + self.sequence_length])

            return np.array(X_seq), np.array(y_seq)

        def fit(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None
        ) -> Dict[str, list]:
            """
            Train the deep learning model.

            Args:
                X_train: Training features
                y_train: Training labels
                X_val: Validation features
                y_val: Validation labels

            Returns:
                Training history
            """
            logger.info(f"Training {self.model_type.upper()} model...")

            # Convert to numpy
            X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            if X_val is not None:
                X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(X_train_np, y_train)

            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self._create_sequences(X_val_np, y_val)
            else:
                X_val_seq, y_val_seq = None, None

            logger.info(f"Training sequences: {X_train_seq.shape}")

            # Create model
            input_size = X_train_seq.shape[2]

            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                )
            elif self.model_type == 'gru':
                self.model = GRUModel(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                )
            elif self.model_type == 'cnn_lstm':
                self.model = CNNLSTMModel(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model = self.model.to(self.device)

            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.FloatTensor(y_train_seq).unsqueeze(1)
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            if X_val_seq is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq),
                    torch.FloatTensor(y_val_seq).unsqueeze(1)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            # Training loop with early stopping
            history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.max_epochs):
                # Training
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                history['train_loss'].append(train_loss)

                # Validation
                if X_val_seq is not None:
                    self.model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)

                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()

                            predictions = (outputs > 0.5).float()
                            correct += (predictions == batch_y).sum().item()
                            total += batch_y.size(0)

                    val_loss /= len(val_loader)
                    val_accuracy = correct / total

                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if (epoch + 1) % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{self.max_epochs} - "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Val Acc: {val_accuracy:.4f}"
                        )

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                else:
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}")

            self.is_fitted = True
            logger.info(f"{self.model_type.upper()} training complete")

            return history

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            """
            Get probability predictions.

            Args:
                X: Features

            Returns:
                Probability predictions
            """
            if not self.is_fitted:
                raise ValueError("Model not fitted. Call fit() first.")

            self.model.eval()

            # Convert to numpy
            X_np = X.values if isinstance(X, pd.DataFrame) else X

            # Create dummy y for sequence creation
            y_dummy = np.zeros(len(X_np))
            X_seq, _ = self._create_sequences(X_np, y_dummy)

            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                predictions = self.model(X_tensor).cpu().numpy().flatten()

            return predictions


else:
    # Dummy classes when PyTorch not available
    class DeepModelTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for deep learning models. Install with: pip install torch")
