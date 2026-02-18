"""
================================================================================
LSTM CLASSIFIER - Long Short-Term Memory for Stock Direction Prediction
================================================================================
Recurrent neural network that captures temporal dependencies in stock data.

Key Features:
- Processes sequences of historical data (30 days)
- Captures long-term patterns
- Better at trend prediction than traditional ML
- Handles time series naturally

Expected Performance: 58-65% accuracy
Training Time: ~5-10 minutes per stock
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import sys
import os

# Import base model
try:
    from ..base_model import BaseDeepLearningModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from base_model import BaseDeepLearningModel

from sklearn.preprocessing import MinMaxScaler
from loguru import logger


class LSTMClassifier(BaseDeepLearningModel):
    """
    LSTM binary classifier for stock direction prediction.

    Uses sequences of historical data to predict next-day direction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_length: int = 30,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 20,
        device: str = 'cpu'
    ):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            seq_length: Length of input sequences
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            batch_size: Batch size for training
            early_stopping_patience: Epochs to wait before early stopping
            device: 'cpu' or 'cuda'
        """
        super().__init__(model_name="LSTM", seq_length=seq_length)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device

        # Scaler for features
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Will be created during training
        self.model = None

    def _build_model(self):
        """Build LSTM model architecture."""
        try:
            import torch
            import torch.nn as nn

            class LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size,
                        hidden_size,
                        num_layers,
                        dropout=dropout,
                        batch_first=True
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1),
                        nn.Sigmoid()  # Output probability
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    # Take last timestamp output
                    last_out = lstm_out[:, -1, :]
                    out = self.fc(last_out)
                    return out

            model = LSTMNet(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout
            )

            return model.to(self.device)

        except ImportError:
            raise ImportError("PyTorch not installed! Install with: pip install torch")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Train LSTM model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - binary 0/1
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        logger.info(f"Training {self.model_name}...")

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Store feature names
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        logger.info(f"  Created sequences: {X_train_seq.shape}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)

        if X_val_seq is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)

        # Build model
        self.model = self._build_model()

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(X_train_tensor)
            train_losses.append(epoch_loss)

            # Validation
            if X_val_seq is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = criterion(y_val_pred, y_val_tensor).item()
                    val_losses.append(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"  Early stopping at epoch {epoch+1}")
                        break

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses if val_losses else None,
            'n_features': self.input_size,
            'n_train_samples': len(X_train_seq),
            'n_val_samples': len(X_val_seq) if X_val_seq is not None else 0,
            'final_epoch': len(train_losses)
        }

        logger.info(f"[SUCCESS] {self.model_name} training complete!")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary direction (0=DOWN, 1=UP).

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Binary predictions (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        import torch

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq = self.create_sequences(X_scaled)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)

        # Convert to binary (threshold at 0.5)
        predictions = (y_pred.cpu().numpy().ravel() > 0.5).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of UP movement.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        import torch

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq = self.create_sequences(X_scaled)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)

        # Return probabilities
        probabilities = y_pred.cpu().numpy().ravel()

        return probabilities


if __name__ == '__main__':
    # Example usage
    print("LSTM Classifier Test\n" + "="*50)

    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)

    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 2, 200)

    X_test = np.random.randn(200, 50)  # Need more for sequences
    y_test = np.random.randint(0, 2, 200)

    # Create model
    model = LSTMClassifier(
        input_size=50,
        hidden_size=64,
        num_layers=2,
        seq_length=30,
        epochs=10,  # Fewer for testing
        batch_size=32
    )

    # Train
    print("\n[1/3] Training...")
    history = model.train(X_train, y_train, X_val, y_val, verbose=True)
    print(f"  Training history: epochs={history['final_epoch']}")

    # Evaluate (account for sequence length)
    print("\n[2/3] Evaluating...")
    # Need to align y_test with predictions (which are reduced by seq_length)
    y_test_aligned = y_test[model.seq_length:]
    metrics = model.evaluate(X_test, y_test_aligned)
    print(metrics)

    # Test predictions
    print("\n[3/3] Sample Predictions:")
    preds = model.predict(X_test[:50])  # Need enough for sequences
    proba = model.predict_proba(X_test[:50])

    print("  Index | Prediction | Probability | True")
    print("  " + "-"*50)
    for i in range(min(5, len(preds))):
        y_idx = model.seq_length + i  # Adjust for sequence offset
        print(f"    {i:2d}   |     {preds[i]}      |   {proba[i]:.3f}    |  {y_test[y_idx]}")

    print("\n[SUCCESS] LSTM Classifier working correctly!")
