"""
================================================================================
V3 NEURAL MODELS - Advanced deep learning models for stock price prediction
================================================================================
Includes: LSTM, BiLSTM, Transformer, CNN-LSTM Hybrid, Temporal Convolutional Networks (TCN)

Research shows these models often outperform traditional ML for time series:
- BiLSTM: Better feature extraction due to bidirectional processing
- Transformer: Parallel processing, excellent for long-term dependencies
- CNN-LSTM: CNN extracts local patterns, LSTM captures temporal dependencies
- TCN: Dilated causal convolutions for efficient long-range dependency capture

Target: Increase directional accuracy from 50-57% → 60-70% and win ratio to 70%+
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# ============================================================================
# BASE NEURAL NETWORK CLASS
# ============================================================================

class BaseNeuralModel:
    """Base class for all neural network models."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu'):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: 'cpu' or 'cuda'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_losses = []
        self.val_losses = []
        
    def create_sequences(self, X, y, seq_length=30):
        """Create sequences for time series modeling."""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def scale_data(self, X_train, X_test=None, X_val=None):
        """Scale features using training data."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
            
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
            
        return X_train_scaled, X_test_scaled, X_val_scaled
    
    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, seq_length=30):
        """Prepare data for training."""
        # Scale data
        X_train_scaled, X_val_scaled, _ = self.scale_data(X_train, X_val)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, seq_length)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val, seq_length)
        else:
            X_val_seq, y_val_seq = None, None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        
        if X_val_seq is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        else:
            X_val_tensor, y_val_tensor = None, None
        
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    
    def train_model(self, X_train_tensor, y_train_tensor, X_val_tensor=None, y_val_tensor=None):
        """Train the model."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader:
                # Forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
            
            epoch_loss /= len(X_train_tensor)
            self.train_losses.append(epoch_loss)
            
            # Validation loss
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = criterion(y_val_pred, y_val_tensor).item()
                    self.val_losses.append(val_loss)
                self.model.train()
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.epochs}, "
                               f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 20 == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.6f}")
    
    def predict(self, X_test_tensor):
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
        return predictions.cpu().numpy()


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(BaseNeuralModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu'):
        super().__init__(input_size, hidden_size, num_layers, dropout, 
                        learning_rate, epochs, batch_size, device)
        
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   dropout=dropout, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take last timestamp output
                last_out = lstm_out[:, -1, :]
                out = self.fc(last_out)
                return out
        
        self.model = LSTM(input_size, hidden_size, num_layers, dropout).to(device)


# ============================================================================
# BIDIRECTIONAL LSTM MODEL
# ============================================================================

class BiLSTMModel(BaseNeuralModel):
    """Bidirectional LSTM model - processes sequence forwards and backwards."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu'):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        learning_rate, epochs, batch_size, device)
        
        class BiLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                                     bidirectional=True, dropout=dropout, batch_first=True)
                # Bidirectional output is 2*hidden_size
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                lstm_out, (h_n, c_n) = self.bilstm(x)
                # Concatenate last outputs from both directions
                last_out = lstm_out[:, -1, :]
                out = self.fc(last_out)
                return out
        
        self.model = BiLSTM(input_size, hidden_size, num_layers, dropout).to(device)


# ============================================================================
# GRU MODEL (Gated Recurrent Unit - simpler than LSTM)
# ============================================================================

class GRUModel(BaseNeuralModel):
    """GRU model - simplified version of LSTM with fewer parameters."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu'):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        learning_rate, epochs, batch_size, device)
        
        class GRU(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_out = gru_out[:, -1, :]
                out = self.fc(last_out)
                return out
        
        self.model = GRU(input_size, hidden_size, num_layers, dropout).to(device)


# ============================================================================
# CNN-LSTM HYBRID MODEL
# ============================================================================

class CNNLSTMModel(BaseNeuralModel):
    """
    Hybrid CNN-LSTM Model.
    
    CNN: Extracts local patterns and features from price movements
    LSTM: Captures temporal dependencies and long-term patterns
    
    This combination often outperforms vanilla LSTM on stock data.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu',
                 num_filters=32, kernel_size=3):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        learning_rate, epochs, batch_size, device)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        class CNNLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, num_filters, kernel_size):
                super().__init__()
                
                # CNN part: Extract spatial patterns
                self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=1)
                self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool1d(2)
                
                # LSTM part: Capture temporal dependencies on CNN output
                self.lstm = nn.LSTM(num_filters, hidden_size, num_layers,
                                   dropout=dropout, batch_first=True)
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                # CNN part: (batch, seq_len, features) -> (batch, features, seq_len)
                x = x.transpose(1, 2)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                
                # Back to (batch, seq_len, features)
                x = x.transpose(1, 2)
                
                # LSTM part
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                
                # FC part
                out = self.fc(last_out)
                return out
        
        self.model = CNNLSTM(input_size, hidden_size, num_layers, dropout,
                            num_filters, kernel_size).to(device)


# ============================================================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================

class TCNModel(BaseNeuralModel):
    """
    Temporal Convolutional Network (TCN).
    
    Uses dilated causal convolutions to capture long-range dependencies efficiently.
    
    Advantages:
    - Parallel training (unlike RNNs which are sequential)
    - Can capture very long-term dependencies with dilated convolutions
    - Efficient memory usage
    - Often outperforms LSTM on stock prediction tasks
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu',
                 num_filters=32, kernel_size=3):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        learning_rate, epochs, batch_size, device)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        class ResidualBlock(nn.Module):
            def __init__(self, num_filters, kernel_size, dilation, dropout):
                super().__init__()
                padding = (kernel_size - 1) * dilation
                self.conv1 = nn.utils.weight_norm(
                    nn.Conv1d(num_filters, num_filters, kernel_size,
                             dilation=dilation, padding=padding)
                )
                self.dropout1 = nn.Dropout(dropout)
                self.relu = nn.ReLU()
                self.conv2 = nn.utils.weight_norm(
                    nn.Conv1d(num_filters, num_filters, kernel_size,
                             dilation=dilation, padding=padding)
                )
                self.dropout2 = nn.Dropout(dropout)
                
                # Causal padding: remove right side
                self.padding = padding
            
            def forward(self, x):
                # Apply causal padding (remove future information)
                out = self.conv1(x)
                out = out[:, :, :-self.padding] if self.padding > 0 else out
                out = self.dropout1(self.relu(out))
                
                out = self.conv2(out)
                out = out[:, :, :-self.padding] if self.padding > 0 else out
                out = self.dropout2(out)
                
                return out + x
        
        class TCN(nn.Module):
            def __init__(self, input_size, num_filters, num_levels, kernel_size, dropout):
                super().__init__()
                layers = []
                
                # First layer: input_size -> num_filters
                layers.append(nn.Conv1d(input_size, num_filters, 1))
                
                # Residual blocks with increasing dilation
                for level in range(num_levels):
                    dilation = 2 ** level
                    layers.append(ResidualBlock(num_filters, kernel_size, dilation, dropout))
                
                self.network = nn.Sequential(*layers)
                self.fc = nn.Sequential(
                    nn.Linear(num_filters, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                # (batch, seq_len, features) -> (batch, features, seq_len)
                x = x.transpose(1, 2)
                y = self.network(x)
                # Take last timestamp
                y = y[:, :, -1]
                y = self.fc(y)
                return y
        
        self.model = TCN(input_size, num_filters, num_layers, kernel_size, dropout).to(device)


# ============================================================================
# TRANSFORMER-BASED MODEL
# ============================================================================

class TransformerModel(BaseNeuralModel):
    """
    Transformer-based model for time series prediction.
    
    Uses multi-head self-attention to capture dependencies across entire sequence.
    Advantages:
    - Parallel processing (faster than RNNs)
    - Can attend to any position in sequence
    - Excellent for capturing long-term dependencies
    - State-of-the-art for many sequence tasks
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu',
                 n_heads=8):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        learning_rate, epochs, batch_size, device)
        self.n_heads = n_heads
        
        class TransformerBlock(nn.Module):
            def __init__(self, hidden_size, n_heads, dropout):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, n_heads,
                                                      dropout=dropout, batch_first=True)
                self.feed_forward = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # Self-attention with residual connection
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                
                # Feed-forward with residual connection
                ff_out = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_out))
                
                return x
        
        class Transformer(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, n_heads):
                super().__init__()
                # Project input to hidden dimension
                self.embedding = nn.Linear(input_size, hidden_size)
                self.transformer_blocks = nn.ModuleList([
                    TransformerBlock(hidden_size, n_heads, dropout)
                    for _ in range(num_layers)
                ])
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                x = self.embedding(x)
                
                for transformer_block in self.transformer_blocks:
                    x = transformer_block(x)
                
                # Take last timestamp output
                x = x[:, -1, :]
                x = self.fc(x)
                return x
        
        self.model = Transformer(input_size, hidden_size, num_layers, dropout, n_heads).to(device)


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'bilstm': BiLSTMModel,
    'gru': GRUModel,
    'cnn_lstm': CNNLSTMModel,
    'tcn': TCNModel,
    'transformer': TransformerModel,
}

def create_model(model_name, input_size, device='cpu', **kwargs):
    """Factory function to create model instances."""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return MODEL_REGISTRY[model_name](input_size, device=device, **kwargs)


if __name__ == '__main__':
    # Example usage
    logger.info("Available neural models:")
    for name in MODEL_REGISTRY.keys():
        logger.info(f"  - {name}")
