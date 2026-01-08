"""
================================================================================
MULTI-TARGET PREDICTION MODELS
================================================================================
Implements 4 models:
1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit)
3. XGBoost (Gradient Boosting)
4. Ensemble (Stacking)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.models = {}  # One model per target
        self.scalers = {}  # One scaler per target
        self.feature_scaler = None
        
    def save(self, symbol: str, save_dir: str):
        """Save model and scalers."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save feature scaler
        scaler_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save target scalers
        for target, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save models (implementation-specific)
        self._save_models(symbol, save_dir)
        
        logger.info(f"Saved {self.model_name} for {symbol}")
    
    def load(self, symbol: str, save_dir: str):
        """Load model and scalers."""
        # Load feature scaler
        scaler_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_feature_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        # Load target scalers
        for target in ['close', 'high', 'low', 'direction']:
            scaler_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[target] = pickle.load(f)
        
        # Load models (implementation-specific)
        self._load_models(symbol, save_dir)
        
        logger.info(f"Loaded {self.model_name} for {symbol}")


class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory) model.
    Best for: Capturing long-term dependencies in time series.
    """
    
    def __init__(self, sequence_length: int = 10):
        super().__init__('lstm')
        self.sequence_length = sequence_length
    
    def build_model(self, input_shape: Tuple, target_name: str) -> Model:
        """Build LSTM architecture with improved design."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear' if target_name != 'direction' else 'sigmoid')
        ])
        
        loss = 'mse' if target_name != 'direction' else 'binary_crossentropy'
        metrics = ['mae'] if target_name != 'direction' else ['accuracy']
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Create sequences for LSTM."""
        sequences = []
        targets = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences
    
    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        """Train LSTM for all targets."""
        # Scale features
        self.feature_scaler = RobustScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train model for each target
        for target in ['close', 'high', 'low', 'direction']:
            logger.info(f"  Training LSTM for {target}...")
            
            # Scale target (except direction)
            if target != 'direction':
                self.scalers[target] = RobustScaler()
                y_train_scaled = self.scalers[target].fit_transform(
                    y_train_dict[target].reshape(-1, 1)
                ).ravel()
                y_val_scaled = self.scalers[target].transform(
                    y_val_dict[target].reshape(-1, 1)
                ).ravel()
            else:
                y_train_scaled = y_train_dict[target]
                y_val_scaled = y_val_dict[target]
            
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
            
            # Build model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            model = self.build_model(input_shape, target)
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            
            # Train
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            self.models[target] = model
    
    def predict(self, X):
        """Predict all targets."""
        X_scaled = self.feature_scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)
        
        predictions = {}
        for target in ['close', 'high', 'low', 'direction']:
            pred_scaled = self.models[target].predict(X_seq, verbose=0).ravel()
            
            if target != 'direction':
                pred = self.scalers[target].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                pred = (pred_scaled > 0.5).astype(int)
            
            # Pad to match input length
            pred_padded = np.full(len(X), np.nan)
            pred_padded[self.sequence_length:] = pred
            predictions[target] = pred_padded
        
        return predictions
    
    def _save_models(self, symbol, save_dir):
        """Save Keras models."""
        for target, model in self.models.items():
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.h5')
            model.save(model_path)
    
    def _load_models(self, symbol, save_dir):
        """Load Keras models."""
        for target in ['close', 'high', 'low', 'direction']:
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.h5')
            self.models[target] = load_model(model_path)


class GRUModel(BaseModel):
    """
    GRU (Gated Recurrent Unit) model.
    Best for: Faster training than LSTM, similar performance.
    """
    
    def __init__(self, sequence_length: int = 10):
        super().__init__('gru')
        self.sequence_length = sequence_length
    
    def build_model(self, input_shape: Tuple, target_name: str) -> Model:
        """Build GRU architecture with improved design."""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear' if target_name != 'direction' else 'sigmoid')
        ])
        
        loss = 'mse' if target_name != 'direction' else 'binary_crossentropy'
        metrics = ['mae'] if target_name != 'direction' else ['accuracy']
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Create sequences for GRU."""
        sequences = []
        targets = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences
    
    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        """Train GRU for all targets."""
        # Scale features
        self.feature_scaler = RobustScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train model for each target
        for target in ['close', 'high', 'low', 'direction']:
            logger.info(f"  Training GRU for {target}...")
            
            # Scale target (except direction)
            if target != 'direction':
                self.scalers[target] = RobustScaler()
                y_train_scaled = self.scalers[target].fit_transform(
                    y_train_dict[target].reshape(-1, 1)
                ).ravel()
                y_val_scaled = self.scalers[target].transform(
                    y_val_dict[target].reshape(-1, 1)
                ).ravel()
            else:
                y_train_scaled = y_train_dict[target]
                y_val_scaled = y_val_dict[target]
            
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
            
            # Build model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            model = self.build_model(input_shape, target)
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            
            # Train
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            self.models[target] = model
    
    def predict(self, X):
        """Predict all targets."""
        X_scaled = self.feature_scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)
        
        predictions = {}
        for target in ['close', 'high', 'low', 'direction']:
            pred_scaled = self.models[target].predict(X_seq, verbose=0).ravel()
            
            if target != 'direction':
                pred = self.scalers[target].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                pred = (pred_scaled > 0.5).astype(int)
            
            # Pad to match input length
            pred_padded = np.full(len(X), np.nan)
            pred_padded[self.sequence_length:] = pred
            predictions[target] = pred_padded
        
        return predictions
    
    def _save_models(self, symbol, save_dir):
        """Save Keras models."""
        for target, model in self.models.items():
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.h5')
            model.save(model_path)
    
    def _load_models(self, symbol, save_dir):
        """Load Keras models."""
        for target in ['close', 'high', 'low', 'direction']:
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.h5')
            self.models[target] = load_model(model_path)


class XGBoostModel(BaseModel):
    """
    XGBoost model.
    Best for: Fast training, handles non-linear relationships, feature importance.
    """
    
    def __init__(self):
        super().__init__('xgboost')
    
    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        """Train XGBoost for all targets."""
        # Scale features
        self.feature_scaler = RobustScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train model for each target
        for target in ['close', 'high', 'low', 'direction']:
            logger.info(f"  Training XGBoost for {target}...")
            
            if target == 'direction':
                # Classification
                model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric='logloss',
                    early_stopping_rounds=30
                )
                
                model.fit(
                    X_train_scaled, y_train_dict[target],
                    eval_set=[(X_val_scaled, y_val_dict[target])],
                    verbose=False
                )
            else:
                # Regression
                self.scalers[target] = RobustScaler()
                y_train_scaled = self.scalers[target].fit_transform(
                    y_train_dict[target].reshape(-1, 1)
                ).ravel()
                y_val_scaled = self.scalers[target].transform(
                    y_val_dict[target].reshape(-1, 1)
                ).ravel()
                
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric='rmse',
                    early_stopping_rounds=30
                )
                
                model.fit(
                    X_train_scaled, y_train_scaled,
                    eval_set=[(X_val_scaled, y_val_scaled)],
                    verbose=False
                )
            
            self.models[target] = model
    
    def predict(self, X):
        """Predict all targets."""
        X_scaled = self.feature_scaler.transform(X)
        
        predictions = {}
        for target in ['close', 'high', 'low', 'direction']:
            pred_scaled = self.models[target].predict(X_scaled)
            
            if target != 'direction':
                pred = self.scalers[target].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                pred = pred_scaled
            
            predictions[target] = pred
        
        return predictions
    
    def _save_models(self, symbol, save_dir):
        """Save XGBoost models."""
        for target, model in self.models.items():
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.json')
            model.save_model(model_path)
    
    def _load_models(self, symbol, save_dir):
        """Load XGBoost models."""
        for target in ['close', 'high', 'low', 'direction']:
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}.json')
            if target == 'direction':
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.models[target] = model


class EnsembleModel(BaseModel):
    """
    Ensemble model using stacking.
    Combines predictions from LSTM, GRU, and XGBoost.
    Best for: Maximum accuracy by leveraging strengths of all models.
    """
    
    def __init__(self, base_models: list = None):
        super().__init__('ensemble')
        self.base_models = base_models or []
        self.meta_models = {}  # Meta-learner for each target
    
    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        """
        Train ensemble using stacking.
        Base models should already be trained.
        """
        logger.info("  Training Ensemble (Stacking)...")
        
        # Get predictions from base models on validation set
        base_predictions_train = []
        base_predictions_val = []
        
        for model in self.base_models:
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            base_predictions_train.append(pred_train)
            base_predictions_val.append(pred_val)
        
        # Train meta-model for each target
        for target in ['close', 'high', 'low', 'direction']:
            logger.info(f"    Meta-model for {target}...")
            
            # Stack predictions from all base models
            meta_features_train = []
            meta_features_val = []
            
            for pred_dict in base_predictions_train:
                meta_features_train.append(pred_dict[target])
            
            for pred_dict in base_predictions_val:
                meta_features_val.append(pred_dict[target])
            
            # Convert to array (n_samples, n_base_models)
            meta_X_train = np.column_stack(meta_features_train)
            meta_X_val = np.column_stack(meta_features_val)
            
            # Remove NaN rows (from LSTM/GRU sequence padding)
            valid_train = ~np.any(np.isnan(meta_X_train), axis=1)
            valid_val = ~np.any(np.isnan(meta_X_val), axis=1)
            
            meta_X_train_clean = meta_X_train[valid_train]
            meta_y_train_clean = y_train_dict[target][valid_train]
            meta_X_val_clean = meta_X_val[valid_val]
            meta_y_val_clean = y_val_dict[target][valid_val]
            
            # Train meta-model (simple linear combination)
            if target == 'direction':
                from sklearn.linear_model import LogisticRegression
                meta_model = LogisticRegression(random_state=42)
            else:
                from sklearn.linear_model import Ridge
                meta_model = Ridge(alpha=1.0, random_state=42)
            
            meta_model.fit(meta_X_train_clean, meta_y_train_clean)
            self.meta_models[target] = meta_model
    
    def predict(self, X):
        """Predict using ensemble."""
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Combine using meta-models
        predictions = {}
        for target in ['close', 'high', 'low', 'direction']:
            # Stack predictions
            meta_features = []
            for pred_dict in base_predictions:
                meta_features.append(pred_dict[target])
            
            meta_X = np.column_stack(meta_features)
            
            # Handle NaN (use simple average as fallback)
            valid = ~np.any(np.isnan(meta_X), axis=1)
            
            pred = np.full(len(X), np.nan)
            if valid.sum() > 0:
                meta_X_clean = meta_X[valid]
                pred[valid] = self.meta_models[target].predict(meta_X_clean)
            
            predictions[target] = pred
        
        return predictions
    
    def _save_models(self, symbol, save_dir):
        """Save meta-models."""
        for target, model in self.meta_models.items():
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}_meta.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    def _load_models(self, symbol, save_dir):
        """Load meta-models."""
        for target in ['close', 'high', 'low', 'direction']:
            model_path = os.path.join(save_dir, f'{symbol}_{self.model_name}_{target}_meta.pkl')
            with open(model_path, 'rb') as f:
                self.meta_models[target] = pickle.load(f)
