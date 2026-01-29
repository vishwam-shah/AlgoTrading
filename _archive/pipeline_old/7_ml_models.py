"""
================================================================================
STEP 7: MACHINE LEARNING MODELS FOR STOCK PREDICTION
================================================================================

This module implements ML models for next-day close price prediction:
- XGBoost (Gradient Boosting)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer (Attention-based)
- Ensemble (combines all models)

Usage:
    from pipeline.7_ml_models import MLModelTrainer
    trainer = MLModelTrainer()
    models = trainer.train_all_models(features, targets)
    predictions = trainer.predict_ensemble(models, features)

================================================================================
"""

import os
import sys
import warnings
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    r2: float
    direction_accuracy: float
    train_samples: int
    test_samples: int
    train_time: float
    feature_count: int


@dataclass
class ModelPredictions:
    """Predictions from a model."""
    symbol: str
    model_name: str
    dates: List[datetime]
    actual: np.ndarray
    predicted: np.ndarray
    direction_actual: np.ndarray
    direction_predicted: np.ndarray
    confidence: np.ndarray = None


class XGBoostModel:
    """XGBoost model for close price prediction."""

    def __init__(self, params: Dict = None):
        """Initialize XGBoost model."""
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            logger.warning("XGBoost not available, using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingRegressor
            self.xgb = None
            self.sklearn_model = GradientBoostingRegressor

        self.params = params or config.XGBOOST_PARAMS.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None) -> Dict:
        """Train the model."""
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.xgb is not None:
            # Use XGBoost
            dtrain = self.xgb.DMatrix(X_train_scaled, label=y_train)

            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': self.params.get('max_depth', 6),
                'learning_rate': self.params.get('learning_rate', 0.01),
                'subsample': self.params.get('subsample', 0.8),
                'colsample_bytree': self.params.get('colsample_bytree', 0.8),
                'min_child_weight': self.params.get('min_child_weight', 5),
                'gamma': self.params.get('gamma', 0.1),
                'reg_alpha': self.params.get('reg_alpha', 0.1),
                'reg_lambda': self.params.get('reg_lambda', 1.0),
                'seed': config.RANDOM_SEED
            }

            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                dval = self.xgb.DMatrix(X_val_scaled, label=y_val)
                evals.append((dval, 'val'))

            self.model = self.xgb.train(
                params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 500),
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False
            )
        else:
            # Use sklearn fallback
            self.model = self.sklearn_model(
                n_estimators=min(self.params.get('n_estimators', 500), 200),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.01),
                subsample=self.params.get('subsample', 0.8),
                random_state=config.RANDOM_SEED
            )
            self.model.fit(X_train_scaled, y_train)

        return {'status': 'trained'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)

        if self.xgb is not None:
            dtest = self.xgb.DMatrix(X_scaled)
            return self.model.predict(dtest)
        else:
            return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.xgb is not None and self.model is not None:
            importance = self.model.get_score(importance_type='gain')
            if self.feature_names:
                return {self.feature_names[int(k[1:])]: v for k, v in importance.items() if k.startswith('f')}
            return importance
        elif self.model is not None:
            importance = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            return dict(enumerate(importance))
        return {}


class LSTMModel:
    """LSTM model for sequence prediction."""

    def __init__(self, params: Dict = None):
        """Initialize LSTM model."""
        self.params = params or config.LSTM_PARAMS.copy()
        self.model = None
        self.scaler = RobustScaler()
        self.sequence_length = config.SEQUENCE_LENGTH

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam

            units = self.params.get('units', [128, 64, 32])
            dropout = self.params.get('dropout', 0.3)

            model = Sequential()

            # First LSTM layer
            model.add(LSTM(units[0], return_sequences=len(units) > 1,
                          input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            # Additional LSTM layers
            for i, u in enumerate(units[1:]):
                return_seq = i < len(units) - 2
                model.add(LSTM(u, return_sequences=return_seq))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            # Output layer
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))

            model.compile(
                optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )

            return model
        except ImportError:
            logger.warning("TensorFlow not available for LSTM")
            return None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])

        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences), None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None) -> Dict:
        """Train the LSTM model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train)

        if len(X_seq) < 50:
            logger.warning(f"Insufficient samples for LSTM: {len(X_seq)}")
            return {'status': 'skipped', 'reason': 'insufficient_samples'}

        # Build model
        self.model = self._build_model((self.sequence_length, X_train.shape[1]))

        if self.model is None:
            return {'status': 'skipped', 'reason': 'tensorflow_not_available'}

        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            callbacks = [
                EarlyStopping(patience=self.params.get('early_stopping_patience', 15),
                             restore_best_weights=True),
                ReduceLROnPlateau(patience=self.params.get('reduce_lr_patience', 8),
                                 factor=0.5)
            ]

            # Validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
                if len(X_val_seq) > 0:
                    validation_data = (X_val_seq, y_val_seq)

            # Train
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0
            )

            return {'status': 'trained', 'history': history.history}
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            return np.zeros(len(X) - self.sequence_length)

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.array([])

        return self.model.predict(X_seq, verbose=0).flatten()


class GRUModel:
    """GRU model for sequence prediction (faster than LSTM)."""

    def __init__(self, params: Dict = None):
        """Initialize GRU model."""
        self.params = params or config.GRU_PARAMS.copy()
        self.model = None
        self.scaler = RobustScaler()
        self.sequence_length = config.SEQUENCE_LENGTH

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build GRU architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam

            units = self.params.get('units', [128, 64, 32])
            dropout = self.params.get('dropout', 0.3)

            model = Sequential()

            # First GRU layer
            model.add(GRU(units[0], return_sequences=len(units) > 1,
                         input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            # Additional GRU layers
            for i, u in enumerate(units[1:]):
                return_seq = i < len(units) - 2
                model.add(GRU(u, return_sequences=return_seq))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            # Output layer
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))

            model.compile(
                optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )

            return model
        except ImportError:
            logger.warning("TensorFlow not available for GRU")
            return None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU input."""
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])

        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences), None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None) -> Dict:
        """Train the GRU model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train)

        if len(X_seq) < 50:
            logger.warning(f"Insufficient samples for GRU: {len(X_seq)}")
            return {'status': 'skipped', 'reason': 'insufficient_samples'}

        # Build model
        self.model = self._build_model((self.sequence_length, X_train.shape[1]))

        if self.model is None:
            return {'status': 'skipped', 'reason': 'tensorflow_not_available'}

        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            callbacks = [
                EarlyStopping(patience=self.params.get('early_stopping_patience', 15),
                             restore_best_weights=True),
                ReduceLROnPlateau(patience=self.params.get('reduce_lr_patience', 8),
                                 factor=0.5)
            ]

            # Validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
                if len(X_val_seq) > 0:
                    validation_data = (X_val_seq, y_val_seq)

            # Train
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0
            )

            return {'status': 'trained', 'history': history.history}
        except Exception as e:
            logger.error(f"GRU training failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            return np.zeros(len(X) - self.sequence_length)

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.array([])

        return self.model.predict(X_seq, verbose=0).flatten()


class LightGBMModel:
    """LightGBM model for fast gradient boosting."""

    def __init__(self, params: Dict = None):
        """Initialize LightGBM model."""
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            logger.warning("LightGBM not available, using sklearn")
            from sklearn.ensemble import GradientBoostingRegressor
            self.lgb = None
            self.sklearn_model = GradientBoostingRegressor

        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None) -> Dict:
        """Train the model."""
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.lgb is not None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': config.RANDOM_SEED
            }
            params.update(self.params)

            train_data = self.lgb.Dataset(X_train_scaled, label=y_train)
            valid_sets = [train_data]
            valid_names = ['train']

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_data = self.lgb.Dataset(X_val_scaled, label=y_val)
                valid_sets.append(val_data)
                valid_names.append('val')

            self.model = self.lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[self.lgb.early_stopping(50), self.lgb.log_evaluation(0)]
            )
        else:
            self.model = self.sklearn_model(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                random_state=config.RANDOM_SEED
            )
            self.model.fit(X_train_scaled, y_train)

        return {'status': 'trained'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.lgb is not None and self.model is not None:
            importance = self.model.feature_importance(importance_type='gain')
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            return dict(enumerate(importance))
        elif self.model is not None:
            importance = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            return dict(enumerate(importance))
        return {}


class EnsembleModel:
    """Ensemble combining multiple models using stacking."""

    def __init__(self, models: Dict[str, Any], meta_learner: str = 'ridge'):
        """
        Initialize ensemble model.

        Args:
            models: Dict of model_name -> trained model
            meta_learner: Type of meta learner ('ridge', 'linear', 'average')
        """
        self.models = models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        self.weights = None

    def train_meta(self, X: np.ndarray, y: np.ndarray):
        """Train meta-learner on model predictions."""
        # Get predictions from base models
        base_predictions = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                if len(pred) > 0:
                    base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if len(base_predictions) == 0:
            logger.error("No base model predictions available")
            return

        # Align predictions (handle different lengths from sequence models)
        min_len = min(len(p) for p in base_predictions)
        base_predictions = [p[-min_len:] for p in base_predictions]
        y_aligned = y[-min_len:]

        # Stack predictions
        X_meta = np.column_stack(base_predictions)

        if self.meta_learner_type == 'ridge':
            self.meta_learner = Ridge(alpha=1.0)
            self.meta_learner.fit(X_meta, y_aligned)
            self.weights = self.meta_learner.coef_
        elif self.meta_learner_type == 'average':
            self.weights = np.ones(len(base_predictions)) / len(base_predictions)
        else:
            # Simple average
            self.weights = np.ones(len(base_predictions)) / len(base_predictions)

        logger.info(f"Ensemble weights: {dict(zip(self.models.keys(), self.weights))}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        base_predictions = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                if len(pred) > 0:
                    base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if len(base_predictions) == 0:
            return np.zeros(len(X))

        # Align predictions
        min_len = min(len(p) for p in base_predictions)
        base_predictions = [p[-min_len:] for p in base_predictions]

        # Stack and predict
        X_meta = np.column_stack(base_predictions)

        if self.meta_learner is not None:
            return self.meta_learner.predict(X_meta)
        else:
            # Weighted average
            return np.average(base_predictions, axis=0, weights=self.weights)


class MLModelTrainer:
    """
    Main class for training and evaluating ML models.
    """

    def __init__(self, output_dir: str = None):
        """Initialize trainer."""
        self.output_dir = output_dir or os.path.join(config.MODEL_DIR, 'trained')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.metrics = {}
        self.predictions = {}

        logger.info("MLModelTrainer initialized")

    def prepare_data(self, features_df: pd.DataFrame,
                     target_col: str = 'close_return') -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        """
        Prepare data for model training.

        Args:
            features_df: DataFrame with features and target
            target_col: Name of target column

        Returns:
            X, y, feature_names, dates
        """
        # Compute target if not present
        if target_col not in features_df.columns:
            # Compute close return (next day)
            features_df = features_df.copy()
            features_df['close_return'] = features_df['close'].pct_change().shift(-1)
            features_df = features_df.dropna()

        # Exclude non-feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol',
                       'close_return', 'direction_target', 'timestamp', 'date']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

        # Get feature matrix
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        dates = features_df.index

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)

        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")

        return X, y, feature_cols, dates

    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None,
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all configured ML models.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            test_size: Fraction of data for testing

        Returns:
            Dict of model_name -> trained model
        """
        logger.info("=" * 60)
        logger.info("TRAINING ML MODELS")
        logger.info("=" * 60)

        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Further split training for validation
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        logger.info(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

        trained_models = {}

        # Train XGBoost
        if config.ENABLE_MODELS.get('xgboost', True):
            logger.info("\n[1/4] Training XGBoost...")
            import time
            start = time.time()

            xgb_model = XGBoostModel()
            xgb_model.train(X_tr, y_tr, X_val, y_val, feature_names)
            trained_models['xgboost'] = xgb_model

            # Evaluate
            pred = xgb_model.predict(X_test)
            metrics = self._calculate_metrics('xgboost', y_test, pred,
                                             len(X_tr), len(X_test),
                                             time.time() - start, len(feature_names or []))
            self.metrics['xgboost'] = metrics
            logger.info(f"  RMSE: {metrics.rmse:.4f}, Direction: {metrics.direction_accuracy:.2%}")

        # Train LightGBM
        logger.info("\n[2/4] Training LightGBM...")
        import time
        start = time.time()

        lgb_model = LightGBMModel()
        lgb_model.train(X_tr, y_tr, X_val, y_val, feature_names)
        trained_models['lightgbm'] = lgb_model

        pred = lgb_model.predict(X_test)
        metrics = self._calculate_metrics('lightgbm', y_test, pred,
                                         len(X_tr), len(X_test),
                                         time.time() - start, len(feature_names or []))
        self.metrics['lightgbm'] = metrics
        logger.info(f"  RMSE: {metrics.rmse:.4f}, Direction: {metrics.direction_accuracy:.2%}")

        # Train LSTM
        if config.ENABLE_MODELS.get('lstm', True):
            logger.info("\n[3/4] Training LSTM...")
            start = time.time()

            lstm_model = LSTMModel()
            result = lstm_model.train(X_tr, y_tr, X_val, y_val, feature_names)

            if result.get('status') == 'trained':
                trained_models['lstm'] = lstm_model

                pred = lstm_model.predict(X_test)
                if len(pred) > 0:
                    y_test_aligned = y_test[-len(pred):]
                    metrics = self._calculate_metrics('lstm', y_test_aligned, pred,
                                                     len(X_tr), len(pred),
                                                     time.time() - start, len(feature_names or []))
                    self.metrics['lstm'] = metrics
                    logger.info(f"  RMSE: {metrics.rmse:.4f}, Direction: {metrics.direction_accuracy:.2%}")
            else:
                logger.warning(f"  LSTM skipped: {result.get('reason', 'unknown')}")

        # Train GRU
        if config.ENABLE_MODELS.get('gru', True):
            logger.info("\n[4/4] Training GRU...")
            start = time.time()

            gru_model = GRUModel()
            result = gru_model.train(X_tr, y_tr, X_val, y_val, feature_names)

            if result.get('status') == 'trained':
                trained_models['gru'] = gru_model

                pred = gru_model.predict(X_test)
                if len(pred) > 0:
                    y_test_aligned = y_test[-len(pred):]
                    metrics = self._calculate_metrics('gru', y_test_aligned, pred,
                                                     len(X_tr), len(pred),
                                                     time.time() - start, len(feature_names or []))
                    self.metrics['gru'] = metrics
                    logger.info(f"  RMSE: {metrics.rmse:.4f}, Direction: {metrics.direction_accuracy:.2%}")
            else:
                logger.warning(f"  GRU skipped: {result.get('reason', 'unknown')}")

        # Train Ensemble
        if config.ENABLE_MODELS.get('ensemble', True) and len(trained_models) >= 2:
            logger.info("\n[5/5] Training Ensemble...")
            start = time.time()

            ensemble = EnsembleModel(trained_models, meta_learner='ridge')
            ensemble.train_meta(X_val, y_val)
            trained_models['ensemble'] = ensemble

            pred = ensemble.predict(X_test)
            metrics = self._calculate_metrics('ensemble', y_test[-len(pred):], pred,
                                             len(X_tr), len(pred),
                                             time.time() - start, len(feature_names or []))
            self.metrics['ensemble'] = metrics
            logger.info(f"  RMSE: {metrics.rmse:.4f}, Direction: {metrics.direction_accuracy:.2%}")

        self.models = trained_models
        return trained_models

    def _calculate_metrics(self, model_name: str, y_true: np.ndarray,
                          y_pred: np.ndarray, train_samples: int,
                          test_samples: int, train_time: float,
                          feature_count: int) -> ModelMetrics:
        """Calculate model performance metrics."""
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0

        r2 = r2_score(y_true, y_pred)

        # Direction accuracy
        direction_actual = (y_true > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = (direction_actual == direction_pred).mean()

        return ModelMetrics(
            model_name=model_name,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2=r2,
            direction_accuracy=direction_accuracy,
            train_samples=train_samples,
            test_samples=test_samples,
            train_time=train_time,
            feature_count=feature_count
        )

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if 'ensemble' in self.models:
            return self.models['ensemble'].predict(X)
        elif self.models:
            # Fall back to best individual model
            best_model = min(self.metrics.items(), key=lambda x: x[1].rmse)[0]
            return self.models[best_model].predict(X)
        else:
            return np.zeros(len(X))

    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as DataFrame."""
        rows = []
        for name, metrics in self.metrics.items():
            rows.append({
                'model': name,
                'rmse': metrics.rmse,
                'mae': metrics.mae,
                'mape': metrics.mape,
                'r2': metrics.r2,
                'direction_accuracy': metrics.direction_accuracy,
                'train_samples': metrics.train_samples,
                'test_samples': metrics.test_samples,
                'train_time': metrics.train_time
            })
        return pd.DataFrame(rows)

    def save_models(self, prefix: str = 'model'):
        """Save all trained models."""
        for name, model in self.models.items():
            path = os.path.join(self.output_dir, f'{prefix}_{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {path}")

    def load_models(self, prefix: str = 'model'):
        """Load trained models."""
        for name in ['xgboost', 'lightgbm', 'lstm', 'gru', 'ensemble']:
            path = os.path.join(self.output_dir, f'{prefix}_{name}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded {name} from {path}")


if __name__ == "__main__":
    # Test the models
    print("Testing ML Models...")

    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 500
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)  # Non-linear target

    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Train models
    trainer = MLModelTrainer()
    models = trainer.train_all_models(X, y, feature_names)

    # Print metrics
    print("\nModel Metrics:")
    print(trainer.get_metrics_df().to_string())
