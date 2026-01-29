"""
================================================================================
UNIFIED ML MODELS
================================================================================
Merged ML models from pipeline/ and production/ systems.

From pipeline (7_ml_models.py):
- XGBoostModel, LightGBMModel, LSTMModel, GRUModel, EnsembleModel
- MLModelTrainer with train_all_models(), predict_ensemble()

From production (models.py):
- ProductionModel: Cascaded XGBoost+LightGBM direction classifier + return
  regressor + quantile models + confidence scoring
- ModelEvaluator: Comprehensive prediction evaluation

All models expose unified train()/predict() interface.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ModelPrediction:
    """Structured prediction output (production-style)."""
    direction: int  # 0=down, 1=up
    direction_probability: float
    expected_return: float
    confidence: float  # 0-1
    upper_bound: float
    lower_bound: float
    prediction_5d: Optional[float] = None


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    direction_accuracy: float
    rmse: float
    mae: float
    r2_score: float
    feature_importance: Optional[Dict] = None


@dataclass
class ModelPredictions:
    """Predictions from a single model."""
    model_name: str
    predictions: np.ndarray
    direction_predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None


# ============================================================================
# PIPELINE MODELS (XGBoost, LightGBM, LSTM, GRU, Ensemble)
# ============================================================================

class XGBoostModel:
    """XGBoost model with sklearn fallback."""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.feature_names = feature_names
        X_train_scaled = self.scaler.fit_transform(X_train)

        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=self.params.get('n_estimators', 500),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.01),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                min_child_weight=self.params.get('min_child_weight', 5),
                gamma=self.params.get('gamma', 0.1),
                reg_alpha=self.params.get('reg_alpha', 0.1),
                reg_lambda=self.params.get('reg_lambda', 1.0),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=50
            )

            eval_set = []
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_val_scaled, y_val)]

            self.model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=min(self.params.get('n_estimators', 500), 200),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.01),
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

        logger.info("XGBoost model trained")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


class LightGBMModel:
    """LightGBM model with sklearn fallback."""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.feature_names = feature_names
        X_train_scaled = self.scaler.fit_transform(X_train)

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=self.params.get('n_estimators', 500),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.01),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                num_leaves=self.params.get('num_leaves', 31),
                reg_alpha=self.params.get('reg_alpha', 0.1),
                reg_lambda=self.params.get('reg_lambda', 1.0),
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

            eval_set = []
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_val_scaled, y_val)]

            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None
            )

        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=min(self.params.get('n_estimators', 500), 200),
                max_depth=self.params.get('max_depth', 6),
                learning_rate=self.params.get('learning_rate', 0.01),
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

        logger.info("LightGBM model trained")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


class LSTMModel:
    """LSTM model (requires tensorflow)."""

    def __init__(self, params: Dict = None):
        self.params = params or config.LSTM_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = self.params.get('sequence_length', 60)

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)

            units = self.params.get('units', [128, 64, 32])
            dropout = self.params.get('dropout', 0.3)

            model = Sequential()
            model.add(LSTM(units[0], return_sequences=True,
                          input_shape=(self.sequence_length, X_train.shape[1])))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            for u in units[1:-1]:
                model.add(LSTM(u, return_sequences=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            model.add(LSTM(units[-1], return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(Dense(1))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.params.get('learning_rate', 0.001)
                ),
                loss='mse'
            )

            callbacks = [
                EarlyStopping(patience=self.params.get('early_stopping_patience', 15),
                            restore_best_weights=True),
                ReduceLROnPlateau(patience=self.params.get('reduce_lr_patience', 8),
                                factor=0.5, min_lr=1e-6)
            ]

            val_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
                if len(X_val_seq) > 0:
                    val_data = (X_val_seq, y_val_seq)

            model.fit(
                X_train_seq, y_train_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                validation_data=val_data,
                callbacks=callbacks,
                verbose=0
            )

            self.model = model
            logger.info("LSTM model trained")

        except ImportError:
            logger.warning("TensorFlow not available, LSTM model not trained")

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.zeros(len(X))
        preds = self.model.predict(X_seq, verbose=0).flatten()
        # Pad to match original length
        result = np.zeros(len(X))
        result[self.sequence_length:self.sequence_length + len(preds)] = preds
        return result

    def _create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)


class GRUModel:
    """GRU model (requires tensorflow). Same architecture as LSTM but with GRU layers."""

    def __init__(self, params: Dict = None):
        self.params = params or config.GRU_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = self.params.get('sequence_length', 60)

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)

            units = self.params.get('units', [128, 64, 32])
            dropout = self.params.get('dropout', 0.3)

            model = Sequential()
            model.add(GRU(units[0], return_sequences=True,
                         input_shape=(self.sequence_length, X_train.shape[1])))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            for u in units[1:-1]:
                model.add(GRU(u, return_sequences=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            model.add(GRU(units[-1], return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(Dense(1))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.params.get('learning_rate', 0.001)
                ),
                loss='mse'
            )

            callbacks = [
                EarlyStopping(patience=self.params.get('early_stopping_patience', 15),
                            restore_best_weights=True),
                ReduceLROnPlateau(patience=self.params.get('reduce_lr_patience', 8),
                                factor=0.5, min_lr=1e-6)
            ]

            val_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
                if len(X_val_seq) > 0:
                    val_data = (X_val_seq, y_val_seq)

            model.fit(
                X_train_seq, y_train_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                validation_data=val_data,
                callbacks=callbacks,
                verbose=0
            )

            self.model = model
            logger.info("GRU model trained")

        except ImportError:
            logger.warning("TensorFlow not available, GRU model not trained")

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.zeros(len(X))
        preds = self.model.predict(X_seq, verbose=0).flatten()
        result = np.zeros(len(X))
        result[self.sequence_length:self.sequence_length + len(preds)] = preds
        return result

    def _create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)


class EnsembleModel:
    """Stacking ensemble with Ridge meta-learner."""

    def __init__(self):
        self.base_models = {}
        self.meta_learner = Ridge(alpha=1.0)
        self.is_trained = False

    def train(self, base_predictions: Dict[str, np.ndarray], y_train: np.ndarray):
        """Train meta-learner on base model predictions."""
        # Align predictions to same length
        min_len = min(len(v) for v in base_predictions.values())
        min_len = min(min_len, len(y_train))

        X_meta = np.column_stack([v[:min_len] for v in base_predictions.values()])
        y_meta = y_train[:min_len]

        # Remove NaN
        valid = ~(np.isnan(X_meta).any(axis=1) | np.isnan(y_meta))
        if valid.sum() < 10:
            logger.warning("Not enough valid samples for ensemble training")
            return

        self.meta_learner.fit(X_meta[valid], y_meta[valid])
        self.base_models = list(base_predictions.keys())
        self.is_trained = True
        logger.info(f"Ensemble trained on {len(self.base_models)} base models")

    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            # Simple average fallback
            preds = list(base_predictions.values())
            return np.mean(preds, axis=0)

        min_len = min(len(v) for v in base_predictions.values())
        X_meta = np.column_stack([v[:min_len] for v in base_predictions.values()])
        X_meta = np.nan_to_num(X_meta, nan=0.0)
        return self.meta_learner.predict(X_meta)


class MLModelTrainer:
    """Unified trainer for all ML models."""

    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.predictions = {}
        self.scaler = StandardScaler()

    def prepare_data(self, features_df: pd.DataFrame, target_col: str = 'close'):
        """Prepare train/val/test splits from feature DataFrame."""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                       'date', 'symbol'] + [c for c in features_df.columns if c.startswith('target_')]
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[feature_cols].values
        y = features_df['close'].pct_change().shift(-1).values  # Next day return

        # Remove NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        # Chronological split
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        return {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:],
            'feature_names': feature_cols
        }

    def train_all_models(self, data: Dict, models_to_train: List[str] = None):
        """Train all specified models."""
        if models_to_train is None:
            models_to_train = ['xgboost', 'lightgbm']

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        feature_names = data.get('feature_names')

        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")

            if model_name == 'xgboost':
                model = XGBoostModel(config.XGBOOST_PARAMS)
            elif model_name == 'lightgbm':
                model = LightGBMModel()
            elif model_name == 'lstm':
                model = LSTMModel(config.LSTM_PARAMS)
            elif model_name == 'gru':
                model = GRUModel(config.GRU_PARAMS)
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue

            model.train(X_train, y_train, X_val, y_val, feature_names)
            self.models[model_name] = model

            # Get validation predictions
            val_preds = model.predict(X_val)
            self.predictions[model_name] = val_preds

            # Calculate metrics
            valid = ~np.isnan(val_preds)
            if valid.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_val[valid], val_preds[valid]))
                mae = mean_absolute_error(y_val[valid], val_preds[valid])
                dir_acc = accuracy_score(
                    (y_val[valid] > 0).astype(int),
                    (val_preds[valid] > 0).astype(int)
                )

                self.metrics[model_name] = ModelMetrics(
                    model_name=model_name,
                    direction_accuracy=dir_acc,
                    rmse=rmse,
                    mae=mae,
                    r2_score=1 - (np.sum((y_val[valid] - val_preds[valid])**2) /
                                  (np.sum((y_val[valid] - y_val[valid].mean())**2) + 1e-10)),
                    feature_importance=model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
                )

                logger.info(f"  {model_name}: Dir Acc={dir_acc:.2%}, RMSE={rmse:.6f}")

        # Train ensemble if multiple models
        if len(self.models) >= 2:
            logger.info("Training ensemble...")
            ensemble = EnsembleModel()
            ensemble.train(self.predictions, y_val)
            self.models['ensemble'] = ensemble

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction from all trained models."""
        base_preds = {}
        for name, model in self.models.items():
            if name != 'ensemble':
                base_preds[name] = model.predict(X)

        if 'ensemble' in self.models and self.models['ensemble'].is_trained:
            return self.models['ensemble'].predict(base_preds)
        else:
            return np.mean(list(base_preds.values()), axis=0)

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
        logger.info(f"Models saved to {save_dir}")

    def load_models(self, load_dir: str):
        import glob
        for path in glob.glob(os.path.join(load_dir, '*.pkl')):
            name = os.path.splitext(os.path.basename(path))[0]
            self.models[name] = joblib.load(path)
        logger.info(f"Loaded {len(self.models)} models from {load_dir}")


# ============================================================================
# PRODUCTION MODEL (Cascaded XGBoost+LightGBM ensemble)
# ============================================================================

class ProductionModel:
    """
    Production-grade prediction model with cascaded architecture.

    1. Direction Classifier (XGBoost + LightGBM ensemble)
    2. Return Regressor (XGBoost with quantile estimation)
    3. Confidence Scorer (model agreement + direction strength + return uncertainty)
    """

    def __init__(self, model_name: str = 'production'):
        self.model_name = model_name
        self.direction_xgb = None
        self.direction_lgb = None
        self.return_model = None
        self.return_upper_model = None
        self.return_lower_model = None
        self.return_5d_model = None
        self.feature_scaler = RobustScaler()
        self.feature_names = None
        self.training_date = None
        self.n_features = None
        self.validation_metrics = {}

    def train(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Train all model components."""
        from xgboost import XGBClassifier, XGBRegressor
        import lightgbm as lgb

        logger.info("Training Production Model...")
        self.training_date = datetime.now()
        self.feature_names = feature_names
        self.n_features = X_train.shape[1]

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        metrics = {}

        # 1. Direction Classifier (XGBoost)
        logger.info("Training Direction Classifier (XGBoost)...")
        self.direction_xgb = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=self._compute_class_weight(y_train['direction']),
            random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=50, eval_metric='logloss'
        )
        self.direction_xgb.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])], verbose=False
        )

        xgb_dir_acc = accuracy_score(y_val['direction'],
                                     self.direction_xgb.predict(X_val_scaled))

        # 2. Direction Classifier (LightGBM)
        logger.info("Training Direction Classifier (LightGBM)...")
        self.direction_lgb = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            random_state=42, n_jobs=-1, verbosity=-1
        )
        self.direction_lgb.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        lgb_dir_acc = accuracy_score(y_val['direction'],
                                    self.direction_lgb.predict(X_val_scaled))

        # Ensemble
        dir_proba_xgb = self.direction_xgb.predict_proba(X_val_scaled)[:, 1]
        dir_proba_lgb = self.direction_lgb.predict_proba(X_val_scaled)[:, 1]
        dir_proba_ensemble = (dir_proba_xgb + dir_proba_lgb) / 2
        ensemble_dir_acc = accuracy_score(y_val['direction'],
                                         (dir_proba_ensemble > 0.5).astype(int))

        metrics['direction_accuracy_xgb'] = xgb_dir_acc
        metrics['direction_accuracy_lgb'] = lgb_dir_acc
        metrics['direction_accuracy_ensemble'] = ensemble_dir_acc

        # 3. Return Regressor
        logger.info("Training Return Regressor...")
        self.return_model = XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=50
        )
        self.return_model.fit(
            X_train_scaled, y_train['close_return'],
            eval_set=[(X_val_scaled, y_val['close_return'])], verbose=False
        )

        return_pred = self.return_model.predict(X_val_scaled)
        metrics['return_mae'] = mean_absolute_error(y_val['close_return'], return_pred)
        metrics['return_rmse'] = np.sqrt(mean_squared_error(y_val['close_return'], return_pred))

        # 4. Quantile Models
        logger.info("Training Quantile Models...")
        self.return_upper_model = lgb.LGBMRegressor(
            objective='quantile', alpha=0.75, n_estimators=300, max_depth=4,
            learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1
        )
        self.return_upper_model.fit(X_train_scaled, y_train['close_return'])

        self.return_lower_model = lgb.LGBMRegressor(
            objective='quantile', alpha=0.25, n_estimators=300, max_depth=4,
            learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1
        )
        self.return_lower_model.fit(X_train_scaled, y_train['close_return'])

        # 5. 5-day return model
        if 'close_return_5d' in y_train and y_train['close_return_5d'] is not None:
            close_5d = y_train['close_return_5d']
            valid_mask = ~np.isnan(close_5d) & ~np.isinf(close_5d)
            if valid_mask.sum() > 100:
                logger.info("Training 5-Day Return Model...")
                self.return_5d_model = XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42, n_jobs=-1, verbosity=0
                )
                self.return_5d_model.fit(X_train_scaled[valid_mask], close_5d[valid_mask])

        self.validation_metrics = metrics
        logger.success(f"Training complete. Ensemble accuracy: {ensemble_dir_acc:.2%}")
        return metrics

    def predict(self, X: np.ndarray) -> List[ModelPrediction]:
        """Generate predictions with confidence scores."""
        X_scaled = self.feature_scaler.transform(X)

        dir_proba_xgb = self.direction_xgb.predict_proba(X_scaled)[:, 1]
        dir_proba_lgb = self.direction_lgb.predict_proba(X_scaled)[:, 1]
        dir_proba = (dir_proba_xgb + dir_proba_lgb) / 2
        directions = (dir_proba > 0.5).astype(int)

        expected_returns = self.return_model.predict(X_scaled)
        upper_bounds = self.return_upper_model.predict(X_scaled)
        lower_bounds = self.return_lower_model.predict(X_scaled)

        predictions_5d = None
        if self.return_5d_model is not None:
            predictions_5d = self.return_5d_model.predict(X_scaled)

        confidences = self._compute_confidence(
            dir_proba_xgb, dir_proba_lgb, dir_proba,
            expected_returns, upper_bounds, lower_bounds
        )

        predictions = []
        for i in range(len(X)):
            pred = ModelPrediction(
                direction=int(directions[i]),
                direction_probability=float(dir_proba[i]),
                expected_return=float(expected_returns[i]),
                confidence=float(confidences[i]),
                upper_bound=float(upper_bounds[i]),
                lower_bound=float(lower_bounds[i]),
                prediction_5d=float(predictions_5d[i]) if predictions_5d is not None else None
            )
            predictions.append(pred)

        return predictions

    def predict_single(self, X: np.ndarray) -> ModelPrediction:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.predict(X)[0]

    def _compute_confidence(self, proba_xgb, proba_lgb, proba_ensemble,
                           returns, upper, lower):
        model_agreement = 1 - np.abs(proba_xgb - proba_lgb)
        direction_strength = np.abs(proba_ensemble - 0.5) * 2
        return_spread = upper - lower
        median_spread = np.median(return_spread[~np.isnan(return_spread)])
        return_confidence = np.clip(1 - (return_spread / (median_spread * 2)), 0, 1)
        return 0.4 * model_agreement + 0.4 * direction_strength + 0.2 * return_confidence

    def _compute_class_weight(self, y):
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        if n_positive == 0 or n_negative == 0:
            return 1.0
        return n_negative / n_positive

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.direction_xgb is None:
            raise ValueError("Model not trained yet")
        importance = self.direction_xgb.feature_importances_
        if self.feature_names is not None:
            df = pd.DataFrame({'feature': self.feature_names[:len(importance)],
                             'importance': importance})
        else:
            df = pd.DataFrame({'feature': [f'f_{i}' for i in range(len(importance))],
                             'importance': importance})
        return df.sort_values('importance', ascending=False).head(top_n)

    def predict_with_model(self, X: np.ndarray, model_type: str) -> List[ModelPrediction]:
        """Get predictions from a specific model (xgb or lgb) instead of ensemble."""
        X_scaled = self.feature_scaler.transform(X)
        
        if model_type == 'xgb':
            dir_proba = self.direction_xgb.predict_proba(X_scaled)[:, 1]
        elif model_type == 'lgb':
            dir_proba = self.direction_lgb.predict_proba(X_scaled)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        directions = (dir_proba > 0.5).astype(int)
        expected_returns = self.return_model.predict(X_scaled)
        upper_bounds = self.return_upper_model.predict(X_scaled)
        lower_bounds = self.return_lower_model.predict(X_scaled)
        
        predictions_5d = None
        if self.return_5d_model is not None:
            predictions_5d = self.return_5d_model.predict(X_scaled)
        
        confidences = np.abs(dir_proba - 0.5) * 2
        
        predictions = []
        for i in range(len(X)):
            pred = ModelPrediction(
                direction=int(directions[i]),
                direction_probability=float(dir_proba[i]),
                expected_return=float(expected_returns[i]),
                confidence=float(confidences[i]),
                upper_bound=float(upper_bounds[i]),
                lower_bound=float(lower_bounds[i]),
                prediction_5d=float(predictions_5d[i]) if predictions_5d is not None else None
            )
            predictions.append(pred)
        
        return predictions

    def get_feature_importance(self, feature_names: List[str] = None) -> List[Dict[str, float]]:
        """Get feature importance scores sorted by average importance."""
        if feature_names is None:
            feature_names = self.feature_names
        
        if feature_names is None or len(feature_names) == 0:
            return []
        
        xgb_importance = self.direction_xgb.feature_importances_
        reg_importance = self.return_model.feature_importances_
        avg_importance = (xgb_importance + reg_importance) / 2
        
        importance_list = []
        for i, feat_name in enumerate(feature_names):
            importance_list.append({
                'feature': feat_name,
                'importance': float(avg_importance[i])
            })
        
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        return importance_list

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.direction_xgb, os.path.join(save_dir, 'direction_xgb.pkl'))
        joblib.dump(self.direction_lgb, os.path.join(save_dir, 'direction_lgb.pkl'))
        joblib.dump(self.return_model, os.path.join(save_dir, 'return_model.pkl'))
        joblib.dump(self.return_upper_model, os.path.join(save_dir, 'return_upper.pkl'))
        joblib.dump(self.return_lower_model, os.path.join(save_dir, 'return_lower.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(save_dir, 'scaler.pkl'))
        if self.return_5d_model is not None:
            joblib.dump(self.return_5d_model, os.path.join(save_dir, 'return_5d.pkl'))
        metadata = {
            'model_name': self.model_name,
            'training_date': str(self.training_date),
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'validation_metrics': self.validation_metrics
        }
        joblib.dump(metadata, os.path.join(save_dir, 'metadata.pkl'))
        logger.info(f"Model saved to {save_dir}")

    def load(self, load_dir: str):
        self.direction_xgb = joblib.load(os.path.join(load_dir, 'direction_xgb.pkl'))
        self.direction_lgb = joblib.load(os.path.join(load_dir, 'direction_lgb.pkl'))
        self.return_model = joblib.load(os.path.join(load_dir, 'return_model.pkl'))
        self.return_upper_model = joblib.load(os.path.join(load_dir, 'return_upper.pkl'))
        self.return_lower_model = joblib.load(os.path.join(load_dir, 'return_lower.pkl'))
        self.feature_scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        return_5d_path = os.path.join(load_dir, 'return_5d.pkl')
        if os.path.exists(return_5d_path):
            self.return_5d_model = joblib.load(return_5d_path)
        metadata = joblib.load(os.path.join(load_dir, 'metadata.pkl'))
        self.model_name = metadata['model_name']
        self.training_date = metadata['training_date']
        self.n_features = metadata['n_features']
        self.feature_names = metadata['feature_names']
        self.validation_metrics = metadata['validation_metrics']
        logger.info(f"Model loaded from {load_dir}")


# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation."""

    @staticmethod
    def evaluate_predictions(y_true_direction, y_true_return, predictions):
        pred_directions = np.array([p.direction for p in predictions])
        pred_returns = np.array([p.expected_return for p in predictions])
        confidences = np.array([p.confidence for p in predictions])

        direction_accuracy = accuracy_score(y_true_direction, pred_directions)

        high_conf_mask = confidences > 0.6
        high_conf_accuracy = (accuracy_score(y_true_direction[high_conf_mask],
                                            pred_directions[high_conf_mask])
                             if high_conf_mask.sum() > 0 else 0)

        return_mae = mean_absolute_error(y_true_return, pred_returns)
        return_rmse = np.sqrt(mean_squared_error(y_true_return, pred_returns))

        strategy_returns = y_true_return * (pred_directions * 2 - 1)

        return {
            'direction_accuracy': direction_accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_trades': int(high_conf_mask.sum()),
            'return_mae': return_mae,
            'return_rmse': return_rmse,
            'strategy_total_return': float(strategy_returns.sum()),
            'strategy_sharpe': float(strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)),
            'avg_confidence': float(confidences.mean())
        }
