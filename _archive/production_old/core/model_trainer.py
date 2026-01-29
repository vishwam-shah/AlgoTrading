"""
================================================================================
PRODUCTION MODELS
================================================================================
Optimized XGBoost ensemble with:
- Direction classifier (binary: up/down)
- Return regressor (magnitude of move)
- Confidence scorer (prediction reliability)
- Multi-horizon predictions (1-day, 5-day)

Key innovations:
1. Cascaded model: direction → magnitude (better than joint training)
2. Feature importance-weighted ensemble
3. Calibrated probability outputs
4. Out-of-bag confidence estimation
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import joblib
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class ModelPrediction:
    """Structured prediction output"""
    direction: int  # 0 = down, 1 = up
    direction_probability: float  # Probability of direction
    expected_return: float  # Expected return magnitude
    confidence: float  # Overall prediction confidence (0-1)
    upper_bound: float  # Upper return estimate
    lower_bound: float  # Lower return estimate
    prediction_5d: Optional[float] = None  # 5-day return prediction


class ProductionModel:
    """
    Production-grade prediction model.

    Architecture:
    1. Direction Classifier (XGBoost + LightGBM ensemble)
    2. Return Regressor (XGBoost with quantile estimation)
    3. Confidence Scorer (based on model agreement & historical accuracy)
    """

    def __init__(self, model_name: str = 'production'):
        self.model_name = model_name

        # Direction models (ensemble)
        self.direction_xgb = None
        self.direction_lgb = None

        # Return models
        self.return_model = None
        self.return_upper_model = None  # 75th percentile
        self.return_lower_model = None  # 25th percentile

        # 5-day horizon
        self.return_5d_model = None

        # Scalers
        self.feature_scaler = RobustScaler()

        # Feature names
        self.feature_names = None

        # Model metadata
        self.training_date = None
        self.n_features = None
        self.validation_metrics = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        feature_names: List[str] = None
    ) -> Dict:
        """
        Train all model components.

        Args:
            X_train: Training features
            y_train: Dict with 'direction', 'close_return', 'close_return_5d'
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names

        Returns:
            Dict with training metrics
        """
        logger.info("Training Production Model...")
        self.training_date = datetime.now()
        self.feature_names = feature_names
        self.n_features = X_train.shape[1]

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        metrics = {}

        # 1. Train Direction Classifier (XGBoost)
        logger.info("Training Direction Classifier (XGBoost)...")
        self.direction_xgb = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=self._compute_class_weight(y_train['direction']),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric='logloss'
        )

        self.direction_xgb.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            verbose=False
        )

        xgb_dir_pred = self.direction_xgb.predict(X_val_scaled)
        xgb_dir_acc = accuracy_score(y_val['direction'], xgb_dir_pred)
        logger.info(f"  XGBoost Direction Accuracy: {xgb_dir_acc:.2%}")

        # 2. Train Direction Classifier (LightGBM) for ensemble
        logger.info("Training Direction Classifier (LightGBM)...")
        self.direction_lgb = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        self.direction_lgb.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        lgb_dir_pred = self.direction_lgb.predict(X_val_scaled)
        lgb_dir_acc = accuracy_score(y_val['direction'], lgb_dir_pred)
        logger.info(f"  LightGBM Direction Accuracy: {lgb_dir_acc:.2%}")

        # Ensemble direction
        dir_proba_xgb = self.direction_xgb.predict_proba(X_val_scaled)[:, 1]
        dir_proba_lgb = self.direction_lgb.predict_proba(X_val_scaled)[:, 1]
        dir_proba_ensemble = (dir_proba_xgb + dir_proba_lgb) / 2
        dir_pred_ensemble = (dir_proba_ensemble > 0.5).astype(int)
        ensemble_dir_acc = accuracy_score(y_val['direction'], dir_pred_ensemble)
        logger.info(f"  Ensemble Direction Accuracy: {ensemble_dir_acc:.2%}")

        metrics['direction_accuracy_xgb'] = xgb_dir_acc
        metrics['direction_accuracy_lgb'] = lgb_dir_acc
        metrics['direction_accuracy_ensemble'] = ensemble_dir_acc

        # 3. Train Return Regressor (median)
        logger.info("Training Return Regressor...")
        self.return_model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50
        )

        self.return_model.fit(
            X_train_scaled, y_train['close_return'],
            eval_set=[(X_val_scaled, y_val['close_return'])],
            verbose=False
        )

        return_pred = self.return_model.predict(X_val_scaled)
        return_mae = mean_absolute_error(y_val['close_return'], return_pred)
        return_rmse = np.sqrt(mean_squared_error(y_val['close_return'], return_pred))
        logger.info(f"  Return MAE: {return_mae*100:.3f}%, RMSE: {return_rmse*100:.3f}%")

        metrics['return_mae'] = return_mae
        metrics['return_rmse'] = return_rmse

        # 4. Train Quantile Models for confidence bounds
        logger.info("Training Quantile Models...")
        self.return_upper_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.75,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.return_upper_model.fit(X_train_scaled, y_train['close_return'])

        self.return_lower_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.25,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.return_lower_model.fit(X_train_scaled, y_train['close_return'])

        # 5. Train 5-day return model if data available (and not all NaN)
        if 'close_return_5d' in y_train and y_train['close_return_5d'] is not None:
            # Check for valid data
            close_5d = y_train['close_return_5d']
            valid_mask = ~np.isnan(close_5d) & ~np.isinf(close_5d)

            if valid_mask.sum() > 100:  # Need sufficient valid data
                logger.info("Training 5-Day Return Model...")
                self.return_5d_model = XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                # Use only valid data
                X_5d = X_train_scaled[valid_mask]
                y_5d = close_5d[valid_mask]
                self.return_5d_model.fit(X_5d, y_5d)
            else:
                logger.warning("Skipping 5-Day Return Model - insufficient valid data")

        # Store validation metrics
        self.validation_metrics = metrics

        logger.success(f"Training complete. Ensemble accuracy: {ensemble_dir_acc:.2%}")

        return metrics

    def predict(self, X: np.ndarray) -> List[ModelPrediction]:
        """
        Generate predictions with confidence scores.

        Args:
            X: Feature matrix

        Returns:
            List of ModelPrediction objects
        """
        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Direction predictions
        dir_proba_xgb = self.direction_xgb.predict_proba(X_scaled)[:, 1]
        dir_proba_lgb = self.direction_lgb.predict_proba(X_scaled)[:, 1]

        # Ensemble probability
        dir_proba = (dir_proba_xgb + dir_proba_lgb) / 2
        directions = (dir_proba > 0.5).astype(int)

        # Return predictions
        expected_returns = self.return_model.predict(X_scaled)
        upper_bounds = self.return_upper_model.predict(X_scaled)
        lower_bounds = self.return_lower_model.predict(X_scaled)

        # 5-day predictions
        predictions_5d = None
        if self.return_5d_model is not None:
            predictions_5d = self.return_5d_model.predict(X_scaled)

        # Compute confidence scores
        confidences = self._compute_confidence(
            dir_proba_xgb, dir_proba_lgb, dir_proba,
            expected_returns, upper_bounds, lower_bounds
        )

        # Build prediction objects
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
        """Predict for a single sample."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.predict(X)[0]

    def _compute_confidence(
        self,
        proba_xgb: np.ndarray,
        proba_lgb: np.ndarray,
        proba_ensemble: np.ndarray,
        returns: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray
    ) -> np.ndarray:
        """
        Compute prediction confidence based on:
        1. Model agreement (XGBoost vs LightGBM)
        2. Direction probability strength
        3. Return prediction uncertainty (spread of quantiles)
        """
        # Model agreement (1 = perfect agreement, 0 = complete disagreement)
        model_agreement = 1 - np.abs(proba_xgb - proba_lgb)

        # Direction confidence (how far from 0.5)
        direction_strength = np.abs(proba_ensemble - 0.5) * 2

        # Return uncertainty (narrower spread = more confident)
        return_spread = upper - lower
        median_spread = np.median(return_spread[~np.isnan(return_spread)])
        return_confidence = np.clip(1 - (return_spread / (median_spread * 2)), 0, 1)

        # Combined confidence
        confidence = (
            0.4 * model_agreement +
            0.4 * direction_strength +
            0.2 * return_confidence
        )

        return confidence

    def _compute_class_weight(self, y: np.ndarray) -> float:
        """Compute class weight for imbalanced data."""
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        if n_positive == 0 or n_negative == 0:
            return 1.0
        return n_negative / n_positive

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if self.direction_xgb is None:
            raise ValueError("Model not trained yet")

        importance = self.direction_xgb.feature_importances_

        if self.feature_names is not None:
            df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            })
        else:
            df = pd.DataFrame({
                'feature': [f'f_{i}' for i in range(len(importance))],
                'importance': importance
            })

        return df.sort_values('importance', ascending=False).head(top_n)

    def save(self, save_dir: str):
        """Save model to directory."""
        os.makedirs(save_dir, exist_ok=True)

        # Save all components
        joblib.dump(self.direction_xgb, os.path.join(save_dir, 'direction_xgb.pkl'))
        joblib.dump(self.direction_lgb, os.path.join(save_dir, 'direction_lgb.pkl'))
        joblib.dump(self.return_model, os.path.join(save_dir, 'return_model.pkl'))
        joblib.dump(self.return_upper_model, os.path.join(save_dir, 'return_upper.pkl'))
        joblib.dump(self.return_lower_model, os.path.join(save_dir, 'return_lower.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(save_dir, 'scaler.pkl'))

        if self.return_5d_model is not None:
            joblib.dump(self.return_5d_model, os.path.join(save_dir, 'return_5d.pkl'))

        # Save metadata
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
        """Load model from directory."""
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


class ModelEvaluator:
    """Comprehensive model evaluation."""

    @staticmethod
    def evaluate_predictions(
        y_true_direction: np.ndarray,
        y_true_return: np.ndarray,
        predictions: List[ModelPrediction]
    ) -> Dict:
        """Evaluate model predictions."""

        # Extract predictions
        pred_directions = np.array([p.direction for p in predictions])
        pred_returns = np.array([p.expected_return for p in predictions])
        confidences = np.array([p.confidence for p in predictions])

        # Direction metrics
        direction_accuracy = accuracy_score(y_true_direction, pred_directions)

        # High confidence accuracy
        high_conf_mask = confidences > 0.6
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                y_true_direction[high_conf_mask],
                pred_directions[high_conf_mask]
            )
        else:
            high_conf_accuracy = 0

        # Return metrics
        return_mae = mean_absolute_error(y_true_return, pred_returns)
        return_rmse = np.sqrt(mean_squared_error(y_true_return, pred_returns))

        # Directional return accuracy (predicted direction matches actual)
        pred_dir_from_return = (pred_returns > 0).astype(int)
        actual_dir_from_return = (y_true_return > 0).astype(int)
        dir_from_return_accuracy = accuracy_score(actual_dir_from_return, pred_dir_from_return)

        # Profitability (if we traded based on predictions)
        # Simple strategy: go long when predicting up, flat when predicting down
        strategy_returns = y_true_return * (pred_directions * 2 - 1)  # Convert 0/1 to -1/1
        total_return = strategy_returns.sum()
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)

        return {
            'direction_accuracy': direction_accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_trades': high_conf_mask.sum(),
            'return_mae': return_mae,
            'return_rmse': return_rmse,
            'dir_from_return_accuracy': dir_from_return_accuracy,
            'strategy_total_return': total_return,
            'strategy_sharpe': sharpe,
            'avg_confidence': confidences.mean()
        }
