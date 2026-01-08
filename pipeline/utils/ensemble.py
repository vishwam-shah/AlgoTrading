"""
================================================================================
ENSEMBLE MODEL - STACKING META-LEARNER
================================================================================
Combines XGBoost and LSTM predictions using a meta-learner (stacking).

Architecture:
    Level 0 (Base Models):
        - XGBoost: Gradient boosting on raw features
        - LSTM: Sequential model on time series
    
    Level 1 (Meta-Learner):
        - Linear Regression: Weighted combination of base predictions
        - OR XGBoost: Non-linear combination
        
Features passed to meta-learner:
    - XGBoost prediction
    - LSTM prediction
    - Prediction variance/uncertainty (if available)
    - Latest market features (optional)

Usage:
    from pipeline.utils.ensemble import train_ensemble, predict_ensemble
    
    # Train ensemble
    ensemble_model = train_ensemble(symbol, X_train, y_train, X_val, y_val)
    
    # Predict
    predictions = predict_ensemble(ensemble_model, X_test)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class StackingEnsemble:
    """
    Stacking ensemble that combines multiple model predictions.
    """
    
    def __init__(self, meta_learner_type='ridge', alpha=1.0):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner_type: 'ridge', 'linear', or 'xgboost'
            alpha: Regularization strength for Ridge
        """
        self.meta_learner_type = meta_learner_type
        self.alpha = alpha
        self.meta_learner = None
        self.base_weights = None
        
    def train(self, base_predictions: dict, y_true: np.ndarray):
        """
        Train meta-learner on base model predictions.
        
        Args:
            base_predictions: Dict with keys 'xgboost', 'lstm' -> predictions array
            y_true: True target values
        """
        # Stack predictions as features
        X_meta = self._stack_predictions(base_predictions)
        
        # Train meta-learner
        if self.meta_learner_type == 'ridge':
            self.meta_learner = Ridge(alpha=self.alpha, fit_intercept=True)
        elif self.meta_learner_type == 'linear':
            self.meta_learner = LinearRegression(fit_intercept=True)
        elif self.meta_learner_type == 'xgboost':
            import xgboost as xgb
            self.meta_learner = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")
        
        self.meta_learner.fit(X_meta, y_true)
        
        # Store base model weights (for linear models)
        if hasattr(self.meta_learner, 'coef_'):
            self.base_weights = {
                'xgboost': self.meta_learner.coef_[0] if len(self.meta_learner.coef_) > 0 else 0,
                'lstm': self.meta_learner.coef_[1] if len(self.meta_learner.coef_) > 1 else 0,
            }
            # Normalize to sum to 1
            total = sum(abs(w) for w in self.base_weights.values())
            if total > 0:
                self.base_weights = {k: v/total for k, v in self.base_weights.items()}
        
        logger.info(f"Ensemble trained with {self.meta_learner_type} meta-learner")
        if self.base_weights:
            logger.info(f"Base weights: XGBoost={self.base_weights['xgboost']:.3f}, "
                       f"LSTM={self.base_weights['lstm']:.3f}")
    
    def predict(self, base_predictions: dict) -> np.ndarray:
        """
        Predict using ensemble.
        
        Args:
            base_predictions: Dict with keys 'xgboost', 'lstm' -> predictions array
            
        Returns:
            Ensemble predictions
        """
        X_meta = self._stack_predictions(base_predictions)
        return self.meta_learner.predict(X_meta)
    
    def _stack_predictions(self, base_predictions: dict) -> np.ndarray:
        """Stack base predictions as features for meta-learner."""
        predictions_list = []
        
        # Ensure consistent order
        for model_name in ['xgboost', 'lstm']:
            if model_name in base_predictions:
                predictions_list.append(base_predictions[model_name].reshape(-1, 1))
        
        if len(predictions_list) == 0:
            raise ValueError("No base predictions provided")
        
        return np.hstack(predictions_list)
    
    def save(self, path: str):
        """Save ensemble model."""
        joblib.dump({
            'meta_learner': self.meta_learner,
            'meta_learner_type': self.meta_learner_type,
            'base_weights': self.base_weights,
            'alpha': self.alpha
        }, path)
        logger.info(f"Ensemble saved to {path}")
    
    @staticmethod
    def load(path: str):
        """Load ensemble model."""
        data = joblib.load(path)
        ensemble = StackingEnsemble(
            meta_learner_type=data['meta_learner_type'],
            alpha=data.get('alpha', 1.0)
        )
        ensemble.meta_learner = data['meta_learner']
        ensemble.base_weights = data['base_weights']
        logger.info(f"Ensemble loaded from {path}")
        return ensemble


def train_ensemble_model(symbol: str, 
                        train_predictions: dict,
                        val_predictions: dict,
                        y_train: np.ndarray,
                        y_val: np.ndarray,
                        meta_learner_type: str = 'ridge') -> tuple:
    """
    Train ensemble model and return it with validation predictions.
    
    Args:
        symbol: Stock symbol
        train_predictions: {'xgboost': pred_train, 'lstm': pred_train}
        val_predictions: {'xgboost': pred_val, 'lstm': pred_val}
        y_train: Training targets
        y_val: Validation targets
        meta_learner_type: Type of meta-learner
        
    Returns:
        (ensemble_model, ensemble_path, val_ensemble_predictions)
    """
    ensemble = StackingEnsemble(meta_learner_type=meta_learner_type)
    
    # Train on validation set (meta-learner uses out-of-fold predictions)
    ensemble.train(val_predictions, y_val)
    
    # Get ensemble predictions on validation set
    val_pred_ensemble = ensemble.predict(val_predictions)
    
    # Save ensemble
    ensemble_path = os.path.join(config.MODEL_DIR, 'ensemble', f"{symbol}_ensemble.pkl")
    os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
    ensemble.save(ensemble_path)
    
    return ensemble, ensemble_path, val_pred_ensemble


def predict_with_ensemble(ensemble: StackingEnsemble, base_predictions: dict) -> np.ndarray:
    """
    Make predictions with trained ensemble.
    
    Args:
        ensemble: Trained StackingEnsemble
        base_predictions: {'xgboost': predictions, 'lstm': predictions}
        
    Returns:
        Ensemble predictions
    """
    return ensemble.predict(base_predictions)


def simple_average_ensemble(base_predictions: dict, weights: dict = None) -> np.ndarray:
    """
    Simple weighted average ensemble (no training required).
    
    Args:
        base_predictions: {'xgboost': pred, 'lstm': pred}
        weights: {'xgboost': w1, 'lstm': w2} or None for equal weights
        
    Returns:
        Averaged predictions
    """
    if weights is None:
        weights = {k: 1.0/len(base_predictions) for k in base_predictions.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    ensemble_pred = np.zeros_like(list(base_predictions.values())[0])
    for model_name, pred in base_predictions.items():
        weight = weights.get(model_name, 0)
        ensemble_pred += weight * pred
    
    return ensemble_pred


if __name__ == '__main__':
    # Test ensemble
    np.random.seed(42)
    
    # Simulate predictions
    y_true = np.random.randn(100) * 0.02
    xgb_pred = y_true + np.random.randn(100) * 0.01
    lstm_pred = y_true + np.random.randn(100) * 0.015
    
    base_preds = {'xgboost': xgb_pred, 'lstm': lstm_pred}
    
    # Train ensemble
    ensemble = StackingEnsemble(meta_learner_type='ridge')
    ensemble.train(base_preds, y_true)
    
    # Predict
    ensemble_pred = ensemble.predict(base_preds)
    
    # Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_true, xgb_pred)))
    print("LSTM RMSE:", np.sqrt(mean_squared_error(y_true, lstm_pred)))
    print("Ensemble RMSE:", np.sqrt(mean_squared_error(y_true, ensemble_pred)))
    print("\nEnsemble R²:", r2_score(y_true, ensemble_pred))
