"""
================================================================================
NEURAL MODEL INTEGRATION FOR V3 PIPELINE
================================================================================
Extends pipeline.py with neural network model training and ensemble creation.

Provides:
- train_neural_models(): Train all neural architectures on same data as traditional models
- create_super_ensemble(): Combine traditional + neural models into unified ensemble
- Seamless integration with existing pipeline results format

Models trained:
- LSTM: Recurrent neural network for temporal sequences
- BiLSTM: Bidirectional LSTM for enhanced context
- GRU: Gated Recurrent Unit (simpler, faster LSTM)
- CNN-LSTM: CNN for feature extraction + LSTM for temporal modeling
- TCN: Temporal Convolutional Network with dilated convolutions
- Transformer: Multi-head attention-based architecture

Integration strategy:
1. Train neural models on same train/val/test split as XGB/LGB
2. Generate predictions for validation set
3. Optimize ensemble weights combining traditional + neural models
4. Evaluate ensemble accuracy and metrics
================================================================================
"""

import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize

from loguru import logger

# Try importing neural models with fallback
try:
    from neural_models import (
        LSTMModel, BiLSTMModel, GRUModel,
        CNNLSTMModel, TCNModel, TransformerModel,
        create_model
    )
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False
    logger.warning("Neural models not available - using traditional models only")

warnings.filterwarnings('ignore')


# ============================================================================
# NEURAL MODEL TRAINING
# ============================================================================

def train_neural_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    seq_length: int = 30,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = 'cpu',
) -> Dict:
    """
    Train all neural network models on the provided data.
    
    Args:
        X_train, y_train: Training data (n_samples, n_features)
        X_test, y_test: Test data for early stopping
        X_val, y_val: Validation data for evaluation
        feature_names: List of feature names
        seq_length: Sequence length for RNNs/CNNs/TCN/Transformer
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict with predictions for test and val sets from each model
    """
    
    if not NEURAL_MODELS_AVAILABLE:
        logger.warning("Neural models not available, returning empty dict")
        return {}
    
    input_size = X_train.shape[1]
    results = {}
    
    # Define neural models to train
    models_config = [
        ('lstm', {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}),
        ('bilstm', {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}),
        ('gru', {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}),
        ('cnn_lstm', {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'num_filters': 32, 'kernel_size': 3}),
        ('tcn', {'hidden_size': 128, 'num_layers': 4, 'dropout': 0.2, 'num_filters': 32, 'kernel_size': 3}),
        ('transformer', {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'n_heads': 4}),
    ]
    
    for model_name, hyper_params in models_config:
        try:
            logger.info(f"  Training neural model: {model_name}...")
            
            # Create model
            model = create_model(
                model_name,
                input_size=input_size,
                learning_rate=0.001,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                **hyper_params
            )
            
            # Prepare data (create sequences)
            X_train_t, y_train_t, X_val_t, y_val_t = model.prepare_data(
                X_train, y_train, X_val, y_val, seq_length=seq_length
            )
            
            # Handle case where sequencing reduces data size
            if X_train_t.shape[0] < 10:
                logger.warning(f"  {model_name}: insufficient data after sequencing ({X_train_t.shape[0]} samples)")
                continue
            
            # Train model
            model.train_model(X_train_t, y_train_t, X_val_t, y_val_t)
            
            # Generate predictions on val set
            val_predictions = model.predict(X_val_t).flatten()
            
            results[model_name] = {
                'model': model,
                'pred_val': val_predictions,
                'train_losses': model.train_losses,
                'val_losses': model.val_losses,
            }
            
            logger.info(f"  {model_name}: training complete, final val loss={model.val_losses[-1]:.6f}")
            
        except Exception as e:
            logger.error(f"  {model_name} training failed: {e}")
            continue
    
    return results


# ============================================================================
# ENSEMBLE CREATION WITH NEURAL + TRADITIONAL MODELS
# ============================================================================

def create_super_ensemble(
    xgb_pred_val: np.ndarray,
    lgb_pred_val: np.ndarray,
    neural_predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
) -> Dict:
    """
    Create weighted ensemble combining traditional + neural models.
    
    Optimizes weights to minimize MSE on validation set.
    Constraint: weights sum to 1.0, each weight in [0, 1]
    
    Args:
        xgb_pred_val: XGBoost predictions
        lgb_pred_val: LightGBM predictions
        neural_predictions: Dict[model_name] -> predictions array
        y_val: Validation targets
    
    Returns:
        Dict with optimized weights and ensemble predictions
    """
    
    # Collect all predictions
    all_preds = [xgb_pred_val, lgb_pred_val]
    model_names = ['xgb', 'lgb']
    
    for name, pred in neural_predictions.items():
        # Handle sequence length mismatch: use last n predictions where n = min length
        min_len = min(len(pred), len(y_val))
        all_preds.append(pred[-min_len:])
        model_names.append(name)
    
    # Make sure all predictions have same length as y_val
    min_len = min(len(y_val), min(len(p) for p in all_preds))
    all_preds = [p[-min_len:] for p in all_preds]
    y_val_aligned = y_val[-min_len:]
    
    n_models = len(all_preds)
    
    # Objective: MSE of weighted ensemble
    def ensemble_loss(weights):
        ensemble_pred = np.zeros_like(y_val_aligned)
        for i, pred in enumerate(all_preds):
            ensemble_pred += weights[i] * pred
        return np.mean((y_val_aligned - ensemble_pred) ** 2)
    
    # Optimize weights
    result = minimize(
        ensemble_loss,
        x0=np.ones(n_models) / n_models,  # Equal initial weight
        method='SLSQP',
        bounds=[(0, 1) for _ in range(n_models)],
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
    )
    
    if not result.success:
        logger.warning(f"Ensemble optimization failed: {result.message}, using equal weights")
        weights = np.ones(n_models) / n_models
    else:
        weights = result.x
    
    # Generate ensemble predictions
    ensemble_pred = np.zeros_like(y_val_aligned)
    for i, pred in enumerate(all_preds):
        ensemble_pred += weights[i] * pred
    
    # Log results
    weights_dict = {name: round(float(w), 4) for name, w in zip(model_names, weights)}
    logger.info(f"  Super ensemble weights: {weights_dict}")
    
    return {
        'ensemble_pred': ensemble_pred,
        'weights': weights_dict,
        'model_names': model_names,
        'all_predictions': {name: pred for name, pred in zip(model_names, all_preds)},
    }


# ============================================================================
# EXTENDED METRICS COMPUTATION
# ============================================================================

def compute_neural_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute directional accuracy and win rates from predictions.
    (Simplified version that works with any model's predictions)
    """
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    
    actual_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)
    
    # Directional accuracy
    dir_accuracy = accuracy_score(actual_dir, pred_dir)
    
    # Win rate (long)
    long_mask = pred_dir == 1
    win_rate_long = float(np.mean(actual_dir[long_mask])) if long_mask.sum() > 0 else 0.0
    
    # Win rate (short)
    short_mask = pred_dir == 0
    win_rate_short = float(np.mean(1 - actual_dir[short_mask])) if short_mask.sum() > 0 else 0.0
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'dir_accuracy': round(dir_accuracy, 4),
        'win_rate_long': round(win_rate_long, 4),
        'win_rate_short': round(win_rate_short, 4),
        'rmse': round(rmse, 6),
        'mae': round(mae, 6),
    }


# ============================================================================
# PIPELINE WRAPPER
# ============================================================================

def train_hybrid_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    xgb_pred_val: np.ndarray = None,
    lgb_pred_val: np.ndarray = None,
    **kwargs
) -> Dict:
    """
    Unified training function that trains traditional (XGB/LGB) + neural models.
    
    This is a wrapper that combines results from:
    1. Traditional models (external training)
    2. Neural models (this module)
    3. Creates super ensemble
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test set (for early stopping)
        X_val, y_val: Validation set
        feature_names: List of feature names
        xgb_pred_val: Pre-computed XGBoost predictions on val set
        lgb_pred_val: Pre-computed LightGBM predictions on val set
        **kwargs: Additional args (seq_length, batch_size, epochs, device, etc.)
    
    Returns:
        Dict with all model predictions and ensemble weights
    """
    
    # Train neural models
    neural_results = train_neural_models(
        X_train, y_train, X_test, y_test, X_val, y_val,
        feature_names,
        seq_length=kwargs.get('seq_length', 30),
        batch_size=kwargs.get('batch_size', 32),
        epochs=kwargs.get('epochs', 100),
        device=kwargs.get('device', 'cpu'),
    )
    
    # Extract neural predictions
    neural_preds = {name: res['pred_val'] for name, res in neural_results.items()}
    
    # If traditional model predictions provided, create super ensemble
    if xgb_pred_val is not None and lgb_pred_val is not None:
        ensemble_result = create_super_ensemble(
            xgb_pred_val, lgb_pred_val, neural_preds, y_val
        )
    else:
        # Just combine neural models
        if neural_preds:
            ensemble_result = create_super_ensemble(
                np.zeros_like(y_val), np.zeros_like(y_val), neural_preds, y_val
            )
        else:
            ensemble_result = None
    
    return {
        'neural_models': neural_results,
        'neural_predictions': neural_preds,
        'ensemble_result': ensemble_result,
    }


if __name__ == '__main__':
    logger.info("Neural model integration module loaded successfully")
    if NEURAL_MODELS_AVAILABLE:
        logger.info("Neural models available and ready for use")
    else:
        logger.warning("Neural models not available - install required dependencies")
