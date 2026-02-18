"""
================================================================================
BASE MODEL - Abstract Interface for All Models
================================================================================
All models (XGBoost, LSTM, TCN, etc.) inherit from this base class.

This ensures:
1. Consistent interface: train() and predict()
2. Standard evaluation metrics
3. Easy model swapping
4. Unified pipeline integration

Usage:
    class MyNewModel(BaseModel):
        def train(self, X_train, y_train, X_val, y_val):
            # Your training code
            pass

        def predict(self, X):
            # Your prediction code
            pass
================================================================================
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
from dataclasses import dataclass
import joblib
from pathlib import Path
from loguru import logger


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    logloss: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'logloss': self.logloss,
            'confusion_matrix': self.confusion_matrix.tolist()
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        return f"""
{'='*45}
        MODEL PERFORMANCE
{'='*45}
Accuracy:  {self.accuracy:6.2%}
Precision: {self.precision:6.2%}
Recall:    {self.recall:6.2%}
F1 Score:  {self.f1_score:6.4f}
AUC:       {self.auc:6.4f}
Log Loss:  {self.logloss:6.4f}
{'='*45}
"""


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    All models must implement:
    - train(X_train, y_train, X_val, y_val)
    - predict(X) → predictions
    - predict_proba(X) → probabilities (for classification)
    """

    def __init__(self, model_name: str):
        """
        Initialize base model.

        Args:
            model_name: Name of the model (e.g., "XGBoost", "LSTM", "TCN")
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary with training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features to predict on

        Returns:
            Binary predictions (0 or 1)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Probability of positive class (0.0 to 1.0)

        Note:
            Some models may not support probabilities.
            Override this method if your model does.
        """
        raise NotImplementedError(
            f"{self.model_name} does not support predict_proba()"
        )

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        return_predictions: bool = False
    ) -> Union[ModelMetrics, Tuple[ModelMetrics, np.ndarray, np.ndarray]]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y_true: True labels
            return_predictions: Whether to return predictions

        Returns:
            ModelMetrics object (and optionally predictions & probabilities)
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet!")

        # Get predictions
        y_pred = self.predict(X)

        # Try to get probabilities
        try:
            y_proba = self.predict_proba(X)
            # Ensure y_proba is 1D (probability of positive class)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
        except NotImplementedError:
            # Model doesn't support probabilities
            y_proba = None

        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            auc=roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0,
            logloss=log_loss(y_true, y_proba) if y_proba is not None else 0.0,
            confusion_matrix=confusion_matrix(y_true, y_pred)
        )

        if return_predictions:
            return metrics, y_pred, y_proba
        else:
            return metrics

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_name': self.model_name,
            'model': self.model,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"✅ {self.model_name} saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        save_dict = joblib.load(filepath)

        self.model_name = save_dict['model_name']
        self.model = save_dict['model']
        self.is_trained = save_dict['is_trained']
        self.training_history = save_dict['training_history']
        self.feature_names = save_dict.get('feature_names', None)

        logger.info(f"✅ {self.model_name} loaded from {filepath}")

    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance (if supported).

        Returns:
            Dictionary mapping feature names to importance scores

        Note:
            Override this method if your model supports feature importance
        """
        return None

    def __repr__(self) -> str:
        """String representation."""
        status = "✅ Trained" if self.is_trained else "❌ Not Trained"
        return f"{self.model_name} ({status})"


class BaseMLModel(BaseModel):
    """
    Base class for traditional ML models (XGBoost, LightGBM, etc.).

    Provides common functionality:
    - Feature scaling
    - Feature importance
    - Early stopping
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.scaler = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet!")

        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)

        # Return probability of positive class
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def get_feature_importance(self) -> Optional[Dict]:
        """Get feature importance from tree-based model."""
        if not self.is_trained:
            return None

        if not hasattr(self.model, 'feature_importances_'):
            return None

        if self.feature_names is None:
            return None

        importance_scores = self.model.feature_importances_

        # Create dictionary mapping feature names to scores
        importance_dict = dict(zip(self.feature_names, importance_scores))

        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict


class BaseDeepLearningModel(BaseModel):
    """
    Base class for deep learning models (LSTM, Transformer, etc.).

    Provides common functionality:
    - Sequence creation
    - Data scaling
    - PyTorch device management
    - Training loop with early stopping
    """

    def __init__(self, model_name: str, seq_length: int = 30):
        super().__init__(model_name)
        self.seq_length = seq_length
        self.scaler = None
        self.device = 'cpu'  # Override to 'cuda' if GPU available

    def create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Create sequences for time series modeling.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (optional)

        Returns:
            X_seq: (n_samples - seq_length, seq_length, n_features)
            y_seq: (n_samples - seq_length,) if y provided
        """
        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            if y is not None:
                y_seq.append(y[i + self.seq_length])

        X_seq = np.array(X_seq)

        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compare_models(
    models: Dict[str, BaseModel],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on same test set.

    Args:
        models: Dictionary of {model_name: model_instance}
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, model in models.items():
        try:
            metrics = model.evaluate(X_test, y_test)

            results.append({
                'Model': name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1': metrics.f1_score,
                'AUC': metrics.auc,
                'LogLoss': metrics.logloss
            })
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)

    return df


if __name__ == '__main__':
    # Example usage
    print("Base Model Classes:")
    print("  - BaseModel: Abstract base for all models")
    print("  - BaseMLModel: For XGBoost, LightGBM, etc.")
    print("  - BaseDeepLearningModel: For LSTM, TCN, Transformer, etc.")
    print("\nAll models will inherit from these classes.")
