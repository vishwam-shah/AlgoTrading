"""
================================================================================
WALK-FORWARD EXPANDING WINDOW VALIDATION
================================================================================
Implements rolling window training that expands after each round.

Key Features:
- Expanding window: Training set grows with each iteration
- Fixed test window: Consistent evaluation period
- Realistic backtesting: No look-ahead bias
- Performance tracking: Metrics across all windows

Example:
    Round 1: Train[0:500]   Test[500:550]
    Round 2: Train[0:550]   Test[550:600]  <- Training expands
    Round 3: Train[0:600]   Test[600:650]  <- Training expands again
    ...

This mimics real trading: each day you retrain with all historical data.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import time
from datetime import datetime
import io
import contextlib

# Add paths
v3_path = Path(__file__).parent.parent
sys.path.insert(0, str(v3_path))
sys.path.insert(0, str(v3_path / '02_models'))

from traditional.xgboost_classifier import XGBoostClassifier
from traditional.lightgbm_classifier import LightGBMClassifier
from traditional.catboost_classifier import CatBoostClassifier
from deep_learning.lstm_classifier import LSTMClassifier
from deep_learning.tcn_classifier import TCNClassifier


class WalkForwardValidator:
    """
    Walk-forward expanding window validation for time series models.

    Training window expands, test window slides forward.
    """

    def __init__(
        self,
        initial_train_size: int = 800,
        test_window_size: int = 100,
        step_size: int = 100,
        min_train_samples: int = 500
    ):
        """
        Initialize walk-forward validator.

        Args:
            initial_train_size: Initial training samples
            test_window_size: Size of each test window
            step_size: Number of samples to move forward each round
            min_train_samples: Minimum samples needed for training
        """
        self.initial_train_size = initial_train_size
        self.test_window_size = test_window_size
        self.step_size = step_size
        self.min_train_samples = min_train_samples

        self.results = []
        self.window_metrics = []

    def create_expanding_windows(
        self,
        n_samples: int
    ) -> List[Tuple[slice, slice]]:
        """
        Create expanding training windows with sliding test windows.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_slice, test_slice) tuples
        """
        windows = []

        train_end = self.initial_train_size

        while train_end + self.test_window_size <= n_samples:
            # Train slice: From start to current position (expanding)
            train_slice = slice(0, train_end)

            # Test slice: Fixed window after training
            test_start = train_end
            test_end = train_end + self.test_window_size
            test_slice = slice(test_start, test_end)

            windows.append((train_slice, test_slice))

            # Move forward by step_size (training expands)
            train_end += self.step_size

        return windows

    def validate_model(
        self,
        model_class,
        model_params: Dict,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_names: List[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward validation for a single model.

        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            model_name: Name of the model
            feature_names: List of feature names
            verbose: Whether to print progress

        Returns:
            Dictionary with validation results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" WALK-FORWARD VALIDATION: {model_name}")
            print(f"{'='*70}")

        # Create windows
        windows = self.create_expanding_windows(len(X))

        if verbose:
            print(f"\n[INFO] Created {len(windows)} expanding windows")
            print(f"  Initial train size: {self.initial_train_size}")
            print(f"  Test window size: {self.test_window_size}")
            print(f"  Step size: {self.step_size}")

        window_results = []
        all_predictions = []
        all_true_labels = []
        all_train_times = []

        for window_idx, (train_slice, test_slice) in enumerate(windows, 1):
            if verbose:
                print(f"\n{'-'*70}")
                print(f" Window {window_idx}/{len(windows)}")
                print(f"{'-'*70}")
                print(f"  Train: samples 0 to {train_slice.stop} ({train_slice.stop} total)")
                print(f"  Test:  samples {test_slice.start} to {test_slice.stop} ({test_slice.stop - test_slice.start} total)")

            # Extract data for this window
            X_train = X[train_slice]
            y_train = y[train_slice]
            X_test = X[test_slice]
            y_test = y[test_slice]

            # Create validation set (last 15% of training)
            val_size = int(len(X_train) * 0.15)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_only = X_train[:-val_size]
            y_train_only = y_train[:-val_size]

            # Initialize new model for this window
            model = model_class(**model_params)

            # Train (suppress all output to avoid Unicode errors)
            start_time = time.time()
            try:
                # Redirect stdout and stderr to suppress Unicode characters
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        model.train(
                            X_train_only, y_train_only,
                            X_val, y_val,
                            feature_names=feature_names,
                            verbose=False
                        )
                train_time = time.time() - start_time
            except Exception as e:
                print(f"  [ERROR] Training failed: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                continue

            # Adjust test labels for sequence models
            if hasattr(model, 'seq_length'):
                y_test_eval = y_test[model.seq_length:]
            else:
                y_test_eval = y_test

            # Predict
            try:
                predictions = model.predict(X_test)

                # Calculate metrics manually to avoid Unicode issues
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, roc_auc_score
                )

                accuracy = accuracy_score(y_test_eval, predictions)
                precision = precision_score(y_test_eval, predictions, zero_division=0)
                recall = recall_score(y_test_eval, predictions, zero_division=0)
                f1 = f1_score(y_test_eval, predictions, zero_division=0)

                try:
                    # Get probabilities for AUC
                    probas = model.predict_proba(X_test)
                    auc = roc_auc_score(y_test_eval, probas)
                except:
                    auc = 0.5  # Default if AUC calculation fails

                window_results.append({
                    'window': window_idx,
                    'train_size': train_slice.stop,
                    'test_size': len(y_test_eval),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'train_time': train_time
                })

                # Store predictions
                all_predictions.extend(predictions)
                all_true_labels.extend(y_test_eval)
                all_train_times.append(train_time)

                if verbose:
                    print(f"  Accuracy: {accuracy:.2%}")
                    print(f"  F1 Score: {f1:.4f}")
                    print(f"  Train Time: {train_time:.1f}s")

            except Exception as e:
                import traceback
                print(f"  [ERROR] Evaluation failed: {e}")
                if verbose:
                    traceback.print_exc()
                continue

        # Calculate overall metrics
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        overall_precision = precision_score(all_true_labels, all_predictions, zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predictions, zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predictions, zero_division=0)

        avg_train_time = np.mean(all_train_times)

        results = {
            'model': model_name,
            'n_windows': len(windows),
            'n_completed': len(window_results),
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'avg_train_time': avg_train_time,
            'window_results': window_results,
            'predictions': all_predictions,
            'true_labels': all_true_labels
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f" OVERALL RESULTS: {model_name}")
            print(f"{'='*70}")
            print(f"  Windows completed: {len(window_results)}/{len(windows)}")
            print(f"  Overall Accuracy: {overall_accuracy:.2%}")
            print(f"  Overall F1 Score: {overall_f1:.4f}")
            print(f"  Avg Train Time: {avg_train_time:.1f}s")

        return results

    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        models_to_test: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models using walk-forward validation.

        Args:
            X: Features
            y: Labels
            feature_names: Feature names
            models_to_test: List of model names to test (default: all)
            verbose: Print progress

        Returns:
            DataFrame with comparison results
        """
        if verbose:
            print("\n" + "="*70)
            print(" WALK-FORWARD MODEL COMPARISON")
            print("="*70)
            print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
            print(f"Windows: {len(self.create_expanding_windows(len(X)))}")

        n_features = X.shape[1]

        # Define all available models
        all_models = {
            'XGBoost': (XGBoostClassifier, {
                'n_estimators': 500,
                'max_depth': 5,
                'learning_rate': 0.01,
                'early_stopping_rounds': 50
            }),
            'LightGBM': (LightGBMClassifier, {
                'n_estimators': 500,
                'max_depth': 5,
                'learning_rate': 0.01,
                'early_stopping_rounds': 50
            }),
            'CatBoost': (CatBoostClassifier, {
                'iterations': 500,
                'depth': 5,
                'learning_rate': 0.01,
                'early_stopping_rounds': 50
            }),
            'LSTM': (LSTMClassifier, {
                'input_size': n_features,
                'hidden_size': 128,
                'num_layers': 2,
                'seq_length': 30,
                'epochs': 50,
                'early_stopping_patience': 10
            }),
            'TCN': (TCNClassifier, {
                'input_size': n_features,
                'num_filters': 64,
                'kernel_size': 3,
                'num_levels': 4,
                'seq_length': 30,
                'epochs': 50,
                'early_stopping_patience': 10
            })
        }

        # Select models to test
        if models_to_test is None:
            models_to_test = list(all_models.keys())

        results = []

        for model_name in models_to_test:
            if model_name not in all_models:
                print(f"\n[WARNING] Model '{model_name}' not found, skipping...")
                continue

            model_class, model_params = all_models[model_name]

            try:
                result = self.validate_model(
                    model_class,
                    model_params,
                    X, y,
                    model_name,
                    feature_names,
                    verbose=verbose
                )

                results.append({
                    'Model': model_name,
                    'Type': 'Deep Learning' if model_name in ['LSTM', 'TCN'] else 'Traditional ML',
                    'Accuracy': result['overall_accuracy'],
                    'Precision': result['overall_precision'],
                    'Recall': result['overall_recall'],
                    'F1': result['overall_f1'],
                    'AvgTrainTime': result['avg_train_time'],
                    'WindowsCompleted': result['n_completed'],
                    'TotalWindows': result['n_windows']
                })

            except Exception as e:
                print(f"\n[ERROR] {model_name} failed: {e}")
                import traceback
                print("\nFull traceback:")
                traceback.print_exc()
                continue

        # Create results DataFrame
        if not results:
            print("\n[ERROR] No models completed successfully!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)

        if verbose:
            print("\n" + "="*70)
            print(" FINAL WALK-FORWARD COMPARISON")
            print("="*70)
            print(results_df.to_string(index=False))

        return results_df


def load_stock_data(symbol='SBIN', timestamp='20260215_221140'):
    """Load stock data with features."""
    data_path = v3_path / 'data' / 'features' / timestamp / f'{symbol}_features.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    print(f"Loading {symbol} data from: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def prepare_data(df):
    """Prepare features and target."""
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'date',
        'target_close_return', 'target_direction',
        'target_high', 'target_low', 'next_day_log_return'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    X = X[:-1]
    y = df['target'].values[:-1]

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"\n[Data Info]")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  UP days: {(y == 1).sum()} ({(y == 1).mean():.1%})")

    return X, y, feature_cols


def main():
    """Main function for walk-forward validation."""
    print("="*70)
    print(" WALK-FORWARD EXPANDING WINDOW VALIDATION")
    print("="*70)

    # Load data
    print("\n[1/3] Loading SBIN data...")
    df = load_stock_data('SBIN')

    # Prepare
    print("\n[2/3] Preparing features...")
    X, y, feature_names = prepare_data(df)

    # Create validator
    print("\n[3/3] Running walk-forward validation...")
    validator = WalkForwardValidator(
        initial_train_size=800,
        test_window_size=100,
        step_size=100
    )

    # Test all models including deep learning
    models_to_test = ['LightGBM', 'XGBoost', 'CatBoost', 'LSTM', 'TCN']

    # Compare models
    results_df = validator.compare_models(
        X, y,
        feature_names=feature_names,
        models_to_test=models_to_test,
        verbose=True
    )

    # Save results
    output_path = v3_path / '06_results' / 'walk_forward_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n[COMPLETE] Results saved to: {output_path}")

    # Create Excel report
    try:
        excel_path = v3_path / '06_results' / 'WALK_FORWARD_REPORT.xlsx'
        results_df.to_excel(excel_path, index=False, sheet_name='Walk Forward Results')
        print(f"[COMPLETE] Excel report saved to: {excel_path}")
    except:
        print("[WARNING] Excel export failed (openpyxl may not be installed)")


if __name__ == '__main__':
    main()
