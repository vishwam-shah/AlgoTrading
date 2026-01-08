"""
================================================================================
VISUALIZER MODULE
================================================================================
Creates visualizations for model performance:
- Confusion Matrix (for direction classification)
- Predicted vs Actual plots
- Training history plots
- Error distribution plots

All plots are saved with timestamps in results/plots/

Usage:
    from pipeline.utils.visualizer import Visualizer

    viz = Visualizer(symbol='HDFCBANK', model_type='xgboost')
    viz.plot_confusion_matrix(y_true, y_pred)
    viz.plot_predicted_vs_actual(actual, predicted, dates)
    viz.plot_error_distribution(errors)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set matplotlib backend for non-GUI environments
plt.switch_backend('Agg')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


class Visualizer:
    """Create and save visualizations for model performance."""

    def __init__(self, symbol: str, model_type: str, run_id: str = None):
        self.symbol = symbol
        self.model_type = model_type
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create output directory for this run
        self.output_dir = os.path.join(config.PLOTS_DIR, f"{self.run_id}_{symbol}_{model_type}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_direction_class(self, returns: np.ndarray) -> np.ndarray:
        """Convert returns to direction classes (0-4)."""
        classes = np.zeros(len(returns), dtype=int)

        classes[returns < config.DIRECTION_THRESHOLDS['strong_bear']] = 0  # Strong Bear
        classes[(returns >= config.DIRECTION_THRESHOLDS['strong_bear']) &
                (returns < config.DIRECTION_THRESHOLDS['weak_bear'])] = 1  # Weak Bear
        classes[(returns >= config.DIRECTION_THRESHOLDS['weak_bear']) &
                (returns < config.DIRECTION_THRESHOLDS['neutral'])] = 2  # Neutral
        classes[(returns >= config.DIRECTION_THRESHOLDS['neutral']) &
                (returns < config.DIRECTION_THRESHOLDS['weak_bull'])] = 3  # Weak Bull
        classes[returns >= config.DIRECTION_THRESHOLDS['weak_bull']] = 4  # Strong Bull

        return classes

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = None, split_name: str = 'test') -> str:
        """
        Plot confusion matrix for direction classification.

        Args:
            y_true: Actual returns (will be converted to direction classes)
            y_pred: Predicted returns (will be converted to direction classes)
            title: Custom title
            split_name: 'train', 'val', or 'test'

        Returns:
            Path to saved figure
        """
        # Convert to direction classes
        y_true_class = self._get_direction_class(y_true)
        y_pred_class = self._get_direction_class(y_pred)

        # Create confusion matrix
        labels = ['Strong Bear', 'Weak Bear', 'Neutral', 'Weak Bull', 'Strong Bull']
        cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2, 3, 4])

        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm) * 100

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d')

        # Title
        if title is None:
            title = f'{self.symbol} - {self.model_type.upper()} Direction Confusion Matrix ({split_name.title()})'
        ax.set_title(f'{title}\nAccuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save
        filename = f'confusion_matrix_{split_name}_{self.symbol}_{self.model_type}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_predicted_vs_actual(self, actual: np.ndarray, predicted: np.ndarray,
                                  dates: np.ndarray = None, split_name: str = 'test',
                                  show_prices: bool = True, base_price: float = None) -> str:
        """
        Plot predicted vs actual values over time.

        Args:
            actual: Actual values (returns or prices)
            predicted: Predicted values
            dates: Date array for x-axis
            split_name: 'train', 'val', or 'test'
            show_prices: If True and base_price provided, convert returns to prices
            base_price: Starting price for conversion

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Convert to prices if requested
        if show_prices and base_price is not None:
            actual_prices = base_price * np.cumprod(1 + actual)
            predicted_prices = base_price * np.cumprod(1 + predicted)
            y_label = 'Price (INR)'
            actual_plot = actual_prices
            predicted_plot = predicted_prices
        else:
            y_label = 'Return'
            actual_plot = actual
            predicted_plot = predicted

        # Create x-axis
        if dates is not None:
            dates = pd.to_datetime(dates)
            x_axis = dates
        else:
            x_axis = np.arange(len(actual))

        # Plot 1: Actual vs Predicted
        ax1 = axes[0]
        ax1.plot(x_axis, actual_plot, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
        ax1.plot(x_axis, predicted_plot, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Date' if dates is not None else 'Time')
        ax1.set_ylabel(y_label)
        ax1.set_title(f'{self.symbol} - {self.model_type.upper()}: Predicted vs Actual ({split_name.title()})',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        if dates is not None:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Prediction Error
        ax2 = axes[1]
        error = predicted - actual
        error_pct = error * 100 if not show_prices else (predicted_plot - actual_plot) / actual_plot * 100

        ax2.bar(x_axis, error_pct, color=['green' if e >= 0 else 'red' for e in error_pct], alpha=0.6, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.mean(error_pct), color='blue', linestyle='--', linewidth=1,
                    label=f'Mean Error: {np.mean(error_pct):.2f}%')
        ax2.set_xlabel('Date' if dates is not None else 'Time')
        ax2.set_ylabel('Error (%)')
        ax2.set_title('Prediction Error Over Time', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        if dates is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save
        filename = f'predicted_vs_actual_{split_name}_{self.symbol}_{self.model_type}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_error_distribution(self, actual: np.ndarray, predicted: np.ndarray,
                                 split_name: str = 'test') -> str:
        """
        Plot error distribution histogram.

        Args:
            actual: Actual values
            predicted: Predicted values
            split_name: 'train', 'val', or 'test'

        Returns:
            Path to saved figure
        """
        error = (predicted - actual) * 100  # Convert to percentage

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Error Distribution
        ax1 = axes[0]
        ax1.hist(error, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(x=np.mean(error), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(error):.3f}%')
        ax1.axvline(x=np.median(error), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(error):.3f}%')
        ax1.set_xlabel('Prediction Error (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{self.symbol} - Error Distribution ({split_name.title()})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Actual vs Predicted Scatter
        ax2 = axes[1]
        ax2.scatter(actual * 100, predicted * 100, alpha=0.5, s=20, c='steelblue')

        # Perfect prediction line
        min_val = min(actual.min(), predicted.min()) * 100
        max_val = max(actual.max(), predicted.max()) * 100
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax2.set_xlabel('Actual Return (%)')
        ax2.set_ylabel('Predicted Return (%)')
        ax2.set_title(f'Actual vs Predicted Scatter ({split_name.title()})', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(actual, predicted)
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                 fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        filename = f'error_distribution_{split_name}_{self.symbol}_{self.model_type}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_training_summary(self, train_metrics: dict, val_metrics: dict, test_metrics: dict,
                               data_info: dict) -> str:
        """
        Create a summary plot showing all metrics for train/val/test.

        Args:
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            test_metrics: Test metrics dict
            data_info: Data split information

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['rmse', 'mae', 'mape', 'direction_accuracy']
        titles = ['RMSE', 'MAE', 'MAPE (%)', 'Direction Accuracy (%)']
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            values = [
                train_metrics.get(metric, 0),
                val_metrics.get(metric, 0),
                test_metrics.get(metric, 0)
            ]

            # Convert to percentage for display
            if metric in ['direction_accuracy']:
                values = [v * 100 for v in values]

            bars = ax.bar(['Train', 'Validation', 'Test'], values, color=colors, edgecolor='black', alpha=0.8)

            ax.set_ylabel(title)
            ax.set_title(f'{title} by Split', fontsize=12, fontweight='bold')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.4f}' if metric != 'direction_accuracy' else f'{val:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

            ax.grid(True, alpha=0.3, axis='y')

        # Add data info as text
        info_text = (
            f"Data Split Summary\n"
            f"{'=' * 30}\n"
            f"Train: {data_info.get('train_start', 'N/A')} to {data_info.get('train_end', 'N/A')}\n"
            f"       ({data_info.get('train_rows', 0):,} rows)\n"
            f"Val:   {data_info.get('val_start', 'N/A')} to {data_info.get('val_end', 'N/A')}\n"
            f"       ({data_info.get('val_rows', 0):,} rows)\n"
            f"Test:  {data_info.get('test_start', 'N/A')} to {data_info.get('test_end', 'N/A')}\n"
            f"       ({data_info.get('test_rows', 0):,} rows)\n"
            f"{'=' * 30}\n"
            f"Features: {data_info.get('n_features', 0)}"
        )

        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(f'{self.symbol} - {self.model_type.upper()} Training Summary\n'
                     f'Run: {self.run_id}', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0.15, 0.15, 1, 0.95])

        # Save
        filename = f'training_summary_{self.symbol}_{self.model_type}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def create_all_plots(self, actual_train: np.ndarray, predicted_train: np.ndarray,
                          actual_val: np.ndarray, predicted_val: np.ndarray,
                          actual_test: np.ndarray, predicted_test: np.ndarray,
                          dates_train: np.ndarray = None, dates_val: np.ndarray = None,
                          dates_test: np.ndarray = None,
                          train_metrics: dict = None, val_metrics: dict = None,
                          test_metrics: dict = None, data_info: dict = None,
                          feature_importance: dict = None) -> dict:
        """
        Create all visualizations and return paths.

        Returns:
            dict with paths to all created plots
        """
        plots = {}

        # Confusion matrices
        plots['confusion_train'] = self.plot_confusion_matrix(actual_train, predicted_train, split_name='train')
        plots['confusion_val'] = self.plot_confusion_matrix(actual_val, predicted_val, split_name='val')
        plots['confusion_test'] = self.plot_confusion_matrix(actual_test, predicted_test, split_name='test')

        # Predicted vs Actual
        plots['pred_vs_actual_test'] = self.plot_predicted_vs_actual(
            actual_test, predicted_test, dates_test, split_name='test')

        # Error distribution
        plots['error_dist_test'] = self.plot_error_distribution(actual_test, predicted_test, split_name='test')

        # Training summary
        if train_metrics and val_metrics and test_metrics and data_info:
            plots['training_summary'] = self.plot_training_summary(
                train_metrics, val_metrics, test_metrics, data_info)
        
        # Feature importance (if provided)
        if feature_importance:
            plots['feature_importance'] = self.plot_feature_importance(feature_importance)

        return plots
    
    def plot_feature_importance(self, feature_importance: dict, top_n: int = 20) -> str:
        """
        Plot top feature importances.
        
        Args:
            feature_importance: Dict with feature names as keys and importance scores as values
            top_n: Number of top features to show
            
        Returns:
            Path to saved figure
        """
        if not feature_importance:
            return None
            
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{self.symbol} - Top {top_n} Feature Importances ({self.model_type.upper()})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        filename = f'feature_importance_{self.symbol}_{self.model_type}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath


if __name__ == '__main__':
    # Test visualization
    np.random.seed(42)

    # Generate sample data
    n_samples = 100
    actual = np.random.randn(n_samples) * 0.02
    predicted = actual + np.random.randn(n_samples) * 0.01

    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

    # Create visualizer
    viz = Visualizer(symbol='TEST', model_type='xgboost')

    # Create plots
    print("Creating confusion matrix...")
    cm_path = viz.plot_confusion_matrix(actual, predicted, split_name='test')
    print(f"Saved: {cm_path}")

    print("\nCreating predicted vs actual plot...")
    pva_path = viz.plot_predicted_vs_actual(actual, predicted, dates.values, split_name='test')
    print(f"Saved: {pva_path}")

    print("\nCreating error distribution plot...")
    err_path = viz.plot_error_distribution(actual, predicted, split_name='test')
    print(f"Saved: {err_path}")

    print(f"\nAll plots saved to: {viz.output_dir}")
