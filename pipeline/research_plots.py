"""
================================================================================
RESEARCH PLOTS GENERATOR
================================================================================
Generates comprehensive research plots for stock prediction analysis:
- Confusion matrices for direction prediction
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance (XGBoost)
- Error distribution plots
- Combined analysis across all stocks

Usage:
    from pipeline.research_plots import generate_research_plots, generate_combined_plots
    
    # For single stock
    generate_research_plots(symbol, test_df, predictions, model_objects)
    
    # For combined analysis
    generate_combined_plots()
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from loguru import logger
import config


def generate_research_plots(symbol, test_df, all_predictions, model_objects, n_features):
    """
    Generate comprehensive research plots for a single stock.
    
    Args:
        symbol: Stock symbol
        test_df: Test dataframe with actuals
        all_predictions: Dict of predictions {model_name: {target: values}}
        model_objects: Dict of trained model objects {model_name: model}
        n_features: Number of features used
    """
    logger.info(f"Generating research plots for {symbol}...")
    
    # Create plots directory
    plots_dir = os.path.join(config.BASE_DIR, 'evaluation_results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Confusion Matrix for each model
    _plot_confusion_matrices(symbol, test_df, all_predictions, plots_dir)
    
    # 2. ROC Curves
    _plot_roc_curves(symbol, test_df, all_predictions, plots_dir)
    
    # 3. Precision-Recall Curves
    _plot_precision_recall_curves(symbol, test_df, all_predictions, plots_dir)
    
    # 4. Feature Importance (XGBoost only)
    if 'XGBoost' in model_objects:
        _plot_feature_importance(symbol, model_objects['XGBoost'], plots_dir)
    
    # 5. Error Distribution
    _plot_error_distribution(symbol, test_df, all_predictions, plots_dir)
    
    # 6. Prediction Scatter Plots
    _plot_prediction_scatter(symbol, test_df, all_predictions, plots_dir)
    
    logger.success(f"Research plots generated for {symbol}")


def _plot_confusion_matrices(symbol, test_df, all_predictions, plots_dir):
    """Generate confusion matrix plots for direction prediction."""
    
    # Get actual direction
    actual_direction = test_df['target_direction'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(actual_direction)
    actual_direction = actual_direction[valid_mask]
    
    # Create figure with 2x2 subplots (4 models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{symbol} - Confusion Matrices (Direction Prediction)', 
                 fontsize=16, fontweight='bold')
    
    model_names = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['Blues', 'Reds', 'Greens', 'Purples']
    
    for idx, (ax, model_name, cmap) in enumerate(zip(axes.flat, model_names, colors)):
        if model_name in all_predictions:
            pred_direction = all_predictions[model_name]['direction']
            
            # Apply same mask to predictions and remove NaN
            pred_direction = pred_direction[valid_mask]
            pred_valid_mask = ~np.isnan(pred_direction)
            actual_clean = actual_direction[pred_valid_mask]
            pred_clean = pred_direction[pred_valid_mask]
            
            if len(actual_clean) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
                continue
            
            # Calculate confusion matrix
            cm = confusion_matrix(actual_clean, pred_clean)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, 
                       cbar_kws={'label': 'Count'})
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Down (0)', 'Up (1)'])
            ax.set_yticklabels(['Down (0)', 'Up (1)'])
            
            # Add accuracy to subtitle
            accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
            ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                   transform=ax.transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, f'{symbol}_confusion_matrices.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Confusion matrices saved: {plot_file}")
    plt.close()


def _plot_roc_curves(symbol, test_df, all_predictions, plots_dir):
    """Generate ROC curve plots."""
    
    actual_direction = test_df['target_direction'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(actual_direction)
    actual_direction = actual_direction[valid_mask]
    
    if len(actual_direction) == 0:
        logger.warning("No valid data for ROC curves")
        return
    
    plt.figure(figsize=(10, 8))
    
    model_names = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for model_name, color in zip(model_names, colors):
        if model_name in all_predictions:
            # Use prediction probabilities if available, otherwise use predictions
            pred_proba = all_predictions[model_name]['direction']
            
            # Apply mask and remove NaN
            pred_proba = pred_proba[valid_mask]
            pred_valid_mask = ~np.isnan(pred_proba)
            actual_clean = actual_direction[pred_valid_mask]
            pred_clean = pred_proba[pred_valid_mask]
            
            if len(actual_clean) == 0:
                continue
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(actual_clean, pred_clean)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{symbol} - ROC Curves (Direction Prediction)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, f'{symbol}_roc_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  ROC curves saved: {plot_file}")
    plt.close()


def _plot_precision_recall_curves(symbol, test_df, all_predictions, plots_dir):
    """Generate Precision-Recall curve plots."""
    
    actual_direction = test_df['target_direction'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(actual_direction)
    actual_direction = actual_direction[valid_mask]
    
    if len(actual_direction) == 0:
        logger.warning("No valid data for Precision-Recall curves")
        return
    
    plt.figure(figsize=(10, 8))
    
    model_names = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for model_name, color in zip(model_names, colors):
        if model_name in all_predictions:
            pred_proba = all_predictions[model_name]['direction']
            
            # Apply mask and remove NaN
            pred_proba = pred_proba[valid_mask]
            pred_valid_mask = ~np.isnan(pred_proba)
            actual_clean = actual_direction[pred_valid_mask]
            pred_clean = pred_proba[pred_valid_mask]
            
            if len(actual_clean) == 0:
                continue
            
            # Calculate Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(actual_clean, pred_clean)
            avg_precision = average_precision_score(actual_clean, pred_clean)
            
            # Plot
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{symbol} - Precision-Recall Curves (Direction Prediction)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, f'{symbol}_precision_recall_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Precision-Recall curves saved: {plot_file}")
    plt.close()


def _plot_feature_importance(symbol, xgb_model, plots_dir):
    """Generate feature importance plot for XGBoost model."""
    
    try:
        # Get feature importance for direction model
        if hasattr(xgb_model, 'models') and 'direction' in xgb_model.models:
            model = xgb_model.models['direction']
            
            # Get feature importance
            importance = model.get_score(importance_type='gain')
            
            # Convert to dataframe and sort
            importance_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False).head(30)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.barh(range(len(importance_df)), importance_df['importance'].values)
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('Importance (Gain)', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'{symbol} - Top 30 Feature Importance (XGBoost Direction Model)',
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plot_file = os.path.join(plots_dir, f'{symbol}_feature_importance.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"  Feature importance saved: {plot_file}")
            plt.close()
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")


def _plot_error_distribution(symbol, test_df, all_predictions, plots_dir):
    """Generate error distribution plots for price predictions."""
    
    # Calculate errors for close price prediction
    actual_close = test_df['target_close'].values
    current_close = test_df['close'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{symbol} - Prediction Error Distribution (Close Price)',
                 fontsize=16, fontweight='bold')
    
    model_names = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for idx, (ax, model_name, color) in enumerate(zip(axes.flat, model_names, colors)):
        if model_name in all_predictions:
            # Convert return predictions to prices
            pred_close = current_close * (1 + all_predictions[model_name]['close'])
            
            # Calculate percentage errors
            errors = ((pred_close - actual_close) / actual_close) * 100
            
            # Plot histogram
            ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            
            # Add statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.axvline(x=mean_error, color='green', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_error:.2f}%')
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Prediction Error (%)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            textstr = f'Mean: {mean_error:.2f}%\nStd: {std_error:.2f}%\nMedian: {np.median(errors):.2f}%'
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, f'{symbol}_error_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Error distribution saved: {plot_file}")
    plt.close()


def _plot_prediction_scatter(symbol, test_df, all_predictions, plots_dir):
    """Generate scatter plots of predicted vs actual values."""
    
    actual_close = test_df['target_close'].values
    current_close = test_df['close'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{symbol} - Predicted vs Actual Close Price',
                 fontsize=16, fontweight='bold')
    
    model_names = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for idx, (ax, model_name, color) in enumerate(zip(axes.flat, model_names, colors)):
        if model_name in all_predictions:
            # Convert return predictions to prices
            pred_close = current_close * (1 + all_predictions[model_name]['close'])
            
            # Scatter plot
            ax.scatter(actual_close, pred_close, alpha=0.5, color=color, s=20)
            
            # Perfect prediction line
            min_val = min(actual_close.min(), pred_close.min())
            max_val = max(actual_close.max(), pred_close.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(actual_close, pred_close)
            
            ax.set_title(f'{model_name} (R² = {r2:.4f})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Actual Price', fontsize=10)
            ax.set_ylabel('Predicted Price', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, f'{symbol}_prediction_scatter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Prediction scatter saved: {plot_file}")
    plt.close()


def generate_combined_plots():
    """
    Generate combined analysis plots across all stocks.
    Reads all model comparison CSVs and creates aggregate visualizations.
    """
    logger.info("Generating combined analysis plots for all stocks...")
    
    # Load all results
    results_dir = os.path.join(config.BASE_DIR, 'evaluation_results', 'multi_target')
    all_results = []
    
    for file in os.listdir(results_dir):
        if file.endswith('_model_comparison.csv'):
            symbol = file.replace('_model_comparison.csv', '')
            df = pd.read_csv(os.path.join(results_dir, file))
            df['symbol'] = symbol
            all_results.append(df)
    
    if not all_results:
        logger.warning("No model comparison files found!")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Loaded results for {len(combined_df['symbol'].unique())} stocks")
    
    plots_dir = os.path.join(config.BASE_DIR, 'evaluation_results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model Performance Comparison
    _plot_combined_model_performance(combined_df, plots_dir)
    
    # 2. Direction Accuracy Distribution
    _plot_combined_direction_accuracy(combined_df, plots_dir)
    
    # 3. R² Score Distribution
    _plot_combined_r2_distribution(combined_df, plots_dir)
    
    # 4. Best Model Distribution
    _plot_best_model_distribution(combined_df, plots_dir)
    
    # 5. Performance Heatmap
    _plot_performance_heatmap(combined_df, plots_dir)
    
    logger.success(f"Combined analysis plots generated in {plots_dir}")


def _plot_combined_model_performance(combined_df, plots_dir):
    """Plot overall model performance comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Combined Model Performance Across All Stocks', 
                 fontsize=18, fontweight='bold')
    
    models = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    # Direction Accuracy
    ax = axes[0, 0]
    data = [combined_df[combined_df['model'] == m]['direction_accuracy'].values 
            for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title('Direction Accuracy Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Close R²
    ax = axes[0, 1]
    data = [combined_df[combined_df['model'] == m]['close_r2'].values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title('Close Price R² Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Close MAPE
    ax = axes[1, 0]
    data = [combined_df[combined_df['model'] == m]['close_mape'].values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title('Close Price MAPE Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 Score
    ax = axes[1, 1]
    data = [combined_df[combined_df['model'] == m]['direction_f1'].values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title('Direction F1 Score Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, 'COMBINED_model_performance.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Combined performance plot saved: {plot_file}")
    plt.close()


def _plot_combined_direction_accuracy(combined_df, plots_dir):
    """Plot direction accuracy distribution."""
    
    plt.figure(figsize=(14, 8))
    
    models = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for model, color in zip(models, colors):
        model_data = combined_df[combined_df['model'] == model]['direction_accuracy']
        plt.hist(model_data, bins=30, alpha=0.5, label=model, color=color, edgecolor='black')
    
    plt.xlabel('Direction Accuracy', fontsize=12)
    plt.ylabel('Frequency (Number of Stocks)', fontsize=12)
    plt.title('Direction Accuracy Distribution Across All Stocks', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add median lines
    for model, color in zip(models, colors):
        model_data = combined_df[combined_df['model'] == model]['direction_accuracy']
        median = model_data.median()
        plt.axvline(x=median, color=color, linestyle='--', linewidth=2, 
                   label=f'{model} Median: {median:.2%}')
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, 'COMBINED_direction_accuracy_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Direction accuracy distribution saved: {plot_file}")
    plt.close()


def _plot_combined_r2_distribution(combined_df, plots_dir):
    """Plot R² score distribution."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('R² Score Distribution Across All Stocks',
                 fontsize=16, fontweight='bold')
    
    targets = ['close', 'high', 'low']
    target_labels = ['Close Price', 'High Price', 'Low Price']
    
    models = ['XGBoost', 'LSTM', 'GRU', 'Ensemble']
    colors = ['blue', 'red', 'green', 'purple']
    
    for ax, target, label in zip(axes, targets, target_labels):
        for model, color in zip(models, colors):
            model_data = combined_df[combined_df['model'] == model][f'{target}_r2']
            ax.hist(model_data, bins=30, alpha=0.5, label=model, color=color, edgecolor='black')
        
        ax.set_xlabel('R² Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, 'COMBINED_r2_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  R² distribution saved: {plot_file}")
    plt.close()


def _plot_best_model_distribution(combined_df, plots_dir):
    """Plot which model performs best for each metric."""
    
    # Find best model for each stock and metric
    metrics = {
        'Direction Accuracy': 'direction_accuracy',
        'Close R²': 'close_r2',
        'Close MAPE': 'close_mape',
        'F1 Score': 'direction_f1'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best Model Distribution Across Stocks',
                 fontsize=16, fontweight='bold')
    
    for ax, (metric_name, metric_col) in zip(axes.flat, metrics.items()):
        # For each stock, find the best model
        if 'mape' in metric_col.lower():
            # Lower is better for MAPE
            best_models = combined_df.loc[combined_df.groupby('symbol')[metric_col].idxmin(), 'model']
        else:
            # Higher is better for others
            best_models = combined_df.loc[combined_df.groupby('symbol')[metric_col].idxmax(), 'model']
        
        # Count occurrences
        counts = best_models.value_counts()
        
        # Plot
        colors_map = {'XGBoost': 'blue', 'LSTM': 'red', 'GRU': 'green', 'Ensemble': 'purple'}
        colors = [colors_map.get(m, 'gray') for m in counts.index]
        
        bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=0)
        ax.set_ylabel('Number of Stocks', fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, 'COMBINED_best_model_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Best model distribution saved: {plot_file}")
    plt.close()


def _plot_performance_heatmap(combined_df, plots_dir):
    """Plot performance heatmap for all stocks and models."""
    
    # Pivot to get model performance matrix
    pivot = combined_df.pivot_table(
        index='symbol',
        columns='model',
        values='direction_accuracy'
    )
    
    # Sort by best average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    pivot = pivot.drop('avg', axis=1)
    
    # Plot heatmap
    plt.figure(figsize=(12, max(20, len(pivot)*0.3)))
    sns.heatmap(pivot, annot=False, cmap='RdYlGn', center=0.5, 
                cbar_kws={'label': 'Direction Accuracy'},
                linewidths=0.5)
    plt.title('Direction Accuracy Heatmap (All Stocks & Models)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Stock Symbol', fontsize=12)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, 'COMBINED_performance_heatmap.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Performance heatmap saved: {plot_file}")
    plt.close()


if __name__ == '__main__':
    # Generate combined plots
    generate_combined_plots()
