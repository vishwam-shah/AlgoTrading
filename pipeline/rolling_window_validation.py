"""
================================================================================
ROLLING WINDOW VALIDATION & MODEL COMPARISON
================================================================================
Implements walk-forward validation and comprehensive model comparison.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from multi_target_prediction_system import MultiTargetFeatureEngineering, get_feature_columns
from multi_target_models import LSTMModel, GRUModel, XGBoostModel, EnsembleModel
from pipeline.research_plots import generate_research_plots


def calculate_metrics(y_true, y_pred, target_type='regression'):
    """Calculate comprehensive metrics with trend-neutral accuracy."""
    # Remove NaN
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid]
    y_pred_clean = y_pred[valid]
    
    if len(y_true_clean) == 0:
        return {}
    
    if target_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # For returns, use absolute MAE as "MAPE" equivalent (since returns can be negative)
        # Regular MAPE doesn't make sense for values close to 0
        mape_proxy = mae * 100  # Convert to percentage points
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape_proxy,  # This is MAE in percentage points for returns
            'n_samples': len(y_true_clean)
        }
    else:  # classification
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        
        # Calculate precision and recall manually
        tp = ((y_pred_clean == 1) & (y_true_clean == 1)).sum()
        fp = ((y_pred_clean == 1) & (y_true_clean == 0)).sum()
        fn = ((y_pred_clean == 0) & (y_true_clean == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(y_true_clean)
        }


class RollingWindowValidator:
    """
    Implements rolling window (walk-forward) validation.
    
    This mimics real-world scenario:
    - Train on historical data
    - Validate on next period
    - Test on out-of-sample data
    - Retrain periodically
    """
    
    def __init__(self, train_size=0.6, val_size=0.2, test_size=0.2):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
    
    def split_data(self, df: pd.DataFrame):
        """Split data chronologically."""
        n = len(df)
        
        train_end = int(n * self.train_size)
        val_end = int(n * (self.train_size + self.val_size))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: list):
        """Prepare features and targets."""
        X = df[feature_cols].values
        
        y_dict = {
            'close': df['target_close_return'].values,  # Predict RETURNS not absolute prices
            'high': df['target_high_return'].values,
            'low': df['target_low_return'].values,
            'direction': df['target_direction'].values
        }
        
        return X, y_dict


def train_and_evaluate_model(
    model, 
    model_name: str,
    X_train, y_train_dict, 
    X_val, y_val_dict, 
    X_test, y_test_dict
) -> dict:
    """Train a model and evaluate on test set."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Train
    model.train(X_train, y_train_dict, X_val, y_val_dict)
    
    # Predict on test set
    logger.info("Evaluating on test set...")
    predictions = model.predict(X_test)
    
    # Calculate metrics for each target
    results = {
        'model': model_name,
        'n_features': X_train.shape[1]  # Number of features used
    }
    
    for target in ['close', 'high', 'low']:
        metrics = calculate_metrics(
            y_test_dict[target], 
            predictions[target],
            target_type='regression'
        )
        for metric_name, value in metrics.items():
            results[f'{target}_{metric_name}'] = value
        
        logger.info(f"  {target.upper()}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                   f"R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")
    
    # Direction (classification)
    metrics = calculate_metrics(
        y_test_dict['direction'],
        predictions['direction'],
        target_type='classification'
    )
    for metric_name, value in metrics.items():
        results[f'direction_{metric_name}'] = value
    
    logger.info(f"  DIRECTION: Accuracy={metrics['accuracy']:.2%}, Precision={metrics['precision']:.2%}, "
               f"Recall={metrics['recall']:.2%}, F1={metrics['f1']:.4f}")
    
    # Add sample counts
    for target in ['close', 'high', 'low', 'direction']:
        results[f'{target}_n_samples'] = len(y_test_dict[target])
    
    return results, predictions


def compare_models(symbol: str):
    """
    Train all models and compare their performance.
    """
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# MULTI-TARGET PREDICTION SYSTEM")
    logger.info(f"# Stock: {symbol}")
    logger.info(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*80}\n")
    
    # Prepare dataset
    df = MultiTargetFeatureEngineering.prepare_full_dataset(symbol)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    
    # Split data
    validator = RollingWindowValidator()
    train_df, val_df, test_df = validator.split_data(df)
    
    # Prepare data
    X_train, y_train_dict = validator.prepare_data(train_df, feature_cols)
    X_val, y_val_dict = validator.prepare_data(val_df, feature_cols)
    X_test, y_test_dict = validator.prepare_data(test_df, feature_cols)
    
    # Store results
    all_results = []
    all_predictions = {}
    
    # ========== TRAIN MODELS ==========
    
    # 1. XGBoost (fastest, good baseline)
    logger.info("\n" + "="*80)
    logger.info("MODEL 1/4: XGBoost")
    logger.info("="*80)
    xgb_model = XGBoostModel()
    xgb_results, xgb_predictions = train_and_evaluate_model(
        xgb_model, 'XGBoost',
        X_train, y_train_dict,
        X_val, y_val_dict,
        X_test, y_test_dict
    )
    all_results.append(xgb_results)
    all_predictions['XGBoost'] = xgb_predictions
    
    # 2. LSTM
    logger.info("\n" + "="*80)
    logger.info("MODEL 2/4: LSTM")
    logger.info("="*80)
    lstm_model = LSTMModel(sequence_length=10)
    lstm_results, lstm_predictions = train_and_evaluate_model(
        lstm_model, 'LSTM',
        X_train, y_train_dict,
        X_val, y_val_dict,
        X_test, y_test_dict
    )
    all_results.append(lstm_results)
    all_predictions['LSTM'] = lstm_predictions
    
    # 3. GRU
    logger.info("\n" + "="*80)
    logger.info("MODEL 3/4: GRU")
    logger.info("="*80)
    gru_model = GRUModel(sequence_length=10)
    gru_results, gru_predictions = train_and_evaluate_model(
        gru_model, 'GRU',
        X_train, y_train_dict,
        X_val, y_val_dict,
        X_test, y_test_dict
    )
    all_results.append(gru_results)
    all_predictions['GRU'] = gru_predictions
    
    # Save trained models
    logger.info("\n" + "="*80)
    logger.info("SAVING TRAINED MODELS")
    logger.info("="*80)
    
    # Create model directories
    xgboost_dir = os.path.join(config.MODEL_DIR, 'xgboost', symbol)
    lstm_dir = os.path.join(config.MODEL_DIR, 'lstm', symbol)
    gru_dir = os.path.join(config.MODEL_DIR, 'gru', symbol)
    ensemble_dir = os.path.join(config.MODEL_DIR, 'ensemble', symbol)
    
    # Save each model
    xgb_model.save(symbol, xgboost_dir)
    lstm_model.save(symbol, lstm_dir)
    gru_model.save(symbol, gru_dir)
    
    # 4. Ensemble
    logger.info("\n" + "="*80)
    logger.info("MODEL 4/4: Ensemble (Stacking)")
    logger.info("="*80)
    ensemble_model = EnsembleModel(base_models=[xgb_model, lstm_model, gru_model])
    ensemble_results, ensemble_predictions = train_and_evaluate_model(
        ensemble_model, 'Ensemble',
        X_train, y_train_dict,
        X_val, y_val_dict,
        X_test, y_test_dict
    )
    all_results.append(ensemble_results)
    all_predictions['Ensemble'] = ensemble_predictions
    
    # Save ensemble model
    ensemble_model.save(symbol, ensemble_dir)
    logger.success(f"All models saved for {symbol}")
    
    # Store model objects for research plots
    model_objects = {
        'XGBoost': xgb_model,
        'LSTM': lstm_model,
        'GRU': gru_model,
        'Ensemble': ensemble_model
    }
    
    # ========== COMPARISON ==========
    
    logger.info(f"\n{'#'*80}")
    logger.info("# MODEL COMPARISON SUMMARY")
    logger.info(f"{'#'*80}\n")
    
    df_results = pd.DataFrame(all_results)
    
    # Display comparison
    logger.info("CLOSING RETURN PREDICTION:")
    logger.info("-" * 80)
    comparison = df_results[['model', 'close_rmse', 'close_mae', 'close_r2', 'close_mape']].copy()
    comparison.columns = ['Model', 'RMSE (%)', 'MAE (%)', 'R²', 'MAE pp']  # pp = percentage points
    # Convert to percentage for display
    comparison['RMSE (%)'] = comparison['RMSE (%)'] * 100
    comparison['MAE (%)'] = comparison['MAE (%)'] * 100
    logger.info("\n" + comparison.to_string(index=False))
    
    logger.info("\n\nHIGH RETURN PREDICTION:")
    logger.info("-" * 80)
    comparison = df_results[['model', 'high_rmse', 'high_mae', 'high_r2', 'high_mape']].copy()
    comparison.columns = ['Model', 'RMSE (%)', 'MAE (%)', 'R²', 'MAE pp']
    comparison['RMSE (%)'] = comparison['RMSE (%)'] * 100
    comparison['MAE (%)'] = comparison['MAE (%)'] * 100
    logger.info("\n" + comparison.to_string(index=False))
    
    logger.info("\n\nLOW RETURN PREDICTION:")
    logger.info("-" * 80)
    comparison = df_results[['model', 'low_rmse', 'low_mae', 'low_r2', 'low_mape']].copy()
    comparison.columns = ['Model', 'RMSE (%)', 'MAE (%)', 'R²', 'MAE pp']
    comparison['RMSE (%)'] = comparison['RMSE (%)'] * 100
    comparison['MAE (%)'] = comparison['MAE (%)'] * 100
    logger.info("\n" + comparison.to_string(index=False))
    
    logger.info("\n\nDIRECTION PREDICTION:")
    logger.info("-" * 80)
    comparison = df_results[['model', 'direction_accuracy', 'direction_precision', 'direction_recall', 'direction_f1']].copy()
    comparison.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    logger.info("\n" + comparison.to_string(index=False))
    
    # Find best model for each target
    logger.info("\n\nBEST MODELS:")
    logger.info("-" * 80)
    best_close = df_results.loc[df_results['close_rmse'].idxmin(), 'model']
    best_high = df_results.loc[df_results['high_rmse'].idxmin(), 'model']
    best_low = df_results.loc[df_results['low_rmse'].idxmin(), 'model']
    best_direction = df_results.loc[df_results['direction_accuracy'].idxmax(), 'model']
    
    logger.success(f"  Closing Price:  {best_close}")
    logger.success(f"  High Price:     {best_high}")
    logger.success(f"  Low Price:      {best_low}")
    logger.success(f"  Direction:      {best_direction}")
    
    # Save results
    output_dir = os.path.join(config.BASE_DIR, 'evaluation_results', 'multi_target')
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'{symbol}_model_comparison.csv')
    df_results.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save predictions
    test_df_with_predictions = test_df.copy()
    for model_name, predictions in all_predictions.items():
        for target in ['close', 'high', 'low', 'direction']:
            test_df_with_predictions[f'{model_name}_{target}_pred'] = predictions[target]
    
    predictions_file = os.path.join(output_dir, f'{symbol}_predictions.csv')
    test_df_with_predictions.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Create visualization
    create_comparison_plots(symbol, test_df_with_predictions, all_predictions, output_dir)
    
    # Generate comprehensive research plots
    logger.info("\nGenerating research plots...")
    n_features = X_train.shape[1]
    generate_research_plots(symbol, test_df_with_predictions, all_predictions, model_objects, n_features)
    
    # Generate comprehensive research plots
    logger.info("\nGenerating research plots...")
    generate_research_plots(symbol, test_df_with_predictions, all_predictions, model_objects, n_features)
    
    # Return summary results for programmatic access
    best_model_idx = df_results['direction_accuracy'].idxmax()
    best_model_row = df_results.loc[best_model_idx]
    
    return {
        'symbol': symbol,
        'best_model': best_model_row['model'],
        'direction_accuracy': best_model_row['direction_accuracy'] * 100,
        'close_mape': best_model_row['close_mape'],
        'close_r2': best_model_row['close_r2'],
        'test_samples': best_model_row['direction_n_samples'],
        'all_results': df_results.to_dict('records')
    }
    
    return df_results, all_predictions


def create_comparison_plots(symbol: str, test_df: pd.DataFrame, all_predictions: dict, output_dir: str):
    """Create comparison plots."""
    
    # Get actual prices (not returns)
    actual_close = test_df['target_close'].values
    actual_high = test_df['target_high'].values
    actual_low = test_df['target_low'].values
    current_close = test_df['close'].values
    
    # Convert return predictions back to prices
    predicted_prices = {}
    for model_name, predictions in all_predictions.items():
        # Close: predicted_close = current_close * (1 + return_pred)
        pred_close = current_close * (1 + predictions['close'])
        # High: predicted_high = current_close * (1 + high_return_pred)
        pred_high = current_close * (1 + predictions['high'])
        # Low: predicted_low = current_close * (1 + low_return_pred)
        pred_low = current_close * (1 + predictions['low'])
        
        predicted_prices[model_name] = {
            'close': pred_close,
            'high': pred_high,
            'low': pred_low
        }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol} - Model Predictions Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Closing price
    ax = axes[0]
    ax.plot(actual_close, label='Actual', color='black', linewidth=2, alpha=0.7)
    colors = ['blue', 'red', 'green', 'purple']
    for i, (model_name, prices) in enumerate(predicted_prices.items()):
        ax.plot(prices['close'], label=f'{model_name}', 
               color=colors[i], alpha=0.6, linestyle='--')
    ax.set_title('Closing Price Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: High price
    ax = axes[1]
    ax.plot(actual_high, label='Actual', color='black', linewidth=2, alpha=0.7)
    for i, (model_name, prices) in enumerate(predicted_prices.items()):
        ax.plot(prices['high'], label=f'{model_name}', 
               color=colors[i], alpha=0.6, linestyle='--')
    ax.set_title('High Price Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Low price
    ax = axes[2]
    ax.plot(actual_low, label='Actual', color='black', linewidth=2, alpha=0.7)
    for i, (model_name, prices) in enumerate(predicted_prices.items()):
        ax.plot(prices['low'], label=f'{model_name}', 
               color=colors[i], alpha=0.6, linestyle='--')
    ax.set_title('Low Price Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots folder
    evaluation_dir = os.path.join(config.BASE_DIR, 'evaluation_results')
    plots_dir = os.path.join(evaluation_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, f'{symbol}_comparison_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to: {plot_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Target Stock Prediction System")
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., HDFCBANK)')
    args = parser.parse_args()
    
    # Run comparison
    compare_models(args.symbol.upper())


if __name__ == '__main__':
    main()
