"""
================================================================================
STEP 4: PREDICT & EVALUATE
================================================================================
Compares all 4 trained models and selects the best performer.
Generates predictions, plots, and comprehensive metrics.

Uses walk-forward validation to ensure no look-ahead bias.

Usage:
    # Single stock
    python pipeline/04_predict.py --symbol RELIANCE
    
    # With trend-neutral evaluation
    python pipeline/04_predict.py --symbol RELIANCE --evaluate-trend
    
    # Multiple stocks
    python pipeline/04_predict.py --symbols RELIANCE TCS INFY
    
    # All stocks
    python pipeline/04_predict.py --all

Output:
    - evaluation_results/multi_target/{STOCK}_model_comparison.csv - Model metrics
    - evaluation_results/multi_target/{STOCK}_predictions.csv - Full predictions
    - evaluation_results/multi_target/{STOCK}_comparison_plot.png - Visualization
    - logs/prediction_log.csv - Prediction metadata
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.rolling_window_validation import compare_models
from pipeline.utils.pipeline_logger import PipelineLogger


def calculate_trend_neutral_accuracy(predictions_df: pd.DataFrame, stock_name: str) -> dict:
    """
    Calculate trend-neutral accuracy by removing linear trend.
    
    This reveals the true predictive skill without trend bias.
    
    Args:
        predictions_df: DataFrame with predictions and actual values
        stock_name: Stock symbol
    
    Returns:
        Dictionary with trend-neutral metrics
    """
    logger.info(f"\n{'='*40}")
    logger.info("Calculating Trend-Neutral Accuracy")
    logger.info(f"{'='*40}")
    
    try:
        # Get actual prices
        prices = predictions_df['close'].values
        
        # Remove linear trend using regression
        time_index = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, prices)
        
        # Detrended prices
        trend_line = slope * time_index + intercept
        detrended_prices = prices - trend_line
        
        # Calculate direction on detrended prices
        detrended_returns = np.diff(detrended_prices) / np.abs(detrended_prices[:-1])
        detrended_direction = (detrended_returns > 0).astype(int)
        
        # Get model predictions (binary direction)
        # Try different column names for model predictions
        pred_col = None
        for col in ['xgboost_direction', 'ensemble_direction', 'lstm_direction', 'gru_direction']:
            if col in predictions_df.columns:
                pred_col = col
                break
        
        if pred_col is None:
            logger.warning("No model prediction column found, skipping trend-neutral evaluation")
            return None
        
        model_predictions = predictions_df[pred_col].values[1:]  # Align with returns
        
        # Ensure same length
        min_len = min(len(detrended_direction), len(model_predictions))
        detrended_dir_valid = detrended_direction[:min_len]
        model_preds_valid = model_predictions[:min_len]
        
        # Calculate trend-neutral accuracy
        trend_neutral_acc = (model_preds_valid == detrended_dir_valid).mean()
        
        # Original direction accuracy
        if 'direction_target' in predictions_df.columns:
            actual_direction = predictions_df['direction_target'].values[1:]
            actual_dir_valid = actual_direction[:min_len]
            original_acc = (model_preds_valid == actual_dir_valid).mean()
        else:
            original_acc = None
        
        # Calculate trend advantage
        trend_advantage = original_acc - trend_neutral_acc if original_acc else 0
        
        results = {
            'stock': stock_name,
            'original_accuracy': original_acc,
            'trend_neutral_accuracy': trend_neutral_acc,
            'trend_correlation': r_value,
            'trend_advantage': trend_advantage,
            'trend_slope': slope,
            'detrended_samples': len(detrended_dir_valid)
        }
        
        logger.info(f"Results:")
        logger.info(f"  Original Accuracy:       {original_acc*100:.2f}%" if original_acc else "  N/A")
        logger.info(f"  Trend-Neutral Accuracy:  {trend_neutral_acc*100:.2f}%")
        logger.info(f"  Trend Correlation:       {r_value:.4f}")
        logger.info(f"  Trend Advantage:         {trend_advantage*100:.2f} pp")
        
        if abs(r_value) > 0.7:
            logger.warning(f"⚠️  STRONG TREND DETECTED (r={r_value:.2f})")
            logger.warning(f"   Model gained {trend_advantage*100:.1f}pp just from the trend")
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating trend-neutral accuracy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def predict_and_evaluate(symbol: str, evaluate_trend: bool = False, pipeline_logger: PipelineLogger = None) -> dict:
    """
    Run predictions and evaluation for a single stock.
    
    Args:
        symbol: Stock symbol
        evaluate_trend: Whether to calculate trend-neutral accuracy
        pipeline_logger: PipelineLogger instance for logging
    
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PREDICT & EVALUATE: {symbol}")
    logger.info(f"{'='*80}")
    
    try:
        # Check if models exist
        model_dirs = ['xgboost', 'lstm', 'gru', 'ensemble']
        models_exist = False
        for model_dir in model_dirs:
            model_path = os.path.join(config.MODEL_DIR, model_dir, f"{symbol}_{model_dir}_model.*")
            if os.path.exists(os.path.join(config.MODEL_DIR, model_dir)):
                models_exist = True
                break
        
        if not models_exist:
            logger.error(f"Models not found for {symbol}")
            logger.info(f"Run: python pipeline/03_train_models.py --symbol {symbol}")
            return None
        
        # Run comparison (trains and evaluates all models)
        logger.info("Comparing all 4 models...")
        results = compare_models(symbol)
        
        if results is None:
            logger.error(f"Failed to compare models for {symbol}")
            return None
        
        # Display results
        logger.info(f"\n{'='*40}")
        logger.info("Model Comparison Results")
        logger.info(f"{'='*40}")
        logger.info(f"Best Model: {results['best_model']}")
        logger.info(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
        logger.info(f"Close MAPE: {results['close_mape']:.2f}%")
        logger.info(f"Close R²: {results['close_r2']:.4f}")
        logger.info(f"Test Samples: {results['test_samples']}")
        
        # Trend-neutral evaluation
        trend_results = None
        if evaluate_trend:
            predictions_file = os.path.join(
                config.EVALUATION_DIR, 
                'multi_target',
                f"{symbol}_predictions.csv"
            )
            
            if os.path.exists(predictions_file):
                predictions_df = pd.read_csv(predictions_file)
                trend_results = calculate_trend_neutral_accuracy(predictions_df, symbol)
                
                if trend_results:
                    results['trend_neutral_accuracy'] = trend_results['trend_neutral_accuracy']
                    results['trend_correlation'] = trend_results['trend_correlation']
                    results['trend_advantage'] = trend_results['trend_advantage']
        
        # Log to pipeline logger
        if pipeline_logger:
            pipeline_logger.log_prediction(
                symbol=symbol,
                model=results['best_model'],
                predictions_df=None  # Already saved by compare_models
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error predicting/evaluating {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main entry point for prediction and evaluation."""
    parser = argparse.ArgumentParser(description='Step 4: Predict and evaluate models')
    parser.add_argument('--symbol', type=str, help='Single stock symbol to predict')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols to predict')
    parser.add_argument('--all', action='store_true', help='Predict all stocks from config')
    parser.add_argument('--evaluate-trend', action='store_true', help='Calculate trend-neutral accuracy')
    
    args = parser.parse_args()
    
    # Determine which stocks to process
    if args.symbol:
        stocks = [args.symbol]
    elif args.symbols:
        stocks = args.symbols
    elif args.all:
        # Get all stocks from models directory
        xgboost_dir = os.path.join(config.MODEL_DIR, 'xgboost')
        if os.path.exists(xgboost_dir):
            stocks = list(set([
                f.replace('_xgboost_model.pkl', '') 
                for f in os.listdir(xgboost_dir) 
                if f.endswith('_xgboost_model.pkl')
            ]))
        else:
            logger.error(f"Models directory not found: {xgboost_dir}")
            return
    else:
        parser.print_help()
        return
    
    # Initialize logger
    pipeline_logger = PipelineLogger()
    
    # Process each stock
    logger.info(f"\nPredicting and evaluating {len(stocks)} stocks...")
    results_list = []
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(stocks, 1):
        logger.info(f"\nProgress: {i}/{len(stocks)}")
        
        results = predict_and_evaluate(
            symbol, 
            evaluate_trend=args.evaluate_trend,
            pipeline_logger=pipeline_logger
        )
        
        if results:
            results_list.append(results)
            success_count += 1
        else:
            fail_count += 1
    
    # Create summary
    if results_list:
        summary_df = pd.DataFrame(results_list)
        summary_df = summary_df.sort_values('direction_accuracy', ascending=False)
        
        # Save summary
        summary_file = os.path.join(config.EVALUATION_DIR, 'multi_target', 'BATCH_SUMMARY.csv')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        summary_df.to_csv(summary_file, index=False)
        logger.success(f"\nSaved batch summary to {summary_file}")
        
        # Display top performers
        logger.info(f"\n{'='*80}")
        logger.info("TOP 10 PERFORMERS")
        logger.info(f"{'='*80}")
        
        top_10 = summary_df.head(10)
        for idx, row in top_10.iterrows():
            logger.info(f"{row['symbol']:15} - {row['best_model']:10} - {row['direction_accuracy']:.2f}% - MAPE: {row['close_mape']:.2f}%")
        
        # Statistics
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Average Direction Accuracy: {summary_df['direction_accuracy'].mean():.2f}%")
        logger.info(f"Average Close MAPE: {summary_df['close_mape'].mean():.2f}%")
        logger.info(f"Average Close R²: {summary_df['close_r2'].mean():.4f}")
        logger.info(f"Stocks > 70% Accuracy: {(summary_df['direction_accuracy'] > 70).sum()}/{len(summary_df)}")
        
        # Model selection breakdown
        logger.info(f"\nBest Model Distribution:")
        model_counts = summary_df['best_model'].value_counts()
        for model, count in model_counts.items():
            logger.info(f"  {model}: {count} stocks ({count/len(summary_df)*100:.1f}%)")
        
        if args.evaluate_trend and 'trend_neutral_accuracy' in summary_df.columns:
            logger.info(f"\nTrend Analysis:")
            logger.info(f"  Average Original Accuracy: {summary_df['direction_accuracy'].mean():.2f}%")
            logger.info(f"  Average Trend-Neutral Accuracy: {summary_df['trend_neutral_accuracy'].mean()*100:.2f}%")
            logger.info(f"  Average Trend Advantage: {summary_df['trend_advantage'].mean()*100:.2f} pp")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"PREDICTION & EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {success_count}/{len(stocks)}")
    logger.info(f"Failed: {fail_count}/{len(stocks)}")
    logger.info(f"Results saved to: {os.path.join(config.EVALUATION_DIR, 'multi_target')}")
    logger.info(f"Logs saved to: {config.PREDICTION_LOG}")


if __name__ == '__main__':
    main()
