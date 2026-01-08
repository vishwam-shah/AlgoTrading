"""
================================================================================
MAIN PIPELINE - Stock Prediction System
================================================================================
Unified entry point for the complete 4-step stock prediction pipeline.

Pipeline Steps:
    1. Data Collection     - Download OHLCV data from NSE
    2. Feature Engineering - Generate 244 professional features
    3. Train Models        - Train XGBoost, LSTM, GRU, Ensemble
    4. Predict & Evaluate  - Compare models, select best, generate predictions

Usage:
    # Run complete pipeline for single stock
    python main_pipeline.py --symbol RELIANCE
    
    # Run complete pipeline with trend evaluation
    python main_pipeline.py --symbol RELIANCE --evaluate-trend-bias
    
    # Run specific steps only
    python main_pipeline.py --symbol RELIANCE --steps 1 2      # Data + Features only
    python main_pipeline.py --symbol RELIANCE --steps 3 4      # Train + Predict only
    python main_pipeline.py --symbol RELIANCE --step 2         # Feature engineering only
    
    # Batch processing
    python main_pipeline.py --batch --all
    python main_pipeline.py --batch --stocks RELIANCE TCS INFY

Features:
    - Modular 4-step pipeline
    - Multi-target prediction (Close, High, Low, Direction)
    - 4 Models (XGBoost, LSTM, GRU, Ensemble)
    - 244 Professional features
    - Walk-forward validation
    - Trend-neutral evaluation
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Import pipeline steps (will call as modules)
# We'll import the functions directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.utils.pipeline_logger import PipelineLogger


def run_step_1(symbol: str, pipeline_logger: PipelineLogger) -> bool:
    """Step 1: Data Collection"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"STEP 1: DATA COLLECTION - {symbol}")
    logger.info(f"{'#'*80}\n")
    
    # Import and run
    from pipeline import collect_stock
    result = collect_stock(symbol, start_date='2015-01-01', pipeline_logger=pipeline_logger)
    
    if result:
        logger.success(f"✓ Step 1 complete: Data collected for {symbol}")
    else:
        logger.error(f"✗ Step 1 failed for {symbol}")
    
    return result


def run_step_2(symbol: str, pipeline_logger: PipelineLogger) -> bool:
    """Step 2: Feature Engineering"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"STEP 2: FEATURE ENGINEERING - {symbol}")
    logger.info(f"{'#'*80}\n")
    
    # Import and run
    from pipeline import engineer_features
    result = engineer_features(symbol, pipeline_logger=pipeline_logger)
    
    if result:
        logger.success(f"✓ Step 2 complete: Features engineered for {symbol}")
    else:
        logger.error(f"✗ Step 2 failed for {symbol}")
    
    return result


def run_step_3(symbol: str, pipeline_logger: PipelineLogger) -> bool:
    """Step 3: Train Models"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"STEP 3: TRAIN MODELS - {symbol}")
    logger.info(f"{'#'*80}\n")
    
    # Import and run
    from pipeline import train_models
    result = train_models(symbol, pipeline_logger=pipeline_logger)
    
    if result:
        logger.success(f"✓ Step 3 complete: Models trained for {symbol}")
    else:
        logger.error(f"✗ Step 3 failed for {symbol}")
    
    return result


def run_step_4(symbol: str, evaluate_trend: bool, pipeline_logger: PipelineLogger) -> dict:
    """Step 4: Predict & Evaluate"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"STEP 4: PREDICT & EVALUATE - {symbol}")
    logger.info(f"{'#'*80}\n")
    
    # Import and run
    from pipeline import predict_and_evaluate
    results = predict_and_evaluate(symbol, evaluate_trend=evaluate_trend, pipeline_logger=pipeline_logger)
    
    if results:
        logger.success(f"✓ Step 4 complete: Predictions generated for {symbol}")
    else:
        logger.error(f"✗ Step 4 failed for {symbol}")
    
    return results


def run_single_stock(symbol: str, steps: list = None, evaluate_trend: bool = False):
    """
    Run pipeline for a single stock.
    
    Args:
        symbol: Stock symbol
        steps: List of steps to run (1-4). If None, runs all steps.
        evaluate_trend: Whether to calculate trend-neutral accuracy
    
    Returns:
        dict: Results from final step
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING PIPELINE: {symbol}")
    logger.info(f"{'='*80}")
    
    # Initialize logger
    pipeline_logger = PipelineLogger()
    
    # Determine which steps to run
    if steps is None:
        steps = [1, 2, 3, 4]  # Run all steps by default
    
    logger.info(f"Steps to run: {steps}")
    
    results = None
    
    try:
        # Step 1: Data Collection
        if 1 in steps:
            if not run_step_1(symbol, pipeline_logger):
                return None
        
        # Step 2: Feature Engineering
        if 2 in steps:
            if not run_step_2(symbol, pipeline_logger):
                return None
        
        # Step 3: Train Models
        if 3 in steps:
            if not run_step_3(symbol, pipeline_logger):
                return None
        
        # Step 4: Predict & Evaluate
        if 4 in steps:
            results = run_step_4(symbol, evaluate_trend, pipeline_logger)
            if results is None:
                return None
        
        # Success message
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ PIPELINE COMPLETE: {symbol}")
        logger.info(f"{'='*80}")
        
        if results:
            logger.info(f"Best Model: {results['best_model']}")
            logger.info(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
            logger.info(f"Close MAPE: {results['close_mape']:.2f}%")
            logger.info(f"Close R²: {results['close_r2']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_batch(stock_list: list, steps: list = None, evaluate_trend: bool = False, output_dir: str = None):
    """
    Run pipeline for multiple stocks.
    
    Args:
        stock_list: List of stock symbols
        steps: List of steps to run (1-4). If None, runs all steps.
        evaluate_trend: Whether to calculate trend-neutral accuracy
        output_dir: Directory to save results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH PROCESSING: {len(stock_list)} STOCKS")
    logger.info(f"{'='*80}\n")
    
    if output_dir is None:
        output_dir = os.path.join(config.EVALUATION_DIR, 'multi_target')
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = []
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(stock_list, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROGRESS: {i}/{len(stock_list)} - {symbol}")
        logger.info(f"{'='*80}")
        
        results = run_single_stock(symbol, steps=steps, evaluate_trend=evaluate_trend)
        
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
        summary_file = os.path.join(output_dir, 'BATCH_SUMMARY.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.success(f"\nSaved batch summary to {summary_file}")
        
        # Display results
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Success: {success_count}/{len(stock_list)}")
        logger.info(f"Failed: {fail_count}/{len(stock_list)}")
        logger.info(f"\nTop 10 Performers:")
        logger.info(f"{'-'*80}")
        
        for idx, row in summary_df.head(10).iterrows():
            logger.info(f"{row['symbol']:15} - {row['best_model']:10} - {row['direction_accuracy']:.2f}% - MAPE: {row['close_mape']:.2f}%")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Average Direction Accuracy: {summary_df['direction_accuracy'].mean():.2f}%")
        logger.info(f"Average Close MAPE: {summary_df['close_mape'].mean():.2f}%")
        logger.info(f"Stocks > 70% Accuracy: {(summary_df['direction_accuracy'] > 70).sum()}/{len(summary_df)}")
        
        # Model selection breakdown
        logger.info(f"\nBest Model Distribution:")
        model_counts = summary_df['best_model'].value_counts()
        for model, count in model_counts.items():
            logger.info(f"  {model}: {count} stocks ({count/len(summary_df)*100:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Stock Prediction Pipeline - 4-Step Modular System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for single stock
  python main_pipeline.py --symbol RELIANCE
  
  # Run with trend evaluation
  python main_pipeline.py --symbol RELIANCE --evaluate-trend-bias
  
  # Run specific steps only
  python main_pipeline.py --symbol RELIANCE --step 2          # Feature engineering only
  python main_pipeline.py --symbol RELIANCE --steps 1 2       # Data + Features
  python main_pipeline.py --symbol RELIANCE --steps 3 4       # Train + Predict
  
  # Batch processing
  python main_pipeline.py --batch --all
  python main_pipeline.py --batch --stocks RELIANCE TCS INFY
  
Pipeline Steps:
  1. Data Collection     - Download OHLCV data
  2. Feature Engineering - Generate 244 features
  3. Train Models        - Train XGBoost, LSTM, GRU, Ensemble
  4. Predict & Evaluate  - Compare models and generate predictions
        """
    )
    
    # Single stock options
    parser.add_argument('--symbol', type=str, help='Single stock symbol to process')
    
    # Batch options
    parser.add_argument('--batch', action='store_true', help='Run batch processing')
    parser.add_argument('--all', action='store_true', help='Process all stocks (with --batch)')
    parser.add_argument('--stocks', nargs='+', help='List of stocks to process (with --batch)')
    
    # Step options
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], help='Run single step only')
    parser.add_argument('--steps', nargs='+', type=int, choices=[1, 2, 3, 4], help='Run specific steps')
    
    # Evaluation options
    parser.add_argument('--evaluate-trend-bias', action='store_true', 
                       help='Calculate trend-neutral accuracy (Step 4 only)')
    
    args = parser.parse_args()
    
    # Determine steps to run
    steps = None
    if args.step:
        steps = [args.step]
    elif args.steps:
        steps = args.steps
    # If neither specified, run all steps (steps = None will trigger default [1,2,3,4])
    
    # Single stock mode
    if args.symbol:
        run_single_stock(
            symbol=args.symbol,
            steps=steps,
            evaluate_trend=args.evaluate_trend_bias
        )
    
    # Batch mode
    elif args.batch:
        if args.all:
            # Get all stocks from raw data directory
            if os.path.exists(config.RAW_DATA_DIR):
                stock_list = [f.replace('.csv', '') for f in os.listdir(config.RAW_DATA_DIR) 
                            if f.endswith('.csv')]
            else:
                stock_list = config.TOP_100_STOCKS if hasattr(config, 'TOP_100_STOCKS') else []
        elif args.stocks:
            stock_list = args.stocks
        else:
            logger.error("Batch mode requires either --all or --stocks")
            parser.print_help()
            return
        
        run_batch(
            stock_list=stock_list,
            steps=steps,
            evaluate_trend=args.evaluate_trend_bias
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
