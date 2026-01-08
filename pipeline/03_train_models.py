"""
================================================================================
STEP 3: TRAIN MODELS
================================================================================
Trains 4 models on engineered features using rolling window validation.

Models:
1. XGBoost - Gradient boosting (fast, interpretable)
2. LSTM - Long Short-Term Memory (sequential patterns)
3. GRU - Gated Recurrent Unit (faster than LSTM)
4. Ensemble - Stacking all 3 models (best of all worlds)

Usage:
    # Single stock
    python pipeline/03_train_models.py --symbol RELIANCE
    
    # Multiple stocks
    python pipeline/03_train_models.py --symbols RELIANCE TCS INFY
    
    # All stocks
    python pipeline/03_train_models.py --all

Note: This step is combined with Step 4 (prediction) in the actual pipeline.
      Use main_pipeline.py for the complete workflow.

Output:
    - models/xgboost/{STOCK}_xgboost_model.pkl - Trained XGBoost model
    - models/lstm/{STOCK}_lstm_model.keras - Trained LSTM model
    - models/gru/{STOCK}_gru_model.keras - Trained GRU model
    - models/ensemble/{STOCK}_ensemble_model.pkl - Trained Ensemble model
    - models/scalers/{STOCK}_scaler.pkl - Feature scaler
    - logs/training_log.csv - Training metadata
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.rolling_window_validation import compare_models
from pipeline.utils.pipeline_logger import PipelineLogger


def train_models(symbol: str, pipeline_logger: PipelineLogger = None) -> bool:
    """
    Train all 4 models for a single stock.
    
    This is a wrapper around rolling_window_validation.compare_models()
    which handles all the training logic.
    
    Args:
        symbol: Stock symbol
        pipeline_logger: PipelineLogger instance for logging
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING MODELS: {symbol}")
    logger.info(f"{'='*80}")
    
    try:
        # Check if features exist
        features_file = os.path.join(config.FEATURES_DIR, f"{symbol}_features.csv")
        if not os.path.exists(features_file):
            logger.error(f"Features not found: {features_file}")
            logger.info(f"Run: python pipeline/02_feature_engineering.py --symbol {symbol}")
            return False
        
        # Train models using rolling window validation
        logger.info("Training 4 models with walk-forward validation...")
        logger.info("  - XGBoost (Gradient Boosting)")
        logger.info("  - LSTM (Long Short-Term Memory)")
        logger.info("  - GRU (Gated Recurrent Unit)")
        logger.info("  - Ensemble (Stacking all 3)")
        
        results = compare_models(symbol)
        
        if results is None:
            logger.error(f"Failed to train models for {symbol}")
            return False
        
        logger.success(f"\nAll 4 models trained successfully for {symbol}")
        logger.info(f"Best Model: {results['best_model']}")
        logger.info(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
        logger.info(f"Close MAPE: {results['close_mape']:.2f}%")
        logger.info(f"Close R²: {results['close_r2']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description='Step 3: Train models on engineered features')
    parser.add_argument('--symbol', type=str, help='Single stock symbol to train')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols to train')
    parser.add_argument('--all', action='store_true', help='Train all stocks from config')
    
    args = parser.parse_args()
    
    # Determine which stocks to process
    if args.symbol:
        stocks = [args.symbol]
    elif args.symbols:
        stocks = args.symbols
    elif args.all:
        # Get all stocks from features directory
        if os.path.exists(config.FEATURE_DATA_DIR):
            stocks = [f.replace('_features.csv', '') for f in os.listdir(config.FEATURE_DATA_DIR) 
                     if f.endswith('_features.csv')]
        else:
            logger.error(f"Features directory not found: {config.FEATURE_DATA_DIR}")
            return
    else:
        parser.print_help()
        return
    
    # Initialize logger
    pipeline_logger = PipelineLogger()
    
    # Train models for each stock
    logger.info(f"\nTraining models for {len(stocks)} stocks...")
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(stocks, 1):
        logger.info(f"\nProgress: {i}/{len(stocks)}")
        
        if train_models(symbol, pipeline_logger=pipeline_logger):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"MODEL TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {success_count}/{len(stocks)}")
    logger.info(f"Failed: {fail_count}/{len(stocks)}")
    logger.info(f"Models saved to: {config.MODEL_DIR}")
    logger.info(f"Logs saved to: {config.TRAINING_LOG}")


if __name__ == '__main__':
    main()
