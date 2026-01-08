"""
================================================================================
STEP 2: FEATURE ENGINEERING
================================================================================
Generates 244 professional features from raw stock data.

Features:
- 60+ Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
- 20+ Market Features (NIFTY, BANKNIFTY correlation)
- 15+ Temporal Features (day/month/quarter encoding)
- 25+ Regime Features (bull/bear/ranging detection)
- 15+ Statistical Features (Hurst, autocorrelation)
- 10+ Liquidity Features (Kyle's lambda, Amihud)
- 20+ Volume Features (VWAP, volume oscillators)
- 15+ Price Level Features (pivot points, support/resistance)
- 10+ Cycle Features (dominant cycle detection)
- 30+ Interaction Features (price × volume, momentum × volatility)
- 7 Sentiment Features (placeholder for news sentiment)
- 17 Lag Features (price/return lags)

Usage:
    # Single stock
    python pipeline/02_feature_engineering.py --symbol RELIANCE
    
    # Multiple stocks
    python pipeline/02_feature_engineering.py --symbols RELIANCE TCS INFY
    
    # All stocks
    python pipeline/02_feature_engineering.py --all

Output:
    - data/features/{STOCK}_features.csv - 244 features per stock
    - data/features/{STOCK}_feature_list.txt - List of feature names
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
from pipeline.multi_target_prediction_system import MultiTargetFeatureEngineering
from pipeline.utils.pipeline_logger import PipelineLogger


def engineer_features(symbol: str, pipeline_logger: PipelineLogger = None) -> bool:
    """
    Generate features for a single stock.
    
    Args:
        symbol: Stock symbol
        pipeline_logger: PipelineLogger instance for logging
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"FEATURE ENGINEERING: {symbol}")
    logger.info(f"{'='*80}")
    
    try:
        # Check if raw data exists
        raw_file = os.path.join(config.RAW_DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(raw_file):
            logger.error(f"Raw data not found: {raw_file}")
            logger.info(f"Run: python pipeline/01_data_collection.py --symbol {symbol}")
            return False
        
        # Initialize prediction system
        logger.info(f"Loading raw data from {raw_file}")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = MultiTargetFeatureEngineering.load_stock_data(symbol)
        
        if df is None or df.empty:
            logger.error(f"Failed to load data for {symbol}")
            return False
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Create features
        logger.info("Creating 244 professional features...")
        logger.info("  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)")
        logger.info("  - Market features (NIFTY, BANKNIFTY correlation)")
        logger.info("  - Temporal features (day/month/quarter encoding)")
        logger.info("  - Regime features (bull/bear/ranging detection)")
        logger.info("  - Statistical features (Hurst, autocorrelation)")
        logger.info("  - Liquidity features (Kyle's lambda, Amihud)")
        logger.info("  - Volume features (VWAP, volume oscillators)")
        logger.info("  - Price level features (pivot points)")
        logger.info("  - Cycle features (dominant cycle detection)")
        logger.info("  - Interaction features (price × volume)")
        logger.info("  - Sentiment features (placeholder)")
        logger.info("  - Lag features (price/return lags)")
        
        df = MultiTargetFeatureEngineering.create_all_features(df, symbol)
        
        if df is None or df.empty:
            logger.error(f"Failed to create features for {symbol}")
            return False
        
        # Count features
        feature_cols = MultiTargetFeatureEngineering.get_feature_columns(df)
        num_features = len(feature_cols)
        
        logger.info(f"Created {num_features} features")
        
        # Save features
        output_dir = config.FEATURES_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{symbol}_features.csv")
        df.to_csv(output_file, index=False)
        logger.success(f"Saved features to {output_file}")
        
        # Save feature list
        feature_list_file = os.path.join(output_dir, f"{symbol}_feature_list.txt")
        with open(feature_list_file, 'w') as f:
            f.write(f"Feature List for {symbol}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total Features: {num_features}\n")
            f.write(f"{'='*80}\n\n")
            
            # Group features by category
            categories = {
                'Technical Indicators': [col for col in feature_cols if any(x in col.lower() for x in ['rsi', 'macd', 'ema', 'sma', 'bb', 'atr', 'adx', 'cci', 'willr', 'mfi', 'stoch', 'roc'])],
                'Market Features': [col for col in feature_cols if any(x in col.lower() for x in ['nifty', 'banknifty', 'vix', 'market'])],
                'Temporal Features': [col for col in feature_cols if any(x in col.lower() for x in ['day', 'month', 'quarter', 'year', 'sin', 'cos'])],
                'Regime Features': [col for col in feature_cols if any(x in col.lower() for x in ['regime', 'bull', 'bear', 'ranging', 'volatility_regime'])],
                'Volume Features': [col for col in feature_cols if any(x in col.lower() for x in ['volume', 'vwap', 'obv', 'cmf', 'mfi'])],
                'Statistical Features': [col for col in feature_cols if any(x in col.lower() for x in ['hurst', 'autocorr', 'skew', 'kurt'])],
                'Liquidity Features': [col for col in feature_cols if any(x in col.lower() for x in ['kyle', 'amihud', 'turnover', 'spread'])],
                'Price Level Features': [col for col in feature_cols if any(x in col.lower() for x in ['pivot', 'support', 'resistance', 'round'])],
                'Cycle Features': [col for col in feature_cols if any(x in col.lower() for x in ['cycle', 'detrend', 'swing'])],
                'Interaction Features': [col for col in feature_cols if '_x_' in col.lower() or 'interaction' in col.lower()],
                'Sentiment Features': [col for col in feature_cols if 'sentiment' in col.lower()],
                'Lag Features': [col for col in feature_cols if 'lag' in col.lower() or 'prev' in col.lower()]
            }
            
            for category, features in categories.items():
                if features:
                    f.write(f"\n{category} ({len(features)}):\n")
                    f.write('-' * 80 + '\n')
                    for feat in sorted(features):
                        f.write(f"  - {feat}\n")
            
            # Uncategorized features
            categorized = set()
            for features in categories.values():
                categorized.update(features)
            uncategorized = [col for col in feature_cols if col not in categorized]
            
            if uncategorized:
                f.write(f"\nOther Features ({len(uncategorized)}):\n")
                f.write('-' * 80 + '\n')
                for feat in sorted(uncategorized):
                    f.write(f"  - {feat}\n")
        
        logger.success(f"Saved feature list to {feature_list_file}")
        
        # Log to pipeline logger
        if pipeline_logger:
            pipeline_logger.log_feature_engineering(
                symbol=symbol,
                features=num_features,
                rows=len(df)
            )
        
        # Display sample
        logger.info(f"\nSample features (first 5 rows):")
        logger.info(f"\n{df[feature_cols[:10]].head()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error engineering features for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point for feature engineering."""
    parser = argparse.ArgumentParser(description='Step 2: Engineer features from raw data')
    parser.add_argument('--symbol', type=str, help='Single stock symbol to process')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols to process')
    parser.add_argument('--all', action='store_true', help='Process all stocks from config')
    
    args = parser.parse_args()
    
    # Determine which stocks to process
    if args.symbol:
        stocks = [args.symbol]
    elif args.symbols:
        stocks = args.symbols
    elif args.all:
        # Get all stocks from raw data directory
        if os.path.exists(config.RAW_DATA_DIR):
            stocks = [f.replace('.csv', '') for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.csv')]
        else:
            logger.error(f"Raw data directory not found: {config.RAW_DATA_DIR}")
            return
    else:
        parser.print_help()
        return
    
    # Initialize logger
    pipeline_logger = PipelineLogger()
    
    # Process each stock
    logger.info(f"\nEngineering features for {len(stocks)} stocks...")
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(stocks, 1):
        logger.info(f"\nProgress: {i}/{len(stocks)}")
        
        if engineer_features(symbol, pipeline_logger=pipeline_logger):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"FEATURE ENGINEERING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {success_count}/{len(stocks)}")
    logger.info(f"Failed: {fail_count}/{len(stocks)}")
    logger.info(f"Features saved to: {config.FEATURE_DATA_DIR}")


if __name__ == '__main__':
    main()
