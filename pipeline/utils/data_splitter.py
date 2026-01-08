"""
================================================================================
DATA SPLITTER UTILITY
================================================================================
Splits stock data into train/validation/test sets based on date ranges.

Train:      2015-01-01 to 2021-12-31 (7 years)   → 70%
Validation: 2022-01-01 to 2023-12-31 (2 years)   → 20%
Test:       2024-01-01 to present    (1 year)    → 10%

Usage:
    from pipeline.utils.data_splitter import split_data, load_split_data

    # Split a dataframe
    train_df, val_df, test_df = split_data(df)

    # Or load pre-split data for a symbol
    train, val, test = load_split_data('TCS')
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def split_data(df: pd.DataFrame,
               train_end: str = '2021-12-31',
               val_end: str = '2023-12-31',
               date_col: str = 'timestamp',
               rolling: bool = True,
               train_years: int = 5,
               val_years: int = 1) -> tuple:
    """
    Split dataframe into train/validation/test sets by date.
    
    Rolling mode (recommended for production):
        - Uses last N years of data dynamically
        - Train: last 5 years (default)
        - Val: 1 year before test period
        - Test: most recent data after val
    
    Fixed mode (for backtesting):
        - Uses fixed date ranges
        - Train: up to train_end
        - Val: train_end to val_end
        - Test: after val_end

    Args:
        df: DataFrame with date column
        train_end: End date for training data (fixed mode only)
        val_end: End date for validation data (fixed mode only)
        date_col: Name of date column
        rolling: If True, use rolling window; if False, use fixed dates
        train_years: Number of years for training (rolling mode)
        val_years: Number of years for validation (rolling mode)

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if rolling:
        # Rolling window mode
        max_date = df[date_col].max()
        
        # Test: most recent ~6 months of data
        test_start = max_date - pd.DateOffset(months=6)
        
        # Val: 1 year before test
        val_end_dt = test_start - pd.Timedelta(days=1)
        val_start = val_end_dt - pd.DateOffset(years=val_years)
        
        # Train: N years before val
        train_end_dt = val_start - pd.Timedelta(days=1)
        train_start = train_end_dt - pd.DateOffset(years=train_years)
        
        train_df = df[(df[date_col] >= train_start) & (df[date_col] <= train_end_dt)]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] <= val_end_dt)]
        test_df = df[df[date_col] >= test_start]
        
        logger.info(f"Rolling split: Train {train_start.date()} to {train_end_dt.date()}, "
                   f"Val {val_start.date()} to {val_end_dt.date()}, "
                   f"Test {test_start.date()} to {max_date.date()}")
    else:
        # Fixed date mode (original behavior)
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)

        train_df = df[df[date_col] <= train_end_dt]
        val_df = df[(df[date_col] > train_end_dt) & (df[date_col] <= val_end_dt)]
        test_df = df[df[date_col] > val_end_dt]

    return train_df, val_df, test_df


def get_split_info(df: pd.DataFrame,
                   train_end: str = '2021-12-31',
                   val_end: str = '2023-12-31',
                   date_col: str = 'timestamp') -> dict:
    """
    Get information about the data split.

    Returns:
        dict with split statistics
    """
    train_df, val_df, test_df = split_data(df, train_end, val_end, date_col)

    total = len(df)

    info = {
        'total_rows': total,
        'train': {
            'rows': len(train_df),
            'pct': len(train_df) / total * 100 if total > 0 else 0,
            'start': train_df[date_col].min() if len(train_df) > 0 else None,
            'end': train_df[date_col].max() if len(train_df) > 0 else None,
        },
        'val': {
            'rows': len(val_df),
            'pct': len(val_df) / total * 100 if total > 0 else 0,
            'start': val_df[date_col].min() if len(val_df) > 0 else None,
            'end': val_df[date_col].max() if len(val_df) > 0 else None,
        },
        'test': {
            'rows': len(test_df),
            'pct': len(test_df) / total * 100 if total > 0 else 0,
            'start': test_df[date_col].min() if len(test_df) > 0 else None,
            'end': test_df[date_col].max() if len(test_df) > 0 else None,
        }
    }

    return info


def load_split_data(symbol: str,
                    data_type: str = 'features',
                    train_end: str = '2021-12-31',
                    val_end: str = '2023-12-31') -> tuple:
    """
    Load and split data for a symbol.

    Args:
        symbol: Stock symbol
        data_type: 'raw' or 'features'
        train_end: End date for training
        val_end: End date for validation

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if data_type == 'raw':
        file_path = os.path.join(config.RAW_DATA_DIR, f"{symbol}.csv")
    else:
        file_path = os.path.join(config.FEATURES_DIR, f"{symbol}_features.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found: {file_path}")

    df = pd.read_csv(file_path)
    return split_data(df, train_end, val_end)


def create_sequences(df: pd.DataFrame,
                     sequence_length: int = 60,
                     feature_cols: list = None,
                     target_col: str = 'target') -> tuple:
    """
    Create sequences for LSTM/Transformer models.

    Args:
        df: DataFrame with features and target
        sequence_length: Number of time steps in sequence
        feature_cols: List of feature columns (if None, infer from df)
        target_col: Target column name

    Returns:
        tuple: (X, y) as numpy arrays
            X shape: (n_samples, sequence_length, n_features)
            y shape: (n_samples,)
    """
    if feature_cols is None:
        # Infer feature columns
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'symbol', 'target', 'next_close']
        feature_cols = [col for col in df.columns if col not in exclude]

    data = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def verify_data_quality(df: pd.DataFrame, date_col: str = 'timestamp') -> dict:
    """
    Verify data quality for a stock dataframe.

    Returns:
        dict with quality metrics
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    quality = {
        'total_rows': len(df),
        'date_range': {
            'start': df[date_col].min(),
            'end': df[date_col].max(),
            'years': (df[date_col].max() - df[date_col].min()).days / 365
        },
        'missing_values': df.isnull().sum().sum(),
        'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
        'columns': list(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
    }

    # Check for price columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            quality[f'{col}_valid'] = df[col].notna().all()

    return quality


def print_split_summary(symbols: list = None):
    """Print summary of data splits for all or specified symbols."""
    if symbols is None:
        symbols = config.ALL_STOCKS

    print("=" * 80)
    print("DATA SPLIT SUMMARY")
    print("Train: 2015-01-01 to 2021-12-31 | Val: 2022-01-01 to 2023-12-31 | Test: 2024-01-01+")
    print("=" * 80)
    print(f"{'Symbol':<12} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8} {'Train%':>7} {'Val%':>6} {'Test%':>6}")
    print("-" * 80)

    total_train = total_val = total_test = 0

    for symbol in symbols:
        file_path = os.path.join(config.RAW_DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path):
            print(f"{symbol:<12} {'NOT FOUND':>8}")
            continue

        try:
            df = pd.read_csv(file_path)
            info = get_split_info(df)

            total_train += info['train']['rows']
            total_val += info['val']['rows']
            total_test += info['test']['rows']

            print(f"{symbol:<12} {info['total_rows']:>8} {info['train']['rows']:>8} "
                  f"{info['val']['rows']:>8} {info['test']['rows']:>8} "
                  f"{info['train']['pct']:>6.1f}% {info['val']['pct']:>5.1f}% "
                  f"{info['test']['pct']:>5.1f}%")
        except Exception as e:
            print(f"{symbol:<12} {'ERROR':>8} - {str(e)[:30]}")

    print("-" * 80)
    grand_total = total_train + total_val + total_test
    print(f"{'TOTAL':<12} {grand_total:>8} {total_train:>8} {total_val:>8} {total_test:>8} "
          f"{total_train/grand_total*100:>6.1f}% {total_val/grand_total*100:>5.1f}% "
          f"{total_test/grand_total*100:>5.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data Split Utility')
    parser.add_argument('--summary', action='store_true', help='Print split summary')
    parser.add_argument('--symbol', type=str, help='Show details for specific symbol')
    args = parser.parse_args()

    if args.summary:
        print_split_summary()
    elif args.symbol:
        file_path = os.path.join(config.RAW_DATA_DIR, f"{args.symbol.upper()}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            info = get_split_info(df)
            quality = verify_data_quality(df)

            print(f"\n{args.symbol.upper()} Data Analysis")
            print("=" * 50)
            print(f"Total rows: {info['total_rows']}")
            print(f"Date range: {quality['date_range']['start']} to {quality['date_range']['end']}")
            print(f"Years: {quality['date_range']['years']:.1f}")
            print("\nSplit Details:")
            print(f"  Train: {info['train']['rows']} rows ({info['train']['pct']:.1f}%)")
            print(f"         {info['train']['start']} to {info['train']['end']}")
            print(f"  Val:   {info['val']['rows']} rows ({info['val']['pct']:.1f}%)")
            print(f"         {info['val']['start']} to {info['val']['end']}")
            print(f"  Test:  {info['test']['rows']} rows ({info['test']['pct']:.1f}%)")
            print(f"         {info['test']['start']} to {info['test']['end']}")
        else:
            print(f"File not found: {file_path}")
    else:
        print_split_summary()
