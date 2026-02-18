"""
================================================================================
GOOGLE TRENDS TEST - Using Recent Data Only (2024-2026)
================================================================================
Since Google Trends has limited historical data, test on recent period only
where we have actual trends data.

This is a fairer test of the Google Trends hypothesis.
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add paths
v3_path = Path(__file__).parent.parent
sys.path.insert(0, str(v3_path))
sys.path.insert(0, str(v3_path / '02_models'))
sys.path.insert(0, str(v3_path / '01_data' / 'alternative'))

from traditional.lightgbm_classifier import LightGBMClassifier
from traditional.xgboost_classifier import XGBoostClassifier
from google_trends import GoogleTrendsCollector


def load_and_filter_recent_data(
    symbol: str = 'SBIN',
    timestamp: str = '20260202_154436',
    start_date: str = '2024-01-01'
):
    """Load data and filter to recent period only."""
    # Load data
    data_path = v3_path / 'data' / 'features' / timestamp / f'{symbol}_features.csv'
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to recent period
    df_recent = df[df['date'] >= start_date].copy()
    
    print(f"[DATA] Full dataset: {len(df)} rows ({df['date'].min()} to {df['date'].max()})")
    print(f"[DATA] Recent subset: {len(df_recent)} rows ({df_recent['date'].min()} to {df_recent['date'].max()})")
    
    return df_recent


def prepare_features_and_target(df, include_trends=True):
    """Prepare features for training."""
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'date',
        'target_close_return', 'target_direction',
        'target_high', 'target_low', 'next_day_log_return'
    ]
    
    if not include_trends:
        exclude_cols += [col for col in df.columns if col.startswith('trends_')]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    
    # Create target
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    X = X[:-1]
    y = df['target'].values[:-1]
    
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols


def simple_train_test(X, y, feature_names, model_class, model_name, train_ratio=0.7):
    """Simple train/test split for quick evaluation."""
    from sklearn.metrics import accuracy_score, f1_score
    
    n = len(X)
    train_end = int(n * train_ratio)
    
    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    
    # Validation from training
    val_size = int(len(X_train) * 0.15)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_only = X_train[:-val_size]
    y_train_only = y_train[:-val_size]
    
    # Train
    model = model_class()
    
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model.train(
                X_train_only, y_train_only,
                X_val, y_val,
                feature_names=feature_names,
                verbose=False
            )
    
    # Predict
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'n_train': len(X_train_only),
        'n_test': len(X_test)
    }


def main():
    """Test Google Trends on recent data only."""
    print("="*70)
    print(" GOOGLE TRENDS TEST - RECENT DATA ONLY (2024-2026)")
    print("="*70)
    print(" Hypothesis: Google Trends helps when we have REAL trends data")
    print(" (not forward-filled placeholder values)")
    print("="*70)
    
    # Initialize
    collector = GoogleTrendsCollector()
    
    symbols = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'TCS', 'INFY']
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f" {symbol}")
        print(f"{'='*50}")
        
        try:
            # Load recent data only
            df = load_and_filter_recent_data(symbol, start_date='2024-01-01')
            
            if len(df) < 200:
                print(f"[SKIP] Not enough data: {len(df)} rows")
                continue
            
            # Get Google Trends
            trends = collector.get_interest_over_time(
                symbol=symbol,
                start_date='2024-01-01',
                end_date='2026-02-18'
            )
            
            print(f"[TRENDS] Got {len(trends)} trends data points")
            
            # Merge
            df_enhanced = collector.merge_with_price_data(df, trends)
            
            # Test WITHOUT trends (baseline)
            print("\n[BASELINE] Without trends:")
            X_base, y_base, features_base = prepare_features_and_target(df_enhanced, include_trends=False)
            print(f"  Features: {len(features_base)}, Samples: {len(X_base)}")
            
            for model_class, model_name in [(LightGBMClassifier, 'LightGBM'), (XGBoostClassifier, 'XGBoost')]:
                result = simple_train_test(X_base, y_base, features_base, model_class, model_name)
                result['symbol'] = symbol
                result['with_trends'] = False
                all_results.append(result)
                print(f"  {model_name}: {result['accuracy']:.2%} accuracy")
            
            # Test WITH trends
            print("\n[ENHANCED] With trends:")
            X_enh, y_enh, features_enh = prepare_features_and_target(df_enhanced, include_trends=True)
            print(f"  Features: {len(features_enh)}, Samples: {len(X_enh)}")
            
            for model_class, model_name in [(LightGBMClassifier, 'LightGBM'), (XGBoostClassifier, 'XGBoost')]:
                result = simple_train_test(X_enh, y_enh, features_enh, model_class, model_name)
                result['symbol'] = symbol
                result['with_trends'] = True
                all_results.append(result)
                print(f"  {model_name}: {result['accuracy']:.2%} accuracy")
            
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY: GOOGLE TRENDS IMPACT ON RECENT DATA")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    for model_name in ['LightGBM', 'XGBoost']:
        baseline = results_df[(results_df['model'] == model_name) & (~results_df['with_trends'])]['accuracy'].mean()
        enhanced = results_df[(results_df['model'] == model_name) & (results_df['with_trends'])]['accuracy'].mean()
        
        diff = enhanced - baseline
        print(f"\n{model_name}:")
        print(f"  Baseline avg: {baseline:.2%}")
        print(f"  Enhanced avg: {enhanced:.2%}")
        print(f"  Improvement:  {diff:+.2%} ({diff*100:+.1f} pp)")
        
        status = "✅ IMPROVED" if diff > 0.01 else ("⚠️ MARGINAL" if diff > 0 else "❌ NO GAIN")
        print(f"  Status: {status}")
    
    # Per-stock breakdown
    print("\n" + "-"*70)
    print(" Per-Stock Results:")
    print("-"*70)
    
    for symbol in results_df['symbol'].unique():
        sym_data = results_df[results_df['symbol'] == symbol]
        
        for model in ['LightGBM']:
            base = sym_data[(sym_data['model'] == model) & (~sym_data['with_trends'])]['accuracy'].values[0]
            enh = sym_data[(sym_data['model'] == model) & (sym_data['with_trends'])]['accuracy'].values[0]
            diff = enh - base
            
            trend_icon = "📈" if diff > 0.01 else ("➡️" if diff > -0.01 else "📉")
            print(f"  {symbol} ({model}): {base:.1%} → {enh:.1%} ({diff:+.1%}) {trend_icon}")
    
    # Save results
    results_path = v3_path / 'results' / 'google_trends_recent_data.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVED] {results_path}")
    
    return results_df


if __name__ == '__main__':
    main()
