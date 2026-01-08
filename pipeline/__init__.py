"""
AI Stock Prediction Pipeline
============================
Multi-target deep learning for NSE stock prediction.

Pipeline Steps:
    01_data_collection.py     - Download OHLCV data from NSE
    02_feature_engineering.py - Generate 244 professional features
    03_train_models.py        - Train XGBoost, LSTM, GRU, Ensemble
    04_predict.py             - Predict & evaluate with trend-neutral analysis

Usage:
    # Complete pipeline
    python main_pipeline.py --symbol TCS
    
    # Individual steps
    python pipeline/01_data_collection.py --symbol TCS
    python pipeline/02_feature_engineering.py --symbol TCS
    python pipeline/03_train_models.py --symbol TCS
    python pipeline/04_predict.py --symbol TCS
"""

__version__ = '2.0.0'
__author__ = 'AI Stock Prediction System'

# Import step functions directly - files are named with prefix numbers
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import using importlib to handle filenames with numbers
import importlib.util

def _import_from_file(filepath, func_name):
    """Import a function from a file with numeric prefix."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

pipeline_dir = os.path.dirname(__file__)

collect_stock = _import_from_file(
    os.path.join(pipeline_dir, '01_data_collection.py'),
    'collect_stock'
)

engineer_features = _import_from_file(
    os.path.join(pipeline_dir, '02_feature_engineering.py'),
    'engineer_features'
)

train_models = _import_from_file(
    os.path.join(pipeline_dir, '03_train_models.py'),
    'train_models'
)

predict_and_evaluate = _import_from_file(
    os.path.join(pipeline_dir, '04_predict.py'),
    'predict_and_evaluate'
)

__all__ = [
    'collect_stock',
    'engineer_features',
    'train_models',
    'predict_and_evaluate'
]
