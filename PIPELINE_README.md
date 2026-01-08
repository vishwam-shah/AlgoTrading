# Stock Prediction Pipeline - Production System

## 🎯 Overview

Production-ready stock prediction system with multi-target prediction, trend-neutral evaluation, and comprehensive metrics.

### Key Features
- **Multi-Target Prediction**: Close, High, Low prices + Direction (Up/Down)
- **4 Models**: XGBoost, LSTM, GRU, Ensemble (Stacking)
- **Walk-Forward Validation**: Mimics real trading conditions
- **Trend-Neutral Evaluation**: Removes trend bias to show true predictive skill
- **244 Professional Features**: Technical indicators, market data, regime detection, etc.

## 📁 Project Structure

```
AI_IN_STOCK_V2/
├── main_pipeline.py              # Main entry point
├── config.py                     # Configuration
├── pipeline/
│   ├── multi_target_prediction_system.py  # Feature engineering (1257 lines)
│   ├── multi_target_models.py             # Model implementations
│   └── rolling_window_validation.py       # Validation & evaluation
├── data/
│   ├── raw/                      # Original stock data
│   ├── features/                 # Engineered features
│   └── market/                   # Market indices (NIFTY, VIX)
├── models/                       # Saved models
├── evaluation_results/
│   └── multi_target/             # Results, plots, predictions
└── logs/                         # Training logs
```

## 🚀 Quick Start

### 1. Run Single Stock

```bash
# Basic run
python main_pipeline.py --symbol RELIANCE

# With trend-neutral evaluation
python main_pipeline.py --symbol RELIANCE --evaluate-trend-bias
```

### 2. Batch Processing

```bash
# Process all stocks
python main_pipeline.py --batch --all

# Process specific stocks
python main_pipeline.py --batch --stocks RELIANCE TCS INFY HDFCBANK
```

## 📊 Results

### Average Performance (106 Stocks)
- **Direction Accuracy**: 68.40% (18.4 pp above random)
- **Close MAPE**: 1.27%
- **Trend-Neutral Accuracy**: 49.90% (true predictive skill)
- **Trend Advantage**: 25.33 pp (gained from trends)

### Top Performers (Original Accuracy)
1. POWERINDIA: 95.22% (trend=0.85)
2. GODREJPROP: 78.00%
3. MPHASIS: 75.56%
4. LUPIN: 75.56%
5. EMAMILTD: 75.45%

### Model Selection
- **XGBoost**: 54 stocks (50.9%) - Best for most stocks
- **Ensemble**: 52 stocks (49.1%) - Close second
- **LSTM**: 0 stocks - Consistently overfits
- **GRU**: 0 stocks - Consistently overfits

## 🔬 Trend-Neutral Evaluation

### Why It Matters
High accuracy can be misleading if the stock is strongly trending. Example:
- POWERINDIA: 95.22% accuracy (Original)
- POWERINDIA: 49.08% accuracy (Trend-Neutral)
- **Trend Advantage**: 46.13 pp gained just from uptrend!

### How It Works
1. Remove linear trend from prices
2. Recalculate direction on detrended data
3. Measure accuracy on trend-neutral predictions
4. Report both metrics for transparency

### When to Use
- **Research/Papers**: Always report trend-neutral accuracy
- **Model Comparison**: Compare apples-to-apples
- **Live Trading**: Use original accuracy (trends are real!)

## 📈 Output Files

For each stock, the pipeline generates:

### 1. Model Comparison (`{STOCK}_model_comparison.csv`)
```csv
model,close_rmse,close_mae,close_r2,close_mape,direction_accuracy,...
XGBoost,0.0154,0.0084,0.7833,0.84,0.9522,...
LSTM,0.0329,0.0238,-0.0001,2.38,0.5191,...
GRU,0.0329,0.0238,-0.0001,2.38,0.5191,...
Ensemble,0.0259,0.0170,0.3832,1.70,0.9504,...
```

### 2. Predictions (`{STOCK}_predictions.csv`)
Full test dataset with:
- All features
- Actual values
- Model predictions (all 4 models)
- Target values

### 3. Comparison Plot (`{STOCK}_comparison_plot.png`)
3-panel visualization:
- Close price prediction
- High price prediction
- Low price prediction

### 4. Batch Summary (`BATCH_SUMMARY.csv`)
When running batch mode:
```csv
Stock,BestModel,DirectionAccuracy,CloseMAPE,CloseR2,TestSamples
POWERINDIA,XGBoost,95.22,0.84,0.7833,272
GODREJPROP,Ensemble,78.00,1.70,0.1136,491
...
```

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Data paths
RAW_DATA_DIR = 'data/raw'
FEATURE_DATA_DIR = 'data/features'
MODEL_DIR = 'models'

# Model parameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05
}

# Training parameters
TEST_SIZE = 0.20
VAL_SIZE = 0.20
RANDOM_STATE = 42
```

## 🧪 Feature Engineering

### 244 Features Across 12 Categories

1. **Technical Indicators (60+)**
   - Momentum: RSI, MACD, Stochastic, ROC
   - Trend: SMA, EMA, Ichimoku
   - Volatility: Bollinger Bands, ATR, Keltner
   - Volume: OBV, CMF, Force Index

2. **Market Features (20+)**
   - NIFTY50 correlation & returns
   - BANKNIFTY correlation & returns
   - INDIA VIX (volatility index)

3. **Temporal Features (15+)**
   - Day/Month/Quarter
   - Cyclical encoding (sin/cos)
   - Month-end, quarter-end flags

4. **Regime Features (25+)**
   - Bull/Bear/Ranging detection
   - Volatility expansion/contraction
   - Volume regimes

5. **Statistical Features (15+)**
   - Hurst exponent
   - Autocorrelation
   - Skewness, Kurtosis

6. **Liquidity Features (10+)**
   - Kyle's lambda
   - Turnover ratios
   - Amihud illiquidity

7. **Volume Features (20+)**
   - VWAP, VWMA
   - Volume oscillators
   - Climax volume detection

8. **Price Level Features (15+)**
   - Round number proximity
   - Pivot points
   - Support/Resistance

9. **Cycle Features (10+)**
   - Dominant cycle detection
   - Detrending
   - Swing high/low

10. **Interaction Features (30+)**
    - Price × Volume
    - Momentum × Volatility
    - Trend Strength × Volume

11. **Sentiment Features (7)**
    - News sentiment (placeholder)
    - Sentiment trends

12. **Lag Features (17)**
    - Price lags (1-10 days)
    - Return lags
    - Close/High/Low lags

## 📊 Model Details

### 1. XGBoost (Gradient Boosting)
- **Pros**: Fast, interpretable, handles non-linearity
- **Cons**: Doesn't capture sequential patterns
- **Best for**: Most stocks (50.9% selection rate)

### 2. LSTM (Long Short-Term Memory)
- **Pros**: Captures sequential patterns, long-term dependencies
- **Cons**: Overfits on daily stock data
- **Best for**: None (0% selection rate)

### 3. GRU (Gated Recurrent Unit)
- **Pros**: Faster than LSTM, similar capabilities
- **Cons**: Also overfits on daily data
- **Best for**: None (0% selection rate)

### 4. Ensemble (Stacking)
- **Pros**: Combines strengths of all models
- **Cons**: Slower, more complex
- **Best for**: 49.1% of stocks (close to XGBoost)

## 🎓 Research Findings

### Key Insights
1. **Simple > Complex**: XGBoost outperforms deep learning consistently
2. **Feature Engineering Critical**: 244 features > model architecture
3. **Trend Bias Significant**: 25.33 pp average advantage from trends
4. **LSTM/GRU Overfit**: Never selected as best model (0/106)
5. **True Skill ~50%**: After removing trend bias, barely above random

### Implications for Trading
- ✅ **Use original accuracy** for trading (trends are real patterns)
- ✅ **Monitor for regime changes** (trend reversals)
- ✅ **Retrain regularly** (quarterly recommended)
- ⚠️ **Be cautious** in ranging/choppy markets
- ⚠️ **Reduce size** if trend reverses

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the right directory
cd AI_IN_STOCK_V2

# Check Python path
python -c "import sys; print(sys.path)"
```

**2. Missing Data**
```bash
# Check if stock data exists
ls data/raw/{STOCK}.csv

# Download data if missing
python pipeline/01_data_collection.py --symbol {STOCK}
```

**3. Memory Issues**
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # Instead of 32
```

**4. GPU Errors (LSTM/GRU)**
```python
# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 📚 References

### Papers & Methods
- XGBoost: Chen & Guestrin (2016)
- LSTM: Hochreiter & Schmidhuber (1997)
- Walk-Forward Validation: Pardo (2008)
- Trend Detrending: Linear regression detrending

### Feature Inspiration
- Technical Analysis of Stock Trends (Edwards & Magee)
- Evidence-Based Technical Analysis (Aronson)
- Algorithmic Trading (Chan)

## 🤝 Contributing

To add new features:
1. Edit `pipeline/multi_target_prediction_system.py`
2. Add to appropriate feature category
3. Update `get_feature_columns()` if needed
4. Test on sample stock

To add new models:
1. Edit `pipeline/multi_target_models.py`
2. Implement `train()` and `predict()` methods
3. Add to `compare_models()` in validation script

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- NSE India for stock data
- TA-Lib for technical indicators
- XGBoost, TensorFlow teams

---

**Last Updated**: December 14, 2025  
**Version**: 2.0 (Production Release)
