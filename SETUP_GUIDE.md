# V3 Pipeline Setup & TensorFlow Guide

## ✅ Current Setup Status

**Python Version**: 3.14  
**Environment**: Virtual environment with compatible packages  
**ML Models**: Scikit-Learn based (no libomp/TensorFlow issues)

### Installed Packages
```
✓ numpy 2.4.4       — Numerical computing
✓ pandas 3.0.2      — Data manipulation
✓ scikit-learn 1.8  — ML models (GradientBoosting, RandomForest)
✓ yfinance 1.0      — Stock data download
✓ ta 0.11.0         — Technical indicators
✓ loguru 0.7.0      — Logging
```

---

## ❓ TensorFlow Support for Python 3.14

### Current Status
**TensorFlow 2.18 does NOT support Python 3.14** (Python 3.14 is very new)

### TensorFlow Compatibility Matrix
| Python | TensorFlow | Status |
|--------|-----------|--------|
| 3.11 | 2.16-2.18 | ✓ Supported |
| 3.12 | 2.16-2.18 | ✓ Supported |
| 3.13 | 2.17-2.18 | ✓ Supported |
| 3.14 | - | ✗ Not yet (expected mid-2026) |

**Expected Timeline**: TensorFlow 2.19+ or 3.0 will likely support Python 3.14

---

## 🔄 Alternatives to TensorFlow for Deep Learning

Since TensorFlow isn't available for Python 3.14, here are your options:

### Option 1: **Use Python 3.13 Instead** (Recommended for TensorFlow)
```bash
# Switch to Python 3.13
python3.13 -m venv venv
source venv/bin/activate
pip install tensorflow==2.18 -r requirements.txt
```
✅ **Pros**: Full TensorFlow support, LSTM/Transformer models  
❌ **Cons**: Need to switch Python versions

### Option 2: **PyTorch** (Alternative to TensorFlow)
Works with Python 3.14. More flexible for research.
```bash
pip install torch torchvision torchaudio
# Then install PyTorch models: LSTM, Transformer, etc.
```

### Option 3: **Current Setup** (Scikit-Learn Only)
What we're using now - no deep learning, only traditional ML.
```bash
✓ GradientBoosting (CPU-optimized)
✓ RandomForest (Parallel)
✓ Ensemble voting
```
✅ **Pros**: No compilation issues, fast training, stable  
❌ **Cons**: No neural networks (but XGBoost/RF often outperform NNs on tabular data)

---

## 📊 Pipeline Outputs Per Stock

### Each Training Run Generates

#### 1. **CSV Files**
```
results_detailed.csv  — All stocks with metrics
{symbol}_predictions.csv — Detailed predictions for stock
```

**Example {SBIN}_predictions.csv:**
```
timestamp,y_true,y_pred,y_pred_proba,log_return
2025-01-15,1,1,0.652,-0.001
2025-01-16,0,1,0.513,0.005
2025-01-17,1,1,0.621,-0.002
...
```

#### 2. **Metrics per Stock**
For each stock, we calculate:

| Metric | Definition | Interpretation |
|--------|-----------|---|
| **Directional Accuracy** | % correct up/down predictions | >50% = better than random |
| **Win Rate** | % wins on high-confidence trades | >40% is good |
| **Profit Factor** | Winning $ / Losing $ | >1.5 = profitable |
| **Precision (Bullish)** | % of bullish signals correct | Quality of long signals |
| **Recall (Bullish)** | % of true bulls caught | Coverage of rallies |

#### 3. **Reports**
```
REPORT.md          — Markdown summary (human readable)
results_detailed.csv  — All metrics sortable
metadata.json      — Run metadata + summary stats
```

#### 4. **Example Output Structure**
```
V3/06_results/runs/20260408_140456/
├── REPORT.md                    ← Human-readable summary
├── results_detailed.csv         ← All stocks ranked
├── metadata.json                ← Run metadata
├── SBIN_predictions.csv         ← Per-stock predictions
├── HDFCBANK_predictions.csv
├── INFY_predictions.csv
└── ... (one CSV per stock)
```

---

## 🚀 Running the Pipeline

### 1. **Single Stock**
```bash
cd AlgoTrading
source venv/bin/activate
python V3/07_pipeline/train_pipeline.py --symbols SBIN --test-size 100
```

### 2. **Multiple Stocks**
```bash
python V3/07_pipeline/train_pipeline.py --symbols SBIN HDFCBANK INFY TCS AXISBANK --test-size 100
```

### 3. **All 100 Stocks**
```bash
python V3/07_pipeline/train_pipeline.py --all-symbols --test-size 100
```

### 4. **Fresh Data (Skip Cache)**
```bash
python V3/07_pipeline/train_pipeline.py --symbols SBIN --fresh --test-size 100
```

---

## 📈 What Each Run Contains

### Training Details
- **Train Period**: 2019-01-01 to T-100 days
- **Test Period**: Last 100 days (configurable)
- **Features**: 50+ technical indicators (SMA, RSI, MACD, etc.)
- **Models**: 
  - GradientBoosting (1000 trees, max_depth=5)
  - RandomForest (200 trees, max_depth=10)
  - Ensemble: Average of both probabilities

### Output Example (SBIN)
```
Symbol:              SBIN
Directional Accuracy: 52.00%
Win Rate:            53.52%
Profit Factor:       1.22
Precision (Bullish): 52.00%
Recall (Bullish):    100.00%
Total Trades:        100
```

---

## 🔧 Configuration

Edit `V3/config_v3.py` to change:
```python
SYMBOLS_100          # Stock universe
DATA_START_DATE      # Historical data start
YFINANCE_DELAY       # Download delay (rate limiting)
```

---

## 📝 Next Steps

1. **Run pipeline on 100 stocks**:
   ```bash
   python V3/07_pipeline/train_pipeline.py --all-symbols --test-size 150
   ```

2. **Analyze results**:
   ```bash
   # View generated REPORT.md for summary
   cat V3/06_results/runs/LATEST/REPORT.md
   ```

3. **Add TensorFlow** (if switching to Python 3.13):
   ```bash
   # Install TensorFlow models in V3/02_models/deep_learning/
   # Then enable in train_pipeline.py
   ```

4. **Optimize features**:
   - Edit `V3/01_data/features.py` to add/remove indicators
   - Run pipeline again to see impact on accuracy

---

## ✅ Checklist

- [x] Python 3.14 venv created
- [x] Core packages installed (numpy, pandas, scikit-learn)
- [x] Data downloader working (yfinance)
- [x] Feature engineering module (50+ indicators)
- [x] Model training (GradientBoosting + RandomForest)
- [x] Directional accuracy calculation
- [x] Comprehensive reporting (CSV, Markdown, JSON)
- [ ] TensorFlow (blocked on Python 3.14 support)
- [ ] Walk-forward validation (coming next)
- [ ] 100-stock full run & analysis

---

## 🎯 Performance Expectations

**Current Results (5 stocks, 100-day test)**:
- Average Accuracy: 49.2%
- Best: 52% (SBIN)
- Profit Factor: 0.74 (not yet profitable, but close)

**Research Goal**: >52% accuracy = profitable at scale (with transaction costs <0.5%)

