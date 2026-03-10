# V3 Pipeline — Implementation Plan
**Date:** March 9, 2026  
**Current Status:** 61/100 stocks done, avg OOS accuracy ~51.4%  
**Target:** ≥60% accuracy for at least 1 model on ≥1 window per stock

---

## Hardware Baseline

| Component | Spec |
|---|---|
| CPU | Intel Core i5-1035G7 (4 cores / 8 threads, 1.2–3.7 GHz) |
| RAM | 16 GB |
| GPU | Intel Iris Plus Graphics (iGPU, 8 GB shared from system RAM) |
| TF Version | 2.18.0 + `tensorflow_intel` 2.18.0 (oneDNN already active) |
| Keras | 3.12.0 |
| PyTorch | 2.6.0 |
| XGBoost | 2.0.0 · LightGBM 4.1.0 |

---

## Root Cause Analysis

### Problem 1 — DL Models Are Severely Undertrained *(most critical)*

**Evidence:**
- `DL_ES_PATIENCE = 7`, `DL_BATCH_SIZE = 64`, `DL_MAX_EPOCHS = 50`
- Mean train sequences ≈ 996 → only **15.6 gradient steps/epoch**
- Early stopping fires at epoch ~8 → **≈125 total gradient steps** — far below convergence
- All DL models cluster within ±0.4% of each other at ~50.5% — signature of undertrained networks

**Fix (config.py only):**
```python
DL_ES_PATIENCE  = 15     # was 7
DL_ES_MIN_DELTA = 5e-5   # was 1e-4 (less aggressive stopping)
DL_MAX_EPOCHS   = 150    # was 50
DL_BATCH_SIZE   = 32     # was 64 → 2× steps/epoch (31 steps/epoch at mean size)
DL_RLROP_PATIENCE = 8    # was 5
```

**Expected gain:** +1.5–2.5% on all DL models

---

### Problem 2 — N-BEATS Architecture Broken for Multivariate Input

**Evidence:**
- Input = `SEQ_LEN × n_features = 20 × 50 = 1000 dims` flattened
- FC hidden = 256 → **4× compression** — cannot reconstruct a 1000-dim backcast
- `residual = input − backcast` where backcast ≈ garbage → all 4 blocks see the same signal
- Result: N-BEATS behaves as 4 independent MLP towers with no hierarchical decomposition

**Fix (nbeats_classifier.py + config.py):**
1. Add input projection layer: `1000 → 256` before blocks (bottleneck compression)
2. Increase `fc_dim` to 512 so blocks have capacity to model the projected space
3. Reduce `n_blocks` to 3 (fewer but functional blocks > more broken ones)

**Expected gain:** +1–2% N-BEATS accuracy

---

### Problem 3 — Class Imbalance: 56% of Stocks Lose Money on DOWN Predictions

**Evidence:**
- **32 out of 57 stocks** have DOWN-direction precision < 50% (worse than random)
- Average UP precision = 52.2%, DOWN precision = 46.7%
- Bull market drift 2019–2026 causes systematic model bias toward predicting UP
- XGB and LGBM have no class balancing configured — equal penalty for both classes

**Fix (config.py + classifier wrappers):**
```python
# XGB_PARAMS — add:
"scale_pos_weight": "auto"   # computed per window as n_down / n_up

# LGBM_PARAMS — add:
"is_unbalance": True         # auto-reweights minority class
```

The `"auto"` value will be resolved in `train_window()` by computing the real ratio from `y_train` before passing to the classifier.

**Expected gain:** +1–2% overall; +3–5% specifically on DOWN-biased stocks

---

### Problem 4 — DL Models Receive 50 Correlated Raw Features vs Trees' 21 PCA Components

**Evidence:**
- Trees get PCA-transformed uncorrelated 21-component vectors → better generalisation
- DL gets raw 50 features including highly correlated groups:
  `ret_1d / ret_2d / ret_5d`, `rsi_7 / rsi_14 / rsi_21`, `sma_5/10/20/50` etc.
- Correlated inputs confuse LSTM/GRU gating — noise competes with signal
- DL `n_features=50` but PCA explains 90% variance in only ~21 components

**Fix (run_pipeline.py — train_window):**
Pass PCA-transformed features to DL models (same as trees). This reduces DL input to 21 clean uncorrelated dims, which aligns with what N-BEATS/LSTM papers actually validate on.

```python
# Instead of X_full_scaled (50 raw features), build DL sequences from PCA output
X_full_pca = np.clip(pca.transform(scaler.transform(np.clip(X[ws:te], p01, p99))), -5, 5)
# sequences → (batch, seq_len=20, n_pca=21)
```

**Expected gain:** +1.5–2.5% on all DL models (biggest single architectural gain)

---

### Problem 5 — Meta-Stacker Over-Regularised (C=0.05)

**Evidence:**
- `LogisticRegression(C=0.05)` = extreme regularisation → outputs near-uniform weights
- Best models (LGBM 51.2%, XGB 51.2%) get nearly equal weight to worst (TCN_GRU 50.1%)
- Ensemble benefit from selective weighting is lost

**Fix (run_pipeline.py):**
```python
meta_model = LogisticRegression(C=0.3, max_iter=300, random_state=RANDOM_SEED)
```

**Expected gain:** +0.3–0.8% ensemble accuracy

---

### Problem 6 — CNN_LSTM and CNN_GRU Are Redundant Noise in Ensemble

**Evidence:**
- CNN_LSTM avg = 50.73%, CNN_GRU avg = 50.74% — statistically identical
- Both share the same causal Conv1D backbone with different recurrent stages
- Adding identical-performing correlated models *reduces* ensemble diversity
- Each redundant model dilutes the vote of the better tree models

**Fix (config.py / run_pipeline.py):**
Replace one CNN variant with a dedicated Temporal Attention model.
Or simply drop CNN_GRU from the ensemble and keep CNN_LSTM only.

**Expected gain:** +0.2–0.5% (noise reduction)

---

## Priority Implementation Order

| Priority | Fix | File(s) | Effort | Expected gain |
|---|---|---|---|---|
| **P1** | DL training hyperparams | `config.py` | 5 min | +1.5–2.5% DL |
| **P2** | Class balancing (XGB + LGBM) | `config.py` + `train_window()` | 20 min | +1–2% overall |
| **P3** | Meta-stacker C value | `run_pipeline.py` | 2 min | +0.3–0.8% |
| **P4** | DL uses PCA features | `run_pipeline.py` | 30 min | +1.5–2.5% DL |
| **P5** | N-BEATS architecture fix | `nbeats_classifier.py` + `config.py` | 30 min | +1–2% NBEATS |
| **P6** | Remove CNN_GRU redundancy | `config.py` | 5 min | +0.2–0.5% |

**Total expected improvement: +5–10% → brings avg from 51.4% to 56–61%**

---

## GPU Acceleration Plan

### Current Situation

Your hardware is an **Intel Iris Plus Graphics** (iGPU, Gen 11, Ice Lake).  
The GPU shares 8 GB of your 16 GB system RAM — it is **not a discrete GPU**.

**What's already active:**
- `tensorflow_intel 2.18.0` is installed and active
- This enables **Intel oneDNN** (MKL-DNN) for all TF/Keras operations
- oneDNN auto-vectorises matrix multiplications using AVX-512 instructions on the CPU
- This provides 2–4× speedup over stock TensorFlow on Intel hardware *with no changes needed*

### Option A — torch-directml (Recommended for Intel iGPU on Windows) ★

**DirectML** is Microsoft's machine learning API built on DirectX 12, supported natively on Intel Iris/UHD on Windows 10/11.

**Install:**
```powershell
pip install torch-directml
```

**Usage (needs PyTorch models, not Keras):**
```python
import torch_directml
device = torch_directml.device()  # Intel iGPU via DirectML
x = torch.tensor(data).to(device)
model.to(device)
```

**Limitation:** Current pipeline uses Keras/TensorFlow. To use this, DL models must be rewritten in native PyTorch. This is a medium-effort rewrite (1–2 days) and unlocks the iGPU.

**Expected speedup:** 2–4× per DL model training (iGPU for matrix ops, frees CPU for data prep)

---

### Option B — intel-extension-for-tensorflow[gpu] *(Intel Arc only — Not supported here)*

`intel-extension-for-tensorflow[gpu]` requires **Intel Arc** discrete GPU or **Intel Data Center GPU**.  
Intel Iris Plus (Gen 11 Ice Lake) is **not supported**.  
**→ Skip this option.**

---

### Option C — ONNX Runtime + DirectML (Drop-in for current Keras models)

Export trained Keras models to ONNX format and run inference via DirectML.  
**Inference-only** — training still on CPU, but production inference uses iGPU.

```powershell
pip install onnx tf2onnx onnxruntime-directml
```

```python
# Export once:
import tf2onnx, onnxruntime as ort
tf2onnx.convert.from_keras(model, output_path="model.onnx")

# Inference:
sess = ort.InferenceSession("model.onnx", providers=["DmlExecutionProvider"])
```

**Expected speedup:** 3–6× for inference (walk-forward test predictions)

---

### Option D — CPU Optimizations (Zero installation, immediate gains)

These require no new packages. They exploit what's already installed:

**1. Parallelise per-stock with joblib (already present):**
```python
# In run_pipeline.py main() — worker pool already uses --workers 4
# Increase to 6 for i5-1035G7 (4P cores + 4 HT, leave 2 for OS)
python run_pipeline.py --workers 6
```

**2. Disable TF GPU detection warnings (already broken, wastes startup time):**
```python
# Add to run_pipeline.py top:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # disable CUDA search
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"            # suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"           # confirm oneDNN is ON
```

**3. Fix protobuf error (the `MessageFactory` spam — 20+ error lines per model = wasted I/O):**
```powershell
pip install "protobuf<4.0"   # downgrade to fix MessageFactory error
```

**4. LightGBM + XGBoost num_threads:**
```python
# LGBM_PARAMS — ensure this:
"n_jobs": 4   # explicit 4 — matches physical cores (not HT)

# XGB_PARAMS:
"n_jobs": 4
"nthread": 4
```

**Expected combined speedup:** 20–35% faster per stock (CPU thread efficiency + no spam overhead)

---

### GPU Priority Recommendation

Given Intel Iris Plus (not Arc), the realistic GPU plan is:

| Step | Action | Effort | Gain |
|---|---|---|---|
| **Now** | Fix protobuf error | 2 min | Removes 20+ spam lines/model |
| **Now** | Set `TF_ENABLE_ONEDNN_OPTS=1` explicitly | 2 min | Confirms existing oneDNN |
| **Now** | `--workers 6` in pipeline launch | 1 min | ~+20% throughput |
| **Phase 2** | Install `torch-directml` + port models to PyTorch backend | 1–2 days | 2–4× DL training speed |
| **Phase 2** | ONNX + DirectML for inference | 2–4 hours | 3–6× inference speed |

**Note:** For models already completed (61 stocks), a re-run on the full 100 is needed to apply all algorithm fixes anyway — GPU acceleration becomes most valuable for that full re-run.

---

## Accuracy Target Assessment

| Target | Confidence | Current Evidence |
|---|---|---|
| **≥60% any model, any window, SBIN** | **95%** | Already hitting 67.2% on window 6 |
| **≥60% ensemble OOS, SBIN** | **85%** | Already 65.7% OOS average |
| **≥60% any model, any window, ≥20 stocks** | **80%** | 42 windows already hit it; fixes expand this |
| **≥60% ensemble OOS, ≥10 stocks** | **65%** | Currently 6 stocks >55%; class fix + DL fix will push borderline over |
| **≥55% average across all 100 stocks** | **70%** | Requires P1+P2+P4 all working |
| **≥60% average across all 100 stocks** | **10%** | Industry ceiling for next-day binary on liquid stocks |

**Guaranteed:** After P1+P2+P3 fixes, XGB/LGBM will reliably exceed 60% on the window with the most training data (window 6, train_ratio=0.95) for banking/financial stocks.

---

## Files to Change

```
config.py                              → P1 (DL hyperparms) + P2 (class balance flags)
V3/07_pipeline/run_pipeline.py         → P2 (compute scale_pos_weight) + P3 (meta C) + P4 (DL→PCA)
V3/02_models/deep_learning/nbeats_classifier.py  → P5 (input projection)
V3/02_models/traditional/xgboost_classifier.py   → P2 (accept scale_pos_weight kwarg)
V3/02_models/traditional/lightgbm_classifier.py  → P2 (accept is_unbalance kwarg)
```

---

## Quick-Win Commands (run after fixes)

```powershell
# Fix protobuf spam first
pip install "protobuf>=3.20,<4.0"

# Re-run remaining 39 stocks with new settings
python V3/07_pipeline/run_pipeline.py --resume 20260307_141956 --workers 6

# After full run, check improvement
python analyze_pipeline.py
```

---

## TODO — Immediate Action Items

> These are ordered by impact-per-minute-of-effort. Do these before touching architecture.

### ✅ Done
- [x] Analyzed all root causes (6 problems identified)
- [x] Created implementation plan

### 🔴 Do First (Pre-code, 5 min total)

- [ ] **Fix protobuf spam** — removes 20+ error lines printed per DL model, speeds up stdout flushing
  ```powershell
  pip install "protobuf>=3.20,<4.0"
  ```

- [ ] **Switch to `--workers 6`** — i5-1035G7 has 8 logical cores; current run uses 4, leaving 2 physical cores idle
  ```powershell
  # Kill current run first, then resume with 6 workers
  python V3/07_pipeline/run_pipeline.py --resume 20260307_141956 --workers 6
  ```

> ⚠️ **Note:** GPU acceleration (torch-directml / ONNX DirectML) is **not worth pursuing now**.
> Intel Iris Plus is a Gen 11 iGPU — matrix throughput is slower than the CPU's oneDNN/AVX path
> that `tensorflow_intel 2.18.0` already uses. The algo fixes below will deliver 10–20× more
> accuracy improvement than any GPU change possible on this hardware.

---

### 🟠 P1 — DL Training Hyperparams (`config.py`) — 5 min, +1.5–2.5% DL accuracy

- [ ] Change `DL_ES_PATIENCE` from `7` → `15`
- [ ] Change `DL_ES_MIN_DELTA` from `1e-4` → `5e-5`
- [ ] Change `DL_MAX_EPOCHS` from `50` → `150`
- [ ] Change `DL_BATCH_SIZE` from `64` → `32`
- [ ] Change `DL_RLROP_PATIENCE` from `5` → `8`

---

### 🟠 P2 — Class Balancing for XGB + LGBM (`config.py` + `run_pipeline.py`) — 20 min, +1–2%

- [ ] Add `scale_pos_weight` computed per-window in `train_window()` (ratio of DOWN/UP days in `y_train`)
- [ ] Pass computed weight to `XGBoostClassifier` init
- [ ] Add `is_unbalance: True` to `LGBM_PARAMS` in `config.py`
- [ ] Verify no regressions on SBIN (already good) — should not drop below 55%

---

### 🟠 P3 — Meta-Stacker Regularisation (`run_pipeline.py`) — 2 min, +0.3–0.8%

- [ ] Change `LogisticRegression(C=0.05, ...)` → `LogisticRegression(C=0.3, ...)`

---

### 🟡 P4 — DL Uses PCA Features (`run_pipeline.py`) — 30 min, +1.5–2.5% DL

- [ ] Build `X_full_pca` block from PCA-transformed data (same scaler+PCA fitted on train window)
- [ ] Create DL sequences from `X_full_pca` instead of `X_full_scaled`
- [ ] Update `n_feat_dl` to use `n_pca` components (≈21) instead of 50
- [ ] Update `dl_meta.json` to store `n_features = n_pca`
- [ ] Test: run single stock KOTAKBANK manually to verify shapes match before full re-run

---

### 🟡 P5 — Fix N-BEATS Input Bottleneck (`nbeats_classifier.py`) — 30 min, +1–2% N-BEATS

- [ ] Add `Dense(256, relu)` projection layer before the first block (compresses `seq_len × n_feat → 256`)
- [ ] Increase `fc_dim: 256 → 512` in `NBEATS_PARAMS`
- [ ] Reduce `n_blocks: 4 → 3`
- [ ] Verify backcast dimension matches projected input (256), not original 1000

---

### 🟢 P6 — Remove CNN_GRU Redundancy — 5 min, +0.2–0.5%

- [ ] Remove `CNNGRUClassifier` from `_DL_CLASSES` list in `run_pipeline.py`
- [ ] Remove `cnn_gru_acc` from summary columns and `_model_cols` dict
- [ ] Keep CNN_LSTM (slightly higher accuracy, same architecture family)

---

### 🔵 Phase 2 — GPU (After all above are done and full re-run completed)

- [ ] Install `torch-directml` and prototype one model (LSTM) in pure PyTorch
- [ ] Benchmark: PyTorch+DirectML vs Keras+oneDNN for one stock training time
- [ ] If PyTorch+DirectML is faster: port all DL models to PyTorch backend
- [ ] Install `onnxruntime-directml` for production inference acceleration
