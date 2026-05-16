"""
step 3 — Walk-Forward Training
================================
Builds expanding-window schedule and trains the 5-model ensemble per window:
  Tree branch  : LightGBM + XGBoost (PCA-transformed features)
  DL branch    : BiLSTM + TCN-Transformer + NBEATS (PCA-compressed sequences)
  Stacking     : LogisticRegression meta-learner on val-set model probabilities
  Regime       : Per-regime LightGBM blended with global ensemble
  Calibration  : Temperature scaling (single-parameter NLL minimisation on val)
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Backend: TensorFlow CPU ───────────────────────────────────────────────────
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_V3_ROOT   = _STEPS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "02_models"))

from config_v3 import (  # type: ignore  # noqa: E402
    INITIAL_TRAIN_RATIO, EXPANSION_STEP, MAX_TRAIN_RATIO,
    MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES,
    DL_SEQ_LEN, DL_BATCH_SIZE, DL_MAX_EPOCHS,
    DL_ES_PATIENCE, DL_ES_MIN_DELTA,
    DL_RLROP_FACTOR, DL_RLROP_PATIENCE, DL_RLROP_MIN_LR,
    MIN_REGIME_SAMPLES, RANDOM_SEED,
)
from traditional.lightgbm_classifier  import LightGBMClassifier   # type: ignore
from traditional.xgboost_classifier   import XGBoostClassifier    # type: ignore

# ── Deep Learning — lazy-loaded once ─────────────────────────────────────────
_DL_AVAILABLE: bool = False
_DL_CLASSES: list   = []
_N_JOBS: int        = -1   # set by worker initialiser in parallel mode
_FAST_MODE: bool    = False  # True → skip all DL, use tree models only


def set_fast_mode(fast: bool) -> None:
    """
    Call before training to disable all DL models (trees only).
    Use for production runs / quick iteration.  Persists for the process lifetime.
    """
    global _FAST_MODE, _DL_CLASSES
    _FAST_MODE = fast
    if fast:
        _DL_CLASSES = []
        print("  [train] FAST MODE — DL models disabled, trees only (LightGBM + XGBoost)")

def _load_dl_models() -> None:
    global _DL_AVAILABLE, _DL_CLASSES
    if _DL_AVAILABLE or _FAST_MODE:
        return
    try:
        from deep_learning.lstm_classifier            import LSTMClassifier
        from deep_learning.bilstm_classifier          import BiLSTMClassifier
        from deep_learning.gru_classifier             import GRUClassifier
        from deep_learning.cnn_lstm_classifier        import CNNLSTMClassifier
        from deep_learning.tcn_gru_classifier         import TCNGRUClassifier
        from deep_learning.tcn_transformer_classifier import TCNTransformerClassifier
        from deep_learning.nbeats_classifier          import NBEATSClassifier
        from deep_learning.base_deep                  import get_dl_splits  # noqa: F401
        _DL_AVAILABLE = True
        # Top-3 DL by OOS accuracy (SBIN benchmark + cross-stock validation):
        # BiLSTM=54.7%, NBEATS=53.7%, TCN_Transformer=52.5%
        # Dropped: LSTM(50.5%), GRU(49.5%), CNN_LSTM(51.6%), TCN_GRU(51.4%)
        # Reduces DL training from 7→3 models: ~3× speedup per stock.
        _DL_CLASSES = [
            (BiLSTMClassifier,         "BiLSTM"),
            (TCNTransformerClassifier, "TCN_Transformer"),
            (NBEATSClassifier,         "NBEATS"),
        ]
    except ImportError as e:
        print(f"  [DL] unavailable: {e}")

_load_dl_models()


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD WINDOW SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(n: int) -> List[Dict]:
    """
    Build expanding-window schedule.
    Train ratio steps from INITIAL_TRAIN_RATIO to MAX_TRAIN_RATIO.
    Each window's test set is the gap to the next window's train end.
    """
    windows = []
    ratio   = INITIAL_TRAIN_RATIO
    while ratio <= MAX_TRAIN_RATIO:
        train_end = int(n * ratio)
        val_size  = max(int(train_end * 0.10), 20)
        tr_end    = train_end - val_size
        va_start  = tr_end;  va_end = train_end
        te_start  = train_end
        next_r    = ratio + EXPANSION_STEP
        te_end    = int(n * (next_r + EXPANSION_STEP)) if next_r <= MAX_TRAIN_RATIO else n
        te_end    = min(te_end, n)
        if te_end - te_start < MIN_TEST_SAMPLES:
            ratio = round(ratio + EXPANSION_STEP, 4); continue
        windows.append({
            "id": len(windows) + 1,
            "train_start": 0,
            "train_end": tr_end,
            "val_start": va_start, "val_end": va_end,
            "test_start": te_start, "test_end": te_end,
            "train_ratio": ratio,
        })
        ratio = round(ratio + EXPANSION_STEP, 4)
    return windows


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def temperature_scale(val_probs: "np.ndarray", y_val: "np.ndarray") -> float:
    """Find temperature T via NLL minimisation on the validation set."""
    from scipy.optimize import minimize_scalar
    from scipy.special  import expit

    vp     = np.clip(val_probs, 1e-7, 1 - 1e-7)
    logits = np.log(vp / (1 - vp))

    def nll(T: float) -> float:
        T   = max(T, 0.05)
        cal = np.clip(expit(logits / T), 1e-7, 1 - 1e-7)
        return -float(np.mean(y_val * np.log(cal) + (1 - y_val) * np.log(1 - cal)))

    try:
        res = minimize_scalar(nll, bounds=(0.1, 3.0), method="bounded")
        T = float(res.x)
        # If T hits the upper bound or NLL barely improved, calibration failed —
        # fall back to T=1 (no calibration) rather than a hard-clamped value.
        nll_uncal = nll(1.0)
        nll_cal   = nll(T)
        if T >= 2.9 or (nll_cal > nll_uncal - 1e-4):
            return 1.0
        return T
    except Exception:
        return 1.0


def apply_temperature(probs: "np.ndarray", T: float) -> "np.ndarray":
    from scipy.special import expit
    probs  = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1 - probs))
    return np.array(expit(logits / max(T, 0.05)), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def train_window(
    X: "np.ndarray",
    y: "np.ndarray",
    window: Dict,
    feature_names: List[str],
    symbol: str,
    window_save_path: Path,
    regimes: Optional["np.ndarray"] = None,
    save_dl_keras: bool = True,
    next_ret: Optional["np.ndarray"] = None,     # NEW — used for meta-labeling
) -> Optional[Dict]:
    """
    Train 9-model ensemble on one expanding window.

    Preprocessing fork after RobustScaler:
      PCA(90% var)          → LightGBM + XGBoost (tree branch)
      create_sequences      → 7 DL models (sequence branch)

    Pipeline:
      Winsorise → RobustScaler → PCA
      ├── Tree: LightGBM + XGBoost → val-logloss weighted soft vote
      └── DL: 7 sequence models → soft vote
      → Meta-learner stacking (LogisticRegression C=0.3)
      → Regime-specific LightGBM (adaptive blend)
      → Temperature scaling calibration
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition  import PCA
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )

    ws, we = window["train_start"], window["train_end"]
    vs, ve = window["val_start"],   window["val_end"]
    ts, te = window["test_start"],  window["test_end"]

    X_tr_raw = X[ws:we]; y_train = y[ws:we]
    X_va_raw = X[vs:ve]; y_val   = y[vs:ve]
    X_te_raw = X[ts:te]; y_test  = y[ts:te]

    if len(y_train) < MIN_TRAIN_SAMPLES or len(y_test) < MIN_TEST_SAMPLES:
        return None
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    # Winsorise (fit on train)
    p01 = np.nanpercentile(X_tr_raw, 1,  axis=0)
    p99 = np.nanpercentile(X_tr_raw, 99, axis=0)
    X_tr_raw = np.clip(X_tr_raw, p01, p99)
    X_va_raw = np.clip(X_va_raw, p01, p99)
    X_te_raw = np.clip(X_te_raw, p01, p99)

    # RobustScaler (fit on train)
    scaler  = RobustScaler()
    X_train = np.clip(np.nan_to_num(scaler.fit_transform(X_tr_raw), nan=0.0), -5, 5)
    X_val   = np.clip(np.nan_to_num(scaler.transform(X_va_raw),     nan=0.0), -5, 5)
    X_test  = np.clip(np.nan_to_num(scaler.transform(X_te_raw),     nan=0.0), -5, 5)

    out_pct = np.mean((X_train < -1) | (X_train > 1)) * 100
    print(f"      [scaler] range [{X_train.min():.2f}, {X_train.max():.2f}]  {out_pct:.1f}% outside [-1,1]")

    # PCA (fit on train, 90% variance retained)
    X_train_sc = X_train.copy()   # pre-PCA: used for DL
    X_val_sc   = X_val.copy()
    X_test_sc  = X_test.copy()    # noqa: F841

    pca    = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_SEED)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)
    n_pc    = pca.n_components_
    evr     = float(pca.explained_variance_ratio_.sum())
    print(f"      [PCA]    {n_pc} components → {evr:.1%} explained variance")
    pca_names = [f"PC{i+1}" for i in range(n_pc)]

    # DL branch: apply PCA to full contiguous block [ws..te)
    X_full_block  = np.nan_to_num(np.clip(X[ws:te], p01, p99), nan=0.0)
    X_full_scaled = np.clip(scaler.transform(X_full_block), -5, 5)
    X_full_pca    = np.clip(pca.transform(X_full_scaled),   -5, 5)
    _dl_local     = dict(ws=0, we=we-ws, vs=vs-ws, ve=ve-ws, ts=ts-ws, te=te-ws)

    _dl_splits = None
    if _DL_AVAILABLE:
        try:
            from deep_learning.base_deep import get_dl_splits
            _dl_splits = get_dl_splits(
                X_scaled=X_full_pca, y=y[ws:te],
                seq_len=DL_SEQ_LEN, **_dl_local,
            )
        except Exception:
            pass

    if _dl_splits is not None:
        X_tr_seq, y_tr_dl, X_va_seq, y_va_dl, X_te_seq, y_te_dl = _dl_splits
    else:
        X_tr_seq = y_tr_dl = X_va_seq = y_va_dl = X_te_seq = y_te_dl = None

    # Temporal sample weights (exponential decay, half-life 1 year)
    ages = np.arange(len(y_train) - 1, -1, -1, dtype=float)
    w    = (np.exp(-ages / 252.0) / np.exp(-ages / 252.0).mean()).astype(np.float32)

    w_dl = None
    if X_tr_seq is not None and len(y_tr_dl) > 0:
        n_dl_tr = len(y_tr_dl)
        dl_ages = np.arange(n_dl_tr - 1, -1, -1, dtype=float)
        w_dl    = (np.exp(-dl_ages / 252.0) / np.exp(-dl_ages / 252.0).mean()).astype(np.float32)

    # Class imbalance ratio
    _n_up   = max((y_train == 1).sum(), 1)
    _n_down = max((y_train == 0).sum(), 1)
    _spw    = float(_n_down) / float(_n_up)

    trained_models: Dict = {}
    test_preds:     Dict = {}
    test_probs:     Dict = {}
    val_probs_each: Dict = {}

    # ── Tree models ───────────────────────────────────────────────────────────
    _tree_specs = [
        (LightGBMClassifier,  "LightGBM",  dict(n_jobs=_N_JOBS, is_unbalance=True)),
        (XGBoostClassifier,   "XGBoost",   dict(n_jobs=_N_JOBS, scale_pos_weight=_spw)),
    ]
    for ModelClass, name, extra_kw in _tree_specs:
        try:
            mdl = ModelClass(**extra_kw)
            mdl.train(X_train, y_train, X_val, y_val,
                      feature_names=pca_names, verbose=False, sample_weight=w)
            test_preds[name]     = mdl.predict(X_test)
            test_probs[name]     = mdl.predict_proba(X_test)
            val_probs_each[name] = mdl.predict_proba(X_val)
            trained_models[name] = mdl
            tree_acc = float(np.mean(test_preds[name] == y_test))
            print(f"      [{name:8s}] acc={tree_acc:.3f}")
        except Exception as exc:
            print(f"      [{name}] ✗ {exc}")

    if not test_preds:
        return None

    # ── DL models ─────────────────────────────────────────────────────────────
    if _DL_AVAILABLE and X_tr_seq is not None and len(X_tr_seq) >= 50:
        n_feat_dl = X_full_pca.shape[1]
        for DLClass, dl_name in _DL_CLASSES:
            try:
                mdl = DLClass(seq_len=DL_SEQ_LEN, n_features=n_feat_dl)
                mdl.train(X_tr_seq, y_tr_dl, X_va_seq, y_va_dl,
                          verbose=False, sample_weight=w_dl)
                test_preds[dl_name]     = mdl.predict(X_te_seq)
                test_probs[dl_name]     = mdl.predict_proba(X_te_seq)
                val_probs_each[dl_name] = mdl.predict_proba(X_va_seq)
                trained_models[dl_name] = mdl
                dl_acc = accuracy_score(y_test, test_preds[dl_name])
                print(f"      [{dl_name:8s}] acc={dl_acc:.3f}")
            except Exception as exc:
                print(f"      [{dl_name}] ✗ {exc}")
            finally:
                try:
                    import keras; keras.backend.clear_session()
                except Exception:
                    pass
                gc.collect()

    # ── Val-logloss weighted ensemble ─────────────────────────────────────────
    try:
        from sklearn.metrics import log_loss
        wts    = {n: 1.0 / max(log_loss(y_val, val_probs_each[n]), 1e-6) for n in trained_models}
        total_w = sum(wts.values())
        avg_prob = sum((wts[n] / total_w) * test_probs[n] for n in trained_models)
        val_avg  = sum((wts[n] / total_w) * val_probs_each[n] for n in trained_models)
    except Exception:
        avg_prob = np.mean(list(test_probs.values()), axis=0)
        val_avg  = np.mean(list(val_probs_each.values()), axis=0)

    # ── Meta-learner stacking ─────────────────────────────────────────────────
    # Elastic-net logistic regression: sparse (L1 drops noisy models) + stable (L2).
    # C=2.0 (was 0.3) — previous over-regularization degenerated 49% of stocks to
    # near-uniform averaging (max|coef|<0.05). Accept meta only if it beats base
    # on val AND has non-trivial coefs (L1 norm > 0.1) — else keep simple average.
    meta_model = None
    if len(trained_models) >= 2:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics      import accuracy_score as _acc

            meta_Xv  = np.column_stack([val_probs_each[n] for n in trained_models])
            meta_Xt  = np.column_stack([test_probs[n]     for n in trained_models])
            base_acc = _acc(y_val, (np.mean(meta_Xv, axis=1) >= 0.5).astype(int))

            if len(np.unique(y_val)) == 2:
                candidate = LogisticRegression(
                    C=2.0, max_iter=500, random_state=RANDOM_SEED,
                    penalty="elasticnet", l1_ratio=0.3, solver="saga",
                )
                candidate.fit(meta_Xv, y_val)
                meta_acc  = _acc(y_val, candidate.predict(meta_Xv))
                meta_prob = candidate.predict_proba(meta_Xt)[:, 1]
                meta_pred = (meta_prob >= 0.5).astype(int)
                coef_l1   = float(np.sum(np.abs(candidate.coef_[0])))
                if meta_acc > base_acc and len(np.unique(meta_pred)) > 1 and coef_l1 > 0.1:
                    meta_model = candidate
                    avg_prob   = meta_prob
                    val_avg    = candidate.predict_proba(meta_Xv)[:, 1]
        except Exception:
            pass

    # ── Regime-specific LightGBM ──────────────────────────────────────────────
    regime_lgb_models: Dict = {}
    win_path = window_save_path / f"window_{window['id']:02d}"

    if regimes is not None:
        tr_reg   = regimes[ws:we]; te_reg = regimes[ts:te]
        global_s = np.mean(list(test_probs.values()), axis=0)
        routed   = global_s.copy()

        for rv, rtag in [(2, "bull"), (1, "sideways"), (0, "bear")]:
            mask_tr = (tr_reg == rv)
            if mask_tr.sum() < MIN_REGIME_SAMPLES or len(np.unique(y_train[mask_tr])) < 2:
                continue
            try:
                rmdl = LightGBMClassifier(n_jobs=_N_JOBS)
                rmdl.train(X_train[mask_tr], y_train[mask_tr], X_val, y_val,
                           feature_names=pca_names, verbose=False)
                regime_lgb_models[rv] = rmdl
                mask_te = (te_reg == rv)
                if mask_te.any():
                    routed[mask_te] = rmdl.predict_proba(X_test[mask_te])
                win_path.mkdir(parents=True, exist_ok=True)
                with open(win_path / f"lgb_{rtag}.pkl", "wb") as f:
                    pickle.dump(rmdl, f)
            except Exception:
                pass

        if regime_lgb_models:
            try:
                from sklearn.metrics import log_loss as _ll
                val_reg      = regimes[vs:ve]
                global_val_s = np.mean(list(val_probs_each.values()), axis=0)
                routed_val   = global_val_s.copy()
                for rv2 in regime_lgb_models:
                    mask_va = (val_reg == rv2)
                    if mask_va.any():
                        routed_val[mask_va] = regime_lgb_models[rv2].predict_proba(X_val[mask_va])
                regime_ll = _ll(y_val, np.clip(routed_val,   1e-7, 1 - 1e-7))
                global_ll = _ll(y_val, np.clip(global_val_s, 1e-7, 1 - 1e-7))
                rw = np.exp(-regime_ll); gw = np.exp(-global_ll)
                blend_alpha = float(np.clip(rw / (rw + gw), 0.40, 0.75))
                print(f"      [regime] adaptive blend α={blend_alpha:.2f}  "
                      f"(regime_ll={regime_ll:.4f}  global_ll={global_ll:.4f})")
            except Exception:
                blend_alpha = 0.6
            avg_prob = blend_alpha * routed + (1.0 - blend_alpha) * global_s

    # ── Temperature scaling calibration ──────────────────────────────────────
    temperature = 1.0
    try:
        temperature = temperature_scale(val_avg, y_val)
        avg_prob    = apply_temperature(avg_prob, temperature)
        val_avg     = apply_temperature(val_avg, temperature)
        print(f"      [calib]  T={temperature:.3f}  "
              f"(prob spread: [{avg_prob.min():.3f}, {avg_prob.max():.3f}])")
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    #  META-LABELING  (López de Prado 2018 — trade-selection secondary)
    # ══════════════════════════════════════════════════════════════════════════
    # Secondary (M2) learns "given primary says UP, will the next-day trade be
    # profitable after round-trip cost". Features = PCA X + primary prob.
    # Trained on rows where the tree-ensemble primary said UP.
    secondary_model = None
    meta_prob_test  = np.full(len(y_test), 0.5, dtype=float)
    meta_info       = {"trained": False, "n_train_pos": 0, "val_auc": 0.0}

    if next_ret is not None:
        try:
            # Build per-split tree-ensemble primary prob (DL excluded — uses sequences,
            # different sample indexing; tree branch is enough for a meta-filter).
            _tree_names = [n for n in ("LightGBM", "XGBoost") if n in trained_models]
            if _tree_names:
                from sklearn.metrics import log_loss as _ll
                _wts = {n: 1.0 / max(_ll(y_val, val_probs_each[n]), 1e-6) for n in _tree_names}
                _tw  = sum(_wts.values())
                p_tr_primary = sum((_wts[n] / _tw) * trained_models[n].predict_proba(X_train) for n in _tree_names)
                p_va_primary = sum((_wts[n] / _tw) * val_probs_each[n] for n in _tree_names)
                # p_te_primary already in avg_prob (calibrated, includes DL) — use as test-time meta feature
                # For train/val we use tree-only to keep sample indexing simple; M2 is a filter, not the predictor.

                nr_tr = next_ret[ws:we]; nr_va = next_ret[vs:ve]
                # Meta label: trade profitable after cost
                ROUND_TRIP = 0.0025
                y2_tr = (nr_tr > ROUND_TRIP).astype(int)
                y2_va = (nr_va > ROUND_TRIP).astype(int)

                # Only train on rows where primary says UP (>= 0.5)
                up_tr = (p_tr_primary >= 0.5)
                up_va = (p_va_primary >= 0.5)

                if up_tr.sum() >= 100 and up_va.sum() >= 30 \
                        and len(np.unique(y2_tr[up_tr])) >= 2 and len(np.unique(y2_va[up_va])) >= 2:
                    X2_tr = np.column_stack([X_train[up_tr], p_tr_primary[up_tr]])
                    X2_va = np.column_stack([X_val[up_va],   p_va_primary[up_va]])
                    # For test we feed the ensemble avg_prob (already calibrated, trained on all)
                    X2_te = np.column_stack([X_test, avg_prob])

                    from lightgbm import LGBMClassifier, early_stopping as _lgb_es, log_evaluation as _lgb_log
                    import contextlib, io as _io
                    m2 = LGBMClassifier(
                        n_estimators=400, max_depth=5, learning_rate=0.03,
                        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.3, reg_lambda=1.5, min_child_samples=20,
                        is_unbalance=True, random_state=RANDOM_SEED,
                        n_jobs=_N_JOBS, verbosity=-1,
                    )
                    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
                        m2.fit(X2_tr, y2_tr[up_tr], eval_set=[(X2_va, y2_va[up_va])],
                               callbacks=[_lgb_es(50, verbose=False), _lgb_log(period=-1)])
                    meta_prob_test = m2.predict_proba(X2_te)[:, 1]

                    # Val AUC for diagnostics
                    from sklearn.metrics import roc_auc_score as _auc
                    try:
                        p2_va = m2.predict_proba(X2_va)[:, 1]
                        val_auc = float(_auc(y2_va[up_va], p2_va))
                    except Exception:
                        val_auc = 0.0

                    secondary_model = m2
                    meta_info = {"trained": True, "n_train_pos": int(up_tr.sum()), "val_auc": round(val_auc, 3)}
                    print(f"      [meta]   trained on {up_tr.sum()} primary-UP rows  val_AUC={val_auc:.3f}")
                else:
                    print(f"      [meta]   skipped (primary-UP rows: tr={up_tr.sum()}, va={up_va.sum()})")
        except Exception as _me:
            print(f"      [meta]   error: {_me}")

    ens_pred = (avg_prob > 0.5).astype(int)

    # Directional accuracy breakdown
    _mask_up      = avg_prob > 0.5
    _mask_down    = avg_prob < 0.5
    _mask_neutral = avg_prob == 0.5
    _dir_acc_up   = float(y_test[_mask_up].mean())           if _mask_up.any()   else 0.0
    _dir_acc_down = float(1 - y_test[_mask_down].mean())     if _mask_down.any() else 0.0

    acc  = accuracy_score(y_test, ens_pred)
    f1   = f1_score(y_test, ens_pred, zero_division=0)
    prec = precision_score(y_test, ens_pred, zero_division=0)
    rec  = recall_score(y_test, ens_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, avg_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, ens_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    per_model = {n: float(accuracy_score(y_test, p)) for n, p in test_preds.items()}

    # ── Save checkpoint ───────────────────────────────────────────────────────
    win_path.mkdir(parents=True, exist_ok=True)

    for name in ("LightGBM", "XGBoost"):
        if name in trained_models:
            with open(win_path / f"{name.lower()}.pkl", "wb") as f:
                pickle.dump(trained_models[name], f)

    _dl_model_keys  = [dn for _, dn in _DL_CLASSES]
    _dl_models_saved = []
    if save_dl_keras:
        for dl_name in _dl_model_keys:
            if dl_name in trained_models:
                keras_path = win_path / f"{dl_name.lower()}.keras"
                try:
                    trained_models[dl_name].save(str(keras_path))
                    _dl_models_saved.append(dl_name)
                except Exception as _se:
                    print(f"      [{dl_name}] save error: {_se}")

    with open(win_path / "scaler.pkl",        "wb") as f: pickle.dump(scaler,     f)
    with open(win_path / "pca.pkl",           "wb") as f: pickle.dump(pca,        f)
    with open(win_path / "winsor_bounds.pkl", "wb") as f: pickle.dump((p01, p99), f)
    if meta_model is not None:
        with open(win_path / "meta_model.pkl", "wb") as f: pickle.dump(meta_model, f)
    # NEW — trade-selection secondary (López de Prado meta-labeling)
    if secondary_model is not None:
        with open(win_path / "secondary.pkl", "wb") as f: pickle.dump(secondary_model, f)

    with open(win_path / "calibration.json", "w") as f:
        json.dump({"temperature": temperature, "meta_info": meta_info}, f)

    _meta_col_order = list(trained_models.keys())
    with open(win_path / "dl_meta.json", "w") as f:
        json.dump({
            "seq_len":      DL_SEQ_LEN,
            "n_features":   int(X_full_pca.shape[1]) if _DL_AVAILABLE and X_tr_seq is not None else 0,
            "dl_models":    _dl_models_saved,
            "meta_columns": _meta_col_order,
        }, f, indent=2)

    with open(win_path / "meta.json", "w") as f:
        json.dump({
            "symbol": symbol, "window_id": window["id"],
            "train_ratio": window["train_ratio"],
            "train_size": we-ws, "val_size": ve-vs, "test_size": te-ts,
            "accuracy": acc, "f1": f1, "auc": auc, "temperature": temperature,
            "per_model": per_model,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "dl_models": _dl_models_saved,
            "dir_acc_up": _dir_acc_up, "dir_acc_down": _dir_acc_down,
            "pct_neutral": float(_mask_neutral.mean()),
            "pct_up": float(_mask_up.mean()), "pct_down": float(_mask_down.mean()),
        }, f, indent=2)

    # Free DL models from memory (checkpoints saved to disk)
    for dn in _dl_model_keys:
        if dn in trained_models:
            del trained_models[dn]
    try:
        import keras; keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return {
        "window": window, "models": trained_models,
        "meta_model": meta_model, "regime_lgb_models": regime_lgb_models,
        "secondary_model": secondary_model,                    # NEW — meta-labeling M2
        "scaler": scaler, "pca": pca, "winsor_bounds": (p01, p99),
        "temperature": temperature,
        "win_path": win_path,
        "dl_meta": {
            "seq_len":      DL_SEQ_LEN,
            "n_features":   int(X_full_pca.shape[1]) if _DL_AVAILABLE and X_tr_seq is not None else 0,
            "dl_models":    _dl_models_saved,
            "meta_columns": _meta_col_order,
        },
        "y_test": y_test, "ens_pred": ens_pred, "avg_prob": avg_prob,
        "meta_prob": meta_prob_test,                            # NEW — per-test meta filter prob
        "meta_info": meta_info,                                 # NEW — diagnostics
        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc,
        "per_model": per_model, "test_preds": test_preds,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "dir_acc_up":  _dir_acc_up,  "dir_acc_down": _dir_acc_down,
        "pct_neutral": float(_mask_neutral.mean()),
        "pct_up": float(_mask_up.mean()), "pct_down": float(_mask_down.mean()),
    }
