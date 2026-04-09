"""
step 5 — Next-Day Prediction
==============================
Loads production models for each symbol and generates tomorrow's directional signal.
Uses the exact preprocessing pipeline from training: winsorise → scale → PCA → ensemble → meta → regime → temperature.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_V3_ROOT   = _STEPS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "02_models"))

from config_v3 import (  # type: ignore  # noqa: E402
    MODELS_PROD_DIR, CONFIDENCE_THRESHOLD, DL_SEQ_LEN,
)
from steps.features import compute_features  # type: ignore
from steps.train    import apply_temperature, _DL_AVAILABLE, _DL_CLASSES  # type: ignore


def predict_next_day(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    Load production model for `symbol` and predict tomorrow's direction.

    Returns dict with keys: symbol, last_date, last_close, direction, action,
    confidence, avg_prob, signal_active, regime, regime_label, temperature.
    Returns None if no production model exists.
    """
    prod_path = MODELS_PROD_DIR / symbol
    if not (prod_path / "metadata.json").exists():
        return None

    try:
        with open(prod_path / "metadata.json") as f:
            meta = json.load(f)
    except Exception:
        return None

    feat_cols   = meta["feature_names"]
    scaler_path = prod_path / "scaler.pkl"
    if not scaler_path.exists():
        return None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load tree models
    models = {}
    for pkl in prod_path.glob("*.pkl"):
        if pkl.stem in ("scaler", "meta_model", "pca", "winsor_bounds"):
            continue
        try:
            with open(pkl, "rb") as f:
                models[pkl.stem] = pickle.load(f)
        except Exception:
            pass

    if not models:
        return None

    # Temperature calibration
    temperature = 1.0
    cal_path = prod_path / "calibration.json"
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                temperature = float(json.load(f).get("temperature", 1.0))
        except Exception:
            pass

    # Compute features
    df = compute_features(raw_df, symbol=symbol, peer_returns=peer_returns,
                          market_df=market_df, usdinr_df=usdinr_df,
                          global_cues_df=global_cues_df)
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 10)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df = df.reset_index(drop=True)
    if df.empty or len(feat_cols) != scaler.n_features_in_:
        return None

    # Load PCA and winsor bounds
    pca = None
    if (prod_path / "pca.pkl").exists():
        with open(prod_path / "pca.pkl", "rb") as f:
            pca = pickle.load(f)
    wb = None
    if (prod_path / "winsor_bounds.pkl").exists():
        with open(prod_path / "winsor_bounds.pkl", "rb") as f:
            wb = pickle.load(f)

    # Load DL metadata
    dl_meta_inf = {}
    if (prod_path / "dl_meta.json").exists():
        try:
            with open(prod_path / "dl_meta.json") as f:
                dl_meta_inf = json.load(f)
        except Exception:
            pass
    dl_seq_len_inf = int(dl_meta_inf.get("seq_len",    DL_SEQ_LEN))
    dl_n_feat_inf  = int(dl_meta_inf.get("n_features", len(feat_cols)))
    dl_models_inf  = dl_meta_inf.get("dl_models",    [])
    meta_col_order = dl_meta_inf.get("meta_columns", [])

    if len(df) < max(1, dl_seq_len_inf):
        return None

    # Preprocessing
    X_raw_N = df[feat_cols].iloc[-dl_seq_len_inf:].values.astype(float)
    X_raw_N = np.nan_to_num(X_raw_N, nan=0.0)
    if wb is not None:
        X_raw_N = np.clip(X_raw_N, wb[0], wb[1])
    X_sc_N = np.clip(scaler.transform(X_raw_N), -5, 5)

    X_sc_1 = X_sc_N[[-1]]
    X_tree = pca.transform(X_sc_1) if pca is not None else X_sc_1
    X_seq_inf = X_sc_N[np.newaxis, :, :]

    # Tree model inference
    probs_dict: Dict[str, float] = {}
    for name, mdl in models.items():
        try:
            p = mdl.predict_proba(X_tree)
            probs_dict[name] = float(p.ravel()[0])
        except Exception:
            pass

    # DL model inference
    if _DL_AVAILABLE and dl_models_inf:
        _dl_name_to_class = {dn: cls for cls, dn in _DL_CLASSES}
        for dl_name in dl_models_inf:
            keras_path = prod_path / f"{dl_name.lower()}.keras"
            if not keras_path.exists() or dl_name not in _dl_name_to_class:
                continue
            try:
                mdl = _dl_name_to_class[dl_name](seq_len=dl_seq_len_inf, n_features=dl_n_feat_inf)
                mdl.load(str(keras_path))
                p = mdl.predict_proba(X_seq_inf)
                probs_dict[dl_name] = float(p.ravel()[0])
            except Exception:
                pass

    if not probs_dict:
        return None

    avg_prob = float(np.mean(list(probs_dict.values())))

    # Meta-learner stacking
    meta_path = prod_path / "meta_model.pkl"
    if meta_path.exists() and meta_col_order:
        try:
            with open(meta_path, "rb") as f:
                mm = pickle.load(f)
            mx_cols = [probs_dict.get(col, 0.5) for col in meta_col_order
                       if col in probs_dict or col.lower() in {k.lower() for k in probs_dict}]
            if len(mx_cols) == mm.n_features_in_:
                avg_prob = float(mm.predict_proba(np.array(mx_cols).reshape(1, -1))[0, 1])
            else:
                mx = np.array(list(probs_dict.values())).reshape(1, -1)
                if mx.shape[1] == mm.n_features_in_:
                    avg_prob = float(mm.predict_proba(mx)[0, 1])
        except Exception:
            pass

    avg_prob = float(apply_temperature(np.array([avg_prob]), temperature)[0])

    regime_val = int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 1
    regime_lbl = {0: "bear", 1: "sideways", 2: "bull"}.get(regime_val, "sideways")

    # Regime LightGBM blend
    rlgb_path = prod_path / f"lgb_{regime_lbl}.pkl"
    if rlgb_path.exists():
        try:
            with open(rlgb_path, "rb") as f:
                rlgb = pickle.load(f)
            r_prob = float(apply_temperature(
                np.array([float(rlgb.predict_proba(X_tree).ravel()[0])]), temperature)[0])
            avg_prob = 0.6 * r_prob + 0.4 * avg_prob
        except Exception:
            pass

    direction     = 1 if avg_prob >= 0.5 else 0
    confidence    = avg_prob if direction == 1 else 1 - avg_prob
    signal_active = confidence >= CONFIDENCE_THRESHOLD

    return {
        "symbol":        symbol,
        "last_date":     str(df["date"].iloc[-1])[:10],
        "last_close":    float(raw_df["close"].iloc[-1]),
        "direction":     "UP" if direction == 1 else "DOWN",
        "action":        ("BUY" if direction == 1 else "SELL") if signal_active else "HOLD",
        "confidence":    round(confidence, 4),
        "avg_prob":      round(avg_prob, 4),
        "signal_active": signal_active,
        "regime":        regime_val,
        "regime_label":  regime_lbl,
        "temperature":   round(temperature, 3),
    }


def predict_worker(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns,
    market_df,
    usdinr_df,
    global_cues_df,
) -> Optional[Dict]:
    """Thread-safe wrapper for ThreadPoolExecutor."""
    try:
        return predict_next_day(
            symbol, raw_df, peer_returns=peer_returns,
            market_df=market_df, usdinr_df=usdinr_df,
            global_cues_df=global_cues_df,
        )
    except Exception:
        return None
