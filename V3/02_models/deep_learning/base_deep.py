"""
base_deep.py
================================================================================
Shared utilities and abstract base class for all V3 deep learning classifiers.

Sequence Indexing Contract
--------------------------
  create_sequences(X_scaled, SEQ_LEN=20) → X_seq of shape (N-19, 20, F)
  X_seq[i] = X_scaled[i : i + SEQ_LEN]
  Label alignment: X_seq[i] → y[i + SEQ_LEN - 1]

  For a window with integer row positions [ts, te) in the full (filtered) dataset:
    DL seq indices into X_seq_all:  [ts - SEQ_LEN + 1 : te - SEQ_LEN + 1]
    Aligned labels:                 y[ts : te]
    These are IDENTICAL to the tree-model test labels y[ts:te].  ✓

  The first few val/test sequences look back into the preceding split period —
  these are historical *feature* values only; no target leakage occurs because
  targets were computed forward (next-day return) and are separate from features.

Leakage Safety for BiLSTM
--------------------------
  All features in the V3 pipeline are lagged (rolling windows, previous-day
  returns, etc.).  The backward pass of a BiLSTM operates on these feature
  values, NOT on future price targets.  Using BiLSTM here is safe.
================================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Resolve config path ────────────────────────────────────────────────────────
_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import (                        # noqa: E402
    DL_SEQ_LEN, DL_BATCH_SIZE, DL_MAX_EPOCHS,
    DL_ES_PATIENCE, DL_ES_MIN_DELTA,
    DL_RLROP_FACTOR, DL_RLROP_PATIENCE, DL_RLROP_MIN_LR,
    RANDOM_SEED,
)


# ── Directional loss (module-level so load() can reference same object) ────────
# Penalises wrong-direction predictions by ALPHA×.
# Source: MDPI AI 2025 — +3–6 pp directional accuracy vs binary cross-entropy.
_DIRECTIONAL_ALPHA: float = 2.0

def _make_directional_loss(alpha: float = _DIRECTIONAL_ALPHA):
    """
    Asymmetric directional loss for binary stock direction classification.

    When the model predicts the wrong direction (pred>0.5 but actual=0, or
    pred<0.5 but actual=1), the standard BCE loss for that sample is multiplied
    by `alpha`.  Correct-direction errors (confident but not extreme) are left
    at weight 1.0.  This focuses gradient updates on direction errors rather
    than magnitude errors.

    alpha=1.0 → identical to standard binary cross-entropy.
    alpha=2.0 → direction errors cost 2× (recommended by literature).

    Uses keras.ops (Keras 3.x API) — keras.backend tensor ops were removed in Keras 3.
    """
    def dir_loss(y_true, y_pred):
        import keras
        _ops  = keras.ops
        eps   = 1e-7
        y_p   = _ops.clip(y_pred, eps, 1.0 - eps)
        bce   = -(y_true * _ops.log(y_p) + (1.0 - y_true) * _ops.log(1.0 - y_p))
        # direction error: model output is on the wrong side of 0.5
        pred_dir  = _ops.cast(y_p > 0.5, dtype=y_true.dtype)
        dir_wrong = _ops.cast(_ops.abs(pred_dir - y_true) > 0.5, dtype=y_true.dtype)
        weight    = 1.0 + (alpha - 1.0) * dir_wrong
        return _ops.mean(bce * weight)
    dir_loss.__name__ = "dir_loss"
    return dir_loss

_DIRECTIONAL_LOSS = _make_directional_loss(_DIRECTIONAL_ALPHA)


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build overlapping sliding windows (zero-copy via stride tricks).

    Parameters
    ----------
    X       : (N, F) float array
    seq_len : window length

    Returns
    -------
    X_seq : (N - seq_len + 1, seq_len, F) float array
            X_seq[i] = X[i : i + seq_len]
    """
    N, F = X.shape
    if N < seq_len:
        raise ValueError(
            f"create_sequences: need at least {seq_len} rows, got {N}."
        )
    # sliding_window_view shape: (N - seq_len + 1, F, seq_len, 1) — reshape
    view = np.lib.stride_tricks.sliding_window_view(X, (seq_len, F))
    # view.shape = (N - seq_len + 1, 1, seq_len, F)  →  squeeze axis 1
    return view[:, 0, :, :].astype(np.float32)


def get_dl_splits(
    X_scaled: np.ndarray,
    y: np.ndarray,
    ws: int, we: int,
    vs: int, ve: int,
    ts: int, te: int,
    seq_len: int,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Return leakage-free (X_seq, y) splits for train / val / test.

    All index arguments are LOCAL offsets into X_scaled / y
    (i.e., row 0 of X_scaled = row ws of the original full dataset).

    The function builds ALL sequences first, then slices out the three splits
    based on the label-alignment formula:
        DL seq index i → label index (i + seq_len - 1)
        ⟹ to get labels in [a, b): use X_seq[a - seq_len + 1 : b - seq_len + 1]

    Parameters
    ----------
    X_scaled : (n_total, F) — continuous scaled block [ws..te)
    y        : (n_total,)  — labels aligned with X_scaled
    ws/we    : train start/end (local indices)
    vs/ve    : val   start/end (local indices)
    ts/te    : test  start/end (local indices)
    seq_len  : sequence length

    Returns
    -------
    X_tr, y_tr, X_va, y_va, X_te, y_te
    """
    off = seq_len - 1
    X_seq_all = create_sequences(X_scaled, seq_len)

    # ── train ─────────────────────────────────────────────────────────────────
    tr_s = max(0, ws - off)
    tr_e = max(0, we - off)
    X_tr = X_seq_all[tr_s:tr_e]
    y_tr = y[tr_s + off : tr_e + off]   # = y[max(seq_len-1,ws):we]

    # ── val ───────────────────────────────────────────────────────────────────
    va_s = max(0, vs - off)
    va_e = max(0, ve - off)
    X_va = X_seq_all[va_s:va_e]
    y_va = y[vs:ve]                     # = y[val_start:val_end] ← same as tree

    # ── test ──────────────────────────────────────────────────────────────────
    te_s = max(0, ts - off)
    te_e = max(0, te - off)
    X_te = X_seq_all[te_s:te_e]
    y_te = y[ts:te]                     # = y[test_start:test_end] ← same as tree

    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class BaseDLClassifier:
    """
    Abstract base for all V3 deep learning classifiers.

    Consistent API — matches LightGBMClassifier / XGBoostClassifier exactly:

        model = SomeClassifier(seq_len=20, n_features=50)
        model.train(X_3D, y, X_val, y_val, verbose=False, sample_weight=None)
        proba = model.predict_proba(X_3D)   # shape (n,), P(class=1)
        preds = model.predict(X_3D)         # shape (n,), int {0, 1}
        model.save("path/to/model.keras")
        model.load("path/to/model.keras")

    Differences from tree wrappers:
        - Input X must be 3D: (n_samples, seq_len, n_features)
        - Saving uses Keras native format (.keras), not joblib pkl
    """

    model_name: str = "BaseDL"

    def __init__(self, seq_len: int = DL_SEQ_LEN, n_features: int = 50):
        self.seq_len    = seq_len
        self.n_features = n_features
        self._lr: float = 1e-3
        self._model     = None          # keras.Model set in subclass _build()
        self.is_trained: bool = False

    # ── subclass must override ─────────────────────────────────────────────────
    def _build(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build()."
        )

    # ── training ──────────────────────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build, compile, and fit the Keras model.

        X_train/X_val shape : (n, seq_len, n_features)
        y_train/y_val shape : (n,)  binary int
        sample_weight       : (n,)  temporal decay weights (passed to model.fit)
        """
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("KERAS_BACKEND",         "tensorflow")

        import keras                                    # lazy import — startup speed
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        np.random.seed(RANDOM_SEED)
        try:
            import tensorflow as tf
            tf.random.set_seed(RANDOM_SEED)
        except ImportError:
            pass

        if self._model is None:
            self._model = self._build()

        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._lr),
            loss=_DIRECTIONAL_LOSS,   # asymmetric: direction errors cost 2×
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=DL_ES_PATIENCE,
                min_delta=DL_ES_MIN_DELTA,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=DL_RLROP_FACTOR,
                patience=DL_RLROP_PATIENCE,
                min_lr=DL_RLROP_MIN_LR,
                verbose=0,
            ),
        ]

        val_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        self._model.fit(
            X_train,
            y_train.astype(np.float32),
            validation_data=val_data,
            epochs=DL_MAX_EPOCHS,
            batch_size=DL_BATCH_SIZE,
            sample_weight=sample_weight,
            callbacks=callbacks,
            verbose=1 if verbose else 0,
        )
        self.is_trained = True

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1), shape (n,)."""
        if self._model is None:
            raise RuntimeError(f"{self.model_name}: model not built. Call train() first.")
        raw = self._model.predict(X.astype(np.float32), verbose=0)
        return raw.ravel().astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions {0, 1}, shape (n,)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save model in Keras 3.x native format (.keras)."""
        if self._model is not None:
            self._model.save(path)

    def load(self, path: str) -> None:
        """Load model from .keras file."""
        import keras
        self._model     = keras.models.load_model(
            path,
            custom_objects={"dir_loss": _DIRECTIONAL_LOSS},
        )
        self.is_trained = True
