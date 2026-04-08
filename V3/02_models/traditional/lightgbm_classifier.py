"""LightGBM binary classifier for stock direction prediction (V3 pipeline)."""

import contextlib
import io
import numpy as np
from typing import List, Optional


# ── Focal loss custom objective ────────────────────────────────────────────────
# Handles hard-to-classify samples (uncertain predictions near 0.5) more
# aggressively than standard binary log-loss.
# Source: ACM RACS 2022 — improved minority-class recall on imbalanced datasets.
# gamma=0 → identical to binary log-loss; gamma=2 is the standard literature default.
_FOCAL_GAMMA: float = 2.0


class _FocalLossLGB:
    """
    Picklable focal loss objective for LightGBM (callable class, not closure).

    Using a top-level class instead of a closure ensures pickle/joblib can
    serialise the LGBMClassifier model object (closures are not picklable).

    Signature: (y_true, y_pred) → (gradient, hessian)
      y_pred is the raw logit score from the booster.

    gamma=0 → identical to binary log-loss.
    gamma=2 → standard focal loss (Lin et al. 2017); handles hard examples.
    Source: ACM RACS 2022 — improved minority-class recall on imbalanced datasets.
    """
    def __init__(self, gamma: float = _FOCAL_GAMMA):
        self.gamma = gamma

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        p    = 1.0 / (1.0 + np.exp(-y_pred))      # sigmoid of raw logit
        pt   = np.where(y_true == 1, p, 1.0 - p)  # prob of the true class
        fw   = (1.0 - pt) ** self.gamma            # focal weight
        grad = fw * (p - y_true)
        hess = np.maximum(fw * p * (1.0 - p), 1e-6)
        return grad, hess

    def __repr__(self) -> str:
        return f"FocalLossLGB(gamma={self.gamma})"


class LightGBMClassifier:
    """
    Thin wrapper around LGBMClassifier with the standard V3 train/predict API.

    API contract
    ------------
    train(X_tr, y_tr, X_val, y_val, feature_names, verbose)
    predict(X)        -> np.ndarray of int   (0 or 1)
    predict_proba(X)  -> np.ndarray of float (P(class=1))
    feature_importances_  -> np.ndarray (split importance, same length as features)
    """

    def __init__(self, **kwargs):
        from lightgbm import LGBMClassifier as _LGBM
        params = {
            "n_estimators":      1000,
            "max_depth":         5,
            "learning_rate":     0.01,
            "num_leaves":        31,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "reg_alpha":         0.3,
            "reg_lambda":        1.5,
            "min_child_samples": 20,
            # objective intentionally omitted here — set to focal loss below
            "metric":            "binary_logloss",   # eval metric for early stopping
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         -1,
        }
        params.update(kwargs)
        self._early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        # Use focal loss as training objective; metric stays binary_logloss for ES
        params["objective"] = _FocalLossLGB(_FOCAL_GAMMA)
        self._use_focal_loss = True
        self.model = _LGBM(**params)
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        from lightgbm import early_stopping as lgb_es, log_evaluation as lgb_log

        if feature_names:
            self.feature_names = feature_names

        callbacks = []
        eval_set  = []
        if X_val is not None and y_val is not None:
            eval_set  = [(X_val, y_val)]
            callbacks = [
                lgb_es(self._early_stopping_rounds, verbose=False),
                lgb_log(period=-1),
            ]

        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx, contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(
                X_train, y_train,
                eval_set        = eval_set or None,
                callbacks       = callbacks or None,
                sample_weight   = sample_weight,
            )

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) — shape (n_samples,).
        With a custom objective, LightGBM outputs raw logits; apply sigmoid explicitly.
        """
        if self._use_focal_loss:
            raw = self.model.predict(X, raw_score=True)
            return 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
