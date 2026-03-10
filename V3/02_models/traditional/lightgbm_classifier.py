"""LightGBM binary classifier for stock direction prediction (V3 pipeline)."""

import contextlib
import io
import numpy as np
from typing import List, Optional


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
            "objective":         "binary",
            "metric":            "binary_logloss",
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         -1,
        }
        params.update(kwargs)
        self._early_stopping_rounds = params.pop("early_stopping_rounds", 50)
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
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) — shape (n_samples,)."""
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
