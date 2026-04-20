"""XGBoost binary classifier for stock direction prediction (V3 pipeline)."""

import contextlib
import io
import numpy as np
from typing import List, Optional


class XGBoostClassifier:
    """
    Thin wrapper around xgboost.XGBClassifier with the standard V3 train/predict API.

    API contract
    ------------
    train(X_tr, y_tr, X_val, y_val, feature_names, verbose)
    predict(X)        -> np.ndarray of int   (0 or 1)
    predict_proba(X)  -> np.ndarray of float (P(class=1))
    feature_importances_  -> np.ndarray (gain importance)
    """

    def __init__(self, **kwargs):
        import xgboost as xgb
        params = {
            "n_estimators":    1000,
            "max_depth":       5,
            "learning_rate":   0.01,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":       0.3,
            "reg_lambda":      1.5,
            "eval_metric":     "logloss",
            "objective":       "binary:logistic",
            "random_state":    42,
            "n_jobs":          -1,
            "verbosity":       0,
        }
        params.update(kwargs)
        # XGBoost ≥2.0: early_stopping_rounds moves to constructor, not fit()
        params.setdefault("early_stopping_rounds", 50)
        self._early_stopping_rounds = params["early_stopping_rounds"]
        self.model = xgb.XGBClassifier(**params)
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
        if feature_names:
            self.feature_names = feature_names

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            # early_stopping_rounds is in constructor (XGBoost ≥2.0)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"]  = False

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx, contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(X_train, y_train, **fit_kwargs)

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
