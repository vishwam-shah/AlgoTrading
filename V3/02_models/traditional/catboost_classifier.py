"""CatBoost binary classifier for stock direction prediction (V3 pipeline)."""

import contextlib
import io
import numpy as np
from typing import List, Optional


class CatBoostClassifier:
    """
    Thin wrapper around catboost.CatBoostClassifier with the standard V3 train/predict API.

    API contract
    ------------
    train(X_tr, y_tr, X_val, y_val, feature_names, verbose)
    predict(X)        -> np.ndarray of int   (0 or 1)
    predict_proba(X)  -> np.ndarray of float (P(class=1))
    feature_importances_  -> np.ndarray (PredictionValuesChange importance)

    Why CatBoost adds value over XGB + LGB:
    - Ordered boosting reduces target leakage on small windows
    - Native handling of temporal feature ordering (no shuffling)
    - Different regularisation path → lower ensemble correlation
    """

    def __init__(self, **kwargs):
        from catboost import CatBoostClassifier as _CB
        params = {
            "iterations":        1000,
            "depth":             6,
            "learning_rate":     0.01,
            "subsample":         0.8,
            "colsample_bylevel": 0.8,
            "l2_leaf_reg":       3.0,
            "min_data_in_leaf":  20,
            "loss_function":     "Logloss",
            "eval_metric":       "Logloss",
            "random_seed":       42,
            "thread_count":      -1,
            "verbose":           False,
            "allow_writing_files": False,   # no filesystem artifacts
        }
        params.update(kwargs)
        self._early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        self.model = _CB(**params)
        self.feature_names: List[str] = []

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
        from catboost import Pool

        if feature_names:
            self.feature_names = feature_names

        train_pool = Pool(X_train, label=y_train, weight=sample_weight)

        fit_kwargs: dict = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = Pool(X_val, label=y_val)
            fit_kwargs["early_stopping_rounds"] = self._early_stopping_rounds

        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx, contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(train_pool, **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) — shape (n_samples,)."""
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.get_feature_importance()
