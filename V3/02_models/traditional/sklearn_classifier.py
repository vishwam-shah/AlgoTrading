"""Scikit-learn GradientBoosting classifier for stock direction prediction (no libomp needed)."""

import contextlib
import io
import numpy as np
from typing import List, Optional


class SKLearnGradientBoostingClassifier:
    """
    Gradient Boosting wrapper using scikit-learn.
    No external dependencies like libomp.

    API contract
    ------------
    train(X_tr, y_tr, X_val, y_val, feature_names, verbose)
    predict(X)        -> np.ndarray of int   (0 or 1)
    predict_proba(X)  -> np.ndarray of float (P(class=1))
    feature_importances_  -> np.ndarray
    """

    def __init__(self, **kwargs):
        from sklearn.ensemble import GradientBoostingClassifier as GBDT

        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
        }
        params.update(kwargs)
        self.model = GBDT(**params)
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        if feature_names:
            self.feature_names = feature_names

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx, contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(X_train, y_train, **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) — shape (n_samples,)."""
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


class SKLearnRandomForestClassifier:
    """Random Forest wrapper using scikit-learn."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as RFC

        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = RFC(**params)
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        if feature_names:
            self.feature_names = feature_names

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self.model.fit(X_train, y_train, **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) — shape (n_samples,)."""
        p = self.model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
