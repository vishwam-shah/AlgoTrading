"""
Target computation for next-day directional prediction.
Computes binary target: 1 if tomorrow's close > today's close, else 0.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TargetComputer:
    """Compute next-day direction targets."""

    @staticmethod
    def compute_direction_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute binary direction target: 1 if next day closes up, 0 otherwise.

        Uses shift(-1) BEFORE any train/test split to avoid leakage.

        Input: DataFrame with 'close' column
        Output: Same DataFrame + 'direction_target' column (0 or 1)
        """
        df = df.copy()

        # Next day's close (shift -1 = tomorrow's value)
        tomorrow_close = df["close"].shift(-1)

        # Direction: 1 if up, 0 if down or flat
        df["direction_target"] = (tomorrow_close > df["close"]).astype(int)

        # Drop the last row (no tomorrow for the last day)
        df = df[:-1].reset_index(drop=True)

        return df

    @staticmethod
    def compute_next_day_return(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute next-day log return for regression targets.

        Returns log(close_t+1 / close_t)
        """
        df = df.copy()

        tomorrow_close = df["close"].shift(-1)
        df["next_day_log_return"] = np.log(tomorrow_close / df["close"])

        # Drop last row
        df = df[:-1].reset_index(drop=True)

        return df

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> np.ndarray:
        """
        Compute class weights for imbalanced data.
        More weight to minority class.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)

        weights = np.zeros(len(y))
        for cls, count in zip(unique, counts):
            weight = total / (2 * count)
            weights[y == cls] = weight

        return weights

    @staticmethod
    def check_target_consistency(df: pd.DataFrame) -> dict:
        """
        Validate that targets make sense.
        Returns stats about target distribution.
        """
        if "direction_target" not in df.columns:
            return {"error": "No direction_target column"}

        y = df["direction_target"].values
        total = len(y)

        return {
            "total_samples": total,
            "class_0_count": (y == 0).sum(),
            "class_1_count": (y == 1).sum(),
            "class_0_pct": (y == 0).sum() / total * 100,
            "class_1_pct": (y == 1).sum() / total * 100,
        }
