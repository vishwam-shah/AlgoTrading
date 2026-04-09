"""
Performance metrics for directional prediction.
Calculates accuracy, profit factors, and other trading metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class AccuracyMetrics:
    """Calculate directional accuracy and trading metrics."""

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Percentage of correct direction predictions."""
        return (y_true == y_pred).sum() / len(y_true) * 100

    @staticmethod
    def win_rate(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Win rate: % of predictions where direction matched and confidence high.
        Uses predicted probability for confidence weighting.
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)
        correct = (y_true == y_pred)
        confident = y_pred_proba >= 0.6

        if confident.sum() == 0:
            return 0.0

        return correct[confident].sum() / confident.sum() * 100

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, class_label: int = 1) -> float:
        """Precision for a specific class (default: bullish)."""
        predicted_positive = y_pred == class_label
        if predicted_positive.sum() == 0:
            return 0.0
        return (y_true[predicted_positive] == class_label).sum() / predicted_positive.sum() * 100

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, class_label: int = 1) -> float:
        """Recall for a specific class (default: bullish)."""
        actual_positive = y_true == class_label
        if actual_positive.sum() == 0:
            return 0.0
        return (y_pred[actual_positive] == class_label).sum() / actual_positive.sum() * 100

    @staticmethod
    def calculate_pnl(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: np.ndarray,
        confidence_threshold: float = 0.55,
    ) -> Tuple[float, float, float]:
        """
        Calculate P&L based on predictions and actual returns.

        Returns:
            (total_pnl, win_pnl, loss_pnl)
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)
        high_confidence = y_pred_proba >= confidence_threshold

        # Only trade high-confidence predictions
        trades = y_pred[high_confidence].copy()
        actual_direction = (returns[high_confidence] > 0).astype(int)
        trade_returns = returns[high_confidence]

        if len(trades) == 0:
            return 0.0, 0.0, 0.0

        # P&L: +1 for correct, -1 for incorrect
        pnl = np.where(trades == actual_direction, np.abs(trade_returns), -np.abs(trade_returns))

        total_pnl = pnl.sum()
        win_pnl = pnl[pnl > 0].sum()
        loss_pnl = pnl[pnl < 0].sum()

        return float(total_pnl), float(win_pnl), float(loss_pnl)

    @staticmethod
    def profit_factor(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: np.ndarray,
        confidence_threshold: float = 0.55,
    ) -> float:
        """
        Profit Factor: sum of wins / abs(sum of losses).
        Higher is better (>1.5 is good).
        """
        _, win_pnl, loss_pnl = AccuracyMetrics.calculate_pnl(
            y_true, y_pred_proba, returns, confidence_threshold
        )

        if loss_pnl >= 0:
            return 0.0 if win_pnl == 0 else float("inf")

        return float(abs(win_pnl / loss_pnl))

    @staticmethod
    def generate_report(
        symbol: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: np.ndarray,
    ) -> Dict:
        """Generate comprehensive metrics report for a stock."""
        accuracy = AccuracyMetrics.directional_accuracy(y_true, y_pred)
        win_rate = AccuracyMetrics.win_rate(y_true, y_pred_proba)
        precision_1 = AccuracyMetrics.precision(y_true, y_pred, class_label=1)
        recall_1 = AccuracyMetrics.recall(y_true, y_pred, class_label=1)
        pf = AccuracyMetrics.profit_factor(y_true, y_pred_proba, returns)

        return {
            "symbol": symbol,
            "accuracy": accuracy,
            "win_rate": win_rate,
            "precision_bullish": precision_1,
            "recall_bullish": recall_1,
            "profit_factor": pf,
            "sample_count": len(y_true),
            "bullish_true": (y_true == 1).sum(),
            "bullish_pred": (y_pred == 1).sum(),
        }
