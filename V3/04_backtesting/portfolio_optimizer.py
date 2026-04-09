"""
Portfolio Optimization — HRP (Hierarchical Risk Parity).

HRP is the replacement for RL-based position sizing.

Why HRP instead of RL?
- RL needs daily retraining (not feasible)
- RL agents are slow for real-time decisions
- HRP is fast (<100ms), stable, and used by hedge funds
- HRP is clustering-based: groups correlated stocks, invests equally across clusters
- HRP doesn't require matrix inversion (unlike Markowitz) — numerically stable with 100 stocks

Reference: López de Prado, M. (2016). "Building Diversified Portfolios that Outperform in Down Markets"
"""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimization.

    Constructs a portfolio by:
    1. Computing correlation matrix of stock returns
    2. Building hierarchical clustering tree (minimizes within-cluster correlation)
    3. Assigning equal risk to clusters (top-down recursive bisection)
    4. Scaling by inverse volatility within clusters
    """

    @staticmethod
    def _distance_matrix(correlation_matrix: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to distance matrix (1 - abs(correlation))."""
        return 1.0 - np.abs(correlation_matrix)

    @staticmethod
    def _quasi_diagonal_ordering(link_matrix: np.ndarray, n: int) -> np.ndarray:
        """
        Sort clusters to minimize distance between nearby assets (diagonal ordering).

        This reduces the number of times we need to split clusters.
        """
        from scipy.cluster.hierarchy import dendrogram

        dendro = dendrogram(link_matrix, no_plot=True)
        return np.array(dendro["leaves"], dtype=int)

    @staticmethod
    def allocate(
        returns_df: pd.DataFrame,
        symbols: List[str],
        lookback_days: int = 60,
        min_weight: float = 0.001,
        max_weight: float = 0.15,
    ) -> Dict[str, float]:
        """
        Compute HRP weights for a portfolio.

        Args:
            returns_df: DataFrame with daily returns per symbol (columns = symbols, index = dates)
            symbols: List of symbols to include
            lookback_days: Window for correlation estimation (default 60 days)
            min_weight: Minimum weight per stock (default 0.1%)
            max_weight: Maximum weight per stock (default 15%)

        Returns:
            {symbol: weight} — normalized to sum to 1.0
        """
        # Filter to requested symbols and recent data
        returns = returns_df[symbols].tail(lookback_days).copy()

        # Drop stocks with insufficient data
        returns = returns.dropna(axis=1, how='any')
        valid_symbols = returns.columns.tolist()

        if len(valid_symbols) < 2:
            # Not enough data for HRP, use equal weight
            equal_weight = 1.0 / len(valid_symbols) if valid_symbols else 0
            return {s: equal_weight for s in valid_symbols}

        # Compute correlation matrix
        correlation_matrix = returns.corr().values
        n = len(valid_symbols)

        # Distance matrix (1 - abs(correlation))
        distance_matrix = HierarchicalRiskParity._distance_matrix(correlation_matrix)

        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        link_matrix = hierarchy.linkage(condensed_dist, method="single")

        # Quasi-diagonal ordering (sort to reduce split count)
        ordered_indices = HierarchicalRiskParity._quasi_diagonal_ordering(
            link_matrix, n
        )

        # Compute weights recursively (top-down bisection)
        weights = np.ones(n)
        clustered_indices = [ordered_indices]

        while len(clustered_indices) > 0:
            # Pop a cluster
            cluster = clustered_indices.pop(0)

            if len(cluster) > 1:
                # Compute within-cluster covariance
                cov = returns.iloc[:, cluster].cov().values

                # Split this cluster into two halves
                mid = len(cluster) // 2
                cluster1 = cluster[:mid]
                cluster2 = cluster[mid:]

                # Variance per cluster
                var1 = np.sum(cov[np.ix_(range(mid), range(mid))])
                var2 = np.sum(cov[np.ix_(range(mid, len(cluster)), range(mid, len(cluster)))])

                # Allocate inverse to variance
                alpha = 1.0 - var1 / (var1 + var2) if (var1 + var2) > 0 else 0.5

                # Scale weights
                weights[cluster1] *= alpha / len(cluster1) if len(cluster1) > 0 else 0
                weights[cluster2] *= (1.0 - alpha) / len(cluster2) if len(cluster2) > 0 else 0

                # Add sub-clusters for further processing
                if len(cluster1) > 1:
                    clustered_indices.insert(0, cluster1)
                if len(cluster2) > 1:
                    clustered_indices.insert(0, cluster2)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Apply min/max constraints
        weights = np.maximum(weights, min_weight)
        weights = np.minimum(weights, max_weight)
        weights = weights / np.sum(weights)  # Renormalize

        # Return as dict
        return {valid_symbols[i]: float(weights[i]) for i in range(len(valid_symbols))}

    @staticmethod
    def allocate_with_confidence(
        returns_df: pd.DataFrame,
        symbols: List[str],
        confidences: Dict[str, float],
        lookback_days: int = 60,
        min_weight: float = 0.001,
        max_weight: float = 0.15,
    ) -> Dict[str, float]:
        """
        Compute HRP weights, scaled by model confidence per stock.

        High-confidence stocks get more weight, low-confidence stocks get less.

        Args:
            returns_df: Daily returns DataFrame
            symbols: List of symbols
            confidences: {symbol: confidence [0, 1]}
            lookback_days: Window for correlation (default 60)
            min_weight: Minimum weight (default 0.1%)
            max_weight: Maximum weight (default 15%)

        Returns:
            {symbol: weight} — HRP scaled by confidence
        """
        # Get base HRP weights
        hrp_weights = HierarchicalRiskParity.allocate(
            returns_df, symbols, lookback_days, min_weight=0, max_weight=1.0
        )

        # Scale by confidence
        confidence_scaled = {}
        for s in hrp_weights:
            confidence = confidences.get(s, 0.55)  # Default to threshold
            confidence_scaled[s] = hrp_weights[s] * confidence

        # Normalize
        total = sum(confidence_scaled.values())
        if total > 0:
            confidence_scaled = {s: w / total for s, w in confidence_scaled.items()}

        # Apply min/max constraints
        for s in confidence_scaled:
            confidence_scaled[s] = max(confidence_scaled[s], min_weight)
            confidence_scaled[s] = min(confidence_scaled[s], max_weight)

        # Renormalize
        total = sum(confidence_scaled.values())
        if total > 0:
            confidence_scaled = {s: w / total for s, w in confidence_scaled.items()}

        return confidence_scaled
