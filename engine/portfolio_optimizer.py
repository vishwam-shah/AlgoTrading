"""
================================================================================
STEP 4: PORTFOLIO OPTIMIZATION
================================================================================
Multiple optimization methods for portfolio construction.

Optimization Methods:
1. EQUAL_WEIGHT      - Simple 1/N allocation
2. RISK_PARITY       - Equal risk contribution (recommended)
3. MEAN_VARIANCE     - Classic Markowitz optimization
4. MAX_SHARPE        - Maximum Sharpe ratio portfolio
5. MIN_VOLATILITY    - Minimum variance portfolio
6. HIERARCHICAL_RP   - Hierarchical Risk Parity

Research Basis:
- Markowitz (1952): Modern Portfolio Theory
- Maillard et al. (2010): Risk Parity
- Lopez de Prado (2016): Hierarchical Risk Parity

Usage:
    from engine.portfolio_optimizer import PortfolioOptimizer
    optimizer = PortfolioOptimizer()
    allocation = optimizer.optimize(price_data, factor_scores, n_holdings=15)

Or run directly:
    python pipeline/step_4_portfolio_optimization.py
================================================================================
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    
    # Constraints satisfied
    max_position: float = 0.10
    max_sector: float = 0.30
    
    # Diagnostics
    n_positions: int = 0
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.n_positions = len([w for w in self.weights.values() if w > 0.01])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'method': self.method,
            'n_positions': self.n_positions,
            'risk_contributions': self.risk_contributions
        }


class PortfolioOptimizer:
    """
    Multiple portfolio optimization methods.
    
    Supports various optimization approaches with risk constraints.
    Recommended: Risk Parity for robust, diversified portfolios.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 5% risk-free rate for India
        max_position: float = 0.10,     # Max 10% per stock
        max_sector: float = 0.30        # Max 30% per sector
    ):
        """Initialize PortfolioOptimizer."""
        self.risk_free_rate = risk_free_rate
        self.max_position = max_position
        self.max_sector = max_sector
        
        logger.info("PortfolioOptimizer initialized")
        logger.info(f"  Risk-free rate: {risk_free_rate:.2%}")
        logger.info(f"  Max position: {max_position:.0%}")
        logger.info(f"  Max sector: {max_sector:.0%}")
    
    def optimize(
        self,
        price_data: Dict[str, pd.DataFrame],
        factor_scores: List = None,
        n_holdings: int = 15,
        method: str = 'risk_parity',
        sector_map: Dict[str, str] = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio weights.
        
        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            factor_scores: List of FactorScore objects (for stock selection)
            n_holdings: Number of stocks to hold
            method: Optimization method
            sector_map: Dict of symbol -> sector
            
        Returns:
            PortfolioAllocation object
        """
        logger.info("=" * 60)
        logger.info("STEP 4: PORTFOLIO OPTIMIZATION")
        logger.info("=" * 60)
        logger.info(f"Method: {method}")
        logger.info(f"Target holdings: {n_holdings}")
        
        # Select top stocks based on factor scores
        if factor_scores:
            # Sort by combined score
            sorted_scores = sorted(factor_scores, key=lambda x: x.combined_score, reverse=True)
            selected_symbols = [s.symbol for s in sorted_scores[:n_holdings]]
            logger.info(f"Selected top {n_holdings} stocks by factor score")
        else:
            selected_symbols = list(price_data.keys())[:n_holdings]
            logger.info(f"Using first {n_holdings} stocks (no factor scores)")
        
        # Filter price data to selected symbols
        selected_data = {s: price_data[s] for s in selected_symbols if s in price_data}
        
        if len(selected_data) < 3:
            logger.error("Insufficient stocks for optimization")
            return None
        
        # Calculate returns and covariance
        returns_df = pd.DataFrame()
        for symbol, df in selected_data.items():
            returns_df[symbol] = df['close'].pct_change()
        
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 60:
            logger.error("Insufficient data for optimization")
            return None
        
        # Annualize
        expected_returns = returns_df.mean().values * 252
        cov_matrix = returns_df.cov().values * 252
        symbols = returns_df.columns.tolist()
        
        # Optimize based on method
        if method == 'equal':
            weights = self._equal_weight(len(symbols))
        elif method == 'risk_parity':
            weights = self._risk_parity(cov_matrix)
        elif method == 'mean_variance':
            weights = self._mean_variance(expected_returns, cov_matrix)
        elif method == 'max_sharpe':
            weights = self._max_sharpe(expected_returns, cov_matrix)
        elif method == 'min_volatility':
            weights = self._min_volatility(cov_matrix)
        elif method == 'hierarchical_rp':
            weights = self._hierarchical_risk_parity(returns_df)
        else:
            logger.warning(f"Unknown method '{method}', using risk_parity")
            weights = self._risk_parity(cov_matrix)
        
        # Apply constraints
        weights = self._apply_constraints(weights, symbols, sector_map)
        
        # Calculate portfolio metrics
        port_return = np.sum(expected_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Calculate risk contributions
        risk_contribs = self._calculate_risk_contributions(weights, cov_matrix)
        
        # Create allocation object
        weights_dict = {s: float(w) for s, w in zip(symbols, weights) if w > 0.001}
        risk_contribs_dict = {s: float(rc) for s, rc in zip(symbols, risk_contribs) if weights_dict.get(s, 0) > 0.001}
        
        allocation = PortfolioAllocation(
            weights=weights_dict,
            expected_return=float(port_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method=method,
            max_position=self.max_position,
            max_sector=self.max_sector,
            risk_contributions=risk_contribs_dict
        )
        
        # Log results
        logger.info(f"\nPortfolio metrics:")
        logger.info(f"  Expected return: {port_return:.2%}")
        logger.info(f"  Volatility: {port_vol:.2%}")
        logger.info(f"  Sharpe ratio: {sharpe:.2f}")
        logger.info(f"  Positions: {allocation.n_positions}")
        
        logger.info(f"\nWeights:")
        for sym, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
            sector = sector_map.get(sym, 'Other') if sector_map else 'N/A'
            logger.info(f"  {sym:12} {weight:>6.2%}  ({sector})")
        
        logger.success("Portfolio optimization complete")
        
        return allocation
    
    def _equal_weight(self, n: int) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(n) / n
    
    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk parity optimization.
        
        Each asset contributes equally to portfolio risk.
        More robust than mean-variance as it doesn't rely on return estimates.
        """
        n = len(cov_matrix)
        
        def objective(weights):
            """Minimize sum of squared risk contribution differences."""
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_risk = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_risk
            avg_contrib = np.mean(risk_contrib)
            return np.sum((risk_contrib - avg_contrib) ** 2)
        
        # Constraints: weights sum to 1, all positive
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0.01, self.max_position) for _ in range(n)]
        
        # Initial guess: equal weight
        x0 = np.ones(n) / n
        
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            weights = np.ones(n) / n
        
        # Normalize
        weights = weights / np.sum(weights)
        return weights
    
    def _mean_variance(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Mean-variance optimization with risk aversion.
        """
        n = len(expected_returns)
        risk_aversion = 2.0  # Higher = more conservative
        
        def objective(weights):
            port_return = np.sum(expected_returns * weights)
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(port_return - risk_aversion * port_var)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_position) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else np.ones(n) / n
        return weights / np.sum(weights)
    
    def _max_sharpe(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Maximum Sharpe ratio portfolio.
        """
        n = len(expected_returns)
        
        def neg_sharpe(weights):
            port_return = np.sum(expected_returns * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / (port_vol + 1e-10)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_position) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(
            neg_sharpe, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else np.ones(n) / n
        return weights / np.sum(weights)
    
    def _min_volatility(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Minimum variance portfolio.
        """
        n = len(cov_matrix)
        
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_position) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else np.ones(n) / n
        return weights / np.sum(weights)
    
    def _hierarchical_risk_parity(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP).
        
        Uses hierarchical clustering to group similar assets,
        then allocates risk equally within and across groups.
        More robust to estimation error.
        """
        # Calculate correlation matrix
        corr = returns_df.corr()
        
        # Convert correlation to distance
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Hierarchical clustering
        linkage_matrix = linkage(dist, method='ward')
        sort_ix = leaves_list(linkage_matrix)
        
        # Reorder covariance matrix
        cov_matrix = returns_df.cov().values * 252
        sorted_cov = cov_matrix[sort_ix][:, sort_ix]
        
        # Recursive bisection
        weights = self._recursive_bisection(sorted_cov)
        
        # Reorder weights back
        final_weights = np.zeros(len(weights))
        for i, ix in enumerate(sort_ix):
            final_weights[ix] = weights[i]
        
        return final_weights / np.sum(final_weights)
    
    def _recursive_bisection(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP."""
        n = len(cov_matrix)
        weights = np.ones(n)
        
        def get_cluster_var(cov, idx):
            sub_cov = cov[np.ix_(idx, idx)]
            inv_var = 1 / np.diag(sub_cov)
            w = inv_var / np.sum(inv_var)
            return np.dot(w.T, np.dot(sub_cov, w))
        
        clusters = [[i for i in range(n)]]
        
        while len(clusters) < n:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    new_clusters.extend([cluster[:mid], cluster[mid:]])
                else:
                    new_clusters.append(cluster)
            clusters = new_clusters
        
        # Allocate weights based on inverse variance
        for i, cluster in enumerate(clusters):
            idx = cluster[0] if cluster else i
            var = cov_matrix[idx, idx]
            weights[idx] = 1 / var if var > 0 else 1
        
        return weights / np.sum(weights)
    
    def _apply_constraints(
        self,
        weights: np.ndarray,
        symbols: List[str],
        sector_map: Dict[str, str] = None
    ) -> np.ndarray:
        """Apply position and sector constraints."""
        # Cap individual positions
        weights = np.clip(weights, 0, self.max_position)
        
        # Apply sector constraints if sector map provided
        if sector_map:
            sector_weights = {}
            for i, sym in enumerate(symbols):
                sector = sector_map.get(sym, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
            
            # Scale down over-allocated sectors
            for sector, total_weight in sector_weights.items():
                if total_weight > self.max_sector:
                    scale = self.max_sector / total_weight
                    for i, sym in enumerate(symbols):
                        if sector_map.get(sym, 'Other') == sector:
                            weights[i] *= scale
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if port_vol > 0:
            marginal_risk = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_risk
            risk_contrib = risk_contrib / np.sum(risk_contrib)  # Normalize to %
        else:
            risk_contrib = weights
        
        return risk_contrib


def test_portfolio_optimization():
    """Test portfolio optimization with sample data."""
    print("\n" + "=" * 80)
    print("TESTING STEP 4: PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    # Run previous steps
    from step_1_data_collection import DataCollector
    from step_2_feature_engineering import FeatureEngineer
    from step_3_factor_analysis import FactorAnalyzer
    
    test_symbols = [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
        'TCS', 'INFY', 'RELIANCE', 'TATASTEEL', 'HINDUNILVR',
        'MARUTI', 'SUNPHARMA', 'LT', 'BHARTIARTL', 'ITC'
    ]
    
    # Step 1-3
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols=test_symbols, start_date='2022-01-01')
    
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)
    
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features, market_data.get('NIFTY50'))
    
    # Step 4: Portfolio optimization
    optimizer = PortfolioOptimizer()
    
    # Test different methods
    methods = ['equal', 'risk_parity', 'max_sharpe', 'min_volatility']
    results = {}
    
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method.upper()}")
        print(f"{'='*60}")
        
        allocation = optimizer.optimize(
            price_data=price_data,
            factor_scores=factor_scores,
            n_holdings=10,
            method=method,
            sector_map=config.STOCK_SECTOR_MAP
        )
        
        if allocation:
            results[method] = allocation
    
    # Compare methods
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)
    print(f"{'Method':<15} {'Return':>10} {'Volatility':>12} {'Sharpe':>8} {'Positions':>10}")
    print("-" * 60)
    
    for method, alloc in results.items():
        print(f"{method:<15} {alloc.expected_return:>10.2%} {alloc.expected_volatility:>12.2%} "
              f"{alloc.sharpe_ratio:>8.2f} {alloc.n_positions:>10}")
    
    # Validation tests
    print("\n" + "-" * 40)
    print("VALIDATION TESTS")
    print("-" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: All methods produce results
    tests_total += 1
    if len(results) == len(methods):
        print(f"✓ Test 1: All {len(methods)} methods successful - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 1: Only {len(results)}/{len(methods)} methods successful - FAILED")
    
    # Test 2: Weights sum to 1
    tests_total += 1
    weights_valid = True
    for method, alloc in results.items():
        weight_sum = sum(alloc.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            weights_valid = False
            break
    if weights_valid:
        print("✓ Test 2: Weights sum to 1 - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 2: Weights sum to 1 - FAILED")
    
    # Test 3: Max position constraint
    tests_total += 1
    position_valid = True
    for method, alloc in results.items():
        if max(alloc.weights.values()) > alloc.max_position + 0.01:
            position_valid = False
            break
    if position_valid:
        print(f"✓ Test 3: Max position <= {optimizer.max_position:.0%} - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 3: Max position constraint violated - FAILED")
    
    # Test 4: Sharpe ratio positive for best method
    tests_total += 1
    best_sharpe = max(alloc.sharpe_ratio for alloc in results.values())
    if best_sharpe > 0:
        print(f"✓ Test 4: Best Sharpe ratio > 0 ({best_sharpe:.2f}) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 4: Best Sharpe ratio <= 0 - FAILED")
    
    # Test 5: Risk parity has balanced risk contributions
    tests_total += 1
    if 'risk_parity' in results:
        rp_alloc = results['risk_parity']
        risk_contribs = list(rp_alloc.risk_contributions.values())
        if risk_contribs:
            cv = np.std(risk_contribs) / np.mean(risk_contribs) if np.mean(risk_contribs) > 0 else 1
            if cv < 0.5:  # CV < 50% is reasonably balanced
                print(f"✓ Test 5: Risk parity balanced (CV: {cv:.2f}) - PASSED")
                tests_passed += 1
            else:
                print(f"✗ Test 5: Risk parity not balanced (CV: {cv:.2f}) - FAILED")
        else:
            print("✗ Test 5: No risk contributions calculated - FAILED")
    else:
        print("✗ Test 5: Risk parity method failed - FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"TESTS: {tests_passed}/{tests_total} passed")
    print("=" * 40)
    
    return results


if __name__ == "__main__":
    test_portfolio_optimization()
