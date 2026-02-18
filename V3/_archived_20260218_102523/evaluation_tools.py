"""
================================================================================
V3 NEURAL MODELS - EVALUATION AND ANALYSIS TOOLS
================================================================================
Tools for evaluating and comparing traditional (XGB, LGB) and neural models.

Provides:
- Statistical comparison of model performance
- Ensemble optimization analysis
- Feature importance aggregation
- Performance visualization-ready data
- Cross-model accuracy trends
- Win rate analysis
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from loguru import logger


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    dir_accuracy: float
    win_rate_long: float
    win_rate_short: float
    sharpe: float
    profit_factor: float
    rmse: float
    mae: float
    n_samples: int
    
    def to_dict(self):
        return {
            'model': self.model_name,
            'dir_accuracy': self.dir_accuracy,
            'win_rate_long': self.win_rate_long,
            'win_rate_short': self.win_rate_short,
            'sharpe': self.sharpe,
            'profit_factor': self.profit_factor,
            'rmse': self.rmse,
            'mae': self.mae,
            'n_samples': self.n_samples,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation of all models on a dataset."""
    symbol: str
    window_config: str
    models: Dict[str, ModelMetrics]
    ensemble_weights: Dict[str, float]
    best_model: str
    timestamp: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        rows = []
        for model_name, metrics in self.models.items():
            row = metrics.to_dict()
            row['symbol'] = self.symbol
            row['window'] = self.window_config
            row['ensemble_weight'] = self.ensemble_weights.get(model_name, 0)
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# PERFORMANCE COMPARISONS
# ============================================================================

class ModelComparator:
    """Compare performance across multiple models."""
    
    @staticmethod
    def rank_models_by_metric(
        results: Dict[str, ModelMetrics],
        metric: str = 'dir_accuracy'
    ) -> List[Tuple[str, float]]:
        """
        Rank models by a specific metric.
        
        Args:
            results: Dict[model_name] -> ModelMetrics
            metric: Metric to rank by ('dir_accuracy', 'sharpe', 'profit_factor', etc.)
        
        Returns:
            List of (model_name, metric_value) sorted descending
        """
        rankings = []
        for model_name, metrics in results.items():
            value = getattr(metrics, metric, 0)
            rankings.append((model_name, value))
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def compute_ensemble_statistics(
        results: Dict[str, ModelMetrics],
        metric: str = 'dir_accuracy'
    ) -> Dict:
        """
        Compute statistical summary of model performance.
        
        Args:
            results: Dict[model_name] -> ModelMetrics
            metric: Metric to analyze
        
        Returns:
            Dict with mean, std, min, max, etc.
        """
        values = []
        for metrics in results.values():
            values.append(getattr(metrics, metric, 0))
        
        values = np.array(values)
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
        }
    
    @staticmethod
    def compute_improvement(
        baseline_value: float,
        improved_value: float
    ) -> float:
        """
        Compute percentage improvement from baseline to improved value.
        
        Args:
            baseline_value: Original/baseline metric value
            improved_value: New/improved metric value
        
        Returns:
            Percentage improvement
        """
        if baseline_value == 0:
            return float('inf') if improved_value > 0 else 0
        return ((improved_value - baseline_value) / abs(baseline_value)) * 100
    
    @staticmethod
    def compare_model_groups(
        traditional_results: Dict[str, ModelMetrics],
        neural_results: Dict[str, ModelMetrics],
        metric: str = 'dir_accuracy'
    ) -> Dict:
        """
        Compare performance between traditional and neural model groups.
        
        Args:
            traditional_results: Dict of XGB, LGB, etc.
            neural_results: Dict of neural models
            metric: Metric to compare on
        
        Returns:
            Dict with comparison statistics
        """
        
        traditional_values = [getattr(m, metric) for m in traditional_results.values()]
        neural_values = [getattr(m, metric) for m in neural_results.values()]
        
        traditional_mean = np.mean(traditional_values)
        neural_mean = np.mean(neural_values)
        
        improvement = ModelComparator.compute_improvement(traditional_mean, neural_mean)
        
        return {
            'traditional_mean': float(traditional_mean),
            'traditional_std': float(np.std(traditional_values)),
            'neural_mean': float(neural_mean),
            'neural_std': float(np.std(neural_values)),
            'improvement_percent': improvement,
            'better_group': 'neural' if neural_mean > traditional_mean else 'traditional',
        }


# ============================================================================
# ENSEMBLE ANALYSIS
# ============================================================================

class EnsembleAnalyzer:
    """Analyze ensemble composition and optimization."""
    
    @staticmethod
    def analyze_ensemble_weights(
        weights: Dict[str, float]
    ) -> Dict:
        """
        Analyze ensemble weight distribution.
        
        Args:
            weights: Dict[model_name] -> weight
        
        Returns:
            Analysis dict with concentration, diversity metrics
        """
        weights_array = np.array(list(weights.values()))
        
        # Gini coefficient (0=equal, 1=concentrated)
        sorted_weights = np.sort(weights_array)
        n = len(sorted_weights)
        gini = (2 * np.sum(np.arange(1, n+1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n+1)/n
        
        # Herfindahl index (portfolio concentration)
        hhi = np.sum(weights_array ** 2)
        
        # Count models with >5% weight
        major_models = sum(1 for w in weights_array if w > 0.05)
        
        return {
            'gini_coefficient': max(0, gini),  # Ensure non-negative
            'herfindahl_index': float(hhi),
            'n_major_models': major_models,
            'n_total_models': len(weights),
            'max_weight': float(np.max(weights_array)),
            'min_weight': float(np.min(weights_array[weights_array > 0.0001])),  # Ignore near-zero weights
            'top_model': max(weights, key=weights.get),
            'top_weight': float(np.max(weights_array)),
        }
    
    @staticmethod
    def compare_ensemble_compositions(
        ensemble1_weights: Dict[str, float],
        ensemble2_weights: Dict[str, float],
    ) -> Dict:
        """
        Compare two ensemble compositions (e.g., traditional vs super ensemble).
        
        Args:
            ensemble1_weights: First ensemble weights
            ensemble2_weights: Second ensemble weights
        
        Returns:
            Comparison metrics
        """
        # Get all model names
        all_models = set(ensemble1_weights.keys()) | set(ensemble2_weights.keys())
        
        weight_diffs = []
        for model in all_models:
            w1 = ensemble1_weights.get(model, 0)
            w2 = ensemble2_weights.get(model, 0)
            weight_diffs.append(abs(w1 - w2))
        
        # Models that changed significantly (>10% absolute difference)
        significant_changes = sum(1 for d in weight_diffs if d > 0.1)
        
        return {
            'n_common_models': len(all_models),
            'mean_weight_change': float(np.mean(weight_diffs)),
            'max_weight_change': float(np.max(weight_diffs)),
            'significant_changes': significant_changes,
            'weight_shift_magnitude': float(np.sum(weight_diffs)),
        }


# ============================================================================
# ACCURACY ANALYSIS
# ============================================================================

class AccuracyAnalyzer:
    """Analyze and compare directional accuracy and win rates."""
    
    @staticmethod
    def compute_win_rate_statistics(
        models_results: Dict[str, ModelMetrics]
    ) -> Dict:
        """
        Compute statistics on win rates across models.
        
        Args:
            models_results: Dict[model_name] -> ModelMetrics
        
        Returns:
            Statistics on long and short win rates
        """
        long_rates = [m.win_rate_long for m in models_results.values()]
        short_rates = [m.win_rate_short for m in models_results.values()]
        
        return {
            'long_win_rate_mean': float(np.mean(long_rates)),
            'long_win_rate_std': float(np.std(long_rates)),
            'short_win_rate_mean': float(np.mean(short_rates)),
            'short_win_rate_std': float(np.std(short_rates)),
            'avg_win_rate': float((np.mean(long_rates) + np.mean(short_rates)) / 2),
        }
    
    @staticmethod
    def identify_biased_models(
        models_results: Dict[str, ModelMetrics],
        threshold: float = 0.1
    ) -> Dict[str, str]:
        """
        Identify models with directional bias (overpredicting UP or DOWN).
        
        A model is considered biased if win_rate_long and win_rate_short
        differ significantly (threshold = 0.1 means 10% difference).
        
        Args:
            models_results: Dict[model_name] -> ModelMetrics
            threshold: Minimum difference to flag as biased
        
        Returns:
            Dict[model_name] -> bias_direction ('LONG_BIASED', 'SHORT_BIASED', or 'BALANCED')
        """
        biases = {}
        for model_name, metrics in models_results.items():
            diff = metrics.win_rate_long - metrics.win_rate_short
            if diff > threshold:
                biases[model_name] = 'LONG_BIASED'
            elif diff < -threshold:
                biases[model_name] = 'SHORT_BIASED'
            else:
                biases[model_name] = 'BALANCED'
        return biases


# ============================================================================
# PROFIT ANALYSIS
# ============================================================================

class ProfitAnalyzer:
    """Analyze profitability metrics."""
    
    @staticmethod
    def evaluate_sharpe_ratios(
        models_results: Dict[str, ModelMetrics]
    ) -> Dict:
        """
        Analyze Sharpe ratio distribution.
        
        Args:
            models_results: Dict[model_name] -> ModelMetrics
        
        Returns:
            Sharpe ratio statistics
        """
        sharpes = [m.sharpe for m in models_results.values()]
        positive_sharpes = [s for s in sharpes if s > 0]
        
        return {
            'mean_sharpe': float(np.mean(sharpes)),
            'std_sharpe': float(np.std(sharpes)),
            'max_sharpe': float(np.max(sharpes)),
            'min_sharpe': float(np.min(sharpes)),
            'positive_sharpe_count': len(positive_sharpes),
            'positive_sharpe_mean': float(np.mean(positive_sharpes)) if positive_sharpes else 0.0,
        }
    
    @staticmethod
    def evaluate_profit_factors(
        models_results: Dict[str, ModelMetrics],
        min_threshold: float = 1.0,
        good_threshold: float = 2.0
    ) -> Dict:
        """
        Analyze profit factor distribution.
        
        Profit Factor > 1.0 means net profitable
        Profit Factor > 2.0 is considered good
        
        Args:
            models_results: Dict[model_name] -> ModelMetrics
            min_threshold: Minimum acceptable profit factor
            good_threshold: Good profit factor threshold
        
        Returns:
            Profit factor analysis
        """
        pfs = [m.profit_factor for m in models_results.values()]
        pfs_valid = [pf for pf in pfs if np.isfinite(pf)]
        
        if not pfs_valid:
            return {
                'mean_profit_factor': 0.0,
                'profitable_models': 0,
                'good_models': 0,
            }
        
        profitable = sum(1 for pf in pfs_valid if pf > min_threshold)
        good = sum(1 for pf in pfs_valid if pf > good_threshold)
        
        return {
            'mean_profit_factor': float(np.mean(pfs_valid)),
            'std_profit_factor': float(np.std(pfs_valid)),
            'min_profit_factor': float(np.min(pfs_valid)),
            'max_profit_factor': float(np.max(pfs_valid)),
            'profitable_models': profitable,
            'good_models': good,
            'n_total_models': len(pfs_valid),
        }


# ============================================================================
# AGGREGATE REPORTING
# ============================================================================

class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    @staticmethod
    def create_model_comparison_report(
        evaluation_results: List[EvaluationResult],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive model comparison report.
        
        Args:
            evaluation_results: List of EvaluationResult objects
            output_path: Optional path to save CSV
        
        Returns:
            DataFrame with all metrics across symbols and windows
        """
        dfs = []
        for result in evaluation_results:
            df = result.to_dataframe()
            dfs.append(df)
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        if output_path:
            full_df.to_csv(output_path, index=False)
            logger.info(f"Saved comparison report to {output_path}")
        
        return full_df
    
    @staticmethod
    def create_aggregated_summary(
        comparison_df: pd.DataFrame
    ) -> Dict:
        """
        Create aggregated summary across all runs.
        
        Args:
            comparison_df: Model comparison DataFrame
        
        Returns:
            Summary dict with key metrics by model
        """
        summary = {}
        
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]
            
            summary[model] = {
                'n_runs': len(model_data),
                'dir_accuracy_mean': float(model_data['dir_accuracy'].mean()),
                'dir_accuracy_std': float(model_data['dir_accuracy'].std()),
                'dir_accuracy_min': float(model_data['dir_accuracy'].min()),
                'dir_accuracy_max': float(model_data['dir_accuracy'].max()),
                'win_rate_long_mean': float(model_data['win_rate_long'].mean()),
                'sharpe_mean': float(model_data['sharpe'].mean()),
                'profit_factor_mean': float(model_data['profit_factor'].mean()),
            }
        
        return summary
    
    @staticmethod
    def identify_best_model_combination(
        comparison_df: pd.DataFrame,
        metric: str = 'dir_accuracy'
    ) -> Dict:
        """
        Identify best performing model overall.
        
        Args:
            comparison_df: Model comparison DataFrame
            metric: Metric to optimize for
        
        Returns:
            Best model info
        """
        best_row = comparison_df.loc[comparison_df[metric].idxmax()]
        
        return {
            'model': best_row['model'],
            'symbol': best_row['symbol'],
            'window': best_row['window'],
            'metric': metric,
            'value': float(best_row[metric]),
            'run_id': best_row.name,
        }


# ============================================================================
# TREND ANALYSIS
# ============================================================================

class TrendAnalyzer:
    """Analyze trends in model performance."""
    
    @staticmethod
    def compute_performance_by_symbol(
        comparison_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compute average performance by stock symbol.
        
        Args:
            comparison_df: Model comparison DataFrame
        
        Returns:
            Dict[symbol] -> Dict of average metrics
        """
        summary = {}
        
        for symbol in comparison_df['symbol'].unique():
            symbol_data = comparison_df[comparison_df['symbol'] == symbol]
            
            summary[symbol] = {
                'n_models': len(symbol_data),
                'dir_accuracy_mean': float(symbol_data['dir_accuracy'].mean()),
                'dir_accuracy_std': float(symbol_data['dir_accuracy'].std()),
                'sharpe_mean': float(symbol_data['sharpe'].mean()),
                'profit_factor_mean': float(symbol_data['profit_factor'].mean()),
                'best_model': symbol_data.loc[symbol_data['dir_accuracy'].idxmax(), 'model'],
                'best_accuracy': float(symbol_data['dir_accuracy'].max()),
            }
        
        return summary
    
    @staticmethod
    def compute_performance_by_window(
        comparison_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compute average performance by window configuration.
        
        Args:
            comparison_df: Model comparison DataFrame
        
        Returns:
            Dict[window] -> Dict of average metrics
        """
        summary = {}
        
        for window in comparison_df['window'].unique():
            window_data = comparison_df[comparison_df['window'] == window]
            
            summary[str(window)] = {
                'n_models': len(window_data),
                'dir_accuracy_mean': float(window_data['dir_accuracy'].mean()),
                'dir_accuracy_std': float(window_data['dir_accuracy'].std()),
                'sharpe_mean': float(window_data['sharpe'].mean()),
                'profit_factor_mean': float(window_data['profit_factor'].mean()),
                'best_model': window_data.loc[window_data['dir_accuracy'].idxmax(), 'model'],
                'best_accuracy': float(window_data['dir_accuracy'].max()),
            }
        
        return summary


if __name__ == '__main__':
    logger.info("Evaluation tools module loaded successfully")
