"""
================================================================================
EXPERIMENT TRACKER - Research Results Management
================================================================================
Track all experiments with:
- Configuration snapshots
- Results logging
- Statistical validation
- Comparison reports
- Version control

Every experiment is reproducible and comparable.
================================================================================
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from loguru import logger
from scipy import stats


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Strategy
    strategy_type: str  # 'factor', 'ml', 'hybrid'
    factors: List[str]
    factor_weights: Dict[str, float]

    # Universe
    universe: str  # 'NIFTY50', 'NIFTY100', 'custom'
    n_holdings: int

    # Portfolio
    optimization_method: str  # 'equal', 'risk_parity', 'mean_variance'
    rebalance_frequency: str  # 'monthly', 'quarterly'
    max_position: float
    max_sector: float

    # Backtest
    start_date: str
    end_date: str
    initial_capital: float
    transaction_cost: float

    # Model (if ML)
    model_type: Optional[str] = None
    model_params: Optional[Dict] = None
    features: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResults:
    """Experiment results."""
    # Returns
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float  # Alpha

    # Risk
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trading
    n_trades: int
    turnover: float  # Annual
    transaction_costs: float

    # By period
    monthly_returns: List[float] = field(default_factory=list)
    win_rate_monthly: float = 0
    best_month: float = 0
    worst_month: float = 0

    # Factor attribution (optional)
    factor_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StatisticalValidation:
    """Statistical validation of results."""
    # Sample size
    n_periods: int
    n_trades: int

    # Significance tests
    t_stat: float
    p_value: float
    is_significant: bool  # p < 0.05

    # Confidence intervals
    return_ci_lower: float
    return_ci_upper: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    # Robustness
    information_ratio: float
    hit_rate: float  # % of periods beating benchmark

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Experiment:
    """Complete experiment record."""
    # Metadata
    experiment_id: str
    name: str
    description: str
    timestamp: datetime
    status: str  # 'running', 'completed', 'failed'

    # Components
    config: ExperimentConfig
    results: Optional[ExperimentResults] = None
    validation: Optional[StatisticalValidation] = None

    # Conclusion
    conclusion: str = ""
    next_steps: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'timestamp': str(self.timestamp),
            'status': self.status,
            'config': self.config.to_dict(),
            'results': self.results.to_dict() if self.results else None,
            'validation': self.validation.to_dict() if self.validation else None,
            'conclusion': self.conclusion,
            'next_steps': self.next_steps,
            'tags': self.tags
        }


class ExperimentTracker:
    """
    Track and manage all experiments.
    """

    def __init__(self, experiments_dir: str = None):
        if experiments_dir is None:
            experiments_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'experiments'
            )
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Index file for quick lookup
        self.index_file = self.experiments_dir / 'index.json'
        self.index = self._load_index()

        logger.info(f"ExperimentTracker initialized: {self.experiments_dir}")
        logger.info(f"Loaded {len(self.index)} existing experiments")

    def _load_index(self) -> Dict:
        """Load experiment index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save experiment index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, default=str)

    def create_experiment(
        self,
        name: str,
        description: str,
        config: ExperimentConfig,
        tags: List[str] = None
    ) -> Experiment:
        """
        Create a new experiment.
        """
        # Generate unique ID
        config_hash = config.get_hash()
        timestamp = datetime.now()
        exp_id = f"EXP_{timestamp.strftime('%Y%m%d_%H%M')}_{config_hash}"

        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            timestamp=timestamp,
            status='running',
            config=config,
            tags=tags or []
        )

        # Save to disk
        self._save_experiment(experiment)

        # Update index
        self.index[exp_id] = {
            'name': name,
            'timestamp': str(timestamp),
            'status': 'running',
            'config_hash': config_hash
        }
        self._save_index()

        logger.info(f"Created experiment: {exp_id}")
        return experiment

    def update_results(
        self,
        experiment: Experiment,
        results: ExperimentResults,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Experiment:
        """
        Update experiment with results and compute validation.
        """
        experiment.results = results

        # Compute statistical validation
        if results.monthly_returns:
            validation = self._compute_validation(
                results.monthly_returns,
                benchmark_returns
            )
            experiment.validation = validation

        experiment.status = 'completed'

        # Save
        self._save_experiment(experiment)

        # Update index
        self.index[experiment.experiment_id]['status'] = 'completed'
        if results:
            self.index[experiment.experiment_id]['sharpe'] = results.sharpe_ratio
            self.index[experiment.experiment_id]['annual_return'] = results.annual_return
        self._save_index()

        logger.info(f"Updated experiment: {experiment.experiment_id}")
        return experiment

    def _compute_validation(
        self,
        returns: List[float],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> StatisticalValidation:
        """
        Compute statistical validation of results.
        """
        returns = np.array(returns)
        n = len(returns)

        warnings = []

        # Minimum sample checks
        if n < 12:
            warnings.append(f"Only {n} periods - need 12+ for significance")
        if n < 36:
            warnings.append(f"Only {n} periods - 36+ recommended for reliability")

        # T-test against zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        is_significant = p_value < 0.05 and np.mean(returns) > 0

        # Confidence intervals for returns
        mean_return = np.mean(returns)
        se_return = np.std(returns) / np.sqrt(n)
        return_ci = stats.t.interval(0.95, n-1, loc=mean_return, scale=se_return)

        # Sharpe ratio and its CI
        annual_return = (1 + mean_return) ** 12 - 1
        annual_vol = np.std(returns) * np.sqrt(12)
        sharpe = (annual_return - 0.06) / annual_vol if annual_vol > 0 else 0

        # Sharpe CI (approximation)
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n)
        sharpe_ci = (sharpe - 1.96 * sharpe_se, sharpe + 1.96 * sharpe_se)

        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == n:
            excess = returns - benchmark_returns
            ir = np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0
            hit_rate = np.mean(excess > 0)
        else:
            ir = 0
            hit_rate = np.mean(returns > 0)

        # Check for concerning patterns
        if np.mean(returns[-6:]) < np.mean(returns[:6]):
            warnings.append("Performance declining in recent periods")

        negative_streak = 0
        max_negative_streak = 0
        for r in returns:
            if r < 0:
                negative_streak += 1
                max_negative_streak = max(max_negative_streak, negative_streak)
            else:
                negative_streak = 0

        if max_negative_streak >= 4:
            warnings.append(f"Max {max_negative_streak} consecutive losing periods")

        return StatisticalValidation(
            n_periods=n,
            n_trades=0,  # Set separately
            t_stat=float(t_stat),
            p_value=float(p_value),
            is_significant=is_significant,
            return_ci_lower=float(return_ci[0]),
            return_ci_upper=float(return_ci[1]),
            sharpe_ci_lower=float(sharpe_ci[0]),
            sharpe_ci_upper=float(sharpe_ci[1]),
            information_ratio=float(ir),
            hit_rate=float(hit_rate),
            warnings=warnings
        )

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        exp_dir = self.experiments_dir / experiment.experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save main file
        with open(exp_dir / 'experiment.json', 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from disk."""
        exp_dir = self.experiments_dir / experiment_id
        exp_file = exp_dir / 'experiment.json'

        if not exp_file.exists():
            logger.warning(f"Experiment not found: {experiment_id}")
            return None

        with open(exp_file, 'r') as f:
            data = json.load(f)

        # Reconstruct experiment
        config = ExperimentConfig(**data['config'])
        results = ExperimentResults(**data['results']) if data['results'] else None
        validation = StatisticalValidation(**data['validation']) if data['validation'] else None

        return Experiment(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data['description'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            status=data['status'],
            config=config,
            results=results,
            validation=validation,
            conclusion=data.get('conclusion', ''),
            next_steps=data.get('next_steps', ''),
            tags=data.get('tags', [])
        )

    def list_experiments(
        self,
        status: str = None,
        tags: List[str] = None,
        sort_by: str = 'timestamp'
    ) -> List[Dict]:
        """List all experiments with optional filters."""
        experiments = []

        for exp_id, info in self.index.items():
            if status and info.get('status') != status:
                continue

            experiments.append({
                'experiment_id': exp_id,
                **info
            })

        # Sort
        if sort_by == 'timestamp':
            experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == 'sharpe':
            experiments.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
        elif sort_by == 'annual_return':
            experiments.sort(key=lambda x: x.get('annual_return', 0), reverse=True)

        return experiments

    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple experiments side by side.
        """
        data = []

        for exp_id in experiment_ids:
            exp = self.load_experiment(exp_id)
            if exp is None:
                continue

            row = {
                'experiment_id': exp_id,
                'name': exp.name,
                'status': exp.status,
            }

            # Config
            row['strategy'] = exp.config.strategy_type
            row['n_holdings'] = exp.config.n_holdings
            row['optimization'] = exp.config.optimization_method
            row['factors'] = ', '.join(exp.config.factors)

            # Results
            if exp.results:
                row['annual_return'] = f"{exp.results.annual_return:.2%}"
                row['sharpe'] = f"{exp.results.sharpe_ratio:.2f}"
                row['max_drawdown'] = f"{exp.results.max_drawdown:.2%}"
                row['volatility'] = f"{exp.results.volatility:.2%}"

            # Validation
            if exp.validation:
                row['p_value'] = f"{exp.validation.p_value:.3f}"
                row['significant'] = '✓' if exp.validation.is_significant else '✗'
                row['warnings'] = len(exp.validation.warnings)

            data.append(row)

        return pd.DataFrame(data)

    def generate_report(self, experiment_id: str) -> str:
        """Generate detailed report for an experiment."""
        exp = self.load_experiment(experiment_id)
        if exp is None:
            return f"Experiment {experiment_id} not found"

        lines = []
        lines.append("=" * 80)
        lines.append(f"EXPERIMENT REPORT: {exp.name}")
        lines.append("=" * 80)
        lines.append(f"ID: {exp.experiment_id}")
        lines.append(f"Date: {exp.timestamp}")
        lines.append(f"Status: {exp.status}")
        lines.append(f"Tags: {', '.join(exp.tags)}")
        lines.append("")

        # Description
        lines.append("-" * 40)
        lines.append("DESCRIPTION")
        lines.append("-" * 40)
        lines.append(exp.description)
        lines.append("")

        # Configuration
        lines.append("-" * 40)
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Strategy: {exp.config.strategy_type}")
        lines.append(f"Universe: {exp.config.universe}")
        lines.append(f"Holdings: {exp.config.n_holdings}")
        lines.append(f"Optimization: {exp.config.optimization_method}")
        lines.append(f"Rebalance: {exp.config.rebalance_frequency}")
        lines.append(f"Factors: {exp.config.factors}")
        lines.append(f"Factor Weights: {exp.config.factor_weights}")
        lines.append(f"Period: {exp.config.start_date} to {exp.config.end_date}")
        lines.append("")

        # Results
        if exp.results:
            lines.append("-" * 40)
            lines.append("RESULTS")
            lines.append("-" * 40)
            lines.append(f"Total Return: {exp.results.total_return:.2%}")
            lines.append(f"Annual Return: {exp.results.annual_return:.2%}")
            lines.append(f"Benchmark: {exp.results.benchmark_return:.2%}")
            lines.append(f"Excess Return: {exp.results.excess_return:.2%}")
            lines.append("")
            lines.append(f"Volatility: {exp.results.volatility:.2%}")
            lines.append(f"Sharpe Ratio: {exp.results.sharpe_ratio:.2f}")
            lines.append(f"Sortino Ratio: {exp.results.sortino_ratio:.2f}")
            lines.append(f"Max Drawdown: {exp.results.max_drawdown:.2%}")
            lines.append("")
            lines.append(f"Win Rate (Monthly): {exp.results.win_rate_monthly:.1%}")
            lines.append(f"Best Month: {exp.results.best_month:.2%}")
            lines.append(f"Worst Month: {exp.results.worst_month:.2%}")
            lines.append("")

        # Validation
        if exp.validation:
            lines.append("-" * 40)
            lines.append("STATISTICAL VALIDATION")
            lines.append("-" * 40)
            lines.append(f"Sample Size: {exp.validation.n_periods} periods")
            lines.append(f"T-Statistic: {exp.validation.t_stat:.2f}")
            lines.append(f"P-Value: {exp.validation.p_value:.4f}")
            lines.append(f"Significant: {'YES' if exp.validation.is_significant else 'NO'}")
            lines.append("")
            lines.append(f"Return 95% CI: [{exp.validation.return_ci_lower:.2%}, {exp.validation.return_ci_upper:.2%}]")
            lines.append(f"Sharpe 95% CI: [{exp.validation.sharpe_ci_lower:.2f}, {exp.validation.sharpe_ci_upper:.2f}]")
            lines.append(f"Information Ratio: {exp.validation.information_ratio:.2f}")
            lines.append(f"Hit Rate: {exp.validation.hit_rate:.1%}")
            lines.append("")

            if exp.validation.warnings:
                lines.append("WARNINGS:")
                for warning in exp.validation.warnings:
                    lines.append(f"  ⚠️ {warning}")
                lines.append("")

        # Conclusion
        if exp.conclusion:
            lines.append("-" * 40)
            lines.append("CONCLUSION")
            lines.append("-" * 40)
            lines.append(exp.conclusion)
            lines.append("")

        if exp.next_steps:
            lines.append("NEXT STEPS:")
            lines.append(exp.next_steps)
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_best_experiments(
        self,
        metric: str = 'sharpe',
        n: int = 5
    ) -> List[Dict]:
        """Get top N experiments by metric."""
        experiments = self.list_experiments(status='completed', sort_by=metric)
        return experiments[:n]


if __name__ == "__main__":
    # Test experiment tracker
    tracker = ExperimentTracker()

    # Create test config
    config = ExperimentConfig(
        strategy_type='factor',
        factors=['value', 'momentum', 'quality'],
        factor_weights={'value': 0.33, 'momentum': 0.34, 'quality': 0.33},
        universe='NIFTY50',
        n_holdings=20,
        optimization_method='risk_parity',
        rebalance_frequency='monthly',
        max_position=0.10,
        max_sector=0.30,
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=1000000,
        transaction_cost=0.001
    )

    # Create experiment
    exp = tracker.create_experiment(
        name="Test: Value+Momentum+Quality",
        description="Testing combined factor strategy with risk parity",
        config=config,
        tags=['factor', 'test']
    )

    print(f"Created: {exp.experiment_id}")

    # Simulate results
    results = ExperimentResults(
        total_return=0.45,
        annual_return=0.12,
        benchmark_return=0.10,
        excess_return=0.02,
        volatility=0.15,
        sharpe_ratio=0.80,
        sortino_ratio=1.10,
        max_drawdown=-0.18,
        max_drawdown_duration=45,
        n_trades=120,
        turnover=1.2,
        transaction_costs=0.012,
        monthly_returns=[0.02, -0.01, 0.03, 0.01, -0.02, 0.04] * 6,
        win_rate_monthly=0.58,
        best_month=0.08,
        worst_month=-0.06
    )

    exp = tracker.update_results(exp, results)

    # Generate report
    print(tracker.generate_report(exp.experiment_id))
