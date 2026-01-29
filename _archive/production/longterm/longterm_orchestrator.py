"""
================================================================================
LONG-TERM EQUITY ORCHESTRATOR
================================================================================
Complete pipeline for long-term factor-based portfolio management:

1. DATA COLLECTION - Download historical OHLCV
2. FACTOR COMPUTATION - Calculate Value, Momentum, Quality, Low-Vol
3. STOCK RANKING - Rank universe by factor scores
4. PORTFOLIO OPTIMIZATION - Construct optimal portfolio
5. BACKTESTING - Walk-forward validation
6. SIGNAL GENERATION - Buy/Sell/Hold signals
7. EXPERIMENT TRACKING - Log all results

Run modes:
- backtest: Historical simulation
- paper: Paper trading simulation
- live: Real trading (with broker integration)
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from production.longterm.factor_engine import FactorEngine, FactorScores
from production.longterm.portfolio_optimizer import (
    PortfolioOptimizer, PortfolioAllocation, LongTermBacktester
)
from production.longterm.experiment_tracker import (
    ExperimentTracker, ExperimentConfig, ExperimentResults, Experiment
)


class LongTermOrchestrator:
    """
    Orchestrator for long-term equity portfolio strategy.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        n_holdings: int = 20,
        optimization_method: str = 'risk_parity',
        rebalance_frequency: str = 'monthly',
        initial_capital: float = 1000000,
        results_dir: str = None
    ):
        """
        Initialize long-term orchestrator.

        Args:
            symbols: Stock universe (defaults to NIFTY50)
            n_holdings: Number of stocks to hold
            optimization_method: 'equal', 'risk_parity', 'mean_variance', 'max_sharpe'
            rebalance_frequency: 'monthly' or 'quarterly'
            initial_capital: Starting capital
            results_dir: Directory for results
        """
        # Universe
        self.symbols = symbols or config.ALL_STOCKS
        self.n_holdings = n_holdings

        # Strategy
        self.optimization_method = optimization_method
        self.rebalance_frequency = rebalance_frequency

        # Capital
        self.initial_capital = initial_capital

        # Directories
        self.results_dir = Path(results_dir or os.path.join(config.BASE_DIR, 'longterm_results'))
        self._setup_directories()

        # Components
        self.factor_engine = FactorEngine()
        self.portfolio_optimizer = PortfolioOptimizer(
            max_position=0.10,
            min_position=0.02,
            max_sector=0.30
        )
        self.backtester = LongTermBacktester(
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency
        )
        self.experiment_tracker = ExperimentTracker(
            str(self.results_dir / 'experiments')
        )

        # Data caches
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.factor_scores: List[FactorScores] = []
        self.current_allocation: Optional[PortfolioAllocation] = None
        self.current_experiment: Optional[Experiment] = None

        # Portfolio state
        self.portfolio_weights: Dict[str, float] = {}
        self.portfolio_value = initial_capital

        logger.info(f"LongTermOrchestrator initialized")
        logger.info(f"Universe: {len(self.symbols)} stocks")
        logger.info(f"Holdings: {n_holdings}, Method: {optimization_method}")

    def _setup_directories(self):
        """Create result directories."""
        dirs = [
            self.results_dir,
            self.results_dir / 'data',
            self.results_dir / 'factors',
            self.results_dir / 'portfolios',
            self.results_dir / 'backtests',
            self.results_dir / 'signals',
            self.results_dir / 'experiments',
            self.results_dir / 'reports'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ==================== STAGE 1: DATA COLLECTION ====================

    def collect_data(
        self,
        days: int = 1500,  # ~6 years
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Stage 1: Collect historical price data.
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA COLLECTION")
        logger.info("=" * 60)

        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for symbol in self.symbols:
            logger.info(f"Downloading {symbol}...")

            try:
                ticker = f"{symbol}.NS"
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )

                if len(df) > 0:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    df.columns = [c.lower() for c in df.columns]
                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    df['symbol'] = symbol

                    self.price_data[symbol] = df

                    # Save
                    save_path = self.results_dir / 'data' / f'{symbol}.csv'
                    df.to_csv(save_path)

                    logger.info(f"  {symbol}: {len(df)} rows")

            except Exception as e:
                logger.error(f"  {symbol}: Failed - {e}")

        logger.success(f"Data collection complete: {len(self.price_data)} symbols")
        return self.price_data

    # ==================== STAGE 2: FACTOR COMPUTATION ====================

    def compute_factors(self) -> List[FactorScores]:
        """
        Stage 2: Compute factor scores for all stocks.
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: FACTOR COMPUTATION")
        logger.info("=" * 60)

        if not self.price_data:
            logger.error("No price data. Run collect_data() first.")
            return []

        # Compute factors
        self.factor_scores = self.factor_engine.compute_all_factors(self.price_data)

        # Convert to DataFrame and save
        factors_df = self.factor_engine.to_dataframe(self.factor_scores)
        factors_df.to_csv(self.results_dir / 'factors' / 'factor_scores.csv', index=False)

        # Generate report
        report = self.factor_engine.generate_report(self.factor_scores)
        logger.info("\n" + report)

        with open(self.results_dir / 'factors' / 'factor_report.txt', 'w') as f:
            f.write(report)

        logger.success(f"Factor computation complete: {len(self.factor_scores)} stocks scored")
        return self.factor_scores

    # ==================== STAGE 3: PORTFOLIO OPTIMIZATION ====================

    def optimize_portfolio(
        self,
        factor_weights: Dict[str, float] = None
    ) -> PortfolioAllocation:
        """
        Stage 3: Optimize portfolio weights.
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: PORTFOLIO OPTIMIZATION")
        logger.info("=" * 60)

        if not self.factor_scores:
            logger.error("No factor scores. Run compute_factors() first.")
            return None

        # Set factor weights
        if factor_weights:
            self.factor_engine.set_factor_weights(factor_weights)

        # Get top stocks
        top_stocks = self.factor_engine.get_top_stocks(
            self.factor_scores,
            n=self.n_holdings,
            factor='combined'
        )

        top_symbols = [s.symbol for s in top_stocks]
        logger.info(f"Top {self.n_holdings} stocks: {top_symbols}")

        # Get price data for optimization
        available_symbols = [s for s in top_symbols if s in self.price_data]

        if len(available_symbols) < self.n_holdings:
            logger.warning(f"Only {len(available_symbols)} stocks have price data")

        # Calculate expected returns and covariance
        returns_data = {}
        for sym in available_symbols:
            df = self.price_data[sym]
            returns_data[sym] = df['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 60:
            logger.error("Insufficient data for optimization")
            return None

        expected_returns = returns_df.mean().values * 252  # Annualize
        cov_matrix = returns_df.cov().values * 252

        # Optimize
        self.current_allocation = self.portfolio_optimizer.optimize(
            symbols=available_symbols,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            method=self.optimization_method,
            sector_map=config.STOCK_SECTOR_MAP
        )

        # Log results
        logger.info(f"\nOptimized Portfolio ({self.optimization_method}):")
        logger.info(f"  Expected Return: {self.current_allocation.expected_return:.2%}")
        logger.info(f"  Volatility: {self.current_allocation.expected_volatility:.2%}")
        logger.info(f"  Sharpe Ratio: {self.current_allocation.sharpe_ratio:.2f}")
        logger.info(f"  Positions: {self.current_allocation.n_positions}")

        logger.info("\nWeights:")
        for sym, weight in sorted(self.current_allocation.weights.items(),
                                  key=lambda x: x[1], reverse=True):
            logger.info(f"  {sym:12}: {weight:.2%}")

        # Save
        with open(self.results_dir / 'portfolios' / 'current_allocation.json', 'w') as f:
            json.dump(self.current_allocation.to_dict(), f, indent=2, default=str)

        self.portfolio_weights = self.current_allocation.weights

        logger.success("Portfolio optimization complete")
        return self.current_allocation

    # ==================== STAGE 4: BACKTESTING ====================

    def run_backtest(
        self,
        start_date: str = '2020-01-01',
        end_date: str = None,
        experiment_name: str = None
    ) -> Dict:
        """
        Stage 4: Run historical backtest.
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: BACKTESTING")
        logger.info("=" * 60)

        if not self.price_data or not self.factor_scores:
            logger.error("Run collect_data() and compute_factors() first")
            return {}

        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        # Create experiment
        exp_config = ExperimentConfig(
            strategy_type='factor',
            factors=['value', 'momentum', 'quality', 'low_vol', 'sentiment'],  # 5 factors
            factor_weights=self.factor_engine.factor_weights,
            universe=f"Custom ({len(self.symbols)} stocks)",
            n_holdings=self.n_holdings,
            optimization_method=self.optimization_method,
            rebalance_frequency=self.rebalance_frequency,
            max_position=0.10,
            max_sector=0.30,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            transaction_cost=0.001
        )

        self.current_experiment = self.experiment_tracker.create_experiment(
            name=experiment_name or f"Backtest {datetime.now().strftime('%Y%m%d')}",
            description=f"Factor-based strategy with {self.optimization_method} optimization",
            config=exp_config,
            tags=['backtest', self.optimization_method]
        )

        logger.info(f"Experiment: {self.current_experiment.experiment_id}")

        # Prepare factor scores DataFrame
        factors_df = self.factor_engine.to_dataframe(self.factor_scores)

        # Run backtest
        results = self.backtester.run_backtest(
            price_data=self.price_data,
            factor_scores=factors_df,
            optimizer=self.portfolio_optimizer,
            start_date=start_date,
            end_date=end_date,
            n_holdings=self.n_holdings,
            method=self.optimization_method
        )

        if 'error' in results:
            logger.error(f"Backtest failed: {results['error']}")
            return results

        # Calculate benchmark return (NIFTY50 proxy - use average of stocks)
        benchmark_return = 0
        for sym, df in self.price_data.items():
            try:
                df_period = df.loc[start_date:end_date]
                if len(df_period) > 0:
                    ret = (df_period['close'].iloc[-1] / df_period['close'].iloc[0]) - 1
                    benchmark_return += ret
            except:
                pass
        benchmark_return /= len(self.price_data)

        # Create experiment results
        equity_df = results.get('equity_curve')
        monthly_returns = []
        if equity_df is not None:
            monthly_rets = equity_df['portfolio_value'].resample('M').last().pct_change().dropna()
            monthly_returns = monthly_rets.tolist()

        exp_results = ExperimentResults(
            total_return=results['total_return'],
            annual_return=results['annual_return'],
            benchmark_return=benchmark_return,
            excess_return=results['annual_return'] - benchmark_return * (252 / len(equity_df)) if equity_df is not None else 0,
            volatility=results['volatility'],
            sharpe_ratio=results['sharpe_ratio'],
            sortino_ratio=results['sharpe_ratio'] * 1.2,  # Approximation
            max_drawdown=results['max_drawdown'],
            max_drawdown_duration=0,
            n_trades=results['n_trades'],
            turnover=results['n_trades'] / results['n_rebalances'] if results['n_rebalances'] > 0 else 0,
            transaction_costs=results['n_trades'] * 0.001,
            monthly_returns=monthly_returns,
            win_rate_monthly=np.mean([r > 0 for r in monthly_returns]) if monthly_returns else 0,
            best_month=max(monthly_returns) if monthly_returns else 0,
            worst_month=min(monthly_returns) if monthly_returns else 0
        )

        # Update experiment
        self.experiment_tracker.update_results(self.current_experiment, exp_results)

        # Generate report
        report = self._generate_backtest_report(results, exp_results)
        logger.info("\n" + report)

        with open(self.results_dir / 'backtests' / f'{self.current_experiment.experiment_id}.txt', 'w') as f:
            f.write(report)

        # Save equity curve
        if equity_df is not None:
            equity_df.to_csv(self.results_dir / 'backtests' / f'{self.current_experiment.experiment_id}_equity.csv')

        logger.success(f"Backtest complete: {results['annual_return']:.2%} annual return")

        return results

    def _generate_backtest_report(self, results: Dict, exp_results: ExperimentResults) -> str:
        """Generate backtest report."""
        lines = []
        lines.append("=" * 80)
        lines.append("LONG-TERM STRATEGY BACKTEST RESULTS")
        lines.append("=" * 80)

        lines.append(f"\nSTRATEGY:")
        lines.append(f"  Method: {self.optimization_method}")
        lines.append(f"  Holdings: {self.n_holdings}")
        lines.append(f"  Rebalance: {self.rebalance_frequency}")
        lines.append(f"  Factors: Value + Momentum + Quality + Low-Vol + Sentiment (5 factors)")

        lines.append(f"\nPERFORMANCE:")
        lines.append(f"  Total Return:      {results['total_return']:.2%}")
        lines.append(f"  Annual Return:     {results['annual_return']:.2%}")
        lines.append(f"  Benchmark:         {exp_results.benchmark_return:.2%}")
        lines.append(f"  Excess Return:     {exp_results.excess_return:.2%}")

        lines.append(f"\nRISK:")
        lines.append(f"  Volatility:        {results['volatility']:.2%}")
        lines.append(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
        lines.append(f"  Max Drawdown:      {results['max_drawdown']:.2%}")

        lines.append(f"\nTRADING:")
        lines.append(f"  Rebalances:        {results['n_rebalances']}")
        lines.append(f"  Total Trades:      {results['n_trades']}")
        lines.append(f"  Final Value:       Rs {results['final_value']:,.0f}")

        if exp_results.monthly_returns:
            lines.append(f"\nMONTHLY STATS:")
            lines.append(f"  Win Rate:          {exp_results.win_rate_monthly:.1%}")
            lines.append(f"  Best Month:        {exp_results.best_month:.2%}")
            lines.append(f"  Worst Month:       {exp_results.worst_month:.2%}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    # ==================== STAGE 5: SIGNAL GENERATION ====================

    def generate_signals(self) -> Dict[str, Dict]:
        """
        Stage 5: Generate current trading signals.

        Returns:
            Dict of {symbol: {action, target_weight, current_price, ...}}
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: SIGNAL GENERATION")
        logger.info("=" * 60)

        if not self.current_allocation:
            logger.error("No allocation. Run optimize_portfolio() first.")
            return {}

        signals = {}
        target_weights = self.current_allocation.weights

        for symbol in set(list(target_weights.keys()) + list(self.portfolio_weights.keys())):
            current = self.portfolio_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)

            # Get current price
            price = 0
            if symbol in self.price_data:
                price = self.price_data[symbol]['close'].iloc[-1]

            # Determine action
            if target > current + 0.01:
                action = 'BUY'
            elif target < current - 0.01:
                action = 'SELL'
            else:
                action = 'HOLD'

            # Get factor info
            factor_score = None
            for fs in self.factor_scores:
                if fs.symbol == symbol:
                    factor_score = fs
                    break

            signals[symbol] = {
                'action': action,
                'current_weight': current,
                'target_weight': target,
                'weight_change': target - current,
                'current_price': float(price),
                'factor_scores': {
                    'value': factor_score.value_score if factor_score else 0,
                    'momentum': factor_score.momentum_score if factor_score else 0,
                    'quality': factor_score.quality_score if factor_score else 0,
                    'low_vol': factor_score.low_vol_score if factor_score else 0,
                    'sentiment': factor_score.sentiment_score if factor_score else 0,  # NEW
                    'combined': factor_score.combined_score if factor_score else 0,
                } if factor_score else {}
            }

        # Log signals
        buys = [s for s, d in signals.items() if d['action'] == 'BUY']
        sells = [s for s, d in signals.items() if d['action'] == 'SELL']
        holds = [s for s, d in signals.items() if d['action'] == 'HOLD']

        logger.info(f"\nSIGNALS:")
        logger.info(f"  BUY ({len(buys)}):  {buys}")
        logger.info(f"  SELL ({len(sells)}): {sells}")
        logger.info(f"  HOLD ({len(holds)}): {holds}")

        # Save signals
        signals_file = self.results_dir / 'signals' / f'signals_{datetime.now().strftime("%Y%m%d")}.json'
        with open(signals_file, 'w') as f:
            json.dump(signals, f, indent=2, default=str)

        logger.success(f"Signals generated: {len(buys)} buys, {len(sells)} sells")
        return signals

    # ==================== FULL PIPELINE ====================

    def run_pipeline(
        self,
        mode: str = 'backtest',
        start_date: str = '2020-01-01',
        experiment_name: str = None
    ):
        """
        Run complete pipeline.

        Args:
            mode: 'backtest', 'paper', or 'live'
            start_date: Backtest start date
            experiment_name: Name for experiment tracking
        """
        logger.info("#" * 80)
        logger.info("# LONG-TERM EQUITY PIPELINE")
        logger.info(f"# Mode: {mode.upper()}")
        logger.info(f"# Started: {datetime.now()}")
        logger.info("#" * 80)

        # Stage 1: Data
        self.collect_data()

        # Stage 2: Factors
        self.compute_factors()

        # Stage 3: Optimization
        self.optimize_portfolio()

        # Stage 4: Backtest (if applicable)
        if mode in ['backtest', 'paper']:
            self.run_backtest(
                start_date=start_date,
                experiment_name=experiment_name
            )

        # Stage 5: Signals
        signals = self.generate_signals()

        logger.info("#" * 80)
        logger.info("# PIPELINE COMPLETE")
        logger.info(f"# Results: {self.results_dir}")
        logger.info("#" * 80)

        return signals

    # ==================== UTILITIES ====================

    def compare_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """Compare different optimization methods."""
        methods = methods or ['equal', 'risk_parity', 'mean_variance', 'max_sharpe']

        results = []
        original_method = self.optimization_method

        for method in methods:
            logger.info(f"\nTesting {method}...")
            self.optimization_method = method

            # Run backtest
            result = self.run_backtest(
                experiment_name=f"Method Comparison: {method}"
            )

            if 'error' not in result:
                results.append({
                    'method': method,
                    'annual_return': result['annual_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'n_trades': result['n_trades']
                })

        self.optimization_method = original_method

        return pd.DataFrame(results)

    def get_experiment_report(self) -> str:
        """Get report for current experiment."""
        if self.current_experiment:
            return self.experiment_tracker.generate_report(
                self.current_experiment.experiment_id
            )
        return "No current experiment"


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Long-Term Equity Strategy')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default='backtest')
    parser.add_argument('--holdings', type=int, default=20)
    parser.add_argument('--method', default='risk_parity')
    parser.add_argument('--capital', type=float, default=1000000)
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--compare', action='store_true', help='Compare all methods')

    args = parser.parse_args()

    orchestrator = LongTermOrchestrator(
        n_holdings=args.holdings,
        optimization_method=args.method,
        initial_capital=args.capital
    )

    if args.compare:
        comparison = orchestrator.compare_methods()
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        print(comparison.to_string(index=False))
    else:
        orchestrator.run_pipeline(
            mode=args.mode,
            start_date=args.start
        )


if __name__ == '__main__':
    main()
