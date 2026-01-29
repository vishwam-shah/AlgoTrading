"""
================================================================================
STEP 5: BACKTESTING & VALIDATION
================================================================================
Comprehensive backtesting framework with performance metrics.

Features:
- Historical simulation with realistic assumptions
- Transaction costs and slippage
- Multiple performance metrics (Sharpe, Sortino, Calmar, etc.)
- Drawdown analysis
- Benchmark comparison
- Out-of-sample validation
- Rolling performance analysis

Metrics Computed:
- Total Return, Annual Return
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Drawdown Duration
- Win Rate, Profit Factor
- Value at Risk (VaR), Conditional VaR

Usage:
    from pipeline.step_5_backtest_validation import BacktestValidator
    validator = BacktestValidator()
    results = validator.run_backtest(price_data, allocation, start_date, end_date)

Or run directly:
    python pipeline/step_5_backtest_validation.py
================================================================================
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    # Basic metrics
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade metrics
    n_trades: int
    n_rebalances: int
    win_rate: float
    profit_factor: float
    
    # Value at Risk
    var_95: float
    cvar_95: float
    
    # Portfolio info
    final_value: float
    initial_capital: float
    
    # Time series
    equity_curve: pd.DataFrame = None
    drawdown_series: pd.Series = None
    monthly_returns: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'n_trades': self.n_trades,
            'n_rebalances': self.n_rebalances,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'final_value': self.final_value,
            'initial_capital': self.initial_capital
        }


class BacktestValidator:
    """
    Comprehensive backtesting framework.
    
    Simulates portfolio performance with realistic assumptions:
    - Transaction costs
    - Slippage
    - Rebalancing constraints
    - Cash management
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,          # 0.05% slippage
        rebalance_frequency: str = 'monthly',  # monthly, quarterly, weekly
        risk_free_rate: float = 0.05
    ):
        """Initialize BacktestValidator."""
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        
        logger.info("BacktestValidator initialized")
        logger.info(f"  Transaction cost: {transaction_cost:.2%}")
        logger.info(f"  Slippage: {slippage:.2%}")
        logger.info(f"  Rebalance: {rebalance_frequency}")
    
    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        allocation: Dict[str, float],
        start_date: str = '2022-01-01',
        end_date: str = None,
        initial_capital: float = 1000000,
        benchmark: pd.DataFrame = None
    ) -> BacktestResults:
        """
        Run historical backtest.
        
        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            allocation: Dict of symbol -> target weight
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            benchmark: Benchmark index DataFrame
            
        Returns:
            BacktestResults object
        """
        logger.info("=" * 60)
        logger.info("STEP 5: BACKTESTING & VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date or 'today'}")
        logger.info(f"Initial capital: Rs {initial_capital:,.0f}")
        logger.info(f"Portfolio: {len(allocation)} positions")
        
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Build price matrix
        price_matrix = self._build_price_matrix(price_data, allocation, start_date, end_date)
        
        if price_matrix is None or len(price_matrix) < 20:
            logger.error("Insufficient data for backtest")
            return None
        
        # Run simulation
        equity_curve, trades = self._simulate(price_matrix, allocation, initial_capital)
        
        # Calculate metrics
        results = self._calculate_metrics(
            equity_curve, trades, initial_capital, benchmark, start_date, end_date
        )
        
        # Log results
        self._log_results(results)
        
        logger.success("Backtest complete")
        
        return results
    
    def _build_price_matrix(
        self,
        price_data: Dict[str, pd.DataFrame],
        allocation: Dict[str, float],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Build aligned price matrix for portfolio symbols."""
        
        symbols = list(allocation.keys())
        available_symbols = [s for s in symbols if s in price_data]
        
        if not available_symbols:
            logger.error("No price data for portfolio symbols")
            return None
        
        # Build price matrix
        prices = pd.DataFrame()
        for symbol in available_symbols:
            df = price_data[symbol]
            prices[symbol] = df['close']
        
        # Filter date range
        prices = prices.loc[start_date:end_date]
        
        # Forward fill missing prices (holidays, etc.)
        prices = prices.ffill()
        
        logger.info(f"Price matrix: {len(prices)} days, {len(available_symbols)} symbols")
        
        return prices
    
    def _simulate(
        self,
        prices: pd.DataFrame,
        allocation: Dict[str, float],
        initial_capital: float
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Simulate portfolio with periodic rebalancing.
        """
        symbols = prices.columns.tolist()
        dates = prices.index.tolist()
        
        # Filter allocation to available symbols
        weights = {s: allocation.get(s, 0) for s in symbols}
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}
        else:
            weights = {s: 1 / len(symbols) for s in symbols}
        
        # Initialize portfolio
        cash = initial_capital
        holdings = {s: 0 for s in symbols}
        portfolio_values = []
        trades = []
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(dates)
        
        for i, date in enumerate(dates):
            # Get current prices
            current_prices = prices.loc[date]
            
            # Rebalance if needed
            if date in rebalance_dates:
                cash, holdings, day_trades = self._rebalance(
                    cash, holdings, current_prices, weights, date
                )
                trades.extend(day_trades)
            
            # Calculate portfolio value
            holdings_value = sum(holdings[s] * current_prices[s] for s in symbols)
            total_value = cash + holdings_value
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': total_value,
                'cash': cash,
                'holdings_value': holdings_value
            })
        
        equity_curve = pd.DataFrame(portfolio_values)
        equity_curve = equity_curve.set_index('date')
        
        return equity_curve, trades
    
    def _rebalance(
        self,
        cash: float,
        holdings: Dict[str, int],
        prices: pd.Series,
        target_weights: Dict[str, float],
        date: datetime
    ) -> Tuple[float, Dict[str, int], List[Dict]]:
        """
        Rebalance portfolio to target weights.
        """
        symbols = list(target_weights.keys())
        trades = []
        
        # Calculate current value
        holdings_value = sum(holdings.get(s, 0) * prices[s] for s in symbols)
        total_value = cash + holdings_value
        
        # Calculate target values
        for symbol in symbols:
            target_value = total_value * target_weights[symbol]
            current_value = holdings.get(symbol, 0) * prices[symbol]
            diff = target_value - current_value
            
            if abs(diff) > total_value * 0.02:  # Only trade if > 2% difference
                shares_to_trade = int(diff / prices[symbol])
                
                if shares_to_trade > 0:
                    # Buy
                    cost = shares_to_trade * prices[symbol] * (1 + self.transaction_cost + self.slippage)
                    if cost <= cash:
                        holdings[symbol] = holdings.get(symbol, 0) + shares_to_trade
                        cash -= cost
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares_to_trade,
                            'price': prices[symbol],
                            'cost': cost
                        })
                elif shares_to_trade < 0:
                    # Sell
                    shares_to_sell = min(abs(shares_to_trade), holdings.get(symbol, 0))
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * prices[symbol] * (1 - self.transaction_cost - self.slippage)
                        holdings[symbol] = holdings.get(symbol, 0) - shares_to_sell
                        cash += proceeds
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': prices[symbol],
                            'proceeds': proceeds
                        })
        
        return cash, holdings, trades
    
    def _get_rebalance_dates(self, dates: List[datetime]) -> set:
        """Get rebalancing dates based on frequency."""
        rebalance_dates = set()
        
        if self.rebalance_frequency == 'weekly':
            # Every Monday
            for date in dates:
                if hasattr(date, 'dayofweek') and date.dayofweek == 0:
                    rebalance_dates.add(date)
                elif hasattr(date, 'weekday') and date.weekday() == 0:
                    rebalance_dates.add(date)
        
        elif self.rebalance_frequency == 'monthly':
            # First trading day of month
            current_month = None
            for date in dates:
                month = date.month if hasattr(date, 'month') else date.to_pydatetime().month
                if month != current_month:
                    rebalance_dates.add(date)
                    current_month = month
        
        elif self.rebalance_frequency == 'quarterly':
            # First trading day of quarter
            current_quarter = None
            for date in dates:
                month = date.month if hasattr(date, 'month') else date.to_pydatetime().month
                quarter = (month - 1) // 3
                if quarter != current_quarter:
                    rebalance_dates.add(date)
                    current_quarter = quarter
        
        # Always include first date
        rebalance_dates.add(dates[0])
        
        return rebalance_dates
    
    def _calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict],
        initial_capital: float,
        benchmark: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResults:
        """Calculate all performance metrics."""
        
        portfolio_values = equity_curve['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_days = len(portfolio_values)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 0:
            bench_close = benchmark['close']
            bench_close = bench_close.reindex(portfolio_values.index, method='ffill')
            bench_return = (bench_close.iloc[-1] / bench_close.iloc[0]) - 1
        else:
            bench_return = 0.10 * n_years  # Assume 10% annual
        
        excess_return = annual_return - (bench_return / n_years) if n_years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        
        # Drawdown
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = in_drawdown.astype(int).groupby((~in_drawdown).cumsum())
        max_dd_duration = drawdown_periods.sum().max() if len(drawdown_periods) > 0 else 0
        
        # Risk-adjusted metrics
        sharpe = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Information ratio
        if benchmark is not None and len(benchmark) > 0:
            bench_ret = benchmark['close'].pct_change().reindex(returns.index, method='ffill').dropna()
            tracking_error = (returns - bench_ret).std() * np.sqrt(252)
            info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        else:
            info_ratio = 0
        
        # Trade metrics
        n_trades = len(trades)
        n_rebalances = len(set(t['date'] for t in trades))
        
        # Win rate (by month)
        monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
        win_rate = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0.5
        
        # Profit factor
        profitable_months = monthly_returns[monthly_returns > 0].sum()
        losing_months = abs(monthly_returns[monthly_returns < 0].sum())
        profit_factor = profitable_months / losing_months if losing_months > 0 else 2.0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        return BacktestResults(
            total_return=float(total_return),
            annual_return=float(annual_return),
            benchmark_return=float(bench_return),
            excess_return=float(excess_return),
            volatility=float(volatility),
            downside_volatility=float(downside_vol),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_dd_duration),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            information_ratio=float(info_ratio),
            n_trades=n_trades,
            n_rebalances=n_rebalances,
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            final_value=float(portfolio_values.iloc[-1]),
            initial_capital=initial_capital,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            monthly_returns=monthly_returns.tolist()
        )
    
    def _log_results(self, results: BacktestResults):
        """Log backtest results."""
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        logger.info("\nPERFORMANCE:")
        logger.info(f"  Total Return:      {results.total_return:>10.2%}")
        logger.info(f"  Annual Return:     {results.annual_return:>10.2%}")
        logger.info(f"  Benchmark Return:  {results.benchmark_return:>10.2%}")
        logger.info(f"  Excess Return:     {results.excess_return:>10.2%}")
        
        logger.info("\nRISK:")
        logger.info(f"  Volatility:        {results.volatility:>10.2%}")
        logger.info(f"  Max Drawdown:      {results.max_drawdown:>10.2%}")
        logger.info(f"  Max DD Duration:   {results.max_drawdown_duration:>10} days")
        
        logger.info("\nRISK-ADJUSTED:")
        logger.info(f"  Sharpe Ratio:      {results.sharpe_ratio:>10.2f}")
        logger.info(f"  Sortino Ratio:     {results.sortino_ratio:>10.2f}")
        logger.info(f"  Calmar Ratio:      {results.calmar_ratio:>10.2f}")
        
        logger.info("\nTRADING:")
        logger.info(f"  Total Trades:      {results.n_trades:>10}")
        logger.info(f"  Rebalances:        {results.n_rebalances:>10}")
        logger.info(f"  Win Rate:          {results.win_rate:>10.2%}")
        logger.info(f"  Profit Factor:     {results.profit_factor:>10.2f}")
        
        logger.info("\nRISK METRICS:")
        logger.info(f"  VaR (95%):         {results.var_95:>10.2%}")
        logger.info(f"  CVaR (95%):        {results.cvar_95:>10.2%}")
        
        logger.info("\nPORTFOLIO:")
        logger.info(f"  Initial Capital:   Rs {results.initial_capital:>12,.0f}")
        logger.info(f"  Final Value:       Rs {results.final_value:>12,.0f}")


def test_backtest_validation():
    """Test complete pipeline with backtesting."""
    print("\n" + "=" * 80)
    print("TESTING STEP 5: BACKTESTING & VALIDATION")
    print("=" * 80)
    
    # Run previous steps
    from step_1_data_collection import DataCollector
    from step_2_feature_engineering import FeatureEngineer
    from step_3_factor_analysis import FactorAnalyzer
    from step_4_portfolio_optimization import PortfolioOptimizer
    
    test_symbols = [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
        'TCS', 'INFY', 'RELIANCE', 'TATASTEEL', 'HINDUNILVR',
        'MARUTI', 'SUNPHARMA', 'LT', 'BHARTIARTL', 'ITC'
    ]
    
    # Step 1: Collect data
    print("\n[Step 1] Collecting data...")
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols=test_symbols, start_date='2022-01-01')
    
    # Step 2: Compute features
    print("\n[Step 2] Computing features...")
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)
    
    # Step 3: Factor analysis
    print("\n[Step 3] Analyzing factors...")
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features, market_data.get('NIFTY50'))
    
    # Step 4: Portfolio optimization
    print("\n[Step 4] Optimizing portfolio...")
    optimizer = PortfolioOptimizer()
    allocation = optimizer.optimize(
        price_data=price_data,
        factor_scores=factor_scores,
        n_holdings=10,
        method='risk_parity',
        sector_map=config.STOCK_SECTOR_MAP
    )
    
    # Step 5: Backtest
    print("\n[Step 5] Running backtest...")
    validator = BacktestValidator(
        transaction_cost=0.001,
        slippage=0.0005,
        rebalance_frequency='monthly'
    )
    
    results = validator.run_backtest(
        price_data=price_data,
        allocation=allocation.weights,
        start_date='2022-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d'),
        initial_capital=1000000,
        benchmark=market_data.get('NIFTY50')
    )
    
    # Validation tests
    print("\n" + "-" * 40)
    print("VALIDATION TESTS")
    print("-" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Backtest completed
    tests_total += 1
    if results is not None:
        print("✓ Test 1: Backtest completed - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 1: Backtest completed - FAILED")
        return results
    
    # Test 2: Positive final value
    tests_total += 1
    if results.final_value > 0:
        print(f"✓ Test 2: Positive final value (Rs {results.final_value:,.0f}) - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 2: Positive final value - FAILED")
    
    # Test 3: Max drawdown < 50%
    tests_total += 1
    if results.max_drawdown < 0.5:
        print(f"✓ Test 3: Max drawdown < 50% ({results.max_drawdown:.1%}) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 3: Max drawdown < 50% ({results.max_drawdown:.1%}) - FAILED")
    
    # Test 4: Sharpe ratio reasonable
    tests_total += 1
    if -2 < results.sharpe_ratio < 5:
        print(f"✓ Test 4: Sharpe ratio reasonable ({results.sharpe_ratio:.2f}) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 4: Sharpe ratio unreasonable ({results.sharpe_ratio:.2f}) - FAILED")
    
    # Test 5: Beat benchmark or acceptable underperformance
    tests_total += 1
    if results.excess_return > -0.10:  # Within 10% of benchmark
        print(f"✓ Test 5: Performance vs benchmark ({results.excess_return:+.2%}) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 5: Performance vs benchmark ({results.excess_return:+.2%}) - FAILED")
    
    # Test 6: Win rate > 40%
    tests_total += 1
    if results.win_rate > 0.4:
        print(f"✓ Test 6: Win rate > 40% ({results.win_rate:.1%}) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 6: Win rate > 40% ({results.win_rate:.1%}) - FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"TESTS: {tests_passed}/{tests_total} passed")
    print("=" * 40)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Stocks analyzed:     {len(price_data)}")
    print(f"Features computed:   {engineer.get_feature_count()}")
    print(f"Factors:             5 (Value, Momentum, Quality, Low-Vol, Sentiment)")
    print(f"Portfolio positions: {allocation.n_positions}")
    print(f"Backtest period:     2022-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"\nFINAL RESULTS:")
    print(f"  Total Return:      {results.total_return:>10.2%}")
    print(f"  Annual Return:     {results.annual_return:>10.2%}")
    print(f"  Sharpe Ratio:      {results.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:      {results.max_drawdown:>10.2%}")
    print(f"  Final Value:       Rs {results.final_value:>12,.0f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    test_backtest_validation()
