"""
Backtesting module — portfolio simulation and optimization.

- transaction_costs.py: NSE-specific brokerage + STT + slippage model
- position_sizer.py: Kelly criterion + volatility-adjusted sizing
- portfolio_optimizer.py: Hierarchical Risk Parity (HRP) — the RL replacement
- backtest_engine.py: Core portfolio simulation logic
- backtest_runner.py: Orchestrates full backtest from predictions.csv
"""
