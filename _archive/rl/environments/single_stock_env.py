"""
Single Stock Trading Environment
================================

Gym-compatible environment for RL-based stock trading.
Uses existing ML predictions as part of the state space.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os

# Import from parent modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.rl.utils.state_builder import StateBuilder, PortfolioState, create_empty_portfolio
from src.rl.utils.rewards import RewardCalculator, TradeInfo, create_reward_calculator
from src.rl.config.trading_config import TradingConfig, trading_config


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: int  # Step number when entered
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.entry_price + self.unrealized_pnl

    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.quantity > 0:  # Long
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.quantity < 0:  # Short
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.quantity)


class SingleStockEnv(gym.Env):
    """
    Single stock trading environment.

    State Space (12 dimensions):
        - direction_pred: Predicted direction (0 or 1)
        - close_pred: Predicted close return
        - high_pred: Predicted high return (profit target)
        - low_pred: Predicted low return (stop loss)
        - confidence: Model prediction confidence
        - position: Current position (-1, 0, 1)
        - unrealized_pnl: Normalized unrealized P&L
        - days_held: Normalized days in position
        - cash_ratio: Cash as percentage of portfolio
        - volatility: Volatility percentile
        - volume_ratio: Volume vs average
        - regime: Market regime (normalized)

    Action Space (Discrete):
        - 0: Hold
        - 1: Buy (or close short)
        - 2: Sell (or close long)

    Reward:
        Risk-adjusted P&L with transaction cost and drawdown penalties.
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(
        self,
        symbol: str,
        config: TradingConfig = None,
        predictions_df: pd.DataFrame = None,
        prices_df: pd.DataFrame = None,
        initial_capital: float = 100000,
        reward_type: str = 'risk_adjusted',
        max_steps: int = None,
        render_mode: str = None
    ):
        """
        Initialize trading environment.

        Args:
            symbol: Stock symbol to trade
            config: Trading configuration
            predictions_df: DataFrame with predictions (optional, loads from file if None)
            prices_df: DataFrame with price data (optional, extracted from predictions_df)
            initial_capital: Initial capital in INR
            reward_type: Type of reward function
            max_steps: Maximum steps per episode (None = all data)
            render_mode: Rendering mode
        """
        super().__init__()

        self.symbol = symbol
        self.config = config or trading_config
        self.initial_capital = initial_capital
        self.render_mode = render_mode

        # Load data
        self.state_builder = StateBuilder()
        self._load_data(predictions_df, prices_df)

        # Environment settings
        self.max_steps = max_steps or len(self.data) - 1

        # Define spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        # State space: 12 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(StateBuilder.TOTAL_STATE_DIM,),
            dtype=np.float32
        )

        # Reward calculator
        self.reward_calculator = create_reward_calculator(
            reward_type=reward_type,
            transaction_cost_pct=self.config.transaction.total_roundtrip_cost_pct / 2
        )

        # Initialize state
        self._reset_state()

    def _load_data(self, predictions_df: pd.DataFrame = None, prices_df: pd.DataFrame = None):
        """Load prediction and price data"""
        if predictions_df is not None:
            self.predictions_df = predictions_df
        else:
            # Load from file
            self.predictions_df = self.state_builder.load_predictions(self.symbol)

        # Extract price data
        if prices_df is not None:
            self.prices_df = prices_df
        else:
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in price_cols if c in self.predictions_df.columns]
            self.prices_df = self.predictions_df[available_cols].copy()

        # Create unified data DataFrame
        self.data = self.predictions_df.copy()

        # Ensure we have required columns
        required = ['close']
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def _reset_state(self):
        """Reset internal state for new episode"""
        self.current_step = 0
        self.done = False
        self.truncated = False

        # Portfolio state
        self.cash = self.initial_capital
        self.position: Optional[Position] = None
        self.total_value = self.initial_capital

        # Performance tracking
        self.trades: List[TradeInfo] = []
        self.portfolio_values: List[float] = [self.initial_capital]
        self.daily_returns: List[float] = []

        # Peak for drawdown
        self.peak_value = self.initial_capital

        # Reset reward calculator
        self.reward_calculator.reset()

    def reset(
        self,
        seed: int = None,
        options: dict = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset internal state
        self._reset_state()

        # Get initial observation
        obs = self._get_observation()

        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done or self.truncated:
            raise RuntimeError("Episode has ended. Call reset().")

        # Store previous value
        prev_value = self.total_value

        # Get current price
        current_price = self._get_current_price()

        # Execute action
        trade_info = self._execute_action(action, current_price)

        # Update position P&L
        if self.position is not None:
            self.position.update_pnl(current_price)

        # Calculate new portfolio value
        self._update_portfolio_value(current_price)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            action=action,
            current_value=self.total_value,
            previous_value=prev_value,
            position=self._get_position_signal(),
            trade_info=trade_info,
            step=self.current_step
        )

        # Track performance
        self.portfolio_values.append(self.total_value)
        if prev_value > 0:
            self.daily_returns.append((self.total_value - prev_value) / prev_value)

        # Update peak for drawdown tracking
        self.peak_value = max(self.peak_value, self.total_value)

        # Move to next step
        self.current_step += 1

        # Check termination
        self.done = self._check_termination()
        self.truncated = self.current_step >= self.max_steps

        # Get observation
        obs = self._get_observation()

        info = self._get_info()
        if trade_info is not None:
            info['trade'] = {
                'action': action,
                'pnl': trade_info.pnl,
                'is_profitable': trade_info.is_profitable
            }

        return obs, reward, self.done, self.truncated, info

    def _get_current_price(self) -> float:
        """Get current closing price"""
        return float(self.data['close'].iloc[self.current_step])

    def _execute_action(self, action: int, current_price: float) -> Optional[TradeInfo]:
        """
        Execute trading action.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Current stock price

        Returns:
            TradeInfo if a trade was executed, None otherwise
        """
        trade_info = None

        if action == 0:  # Hold
            pass

        elif action == 1:  # Buy
            if self.position is None:  # Open long
                trade_info = self._open_position(current_price, 1)
            elif self.position.quantity < 0:  # Close short
                trade_info = self._close_position(current_price)

        elif action == 2:  # Sell
            if self.position is None:  # Open short (if allowed)
                # For simplicity, we only allow long positions
                pass
            elif self.position.quantity > 0:  # Close long
                trade_info = self._close_position(current_price)

        return trade_info

    def _open_position(self, price: float, direction: int) -> Optional[TradeInfo]:
        """
        Open a new position.

        Args:
            price: Entry price
            direction: 1 for long, -1 for short

        Returns:
            TradeInfo for the entry
        """
        # Calculate position size
        position_value = self.cash * self.config.capital.position_size_pct
        position_value = min(position_value, self.config.capital.max_trade_value)
        position_value = max(position_value, self.config.capital.min_trade_value)

        quantity = int(position_value / price) * direction

        if quantity == 0:
            return None

        # Transaction cost
        tx_cost = abs(quantity) * price * self.config.transaction.total_roundtrip_cost_pct / 2

        # Update cash
        self.cash -= abs(quantity) * price + tx_cost

        # Create position
        self.position = Position(
            symbol=self.symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=self.current_step
        )

        # Return trade info (entry, no P&L yet)
        return TradeInfo(
            action=1 if direction > 0 else 2,
            entry_price=price,
            exit_price=price,
            quantity=abs(quantity),
            is_profitable=False,
            pnl=0.0,
            pnl_pct=0.0,
            holding_days=0,
            transaction_cost=tx_cost
        )

    def _close_position(self, price: float) -> Optional[TradeInfo]:
        """
        Close current position.

        Args:
            price: Exit price

        Returns:
            TradeInfo for the trade
        """
        if self.position is None:
            return None

        # Calculate P&L
        if self.position.quantity > 0:  # Long
            pnl = (price - self.position.entry_price) * self.position.quantity
        else:  # Short
            pnl = (self.position.entry_price - price) * abs(self.position.quantity)

        # Transaction cost
        tx_cost = abs(self.position.quantity) * price * self.config.transaction.total_roundtrip_cost_pct / 2

        # Net P&L
        net_pnl = pnl - tx_cost

        # Update cash
        self.cash += abs(self.position.quantity) * price - tx_cost

        # Calculate metrics
        pnl_pct = net_pnl / (abs(self.position.quantity) * self.position.entry_price)
        holding_days = self.current_step - self.position.entry_date

        # Create trade info
        trade_info = TradeInfo(
            action=2 if self.position.quantity > 0 else 1,
            entry_price=self.position.entry_price,
            exit_price=price,
            quantity=abs(self.position.quantity),
            is_profitable=net_pnl > 0,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            transaction_cost=tx_cost
        )

        self.trades.append(trade_info)

        # Clear position
        self.position = None

        return trade_info

    def _update_portfolio_value(self, current_price: float):
        """Update total portfolio value"""
        if self.position is not None:
            position_value = abs(self.position.quantity) * current_price
        else:
            position_value = 0

        self.total_value = self.cash + position_value

    def _get_position_signal(self) -> int:
        """Get position as -1, 0, or 1"""
        if self.position is None:
            return 0
        elif self.position.quantity > 0:
            return 1
        else:
            return -1

    def _get_observation(self) -> np.ndarray:
        """Get current observation/state"""
        # Build portfolio state
        positions_dict = {}
        if self.position is not None:
            positions_dict[self.symbol] = {
                'qty': self.position.quantity,
                'avg_price': self.position.entry_price,
                'unrealized_pnl': self.position.unrealized_pnl,
                'days_held': self.current_step - self.position.entry_date
            }

        portfolio = PortfolioState(
            cash=self.cash,
            positions=positions_dict,
            total_value=self.total_value,
            daily_pnl=self.daily_returns[-1] * self.total_value if self.daily_returns else 0,
            realized_pnl=sum(t.pnl for t in self.trades)
        )

        # Get date for predictions
        if self.current_step < len(self.data):
            date = self.data.index[self.current_step]
        else:
            date = None

        # Build state
        state = self.state_builder.build_state(
            symbol=self.symbol,
            portfolio=portfolio,
            date=date
        )

        return state

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Check max drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - self.total_value) / self.peak_value
            if drawdown >= self.config.risk.max_drawdown_pct:
                return True

        # Check if capital depleted
        if self.total_value < self.initial_capital * 0.5:  # Lost 50%
            return True

        return False

    def _get_info(self) -> dict:
        """Get additional info"""
        # Calculate metrics
        if len(self.daily_returns) > 1:
            sharpe = np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        if self.peak_value > 0:
            drawdown = (self.peak_value - self.total_value) / self.peak_value
        else:
            drawdown = 0.0

        total_return = (self.total_value - self.initial_capital) / self.initial_capital

        return {
            'step': self.current_step,
            'portfolio_value': self.total_value,
            'cash': self.cash,
            'position': self._get_position_signal(),
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'num_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t.is_profitable) / max(len(self.trades), 1)
        }

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            info = self._get_info()
            print(f"\n{'='*50}")
            print(f"Step: {info['step']} | Symbol: {self.symbol}")
            print(f"Portfolio Value: Rs {info['portfolio_value']:,.2f}")
            print(f"Cash: Rs {info['cash']:,.2f}")
            print(f"Position: {info['position']}")
            print(f"Total Return: {info['total_return']*100:.2f}%")
            print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {info['max_drawdown']*100:.2f}%")
            print(f"Trades: {info['num_trades']} | Win Rate: {info['win_rate']*100:.1f}%")
            print(f"{'='*50}")

    def close(self):
        """Clean up environment"""
        pass

    def get_episode_summary(self) -> dict:
        """Get summary statistics for the episode"""
        stats = self.reward_calculator.get_episode_stats()

        # Add trade statistics
        if self.trades:
            profits = [t.pnl for t in self.trades if t.is_profitable]
            losses = [abs(t.pnl) for t in self.trades if not t.is_profitable]

            stats['avg_profit'] = np.mean(profits) if profits else 0
            stats['avg_loss'] = np.mean(losses) if losses else 0
            stats['profit_factor'] = sum(profits) / sum(losses) if losses else float('inf')
            stats['avg_holding_days'] = np.mean([t.holding_days for t in self.trades])

        stats['final_value'] = self.total_value
        stats['total_trades'] = len(self.trades)

        return stats


# Factory function
def create_trading_env(
    symbol: str,
    **kwargs
) -> SingleStockEnv:
    """
    Factory function to create trading environment.

    Args:
        symbol: Stock symbol
        **kwargs: Additional arguments

    Returns:
        SingleStockEnv instance
    """
    return SingleStockEnv(symbol=symbol, **kwargs)
