"""
Reward Functions for RL Trading Agents
======================================

Multiple reward functions for different optimization objectives:
- Simple P&L
- Risk-adjusted (Sharpe-like)
- Drawdown penalized
- Multi-objective
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of reward functions"""
    SIMPLE_PNL = 'simple_pnl'
    RISK_ADJUSTED = 'risk_adjusted'
    SHARPE_INCREMENTAL = 'sharpe_incremental'
    DRAWDOWN_PENALIZED = 'drawdown_penalized'
    MULTI_OBJECTIVE = 'multi_objective'


@dataclass
class TradeInfo:
    """Information about a trade for reward calculation"""
    action: int  # 0=hold, 1=buy, 2=sell
    entry_price: float
    exit_price: float
    quantity: int
    is_profitable: bool
    pnl: float
    pnl_pct: float
    holding_days: int
    transaction_cost: float


class RewardCalculator:
    """
    Calculates rewards for RL trading agents.

    Supports multiple reward formulations:
    1. Simple P&L - Raw profit/loss
    2. Risk-adjusted - Accounts for volatility
    3. Sharpe incremental - Based on Sharpe ratio contribution
    4. Drawdown penalized - Penalizes drawdowns
    5. Multi-objective - Weighted combination
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.RISK_ADJUSTED,
        transaction_cost_pct: float = 0.001,
        risk_free_rate: float = 0.05,  # 5% annual
        drawdown_penalty: float = 2.0,
        win_bonus: float = 0.1,
        holding_penalty: float = 0.001,
        weights: Dict[str, float] = None
    ):
        """
        Initialize reward calculator.

        Args:
            reward_type: Type of reward function
            transaction_cost_pct: Transaction cost as percentage
            risk_free_rate: Annual risk-free rate
            drawdown_penalty: Multiplier for drawdown penalty
            win_bonus: Bonus for winning trades
            holding_penalty: Penalty per day for holding positions
            weights: Weights for multi-objective reward
        """
        self.reward_type = reward_type
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_free_rate = risk_free_rate / 252  # Daily rate
        self.drawdown_penalty = drawdown_penalty
        self.win_bonus = win_bonus
        self.holding_penalty = holding_penalty

        # Multi-objective weights
        self.weights = weights or {
            'pnl': 1.0,
            'sharpe': 0.5,
            'drawdown': -2.0,
            'win_rate': 0.3,
            'transaction_cost': -1.0
        }

        # History for tracking
        self.returns_history: List[float] = []
        self.peak_value: float = 0.0
        self.trades_history: List[TradeInfo] = []

    def reset(self):
        """Reset history for new episode"""
        self.returns_history = []
        self.peak_value = 0.0
        self.trades_history = []

    def calculate_reward(
        self,
        action: int,
        current_value: float,
        previous_value: float,
        position: int,
        trade_info: TradeInfo = None,
        step: int = 0
    ) -> float:
        """
        Calculate reward based on configured reward type.

        Args:
            action: Action taken (0=hold, 1=buy, 2=sell)
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            position: Current position (-1, 0, 1)
            trade_info: Information about completed trade (if any)
            step: Current step in episode

        Returns:
            Reward value
        """
        if self.reward_type == RewardType.SIMPLE_PNL:
            return self._simple_pnl_reward(current_value, previous_value, action)

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            return self._risk_adjusted_reward(current_value, previous_value, action, position)

        elif self.reward_type == RewardType.SHARPE_INCREMENTAL:
            return self._sharpe_incremental_reward(current_value, previous_value)

        elif self.reward_type == RewardType.DRAWDOWN_PENALIZED:
            return self._drawdown_penalized_reward(current_value, previous_value, action)

        elif self.reward_type == RewardType.MULTI_OBJECTIVE:
            return self._multi_objective_reward(
                current_value, previous_value, action, position, trade_info
            )

        else:
            return self._simple_pnl_reward(current_value, previous_value, action)

    def _simple_pnl_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int
    ) -> float:
        """
        Simple P&L reward.

        reward = (current_value - previous_value) / previous_value - transaction_cost
        """
        if previous_value <= 0:
            return 0.0

        # Calculate return
        ret = (current_value - previous_value) / previous_value

        # Subtract transaction cost if action was buy/sell
        if action != 0:  # Not hold
            ret -= self.transaction_cost_pct

        # Scale reward
        reward = ret * 100  # Scale to percentage points

        return reward

    def _risk_adjusted_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int,
        position: int
    ) -> float:
        """
        Risk-adjusted reward accounting for volatility.

        reward = return / volatility - transaction_cost
        """
        if previous_value <= 0:
            return 0.0

        # Calculate return
        ret = (current_value - previous_value) / previous_value
        self.returns_history.append(ret)

        # Transaction cost
        if action != 0:
            ret -= self.transaction_cost_pct

        # Calculate rolling volatility
        if len(self.returns_history) >= 5:
            volatility = np.std(self.returns_history[-20:]) * np.sqrt(252)
            volatility = max(volatility, 0.01)  # Minimum volatility

            # Risk-adjusted return
            risk_adj_ret = ret / volatility
        else:
            risk_adj_ret = ret

        # Holding penalty (encourage trading when signals are strong)
        if position != 0 and action == 0:  # Holding a position
            risk_adj_ret -= self.holding_penalty

        return risk_adj_ret * 100

    def _sharpe_incremental_reward(
        self,
        current_value: float,
        previous_value: float
    ) -> float:
        """
        Incremental Sharpe ratio reward.

        Rewards actions that improve the Sharpe ratio of the return series.
        """
        if previous_value <= 0:
            return 0.0

        # Calculate return
        ret = (current_value - previous_value) / previous_value

        # Calculate Sharpe before
        if len(self.returns_history) >= 2:
            old_sharpe = self._calculate_sharpe(self.returns_history)
        else:
            old_sharpe = 0.0

        # Add new return
        self.returns_history.append(ret)

        # Calculate Sharpe after
        if len(self.returns_history) >= 2:
            new_sharpe = self._calculate_sharpe(self.returns_history)
        else:
            new_sharpe = 0.0

        # Reward is improvement in Sharpe
        sharpe_improvement = new_sharpe - old_sharpe

        return sharpe_improvement

    def _calculate_sharpe(self, returns: List[float], annualize: bool = True) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0

        excess_returns = np.array(returns) - self.risk_free_rate
        mean_ret = np.mean(excess_returns)
        std_ret = np.std(excess_returns)

        if std_ret < 1e-8:
            return 0.0

        sharpe = mean_ret / std_ret

        if annualize:
            sharpe *= np.sqrt(252)

        return sharpe

    def _drawdown_penalized_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int
    ) -> float:
        """
        Reward with drawdown penalty.

        reward = return - drawdown_penalty * drawdown
        """
        if previous_value <= 0:
            return 0.0

        # Calculate return
        ret = (current_value - previous_value) / previous_value

        # Transaction cost
        if action != 0:
            ret -= self.transaction_cost_pct

        # Update peak
        self.peak_value = max(self.peak_value, current_value)

        # Calculate drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            drawdown = 0.0

        # Apply drawdown penalty
        reward = ret - self.drawdown_penalty * drawdown

        return reward * 100

    def _multi_objective_reward(
        self,
        current_value: float,
        previous_value: float,
        action: int,
        position: int,
        trade_info: TradeInfo = None
    ) -> float:
        """
        Multi-objective reward combining multiple factors.

        Components:
        - P&L
        - Sharpe contribution
        - Drawdown penalty
        - Win rate bonus
        - Transaction cost penalty
        """
        if previous_value <= 0:
            return 0.0

        # Calculate return
        ret = (current_value - previous_value) / previous_value
        self.returns_history.append(ret)

        # 1. P&L component
        pnl_reward = ret * self.weights['pnl']

        # 2. Sharpe component
        if len(self.returns_history) >= 5:
            sharpe = self._calculate_sharpe(self.returns_history[-20:])
            sharpe_reward = np.clip(sharpe / 3.0, -1, 1) * self.weights['sharpe']
        else:
            sharpe_reward = 0.0

        # 3. Drawdown component
        self.peak_value = max(self.peak_value, current_value)
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            drawdown_reward = drawdown * self.weights['drawdown']
        else:
            drawdown_reward = 0.0

        # 4. Win rate component
        if trade_info is not None:
            self.trades_history.append(trade_info)
            if trade_info.is_profitable:
                win_reward = self.weights['win_rate']
            else:
                win_reward = -self.weights['win_rate'] * 0.5
        else:
            win_reward = 0.0

        # 5. Transaction cost component
        if action != 0:
            tx_cost_reward = self.transaction_cost_pct * self.weights['transaction_cost']
        else:
            tx_cost_reward = 0.0

        # Combine all components
        total_reward = pnl_reward + sharpe_reward + drawdown_reward + win_reward + tx_cost_reward

        return total_reward * 100

    def get_episode_stats(self) -> Dict:
        """Get statistics for the episode"""
        if not self.returns_history:
            return {}

        returns = np.array(self.returns_history)

        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        sharpe = self._calculate_sharpe(list(returns))

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = drawdowns.max()

        # Win rate
        if self.trades_history:
            wins = sum(1 for t in self.trades_history if t.is_profitable)
            win_rate = wins / len(self.trades_history)
        else:
            win_rate = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades_history),
            'volatility': returns.std() * np.sqrt(252)
        }


# Factory function
def create_reward_calculator(
    reward_type: str = 'risk_adjusted',
    **kwargs
) -> RewardCalculator:
    """
    Factory function to create reward calculator.

    Args:
        reward_type: Type of reward ('simple_pnl', 'risk_adjusted', 'sharpe', 'drawdown', 'multi')
        **kwargs: Additional arguments

    Returns:
        RewardCalculator instance
    """
    type_map = {
        'simple_pnl': RewardType.SIMPLE_PNL,
        'simple': RewardType.SIMPLE_PNL,
        'risk_adjusted': RewardType.RISK_ADJUSTED,
        'risk': RewardType.RISK_ADJUSTED,
        'sharpe': RewardType.SHARPE_INCREMENTAL,
        'sharpe_incremental': RewardType.SHARPE_INCREMENTAL,
        'drawdown': RewardType.DRAWDOWN_PENALIZED,
        'drawdown_penalized': RewardType.DRAWDOWN_PENALIZED,
        'multi': RewardType.MULTI_OBJECTIVE,
        'multi_objective': RewardType.MULTI_OBJECTIVE
    }

    reward_type_enum = type_map.get(reward_type.lower(), RewardType.RISK_ADJUSTED)

    return RewardCalculator(reward_type=reward_type_enum, **kwargs)
