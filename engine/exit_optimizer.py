"""
================================================================================
EXIT OPTIMIZER - Dynamic Exit Strategies
================================================================================
Optimizes exits using trailing stops, profit targets, and technical signals.
Maximizes profit while protecting capital.

Target: Increase profit factor from ~1.3 to 1.8-2.5
================================================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Exit signal with reason and recommended action."""
    should_exit: bool
    exit_type: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'time', 'technical'
    exit_price: float
    reason: str
    partial_exit: Optional[float] = None  # % of position to exit (for scaling out)


class ExitOptimizer:
    """
    Dynamic exit strategy that adapts to market conditions.
    
    Exit Strategies:
    1. Trailing Stop: Move stop loss as profit increases
    2. Profit Target Ladder: Scale out at multiple levels
    3. Time-based: Exit if no movement
    4. Volatility-adjusted: Stop loss = ATR-based
    5. Technical: Exit on reversal signals
    """
    
    def __init__(
        self,
        atr_multiplier: float = 1.5,
        profit_target_1: float = 0.02,  # 2%
        profit_target_2: float = 0.04,  # 4%
        profit_target_3: float = 0.06,  # 6%
        max_holding_days: int = 5,
        use_trailing_stop: bool = True,
        use_profit_ladder: bool = True,
        use_technical_exit: bool = True
    ):
        self.atr_multiplier = atr_multiplier
        self.profit_target_1 = profit_target_1
        self.profit_target_2 = profit_target_2
        self.profit_target_3 = profit_target_3
        self.max_holding_days = max_holding_days
        self.use_trailing_stop = use_trailing_stop
        self.use_profit_ladder = use_profit_ladder
        self.use_technical_exit = use_technical_exit
    
    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        entry_date: datetime,
        current_date: datetime,
        direction: int,  # 1 for long, -1 for short
        features: pd.Series,
        position_state: Dict
    ) -> ExitSignal:
        """
        Determine if position should be exited.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            entry_date: Entry date/time
            current_date: Current date/time
            direction: 1 for long, -1 for short
            features: Current features
            position_state: Position tracking dict (highest_price, stop_loss, etc.)
            
        Returns:
            ExitSignal with decision and reason
        """
        # Calculate P&L
        if direction == 1:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Get position state
        highest_price = position_state.get('highest_price', entry_price if direction == 1 else entry_price)
        current_stop_loss = position_state.get('stop_loss', None)
        profit_target_1_hit = position_state.get('profit_target_1_hit', False)
        profit_target_2_hit = position_state.get('profit_target_2_hit', False)
        
        # Update highest price
        if direction == 1 and current_price > highest_price:
            highest_price = current_price
            position_state['highest_price'] = highest_price
        elif direction == -1 and current_price < highest_price:
            highest_price = current_price
            position_state['highest_price'] = highest_price
        
        # 1. STOP LOSS CHECK
        atr = features.get('atr_pct', 0.015)
        initial_stop_distance = self.atr_multiplier * atr
        
        if current_stop_loss is None:
            # Set initial stop loss
            if direction == 1:
                current_stop_loss = entry_price * (1 - initial_stop_distance)
            else:
                current_stop_loss = entry_price * (1 + initial_stop_distance)
            position_state['stop_loss'] = current_stop_loss
        
        # Check if stop loss hit
        if direction == 1 and current_price <= current_stop_loss:
            return ExitSignal(
                should_exit=True,
                exit_type='stop_loss',
                exit_price=current_price,
                reason=f"Stop loss hit: {pnl_pct:.2%} loss"
            )
        elif direction == -1 and current_price >= current_stop_loss:
            return ExitSignal(
                should_exit=True,
                exit_type='stop_loss',
                exit_price=current_price,
                reason=f"Stop loss hit: {pnl_pct:.2%} loss"
            )
        
        # 2. PROFIT TARGET LADDER (scale out)
        if self.use_profit_ladder:
            if not profit_target_1_hit and pnl_pct >= self.profit_target_1:
                # Hit first profit target - take 50% off
                position_state['profit_target_1_hit'] = True
                # Move stop to breakeven
                position_state['stop_loss'] = entry_price
                
                return ExitSignal(
                    should_exit=False,  # Don't exit fully
                    exit_type='take_profit',
                    exit_price=current_price,
                    reason=f"First profit target hit: {pnl_pct:.2%}",
                    partial_exit=0.50  # Exit 50% of position
                )
            
            if profit_target_1_hit and not profit_target_2_hit and pnl_pct >= self.profit_target_2:
                # Hit second profit target - take another 30% off (30% of original)
                position_state['profit_target_2_hit'] = True
                # Move stop to +1%
                if direction == 1:
                    position_state['stop_loss'] = entry_price * 1.01
                else:
                    position_state['stop_loss'] = entry_price * 0.99
                
                return ExitSignal(
                    should_exit=False,
                    exit_type='take_profit',
                    exit_price=current_price,
                    reason=f"Second profit target hit: {pnl_pct:.2%}",
                    partial_exit=0.30  # Exit 30% of original position
                )
            
            if profit_target_2_hit and pnl_pct >= self.profit_target_3:
                # Hit third profit target - exit remaining 20%
                return ExitSignal(
                    should_exit=True,
                    exit_type='take_profit',
                    exit_price=current_price,
                    reason=f"Third profit target hit: {pnl_pct:.2%}"
                )
        
        # 3. TRAILING STOP
        if self.use_trailing_stop and pnl_pct > self.profit_target_1:
            # Once in profit, trail the stop
            max_pnl = (highest_price - entry_price) / entry_price if direction == 1 else (entry_price - highest_price) / entry_price
            
            # Trail stop at 50% of max profit
            trailing_stop_pct = max_pnl * 0.5
            
            if direction == 1:
                new_stop = entry_price * (1 + trailing_stop_pct)
                if new_stop > current_stop_loss:
                    position_state['stop_loss'] = new_stop
                    current_stop_loss = new_stop
            else:
                new_stop = entry_price * (1 - trailing_stop_pct)
                if new_stop < current_stop_loss:
                    position_state['stop_loss'] = new_stop
                    current_stop_loss = new_stop
        
        # 4. TIME-BASED EXIT
        days_held = (current_date - entry_date).days
        
        if days_held >= self.max_holding_days:
            # Exit if held too long (prevent dead money)
            if pnl_pct > 0:
                return ExitSignal(
                    should_exit=True,
                    exit_type='time',
                    exit_price=current_price,
                    reason=f"Max holding period reached: {days_held} days, P&L: {pnl_pct:.2%}"
                )
            elif days_held >= self.max_holding_days + 2:
                # Give 2 extra days for losing trades
                return ExitSignal(
                    should_exit=True,
                    exit_type='time',
                    exit_price=current_price,
                    reason=f"Max holding period exceeded: {days_held} days, P&L: {pnl_pct:.2%}"
                )
        
        # 5. TECHNICAL EXIT SIGNALS
        if self.use_technical_exit:
            tech_exit = self._check_technical_exit(direction, features, pnl_pct)
            if tech_exit:
                return ExitSignal(
                    should_exit=True,
                    exit_type='technical',
                    exit_price=current_price,
                    reason=f"Technical reversal signal, P&L: {pnl_pct:.2%}"
                )
        
        # Hold position
        return ExitSignal(
            should_exit=False,
            exit_type='hold',
            exit_price=current_price,
            reason=f"Holding, P&L: {pnl_pct:.2%}, Days: {days_held}"
        )
    
    def _check_technical_exit(
        self, 
        direction: int, 
        features: pd.Series,
        current_pnl: float
    ) -> bool:
        """Check for technical reversal signals."""
        # Only exit on technical signals if in profit or small loss
        if current_pnl < -0.005:  # Don't use tech exit if >0.5% loss
            return False
        
        rsi = features.get('rsi_14', 50)
        macd_hist = features.get('macd_hist', 0)
        bb_position = features.get('bb_position', 0.5)
        
        if direction == 1:  # Long position
            # Exit if:
            # - RSI overbought (>75) and MACD turning down
            # - Price at upper Bollinger Band
            overbought = rsi > 75 and macd_hist < 0
            at_upper_bb = bb_position > 0.95
            
            return overbought or at_upper_bb
        
        else:  # Short position
            # Exit if:
            # - RSI oversold (<25) and MACD turning up
            # - Price at lower Bollinger Band
            oversold = rsi < 25 and macd_hist > 0
            at_lower_bb = bb_position < 0.05
            
            return oversold or at_lower_bb
    
    def calculate_initial_stop_loss(
        self, 
        entry_price: float, 
        direction: int, 
        atr: float
    ) -> float:
        """Calculate initial stop loss based on ATR."""
        stop_distance = self.atr_multiplier * atr
        
        if direction == 1:  # Long
            return entry_price * (1 - stop_distance)
        else:  # Short
            return entry_price * (1 + stop_distance)
    
    def calculate_take_profit_levels(
        self, 
        entry_price: float, 
        direction: int
    ) -> Dict[str, float]:
        """Calculate take profit levels."""
        if direction == 1:  # Long
            return {
                'tp1': entry_price * (1 + self.profit_target_1),
                'tp2': entry_price * (1 + self.profit_target_2),
                'tp3': entry_price * (1 + self.profit_target_3)
            }
        else:  # Short
            return {
                'tp1': entry_price * (1 - self.profit_target_1),
                'tp2': entry_price * (1 - self.profit_target_2),
                'tp3': entry_price * (1 - self.profit_target_3)
            }


if __name__ == "__main__":
    # Example usage
    optimizer = ExitOptimizer(
        atr_multiplier=1.5,
        profit_target_1=0.02,
        profit_target_2=0.04,
        profit_target_3=0.06,
        max_holding_days=5
    )
    
    # Test with sample position
    position_state = {
        'highest_price': 105.0,
        'stop_loss': 98.5,
        'profit_target_1_hit': False,
        'profit_target_2_hit': False
    }
    
    features = pd.Series({
        'rsi_14': 55,
        'macd_hist': 0.5,
        'bb_position': 0.6,
        'atr_pct': 0.015
    })
    
    signal = optimizer.should_exit(
        entry_price=100.0,
        current_price=103.0,
        entry_date=datetime.now() - timedelta(days=2),
        current_date=datetime.now(),
        direction=1,
        features=features,
        position_state=position_state
    )
    
    print(f"Should Exit: {signal.should_exit}")
    print(f"Exit Type: {signal.exit_type}")
    print(f"Reason: {signal.reason}")
    if signal.partial_exit:
        print(f"Partial Exit: {signal.partial_exit:.0%}")
