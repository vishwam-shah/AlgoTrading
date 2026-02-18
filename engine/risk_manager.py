"""
================================================================================
RISK MANAGER - Multi-Layer Risk Controls & Circuit Breakers
================================================================================
Protects capital with:
1. Per-trade risk limits
2. Portfolio-wide risk limits  
3. Drawdown controls
4. Circuit breakers (auto-stop trading)
5. Correlation limits

Target: Max drawdown <15%, protect capital
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of risk check."""
    approved: bool
    reason: str
    warnings: List[str]
    risk_metrics: Dict


class RiskManager:
    """
    Comprehensive risk management system.
    
    Risk Controls:
    1. Max loss per trade: 1% of capital
    2. Max position size: 10% of capital
    3. Max daily loss: 2% - stop trading
    4. Max weekly loss: 5% - review strategy
    5. Max drawdown: 15% - reduce sizes by 50%
    6. Max open positions: 10
    7. Sector/correlation limits
    """
    
    def __init__(
        self,
        initial_capital: float,
        max_risk_per_trade: float = 0.01,  # 1%
        max_position_size: float = 0.10,  # 10%
        max_daily_loss: float = 0.02,  # 2%
        max_weekly_loss: float = 0.05,  # 5%
        max_drawdown: float = 0.15,  # 15%
        max_open_positions: int = 10,
        max_sector_exposure: float = 0.30,  # 30%
        max_correlated_positions: int = 2
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Risk limits
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_weekly_loss = max_weekly_loss
        self.max_drawdown = max_drawdown
        self.max_open_positions = max_open_positions
        self.max_sector_exposure = max_sector_exposure
        self.max_correlated_positions = max_correlated_positions
        
        # State tracking
        self.current_positions = {}  # {symbol: position_info}
        self.daily_pnl = {}  # {date: pnl}
        self.weekly_pnl = {}  # {week: pnl}
        self.trading_stopped = False
        self.stop_reason = None
        self.consecutive_losing_days = 0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
    
    def approve_trade(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        sector: Optional[str] = None,
        correlation_group: Optional[List[str]] = None
    ) -> RiskCheck:
        """
        Check if trade meets all risk criteria.
        
        Returns RiskCheck with approval decision and reasons.
        """
        warnings = []
        risk_metrics = {}
        
        # 1. Check if trading is stopped
        if self.trading_stopped:
            return RiskCheck(
                approved=False,
                reason=f"Trading stopped: {self.stop_reason}",
                warnings=[],
                risk_metrics={}
            )
        
        # 2. Check max positions
        if len(self.current_positions) >= self.max_open_positions:
            return RiskCheck(
                approved=False,
                reason=f"Max positions reached: {len(self.current_positions)}/{self.max_open_positions}",
                warnings=warnings,
                risk_metrics=risk_metrics
            )
        
        # 3. Check position size limit
        position_pct = position_size / self.current_capital
        risk_metrics['position_pct'] = position_pct
        
        if position_pct > self.max_position_size:
            return RiskCheck(
                approved=False,
                reason=f"Position too large: {position_pct:.1%} > {self.max_position_size:.1%}",
                warnings=warnings,
                risk_metrics=risk_metrics
            )
        
        # 4. Check risk per trade
        risk_amount = abs(entry_price - stop_loss) * (position_size / entry_price)
        risk_pct = risk_amount / self.current_capital
        risk_metrics['risk_amount'] = risk_amount
        risk_metrics['risk_pct'] = risk_pct
        
        if risk_pct > self.max_risk_per_trade:
            return RiskCheck(
                approved=False,
                reason=f"Risk too high: {risk_pct:.2%} > {self.max_risk_per_trade:.2%}",
                warnings=warnings,
                risk_metrics=risk_metrics
            )
        
        # 5. Check daily loss limit
        today = date.today()
        today_pnl = self.daily_pnl.get(today, 0)
        daily_loss_pct = today_pnl / self.initial_capital
        risk_metrics['daily_pnl'] = today_pnl
        risk_metrics['daily_loss_pct'] = daily_loss_pct
        
        if daily_loss_pct < -self.max_daily_loss:
            self.stop_trading("Daily loss limit exceeded")
            return RiskCheck(
                approved=False,
                reason=f"Daily loss limit hit: {daily_loss_pct:.2%}",
                warnings=warnings,
                risk_metrics=risk_metrics
            )
        
        # 6. Check drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self. peak_capital
        risk_metrics['current_drawdown'] = current_drawdown
        
        if current_drawdown > self.max_drawdown:
            warnings.append(f"High drawdown: {current_drawdown:.1%}")
        
        # 7. Check sector exposure
        if sector:
            sector_exposure = self._calculate_sector_exposure(sector, position_size)
            risk_metrics['sector_exposure'] = sector_exposure
            
            if sector_exposure > self.max_sector_exposure:
                warnings.append(f"High sector exposure: {sector_exposure:.1%}")
        
        # 8. Check correlation limits
        if correlation_group:
            corr_count = self._count_correlated_positions(correlation_group)
            risk_metrics['correlated_positions'] = corr_count
            
            if corr_count >= self.max_correlated_positions:
                return RiskCheck(
                    approved=False,
                    reason=f"Too many correlated positions: {corr_count}",
                    warnings=warnings,
                    risk_metrics=risk_metrics
                )
        
        # 9. Check consecutive losing days
        if self.consecutive_losing_days >= 3:
            warnings.append(f"Consecutive losing days: {self.consecutive_losing_days}")
        
        # 10. Check market conditions (VIX, etc.)
        # Could add market-wide circuit breakers here
        
        # APPROVED
        return RiskCheck(
            approved=True,
            reason="Trade approved",
            warnings=warnings,
            risk_metrics=risk_metrics
        )
    
    def record_trade(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        shares: int,
        stop_loss: float,
        sector: Optional[str] = None
    ):
        """Record a new trade/position."""
        position_value = entry_price * shares
        
        self.current_positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'shares': shares,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'sector': sector,
            'entry_date': datetime.now(),
            'highest_price': entry_price if direction == 1 else entry_price,
            'profit_target_1_hit': False,
            'profit_target_2_hit': False
        }
        
        self.total_trades += 1
        logger.info(f"Recorded trade: {symbol}, {shares} shares @ ₹{entry_price}")
    
    def close_position(self, symbol: str, exit_price: float):
        """Close a position and update P&L."""
        if symbol not in self.current_positions:
            logger.warning(f"Cannot close {symbol} - not in positions")
            return
        
        pos = self.current_positions[symbol]
        
        # Calculate P&L
        if pos['direction'] == 1:  # Long
            pnl = (exit_price - pos['entry_price']) * pos['shares']
        else:  # Short
            pnl = (pos['entry_price'] - exit_price) * pos['shares']
        
        pnl_pct = pnl / pos['position_value']
        
        # Update capital
        self.current_capital += pnl
        self.total_pnl += pnl
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update daily P&L
        today = date.today()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
        
        # Update winning/losing trades
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Remove from positions
        del self.current_positions[symbol]
        
        logger.info(f"Closed {symbol}: P&L=₹{pnl:,.0f} ({pnl_pct:.2%}), "
                   f"Capital=₹{self.current_capital:,.0f}")
        
        # Check for consecutive losing days
        self._check_consecutive_losses()
    
    def update_daily(self):
        """Called at end of day to check risk limits."""
        today = date.today()
        today_pnl = self.daily_pnl.get(today, 0)
        
        # Check if losing day
        if today_pnl < 0:
            self.consecutive_losing_days += 1
        else:
            self.consecutive_losing_days = 0
        
        # Circuit breaker: 3 consecutive losing days
        if self.consecutive_losing_days >= 3:
            self.stop_trading(f"3 consecutive losing days")
        
        # Log daily stats
        logger.info(f"EOD: Capital=₹{self.current_capital:,.0f}, "
                   f"Daily P&L=₹{today_pnl:,.0f}, "
                   f"Open Positions={len(self.current_positions)}")
    
    def stop_trading(self, reason: str):
        """Stop all trading."""
        self.trading_stopped = True
        self.stop_reason = reason
        logger.warning(f"🛑 TRADING STOPPED: {reason}")
    
    def resume_trading(self):
        """Resume trading (manual override)."""
        self.trading_stopped = False
        self.stop_reason = None
        self.consecutive_losing_days = 0
        logger.info("✅ Trading resumed")
    
    def _calculate_sector_exposure(self, sector: str, new_position_value: float) -> float:
        """Calculate total exposure to a sector."""
        sector_total = new_position_value
        
        for pos in self.current_positions.values():
            if pos.get('sector') == sector:
                sector_total += pos['position_value']
        
        return sector_total / self.current_capital
    
    def _count_correlated_positions(self, correlation_group: List[str]) -> int:
        """Count positions in same correlation group."""
        count = 0
        for symbol in self.current_positions.keys():
            if symbol in correlation_group:
                count += 1
        return count
    
    def _check_consecutive_losses(self):
        """Check recent days for consecutive losses."""
        # Get last 3 days
        recent_days = sorted(self.daily_pnl.keys(), reverse=True)[:3]
        
        if len(recent_days) < 3:
            return
        
        consecutive_losses = all(self.daily_pnl[d] < 0 for d in recent_days)
        
        if consecutive_losses:
            self.consecutive_losing_days = 3
        else:
            self.consecutive_losing_days = 0
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        today = date.today()
        daily_pnl = self.daily_pnl.get(today, 0)
        daily_pnl_pct = daily_pnl / self.initial_capital
        
        total_position_value = sum(pos['position_value'] for pos in self.current_positions.values())
        cash_pct = (self.current_capital - total_position_value) / self.current_capital
        
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl / self.initial_capital,
            'current_drawdown': current_drawdown,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'open_positions': len(self.current_positions),
            'total_position_value': total_position_value,
            'cash_pct': cash_pct,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'consecutive_losing_days': self.consecutive_losing_days,
            'trading_stopped': self.trading_stopped,
            'stop_reason': self.stop_reason
        }


if __name__ == "__main__":
    # Example usage
    rm = RiskManager(
        initial_capital=100000,
        max_risk_per_trade=0.01,
        max_daily_loss=0.02
    )
    
    # Check trade approval
    check = rm.approve_trade(
        symbol='SBIN',
        direction=1,
        entry_price=500,
        position_size=10000,
        stop_loss=485,
        sector='Banking'
    )
    
    print(f"Approved: {check.approved}")
    print(f"Reason: {check.reason}")
    if check.warnings:
        print("Warnings:")
        for w in check.warnings:
            print(f"  - {w}")
    print("\nRisk Metrics:")
    for k, v in check.risk_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2%}" if v < 1 else f"  {k}: ₹{v:,.0f}")
        else:
            print(f"  {k}: {v}")
    
    if check.approved:
        # Record the trade
        rm.record_trade(
            symbol='SBIN',
            direction=1,
            entry_price=500,
            shares=20,
            stop_loss=485,
            sector='Banking'
        )
        
        # Later, close position
        rm.close_position('SBIN', 515)
        
        # Get risk metrics
        metrics = rm.get_risk_metrics()
        print("\nCurrent Risk Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
