"""
State Builder - Constructs RL State from ML Predictions
========================================================

Integrates existing ML model predictions with portfolio state
to create the observation space for RL agents.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: Dict[str, dict]  # {symbol: {qty, avg_price, unrealized_pnl}}
    total_value: float
    daily_pnl: float
    realized_pnl: float


@dataclass
class MarketState:
    """Current market state"""
    current_price: float
    predicted_direction: int  # 0 or 1
    predicted_close_return: float
    predicted_high_return: float
    predicted_low_return: float
    direction_confidence: float
    volatility: float
    volume_ratio: float
    market_regime: int  # 0=bear, 1=neutral, 2=bull


class StateBuilder:
    """
    Builds observation state for RL agents by combining:
    1. ML model predictions (from existing pipeline)
    2. Portfolio state (positions, cash, P&L)
    3. Market context (volatility, regime, etc.)
    """

    # State dimensions
    PREDICTION_DIMS = 5  # direction, close, high, low, confidence
    PORTFOLIO_DIMS = 4   # position, unrealized_pnl, days_held, cash_ratio
    MARKET_DIMS = 3      # volatility, volume_ratio, regime

    TOTAL_STATE_DIM = PREDICTION_DIMS + PORTFOLIO_DIMS + MARKET_DIMS  # 12 dimensions

    def __init__(
        self,
        predictions_dir: str = None,
        model_priority: List[str] = None
    ):
        """
        Initialize StateBuilder.

        Args:
            predictions_dir: Directory containing prediction CSV files
            model_priority: Order of model preference ['Ensemble', 'XGBoost', 'LSTM', 'GRU']
        """
        self.predictions_dir = predictions_dir or os.path.join(
            BASE_DIR, 'evaluation_results', 'multi_target'
        )
        self.model_priority = model_priority or ['Ensemble', 'XGBoost', 'LSTM', 'GRU']

        # Cache for loaded predictions
        self._prediction_cache: Dict[str, pd.DataFrame] = {}

    def load_predictions(self, symbol: str, force_reload: bool = False) -> pd.DataFrame:
        """
        Load predictions for a symbol from CSV.

        Args:
            symbol: Stock symbol
            force_reload: Force reload from disk

        Returns:
            DataFrame with predictions
        """
        if symbol in self._prediction_cache and not force_reload:
            return self._prediction_cache[symbol]

        # Try to find prediction file
        pred_file = os.path.join(self.predictions_dir, f'{symbol}_predictions.csv')

        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions not found for {symbol}: {pred_file}")

        df = pd.read_csv(pred_file)

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        self._prediction_cache[symbol] = df
        return df

    def get_best_model_predictions(
        self,
        df: pd.DataFrame,
        date: datetime = None
    ) -> Dict[str, float]:
        """
        Get predictions from the best available model.

        Args:
            df: Predictions DataFrame
            date: Specific date (None = latest)

        Returns:
            Dict with direction, close, high, low predictions
        """
        # Get row for date
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            row = df.loc[df.index == date].iloc[-1] if date in df.index else df.iloc[-1]
        else:
            row = df.iloc[-1]

        # Find best model's predictions
        for model in self.model_priority:
            direction_col = f'{model}_direction_pred'
            close_col = f'{model}_close_pred'
            high_col = f'{model}_high_pred'
            low_col = f'{model}_low_pred'

            if all(col in df.columns for col in [direction_col, close_col, high_col, low_col]):
                return {
                    'direction': float(row[direction_col]),
                    'close': float(row[close_col]),
                    'high': float(row[high_col]),
                    'low': float(row[low_col]),
                    'model': model
                }

        # Fallback to any available predictions
        for col in df.columns:
            if 'direction_pred' in col:
                model = col.replace('_direction_pred', '')
                return {
                    'direction': float(row[f'{model}_direction_pred']),
                    'close': float(row.get(f'{model}_close_pred', 0)),
                    'high': float(row.get(f'{model}_high_pred', 0)),
                    'low': float(row.get(f'{model}_low_pred', 0)),
                    'model': model
                }

        raise ValueError("No valid predictions found in DataFrame")

    def calculate_direction_confidence(
        self,
        df: pd.DataFrame,
        date: datetime = None
    ) -> float:
        """
        Calculate confidence based on model agreement.

        Args:
            df: Predictions DataFrame
            date: Specific date

        Returns:
            Confidence score 0-1
        """
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            row = df.loc[df.index == date].iloc[-1] if date in df.index else df.iloc[-1]
        else:
            row = df.iloc[-1]

        # Collect all direction predictions
        directions = []
        for model in self.model_priority:
            col = f'{model}_direction_pred'
            if col in df.columns:
                val = row[col]
                # Skip NaN values
                if pd.notna(val):
                    directions.append(val)

        if not directions:
            return 0.5

        # Confidence = proportion of models agreeing
        majority = round(np.nanmean(directions))
        agreement = sum(1 for d in directions if pd.notna(d) and round(d) == majority) / len(directions)

        return agreement

    def calculate_volatility_state(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> float:
        """
        Calculate normalized volatility state.

        Args:
            df: DataFrame with price data
            lookback: Lookback period

        Returns:
            Normalized volatility (0-1 scale)
        """
        if 'close' not in df.columns:
            return 0.5

        # Calculate historical volatility
        returns = df['close'].pct_change().dropna()
        if len(returns) < lookback:
            return 0.5

        current_vol = returns.iloc[-lookback:].std() * np.sqrt(252)

        # Calculate percentile of current volatility
        rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
        percentile = (rolling_vol < current_vol).mean()

        return float(percentile)

    def calculate_volume_ratio(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> float:
        """
        Calculate volume ratio vs average.

        Args:
            df: DataFrame with volume data
            lookback: Lookback period

        Returns:
            Volume ratio (current / average)
        """
        if 'volume' not in df.columns:
            return 1.0

        avg_volume = df['volume'].iloc[-lookback-1:-1].mean()
        current_volume = df['volume'].iloc[-1]

        if avg_volume > 0:
            ratio = current_volume / avg_volume
            return min(ratio, 5.0)  # Cap at 5x

        return 1.0

    def detect_market_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> int:
        """
        Detect market regime (bear/neutral/bull).

        Args:
            df: DataFrame with price data
            lookback: Lookback period

        Returns:
            Regime: 0=bear, 1=neutral, 2=bull
        """
        if 'close' not in df.columns or len(df) < lookback:
            return 1  # Neutral

        returns = df['close'].pct_change().dropna()
        recent_return = returns.iloc[-lookback:].sum()

        if recent_return < -0.05:  # -5%
            return 0  # Bear
        elif recent_return > 0.05:  # +5%
            return 2  # Bull
        else:
            return 1  # Neutral

    def build_state(
        self,
        symbol: str,
        portfolio: PortfolioState,
        date: datetime = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Build complete state vector for RL agent.

        Args:
            symbol: Stock symbol
            portfolio: Current portfolio state
            date: Date for predictions (None = latest)
            normalize: Whether to normalize state values

        Returns:
            State vector of shape (TOTAL_STATE_DIM,)
        """
        # Load predictions
        df = self.load_predictions(symbol)

        # Get ML predictions
        preds = self.get_best_model_predictions(df, date)
        confidence = self.calculate_direction_confidence(df, date)

        # Get market context
        volatility = self.calculate_volatility_state(df)
        volume_ratio = self.calculate_volume_ratio(df)
        regime = self.detect_market_regime(df)

        # Get portfolio context for this symbol
        position = 0
        unrealized_pnl = 0.0
        days_held = 0

        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            position = 1 if pos['qty'] > 0 else (-1 if pos['qty'] < 0 else 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0) / portfolio.total_value
            days_held = pos.get('days_held', 0)

        cash_ratio = portfolio.cash / portfolio.total_value

        # Build state vector
        state = np.array([
            # Prediction features (5)
            preds['direction'],          # 0 or 1
            preds['close'],              # Expected return
            preds['high'],               # Profit target
            preds['low'],                # Stop loss level
            confidence,                  # Model agreement

            # Portfolio features (4)
            position,                    # -1, 0, or 1
            unrealized_pnl,              # Normalized unrealized P&L
            min(days_held / 10.0, 1.0),  # Normalized days held
            cash_ratio,                  # Cash percentage

            # Market features (3)
            volatility,                  # Volatility percentile
            min(volume_ratio / 5.0, 1.0), # Normalized volume ratio
            regime / 2.0                 # Normalized regime
        ], dtype=np.float32)

        if normalize:
            state = self._normalize_state(state)

        return state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to roughly [-1, 1] range"""
        # Most features are already normalized
        # Just clip to prevent extreme values
        return np.clip(state, -10.0, 10.0)

    def get_state_info(self) -> Dict:
        """Get information about state space"""
        return {
            'total_dim': self.TOTAL_STATE_DIM,
            'prediction_dims': self.PREDICTION_DIMS,
            'portfolio_dims': self.PORTFOLIO_DIMS,
            'market_dims': self.MARKET_DIMS,
            'feature_names': [
                'direction_pred', 'close_pred', 'high_pred', 'low_pred', 'confidence',
                'position', 'unrealized_pnl', 'days_held', 'cash_ratio',
                'volatility', 'volume_ratio', 'regime'
            ]
        }

    def build_batch_states(
        self,
        symbol: str,
        portfolio_states: List[PortfolioState],
        dates: List[datetime]
    ) -> np.ndarray:
        """
        Build batch of states for training.

        Args:
            symbol: Stock symbol
            portfolio_states: List of portfolio states
            dates: List of dates

        Returns:
            State batch of shape (batch_size, TOTAL_STATE_DIM)
        """
        states = []
        for portfolio, date in zip(portfolio_states, dates):
            state = self.build_state(symbol, portfolio, date)
            states.append(state)

        return np.array(states, dtype=np.float32)


# Convenience function
def create_empty_portfolio(initial_capital: float = 100000) -> PortfolioState:
    """Create empty portfolio state"""
    return PortfolioState(
        cash=initial_capital,
        positions={},
        total_value=initial_capital,
        daily_pnl=0.0,
        realized_pnl=0.0
    )
