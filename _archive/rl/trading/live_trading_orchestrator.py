"""
Live Trading Orchestrator
=========================

Combines ML predictions + RL signals for optimized live trading.

Features:
- Real-time prediction generation
- RL-based position sizing and timing
- Portfolio optimization
- Risk management
- Order execution via Angel One API
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd
from loguru import logger

# Add parent paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from src.rl.brokers.angel_one.api import AngelOneAPI, PaperTradingAPI, create_broker_api
from src.rl.brokers.angel_one.websocket_handler import (
    LiveDataHandler, MockLiveDataHandler, Tick, create_live_data_handler
)
from src.rl.config.trading_config import TradingConfig, trading_config


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Prediction:
    """ML model prediction"""
    symbol: str
    timestamp: datetime
    close_return: float  # Predicted close return
    high_return: float   # Predicted high (profit target)
    low_return: float    # Predicted low (stop loss)
    direction: int       # 0 or 1
    confidence: float    # Model confidence
    model_name: str = 'ensemble'


@dataclass
class RLSignal:
    """RL agent signal"""
    symbol: str
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    expected_reward: float
    suggested_position_pct: float


@dataclass
class CombinedSignal:
    """Combined ML + RL signal"""
    symbol: str
    signal_type: SignalType
    ml_prediction: Prediction
    rl_signal: Optional[RLSignal]
    final_confidence: float
    suggested_action: str  # 'BUY', 'SELL', 'HOLD'
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    position_size_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioPosition:
    """Active portfolio position"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    unrealized_pnl: float = 0.0
    days_held: int = 0


class MLPredictor:
    """
    Generates predictions using trained models.
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or config.MODEL_DIR
        self.loaded_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    def load_model(self, symbol: str) -> bool:
        """Load trained model for symbol"""
        try:
            import joblib
            from tensorflow.keras.models import load_model

            model_key = symbol

            # Try to load ensemble first, then XGBoost, then LSTM
            ensemble_path = os.path.join(self.models_dir, 'ensemble', symbol)
            xgb_path = os.path.join(self.models_dir, 'xgboost', symbol)

            if os.path.exists(ensemble_path):
                # Load ensemble components
                self.loaded_models[model_key] = {
                    'type': 'ensemble',
                    'path': ensemble_path
                }
                logger.info(f"Loaded ensemble model for {symbol}")

            elif os.path.exists(xgb_path):
                # Load XGBoost models
                models = {}
                for target in ['close', 'high', 'low', 'direction']:
                    model_file = os.path.join(xgb_path, f'{symbol}_{target}_model.pkl')
                    if os.path.exists(model_file):
                        models[target] = joblib.load(model_file)

                if models:
                    self.loaded_models[model_key] = {
                        'type': 'xgboost',
                        'models': models
                    }
                    logger.info(f"Loaded XGBoost models for {symbol}")

            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scalers', f'{symbol}_feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[model_key] = joblib.load(scaler_path)

            return model_key in self.loaded_models

        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False

    def predict(self, symbol: str, features: np.ndarray) -> Optional[Prediction]:
        """
        Generate prediction for symbol.

        Args:
            symbol: Stock symbol
            features: Feature array (1, n_features)

        Returns:
            Prediction object
        """
        try:
            model_key = symbol

            if model_key not in self.loaded_models:
                if not self.load_model(symbol):
                    return None

            model_info = self.loaded_models[model_key]

            # Scale features if scaler available
            if model_key in self.scalers:
                features = self.scalers[model_key].transform(features)

            if model_info['type'] == 'xgboost':
                models = model_info['models']

                close_pred = models['close'].predict(features)[0] if 'close' in models else 0
                high_pred = models['high'].predict(features)[0] if 'high' in models else 0
                low_pred = models['low'].predict(features)[0] if 'low' in models else 0

                if 'direction' in models:
                    direction_prob = models['direction'].predict_proba(features)[0]
                    direction = int(direction_prob[1] > 0.5)
                    confidence = max(direction_prob)
                else:
                    direction = 1 if close_pred > 0 else 0
                    confidence = 0.5 + abs(close_pred) * 10  # Simple confidence

                return Prediction(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    close_return=close_pred,
                    high_return=high_pred,
                    low_return=low_pred,
                    direction=direction,
                    confidence=min(confidence, 1.0),
                    model_name='xgboost'
                )

            return None

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None


class RLAgent:
    """
    RL agent for position sizing and timing.
    """

    def __init__(self, agent_path: str = None):
        self.agent = None
        self.agent_path = agent_path

    def load_agent(self, symbol: str) -> bool:
        """Load trained RL agent"""
        try:
            # For now, use heuristic-based approach
            # TODO: Load trained PPO/DQN agent
            self.agent = 'heuristic'
            return True
        except Exception as e:
            logger.error(f"Error loading RL agent: {e}")
            return False

    def get_signal(
        self,
        symbol: str,
        prediction: Prediction,
        portfolio_state: Dict,
        market_state: Dict
    ) -> RLSignal:
        """
        Get RL signal based on current state.

        Args:
            symbol: Stock symbol
            prediction: ML prediction
            portfolio_state: Current portfolio state
            market_state: Current market conditions

        Returns:
            RLSignal with action and confidence
        """
        # Heuristic-based RL signal (replace with trained agent)
        confidence = prediction.confidence

        # Direction-based action
        if prediction.direction == 1 and confidence > 0.55:
            action = 1  # Buy
            expected_reward = prediction.close_return * 100  # Convert to percentage
        elif prediction.direction == 0 and confidence > 0.55:
            action = 2  # Sell
            expected_reward = -prediction.close_return * 100
        else:
            action = 0  # Hold
            expected_reward = 0

        # Position sizing based on confidence and portfolio state
        if action in [1, 2]:
            # Higher confidence = larger position
            base_position = 0.1  # 10% base
            confidence_multiplier = (confidence - 0.5) * 2  # 0 to 1
            suggested_position = base_position * (1 + confidence_multiplier)

            # Adjust for existing positions
            current_exposure = portfolio_state.get('total_exposure_pct', 0)
            max_exposure = 0.8  # Max 80% exposure

            available_capacity = max(0, max_exposure - current_exposure)
            suggested_position = min(suggested_position, available_capacity)
        else:
            suggested_position = 0

        return RLSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            expected_reward=expected_reward,
            suggested_position_pct=suggested_position
        )


class PortfolioOptimizer:
    """
    Optimizes portfolio allocation using predictions.
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or trading_config

    def optimize(
        self,
        signals: Dict[str, CombinedSignal],
        current_portfolio: Dict[str, PortfolioPosition],
        available_capital: float
    ) -> List[Tuple[str, str, float, float]]:
        """
        Optimize portfolio allocation.

        Args:
            signals: Combined signals for each symbol
            current_portfolio: Current positions
            available_capital: Available cash

        Returns:
            List of (symbol, action, quantity, price) tuples
        """
        orders = []

        # Score all buy signals
        buy_candidates = []
        for symbol, signal in signals.items():
            if signal.suggested_action == 'BUY' and signal.final_confidence > 0.55:
                score = signal.final_confidence * abs(signal.ml_prediction.close_return)
                buy_candidates.append((symbol, signal, score))

        # Sort by score (highest first)
        buy_candidates.sort(key=lambda x: x[2], reverse=True)

        # Allocate capital to top candidates
        remaining_capital = available_capital
        max_positions = 5  # Max concurrent positions

        current_count = len(current_portfolio)
        available_slots = max_positions - current_count

        for symbol, signal, score in buy_candidates[:available_slots]:
            if remaining_capital < self.config.capital.min_trade_value:
                break

            # Calculate position size
            position_value = min(
                remaining_capital * signal.position_size_pct,
                self.config.capital.max_trade_value,
                remaining_capital * 0.25  # Max 25% per position
            )

            if position_value >= self.config.capital.min_trade_value and signal.entry_price:
                quantity = int(position_value / signal.entry_price)
                if quantity > 0:
                    orders.append((symbol, 'BUY', quantity, signal.entry_price))
                    remaining_capital -= quantity * signal.entry_price

        # Check for sell signals on existing positions
        for symbol, position in current_portfolio.items():
            if symbol in signals:
                signal = signals[symbol]
                if signal.suggested_action == 'SELL':
                    orders.append((symbol, 'SELL', position.quantity, signal.entry_price or position.current_price))

        return orders


class LiveTradingOrchestrator:
    """
    Main orchestrator for live trading.

    Combines:
    - ML predictions
    - RL signals
    - Portfolio optimization
    - Risk management
    - Order execution
    """

    def __init__(
        self,
        symbols: List[str],
        paper_trading: bool = True,
        config: TradingConfig = None,
        initial_capital: float = 100000
    ):
        self.symbols = symbols
        self.paper_trading = paper_trading
        self.config = config or trading_config
        self.initial_capital = initial_capital

        # Components
        self.broker = create_broker_api(paper_trading=paper_trading, initial_capital=initial_capital)
        self.data_handler = create_live_data_handler(paper_trading=paper_trading, symbols=symbols)
        self.predictor = MLPredictor()
        self.rl_agent = RLAgent()
        self.optimizer = PortfolioOptimizer(config)

        # State
        self.portfolio: Dict[str, PortfolioPosition] = {}
        self.cash = initial_capital
        self.signals: Dict[str, CombinedSignal] = {}
        self.pending_orders: List[Dict] = []
        self.trade_history: List[Dict] = []

        # Feature cache
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.price_history: Dict[str, deque] = {s: deque(maxlen=100) for s in symbols}

        # Control
        self._running = False
        self._lock = threading.Lock()

        logger.info(f"LiveTradingOrchestrator initialized")
        logger.info(f"  Paper Trading: {paper_trading}")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Initial Capital: Rs {initial_capital:,.2f}")

    def start(self):
        """Start live trading"""
        logger.info("Starting live trading orchestrator...")

        # Authenticate broker
        if not self.paper_trading:
            if not self.broker.authenticate():
                logger.error("Broker authentication failed")
                return False

        # Connect data handler
        self.data_handler.connect()

        # Subscribe to symbols
        # For live trading, we'd need to get symbol tokens
        self.data_handler.subscribe(self.symbols)

        # Add tick callback
        self.data_handler.add_tick_callback(self._on_tick)

        # Load models
        for symbol in self.symbols:
            self.predictor.load_model(symbol)
            self.rl_agent.load_agent(symbol)

        self._running = True

        # Start main loop in background
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()

        logger.success("Live trading orchestrator started")
        return True

    def stop(self):
        """Stop live trading"""
        logger.info("Stopping live trading orchestrator...")
        self._running = False

        # Close all positions
        self._close_all_positions()

        # Disconnect
        self.data_handler.disconnect()

        if not self.paper_trading:
            self.broker.logout()

        logger.info("Live trading orchestrator stopped")

    def _on_tick(self, tick: Tick):
        """Handle incoming tick data"""
        with self._lock:
            # Update price history
            if tick.symbol in self.price_history:
                self.price_history[tick.symbol].append({
                    'timestamp': tick.timestamp,
                    'open': tick.open,
                    'high': tick.high,
                    'low': tick.low,
                    'close': tick.ltp,
                    'volume': tick.volume
                })

            # Update position P&L
            if tick.symbol in self.portfolio:
                pos = self.portfolio[tick.symbol]
                pos.current_price = tick.ltp
                pos.unrealized_pnl = (tick.ltp - pos.entry_price) * pos.quantity

                # Check stop loss / target
                self._check_exit_conditions(tick.symbol, tick.ltp)

    def _main_loop(self):
        """Main trading loop"""
        last_prediction_time = datetime.min

        while self._running:
            try:
                current_time = datetime.now()

                # Generate predictions every minute
                if (current_time - last_prediction_time).seconds >= 60:
                    self._generate_signals()
                    self._evaluate_entries()
                    last_prediction_time = current_time

                # Update portfolio state
                self._update_portfolio()

                # Print status every 30 seconds
                if current_time.second % 30 == 0:
                    self._print_status()

                time.sleep(1)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)

    def _generate_signals(self):
        """Generate combined ML + RL signals"""
        with self._lock:
            for symbol in self.symbols:
                try:
                    # Get latest features (would compute from price history)
                    features = self._get_features(symbol)
                    if features is None:
                        continue

                    # ML prediction
                    prediction = self.predictor.predict(symbol, features)
                    if prediction is None:
                        continue

                    # Portfolio state
                    portfolio_state = self._get_portfolio_state()

                    # Market state
                    market_state = self._get_market_state()

                    # RL signal
                    rl_signal = self.rl_agent.get_signal(
                        symbol, prediction, portfolio_state, market_state
                    )

                    # Combine signals
                    combined = self._combine_signals(symbol, prediction, rl_signal)
                    self.signals[symbol] = combined

                except Exception as e:
                    logger.error(f"Signal generation error for {symbol}: {e}")

    def _get_features(self, symbol: str) -> Optional[np.ndarray]:
        """Get features for prediction"""
        # Get price history
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return None

        history = list(self.price_history[symbol])
        df = pd.DataFrame(history)

        # Compute basic features (simplified)
        features = []

        # Returns
        df['return'] = df['close'].pct_change()
        features.extend([
            df['return'].iloc[-1],  # Latest return
            df['return'].mean(),    # Mean return
            df['return'].std(),     # Volatility
        ])

        # Price ratios
        features.extend([
            df['close'].iloc[-1] / df['close'].iloc[-5] - 1,   # 5-day return
            df['close'].iloc[-1] / df['close'].iloc[-10] - 1,  # 10-day return
        ])

        # RSI (simplified)
        gains = df['return'].clip(lower=0).rolling(14).mean()
        losses = (-df['return'].clip(upper=0)).rolling(14).mean()
        rs = gains / (losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100)

        # Volume ratio
        features.append(df['volume'].iloc[-1] / df['volume'].mean())

        # Fill NaN with 0
        features = [0 if np.isnan(f) else f for f in features]

        return np.array(features).reshape(1, -1)

    def _combine_signals(
        self,
        symbol: str,
        prediction: Prediction,
        rl_signal: RLSignal
    ) -> CombinedSignal:
        """Combine ML prediction and RL signal"""
        # Weight ML and RL signals
        ml_weight = 0.6
        rl_weight = 0.4

        # Combine confidence
        combined_confidence = (
            ml_weight * prediction.confidence +
            rl_weight * rl_signal.confidence
        )

        # Determine action
        if rl_signal.action == 1 and prediction.direction == 1:
            signal_type = SignalType.STRONG_BUY if combined_confidence > 0.7 else SignalType.BUY
            action = 'BUY'
        elif rl_signal.action == 2 or prediction.direction == 0:
            signal_type = SignalType.STRONG_SELL if combined_confidence > 0.7 else SignalType.SELL
            action = 'SELL'
        else:
            signal_type = SignalType.HOLD
            action = 'HOLD'

        # Get current price
        tick = self.data_handler.get_tick(symbol)
        current_price = tick.ltp if tick else None

        # Calculate targets
        if current_price:
            target_price = current_price * (1 + prediction.high_return) if prediction.direction == 1 else None
            stop_loss = current_price * (1 + prediction.low_return) if action != 'HOLD' else None
        else:
            target_price = None
            stop_loss = None

        return CombinedSignal(
            symbol=symbol,
            signal_type=signal_type,
            ml_prediction=prediction,
            rl_signal=rl_signal,
            final_confidence=combined_confidence,
            suggested_action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size_pct=rl_signal.suggested_position_pct,
            timestamp=datetime.now()
        )

    def _evaluate_entries(self):
        """Evaluate and execute entry signals"""
        with self._lock:
            # Get optimized orders
            orders = self.optimizer.optimize(
                self.signals,
                self.portfolio,
                self.cash
            )

            # Execute orders
            for symbol, action, quantity, price in orders:
                self._execute_order(symbol, action, quantity, price)

    def _execute_order(self, symbol: str, action: str, quantity: int, price: float):
        """Execute trade order"""
        try:
            if action == 'BUY':
                order_id = self.broker.place_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    transaction_type='BUY',
                    product_type='DELIVERY'
                )

                if order_id:
                    # Get signal for stop loss/target
                    signal = self.signals.get(symbol)

                    self.portfolio[symbol] = PortfolioPosition(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=price,
                        current_price=price,
                        entry_time=datetime.now(),
                        stop_loss=signal.stop_loss if signal else price * 0.98,
                        target=signal.target_price if signal else price * 1.05
                    )

                    self.cash -= quantity * price
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'order_id': order_id
                    })

                    logger.success(f"BUY {quantity} {symbol} @ Rs {price:.2f}")

            elif action == 'SELL':
                order_id = self.broker.place_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    transaction_type='SELL',
                    product_type='DELIVERY'
                )

                if order_id and symbol in self.portfolio:
                    pos = self.portfolio[symbol]
                    pnl = (price - pos.entry_price) * quantity

                    self.cash += quantity * price
                    del self.portfolio[symbol]

                    self.trade_history.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'order_id': order_id,
                        'pnl': pnl
                    })

                    logger.success(f"SELL {quantity} {symbol} @ Rs {price:.2f} | P&L: Rs {pnl:.2f}")

        except Exception as e:
            logger.error(f"Order execution error: {e}")

    def _check_exit_conditions(self, symbol: str, current_price: float):
        """Check stop loss and target for position"""
        if symbol not in self.portfolio:
            return

        pos = self.portfolio[symbol]

        # Check stop loss
        if pos.stop_loss and current_price <= pos.stop_loss:
            logger.warning(f"Stop loss triggered for {symbol}")
            self._execute_order(symbol, 'SELL', pos.quantity, current_price)

        # Check target
        elif pos.target and current_price >= pos.target:
            logger.info(f"Target reached for {symbol}")
            self._execute_order(symbol, 'SELL', pos.quantity, current_price)

        # Check max holding period (5 days)
        elif (datetime.now() - pos.entry_time).days >= 5:
            logger.info(f"Max holding period reached for {symbol}")
            self._execute_order(symbol, 'SELL', pos.quantity, current_price)

    def _close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.portfolio.keys()):
            pos = self.portfolio[symbol]
            tick = self.data_handler.get_tick(symbol)
            price = tick.ltp if tick else pos.current_price
            self._execute_order(symbol, 'SELL', pos.quantity, price)

    def _update_portfolio(self):
        """Update portfolio state"""
        total_value = self.cash
        for symbol, pos in self.portfolio.items():
            tick = self.data_handler.get_tick(symbol)
            if tick:
                pos.current_price = tick.ltp
                pos.unrealized_pnl = (tick.ltp - pos.entry_price) * pos.quantity
            total_value += pos.quantity * pos.current_price

        self.total_value = total_value

    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        total_value = self.cash
        total_exposure = 0

        for symbol, pos in self.portfolio.items():
            position_value = pos.quantity * pos.current_price
            total_value += position_value
            total_exposure += position_value

        return {
            'cash': self.cash,
            'total_value': total_value,
            'total_exposure': total_exposure,
            'total_exposure_pct': total_exposure / total_value if total_value > 0 else 0,
            'num_positions': len(self.portfolio)
        }

    def _get_market_state(self) -> Dict:
        """Get current market state"""
        # Simplified market state
        return {
            'market_open': True,
            'volatility': 'normal',
            'trend': 'neutral'
        }

    def _print_status(self):
        """Print current status"""
        state = self._get_portfolio_state()

        print(f"\n{'='*60}")
        print(f"  Live Trading Status - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"  Cash: Rs {state['cash']:,.2f}")
        print(f"  Total Value: Rs {state['total_value']:,.2f}")
        print(f"  P&L: Rs {state['total_value'] - self.initial_capital:,.2f} "
              f"({((state['total_value'] / self.initial_capital) - 1) * 100:.2f}%)")
        print(f"  Positions: {state['num_positions']}")

        if self.portfolio:
            print(f"\n  Open Positions:")
            for symbol, pos in self.portfolio.items():
                pnl_pct = ((pos.current_price / pos.entry_price) - 1) * 100
                print(f"    {symbol}: {pos.quantity} @ Rs {pos.entry_price:.2f} "
                      f"-> Rs {pos.current_price:.2f} ({pnl_pct:+.2f}%)")

        if self.signals:
            print(f"\n  Latest Signals:")
            for symbol, signal in self.signals.items():
                print(f"    {symbol}: {signal.suggested_action} "
                      f"(confidence: {signal.final_confidence:.2%})")

        print(f"{'='*60}\n")

    def get_summary(self) -> Dict:
        """Get trading summary"""
        state = self._get_portfolio_state()

        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
        winning_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        total_trades = len([t for t in self.trade_history if 'pnl' in t])

        return {
            'initial_capital': self.initial_capital,
            'current_value': state['total_value'],
            'cash': state['cash'],
            'total_return_pct': ((state['total_value'] / self.initial_capital) - 1) * 100,
            'total_pnl': total_pnl,
            'num_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'open_positions': len(self.portfolio),
            'trade_history': self.trade_history
        }


def run_paper_trading_demo(symbols: List[str] = None, duration_seconds: int = 60):
    """
    Run paper trading demo.

    Args:
        symbols: Symbols to trade
        duration_seconds: How long to run
    """
    if symbols is None:
        symbols = ['HDFCBANK', 'TCS']

    logger.info(f"Starting paper trading demo for {symbols}")

    orchestrator = LiveTradingOrchestrator(
        symbols=symbols,
        paper_trading=True,
        initial_capital=100000
    )

    try:
        orchestrator.start()

        # Run for specified duration
        time.sleep(duration_seconds)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        orchestrator.stop()

        # Print summary
        summary = orchestrator.get_summary()
        print("\n" + "=" * 60)
        print("TRADING SUMMARY")
        print("=" * 60)
        print(f"Initial Capital: Rs {summary['initial_capital']:,.2f}")
        print(f"Final Value: Rs {summary['current_value']:,.2f}")
        print(f"Total Return: {summary['total_return_pct']:.2f}%")
        print(f"Total P&L: Rs {summary['total_pnl']:,.2f}")
        print(f"Trades: {summary['num_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print("=" * 60)


if __name__ == '__main__':
    run_paper_trading_demo(duration_seconds=120)
