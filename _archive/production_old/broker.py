"""
================================================================================
BROKER INTEGRATION MODULE
================================================================================
Unified broker interface supporting:
- Angel One (live trading)
- Paper trading (simulation)
- Trade execution with safety checks
- Real-time market data
================================================================================
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Try importing Angel One API
try:
    from SmartApi import SmartConnect
    import pyotp
    ANGEL_AVAILABLE = True
except ImportError:
    ANGEL_AVAILABLE = False
    logger.warning("SmartAPI not installed. Install with: pip install smartapi-python pyotp")


class OrderStatus(Enum):
    PENDING = 'PENDING'
    EXECUTED = 'EXECUTED'
    REJECTED = 'REJECTED'
    CANCELLED = 'CANCELLED'


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    status: OrderStatus
    quantity: int
    price: float
    transaction_type: str
    timestamp: datetime
    message: str = ''


@dataclass
class Position:
    """Current position"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    pnl: float
    pnl_pct: float


@dataclass
class AccountInfo:
    """Account information"""
    total_value: float
    cash: float
    positions_value: float
    day_pnl: float
    margin_used: float
    margin_available: float


class BaseBroker:
    """Base broker interface"""

    def authenticate(self) -> bool:
        raise NotImplementedError

    def get_quote(self, symbol: str) -> Dict:
        raise NotImplementedError

    def place_market_order(self, symbol: str, quantity: int, side: str) -> OrderResult:
        raise NotImplementedError

    def place_limit_order(self, symbol: str, quantity: int, side: str, price: float) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_positions(self) -> List[Position]:
        raise NotImplementedError

    def get_account_info(self) -> AccountInfo:
        raise NotImplementedError


class AngelOneBroker(BaseBroker):
    """
    Angel One SmartAPI broker integration.

    Credentials from environment:
    - ANGEL_API_KEY
    - ANGEL_CLIENT_ID
    - ANGEL_PASSWORD
    - ANGEL_TOTP_SECRET
    """

    def __init__(self):
        if not ANGEL_AVAILABLE:
            raise ImportError("SmartAPI not available")

        self.api_key = os.getenv('ANGEL_API_KEY')
        self.client_id = os.getenv('ANGEL_CLIENT_ID')
        self.password = os.getenv('ANGEL_PASSWORD')
        self.totp_secret = os.getenv('ANGEL_TOTP_SECRET')

        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            raise ValueError("Missing Angel One credentials in .env file")

        self.smart_api = SmartConnect(api_key=self.api_key)
        self.is_authenticated = False
        self._token_cache = {}

        logger.info("AngelOneBroker initialized")

    def authenticate(self) -> bool:
        """Authenticate with Angel One"""
        try:
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()

            data = self.smart_api.generateSession(
                self.client_id,
                self.password,
                current_totp
            )

            if data.get('status'):
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                self.is_authenticated = True
                logger.success(f"Authenticated with Angel One: {self.client_id}")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def _ensure_auth(self):
        """Ensure authenticated before operations"""
        if not self.is_authenticated:
            if not self.authenticate():
                raise ConnectionError("Failed to authenticate with Angel One")

    def _get_token(self, symbol: str, exchange: str = 'NSE') -> str:
        """Get instrument token for symbol"""
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        try:
            result = self.smart_api.searchScrip(exchange, symbol)
            if result['status'] and result['data']:
                token = result['data'][0]['symboltoken']
                self._token_cache[cache_key] = token
                return token
        except Exception as e:
            logger.error(f"Token lookup error for {symbol}: {e}")

        raise ValueError(f"Token not found for {symbol}")

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        self._ensure_auth()

        try:
            token = self._get_token(symbol)
            response = self.smart_api.ltpData('NSE', f'{symbol}-EQ', token)

            if response['status'] and response['data']:
                data = response['data']
                return {
                    'symbol': symbol,
                    'ltp': float(data.get('ltp', 0)),
                    'open': float(data.get('open', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'close': float(data.get('close', 0)),
                    'volume': int(data.get('volume', 0)),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")

        return {}

    def place_market_order(self, symbol: str, quantity: int, side: str) -> OrderResult:
        """Place market order"""
        self._ensure_auth()

        try:
            token = self._get_token(symbol)

            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": f"{symbol}-EQ",
                "symboltoken": token,
                "transactiontype": side.upper(),
                "exchange": "NSE",
                "ordertype": "MARKET",
                "producttype": "DELIVERY",
                "quantity": quantity,
                "price": 0,
                "duration": "DAY"
            }

            response = self.smart_api.placeOrder(order_params)

            if response['status']:
                order_id = response['data']['orderid']
                logger.info(f"Order placed: {order_id} | {symbol} {side} {quantity}")

                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.EXECUTED,
                    quantity=quantity,
                    price=0,  # Market order
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message='Order placed successfully'
                )
            else:
                return OrderResult(
                    order_id='',
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    quantity=quantity,
                    price=0,
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message=response.get('message', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Order error: {e}")
            return OrderResult(
                order_id='',
                symbol=symbol,
                status=OrderStatus.REJECTED,
                quantity=quantity,
                price=0,
                transaction_type=side,
                timestamp=datetime.now(),
                message=str(e)
            )

    def place_limit_order(self, symbol: str, quantity: int, side: str, price: float) -> OrderResult:
        """Place limit order"""
        self._ensure_auth()

        try:
            token = self._get_token(symbol)

            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": f"{symbol}-EQ",
                "symboltoken": token,
                "transactiontype": side.upper(),
                "exchange": "NSE",
                "ordertype": "LIMIT",
                "producttype": "DELIVERY",
                "quantity": quantity,
                "price": price,
                "duration": "DAY"
            }

            response = self.smart_api.placeOrder(order_params)

            if response['status']:
                order_id = response['data']['orderid']
                logger.info(f"Limit order placed: {order_id} | {symbol} {side} {quantity} @ {price}")

                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.PENDING,
                    quantity=quantity,
                    price=price,
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message='Limit order placed'
                )
            else:
                return OrderResult(
                    order_id='',
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    quantity=quantity,
                    price=price,
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message=response.get('message', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return OrderResult(
                order_id='',
                symbol=symbol,
                status=OrderStatus.REJECTED,
                quantity=quantity,
                price=price,
                transaction_type=side,
                timestamp=datetime.now(),
                message=str(e)
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        self._ensure_auth()

        try:
            response = self.smart_api.cancelOrder(order_id, "NORMAL")
            if response['status']:
                logger.info(f"Order cancelled: {order_id}")
                return True
        except Exception as e:
            logger.error(f"Cancel error: {e}")

        return False

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        self._ensure_auth()

        try:
            response = self.smart_api.position()
            if response['status'] and response['data']:
                positions = []
                for pos in response['data']:
                    if int(pos['netqty']) != 0:
                        positions.append(Position(
                            symbol=pos['tradingsymbol'].replace('-EQ', ''),
                            quantity=int(pos['netqty']),
                            average_price=float(pos['averageprice']),
                            current_price=float(pos.get('ltp', pos['averageprice'])),
                            pnl=float(pos.get('pnl', 0)),
                            pnl_pct=float(pos.get('pnl', 0)) / (float(pos['averageprice']) * int(pos['netqty']) + 1e-10) * 100
                        ))
                return positions
        except Exception as e:
            logger.error(f"Positions error: {e}")

        return []

    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        self._ensure_auth()

        try:
            response = self.smart_api.rmsLimit()
            if response['status'] and response['data']:
                data = response['data']
                cash = float(data.get('availablecash', 0))
                margin_used = float(data.get('utiliseddebits', 0))

                positions = self.get_positions()
                positions_value = sum(p.quantity * p.current_price for p in positions)

                return AccountInfo(
                    total_value=cash + positions_value,
                    cash=cash,
                    positions_value=positions_value,
                    day_pnl=sum(p.pnl for p in positions),
                    margin_used=margin_used,
                    margin_available=cash
                )
        except Exception as e:
            logger.error(f"Account info error: {e}")

        return AccountInfo(0, 0, 0, 0, 0, 0)


class PaperBroker(BaseBroker):
    """
    Paper trading broker for testing.
    Uses real-time prices from Yahoo Finance but simulates orders.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[OrderResult] = []
        self.order_counter = 0
        self.is_authenticated = True

        logger.info(f"PaperBroker initialized with Rs {initial_capital:,.2f}")

    def authenticate(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Dict:
        """Get quote from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period='1d')

            if len(data) > 0:
                return {
                    'symbol': symbol,
                    'ltp': float(data['Close'].iloc[-1]),
                    'open': float(data['Open'].iloc[-1]),
                    'high': float(data['High'].iloc[-1]),
                    'low': float(data['Low'].iloc[-1]),
                    'close': float(data['Close'].iloc[-1]),
                    'volume': int(data['Volume'].iloc[-1]),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")

        return {'symbol': symbol, 'ltp': 0}

    def place_market_order(self, symbol: str, quantity: int, side: str) -> OrderResult:
        """Simulate market order"""
        quote = self.get_quote(symbol)
        price = quote.get('ltp', 0)

        if price == 0:
            return OrderResult(
                order_id='',
                symbol=symbol,
                status=OrderStatus.REJECTED,
                quantity=quantity,
                price=0,
                transaction_type=side,
                timestamp=datetime.now(),
                message='Could not get price'
            )

        # Add slippage
        slippage = price * 0.001
        if side.upper() == 'BUY':
            execution_price = price + slippage
        else:
            execution_price = price - slippage

        # Check funds for buy
        if side.upper() == 'BUY':
            total_cost = execution_price * quantity * 1.001  # Include commission
            if total_cost > self.cash:
                return OrderResult(
                    order_id='',
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    quantity=quantity,
                    price=execution_price,
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message=f'Insufficient funds: need {total_cost:.2f}, have {self.cash:.2f}'
                )

            # Execute buy
            self.cash -= total_cost
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_value = pos.quantity * pos.average_price + quantity * execution_price
                total_qty = pos.quantity + quantity
                pos.average_price = total_value / total_qty
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=execution_price,
                    current_price=price,
                    pnl=0,
                    pnl_pct=0
                )

        else:  # SELL
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                return OrderResult(
                    order_id='',
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    quantity=quantity,
                    price=execution_price,
                    transaction_type=side,
                    timestamp=datetime.now(),
                    message='Insufficient position'
                )

            # Execute sell
            proceeds = execution_price * quantity * 0.999  # Minus commission
            self.cash += proceeds

            pos = self.positions[symbol]
            pos.pnl += (execution_price - pos.average_price) * quantity
            pos.quantity -= quantity

            if pos.quantity == 0:
                del self.positions[symbol]

        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"

        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            status=OrderStatus.EXECUTED,
            quantity=quantity,
            price=execution_price,
            transaction_type=side,
            timestamp=datetime.now(),
            message='Paper order executed'
        )

        self.orders.append(result)
        logger.info(f"Paper {side}: {symbol} x {quantity} @ Rs {execution_price:.2f}")

        return result

    def place_limit_order(self, symbol: str, quantity: int, side: str, price: float) -> OrderResult:
        """Simulate limit order (execute immediately if price is favorable)"""
        quote = self.get_quote(symbol)
        current_price = quote.get('ltp', 0)

        # Check if limit order can be filled
        if side.upper() == 'BUY' and current_price <= price:
            return self.place_market_order(symbol, quantity, side)
        elif side.upper() == 'SELL' and current_price >= price:
            return self.place_market_order(symbol, quantity, side)

        # Order remains pending (not supported in simple paper trading)
        return OrderResult(
            order_id='',
            symbol=symbol,
            status=OrderStatus.PENDING,
            quantity=quantity,
            price=price,
            transaction_type=side,
            timestamp=datetime.now(),
            message='Limit order not immediately fillable'
        )

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> List[Position]:
        """Get paper positions with updated prices"""
        for symbol, pos in self.positions.items():
            quote = self.get_quote(symbol)
            if quote.get('ltp', 0) > 0:
                pos.current_price = quote['ltp']
                pos.pnl = (pos.current_price - pos.average_price) * pos.quantity
                pos.pnl_pct = pos.pnl / (pos.average_price * pos.quantity) * 100

        return list(self.positions.values())

    def get_account_info(self) -> AccountInfo:
        """Get paper account info"""
        positions = self.get_positions()
        positions_value = sum(p.quantity * p.current_price for p in positions)

        return AccountInfo(
            total_value=self.cash + positions_value,
            cash=self.cash,
            positions_value=positions_value,
            day_pnl=sum(p.pnl for p in positions),
            margin_used=0,
            margin_available=self.cash
        )


def create_broker(paper_trading: bool = True, **kwargs) -> BaseBroker:
    """
    Factory function to create appropriate broker.

    Args:
        paper_trading: Use paper trading (default True for safety)
        **kwargs: Additional arguments for broker

    Returns:
        Broker instance
    """
    if paper_trading:
        return PaperBroker(kwargs.get('initial_capital', 100000))
    else:
        return AngelOneBroker()
