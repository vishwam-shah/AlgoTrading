"""
Angel One SmartAPI Integration
==============================

Wrapper for Angel One SmartAPI for live trading in Indian stock market.

Setup:
1. Create account at Angel One
2. Go to https://smartapi.angelbroking.com/
3. Register as developer and create app
4. Get API Key and configure TOTP
5. Store credentials in .env file (gitignored)
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

# Try to import SmartAPI
try:
    from SmartApi import SmartConnect
    import pyotp
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False
    logger.warning("SmartAPI not installed. Run: pip install smartapi-python pyotp")


class OrderType(Enum):
    """Order types supported by Angel One"""
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOPLOSS_LIMIT = 'STOPLOSS_LIMIT'
    STOPLOSS_MARKET = 'STOPLOSS_MARKET'


class ProductType(Enum):
    """Product types"""
    DELIVERY = 'DELIVERY'  # CNC
    INTRADAY = 'INTRADAY'  # MIS
    MARGIN = 'MARGIN'
    BO = 'BO'  # Bracket Order
    CO = 'CO'  # Cover Order


class TransactionType(Enum):
    """Transaction types"""
    BUY = 'BUY'
    SELL = 'SELL'


class Exchange(Enum):
    """Exchanges"""
    NSE = 'NSE'
    BSE = 'BSE'
    NFO = 'NFO'  # NSE F&O


@dataclass
class OrderParams:
    """Order parameters"""
    symbol: str
    token: str
    quantity: int
    transaction_type: TransactionType
    order_type: OrderType = OrderType.MARKET
    product_type: ProductType = ProductType.DELIVERY
    price: float = 0.0
    trigger_price: float = 0.0
    exchange: Exchange = Exchange.NSE
    variety: str = 'NORMAL'


@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: int
    average_price: float
    pnl: float
    ltp: float
    product_type: str


@dataclass
class Order:
    """Order information"""
    order_id: str
    symbol: str
    status: str
    quantity: int
    price: float
    transaction_type: str
    order_type: str


class AngelOneAPI:
    """
    Angel One SmartAPI Wrapper.

    Provides methods for:
    - Authentication
    - Order placement and management
    - Position tracking
    - Real-time quotes
    - Historical data
    """

    def __init__(
        self,
        api_key: str = None,
        client_id: str = None,
        password: str = None,
        totp_secret: str = None,
        config_path: str = None
    ):
        """
        Initialize Angel One API.

        Args:
            api_key: SmartAPI key
            client_id: Angel One client ID
            password: Trading password
            totp_secret: TOTP secret for 2FA
            config_path: Path to config file with credentials
        """
        if not SMARTAPI_AVAILABLE:
            raise ImportError("SmartAPI not available. Install with: pip install smartapi-python pyotp")

        # Load credentials
        self._load_credentials(api_key, client_id, password, totp_secret, config_path)

        # Initialize SmartConnect
        self.smart_api = SmartConnect(api_key=self.api_key)

        # Session state
        self.is_authenticated = False
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None

        # Symbol token cache
        self._token_cache: Dict[str, str] = {}

        logger.info("AngelOneAPI initialized")

    def _load_credentials(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
        config_path: str
    ):
        """Load credentials from arguments, config file, or environment"""
        # Try arguments first
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret

        # Try config file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = self.api_key or config.get('api_key')
                self.client_id = self.client_id or config.get('client_id')
                self.password = self.password or config.get('password')
                self.totp_secret = self.totp_secret or config.get('totp_secret')

        # Try environment variables
        self.api_key = self.api_key or os.getenv('ANGEL_API_KEY')
        self.client_id = self.client_id or os.getenv('ANGEL_CLIENT_ID')
        self.password = self.password or os.getenv('ANGEL_PASSWORD')
        self.totp_secret = self.totp_secret or os.getenv('ANGEL_TOTP_SECRET')

        # Validate
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            logger.warning("Some credentials missing. Set via arguments, config file, or environment variables.")

    def authenticate(self) -> bool:
        """
        Authenticate with Angel One.

        Returns:
            True if authentication successful
        """
        try:
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()

            # Generate session
            data = self.smart_api.generateSession(
                self.client_id,
                self.password,
                current_totp
            )

            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                self.is_authenticated = True
                logger.info(f"Authentication successful for {self.client_id}")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def logout(self) -> bool:
        """Logout from session"""
        try:
            self.smart_api.terminateSession(self.client_id)
            self.is_authenticated = False
            logger.info("Logged out successfully")
            return True
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

    def get_profile(self) -> Dict:
        """Get user profile"""
        self._ensure_authenticated()
        return self.smart_api.getProfile(self.refresh_token)

    def get_token(self, symbol: str, exchange: str = 'NSE') -> str:
        """
        Get symbol token for trading.

        Args:
            symbol: Stock symbol (e.g., 'HDFCBANK')
            exchange: Exchange (NSE, BSE)

        Returns:
            Symbol token
        """
        cache_key = f"{exchange}:{symbol}"

        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        # Search for symbol
        try:
            search_result = self.smart_api.searchScrip(exchange, symbol)
            if search_result['status'] and search_result['data']:
                token = search_result['data'][0]['symboltoken']
                self._token_cache[cache_key] = token
                return token
        except Exception as e:
            logger.error(f"Error getting token for {symbol}: {e}")

        raise ValueError(f"Token not found for {symbol} on {exchange}")

    def place_order(self, params: OrderParams) -> Optional[str]:
        """
        Place an order.

        Args:
            params: Order parameters

        Returns:
            Order ID if successful
        """
        self._ensure_authenticated()

        order_params = {
            "variety": params.variety,
            "tradingsymbol": f"{params.symbol}-EQ",
            "symboltoken": params.token,
            "transactiontype": params.transaction_type.value,
            "exchange": params.exchange.value,
            "ordertype": params.order_type.value,
            "producttype": params.product_type.value,
            "quantity": params.quantity,
            "price": params.price,
            "triggerprice": params.trigger_price,
            "duration": "DAY"
        }

        try:
            response = self.smart_api.placeOrder(order_params)

            if response['status']:
                order_id = response['data']['orderid']
                logger.info(f"Order placed: {order_id} | {params.symbol} {params.transaction_type.value} {params.quantity}")
                return order_id
            else:
                logger.error(f"Order failed: {response.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        transaction_type: str,
        product_type: str = 'DELIVERY'
    ) -> Optional[str]:
        """
        Place a market order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            transaction_type: 'BUY' or 'SELL'
            product_type: 'DELIVERY' or 'INTRADAY'

        Returns:
            Order ID
        """
        token = self.get_token(symbol)

        params = OrderParams(
            symbol=symbol,
            token=token,
            quantity=quantity,
            transaction_type=TransactionType[transaction_type],
            order_type=OrderType.MARKET,
            product_type=ProductType[product_type]
        )

        return self.place_order(params)

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        transaction_type: str,
        price: float,
        product_type: str = 'DELIVERY'
    ) -> Optional[str]:
        """
        Place a limit order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            transaction_type: 'BUY' or 'SELL'
            price: Limit price
            product_type: 'DELIVERY' or 'INTRADAY'

        Returns:
            Order ID
        """
        token = self.get_token(symbol)

        params = OrderParams(
            symbol=symbol,
            token=token,
            quantity=quantity,
            transaction_type=TransactionType[transaction_type],
            order_type=OrderType.LIMIT,
            product_type=ProductType[product_type],
            price=price
        )

        return self.place_order(params)

    def place_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        transaction_type: str,
        trigger_price: float,
        limit_price: float = None,
        product_type: str = 'DELIVERY'
    ) -> Optional[str]:
        """
        Place a stop loss order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            transaction_type: 'BUY' or 'SELL'
            trigger_price: Stop loss trigger price
            limit_price: Limit price (None for market)
            product_type: Product type

        Returns:
            Order ID
        """
        token = self.get_token(symbol)

        order_type = OrderType.STOPLOSS_LIMIT if limit_price else OrderType.STOPLOSS_MARKET
        price = limit_price or 0.0

        params = OrderParams(
            symbol=symbol,
            token=token,
            quantity=quantity,
            transaction_type=TransactionType[transaction_type],
            order_type=order_type,
            product_type=ProductType[product_type],
            price=price,
            trigger_price=trigger_price
        )

        return self.place_order(params)

    def cancel_order(self, order_id: str, variety: str = 'NORMAL') -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            variety: Order variety

        Returns:
            True if cancelled
        """
        self._ensure_authenticated()

        try:
            response = self.smart_api.cancelOrder(order_id, variety)
            if response['status']:
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Cancel failed: {response.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    def modify_order(
        self,
        order_id: str,
        quantity: int = None,
        price: float = None,
        trigger_price: float = None,
        variety: str = 'NORMAL'
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            trigger_price: New trigger price
            variety: Order variety

        Returns:
            True if modified
        """
        self._ensure_authenticated()

        modify_params = {
            "variety": variety,
            "orderid": order_id
        }

        if quantity:
            modify_params["quantity"] = quantity
        if price:
            modify_params["price"] = price
        if trigger_price:
            modify_params["triggerprice"] = trigger_price

        try:
            response = self.smart_api.modifyOrder(modify_params)
            if response['status']:
                logger.info(f"Order modified: {order_id}")
                return True
            else:
                logger.error(f"Modify failed: {response.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Modify error: {e}")
            return False

    def get_order_book(self) -> List[Order]:
        """Get all orders for the day"""
        self._ensure_authenticated()

        try:
            response = self.smart_api.orderBook()
            if response['status'] and response['data']:
                orders = []
                for order_data in response['data']:
                    orders.append(Order(
                        order_id=order_data['orderid'],
                        symbol=order_data['tradingsymbol'],
                        status=order_data['orderstatus'],
                        quantity=order_data['quantity'],
                        price=float(order_data.get('price', 0)),
                        transaction_type=order_data['transactiontype'],
                        order_type=order_data['ordertype']
                    ))
                return orders
            return []
        except Exception as e:
            logger.error(f"Order book error: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        self._ensure_authenticated()

        try:
            response = self.smart_api.position()
            if response['status'] and response['data']:
                positions = []
                for pos_data in response['data']:
                    positions.append(Position(
                        symbol=pos_data['tradingsymbol'],
                        quantity=int(pos_data['netqty']),
                        average_price=float(pos_data['averageprice']),
                        pnl=float(pos_data.get('pnl', 0)),
                        ltp=float(pos_data.get('ltp', 0)),
                        product_type=pos_data['producttype']
                    ))
                return positions
            return []
        except Exception as e:
            logger.error(f"Position error: {e}")
            return []

    def get_holdings(self) -> List[Dict]:
        """Get current holdings (delivery)"""
        self._ensure_authenticated()

        try:
            response = self.smart_api.holding()
            if response['status'] and response['data']:
                return response['data']
            return []
        except Exception as e:
            logger.error(f"Holdings error: {e}")
            return []

    def get_quote(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol
            exchange: Exchange

        Returns:
            Quote data with LTP, bid/ask, etc.
        """
        self._ensure_authenticated()

        token = self.get_token(symbol, exchange)

        try:
            response = self.smart_api.ltpData(exchange, symbol + '-EQ', token)
            if response['status'] and response['data']:
                return response['data']
            return {}
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return {}

    def get_historical_data(
        self,
        symbol: str,
        interval: str = 'ONE_DAY',
        from_date: str = None,
        to_date: str = None,
        exchange: str = 'NSE'
    ) -> List[Dict]:
        """
        Get historical candle data.

        Args:
            symbol: Stock symbol
            interval: 'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY'
            from_date: Start date (YYYY-MM-DD HH:MM)
            to_date: End date
            exchange: Exchange

        Returns:
            List of candles
        """
        self._ensure_authenticated()

        token = self.get_token(symbol, exchange)

        # Default date range
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        if not from_date:
            from_date = (datetime.now().replace(day=1)).strftime('%Y-%m-%d %H:%M')

        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }

        try:
            response = self.smart_api.getCandleData(params)
            if response['status'] and response['data']:
                return response['data']
            return []
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return []

    def get_funds(self) -> Dict:
        """Get available funds and margins"""
        self._ensure_authenticated()

        try:
            response = self.smart_api.rmsLimit()
            if response['status'] and response['data']:
                return response['data']
            return {}
        except Exception as e:
            logger.error(f"Funds error: {e}")
            return {}

    def _ensure_authenticated(self):
        """Ensure API is authenticated"""
        if not self.is_authenticated:
            if not self.authenticate():
                raise ConnectionError("Failed to authenticate with Angel One")


class PaperTradingAPI:
    """
    Paper trading API that simulates Angel One API.

    Use this for testing without real money.
    """

    def __init__(self, initial_capital: float = 100000):
        """
        Initialize paper trading.

        Args:
            initial_capital: Starting capital
        """
        self.capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
        self.is_authenticated = True

        # Price data cache (would be fetched from real API)
        self._price_cache: Dict[str, float] = {}

        logger.info(f"Paper trading initialized with Rs {initial_capital:,.2f}")

    def authenticate(self) -> bool:
        """Simulate authentication"""
        self.is_authenticated = True
        return True

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        transaction_type: str,
        product_type: str = 'DELIVERY'
    ) -> Optional[str]:
        """Simulate market order"""
        # Get simulated price (would fetch from real API)
        price = self._get_simulated_price(symbol)

        return self._execute_order(symbol, quantity, transaction_type, price)

    def _execute_order(
        self,
        symbol: str,
        quantity: int,
        transaction_type: str,
        price: float
    ) -> Optional[str]:
        """Execute simulated order"""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"

        order_value = quantity * price
        commission = order_value * 0.001  # 0.1% commission

        if transaction_type == 'BUY':
            total_cost = order_value + commission
            if total_cost > self.cash:
                logger.error(f"Insufficient funds: need {total_cost:.2f}, have {self.cash:.2f}")
                return None

            self.cash -= total_cost

            if symbol in self.positions:
                pos = self.positions[symbol]
                # Average price calculation
                total_qty = pos.quantity + quantity
                total_value = pos.quantity * pos.average_price + order_value
                pos.average_price = total_value / total_qty
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=price,
                    pnl=0.0,
                    ltp=price,
                    product_type='DELIVERY'
                )

        elif transaction_type == 'SELL':
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                logger.error(f"Insufficient position to sell")
                return None

            self.cash += order_value - commission

            pos = self.positions[symbol]
            pos.pnl += (price - pos.average_price) * quantity
            pos.quantity -= quantity

            if pos.quantity == 0:
                del self.positions[symbol]

        # Record order
        self.orders.append(Order(
            order_id=order_id,
            symbol=symbol,
            status='EXECUTED',
            quantity=quantity,
            price=price,
            transaction_type=transaction_type,
            order_type='MARKET'
        ))

        logger.info(f"Paper order executed: {order_id} | {symbol} {transaction_type} {quantity} @ {price}")
        return order_id

    def get_positions(self) -> List[Position]:
        """Get paper positions"""
        return list(self.positions.values())

    def get_funds(self) -> Dict:
        """Get paper funds"""
        position_value = sum(p.quantity * p.ltp for p in self.positions.values())
        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_value': self.cash + position_value
        }

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price (placeholder)"""
        # In real implementation, fetch from Yahoo Finance or other source
        default_prices = {
            'HDFCBANK': 1650.0,
            'ICICIBANK': 1050.0,
            'SBIN': 620.0,
            'TCS': 3800.0,
            'INFY': 1450.0,
            'RELIANCE': 2500.0,
        }
        return default_prices.get(symbol, 100.0)


def create_broker_api(paper_trading: bool = True, **kwargs) -> Any:
    """
    Factory function to create broker API.

    Args:
        paper_trading: Use paper trading mode
        **kwargs: API credentials

    Returns:
        AngelOneAPI or PaperTradingAPI instance
    """
    if paper_trading:
        return PaperTradingAPI(kwargs.get('initial_capital', 100000))
    else:
        return AngelOneAPI(**kwargs)
