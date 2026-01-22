"""
Angel One WebSocket Handler for Live Market Data
=================================================

Real-time quote streaming using SmartAPI WebSocket.
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from queue import Queue
from loguru import logger

try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    from SmartApi import SmartConnect
    import pyotp
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("SmartAPI WebSocket not available. Install: pip install smartapi-python")


@dataclass
class Tick:
    """Real-time tick data"""
    symbol: str
    token: str
    ltp: float  # Last traded price
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0  # Previous close
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    change: float = 0.0
    change_pct: float = 0.0


@dataclass
class MarketDepth:
    """Market depth (order book)"""
    symbol: str
    bids: List[Dict] = field(default_factory=list)  # [{price, qty}, ...]
    asks: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class LiveDataHandler:
    """
    Handles real-time market data via WebSocket.

    Features:
    - Subscribe to multiple symbols
    - Callbacks for tick updates
    - Automatic reconnection
    - Rate limiting and throttling
    """

    def __init__(
        self,
        api_key: str = None,
        client_id: str = None,
        password: str = None,
        totp_secret: str = None,
        feed_token: str = None,
        auth_token: str = None
    ):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("SmartAPI WebSocket not available")

        # Load credentials
        self.api_key = api_key or os.getenv('ANGEL_API_KEY')
        self.client_id = client_id or os.getenv('ANGEL_CLIENT_ID')
        self.password = password or os.getenv('ANGEL_PASSWORD')
        self.totp_secret = totp_secret or os.getenv('ANGEL_TOTP_SECRET')

        # Will be set after authentication
        self.feed_token = feed_token
        self.auth_token = auth_token

        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Data storage
        self.latest_ticks: Dict[str, Tick] = {}
        self.tick_history: Dict[str, List[Tick]] = defaultdict(list)
        self.subscribed_tokens: Dict[str, str] = {}  # token -> symbol mapping

        # Callbacks
        self.on_tick_callbacks: List[Callable[[Tick], None]] = []
        self.on_connect_callbacks: List[Callable[[], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []

        # Thread safety
        self._lock = threading.Lock()

        # Message queue for processing
        self._message_queue: Queue = Queue()
        self._processor_thread: Optional[threading.Thread] = None

        logger.info("LiveDataHandler initialized")

    def authenticate(self) -> bool:
        """Authenticate and get feed token"""
        try:
            smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()

            # Generate session
            data = smart_api.generateSession(
                self.client_id,
                self.password,
                current_totp
            )

            if data.get('status'):
                self.auth_token = data['data']['jwtToken']
                self.feed_token = smart_api.getfeedToken()
                logger.success(f"WebSocket authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def connect(self) -> bool:
        """Connect to WebSocket"""
        if not self.feed_token or not self.auth_token:
            if not self.authenticate():
                return False

        try:
            # Create WebSocket connection
            self.ws = SmartWebSocketV2(
                self.auth_token,
                self.api_key,
                self.client_id,
                self.feed_token
            )

            # Set callbacks
            self.ws.on_open = self._on_open
            self.ws.on_data = self._on_data
            self.ws.on_error = self._on_error
            self.ws.on_close = self._on_close

            # Connect in a separate thread
            self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self._ws_thread.start()

            # Start message processor
            self._start_processor()

            logger.info("WebSocket connecting...")
            return True

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

    def _run_websocket(self):
        """Run WebSocket connection"""
        try:
            self.ws.connect()
        except Exception as e:
            logger.error(f"WebSocket run error: {e}")
            self._schedule_reconnect()

    def _on_open(self, ws, message):
        """WebSocket connected callback"""
        logger.success("WebSocket connected")
        self.is_connected = True
        self.reconnect_attempts = 0

        # Trigger callbacks
        for callback in self.on_connect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Connect callback error: {e}")

        # Resubscribe to tokens
        if self.subscribed_tokens:
            self._resubscribe()

    def _on_data(self, ws, message):
        """Process incoming tick data"""
        try:
            self._message_queue.put(message)
        except Exception as e:
            logger.error(f"Error queuing message: {e}")

    def _on_error(self, ws, error):
        """WebSocket error callback"""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, code, reason):
        """WebSocket closed callback"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        self.is_connected = False

        # Trigger callbacks
        for callback in self.on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")

        self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
            logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
            threading.Timer(delay, self.connect).start()
        else:
            logger.error("Max reconnection attempts reached")

    def _start_processor(self):
        """Start message processor thread"""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(target=self._process_messages, daemon=True)
            self._processor_thread.start()

    def _process_messages(self):
        """Process messages from queue"""
        while True:
            try:
                message = self._message_queue.get(timeout=1)
                self._handle_tick(message)
            except Exception:
                continue

    def _handle_tick(self, data: Dict):
        """Process tick data"""
        try:
            token = str(data.get('token', ''))

            with self._lock:
                symbol = self.subscribed_tokens.get(token, token)

                tick = Tick(
                    symbol=symbol,
                    token=token,
                    ltp=float(data.get('last_traded_price', 0)) / 100,  # SmartAPI returns price * 100
                    open=float(data.get('open_price_of_the_day', 0)) / 100,
                    high=float(data.get('high_price_of_the_day', 0)) / 100,
                    low=float(data.get('low_price_of_the_day', 0)) / 100,
                    close=float(data.get('closed_price', 0)) / 100,
                    volume=int(data.get('volume_trade_for_the_day', 0)),
                    timestamp=datetime.now()
                )

                # Calculate change
                if tick.close > 0:
                    tick.change = tick.ltp - tick.close
                    tick.change_pct = (tick.change / tick.close) * 100

                # Store tick
                self.latest_ticks[symbol] = tick
                self.tick_history[symbol].append(tick)

                # Keep only last 1000 ticks per symbol
                if len(self.tick_history[symbol]) > 1000:
                    self.tick_history[symbol] = self.tick_history[symbol][-1000:]

            # Trigger callbacks
            for callback in self.on_tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")

        except Exception as e:
            logger.error(f"Error handling tick: {e}")

    def subscribe(self, symbols: List[str], tokens: Dict[str, str] = None):
        """
        Subscribe to symbols for live data.

        Args:
            symbols: List of stock symbols
            tokens: Dict mapping symbol to token (optional, will lookup if not provided)
        """
        if tokens is None:
            # Would need to lookup tokens from symbol master
            logger.warning("Token lookup not implemented, please provide tokens dict")
            return

        with self._lock:
            for symbol in symbols:
                if symbol in tokens:
                    token = tokens[symbol]
                    self.subscribed_tokens[token] = symbol

        if self.is_connected:
            self._send_subscribe()

    def _send_subscribe(self):
        """Send subscribe message"""
        if not self.subscribed_tokens:
            return

        try:
            # SmartAPI subscription format
            token_list = [
                {"exchangeType": 1, "tokens": list(self.subscribed_tokens.keys())}  # 1 = NSE
            ]

            self.ws.subscribe(
                correlation_id="quote_stream",
                mode=3,  # Full mode (LTP + OHLC)
                token_list=token_list
            )
            logger.info(f"Subscribed to {len(self.subscribed_tokens)} symbols")

        except Exception as e:
            logger.error(f"Subscribe error: {e}")

    def _resubscribe(self):
        """Resubscribe after reconnection"""
        self._send_subscribe()

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        with self._lock:
            tokens_to_remove = []
            for token, symbol in self.subscribed_tokens.items():
                if symbol in symbols:
                    tokens_to_remove.append(token)

            for token in tokens_to_remove:
                del self.subscribed_tokens[token]

        if self.is_connected and tokens_to_remove:
            try:
                token_list = [
                    {"exchangeType": 1, "tokens": tokens_to_remove}
                ]
                self.ws.unsubscribe(
                    correlation_id="quote_stream",
                    mode=3,
                    token_list=token_list
                )
            except Exception as e:
                logger.error(f"Unsubscribe error: {e}")

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        with self._lock:
            tick = self.latest_ticks.get(symbol)
            return tick.ltp if tick else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick for symbol"""
        with self._lock:
            return self.latest_ticks.get(symbol)

    def get_all_ticks(self) -> Dict[str, Tick]:
        """Get all latest ticks"""
        with self._lock:
            return self.latest_ticks.copy()

    def add_tick_callback(self, callback: Callable[[Tick], None]):
        """Add callback for tick updates"""
        self.on_tick_callbacks.append(callback)

    def add_connect_callback(self, callback: Callable[[], None]):
        """Add callback for connection"""
        self.on_connect_callbacks.append(callback)

    def add_disconnect_callback(self, callback: Callable[[], None]):
        """Add callback for disconnection"""
        self.on_disconnect_callbacks.append(callback)

    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.ws:
                self.ws.close_connection()
            self.is_connected = False
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")


class MockLiveDataHandler:
    """Mock live data handler for paper trading"""

    def __init__(self, symbols: List[str] = None, base_prices: Dict[str, float] = None):
        self.symbols = symbols or []
        self.base_prices = base_prices or {
            'HDFCBANK': 1650.0,
            'ICICIBANK': 1050.0,
            'TCS': 3800.0,
            'INFY': 1450.0,
            'RELIANCE': 2500.0,
            'SBIN': 620.0,
        }

        self.latest_ticks: Dict[str, Tick] = {}
        self.on_tick_callbacks: List[Callable] = []
        self.is_connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info("MockLiveDataHandler initialized")

    def connect(self) -> bool:
        """Simulate connection"""
        self.is_connected = True
        self._running = True
        self._thread = threading.Thread(target=self._generate_ticks, daemon=True)
        self._thread.start()
        logger.info("Mock WebSocket connected")
        return True

    def _generate_ticks(self):
        """Generate simulated tick data"""
        import random

        while self._running:
            for symbol in self.symbols or self.base_prices.keys():
                base_price = self.base_prices.get(symbol, 100.0)

                # Add random variation (-0.5% to +0.5%)
                variation = random.uniform(-0.005, 0.005)
                ltp = base_price * (1 + variation)

                tick = Tick(
                    symbol=symbol,
                    token=f"mock_{symbol}",
                    ltp=round(ltp, 2),
                    open=base_price,
                    high=base_price * 1.01,
                    low=base_price * 0.99,
                    close=base_price,
                    volume=random.randint(10000, 100000),
                    timestamp=datetime.now(),
                    change=ltp - base_price,
                    change_pct=variation * 100
                )

                self.latest_ticks[symbol] = tick

                # Update base price for next iteration
                self.base_prices[symbol] = ltp

                # Trigger callbacks
                for callback in self.on_tick_callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            time.sleep(1)  # Generate ticks every second

    def subscribe(self, symbols: List[str], tokens: Dict[str, str] = None):
        """Subscribe to symbols"""
        self.symbols = symbols
        for symbol in symbols:
            if symbol not in self.base_prices:
                self.base_prices[symbol] = 100.0

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        self.symbols = [s for s in self.symbols if s not in symbols]

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get latest price"""
        tick = self.latest_ticks.get(symbol)
        return tick.ltp if tick else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick"""
        return self.latest_ticks.get(symbol)

    def get_all_ticks(self) -> Dict[str, Tick]:
        """Get all ticks"""
        return self.latest_ticks.copy()

    def add_tick_callback(self, callback: Callable[[Tick], None]):
        """Add tick callback"""
        self.on_tick_callbacks.append(callback)

    def disconnect(self):
        """Disconnect"""
        self._running = False
        self.is_connected = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Mock WebSocket disconnected")


def create_live_data_handler(paper_trading: bool = True, **kwargs) -> Any:
    """
    Factory function for live data handler.

    Args:
        paper_trading: Use mock handler for paper trading
        **kwargs: Handler arguments

    Returns:
        LiveDataHandler or MockLiveDataHandler
    """
    if paper_trading:
        return MockLiveDataHandler(**kwargs)
    else:
        return LiveDataHandler(**kwargs)
