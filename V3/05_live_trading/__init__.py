"""
Live trading module — Angel One API integration.

- angel_one_client.py: SmartAPI REST + WebSocket wrapper
- order_manager.py: Order routing and fill tracking
- paper_trader.py: Dry-run mode (before live trading)
- risk_guard.py: Pre-trade risk checks (position limits, daily loss, etc.)
- signal_publisher.py: Convert predictions → structured orders
"""
