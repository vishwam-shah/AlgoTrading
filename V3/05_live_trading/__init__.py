"""
Live trading module — Angel One API integration.

- angel_one_client.py : SmartAPI REST + WebSocket wrapper
- order_manager.py    : Order routing and fill tracking
- paper_trader.py     : Dry-run mode (before live trading)
- risk_guard.py       : Pre-trade risk checks (position limits, daily loss, etc.)
- signal_publisher.py : Convert predictions → structured orders
- daily_runner.py     : Master orchestrator (evening → morning → reconcile)
"""

from pathlib import Path
import sys

# Ensure V3 root is on path when this package is imported from backend
_V3_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_V3_ROOT), str(_V3_ROOT / "07_pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
