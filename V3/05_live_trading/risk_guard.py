"""
risk_guard.py — Pre-trade risk checks and circuit breakers
==========================================================
Called by signal_publisher before any order is emitted.
Also used by order_manager for last-mile checks before placement.

Checks:
  1. Position size ≤ MAX_POSITION_PCT (12%)
  2. Sector concentration ≤ MAX_SECTOR_PCT (30%)
  3. Max simultaneous holdings ≤ MAX_HOLDINGS (15)
  4. Daily portfolio loss ≤ MAX_DAILY_LOSS_PCT (2%)
  5. Minimum prob_up threshold (52%)
  6. Market hours guard (orders only 9:15–15:25 IST)
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

# ── NSE sector map for concentration check ────────────────────────────────────
SECTOR_MAP: Dict[str, str] = {
    # Banking
    "SBIN": "Banking", "HDFCBANK": "Banking", "ICICIBANK": "Banking",
    "AXISBANK": "Banking", "KOTAKBANK": "Banking", "INDUSINDBK": "Banking",
    "BANDHANBNK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking",
    "AUBANK": "Banking", "RBLBANK": "Banking",
    # Finance / NBFC
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance",
    "SBILIFE": "Finance", "ICICIGI": "Finance", "MUTHOOTFIN": "Finance",
    "CHOLAFIN": "Finance", "SHRIRAMFIN": "Finance", "MANAPPURAM": "Finance",
    "BAJAJHFL": "Finance",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT",
    "COFORGE": "IT", "TATAELXSI": "IT", "OFSS": "IT",
    # Auto
    "MARUTI": "Auto", "TVSMOTOR": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "MOTHERSON": "Auto", "BOSCHLTD": "Auto", "EXIDEIND": "Auto",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "GODREJCP": "FMCG",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "LUPIN": "Pharma", "TORNTPHARM": "Pharma",
    "AUROPHARMA": "Pharma", "ALKEM": "Pharma",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "NTPC": "Energy", "POWERGRID": "Energy", "COALINDIA": "Energy",
    "GAIL": "Energy", "TATAPOWER": "Energy",
    # Metals
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NMDC": "Metals",
    # Infra / Capital goods
    "LT": "Infra", "BHEL": "Infra", "SIEMENS": "Infra", "ABB": "Infra",
    "HAVELLS": "Infra", "POLYCAB": "Infra", "CUMMINSIND": "Infra",
    "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom",
    # Cement
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "SHREECEM": "Cement",
    # Consumer / Discretionary
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "PIDILITIND": "Consumer",
    "BERGEPAINT": "Consumer", "VOLTAS": "Consumer", "PAGEIND": "Consumer",
    # Real Estate
    "DLF": "RealEstate", "DMART": "Retail", "GODREJPROP": "RealEstate",
    # Conglomerate / Others
    "ADANIENT": "Conglomerate", "ADANIPORTS": "Infra",
    "BEL": "Defence", "HAL": "Defence", "IRFC": "Finance",
    "ETERNAL": "Startup", "NAUKRI": "IT",
}

# ── Config ────────────────────────────────────────────────────────────────────
MAX_POSITION_PCT  = 0.12   # 12% max per stock
MAX_SECTOR_PCT    = 0.30   # 30% max per sector
MAX_HOLDINGS      = 15     # max simultaneous CNC positions
MAX_DAILY_LOSS    = 0.02   # halt trading if daily loss > 2%
MIN_PROB_UP       = 0.52   # must have at least 52% probability

# NSE trading hours IST
_MARKET_OPEN  = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 25)   # stop 5 min before close


class RiskGuard:
    """
    Stateless risk validator — pass current portfolio state each call.
    Also supports check_order(order_dict) for simple dict-based API.
    """

    # ── Simple dict API (used by signal_publisher) ─────────────────────────

    @staticmethod
    def check_order(order: Dict, portfolio_value: float = 1_000_000) -> bool:
        """
        Lightweight check for a single order dict from signal_publisher.

        Args:
            order           : {'symbol', 'prob_up', 'target_pct', 'order_value', ...}
            portfolio_value : Total portfolio value (used for position size check)

        Returns:
            True if order passes all checks
        """
        sym      = order.get("symbol", "")
        prob_up  = float(order.get("prob_up", 0.0))
        tgt_pct  = float(order.get("target_pct", 0.0)) / 100.0  # stored as %

        if prob_up < MIN_PROB_UP:
            return False
        if tgt_pct > MAX_POSITION_PCT * 1.05:   # 5% tolerance
            return False
        if not sym:
            return False
        return True

    # ── Full validation (used by order_manager) ───────────────────────────

    @staticmethod
    def validate_order(
        symbol:           str,
        qty:              int,
        price:            float,
        portfolio_value:  float,
        current_holdings: Dict[str, int],
        current_values:   Dict[str, float],    # {symbol: current_market_value}
        max_position_pct: float = MAX_POSITION_PCT,
        max_sector_pct:   float = MAX_SECTOR_PCT,
        max_holdings:     int   = MAX_HOLDINGS,
    ) -> Tuple[bool, str]:
        """
        Full pre-trade check. Returns (is_valid, reason).

        Args:
            current_values : {symbol: market_value} of existing holdings
        """
        if portfolio_value <= 0:
            return False, "Portfolio value zero"

        order_value  = qty * price
        position_pct = order_value / portfolio_value

        # 1. Position size
        if position_pct > max_position_pct:
            return False, (f"{symbol}: position {position_pct:.1%} "
                           f"exceeds max {max_position_pct:.1%}")

        # 2. Sector concentration
        sector        = SECTOR_MAP.get(symbol, "Other")
        sector_value  = sum(v for s, v in current_values.items()
                            if SECTOR_MAP.get(s, "Other") == sector)
        sector_value += order_value
        sector_pct    = sector_value / portfolio_value
        if sector_pct > max_sector_pct:
            return False, (f"{sector} sector would be {sector_pct:.1%} "
                           f"(max {max_sector_pct:.1%})")

        # 3. Max holdings
        if len(current_holdings) >= max_holdings and symbol not in current_holdings:
            return False, f"Already at max {max_holdings} holdings"

        return True, "OK"

    @staticmethod
    def check_daily_loss(
        portfolio_value: float,
        portfolio_yesterday: float,
        limit_pct: float = MAX_DAILY_LOSS,
    ) -> Tuple[bool, str]:
        """Return (trading_allowed, reason). Call before any order batch."""
        if portfolio_yesterday <= 0:
            return True, "OK"
        daily_loss = max(0.0, (portfolio_yesterday - portfolio_value) / portfolio_yesterday)
        if daily_loss > limit_pct:
            return False, (f"Daily loss {daily_loss:.2%} exceeds circuit breaker "
                           f"{limit_pct:.2%} — trading halted")
        return True, "OK"

    @staticmethod
    def check_market_hours() -> Tuple[bool, str]:
        """Return (can_trade, reason). Prevent orders outside NSE hours."""
        now = datetime.now().time()
        if _MARKET_OPEN <= now <= _MARKET_CLOSE:
            return True, "Market open"
        return False, (f"Market closed (now={now.strftime('%H:%M')} IST, "
                       f"open={_MARKET_OPEN.strftime('%H:%M')}–"
                       f"{_MARKET_CLOSE.strftime('%H:%M')})")

    @staticmethod
    def validate_batch(
        orders: List[Dict],
        portfolio_value: float,
        current_holdings: Dict[str, int],
        current_values:   Dict[str, float],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a list of order dicts.

        Returns:
            (approved_orders, rejected_orders)
        """
        approved, rejected = [], []
        accumulated_values = dict(current_values)   # running sector exposure

        for o in orders:
            sym = o["symbol"]
            qty = o.get("qty", 0)
            px  = o.get("price", 0.0)

            ok, reason = RiskGuard.validate_order(
                symbol=sym, qty=qty, price=px,
                portfolio_value=portfolio_value,
                current_holdings=current_holdings,
                current_values=accumulated_values,
            )
            if ok:
                approved.append(o)
                accumulated_values[sym] = accumulated_values.get(sym, 0) + qty * px
            else:
                o["reject_reason"] = reason
                rejected.append(o)
                print(f"  [risk] BLOCKED {sym}: {reason}")

        return approved, rejected
