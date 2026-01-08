"""
================================================================================
TRADING CALENDAR UTILITY
================================================================================
Handles Indian stock market trading days (NSE).
Excludes weekends and public holidays.

Usage:
    from pipeline.utils.trading_calendar import is_trading_day, get_last_trading_day
    
    if is_trading_day('2025-12-08'):
        # Run data collection
    
    last_trading = get_last_trading_day()
================================================================================
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# NSE holidays 2024-2025 (update annually)
NSE_HOLIDAYS_2024 = [
    '2024-01-26',  # Republic Day
    '2024-03-08',  # Mahashivratri
    '2024-03-25',  # Holi
    '2024-03-29',  # Good Friday
    '2024-04-11',  # Id-Ul-Fitr
    '2024-04-17',  # Shri Ram Navami
    '2024-04-21',  # Mahavir Jayanti
    '2024-05-01',  # Maharashtra Day
    '2024-05-23',  # Buddha Purnima
    '2024-06-17',  # Bakri Id
    '2024-07-17',  # Moharram
    '2024-08-15',  # Independence Day
    '2024-08-26',  # Janmashtami
    '2024-10-02',  # Mahatma Gandhi Jayanti
    '2024-10-12',  # Dussehra
    '2024-11-01',  # Diwali (Laxmi Pujan)
    '2024-11-02',  # Diwali (Balipratipada)
    '2024-11-15',  # Gurunanak Jayanti
    '2024-12-25',  # Christmas
]

NSE_HOLIDAYS_2025 = [
    '2025-01-26',  # Republic Day
    '2025-02-26',  # Mahashivratri
    '2025-03-14',  # Holi
    '2025-03-31',  # Id-Ul-Fitr
    '2025-04-10',  # Mahavir Jayanti
    '2025-04-14',  # Dr. Ambedkar Jayanti
    '2025-04-18',  # Good Friday
    '2025-05-01',  # Maharashtra Day
    '2025-06-07',  # Bakri Id
    '2025-08-15',  # Independence Day
    '2025-08-27',  # Janmashtami
    '2025-10-02',  # Mahatma Gandhi Jayanti
    '2025-10-21',  # Diwali (Laxmi Pujan)
    '2025-11-05',  # Gurunanak Jayanti
    '2025-12-25',  # Christmas
]

# Combine all holidays
NSE_HOLIDAYS = set(NSE_HOLIDAYS_2024 + NSE_HOLIDAYS_2025)


def is_trading_day(date: str | datetime, include_today: bool = True) -> bool:
    """
    Check if a date is a trading day (not weekend or holiday).
    
    Args:
        date: Date string 'YYYY-MM-DD' or datetime object
        include_today: If False, returns False for today (market not closed yet)
        
    Returns:
        True if trading day, False otherwise
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Check if today but market not closed
    if not include_today:
        today = pd.Timestamp.now().normalize()
        if date.normalize() >= today:
            return False
    
    # Check weekend
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check holiday
    date_str = date.strftime('%Y-%m-%d')
    if date_str in NSE_HOLIDAYS:
        return False
    
    return True


def get_last_trading_day(reference_date: Optional[str | datetime] = None) -> datetime:
    """
    Get the last completed trading day.
    Skips weekends and holidays.
    
    Args:
        reference_date: Start searching from this date (default: today)
        
    Returns:
        Last trading day as datetime
    """
    if reference_date is None:
        date = pd.Timestamp.now()
    elif isinstance(reference_date, str):
        date = pd.to_datetime(reference_date)
    else:
        date = reference_date
    
    # Start from yesterday (today's market not closed yet)
    date = date - timedelta(days=1)
    
    # Go back until we find a trading day
    max_lookback = 10  # Prevent infinite loop
    for _ in range(max_lookback):
        if is_trading_day(date, include_today=True):
            return date
        date = date - timedelta(days=1)
    
    raise ValueError(f"No trading day found in last {max_lookback} days")


def get_next_trading_day(reference_date: Optional[str | datetime] = None) -> datetime:
    """
    Get the next trading day.
    Skips weekends and holidays.
    
    Args:
        reference_date: Start searching from this date (default: today)
        
    Returns:
        Next trading day as datetime
    """
    if reference_date is None:
        date = pd.Timestamp.now()
    elif isinstance(reference_date, str):
        date = pd.to_datetime(reference_date)
    else:
        date = reference_date
    
    # Start from tomorrow
    date = date + timedelta(days=1)
    
    # Go forward until we find a trading day
    max_lookforward = 10
    for _ in range(max_lookforward):
        if is_trading_day(date, include_today=True):
            return date
        date = date + timedelta(days=1)
    
    raise ValueError(f"No trading day found in next {max_lookforward} days")


def get_trading_days_between(start_date: str | datetime, 
                             end_date: str | datetime) -> list:
    """
    Get all trading days between two dates (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of trading days as datetime objects
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    trading_days = []
    date = start_date
    
    while date <= end_date:
        if is_trading_day(date, include_today=True):
            trading_days.append(date)
        date = date + timedelta(days=1)
    
    return trading_days


def should_run_pipeline(date: Optional[str | datetime] = None) -> dict:
    """
    Determine if pipeline should run based on trading calendar.
    
    Args:
        date: Date to check (default: today)
        
    Returns:
        dict with 'should_run', 'reason', 'last_trading_day', 'next_trading_day'
    """
    if date is None:
        date = pd.Timestamp.now()
    elif isinstance(date, str):
        date = pd.to_datetime(date)
    
    result = {
        'should_run': False,
        'reason': '',
        'date': date.strftime('%Y-%m-%d'),
        'last_trading_day': None,
        'next_trading_day': None,
    }
    
    try:
        last_trading = get_last_trading_day(date)
        result['last_trading_day'] = last_trading.strftime('%Y-%m-%d')
        
        next_trading = get_next_trading_day(date)
        result['next_trading_day'] = next_trading.strftime('%Y-%m-%d')
        
        # Weekend
        if date.weekday() >= 5:
            result['reason'] = f"{date.strftime('%A')} - Market closed"
            return result
        
        # Holiday
        date_str = date.strftime('%Y-%m-%d')
        if date_str in NSE_HOLIDAYS:
            result['reason'] = f"Market holiday"
            return result
        
        # Trading day - should run
        result['should_run'] = True
        result['reason'] = f"Trading day - predict {result['next_trading_day']}"
        
    except Exception as e:
        result['reason'] = f"Error: {str(e)}"
    
    return result


if __name__ == '__main__':
    # Test the calendar
    import sys
    
    test_date = sys.argv[1] if len(sys.argv) > 1 else None
    
    info = should_run_pipeline(test_date)
    
    print(f"Date: {info['date']}")
    print(f"Should Run: {info['should_run']}")
    print(f"Reason: {info['reason']}")
    print(f"Last Trading Day: {info['last_trading_day']}")
    print(f"Next Trading Day: {info['next_trading_day']}")
    
    # Test week
    print("\nNext 7 days:")
    today = pd.Timestamp.now()
    for i in range(7):
        date = today + timedelta(days=i)
        is_trading = is_trading_day(date)
        day_type = "✅ Trading" if is_trading else "❌ Closed"
        print(f"{date.strftime('%Y-%m-%d %A')}: {day_type}")
