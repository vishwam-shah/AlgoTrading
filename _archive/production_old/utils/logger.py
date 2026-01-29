"""
================================================================================
LOGGING UTILITIES
================================================================================
Centralized logging configuration for the trading system.
================================================================================
"""

import os
import sys
from datetime import datetime
from loguru import logger


def setup_logger(
    name: str = "trading",
    log_dir: str = "logs",
    log_level: str = "INFO",
    console: bool = True,
    file: bool = True
):
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name (used in filename)
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console output
        file: Enable file output
        
    Returns:
        Configured logger instance
    """
    # Convert Path to string if needed
    if hasattr(log_dir, '__fspath__'):
        log_dir = str(log_dir)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    if console:
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level=log_level
        )
    
    # File handler
    if file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{{time:YYYY-MM-DD}}.log")
        logger.add(
            log_file,
            rotation="1 day",
            retention="30 days",
            level="DEBUG"
        )
    
    return logger


def get_logger():
    """Get the configured logger instance."""
    return logger


class TradingLogger:
    """
    Specialized logger for trading operations.
    Logs trades, signals, and performance metrics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.trade_log_path = os.path.join(log_dir, "trades", "trade_log.csv")
        self.signal_log_path = os.path.join(log_dir, "signals", "signal_log.csv")
        
        os.makedirs(os.path.dirname(self.trade_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.signal_log_path), exist_ok=True)
    
    def log_trade(self, trade_data: dict):
        """Log a trade execution."""
        import pandas as pd
        
        trade_data['timestamp'] = datetime.now().isoformat()
        
        df = pd.DataFrame([trade_data])
        
        if os.path.exists(self.trade_log_path):
            df.to_csv(self.trade_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.trade_log_path, index=False)
        
        logger.info(f"Trade logged: {trade_data.get('symbol', 'N/A')} {trade_data.get('side', 'N/A')}")
    
    def log_signal(self, signal_data: dict):
        """Log a trading signal."""
        import pandas as pd
        
        signal_data['timestamp'] = datetime.now().isoformat()
        
        df = pd.DataFrame([signal_data])
        
        if os.path.exists(self.signal_log_path):
            df.to_csv(self.signal_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.signal_log_path, index=False)
    
    def log_performance(self, metrics: dict, period: str = "daily"):
        """Log performance metrics."""
        import pandas as pd
        
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['period'] = period
        
        perf_path = os.path.join(self.log_dir, f"performance_{period}.csv")
        
        df = pd.DataFrame([metrics])
        
        if os.path.exists(perf_path):
            df.to_csv(perf_path, mode='a', header=False, index=False)
        else:
            df.to_csv(perf_path, index=False)
