"""
================================================================================
AI STOCK TRADING SYSTEM - MAIN ENTRY POINT
================================================================================
Unified entry point for the production trading pipeline.

Usage:
------
    python main.py paper --symbols HDFCBANK TCS --capital 100000
    python main.py backtest --symbols HDFCBANK TCS
    python main.py pipeline --symbols HDFCBANK TCS --backtest
    python main.py optimize --symbols HDFCBANK
    python main.py data --symbols HDFCBANK TCS --fresh

For more options:
    python main.py --help
    python main.py paper --help
================================================================================
"""

import sys
from pathlib import Path

# Ensure production module is importable
sys.path.insert(0, str(Path(__file__).parent))

from production.runners.cli import main

if __name__ == "__main__":
    main()
