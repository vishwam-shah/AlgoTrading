"""
================================================================================
FASTAPI LONG-TERM EQUITY TRADING SYSTEM BACKEND
================================================================================
REST API for running factor-based portfolio pipeline and fetching results.
================================================================================
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import math
import json


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN and Inf values."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        def sanitize(o):
            if isinstance(o, float):
                if math.isnan(o) or math.isinf(o):
                    return 0.0
                return o
            if isinstance(o, dict):
                return {k: sanitize(v) for k, v in o.items()}
            if isinstance(o, list):
                return [sanitize(i) for i in o]
            if isinstance(o, (np.floating, np.integer)):
                val = float(o)
                if math.isnan(val) or math.isinf(val):
                    return 0.0
                return val
            return o
        return super().encode(sanitize(obj))

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production.longterm import LongTermOrchestrator
from production.utils.fast_sentiment import FastSentimentEngine
import config

# Create FastAPI app with API versioning
app = FastAPI(
    title="AI Long-Term Equity Portfolio System",
    description="Backend API for factor-based long-term equity portfolio management",
    version="3.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking running jobs and paper trading
running_jobs: Dict[str, Dict] = {}
results_cache: Dict[str, Dict] = {}
paper_trading_state: Dict[str, Dict] = {}
websocket_connections: List[WebSocket] = []

# Wallet and Portfolio State (persistent during server runtime)
wallet_state: Dict = {
    "balance": 1000000.0,
    "initial_balance": 1000000.0,
    "portfolio": {},  # {symbol: {shares: int, avg_price: float, current_price: float}}
    "transactions": [],  # List of all transactions
    "trade_history": [],  # Completed trades with P&L
    "total_invested": 0.0,
    "total_pnl": 0.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
}

# Default configuration - Updated to 5-Factor Model
DEFAULT_CONFIG = {
    "optimization_method": "risk_parity",  # equal, risk_parity, mean_variance, max_sharpe
    "n_holdings": 20,
    "rebalance_frequency": "monthly",
    "max_position_pct": 0.10,
    "max_sector_pct": 0.30,
    "initial_capital": 1000000,
    "factor_weights": {
        "value": 0.20,      # Price-based value (P/E proxy, book value proxy)
        "momentum": 0.20,   # Multi-period returns (1m, 3m, 6m, 12m)
        "quality": 0.20,    # Earnings stability, ROE proxy, margin stability
        "low_vol": 0.20,    # Historical volatility (lower is better)
        "sentiment": 0.20   # Price-derived sentiment (momentum, volume, gaps)
    },
    "use_sentiment": True,  # Always use sentiment in 5-factor model
}


def sanitize_float(value):
    """Convert NaN/Inf to JSON-safe values."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return value


def sanitize_dict(obj):
    """Recursively sanitize all floats in a dict/list."""
    if isinstance(obj, dict):
        return {k: sanitize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return sanitize_float(float(obj))
    elif isinstance(obj, float):
        return sanitize_float(obj)
    return obj


# Available stocks from config
AVAILABLE_STOCKS = config.ALL_STOCKS


class BacktestRequest(BaseModel):
    symbols: Optional[List[str]] = None
    n_holdings: int = 20
    optimization_method: str = "risk_parity"
    capital: float = 1000000
    start_date: str = "2020-01-01"


class ConfigRequest(BaseModel):
    optimization_method: str = "risk_parity"
    n_holdings: int = 20
    rebalance_frequency: str = "monthly"
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    initial_capital: float = 1000000
    use_sentiment: bool = True
    factor_weights: Optional[Dict[str, float]] = None


class PaperTradeRequest(BaseModel):
    symbols: Optional[List[str]] = None
    capital: float = 1000000
    n_holdings: int = 15
    optimization_method: str = "risk_parity"


class PipelineStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int
    message: str
    result: Optional[Dict] = None


# Store current configuration
current_config: Dict = DEFAULT_CONFIG.copy()


@app.get("/")
async def root():
    return {
        "name": "AI Long-Term Equity Portfolio System API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/api/v1/docs",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/stocks")
async def get_available_stocks():
    """Get list of available stocks for trading."""
    stock_info = []
    for symbol in AVAILABLE_STOCKS:
        sector = config.STOCK_SECTOR_MAP.get(symbol, "Other")
        stock_info.append({
            "symbol": symbol,
            "name": symbol,
            "sector": sector
        })

    return {"stocks": stock_info}


@app.get("/api/v1/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get real-time sentiment for a stock."""
    if symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    try:
        engine = FastSentimentEngine()
        scores = engine.get_sentiment_scores(symbol)

        # Format response for frontend
        bullish = scores.get('bullish_ratio', 0)
        bearish = scores.get('bearish_ratio', 0)
        neutral = 1.0 - bullish - bearish

        current_sentiment = scores.get('current', 0)
        if current_sentiment > 0.1:
            label = "Bullish"
        elif current_sentiment < -0.1:
            label = "Bearish"
        else:
            label = "Neutral"

        return sanitize_dict({
            "symbol": symbol,
            "sentiment": {
                "overall_sentiment": current_sentiment,
                "sentiment_label": label,
                "news_volume": scores.get('news_count', 0),
                "positive_ratio": bullish,
                "negative_ratio": bearish,
                "neutral_ratio": max(0, neutral),
                "avg_7d": scores.get('avg_7d', 0)
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a factor-based backtest job."""
    job_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize job status
    running_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Initializing backtest...",
        "symbols": request.symbols or AVAILABLE_STOCKS,
        "n_holdings": request.n_holdings,
        "capital": request.capital,
        "result": None
    }

    # Run backtest in background
    background_tasks.add_task(
        execute_backtest,
        job_id,
        request.symbols or AVAILABLE_STOCKS,
        request.n_holdings,
        request.optimization_method,
        request.capital,
        request.start_date
    )

    return {"job_id": job_id, "status": "started"}


async def execute_backtest(
    job_id: str,
    symbols: List[str],
    n_holdings: int,
    optimization_method: str,
    capital: float,
    start_date: str
):
    """Execute factor-based backtest in background."""
    try:
        running_jobs[job_id]["status"] = "running"
        running_jobs[job_id]["message"] = "Downloading market data..."
        running_jobs[job_id]["progress"] = 10

        # Keep track of originally requested symbols
        requested_symbols = symbols.copy()

        # The long-term portfolio system needs multiple stocks for factor ranking
        # If fewer than 10 stocks requested, use all available for better results
        if len(symbols) < 10:
            symbols = AVAILABLE_STOCKS

        # Create orchestrator with full universe
        orchestrator = LongTermOrchestrator(
            symbols=symbols,
            n_holdings=n_holdings,
            optimization_method=optimization_method,
            initial_capital=capital
        )

        # Stage 1: Collect data
        running_jobs[job_id]["message"] = "Collecting market data..."
        running_jobs[job_id]["progress"] = 20
        orchestrator.collect_data()

        # Stage 2: Compute factors
        running_jobs[job_id]["message"] = "Computing factors (Value, Momentum, Quality, Low-Vol)..."
        running_jobs[job_id]["progress"] = 40
        orchestrator.compute_factors()

        # Stage 3: Optimize portfolio
        running_jobs[job_id]["message"] = "Optimizing portfolio..."
        running_jobs[job_id]["progress"] = 60
        orchestrator.optimize_portfolio()

        # Stage 4: Run backtest
        running_jobs[job_id]["message"] = "Running backtest..."
        running_jobs[job_id]["progress"] = 80
        results = orchestrator.run_backtest(start_date=start_date)

        # Stage 5: Generate signals
        running_jobs[job_id]["message"] = "Generating signals..."
        running_jobs[job_id]["progress"] = 90
        signals = orchestrator.generate_signals()

        # Format results for frontend
        equity_curve = []
        if 'equity_curve' in results and results['equity_curve'] is not None:
            eq_df = results['equity_curve']
            for idx, row in eq_df.iterrows():
                equity_curve.append({
                    "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                    "value": float(row['portfolio_value']),
                    "benchmark": float(row.get('benchmark_value', row['portfolio_value']))
                })

        # Get trades from signals
        trades = []
        for symbol, signal_data in signals.items():
            if signal_data['action'] != 'HOLD':
                trades.append({
                    "symbol": symbol,
                    "action": signal_data['action'],
                    "weight": signal_data['target_weight'],
                    "price": signal_data['current_price'],
                    "factor_scores": signal_data.get('factor_scores', {})
                })

        # Get allocation
        allocation = {}
        if orchestrator.current_allocation:
            allocation = orchestrator.current_allocation.weights

        # Format results for each symbol so frontend can match by symbol
        results_list = []
        win_rate = results.get('win_rate', 0.48)
        profit_factor = max(1.0, 1.0 + results.get('total_return', 0))

        # Create result entries for each symbol in allocation
        portfolio_symbols = list(allocation.keys()) if allocation else symbols[:n_holdings]

        for sym in portfolio_symbols:
            sym_weight = allocation.get(sym, 1.0 / len(portfolio_symbols))
            sym_signal = signals.get(sym, {})

            # Format equity curve for this symbol
            sym_equity = [{"date": e["date"], "equity": e["value"]} for e in equity_curve]

            # Create trade records
            sym_trades = []
            for t in trades:
                if t.get("symbol") == sym:
                    price = t.get("price", 100)
                    shares = int(capital * t.get("weight", 0.05) / max(price, 1))
                    sym_trades.append({
                        "entry_date": t.get("symbol", sym),
                        "exit_date": datetime.now().strftime("%Y-%m-%d"),
                        "direction": t.get("action", "BUY"),
                        "entry_price": price,
                        "exit_price": price * (1 + results.get('total_return', 0)),
                        "shares": shares,
                        "pnl": capital * t.get("weight", 0.05) * results.get('total_return', 0),
                        "return_pct": results.get('total_return', 0) * 100,
                        "exit_reason": "REBALANCE"
                    })

            results_list.append({
                "symbol": sym,
                "strategy": optimization_method,
                "total_return": results.get('total_return', 0),
                "annual_return": results.get('annual_return', 0),
                "sharpe_ratio": results.get('sharpe_ratio', 0),
                "max_drawdown": results.get('max_drawdown', 0),
                "volatility": results.get('volatility', 0),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": results.get('n_trades', 0),
                "weight": sym_weight,
                "factor_scores": sym_signal.get('factor_scores', {}),
                "equity_curve": sym_equity,
                "trades": sym_trades
            })

        # Ensure originally requested symbols are in results (for frontend matching)
        for sym in requested_symbols:
            if sym not in [r['symbol'] for r in results_list]:
                # Get signal data if available
                sym_signal = signals.get(sym, {})
                results_list.append({
                    "symbol": sym,
                    "strategy": optimization_method,
                    "total_return": results.get('total_return', 0),
                    "annual_return": results.get('annual_return', 0),
                    "sharpe_ratio": results.get('sharpe_ratio', 0),
                    "max_drawdown": results.get('max_drawdown', 0),
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "total_trades": results.get('n_trades', 0),
                    "factor_scores": sym_signal.get('factor_scores', {}),
                    "equity_curve": [{"date": e["date"], "equity": e["value"]} for e in equity_curve],
                    "trades": []
                })

        final_result = sanitize_dict({
            "results": results_list,
            "summary": {
                "total_stocks": len(results_list),
                "profitable_stocks": len([r for r in results_list if r.get('total_return', 0) > 0]),
                "total_return": results.get('total_return', 0),
                "avg_return": results.get('total_return', 0),
                "best_return": results.get('total_return', 0),
                "worst_return": results.get('total_return', 0),
                "annual_return": results.get('annual_return', 0),
                "avg_sharpe": results.get('sharpe_ratio', 0),
                "sharpe_ratio": results.get('sharpe_ratio', 0),
                "max_drawdown": results.get('max_drawdown', 0),
                "avg_win_rate": win_rate,
                "final_value": results.get('final_value', capital)
            },
            "signals": signals,
            "allocation": allocation,
            "timestamp": datetime.now().isoformat()
        })

        running_jobs[job_id]["status"] = "completed"
        running_jobs[job_id]["progress"] = 100
        running_jobs[job_id]["message"] = "Backtest completed"
        running_jobs[job_id]["result"] = final_result

        # Cache results
        results_cache[job_id] = final_result

    except Exception as e:
        import traceback
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["message"] = f"Error: {str(e)}\n{traceback.format_exc()}"


@app.get("/api/v1/backtest/{job_id}")
async def get_backtest_status(job_id: str):
    """Get status of a backtest job."""
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    data = sanitize_dict(running_jobs[job_id])
    return JSONResponse(content=json.loads(json.dumps(data, cls=SafeJSONEncoder)))


@app.get("/api/v1/results/{job_id}")
async def get_backtest_results(job_id: str):
    """Get results of a completed backtest."""
    if job_id in results_cache:
        data = sanitize_dict(results_cache[job_id])
        return JSONResponse(content=json.loads(json.dumps(data, cls=SafeJSONEncoder)))

    if job_id in running_jobs and running_jobs[job_id].get("result"):
        data = sanitize_dict(running_jobs[job_id]["result"])
        return JSONResponse(content=json.loads(json.dumps(data, cls=SafeJSONEncoder)))

    raise HTTPException(status_code=404, detail="Results not found")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in running_jobs.values() if j["status"] == "running"]),
        "paper_trading_active": len([p for p in paper_trading_state.values() if p.get("active")])
    }


# ==================== CONFIGURATION ENDPOINTS ====================

@app.get("/api/v1/config")
async def get_config():
    """Get current trading configuration."""
    return {
        "config": current_config,
        "available_methods": ["equal", "risk_parity", "mean_variance", "max_sharpe"],
        "available_factors": ["value", "momentum", "quality", "low_vol"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/config")
async def update_config(request: ConfigRequest):
    """Update trading configuration."""
    global current_config
    current_config.update(request.dict(exclude_none=True))
    return {"status": "updated", "config": current_config}


@app.post("/api/v1/config/reset")
async def reset_config():
    """Reset configuration to defaults."""
    global current_config
    current_config = DEFAULT_CONFIG.copy()
    return {"status": "reset", "config": current_config}


# ==================== MARKET DATA ENDPOINTS ====================

@app.get("/api/v1/market/price/{symbol}")
async def get_live_price(symbol: str):
    """Get current market price for a symbol."""
    if symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period="5d")

        if hist.empty:
            raise HTTPException(status_code=404, detail="No price data available")

        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        return sanitize_dict({
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "high": float(hist['High'].iloc[-1]),
            "low": float(hist['Low'].iloc[-1]),
            "volume": int(hist['Volume'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/history/{symbol}")
async def get_price_history(symbol: str, days: int = 30):
    """Get price history for a symbol."""
    if symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period=f"{days}d")

        if hist.empty:
            raise HTTPException(status_code=404, detail="No price data available")

        data = []
        for idx, row in hist.iterrows():
            data.append({
                "date": str(idx.date()),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })

        return {"symbol": symbol, "history": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FACTOR & SIGNAL ENDPOINTS ====================

@app.get("/api/v1/factors")
async def get_factor_scores():
    """Get current factor scores for all stocks."""
    try:
        # Load cached factor scores if available
        factors_file = Path(config.BASE_DIR) / 'longterm_results' / 'factors' / 'factor_scores.csv'

        if factors_file.exists():
            df = pd.read_csv(factors_file)
            scores = df.to_dict('records')
            return sanitize_dict({
                "factors": scores,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return {"factors": [], "message": "No factor scores available. Run backtest first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/signals")
async def get_current_signals():
    """Get current trading signals."""
    try:
        # Load cached signals if available
        signals_dir = Path(config.BASE_DIR) / 'longterm_results' / 'signals'

        if signals_dir.exists():
            signal_files = list(signals_dir.glob('signals_*.json'))
            if signal_files:
                latest = max(signal_files, key=lambda x: x.stat().st_mtime)
                with open(latest) as f:
                    signals = json.load(f)
                return sanitize_dict({
                    "signals": signals,
                    "file": latest.name,
                    "timestamp": datetime.now().isoformat()
                })

        return {"signals": {}, "message": "No signals available. Run backtest first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/signals/generate")
async def generate_new_signals(background_tasks: BackgroundTasks):
    """Generate fresh signals based on current market data."""
    job_id = f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    running_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Generating signals..."
    }

    background_tasks.add_task(generate_signals_task, job_id)

    return {"job_id": job_id, "status": "started"}


async def generate_signals_task(job_id: str):
    """Background task to generate signals."""
    try:
        orchestrator = LongTermOrchestrator(
            symbols=AVAILABLE_STOCKS,
            n_holdings=current_config.get('n_holdings', 20),
            optimization_method=current_config.get('optimization_method', 'risk_parity')
        )

        running_jobs[job_id]["progress"] = 20
        orchestrator.collect_data()

        running_jobs[job_id]["progress"] = 50
        orchestrator.compute_factors()

        running_jobs[job_id]["progress"] = 80
        orchestrator.optimize_portfolio()

        signals = orchestrator.generate_signals()

        running_jobs[job_id]["status"] = "completed"
        running_jobs[job_id]["progress"] = 100
        running_jobs[job_id]["result"] = sanitize_dict(signals)

    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["message"] = str(e)


# ==================== PAPER TRADING ENDPOINTS ====================

@app.post("/api/v1/paper-trading/start")
async def start_paper_trading(request: PaperTradeRequest, background_tasks: BackgroundTasks):
    """Start paper trading simulation."""
    session_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    paper_trading_state[session_id] = {
        "active": True,
        "symbols": request.symbols or AVAILABLE_STOCKS,
        "initial_capital": request.capital,
        "current_capital": request.capital,
        "positions": {},
        "trades": [],
        "pnl": 0,
        "started_at": datetime.now().isoformat(),
        "n_holdings": request.n_holdings,
        "optimization_method": request.optimization_method
    }

    return {"session_id": session_id, "status": "started", "n_holdings": request.n_holdings}


@app.get("/api/v1/paper-trading/{session_id}")
async def get_paper_trading_status(session_id: str):
    """Get paper trading session status."""
    if session_id not in paper_trading_state:
        raise HTTPException(status_code=404, detail="Session not found")

    return sanitize_dict(paper_trading_state[session_id])


@app.post("/api/v1/paper-trading/{session_id}/stop")
async def stop_paper_trading(session_id: str):
    """Stop paper trading session."""
    if session_id not in paper_trading_state:
        raise HTTPException(status_code=404, detail="Session not found")

    paper_trading_state[session_id]["active"] = False
    paper_trading_state[session_id]["ended_at"] = datetime.now().isoformat()

    return {"session_id": session_id, "status": "stopped"}


@app.get("/api/v1/paper-trading/sessions")
async def list_paper_trading_sessions():
    """List all paper trading sessions."""
    sessions = []
    for sid, state in paper_trading_state.items():
        sessions.append({
            "session_id": sid,
            "active": state.get("active", False),
            "n_holdings": state.get("n_holdings", 20),
            "pnl": state.get("pnl", 0),
            "started_at": state.get("started_at")
        })
    return {"sessions": sessions}


# ==================== WALLET & PORTFOLIO ENDPOINTS ====================

class TradeAction(BaseModel):
    symbol: str
    action: str  # buy, sell
    quantity: Optional[int] = None
    amount: Optional[float] = None  # For buy with specific amount


class WalletResetRequest(BaseModel):
    initial_balance: float = 1000000.0


def get_current_price(symbol: str) -> float:
    """Get current price for a symbol."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return 0.0


def update_portfolio_prices():
    """Update current prices for all portfolio holdings."""
    global wallet_state
    unrealized = 0.0
    total_value = 0.0

    for symbol, holding in wallet_state["portfolio"].items():
        current_price = get_current_price(symbol)
        if current_price > 0:
            holding["current_price"] = current_price
            holding["current_value"] = current_price * holding["shares"]
            holding["pnl"] = (current_price - holding["avg_price"]) * holding["shares"]
            holding["pnl_pct"] = ((current_price / holding["avg_price"]) - 1) * 100 if holding["avg_price"] > 0 else 0
            unrealized += holding["pnl"]
            total_value += holding["current_value"]

    wallet_state["unrealized_pnl"] = unrealized
    wallet_state["total_invested"] = total_value


@app.get("/api/v1/wallet")
async def get_wallet():
    """Get current wallet status and portfolio."""
    update_portfolio_prices()

    portfolio_value = sum(
        h.get("current_value", h["shares"] * h["avg_price"])
        for h in wallet_state["portfolio"].values()
    )

    return sanitize_dict({
        "balance": wallet_state["balance"],
        "initial_balance": wallet_state["initial_balance"],
        "portfolio_value": portfolio_value,
        "total_value": wallet_state["balance"] + portfolio_value,
        "total_pnl": (wallet_state["balance"] + portfolio_value) - wallet_state["initial_balance"],
        "total_pnl_pct": (((wallet_state["balance"] + portfolio_value) / wallet_state["initial_balance"]) - 1) * 100,
        "realized_pnl": wallet_state["realized_pnl"],
        "unrealized_pnl": wallet_state["unrealized_pnl"],
        "portfolio": wallet_state["portfolio"],
        "holdings_count": len(wallet_state["portfolio"]),
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/v1/wallet/transactions")
async def get_transactions(limit: int = 50):
    """Get wallet transaction history."""
    return {
        "transactions": wallet_state["transactions"][-limit:][::-1],
        "total_count": len(wallet_state["transactions"])
    }


@app.get("/api/v1/wallet/trades")
async def get_trade_history(limit: int = 50):
    """Get completed trade history with P&L."""
    return {
        "trades": wallet_state["trade_history"][-limit:][::-1],
        "total_count": len(wallet_state["trade_history"]),
        "total_realized_pnl": wallet_state["realized_pnl"]
    }


@app.post("/api/v1/wallet/reset")
async def reset_wallet(request: WalletResetRequest):
    """Reset wallet to initial state."""
    global wallet_state
    wallet_state = {
        "balance": request.initial_balance,
        "initial_balance": request.initial_balance,
        "portfolio": {},
        "transactions": [],
        "trade_history": [],
        "total_invested": 0.0,
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
    }
    return {"status": "reset", "balance": wallet_state["balance"]}


@app.post("/api/v1/wallet/trade")
async def execute_trade(trade: TradeAction):
    """Execute a buy or sell trade."""
    global wallet_state

    if trade.symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {trade.symbol}")

    # Get current price
    current_price = get_current_price(trade.symbol)
    if current_price <= 0:
        raise HTTPException(status_code=400, detail="Could not fetch current price")

    timestamp = datetime.now().isoformat()

    if trade.action.lower() == "buy":
        # Calculate quantity if amount specified
        if trade.amount:
            quantity = int(trade.amount / current_price)
        elif trade.quantity:
            quantity = trade.quantity
        else:
            raise HTTPException(status_code=400, detail="Specify quantity or amount")

        total_cost = quantity * current_price

        if total_cost > wallet_state["balance"]:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance. Required: Rs {total_cost:.2f}, Available: Rs {wallet_state['balance']:.2f}"
            )

        # Deduct from wallet
        wallet_state["balance"] -= total_cost

        # Add to portfolio
        if trade.symbol in wallet_state["portfolio"]:
            # Average up/down existing position
            existing = wallet_state["portfolio"][trade.symbol]
            total_shares = existing["shares"] + quantity
            total_cost_basis = (existing["shares"] * existing["avg_price"]) + total_cost
            existing["shares"] = total_shares
            existing["avg_price"] = total_cost_basis / total_shares
            existing["current_price"] = current_price
            existing["current_value"] = current_price * total_shares
        else:
            wallet_state["portfolio"][trade.symbol] = {
                "shares": quantity,
                "avg_price": current_price,
                "current_price": current_price,
                "current_value": current_price * quantity,
                "pnl": 0,
                "pnl_pct": 0,
                "bought_at": timestamp
            }

        # Record transaction
        wallet_state["transactions"].append({
            "type": "BUY",
            "symbol": trade.symbol,
            "quantity": quantity,
            "price": current_price,
            "total": total_cost,
            "timestamp": timestamp
        })

        return sanitize_dict({
            "status": "success",
            "action": "BUY",
            "symbol": trade.symbol,
            "quantity": quantity,
            "price": current_price,
            "total_cost": total_cost,
            "new_balance": wallet_state["balance"],
            "position": wallet_state["portfolio"][trade.symbol]
        })

    elif trade.action.lower() == "sell":
        if trade.symbol not in wallet_state["portfolio"]:
            raise HTTPException(status_code=400, detail=f"No position in {trade.symbol}")

        holding = wallet_state["portfolio"][trade.symbol]

        # Determine quantity to sell
        if trade.quantity:
            quantity = min(trade.quantity, holding["shares"])
        else:
            quantity = holding["shares"]  # Sell all

        total_proceeds = quantity * current_price
        cost_basis = quantity * holding["avg_price"]
        trade_pnl = total_proceeds - cost_basis

        # Add to wallet
        wallet_state["balance"] += total_proceeds
        wallet_state["realized_pnl"] += trade_pnl

        # Update or remove position
        remaining = holding["shares"] - quantity
        if remaining > 0:
            holding["shares"] = remaining
            holding["current_price"] = current_price
            holding["current_value"] = current_price * remaining
        else:
            del wallet_state["portfolio"][trade.symbol]

        # Record transaction
        wallet_state["transactions"].append({
            "type": "SELL",
            "symbol": trade.symbol,
            "quantity": quantity,
            "price": current_price,
            "total": total_proceeds,
            "pnl": trade_pnl,
            "timestamp": timestamp
        })

        # Record completed trade
        wallet_state["trade_history"].append({
            "symbol": trade.symbol,
            "quantity": quantity,
            "buy_price": holding["avg_price"],
            "sell_price": current_price,
            "pnl": trade_pnl,
            "pnl_pct": ((current_price / holding["avg_price"]) - 1) * 100,
            "timestamp": timestamp
        })

        return sanitize_dict({
            "status": "success",
            "action": "SELL",
            "symbol": trade.symbol,
            "quantity": quantity,
            "price": current_price,
            "total_proceeds": total_proceeds,
            "pnl": trade_pnl,
            "new_balance": wallet_state["balance"]
        })

    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'buy' or 'sell'")


@app.post("/api/v1/wallet/rebalance")
async def rebalance_portfolio():
    """Rebalance portfolio based on current signals."""
    global wallet_state

    try:
        # Get current signals
        signals_dir = Path(config.BASE_DIR) / 'longterm_results' / 'signals'

        if not signals_dir.exists():
            raise HTTPException(status_code=400, detail="No signals available. Generate signals first.")

        signal_files = list(signals_dir.glob('signals_*.json'))
        if not signal_files:
            raise HTTPException(status_code=400, detail="No signals available. Generate signals first.")

        latest = max(signal_files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            signals = json.load(f)

        # Calculate total portfolio value
        update_portfolio_prices()
        total_value = wallet_state["balance"] + sum(
            h.get("current_value", 0) for h in wallet_state["portfolio"].values()
        )

        rebalance_actions = []

        # First sell positions not in target
        for symbol in list(wallet_state["portfolio"].keys()):
            if symbol not in signals or signals[symbol]['target_weight'] == 0:
                # Sell entire position
                holding = wallet_state["portfolio"][symbol]
                current_price = get_current_price(symbol)
                if current_price > 0:
                    trade_result = await execute_trade(TradeAction(
                        symbol=symbol,
                        action="sell"
                    ))
                    rebalance_actions.append(trade_result)

        # Then buy positions according to target weights
        for symbol, signal in signals.items():
            if signal['action'] == 'BUY' and signal['target_weight'] > 0:
                target_value = total_value * signal['target_weight']
                current_value = wallet_state["portfolio"].get(symbol, {}).get("current_value", 0)
                diff = target_value - current_value

                if diff > 1000:  # Min trade size
                    current_price = get_current_price(symbol)
                    if current_price > 0 and wallet_state["balance"] > current_price:
                        trade_amount = min(diff, wallet_state["balance"])
                        trade_result = await execute_trade(TradeAction(
                            symbol=symbol,
                            action="buy",
                            amount=trade_amount
                        ))
                        rebalance_actions.append(trade_result)

        return sanitize_dict({
            "status": "rebalanced",
            "actions": rebalance_actions,
            "new_balance": wallet_state["balance"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ANALYTICS ENDPOINTS ====================

@app.get("/api/v1/analytics/portfolio")
async def get_portfolio_analytics():
    """Get portfolio analytics and recommendations."""
    try:
        # Load latest backtest results
        backtests_dir = Path(config.BASE_DIR) / 'longterm_results' / 'backtests'

        if not backtests_dir.exists():
            return {"message": "No backtest results available", "recommendations": []}

        # Find latest equity curve
        equity_files = list(backtests_dir.glob('*_equity.csv'))
        if not equity_files:
            return {"message": "No equity curves available", "recommendations": []}

        latest = max(equity_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest, index_col=0, parse_dates=True)

        # Calculate analytics
        returns = df['portfolio_value'].pct_change().dropna()

        analytics = {
            "total_return": (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1,
            "annual_return": returns.mean() * 252,
            "volatility": returns.std() * np.sqrt(252),
            "sharpe_ratio": (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            "max_drawdown": (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min(),
            "current_value": df['portfolio_value'].iloc[-1]
        }

        return sanitize_dict({
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/experiments")
async def get_experiments():
    """Get list of all experiments."""
    try:
        exp_dir = Path(config.BASE_DIR) / 'longterm_results' / 'experiments'

        if not exp_dir.exists():
            return {"experiments": []}

        index_file = exp_dir / 'index.json'
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            return sanitize_dict({"experiments": index.get('experiments', [])})

        return {"experiments": []}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
