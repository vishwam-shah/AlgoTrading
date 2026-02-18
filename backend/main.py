"""
================================================================================
FASTAPI TRADING SYSTEM BACKEND
================================================================================
REST API for running trading pipeline and fetching results.
================================================================================
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import Path
from pathlib import Path as FilePath  # Only use this alias for file paths, not FastAPI params


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

from engine.orchestrator import UnifiedOrchestrator
from engine.sentiment import FastSentimentEngine

# Create FastAPI app with API versioning
app = FastAPI(
    title="AI Stock Trading System",
    description="Backend API for AI-powered stock trading system with deep learning models",
    version="2.0.0",
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
v3_jobs: Dict[str, Dict] = {}
websocket_connections: List[WebSocket] = []

# Wallet and Portfolio State (persistent during server runtime)
wallet_state: Dict = {
    "balance": 100000.0,
    "initial_balance": 100000.0,
    "portfolio": {},  # {symbol: {shares: int, avg_price: float, current_price: float}}
    "transactions": [],  # List of all transactions
    "trade_history": [],  # Completed trades with P&L
    "total_invested": 0.0,
    "total_pnl": 0.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
}

# Default configuration
DEFAULT_CONFIG = {
    "model_type": "xgboost",  # xgboost, lstm, transformer, ensemble
    "lookback_days": 500,
    "train_test_split": 0.8,
    "min_confidence": 0.55,
    "max_position_pct": 0.15,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.05,
    "use_sentiment": True,
    "use_technical": True,
    "use_volume": True,
    "rebalance_frequency": "daily",  # daily, weekly, monthly
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


# Available stocks
AVAILABLE_STOCKS = [
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    "RELIANCE", "TATAMOTORS", "TATASTEEL", "ITC", "LT",
    "BHARTIARTL", "HINDUNILVR", "MARUTI", "BAJFINANCE",
    "ADANIENT", "ADANIPORTS", "ASIANPAINT", "SUNPHARMA"
]


class BacktestRequest(BaseModel):
    symbols: List[str]
    capital: float = 100000
    days: int = 1000
    config: Optional[Dict] = None


class ConfigRequest(BaseModel):
    model_type: str = "xgboost"
    lookback_days: int = 500
    train_test_split: float = 0.8
    min_confidence: float = 0.55
    max_position_pct: float = 0.15
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05
    use_sentiment: bool = True
    use_technical: bool = True
    use_volume: bool = True


class PaperTradeRequest(BaseModel):
    symbols: List[str]
    capital: float = 100000
    action: str = "start"  # start, stop, status


class PipelineStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int
    message: str
    result: Optional[Dict] = None


class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: str


class PipelineRunRequest(BaseModel):
    symbols: Optional[List[str]] = None
    sectors: Optional[List[str]] = None  # Select stocks by sector
    capital: float = 100000
    optimization_method: str = "risk_parity"
    n_holdings: int = 15
    start_date: str = "2022-01-01"
    force_download: bool = True
    models_to_train: Optional[List[str]] = None


class RebalanceRequest(BaseModel):
    target_weights: Optional[Dict[str, float]] = None


# Store current configuration
current_config: Dict = DEFAULT_CONFIG.copy()

# Pipeline orchestrators keyed by job_id
pipeline_orchestrators: Dict[str, UnifiedOrchestrator] = {}


@app.get("/")
async def root():
    return {
        "name": "AI Stock Trading System API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/v1/docs",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/stocks")
async def get_available_stocks():
    """Get list of available stocks for trading."""
    stock_info = {
        "SBIN": {"name": "State Bank of India", "sector": "Banking"},
        "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
        "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
        "AXISBANK": {"name": "Axis Bank", "sector": "Banking"},
        "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
        "TCS": {"name": "Tata Consultancy Services", "sector": "IT"},
        "INFY": {"name": "Infosys", "sector": "IT"},
        "WIPRO": {"name": "Wipro", "sector": "IT"},
        "HCLTECH": {"name": "HCL Technologies", "sector": "IT"},
        "TECHM": {"name": "Tech Mahindra", "sector": "IT"},
        "RELIANCE": {"name": "Reliance Industries", "sector": "Energy"},
        "TATAMOTORS": {"name": "Tata Motors", "sector": "Auto"},
        "TATASTEEL": {"name": "Tata Steel", "sector": "Metals"},
        "ITC": {"name": "ITC Limited", "sector": "FMCG"},
        "LT": {"name": "Larsen & Toubro", "sector": "Infrastructure"},
        "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecom"},
        "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "FMCG"},
        "MARUTI": {"name": "Maruti Suzuki", "sector": "Auto"},
        "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Finance"},
        "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate"},
        "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure"},
        "ASIANPAINT": {"name": "Asian Paints", "sector": "Paints"},
        "SUNPHARMA": {"name": "Sun Pharma", "sector": "Pharma"},
    }
    
    return {
        "stocks": [
            {"symbol": s, **stock_info.get(s, {"name": s, "sector": "Other"})}
            for s in AVAILABLE_STOCKS
        ]
    }


@app.get("/api/v1/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get real-time sentiment for a stock."""
    try:
        if symbol not in AVAILABLE_STOCKS:
            return JSONResponse(status_code=404, content={"error": f"Stock {symbol} not found"})
        engine = FastSentimentEngine()
        scores = engine.get_sentiment_scores(symbol)
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
        return JSONResponse(content=sanitize_dict({
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
        }))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v1/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a backtest job for selected stocks."""
    job_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in AVAILABLE_STOCKS]
    if invalid_symbols:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid symbols: {invalid_symbols}"
        )
    
    # Initialize job status
    running_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Initializing backtest...",
        "symbols": request.symbols,
        "capital": request.capital,
        "result": None
    }
    
    # Run backtest in background
    background_tasks.add_task(
        execute_backtest, 
        job_id, 
        request.symbols, 
        request.capital,
        request.days
    )
    
    return {"job_id": job_id, "status": "started"}


async def execute_backtest(job_id: str, symbols: List[str], capital: float, days: int):
    """Execute backtest in background using UnifiedOrchestrator pipeline."""
    import logging
    try:
        # Log state before pipeline run
        logging.basicConfig(level=logging.INFO)
        logging.info(f"[Pipeline Start] job_id={job_id} running_jobs={running_jobs.get(job_id)} results_cache={results_cache.get(job_id)} pipeline_orchestrators={pipeline_orchestrators.get(job_id)}")

        running_jobs[job_id]["status"] = "running"
        running_jobs[job_id]["message"] = "Running pipeline..."

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        def progress_cb(step_status):
            running_jobs[job_id]["message"] = f"Step {step_status.step_number}/8: {step_status.name} ({step_status.status})"
            running_jobs[job_id]["progress"] = int((step_status.step_number / 8) * 100)

        orchestrator = UnifiedOrchestrator(
            symbols=symbols,
            initial_capital=capital,
            paper_trading=True,
            progress_callback=progress_cb
        )

        pipeline_result = orchestrator.run_pipeline(
            start_date=start_date,
            force_download=True
        )

        # Store orchestrator for later queries
        pipeline_orchestrators[job_id] = orchestrator

        # Format results from backtest step
        backtest_data = orchestrator.get_backtest_results()
        all_results = []
        for symbol, r in backtest_data.items():
            if 'error' in r:
                all_results.append({"symbol": symbol, "error": r["error"],
                                    "total_return": 0, "sharpe_ratio": 0,
                                    "max_drawdown": 0, "win_rate": 0, "total_trades": 0})
            else:
                all_results.append(sanitize_dict({
                    "symbol": symbol,
                    "total_return": r.get("total_return", 0),
                    "sharpe_ratio": r.get("sharpe_ratio", 0),
                    "max_drawdown": r.get("max_drawdown", 0),
                    "win_rate": r.get("win_rate", 0),
                    "total_trades": r.get("total_trades", 0),
                    "profit_factor": r.get("profit_factor", 0),
                    "equity_curve": r.get("equity_curve", []),
                    "trades": r.get("trades", [])
                }))

        running_jobs[job_id]["status"] = "completed"
        running_jobs[job_id]["progress"] = 100
        running_jobs[job_id]["message"] = "Pipeline completed"
        running_jobs[job_id]["result"] = {
            "results": all_results,
            "summary": compute_summary(all_results),
            "pipeline_status": orchestrator.get_status(),
            "signals": orchestrator.get_signals(),
            "allocation": orchestrator.get_allocation(),
            "timestamp": datetime.now().isoformat()
        }

        # Cache results
        results_cache[job_id] = running_jobs[job_id]["result"]

        # Log state after pipeline run
        logging.info(f"[Pipeline End] job_id={job_id} running_jobs={running_jobs.get(job_id)} results_cache={results_cache.get(job_id)} pipeline_orchestrators={pipeline_orchestrators.get(job_id)}")

    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["message"] = str(e)
        logging.error(f"[Pipeline Error] job_id={job_id} error={e}")


def compute_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics from results."""
    if not results:
        return {}
    
    returns = [r.get("total_return", 0) for r in results if "error" not in r]
    
    return sanitize_dict({
        "total_stocks": len(results),
        "profitable_stocks": sum(1 for r in returns if r > 0),
        "avg_return": np.mean(returns) if returns else 0,
        "best_return": max(returns) if returns else 0,
        "worst_return": min(returns) if returns else 0,
        "avg_sharpe": np.mean([r.get("sharpe_ratio", 0) for r in results if "error" not in r]),
        "avg_win_rate": np.mean([r.get("win_rate", 0) for r in results if "error" not in r])
    })


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


@app.get("/api/v1/plot/{symbol}")
async def get_equity_plot(symbol: str):
    """Get equity curve plot for a symbol."""
    plot_path = Path(f"production_results/backtest/{symbol}_equity.png")
    
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path, media_type="image/png")


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
        "available_models": ["xgboost", "lstm", "transformer", "ensemble"],
        "available_features": ["technical", "sentiment", "volume", "volatility"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/config")
async def update_config(request: ConfigRequest):
    """Update trading configuration."""
    global current_config
    current_config.update(request.dict())
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


# ==================== PAPER TRADING ENDPOINTS ====================

@app.post("/api/v1/paper-trading/start")
async def start_paper_trading(request: PaperTradeRequest, background_tasks: BackgroundTasks):
    """Start paper trading simulation."""
    session_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    paper_trading_state[session_id] = {
        "active": True,
        "symbols": request.symbols,
        "initial_capital": request.capital,
        "current_capital": request.capital,
        "positions": {},
        "trades": [],
        "pnl": 0,
        "started_at": datetime.now().isoformat()
    }
    
    return {"session_id": session_id, "status": "started", "symbols": request.symbols}


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
            "symbols": state.get("symbols", []),
            "pnl": state.get("pnl", 0),
            "started_at": state.get("started_at")
        })
    return {"sessions": sessions}


# ==================== ANALYTICS ENDPOINTS ====================

@app.get("/api/v1/analytics/portfolio")
async def get_portfolio_analytics():
    """Get portfolio analytics and recommendations."""
    # Aggregate results from all backtests
    all_results = []
    for job_id, result in results_cache.items():
        if "results" in result:
            all_results.extend(result["results"])
    
    if not all_results:
        return {"message": "No backtest results available", "recommendations": []}
    
    # Calculate analytics
    by_symbol = {}
    for r in all_results:
        symbol = r.get("symbol", "Unknown")
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(r)
    
    recommendations = []
    for symbol, results in by_symbol.items():
        avg_return = np.mean([r.get("total_return", 0) for r in results])
        avg_sharpe = np.mean([r.get("sharpe_ratio", 0) for r in results])
        avg_win_rate = np.mean([r.get("win_rate", 0) for r in results])
        
        score = (avg_return * 0.4) + (avg_sharpe * 0.3) + (avg_win_rate * 0.3)
        
        recommendations.append({
            "symbol": symbol,
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
            "score": score,
            "recommendation": "BUY" if score > 0.5 else "HOLD" if score > 0 else "AVOID"
        })
    
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return sanitize_dict({
        "total_symbols_analyzed": len(by_symbol),
        "recommendations": recommendations[:10],
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/v1/analytics/models")
async def get_model_performance():
    """Get model performance comparison."""
    models_dir = Path("models")
    
    available_models = {
        "xgboost": {"path": "fast", "status": "available", "description": "Fast gradient boosting"},
        "lstm": {"path": "lstm", "status": "available", "description": "Long short-term memory neural network"},
        "transformer": {"path": "transformer", "status": "available", "description": "Attention-based model"},
        "gru": {"path": "gru", "status": "available", "description": "Gated recurrent unit"},
        "ensemble": {"path": "ensemble", "status": "available", "description": "Combined model ensemble"}
    }
    
    for name, info in available_models.items():
        model_path = models_dir / info["path"]
        if model_path.exists():
            files = list(model_path.glob("*"))
            info["trained_symbols"] = len([f for f in files if f.suffix in [".pkl", ".keras", ".h5"]])
        else:
            info["status"] = "not_found"
            info["trained_symbols"] = 0
    
    return {"models": available_models, "timestamp": datetime.now().isoformat()}


# ==================== UNIFIED PIPELINE ENDPOINTS ====================

@app.post("/api/v1/pipeline/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Start full 8-step pipeline (background execution)."""
    import config as cfg

    job_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine symbols from request
    symbols = request.symbols or []
    if request.sectors:
        # Add stocks from selected sectors
        for sector in request.sectors:
            sector_stocks = [s for s, sec in cfg.STOCK_SECTOR_MAP.items() if sec == sector]
            symbols.extend(sector_stocks)
        symbols = list(set(symbols))  # Dedupe

    if not symbols:
        symbols = cfg.ALL_STOCKS[:10]  # Default to first 10

    # Validate symbols
    invalid = [s for s in symbols if s not in cfg.ALL_STOCKS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid symbols: {invalid}")

    running_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Initializing pipeline...",
        "symbols": symbols,
        "capital": request.capital,
        "config": {
            "optimization_method": request.optimization_method,
            "n_holdings": request.n_holdings,
            "start_date": request.start_date
        },
        "steps": [],
        "result": None
    }

    background_tasks.add_task(
        execute_pipeline,
        job_id,
        symbols,
        request.capital,
        request.optimization_method,
        request.n_holdings,
        request.start_date,
        request.force_download,
        request.models_to_train
    )

    return {"job_id": job_id, "status": "started", "symbols": symbols}


async def execute_pipeline(
    job_id: str,
    symbols: List[str],
    capital: float,
    optimization_method: str,
    n_holdings: int,
    start_date: str,
    force_download: bool,
    models_to_train: Optional[List[str]]
):
    """Execute full pipeline in background."""
    try:
        running_jobs[job_id]["status"] = "running"

        def progress_cb(step_status):
            step_info = {
                "step": step_status.step_number,
                "name": step_status.name,
                "status": step_status.status,
                "duration": step_status.duration_seconds,
                "details": step_status.details
            }
            # Update steps list
            steps = running_jobs[job_id].get("steps", [])
            # Replace or append step
            existing = [i for i, s in enumerate(steps) if s["step"] == step_status.step_number]
            if existing:
                steps[existing[0]] = step_info
            else:
                steps.append(step_info)
            running_jobs[job_id]["steps"] = steps
            running_jobs[job_id]["message"] = f"Step {step_status.step_number}/8: {step_status.name}"
            running_jobs[job_id]["progress"] = int((step_status.step_number / 8) * 100)

        orchestrator = UnifiedOrchestrator(
            symbols=symbols,
            initial_capital=capital,
            paper_trading=True,
            progress_callback=progress_cb
        )

        pipeline_result = orchestrator.run_pipeline(
            optimization_method=optimization_method,
            n_holdings=n_holdings,
            start_date=start_date,
            force_download=force_download,
            models_to_train=models_to_train
        )

        # Store orchestrator
        pipeline_orchestrators[job_id] = orchestrator

        running_jobs[job_id]["status"] = pipeline_result.status
        running_jobs[job_id]["progress"] = 100
        running_jobs[job_id]["message"] = f"Pipeline {pipeline_result.status}"
        running_jobs[job_id]["result"] = sanitize_dict({
            "backtest_results": orchestrator.get_backtest_results(),
            "signals": orchestrator.get_signals(),
            "allocation": orchestrator.get_allocation(),
            "pipeline_status": orchestrator.get_status(),
            "timestamp": datetime.now().isoformat()
        })

        results_cache[job_id] = running_jobs[job_id]["result"]

    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["message"] = str(e)


@app.get("/api/v1/pipeline/{job_id}/status")
async def get_pipeline_status(job_id: str):
    """Get per-step progress with details for a pipeline job."""
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = running_jobs[job_id]

    # If orchestrator exists, get live status
    if job_id in pipeline_orchestrators:
        orchestrator = pipeline_orchestrators[job_id]
        status = orchestrator.get_status()
        return sanitize_dict({
            "job_id": job_id,
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "symbols": job.get("symbols", []),
            "config": job.get("config", {}),
            "steps": status.get("steps", []),
            "current_step": status.get("current_step", 0),
            "total_steps": status.get("total_steps", 8),
            "timestamp": datetime.now().isoformat(),
            "result": job.get("result")
        })

    return sanitize_dict({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "symbols": job.get("symbols", []),
        "config": job.get("config", {}),
        "steps": job.get("steps", []),
        "timestamp": datetime.now().isoformat(),
        "result": job.get("result")
    })


@app.get("/api/v1/stock/{symbol}/analysis")
async def get_stock_analysis(symbol: str):
    """Get full analysis for a stock: factors + ML + backtest + sentiment."""
    import config as cfg

    if symbol not in cfg.ALL_STOCKS and symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    analysis = {
        "symbol": symbol,
        "sector": cfg.STOCK_SECTOR_MAP.get(symbol, "Other"),
        "timestamp": datetime.now().isoformat()
    }

    # Get sentiment
    try:
        engine = FastSentimentEngine()
        scores = engine.get_sentiment_scores(symbol)
        analysis["sentiment"] = sanitize_dict({
            "score": scores.get('current', 0),
            "avg_7d": scores.get('avg_7d', 0),
            "bullish_ratio": scores.get('bullish_ratio', 0),
            "bearish_ratio": scores.get('bearish_ratio', 0),
            "news_count": scores.get('news_count', 0)
        })
    except Exception as e:
        analysis["sentiment"] = {"error": str(e)}

    # Get from any completed pipeline that has this symbol
    for job_id, orchestrator in pipeline_orchestrators.items():
        if symbol in orchestrator.features_cache:
            # Factor scores
            for fs in orchestrator.factor_scores:
                if fs.symbol == symbol:
                    analysis["factors"] = sanitize_dict({
                        "value": fs.value_score,
                        "momentum": fs.momentum_score,
                        "quality": fs.quality_score,
                        "low_vol": fs.low_vol_score,
                        "sentiment": fs.sentiment_score,
                        "combined": fs.combined_score
                    })
                    break

            # Backtest results
            if symbol in orchestrator.backtest_results:
                analysis["backtest"] = sanitize_dict(orchestrator.backtest_results[symbol])

            # Signals
            if orchestrator.signals and symbol in orchestrator.signals:
                analysis["signal"] = sanitize_dict(orchestrator.signals[symbol])

            break

    return analysis


@app.get("/api/v1/portfolio/current")
async def get_current_portfolio():
    """Get current allocation, sector weights, and signals from latest pipeline."""
    import config as cfg

    # Find most recent completed pipeline
    latest_orchestrator = None
    latest_job_id = None
    for job_id in sorted(pipeline_orchestrators.keys(), reverse=True):
        orch = pipeline_orchestrators[job_id]
        if orch.pipeline_status and orch.pipeline_status.status == 'completed':
            latest_orchestrator = orch
            latest_job_id = job_id
            break

    if not latest_orchestrator:
        return {
            "message": "No completed pipeline found. Run a pipeline first.",
            "allocation": {},
            "signals": {},
            "timestamp": datetime.now().isoformat()
        }

    allocation = latest_orchestrator.get_allocation()
    signals = latest_orchestrator.get_signals()

    # Compute sector weights
    sector_weights = {}
    if allocation and 'weights' in allocation:
        for symbol, weight in allocation['weights'].items():
            sector = cfg.STOCK_SECTOR_MAP.get(symbol, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

    return sanitize_dict({
        "job_id": latest_job_id,
        "allocation": allocation,
        "sector_weights": sector_weights,
        "signals": signals,
        "holdings_count": len(allocation.get('weights', {})) if allocation else 0,
        "timestamp": datetime.now().isoformat()
    })


@app.post("/api/v1/portfolio/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """Auto-execute trades to match target weights from pipeline allocation."""
    global wallet_state
    import config as cfg

    # Get target weights
    target_weights = request.target_weights
    if not target_weights:
        # Use latest pipeline allocation
        for job_id in sorted(pipeline_orchestrators.keys(), reverse=True):
            orch = pipeline_orchestrators[job_id]
            allocation = orch.get_allocation()
            if allocation and 'weights' in allocation:
                target_weights = allocation['weights']
                break

    if not target_weights:
        raise HTTPException(status_code=400, detail="No target weights provided and no pipeline allocation found")

    # Calculate current portfolio value
    update_portfolio_prices()
    total_value = wallet_state["balance"] + sum(
        h.get("current_value", h["shares"] * h["avg_price"])
        for h in wallet_state["portfolio"].values()
    )

    trades_executed = []

    # Calculate target positions
    for symbol, target_weight in target_weights.items():
        target_value = total_value * target_weight
        current_value = 0
        current_shares = 0

        if symbol in wallet_state["portfolio"]:
            holding = wallet_state["portfolio"][symbol]
            current_shares = holding["shares"]
            current_value = holding.get("current_value", current_shares * holding["avg_price"])

        diff_value = target_value - current_value

        # Get current price
        price = get_current_price(symbol)
        if price <= 0:
            continue

        shares_diff = int(diff_value / price)

        if shares_diff > 0:
            # Buy
            cost = shares_diff * price
            if cost <= wallet_state["balance"]:
                try:
                    result = await execute_trade(TradeAction(
                        symbol=symbol, action="buy", quantity=shares_diff
                    ))
                    trades_executed.append({"action": "BUY", "symbol": symbol, "shares": shares_diff, "result": result})
                except Exception as e:
                    trades_executed.append({"action": "BUY", "symbol": symbol, "shares": shares_diff, "error": str(e)})
        elif shares_diff < 0:
            # Sell
            shares_to_sell = min(abs(shares_diff), current_shares)
            if shares_to_sell > 0:
                try:
                    result = await execute_trade(TradeAction(
                        symbol=symbol, action="sell", quantity=shares_to_sell
                    ))
                    trades_executed.append({"action": "SELL", "symbol": symbol, "shares": shares_to_sell, "result": result})
                except Exception as e:
                    trades_executed.append({"action": "SELL", "symbol": symbol, "shares": shares_to_sell, "error": str(e)})

    return sanitize_dict({
        "status": "completed",
        "trades_executed": len(trades_executed),
        "trades": trades_executed,
        "new_balance": wallet_state["balance"],
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/v1/models/comparison")
async def get_models_comparison():
    """Compare all model metrics side by side from pipeline results."""
    comparisons = []

    for job_id, orchestrator in pipeline_orchestrators.items():
        if orchestrator.pipeline_status and orchestrator.pipeline_status.status == 'completed':
            # Get ML training step details
            for step in orchestrator.pipeline_status.steps:
                if step.name == 'ML Model Training' and step.details:
                    comparisons.append({
                        "job_id": job_id,
                        "symbols_count": len(orchestrator.symbols),
                        "samples_trained": step.details.get('samples_trained', 0),
                        "features_used": step.details.get('features_used', 0),
                        "metrics": step.details.get('metrics', {})
                    })

    # Also aggregate backtest metrics across jobs
    backtest_summary = {}
    for job_id, orchestrator in pipeline_orchestrators.items():
        results = orchestrator.get_backtest_results()
        for symbol, r in results.items():
            if 'error' not in r:
                if symbol not in backtest_summary:
                    backtest_summary[symbol] = []
                backtest_summary[symbol].append({
                    "job_id": job_id,
                    "return": r.get('total_return', 0),
                    "sharpe": r.get('sharpe_ratio', 0),
                    "win_rate": r.get('win_rate', 0),
                    "trades": r.get('total_trades', 0)
                })

    return sanitize_dict({
        "model_training": comparisons,
        "backtest_by_symbol": backtest_summary,
        "total_pipelines": len(comparisons),
        "timestamp": datetime.now().isoformat()
    })


# ==================== V3 PIPELINE ENDPOINTS ====================

class V3RunRequest(BaseModel):
    symbols: List[str] = []
    capital: float = 100000


@app.post("/api/v1/v3/run")
async def run_v3(request: V3RunRequest, background_tasks: BackgroundTasks):
    """Start V3 regression pipeline (background execution)."""
    job_id = f"v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    symbols = request.symbols
    if not symbols:
        symbols = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'TCS', 'INFY']

    v3_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Initializing V3 pipeline...",
        "symbols": symbols,
        "steps": [],
        "current_step": 0,
        "total_steps": 4,
        "result": None,
    }

    background_tasks.add_task(execute_v3, job_id, symbols)
    return {"job_id": job_id, "status": "started", "symbols": symbols}


def execute_v3(job_id: str, symbols: List[str]):
    """Execute V3 pipeline in background."""
    import logging
    try:
        v3_jobs[job_id]["status"] = "running"

        # V3 step names
        v3_step_names = {
            1: "Data Collection",
            2: "Feature Engineering",
            3: "Model Training",
            4: "Evaluation",
        }

        def progress_cb(step: int, total: int, message: str):
            v3_jobs[job_id]["current_step"] = step
            v3_jobs[job_id]["total_steps"] = total
            v3_jobs[job_id]["progress"] = int((step / total) * 100)
            v3_jobs[job_id]["message"] = message

            # Update steps list
            step_info = {
                "step": step,
                "name": v3_step_names.get(step, f"Step {step}"),
                "status": "running",
                "duration": 0,
                "details": message,
            }
            steps = v3_jobs[job_id].get("steps", [])
            # Mark previous steps completed
            for s in steps:
                if s["step"] < step:
                    s["status"] = "completed"
            existing = [i for i, s in enumerate(steps) if s["step"] == step]
            if existing:
                steps[existing[0]] = step_info
            else:
                steps.append(step_info)
            v3_jobs[job_id]["steps"] = steps

        # Import and run V3 pipeline
        from V3.pipeline import run_pipeline_api

        result = run_pipeline_api(
            symbols=symbols,
            progress_callback=progress_cb,
        )

        # Mark all steps completed
        for s in v3_jobs[job_id].get("steps", []):
            s["status"] = "completed"

        v3_jobs[job_id]["status"] = "completed"
        v3_jobs[job_id]["progress"] = 100
        v3_jobs[job_id]["message"] = "V3 pipeline completed"
        v3_jobs[job_id]["result"] = sanitize_dict({
            "backtest_results": result["backtest_results"],
            "signals": result["signals"],
            "allocation": {},
            "pipeline_status": {"status": "completed"},
            "timestamp": datetime.now().isoformat(),
        })

        results_cache[job_id] = v3_jobs[job_id]["result"]

    except Exception as e:
        v3_jobs[job_id]["status"] = "failed"
        v3_jobs[job_id]["message"] = str(e)
        logging.error(f"[V3 Pipeline Error] job_id={job_id} error={e}")


@app.get("/api/v1/v3/{job_id}/status")
async def get_v3_status(job_id: str = Path(...)):
    """Get V3 pipeline status with step details."""
    if job_id not in v3_jobs:
        raise HTTPException(status_code=404, detail="V3 job not found")

    job = v3_jobs[job_id]
    return sanitize_dict({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "symbols": job.get("symbols", []),
        "steps": job.get("steps", []),
        "current_step": job.get("current_step", 0),
        "total_steps": job.get("total_steps", 4),
        "timestamp": datetime.now().isoformat(),
        "result": job.get("result"),
    })


# ==================== WALLET & PORTFOLIO ENDPOINTS ====================

class TradeAction(BaseModel):
    symbol: str
    action: str  # buy, sell
    quantity: Optional[int] = None
    amount: Optional[float] = None  # For buy with specific amount


class WalletResetRequest(BaseModel):
    initial_balance: float = 100000.0


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
                detail=f"Insufficient balance. Required: ₹{total_cost:.2f}, Available: ₹{wallet_state['balance']:.2f}"
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


@app.post("/api/v1/wallet/auto-trade")
async def auto_trade_from_signal(symbol: str):
    """Auto-execute trade based on AI model signal from latest pipeline."""
    global wallet_state

    if symbol not in AVAILABLE_STOCKS:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")

    try:
        # First check if we have a signal from an existing pipeline
        signal = None
        for job_id in sorted(pipeline_orchestrators.keys(), reverse=True):
            orchestrator = pipeline_orchestrators[job_id]
            signals = orchestrator.get_signals()
            if signals and symbol in signals:
                signal = signals[symbol]
                break

        if not signal:
            raise HTTPException(
                status_code=400,
                detail=f"No signal found for {symbol}. Run a pipeline first that includes this symbol."
            )

        current_price = get_current_price(symbol)
        if current_price <= 0:
            raise HTTPException(status_code=400, detail="Could not fetch price")

        result = {
            "symbol": symbol,
            "signal": signal,
            "price": current_price,
            "action_taken": None,
            "timestamp": datetime.now().isoformat()
        }

        # Execute based on signal
        if signal.get("action") == "BUY" and signal.get("confidence", 0) > current_config.get("min_confidence", 0.55):
            # Calculate position size based on config
            max_position = wallet_state["balance"] * current_config.get("max_position_pct", 0.15)
            quantity = int(max_position / current_price)

            if quantity > 0 and wallet_state["balance"] >= quantity * current_price:
                trade_result = await execute_trade(TradeAction(
                    symbol=symbol,
                    action="buy",
                    quantity=quantity
                ))
                result["action_taken"] = trade_result
                result["executed"] = True
            else:
                result["executed"] = False
                result["reason"] = "Insufficient balance or position too small"

        elif signal.get("action") == "SELL" and symbol in wallet_state["portfolio"]:
            trade_result = await execute_trade(TradeAction(
                symbol=symbol,
                action="sell"
            ))
            result["action_taken"] = trade_result
            result["executed"] = True

        else:
            result["executed"] = False
            result["reason"] = f"Signal is {signal.get('action', 'HOLD')} - no action needed"

        return sanitize_dict(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
