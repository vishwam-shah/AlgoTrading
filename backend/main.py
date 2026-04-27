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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Silence noisy third-party loggers immediately at import time
for _noisy in ["SmartApi", "smartConnect", "logzero", "urllib3", "yfinance"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
from fastapi import Path
from pathlib import Path as FilePath  # Only use this alias for file paths, not FastAPI params


from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import threading
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

try:
    from engine.orchestrator import UnifiedOrchestrator  # type: ignore
    from engine.sentiment import FastSentimentEngine      # type: ignore
    _LEGACY_ENGINE = True
except ImportError:
    # Old engine not present — V3-native endpoints still work
    UnifiedOrchestrator = None   # type: ignore
    FastSentimentEngine = None   # type: ignore
    _LEGACY_ENGINE = False

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
        if not _LEGACY_ENGINE:
            return JSONResponse(status_code=503, content={"error": "Legacy engine not available. Use /api/v3/sentiment/{symbol}"})
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

        if not _LEGACY_ENGINE:
            running_jobs[job_id]["status"] = "failed"
            running_jobs[job_id]["message"] = "Legacy engine not available. Use /api/v1/v3/run instead."
            return

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
        if not _LEGACY_ENGINE:
            running_jobs[job_id]["status"] = "failed"
            running_jobs[job_id]["message"] = "Legacy engine not available. Use /api/v1/v3/run."
            return

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
        if not _LEGACY_ENGINE:
            raise ImportError("legacy engine unavailable")
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

_WORKSPACE    = FilePath(__file__).resolve().parent.parent
_PYTHON_BIN   = str(_WORKSPACE / "venv" / "bin" / "python")
_ORCHESTRATOR = str(_WORKSPACE / "V3" / "07_pipeline" / "orchestrator.py")

class V3RunRequest(BaseModel):
    symbols: List[str] = []
    capital: float = 500000
    fast: bool = True
    force_features: bool = False


@app.post("/api/v1/v3/run")
async def run_v3(request: V3RunRequest, background_tasks: BackgroundTasks):
    """
    Start V3 pipeline as a SEPARATE SUBPROCESS — never blocks uvicorn.
    Progress tracked by polling summary.csv file counts.
    """
    import subprocess, os

    job_id  = f"v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    symbols = request.symbols or []

    # Build orchestrator command
    cmd = [_PYTHON_BIN, _ORCHESTRATOR]
    if request.fast:
        cmd.append("--fast")
    if request.force_features:
        cmd.append("--force-features")
    if symbols:
        cmd += ["--symbols"] + symbols

    log_path = _WORKSPACE / "V3" / "07_pipeline" / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    v3_jobs[job_id] = {
        "status":      "running",
        "progress":    0,
        "message":     f"Pipeline started — {len(symbols) or 99} stocks",
        "symbols":     symbols,
        "log_path":    str(log_path),
        "pid":         None,
        "run_id":      None,
        "result":      None,
    }

    background_tasks.add_task(_monitor_v3_subprocess, job_id, cmd, log_path)
    return {"job_id": job_id, "status": "started", "symbols": symbols, "cmd": " ".join(cmd[:4])}


def _monitor_v3_subprocess(job_id: str, cmd: list, log_path: "FilePath"):
    """
    Launch the V3 orchestrator as a subprocess and monitor it.
    Progress is inferred from log lines — no RAM/CPU shared with uvicorn.
    Max runtime: 4 hours. Sleep 2s between polls to keep thread-pool free.
    """
    import subprocess
    import re
    import time as _time

    # Keywords → (progress %, human label)
    PROGRESS_MARKERS = [
        (r"Starting.*pipeline",         5,  "Initializing pipeline"),
        (r"[Dd]ownload.*data|fetch.*data|Collecting data", 15, "Downloading market data"),
        (r"[Cc]omput.*feature|[Ff]eature.*engineer",      35, "Computing features"),
        (r"[Tt]rain.*model|[Ff]itting.*model",             55, "Training models"),
        (r"[Bb]acktest|walk.forward",                      75, "Running backtest"),
        (r"[Ss]ignal|[Pp]rediction",                       90, "Generating signals"),
        (r"[Cc]omplete|[Ff]inished|[Dd]one",              100, "Pipeline complete"),
    ]

    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(_WORKSPACE),
            )

        v3_jobs[job_id]["pid"] = proc.pid

        # Stream log file for progress hints — 2s poll, 4h hard cap
        MAX_RUNTIME_S = 4 * 3600
        started_at    = _time.monotonic()
        with open(log_path, "r") as log_fh:
            while True:
                ret  = proc.poll()
                line = log_fh.readline()
                if line:
                    for pattern, pct, label in PROGRESS_MARKERS:
                        if re.search(pattern, line):
                            if pct > v3_jobs[job_id].get("progress", 0):
                                v3_jobs[job_id]["progress"] = pct
                                v3_jobs[job_id]["message"]  = label
                    if "ERROR" in line or "Traceback" in line:
                        v3_jobs[job_id]["message"] = line.strip()[:200]
                elif ret is not None:
                    break
                elif _time.monotonic() - started_at > MAX_RUNTIME_S:
                    proc.kill()
                    v3_jobs[job_id].update({"status": "failed", "message": "Killed: exceeded 4h limit"})
                    break
                else:
                    _time.sleep(2)   # 2s instead of 0.5s — keeps thread-pool free

        # ── Subprocess finished ──────────────────────────────────────────
        if proc.returncode == 0:
            # Try to load summary from the latest run folder
            summary_csv = _WORKSPACE / "V3" / "07_pipeline" / "runs" / "latest" / "summary.csv"
            result_data: dict = {}
            if summary_csv.exists():
                try:
                    df = pd.read_csv(summary_csv)
                    result_data["signals"] = df.to_dict(orient="records")
                except Exception:
                    pass

            v3_jobs[job_id].update({
                "status":   "completed",
                "progress": 100,
                "message":  "V3 pipeline completed successfully",
                "result":   sanitize_dict({
                    "signals":         result_data.get("signals", []),
                    "backtest_results": {},
                    "equity_curve":    [],
                    "allocation":      {},
                    "pipeline_status": {"status": "completed"},
                    "log_path":        str(log_path),
                    "timestamp":       datetime.now().isoformat(),
                }),
            })
            results_cache[job_id] = v3_jobs[job_id]["result"]
        else:
            # Read tail of log for error context
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-30:])
            except Exception:
                tail = "(log unreadable)"
            v3_jobs[job_id].update({
                "status":  "failed",
                "message": f"Pipeline exited with code {proc.returncode}",
                "result":  {"error_tail": tail},
            })

    except Exception as exc:
        import traceback
        v3_jobs[job_id].update({
            "status":  "failed",
            "message": str(exc),
        })
        logging.error(f"[V3 Subprocess Monitor] job_id={job_id}\n{traceback.format_exc()}")


@app.get("/api/v1/v3/{job_id}/status")
async def get_v3_status(job_id: str = Path(...)):
    """Get V3 pipeline status with step details."""
    if job_id not in v3_jobs:
        raise HTTPException(status_code=404, detail="V3 job not found")

    job = v3_jobs[job_id]
    return sanitize_dict({
        "job_id":   job_id,
        "status":   job.get("status"),
        "progress": job.get("progress", 0),
        "message":  job.get("message", ""),
        "symbols":  job.get("symbols", []),
        "pid":      job.get("pid"),
        "log_path": job.get("log_path"),
        "timestamp": datetime.now().isoformat(),
        "result":   job.get("result"),
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


# ══════════════════════════════════════════════════════════════════════════════
#  V3 NATIVE ENDPOINTS — reads directly from V3 pipeline output files
#  No dependency on old engine/. Pure file readers → zero import issues.
# ══════════════════════════════════════════════════════════════════════════════

_V3_ROOT    = FilePath(__file__).resolve().parent.parent / "V3"
_RUNS_DIR   = _V3_ROOT / "06_results" / "runs"
_ORDERS_DIR = _V3_ROOT / "05_live_trading" / "orders"


def _latest_run_id() -> Optional[str]:
    """
    Return the most recently completed run (newest folder name that has a summary.csv).
    Folders are named YYYYMMDD_HHMMSS so lexicographic descending == chronological.
    """
    for run_dir in sorted(_RUNS_DIR.glob("20*"), reverse=True):
        if (run_dir / "summary.csv").exists():
            return run_dir.name
    return None


def _read_summary(run_id: str) -> dict:
    path = _RUNS_DIR / run_id / "summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return sanitize_dict(df.to_dict(orient="records"))


# ── Latest run id (polling endpoint) ─────────────────────────────────────────

@app.get("/api/v3/runs/latest-id")
async def get_latest_run_id():
    """Lightweight endpoint — returns only the most recent run_id. Frontend polls this."""
    run_id = _latest_run_id()
    if not run_id:
        raise HTTPException(404, detail="No runs found")
    return {"run_id": run_id, "timestamp": datetime.now().isoformat()}


# ── List runs ─────────────────────────────────────────────────────────────────

@app.get("/api/v3/runs")
async def list_v3_runs():
    """List all completed V3 pipeline runs with summary stats."""
    runs = []
    for run_dir in sorted(_RUNS_DIR.glob("20*"), reverse=True):
        meta_path = run_dir / "run_metadata.json"
        summary_path = run_dir / "summary.csv"
        entry = {"run_id": run_dir.name}
        if meta_path.exists():
            with open(meta_path) as f:
                entry.update(json.load(f))
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            avg_row = df[df["symbol"] == "AVERAGE"]
            if not avg_row.empty:
                entry["avg_accuracy"] = round(float(avg_row.iloc[0].get("oos_accuracy", 0)), 4)
                entry["n_stocks"]     = int(df[df["symbol"] != "AVERAGE"].shape[0])
        runs.append(sanitize_dict(entry))
    return {"runs": runs, "total": len(runs)}


# ── Run summary ───────────────────────────────────────────────────────────────

@app.get("/api/v3/runs/{run_id}/summary")
async def get_v3_run_summary(run_id: str):
    """Per-stock OOS accuracy + per-model accuracy + backtest metrics merged."""
    path = _RUNS_DIR / run_id / "summary.csv"
    if not path.exists():
        raise HTTPException(404, detail=f"Run {run_id} not found or still in progress")
    df = pd.read_csv(path)

    # Merge per-model accuracy from each stock's summary_row.json
    _MODEL_COLS = [
        "avg_lgbm_acc", "avg_xgb_acc",
        "avg_lstm_acc", "avg_bilstm_acc", "avg_gru_acc",
        "avg_cnn_lstm_acc", "avg_cnn_gru_acc",
        "avg_tcn_gru_acc", "avg_tcn_transformer_acc", "avg_nbeats_acc",
        "best_model", "best_model_acc",
    ]
    model_rows = []
    run_dir = _RUNS_DIR / run_id
    for sym_dir in run_dir.iterdir():
        json_path = sym_dir / "summary_row.json"
        if not json_path.exists():
            continue
        try:
            with open(json_path) as _f:
                row = json.load(_f)
            model_rows.append({k: row.get(k) for k in ["symbol"] + _MODEL_COLS})
        except Exception:
            pass
    if model_rows:
        model_df = pd.DataFrame(model_rows)
        df = df.merge(model_df, on="symbol", how="left")

    # Merge backtest metrics (sharpe, binary_dir_acc, tradeable, etc.)
    bt_path = _RUNS_DIR / run_id / "backtest_results.csv"
    if bt_path.exists():
        bt_df   = pd.read_csv(bt_path)
        bt_want = ["symbol", "sharpe", "binary_dir_acc", "tradeable",
                   "n_trades", "win_rate", "total_return", "profit_factor",
                   "max_drawdown", "up_signal_acc", "ann_return"]
        bt_df = bt_df[[c for c in bt_want if c in bt_df.columns]]
        df = df.merge(bt_df, on="symbol", how="left")

    return sanitize_dict({
        "run_id": run_id,
        "stocks": df.to_dict(orient="records"),
        "count": len(df),
    })


# ── Next-day predictions ──────────────────────────────────────────────────────

@app.get("/api/v3/predictions/latest")
async def get_latest_predictions():
    """Next-day trading signals from the most recent run."""
    run_id = _latest_run_id()
    if not run_id:
        raise HTTPException(404, detail="No runs found")
    return await get_v3_predictions(run_id)


@app.get("/api/v3/predictions/{run_id}")
async def get_v3_predictions(run_id: str):
    """Next-day trading signals for a specific run."""
    path = _RUNS_DIR / run_id / "next_day_predictions.csv"
    if not path.exists():
        raise HTTPException(404, detail=f"No predictions for run {run_id}")
    df = pd.read_csv(path)
    # Normalise probability column — pipeline writes avg_prob, frontend expects prob_up
    if "avg_prob" in df.columns and "prob_up" not in df.columns:
        df["prob_up"] = df["avg_prob"]
    # Use last_close as price hint if present
    if "last_close" in df.columns:
        df["price"] = df["last_close"]
    return sanitize_dict({
        "run_id": run_id,
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
        "up_count":   int((df["direction"] == "UP").sum())   if "direction" in df.columns else 0,
        "down_count": int((df["direction"] == "DOWN").sum()) if "direction" in df.columns else 0,
        "generated_at": str(path.stat().st_mtime),
    })


# ── Backtest results ──────────────────────────────────────────────────────────

def _load_backtest(run_id: str) -> dict:
    """Load backtest_results.csv for a run. Raises HTTPException if missing."""
    path = _RUNS_DIR / run_id / "backtest_results.csv"
    if not path.exists():
        raise HTTPException(404, detail=f"No backtest data for run {run_id}. Re-run the pipeline.")
    df = pd.read_csv(path)
    profitable = df[df["sharpe"] > 0]
    tradeable  = df[df.get("tradeable", pd.Series(False, index=df.index)) == True]
    portfolio_path = _RUNS_DIR / run_id / "backtest_portfolio.csv"
    portfolio_curve: list = []
    if portfolio_path.exists():
        pf = pd.read_csv(portfolio_path).fillna(0)
        portfolio_curve = sanitize_dict(pf.to_dict(orient="records"))
    # Load supplementary bootstrap + NIFTY summary if available
    bt_summary: dict = {}
    summary_json = _RUNS_DIR / run_id / "backtest_summary.json"
    if summary_json.exists():
        try:
            with open(summary_json) as _f:
                bt_summary = json.load(_f)
        except Exception:
            pass

    return sanitize_dict({
        "run_id":              run_id,
        "stocks":              df.to_dict(orient="records"),
        "n_total":             len(df),
        "n_tradeable":         int(len(tradeable)),
        "n_profitable":        int(len(profitable)),
        "portfolio_sharpe":    round(float(profitable["sharpe"].mean()), 3) if not profitable.empty else 0,
        "portfolio_win_rate":  round(float(profitable["win_rate"].mean()), 4) if not profitable.empty else 0,
        "portfolio_total_ret": round(float(profitable["total_return"].mean()), 4) if not profitable.empty else 0,
        "portfolio_max_dd":    round(float(profitable["max_drawdown"].mean()), 4) if not profitable.empty else 0,
        "portfolio_curve":     portfolio_curve,
        "cost_model":          "0.25% round-trip (STT+brokerage+slippage)",
        # Bootstrap CI for statistical significance
        "bootstrap_significant":  bt_summary.get("bootstrap_significant", False),
        "bootstrap_ci_lower":     bt_summary.get("bootstrap_ci_lower", 0.0),
        "bootstrap_ci_upper":     bt_summary.get("bootstrap_ci_upper", 0.0),
        "bootstrap_acc_mean":     bt_summary.get("bootstrap_acc_mean", 0.0),
        "bootstrap_n_signals":    bt_summary.get("bootstrap_n_signals", 0),
        # NIFTY buy-and-hold comparison
        "nifty_return":           bt_summary.get("nifty_return", None),
        "nifty_start_date":       bt_summary.get("nifty_start_date", ""),
        "nifty_end_date":         bt_summary.get("nifty_end_date", ""),
    })


# IMPORTANT: specific route BEFORE parameterized route so FastAPI doesn't
# capture "latest" as a run_id.
@app.get("/api/v3/runs/latest/backtest")
async def get_latest_backtest():
    """Backtest results for the most recent run."""
    run_id = _latest_run_id()
    if not run_id:
        raise HTTPException(404, detail="No runs found")
    return _load_backtest(run_id)


@app.get("/api/v3/runs/{run_id}/backtest")
async def get_backtest_results(run_id: str):
    """Per-stock simulated P&L metrics for a specific run."""
    return _load_backtest(run_id)


# ── Paper trading ─────────────────────────────────────────────────────────────

_PT_DIR = _V3_ROOT / "05_live_trading" / "paper_trading_logs"

@app.get("/api/v3/paper/sessions")
async def list_paper_sessions():
    """List all paper trading sessions."""
    _PT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for f in sorted(_PT_DIR.glob("session_*.json"), reverse=True):
        try:
            with open(f) as fp:
                s = json.load(fp)
            sessions.append({"session_id": f.stem.replace("session_", ""),
                              "trades": len(s.get("trades", [])),
                              "cash": s.get("cash", 0),
                              "initial_cash": s.get("initial_cash", 0)})
        except Exception:
            pass
    return sanitize_dict({"sessions": sessions, "count": len(sessions)})


@app.post("/api/v3/paper/start")
async def start_paper_session(background_tasks: BackgroundTasks):
    """
    Start a new paper trading session using the best stocks from the latest backtest.
    Selects stocks where: tradeable=True AND sharpe>0.5 (VOLTAS, TECHM, VEDL, FEDERALBNK, ADANIENT).
    """
    run_id = _latest_run_id()
    if not run_id:
        raise HTTPException(404, detail="No pipeline run found. Run the pipeline first.")

    bt_path = _RUNS_DIR / run_id / "backtest_results.csv"
    if not bt_path.exists():
        raise HTTPException(404, detail="No backtest results. Run pipeline first.")

    bt_df = pd.read_csv(bt_path)
    top_stocks = bt_df[
        (bt_df["tradeable"] == True) & (bt_df["sharpe"] >= 0.5)
    ][["symbol", "sharpe", "win_rate", "oos_accuracy"]].to_dict(orient="records")

    if not top_stocks:
        raise HTTPException(400, detail="No stocks qualify (need tradeable=True AND sharpe>=0.5)")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _PT_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = _RUNS_DIR / run_id / "next_day_predictions.csv"
    todays_signals = []
    if pred_path.exists():
        pdf = pd.read_csv(pred_path)
        qualified = set(s["symbol"] for s in top_stocks)
        todays_signals = pdf[
            pdf["symbol"].isin(qualified) & (pdf["direction"] == "UP")
        ].to_dict(orient="records")

    session = {
        "session_id":   session_id,
        "run_id":       run_id,
        "created_at":   datetime.now().isoformat(),
        "initial_cash": 500000,
        "cash":         500000,
        "qualified_stocks": top_stocks,
        "todays_signals":   sanitize_dict(todays_signals),
        "holdings":     {},
        "trades":       [],
        "status":       "active",
        "note": (
            f"Paper trading {len(top_stocks)} stocks with Sharpe>=0.5 from run {run_id}. "
            "Buy signals at tomorrow open, exit at next close."
        ),
    }

    with open(_PT_DIR / f"session_{session_id}.json", "w") as f:
        json.dump(session, f, indent=2)

    return sanitize_dict({
        "session_id":       session_id,
        "qualified_stocks": top_stocks,
        "todays_signals":   sanitize_dict(todays_signals),
        "message": f"Paper trading session started with {len(top_stocks)} stocks",
    })


@app.get("/api/v3/paper/latest")
async def get_latest_paper_session():
    """Most recent paper trading session."""
    _PT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(_PT_DIR.glob("session_*.json"), reverse=True)
    if not files:
        raise HTTPException(404, detail="No paper trading sessions yet")
    with open(files[0]) as f:
        return sanitize_dict(json.load(f))


# ── Per-stock predictions history ─────────────────────────────────────────────

@app.get("/api/v3/runs/{run_id}/stock/{symbol}/predictions")
async def get_stock_predictions(run_id: str, symbol: str):
    """Full OOS prediction history for one stock in a run."""
    path = _RUNS_DIR / run_id / symbol / "predictions.csv"
    if not path.exists():
        raise HTTPException(404, detail=f"{symbol} predictions not found in run {run_id}")
    df = pd.read_csv(path)
    accuracy = float((df["actual"] == df["ensemble_pred"]).mean()) if "actual" in df.columns else None
    return sanitize_dict({
        "run_id": run_id, "symbol": symbol,
        "oos_accuracy": round(accuracy, 4) if accuracy else None,
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
    })


# ── Approved orders ───────────────────────────────────────────────────────────

@app.get("/api/v3/orders/latest")
async def get_latest_orders():
    """Most recent approved order set from signal_publisher."""
    files = sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)
    if not files:
        raise HTTPException(404, detail="No order files found. Run signal_publisher.py first.")
    with open(files[0]) as f:
        orders = json.load(f)
    return sanitize_dict({
        "file": files[0].name,
        "orders": orders,
        "count": len(orders),
        "total_deployed": sum(o.get("order_value", 0) for o in orders),
    })


@app.get("/api/v3/orders/history")
async def get_order_history():
    """All historical order files."""
    files = sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)
    history = []
    for f in files[:30]:  # last 30
        try:
            with open(f) as fh:
                orders = json.load(fh)
            history.append({
                "file": f.name,
                "count": len(orders),
                "total_deployed": round(sum(o.get("order_value", 0) for o in orders), 0),
                "date": f.name.split("_")[-1].replace(".json", ""),
            })
        except Exception:
            pass
    return {"history": history, "total_files": len(files)}


# ── Serve plot images ─────────────────────────────────────────────────────────

@app.get("/api/v3/runs/{run_id}/plots/{filename}")
async def get_run_plot(run_id: str, filename: str):
    """Serve run-level plot PNG (cross_stock_comparison, model_comparison_heatmap, etc.)."""
    path = _RUNS_DIR / run_id / "plots" / filename
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(404, detail=f"Plot {filename} not found")
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/v3/runs/{run_id}/stock/{symbol}/plots/{filename}")
async def get_stock_plot(run_id: str, symbol: str, filename: str):
    """Serve per-stock plot PNG (confidence_timeline, oos_accuracy, confusion_matrix)."""
    path = _RUNS_DIR / run_id / symbol / "plots" / filename
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(404, detail=f"Plot {filename} not found for {symbol}")
    return FileResponse(str(path), media_type="image/png")


# ── Live prices via WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws/live-prices")
async def live_prices_ws(websocket: WebSocket):
    """
    WebSocket endpoint that streams live NSE prices every 5 seconds.
    Client sends: {"symbols": ["SBIN", "HDFCBANK", ...]}
    Server sends: {"prices": {"SBIN": 812.5, ...}, "timestamp": "..."}
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    symbols = []
    try:
        # First message must be symbol list
        msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        symbols = msg.get("symbols", [])[:50]  # cap at 50
        import yfinance as yf

        # Load verified ticker map
        try:
            import sys as _sys
            _sys.path.insert(0, str(_V3_ROOT / "00_config"))
            from tickers import to_yf, to_nse  # type: ignore
        except Exception:
            def to_yf(s: str) -> str: return f"{s}.NS"   # type: ignore
            def to_nse(t: str) -> str: return t.replace(".NS", "")  # type: ignore
        _TICKER_OVERRIDES: dict = {}  # kept for compat — mapping now in tickers.py
        # Cache of last known good prices (avoids gaps when one ticker fails)
        _price_cache: dict = {}

        import logging as _log
        _yf_logger = _log.getLogger("yfinance")
        _yf_logger.setLevel(_log.CRITICAL)  # silence yfinance warnings

        while True:
            try:
                if symbols:
                    tickers = [to_yf(s) for s in symbols]
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                        data = yf.download(
                            tickers, period="1d", interval="1m",
                            auto_adjust=True, progress=False, threads=True,
                            ignore_tz=True,
                        )
                    prices = dict(_price_cache)  # start from cache
                    if not data.empty:
                        closes = data["Close"].iloc[-1] if "Close" in data.columns else data.iloc[-1]
                        for sym, ticker in zip(symbols, tickers):
                            val = closes.get(ticker, float("nan"))
                            if not (isinstance(val, float) and math.isnan(val)):
                                prices[sym] = round(float(val), 2)
                                _price_cache[sym] = prices[sym]
                    await websocket.send_json({
                        "prices": prices,
                        "timestamp": datetime.now().isoformat(),
                    })
            except Exception:
                pass
            await asyncio.sleep(10)  # 10s instead of 5s — less load
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


# ── Sentiment history endpoint ────────────────────────────────────────────────

@app.get("/api/v3/sentiment/{symbol}")
async def get_v3_sentiment(symbol: str):
    """Get FinBERT sentiment history for a symbol from sentiment_history.parquet."""
    sent_path = _V3_ROOT / "01_data" / "news" / "sentiment_history.parquet"
    if not sent_path.exists():
        raise HTTPException(404, detail="No sentiment history found. Run sentiment_history.py first.")
    try:
        df = pd.read_parquet(sent_path)
    except Exception as e:
        raise HTTPException(500, detail=f"Sentiment file corrupted: {e}")
    df = df[df["symbol"] == symbol.upper()].copy()
    if df.empty:
        raise HTTPException(404, detail=f"No sentiment history for {symbol}")
    df["date"] = df["date"].astype(str)
    latest = df.sort_values("date").iloc[-1]
    return sanitize_dict({
        "symbol": symbol,
        "latest_score": float(latest.get("raw_score", 0)),
        "latest_date":  str(latest["date"]),
        "n_articles":   int(latest.get("n_articles", 0)),
        "model_used":   str(latest.get("model_used", "unknown")),
        "history": df.sort_values("date", ascending=False).head(30).to_dict(orient="records"),
    })


@app.get("/api/v3/sentiment/overview")
async def get_v3_sentiment_overview():
    """
    Aggregate FinBERT sentiment across all 100 symbols for dashboard display.
    Returns: latest date, top bullish/bearish, per-stock sentiment, coverage health,
             and sentiment-tagged tradeable stocks from the latest run.
    """
    sent_path = _V3_ROOT / "01_data" / "news" / "sentiment_history.parquet"
    if not sent_path.exists():
        raise HTTPException(404, detail="No sentiment history found.")
    try:
        df = pd.read_parquet(sent_path)
    except Exception as e:
        raise HTTPException(500, detail=f"Sentiment file corrupted: {e}")

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    n_zero = int((latest["n_articles"] == 0).sum())
    n_total = int(len(latest))

    top_bull = latest.nlargest(5, "raw_score")[
        ["symbol", "raw_score", "n_articles", "positive_ratio", "negative_ratio"]
    ].to_dict(orient="records")
    top_bear = latest.nsmallest(5, "raw_score")[
        ["symbol", "raw_score", "n_articles", "positive_ratio", "negative_ratio"]
    ].to_dict(orient="records")

    avg_score  = float(latest["raw_score"].mean()) if len(latest) else 0.0
    avg_articles = float(latest["n_articles"].mean()) if len(latest) else 0.0
    model_used = str(latest["model_used"].mode()[0]) if "model_used" in latest.columns and len(latest) else "unknown"

    # Merge with latest run's tradeable universe
    runs = sorted([d for d in _RUNS_DIR.iterdir() if d.is_dir()])
    tradeable_rows = []
    if runs:
        latest_run = runs[-1]
        pred_path = latest_run / "next_day_predictions.csv"
        if pred_path.exists():
            try:
                pdf = pd.read_csv(pred_path)
                if "tradeable" in pdf.columns:
                    pdf = pdf[pdf["tradeable"] == True]
                merged = pdf.merge(
                    latest[["symbol", "raw_score", "n_articles", "positive_ratio", "negative_ratio"]],
                    on="symbol", how="left"
                )
                for _, r in merged.iterrows():
                    score = r.get("raw_score")
                    if pd.isna(score):
                        score = 0.0
                    # sentiment_prob mirrors predict.py:649
                    import numpy as _np
                    sent_prob = 0.5 + float(_np.clip(score, -1, 1)) * 0.25
                    tradeable_rows.append({
                        "symbol": str(r["symbol"]),
                        "direction": str(r.get("direction", "—")),
                        "confidence": float(r.get("confidence", 0)) if not pd.isna(r.get("confidence", 0)) else 0.0,
                        "sentiment_score": float(score),
                        "sentiment_prob": round(sent_prob, 4),
                        "n_articles": int(r["n_articles"]) if not pd.isna(r.get("n_articles")) else 0,
                        "positive_ratio": float(r["positive_ratio"]) if not pd.isna(r.get("positive_ratio")) else 0.0,
                        "negative_ratio": float(r["negative_ratio"]) if not pd.isna(r.get("negative_ratio")) else 0.0,
                    })
            except Exception:
                pass

    all_scores = latest[["symbol", "raw_score", "n_articles", "positive_ratio", "negative_ratio"]] \
        .sort_values("raw_score", ascending=False).to_dict(orient="records")

    return sanitize_dict({
        "latest_date": str(latest_date.date()),
        "n_symbols": n_total,
        "n_zero_coverage": n_zero,
        "avg_score": round(avg_score, 4),
        "avg_articles": round(avg_articles, 1),
        "model_used": model_used,
        "top_bullish": top_bull,
        "top_bearish": top_bear,
        "tradeable": tradeable_rows,
        "all_scores": all_scores,
        "blend_weight": 0.15,  # from predict.py:650 — 80% ensemble + 15% sentiment
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ANGEL ONE LIVE TRADING ENDPOINTS
#  Thin wrappers over angel_one_client + order_manager + daily_runner.
#  Credentials live in .env — never in source.
# ══════════════════════════════════════════════════════════════════════════════

_LIVE_DIR     = _V3_ROOT / "05_live_trading"
_EXEC_LOG_DIR = _LIVE_DIR / "execution_logs"
_HISTORY_PATH = _LIVE_DIR / "trade_history.parquet"
sys.path.insert(0, str(_LIVE_DIR))


_angel_session_lock  = threading.Lock()
# "fail_until" blocks re-login attempts after a failure; Angel One's TOTP
# code is valid for 30 s and cannot be reused — so on any failure we back
# off for 45 s to let both the rate-limit window and TOTP window reset.
_angel_session_cache: dict = {
    "client": None, "expires_at": 0.0,
    "fail_reason": None, "fail_until": 0.0,
}
_ANGEL_SESSION_TTL     = 270  # 4.5 min — well under Angel One's token expiry
_ANGEL_FAIL_BACKOFF_S  = 45   # must exceed TOTP window (30 s)


def _angel_client():
    """Return a cached logged-in AngelOneClient. Logins once per 4.5 min, not per request."""
    import time as _t
    now = _t.time()
    # Fast path — no lock needed when cache is warm
    if _angel_session_cache["client"] is not None and now < _angel_session_cache["expires_at"]:
        return _angel_session_cache["client"]
    # Failure backoff — don't hammer Angel One after a failed login
    if _angel_session_cache["fail_reason"] and now < _angel_session_cache["fail_until"]:
        raise HTTPException(503, detail=f"Angel One unavailable: {_angel_session_cache['fail_reason']} (retrying in {int(_angel_session_cache['fail_until'] - now)}s)")

    with _angel_session_lock:
        # Re-check inside lock to prevent double login under concurrent requests
        now = _t.time()
        if _angel_session_cache["client"] is not None and now < _angel_session_cache["expires_at"]:
            return _angel_session_cache["client"]
        if _angel_session_cache["fail_reason"] and now < _angel_session_cache["fail_until"]:
            raise HTTPException(503, detail=f"Angel One unavailable: {_angel_session_cache['fail_reason']}")
        try:
            from angel_one_client import AngelOneClient  # type: ignore
            c = AngelOneClient()
            if not c.login():
                raise RuntimeError("Login failed")
            _angel_session_cache["client"]      = c
            _angel_session_cache["expires_at"]  = now + _ANGEL_SESSION_TTL
            _angel_session_cache["fail_reason"] = None
            _angel_session_cache["fail_until"]  = 0.0
            return c
        except EnvironmentError as e:
            _angel_session_cache["client"]      = None
            _angel_session_cache["fail_reason"] = f"credentials missing: {e}"
            _angel_session_cache["fail_until"]  = now + _ANGEL_FAIL_BACKOFF_S
            raise HTTPException(503, detail=f"Angel One credentials missing: {e}")
        except Exception as e:
            _angel_session_cache["client"]      = None
            _angel_session_cache["fail_reason"] = str(e)
            _angel_session_cache["fail_until"]  = now + _ANGEL_FAIL_BACKOFF_S
            raise HTTPException(503, detail=f"Angel One unavailable: {e}")


# ── Account & Portfolio ──────────────────────────────────────────────────────


@app.get("/api/v3/angel/status")
async def angel_status():
    """
    Check Angel One connectivity. Shares the main _angel_client() cache so
    that a single login serves both the status badge and data endpoints —
    avoids triple-login on dashboard load (status + funds + holdings).
    """
    import os, logging
    from dotenv import load_dotenv

    # Silence SmartAPI's internal loggers
    for _log_name in ["SmartApi", "smartConnect", "logzero"]:
        logging.getLogger(_log_name).setLevel(logging.CRITICAL)

    env_path = FilePath(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    has_creds = all(os.getenv(k) for k in [
        "ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET"
    ])
    if not has_creds:
        return {"credentials_present": False, "login_ok": False,
                "message": "Missing credentials in .env", "env_path": str(env_path)}

    sys.path.insert(0, str(_LIVE_DIR))
    try:
        _angel_client()                      # populates or reuses shared cache
        return {"credentials_present": True, "login_ok": True,
                "message": "Connected", "env_path": str(env_path)}
    except HTTPException as e:
        # Surface the same rate-limit / backoff message back to UI
        return {"credentials_present": True, "login_ok": False,
                "message": str(e.detail), "env_path": str(env_path)}
    except Exception as e:
        return {"credentials_present": True, "login_ok": False,
                "message": str(e), "env_path": str(env_path)}


@app.get("/api/v3/angel/funds")
async def get_angel_funds():
    """Return available cash and margin from Angel One."""
    c = _angel_client()
    funds = c.get_funds()
    return sanitize_dict({"funds": funds, "timestamp": datetime.now().isoformat()})


@app.get("/api/v3/angel/holdings")
async def get_angel_holdings():
    """Return current CNC holdings from Angel One."""
    c = _angel_client()
    holdings = c.get_holdings()
    rows = [{"symbol": sym, "qty": p.qty, "avg_price": p.avg_price,
             "ltp": p.ltp, "pnl": p.pnl}
            for sym, p in holdings.items()]
    total_pnl = sum(p.pnl for p in holdings.values())
    return sanitize_dict({
        "holdings": rows,
        "count": len(rows),
        "total_pnl": round(total_pnl, 2),
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/v3/angel/orders")
async def get_angel_order_book():
    """Return today's order book from Angel One."""
    c = _angel_client()
    orders = c.get_order_book()
    return sanitize_dict({"orders": orders, "count": len(orders),
                           "timestamp": datetime.now().isoformat()})


@app.get("/api/v3/angel/ltp/{symbol}")
async def get_angel_ltp(symbol: str):
    """Get live LTP for a symbol from Angel One."""
    c = _angel_client()
    ltp = c.get_ltp(symbol.upper())
    if ltp is None:
        raise HTTPException(404, detail=f"No LTP for {symbol}")
    return {"symbol": symbol.upper(), "ltp": ltp, "timestamp": datetime.now().isoformat()}


# ── Order execution endpoints ─────────────────────────────────────────────────

class PlaceOrderRequest(BaseModel):
    symbol:     str
    qty:        int
    price:      float
    order_type: str = "LIMIT"
    side:       str = "BUY"
    product:    str = "CNC"


@app.post("/api/v3/angel/place-order")
async def place_angel_order(req: PlaceOrderRequest):
    """Place a single order on Angel One (live)."""
    c = _angel_client()
    resp = c.place_order(
        symbol=req.symbol.upper(), qty=req.qty, price=req.price,
        order_type=req.order_type, side=req.side, product=req.product,
    )
    return sanitize_dict({
        "order_id": resp.order_id,
        "symbol":   resp.symbol,
        "status":   resp.status,
        "message":  resp.message,
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/v3/angel/execute-today")
async def execute_today_orders(background_tasks: BackgroundTasks,
                                capital: float = 500_000,
                                paper: bool = True):
    """
    Execute today's approved orders from signal_publisher.
    paper=True (default): simulate fills only, no real orders.
    paper=False: live execution on Angel One.
    """
    job_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v3_jobs[job_id] = {
        "status": "running", "progress": 0,
        "message": "Loading approved orders …", "result": None,
    }

    def _exec():
        try:
            order_files = sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)
            if not order_files:
                v3_jobs[job_id]["status"] = "failed"
                v3_jobs[job_id]["message"] = "No approved orders found. Run evening pipeline first."
                return

            with open(order_files[0]) as f:
                orders = json.load(f)

            v3_jobs[job_id]["message"] = f"Placing {len(orders)} orders …"

            sys.path.insert(0, str(_LIVE_DIR))
            from order_manager import OrderManager  # type: ignore

            if paper:
                mgr = OrderManager(client=None, paper_mode=True)
            else:
                c = _angel_client()
                mgr = OrderManager(client=c, paper_mode=False)

            mgr.execute_orders(orders)
            mgr.wait_for_fills(timeout_min=30) if not paper else None
            log_path = mgr.save_execution_log()
            summary  = mgr.summary()

            v3_jobs[job_id]["status"]   = "completed"
            v3_jobs[job_id]["progress"] = 100
            v3_jobs[job_id]["message"]  = "Execution complete"
            v3_jobs[job_id]["result"]   = sanitize_dict({
                **summary,
                "log_file": log_path.name,
                "paper_mode": paper,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            v3_jobs[job_id]["status"]  = "failed"
            v3_jobs[job_id]["message"] = str(e)

    background_tasks.add_task(_exec)
    return {"job_id": job_id, "paper": paper, "status": "started"}


# ── Execution logs ────────────────────────────────────────────────────────────

@app.get("/api/v3/execution/logs")
async def get_execution_logs():
    """List all execution log files."""
    _EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(_EXEC_LOG_DIR.glob("execution_*.json"), reverse=True)
    logs = []
    for f in files[:20]:
        try:
            with open(f) as fh:
                data = json.load(fh)
            filled    = sum(1 for r in data if r.get("status") == "FILLED")
            total_val = sum(r.get("order_value", 0) for r in data if r.get("status") == "FILLED")
            logs.append({
                "file": f.name,
                "date": f.name.split("_")[1] if "_" in f.name else "",
                "total_orders": len(data),
                "filled": filled,
                "total_value": round(total_val, 0),
            })
        except Exception:
            pass
    return {"logs": logs, "count": len(files)}


@app.get("/api/v3/execution/logs/{filename}")
async def get_execution_log_detail(filename: str):
    """Get fill details from a specific execution log."""
    path = _EXEC_LOG_DIR / filename
    if not path.exists() or path.suffix != ".json":
        raise HTTPException(404, detail="Log not found")
    with open(path) as f:
        data = json.load(f)
    return sanitize_dict({"fills": data, "count": len(data)})


@app.get("/api/v3/execution/history")
async def get_trade_history(limit: int = 200):
    """Return fills from the most recent execution log file."""
    _EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(_EXEC_LOG_DIR.glob("execution_*.json"), reverse=True)
    if not files:
        return {"trades": [], "count": 0}
    try:
        with open(files[0]) as f:
            trades = json.load(f)
        return sanitize_dict({"trades": trades[:limit], "count": len(trades), "file": files[0].name})
    except Exception:
        return {"trades": [], "count": 0}


@app.get("/api/v3/execution/overview")
async def get_execution_overview(scope: str = "latest"):
    """
    Execution overview for the dashboard's paper/live panel.

    Query:
      scope = "latest"  → positions reflect only the most recent session (default)
      scope = "all"     → positions aggregated across every saved session

    Returns:
      mode           : "paper" | "live"   (from TRADING_MODE env)
      scope          : what `positions` represents ("latest" | "all")
      latest_session : summary + fills of most recent file
      positions      : open positions under the chosen scope
      totals         : all-time fills, deployed, charges (for context)
    """
    _EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    if mode not in ("paper", "live"):
        mode = "paper"

    files = sorted(_EXEC_LOG_DIR.glob("execution_*.json"), reverse=True)
    if not files:
        return {
            "mode": mode,
            "scope": scope,
            "latest_session": None,
            "positions": [],
            "totals": {"fills": 0, "deployed": 0.0, "charges": 0.0, "sessions": 0},
        }

    # Totals are always all-time (even if positions are scoped to latest)
    pos: dict = {}
    total_fills     = 0
    total_deployed  = 0.0
    total_charges   = 0.0
    sessions_seen   = 0

    # For scope=latest, only aggregate the first file; for scope=all, every file.
    files_for_positions = [files[0]] if scope == "latest" else files

    for fp in files:
        try:
            with open(fp) as fh:
                rows = json.load(fh)
        except Exception:
            continue
        sessions_seen += 1
        build_positions = fp in files_for_positions
        for r in rows:
            if r.get("status") != "FILLED":
                continue
            sym   = r.get("symbol") or ""
            side  = (r.get("side") or "BUY").upper()
            qty   = int(r.get("filled_qty") or 0)
            price = float(r.get("avg_price") or 0.0)
            value = float(r.get("order_value") or qty * price)
            chg   = (float(r.get("brokerage") or 0.0)
                     + float(r.get("stt") or 0.0)
                     + float(r.get("other_charges") or 0.0))
            is_paper = str(r.get("order_id", "")).startswith("PAPER_")

            total_fills    += 1
            total_deployed += value
            total_charges  += chg

            if not build_positions:
                continue

            entry = pos.setdefault(sym, {
                "symbol":         sym,
                "buy_qty":        0,
                "sell_qty":       0,
                "buy_cost":       0.0,   # sum(qty * price) on buys
                "sell_proceeds":  0.0,
                "last_price":     0.0,
                "last_trade_at":  "",
                "is_paper":       is_paper,
                "n_trades":       0,
            })
            entry["n_trades"]     += 1
            entry["last_price"]    = price
            t = r.get("filled_at") or r.get("placed_at") or ""
            if t > entry["last_trade_at"]:
                entry["last_trade_at"] = t
            if side == "BUY":
                entry["buy_qty"]   += qty
                entry["buy_cost"]  += qty * price
            elif side == "SELL":
                entry["sell_qty"]      += qty
                entry["sell_proceeds"] += qty * price
            entry["is_paper"] = entry["is_paper"] or is_paper

    # Build positions list — only keep symbols with open net_qty > 0
    positions = []
    for p in pos.values():
        net_qty = p["buy_qty"] - p["sell_qty"]
        if net_qty <= 0:
            continue
        avg_buy_price = p["buy_cost"] / p["buy_qty"] if p["buy_qty"] > 0 else 0.0
        positions.append({
            "symbol":        p["symbol"],
            "net_qty":       net_qty,
            "avg_price":     round(avg_buy_price, 2),
            "invested":      round(net_qty * avg_buy_price, 0),
            "last_price":    round(p["last_price"], 2),
            "last_trade_at": p["last_trade_at"],
            "is_paper":      p["is_paper"],
            "n_trades":      p["n_trades"],
        })
    positions.sort(key=lambda x: x["invested"], reverse=True)

    # ── Latest session detail ─────────────────────────────────────────────
    latest = None
    try:
        with open(files[0]) as fh:
            latest_rows = json.load(fh)
        filled_rows = [r for r in latest_rows if r.get("status") == "FILLED"]
        sess_value   = sum(float(r.get("order_value", 0))   for r in filled_rows)
        sess_charges = sum(float(r.get("brokerage", 0))
                           + float(r.get("stt", 0))
                           + float(r.get("other_charges", 0)) for r in filled_rows)
        is_paper_sess = any(str(r.get("order_id", "")).startswith("PAPER_")
                            for r in latest_rows)
        latest = {
            "file":          files[0].name,
            "timestamp":     files[0].name.replace("execution_", "").replace(".json", ""),
            "mode":          "paper" if is_paper_sess else "live",
            "n_fills":       len(filled_rows),
            "n_total":       len(latest_rows),
            "total_value":   round(sess_value, 0),
            "total_charges": round(sess_charges, 2),
            "fills":         latest_rows,
        }
    except Exception:
        latest = None

    return sanitize_dict({
        "mode":           mode,
        "scope":          scope,
        "latest_session": latest,
        "positions":      positions,
        "totals": {
            "fills":    total_fills,
            "deployed": round(total_deployed, 0),
            "charges":  round(total_charges, 2),
            "sessions": sessions_seen,
        },
    })


@app.post("/api/v3/execution/reset")
async def reset_execution_logs(keep_latest: bool = False):
    """
    Archive all execution logs so the paper portfolio resets to zero.
    Files are moved to execution_logs/archive/<timestamp>/ rather than deleted,
    so nothing is lost — you can recover by moving them back.

    Query:
      keep_latest=true → keep the most recent session, archive the rest.
    """
    import shutil
    _EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(_EXEC_LOG_DIR.glob("execution_*.json"), reverse=True)
    if not files:
        return {"archived": 0, "kept": 0, "archive_dir": None}

    to_archive = files[1:] if keep_latest else files
    if not to_archive:
        return {"archived": 0, "kept": len(files), "archive_dir": None}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = _EXEC_LOG_DIR / "archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for fp in to_archive:
        try:
            shutil.move(str(fp), str(archive_dir / fp.name))
            moved += 1
        except Exception:
            pass

    return {
        "archived":    moved,
        "kept":        1 if keep_latest else 0,
        "archive_dir": str(archive_dir),
    }


# ── Dashboard summary endpoint ────────────────────────────────────────────────

@app.get("/api/v3/dashboard")
async def get_dashboard(run_id: Optional[str] = None):
    """
    Single endpoint for frontend dashboard.
    Returns: latest predictions, orders, pipeline status, sentiment health.
    No Angel One call — pure file reads, always fast.
    Pass ?run_id=<id> to view a specific run instead of the latest.
    """
    if not run_id:
        run_id = _latest_run_id()
    out: dict = {"timestamp": datetime.now().isoformat()}

    # Latest run summary
    if run_id:
        out["run_id"] = run_id
        summary_path  = _RUNS_DIR / run_id / "summary.csv"
        # Symbols permanently skipped (insufficient history or structural break)
        # SHRIRAMFIN: merged from Shriram Transport + Shriram City (2022-23) —
        #   model trains on pre-merger data, tests on different entity → 44.1% OOS, worst stock
        _PERMANENTLY_SKIPPED = {"BAJAJHFL", "SHRIRAMFIN"}

        if summary_path.exists():
            df  = pd.read_csv(summary_path)
            trained = df[df["symbol"] != "AVERAGE"]
            avg = df[df["symbol"] == "AVERAGE"]
            n_trained   = int(trained.shape[0])
            # n_total = however many were actually attempted in this run
            n_total     = n_trained  # summary only contains trained stocks; show X/X when complete
            # Check if a symbols file was saved for this run
            sym_file = _RUNS_DIR / run_id / "symbols.txt"
            if sym_file.exists():
                attempted = [l.strip() for l in sym_file.read_text().splitlines() if l.strip()]
                n_total = len(attempted)
            is_complete = n_trained >= n_total
            out["pipeline"] = {
                "n_stocks":    n_trained,
                "n_total":     n_total,
                "is_complete": is_complete,
                "avg_accuracy": round(float(avg.iloc[0]["oos_accuracy"]), 4) if not avg.empty else None,
            }

        pred_path = _RUNS_DIR / run_id / "next_day_predictions.csv"
        if pred_path.exists():
            pdf = pd.read_csv(pred_path)
            # Normalise avg_prob → prob_up
            if "avg_prob" in pdf.columns and "prob_up" not in pdf.columns:
                pdf["prob_up"] = pdf["avg_prob"]
            prob_col = "prob_up" if "prob_up" in pdf.columns else "confidence"
            up = (pdf["direction"] == "UP").sum() if "direction" in pdf.columns else 0
            out["predictions"] = {
                "total": len(pdf),
                "up": int(up),
                "down": int(len(pdf) - up),
                "top5": sanitize_dict(
                    pdf[pdf["direction"] == "UP"]
                    .sort_values(prob_col, ascending=False)
                    .head(5)[["symbol", "direction", prob_col]]
                    .rename(columns={prob_col: "prob_up"})
                    .to_dict(orient="records")
                ) if len(pdf) > 0 else [],
            }

    # Orders — prefer file tied to current run_id, then generate on-the-fly
    orders = None
    orders_source = None
    if run_id:
        run_order_files = sorted(_ORDERS_DIR.glob(f"orders_{run_id}_*.json"), reverse=True)
        if run_order_files:
            with open(run_order_files[0]) as f:
                orders = json.load(f)
            orders_source = run_order_files[0].name

    # If no run-specific file, generate orders from predictions using last_close as price
    if orders is None and run_id:
        pred_path = _RUNS_DIR / run_id / "next_day_predictions.csv"
        if pred_path.exists():
            try:
                pdf = pd.read_csv(pred_path)
                if "avg_prob" in pdf.columns and "prob_up" not in pdf.columns:
                    pdf["prob_up"] = pdf["avg_prob"]
                prob_col = "prob_up" if "prob_up" in pdf.columns else "confidence"
                up_df = pdf[pdf["direction"] == "UP"].copy()
                up_df = up_df[up_df[prob_col] >= 0.52].sort_values(prob_col, ascending=False).head(15)
                capital = 500_000.0
                MAX_POS = 0.12
                built = []
                for _, row in up_df.iterrows():
                    sym = str(row["symbol"])
                    prob = float(row.get(prob_col, 0.55))
                    price = float(row.get("last_close", 0) or row.get("price_hint", 0) or 0)
                    if price <= 0:
                        continue
                    b = 1.5
                    kf = max(0.0, min((b * prob - (1 - prob)) / b * 0.5, MAX_POS))
                    target_inr = capital * kf
                    qty = max(1, int(target_inr / price))
                    built.append({
                        "symbol": sym, "exchange": "NSE", "direction": "BUY",
                        "prob_up": round(prob, 4), "kelly_frac": round(kf, 4),
                        "target_pct": round(kf * 100, 2),
                        "target_inr": round(target_inr, 0),
                        "qty": qty, "price": round(price, 2),
                        "order_value": round(qty * price, 0),
                        "order_type": "LIMIT", "product": "CNC", "validity": "DAY",
                        "generated_at": datetime.now().isoformat(),
                    })
                if built:
                    orders = built
                    orders_source = f"generated:{run_id}"
            except Exception:
                pass

    if orders:
        out["orders"] = {
            "file": orders_source or "",
            "count": len(orders),
            "total_deployed": round(sum(o.get("order_value", 0) for o in orders), 0),
            "orders": sanitize_dict(orders[:10]),
        }
    else:
        out["orders"] = {"count": 0}

    # Sentiment health
    sent_path = _V3_ROOT / "01_data" / "news" / "sentiment_history.parquet"
    if sent_path.exists():
        try:
            sdf = pd.read_parquet(sent_path)
            latest_date = str(sdf["date"].max()) if "date" in sdf.columns else "unknown"
            out["sentiment"] = {
                "n_symbols": int(sdf["symbol"].nunique()),
                "n_dates":   int(sdf["date"].nunique()),
                "latest_date": latest_date,
            }
        except Exception:
            pass  # corrupted sentiment file — skip gracefully

    # Trade history
    if _HISTORY_PATH.exists():
        try:
            df = pd.read_parquet(_HISTORY_PATH)
            out["trade_history"] = {
                "total_trades": len(df),
                "latest_date": str(df["date"].max()) if "date" in df.columns else None,
            }
        except Exception:
            pass  # corrupted history file — skip gracefully

    return sanitize_dict(out)


# ── Angel One postback webhook ────────────────────────────────────────────────
# Register this URL in Angel One SmartAPI app dashboard:
#   Dev:  https://<ngrok-id>.ngrok-free.app/api/v3/angel/webhook
#   Prod: https://yourdomain.com/api/v3/angel/webhook

_WEBHOOK_LOG = _LIVE_DIR / "execution_logs" / "webhook_events.jsonl"


@app.post("/api/v3/angel/webhook")
async def angel_webhook(request: Request):
    """
    Receives order postback events from Angel One SmartAPI.
    Angel One POSTs JSON here when any order status changes
    (placed → pending → complete/rejected).

    Wire this URL in: SmartAPI Dashboard → My Apps → Edit App → Postback URL
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    # Log every event to JSONL for audit trail
    _WEBHOOK_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "received_at": datetime.now().isoformat(),
        "payload": payload,
    }
    with open(_WEBHOOK_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")

    # Extract key fields
    order_id = payload.get("orderid", payload.get("order_id", ""))
    status   = payload.get("orderstatus", payload.get("status", "")).lower()
    symbol   = payload.get("tradingsymbol", payload.get("symbol", ""))
    filled   = payload.get("filledshares", payload.get("filled_qty", 0))
    avg_px   = payload.get("averageprice",  payload.get("avg_price", 0))

    # Broadcast to any connected WebSocket clients (frontend gets live fill updates)
    ws_msg = {
        "type":     "order_update",
        "order_id": order_id,
        "symbol":   symbol,
        "status":   status,
        "filled":   filled,
        "avg_price": avg_px,
        "timestamp": datetime.now().isoformat(),
    }
    dead = []
    for ws in websocket_connections:
        try:
            await ws.send_json(ws_msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        websocket_connections.remove(ws)

    return {"status": "received", "order_id": order_id}


@app.get("/api/v3/angel/webhook/events")
async def get_webhook_events(limit: int = 50):
    """Return recent postback events (for debugging order fills)."""
    if not _WEBHOOK_LOG.exists():
        return {"events": [], "count": 0}
    lines = _WEBHOOK_LOG.read_text().strip().splitlines()
    events = [json.loads(l) for l in lines[-limit:]][::-1]
    return {"events": events, "count": len(lines)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
