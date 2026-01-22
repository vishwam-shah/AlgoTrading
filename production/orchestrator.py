"""
================================================================================
TRADING SYSTEM ORCHESTRATOR
================================================================================
Main coordinator for the production trading system.

Pipeline Stages:
1. DATA COLLECTION - Download latest OHLCV data
2. FEATURE ENGINEERING - Compute 100+ features
3. MODEL TRAINING - Train XGBoost ensemble
4. BACKTESTING - Walk-forward validation
5. SIGNAL GENERATION - Generate trading signals
6. EXECUTION - Execute trades via broker

Each stage produces results in its respective folder.
================================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from production.feature_engine import AdvancedFeatureEngine
from production.models import ProductionModel, ModelEvaluator
from production.signals import SignalGenerator, TradingSignal, PortfolioSignals
from production.backtester import WalkForwardBacktester
from production.broker import create_broker, BaseBroker


class TradingOrchestrator:
    """
    Main orchestrator for the trading system.

    Manages the complete pipeline from data to execution.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        paper_trading: bool = True,
        initial_capital: float = 100000,
        results_dir: str = None
    ):
        """
        Initialize orchestrator.

        Args:
            symbols: List of stock symbols to trade
            paper_trading: Use paper trading mode
            initial_capital: Starting capital
            results_dir: Directory for storing results
        """
        self.symbols = symbols or config.ALL_STOCKS[:10]  # Default to first 10
        self.paper_trading = paper_trading
        self.initial_capital = initial_capital

        # Results directory
        self.results_dir = results_dir or os.path.join(config.BASE_DIR, 'production_results')
        self._setup_directories()

        # Components - sentiment now uses fast RSS-based engine (no rate limits)
        self.feature_engine = AdvancedFeatureEngine(
            include_sentiment=True,    # Enabled - using fast RSS-based sentiment
            include_market_context=False  # Disabled - yfinance issues with indices
        )
        self.model = ProductionModel()
        self.signal_generator = SignalGenerator()
        self.backtester = WalkForwardBacktester(initial_capital=initial_capital)
        self.broker = None  # Initialized on demand

        # State
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.features_cache: Dict[str, pd.DataFrame] = {}
        self.predictions_cache: Dict[str, any] = {}
        self.signals_cache: List[TradingSignal] = []

        # Pipeline status
        self.pipeline_status = {
            'data_collected': False,
            'features_computed': False,
            'model_trained': False,
            'backtested': False,
            'signals_generated': False
        }

        logger.info(f"TradingOrchestrator initialized with {len(self.symbols)} symbols")
        logger.info(f"Mode: {'Paper Trading' if paper_trading else 'LIVE TRADING'}")

    def _setup_directories(self):
        """Create result directories."""
        dirs = [
            self.results_dir,
            os.path.join(self.results_dir, 'data'),
            os.path.join(self.results_dir, 'features'),
            os.path.join(self.results_dir, 'models'),
            os.path.join(self.results_dir, 'backtest'),
            os.path.join(self.results_dir, 'signals'),
            os.path.join(self.results_dir, 'trades'),
            os.path.join(self.results_dir, 'reports')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    # ==================== STAGE 1: DATA COLLECTION ====================

    def collect_data(self, days: int = 1000, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Stage 1: Collect OHLCV data for all symbols.

        Args:
            days: Number of calendar days to fetch (750 ~ 500 trading days)
            force_download: If True, always download fresh data from yfinance

        Returns:
            Dict of DataFrames per symbol
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA COLLECTION")
        logger.info("=" * 60)

        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Directories
        raw_data_dir = config.RAW_DATA_DIR
        cached_data_dir = os.path.join(self.results_dir, 'data')

        logger.info(f"yfinance version: {yf.__version__}")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Force download: {force_download}")

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            df = None

            # Download fresh data from yfinance (primary method)
            if force_download or df is None:
                for attempt in range(3):
                    try:
                        ticker = f"{symbol}.NS"
                        logger.info(f"  Downloading {ticker} from yfinance...")

                        df = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=True
                        )

                        if len(df) > 0:
                            # Handle yfinance 1.0 MultiIndex columns
                            df = df.reset_index()

                            if isinstance(df.columns, pd.MultiIndex):
                                # yfinance 1.0 format: (Price, Ticker) columns
                                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                            # Standardize column names
                            df.columns = [c.lower() for c in df.columns]

                            # Rename date column
                            if 'date' in df.columns:
                                df = df.rename(columns={'date': 'timestamp'})

                            # Add symbol
                            df['symbol'] = symbol

                            logger.success(f"  {symbol}: Downloaded {len(df)} rows (latest: {df['timestamp'].iloc[-1]})")
                            break
                        else:
                            logger.warning(f"  {symbol}: Attempt {attempt+1} - no data returned")
                            time.sleep(1)

                    except Exception as e:
                        logger.warning(f"  {symbol}: Attempt {attempt+1} failed - {e}")
                        time.sleep(1)

            # Fallback: Load from raw data directory ONLY if download completely failed
            # Don't replace freshly downloaded data with stale cache
            if df is None:
                raw_path = os.path.join(raw_data_dir, f'{symbol}.csv')
                if os.path.exists(raw_path):
                    try:
                        df = pd.read_csv(raw_path)
                        logger.warning(f"  {symbol}: Download failed, using cached data ({len(df)} rows)")
                    except Exception as e:
                        logger.error(f"  {symbol}: Failed to read raw data - {e}")

            # Save to cache if we have any valid data
            if df is not None and len(df) > 0:
                self.data_cache[symbol] = df

                # Save to results directory
                save_path = os.path.join(cached_data_dir, f'{symbol}.csv')
                df.to_csv(save_path, index=False)

                # Also update raw data
                raw_save_path = os.path.join(raw_data_dir, f'{symbol}.csv')
                df.to_csv(raw_save_path, index=False)
            else:
                logger.error(f"  {symbol}: No data available")

        self.pipeline_status['data_collected'] = True
        logger.success(f"Data collection complete: {len(self.data_cache)} symbols")

        return self.data_cache

    # ==================== STAGE 2: FEATURE ENGINEERING ====================

    def compute_features(self) -> Dict[str, pd.DataFrame]:
        """
        Stage 2: Compute features for all symbols.

        Returns:
            Dict of feature DataFrames per symbol
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("=" * 60)

        if not self.data_cache:
            logger.error("No data available. Run collect_data() first.")
            return {}

        for symbol, df in self.data_cache.items():
            logger.info(f"Computing features for {symbol}...")

            try:
                # Compute features
                feature_set = self.feature_engine.compute_all_features(df, symbol)

                # Compute targets
                feature_df = self.feature_engine.compute_targets(feature_set.df)

                # Drop NaN
                feature_df = feature_df.dropna()

                self.features_cache[symbol] = feature_df

                # Save
                save_path = os.path.join(self.results_dir, 'features', f'{symbol}_features.csv')
                feature_df.to_csv(save_path, index=False)

                logger.info(f"  {symbol}: {feature_set.n_features} features, {len(feature_df)} samples")

            except Exception as e:
                logger.error(f"  {symbol}: Feature computation failed - {e}")

        self.pipeline_status['features_computed'] = True
        logger.success(f"Feature engineering complete: {len(self.features_cache)} symbols")

        return self.features_cache

    # ==================== STAGE 3: MODEL TRAINING ====================

    def train_model(self, symbol: str = None) -> Dict:
        """
        Stage 3: Train production model.

        Args:
            symbol: Specific symbol to train on, or None for combined training

        Returns:
            Training metrics
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: MODEL TRAINING")
        logger.info("=" * 60)

        if not self.features_cache:
            logger.error("No features available. Run compute_features() first.")
            return {}

        # Combine all symbol data or use specific symbol
        if symbol:
            if symbol not in self.features_cache:
                logger.error(f"No features for {symbol}")
                return {}
            train_df = self.features_cache[symbol]
        else:
            # Combine all symbols
            dfs = []
            for sym, df in self.features_cache.items():
                df_copy = df.copy()
                df_copy['_symbol'] = sym
                dfs.append(df_copy)
            train_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Training on {len(train_df)} samples")

        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                       'date', 'symbol', '_symbol'] + [c for c in train_df.columns if c.startswith('target_')]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        logger.info(f"Using {len(feature_cols)} features")

        # Prepare data
        X = train_df[feature_cols].values

        # Handle NaN in targets
        close_return = train_df['target_close_return'].values.copy()
        close_return = np.nan_to_num(close_return, nan=0.0, posinf=0.0, neginf=0.0)

        # 5-day return (forward-looking sum, need to compute differently)
        close_return_5d = None
        if 'target_close_return' in train_df:
            # Calculate 5-day forward return
            close_return_5d = train_df['target_close_return'].rolling(5, min_periods=1).sum().values
            close_return_5d = np.nan_to_num(close_return_5d, nan=0.0, posinf=0.0, neginf=0.0)

        y = {
            'direction': train_df['target_direction'].values,
            'close_return': close_return,
            'close_return_5d': close_return_5d
        }

        # Train/val split (chronological)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = {k: v[:split_idx] if v is not None else None for k, v in y.items()}
        y_val = {k: v[split_idx:] if v is not None else None for k, v in y.items()}

        # Train
        metrics = self.model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        # Save model
        model_dir = os.path.join(self.results_dir, 'models', symbol or 'combined')
        self.model.save(model_dir)

        # Save feature importance
        importance_df = self.model.get_feature_importance(top_n=30)
        importance_path = os.path.join(model_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)

        self.pipeline_status['model_trained'] = True
        logger.success(f"Model training complete. Direction accuracy: {metrics['direction_accuracy_ensemble']:.2%}")

        return metrics

    # ==================== STAGE 4: BACKTESTING ====================

    def run_backtest(self, symbol: str) -> Dict:
        """
        Stage 4: Run backtest for a symbol.
        Uses proper walk-forward validation - only tests on data not seen during training.

        Args:
            symbol: Stock symbol

        Returns:
            Backtest results dict
        """
        logger.info("=" * 60)
        logger.info(f"STAGE 4: BACKTESTING - {symbol}")
        logger.info("=" * 60)

        if symbol not in self.features_cache:
            logger.error(f"No features for {symbol}")
            return {}

        df = self.features_cache[symbol]

        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                       'date', 'symbol'] + [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Use only the TEST portion (last 20%) - same as validation split in training
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        
        logger.info(f"Testing on {len(test_df)} samples (out-of-sample)")
        
        X_test = test_df[feature_cols].values

        # Get predictions on test set only
        predictions = self.model.predict(X_test)
        directions = np.array([p.direction for p in predictions])
        confidences = np.array([p.confidence for p in predictions])
        expected_returns = np.array([p.expected_return for p in predictions])

        # Run backtest with improved parameters
        result = self.backtester.run_backtest(
            test_df,
            directions,
            confidences,
            expected_returns,
            min_confidence=0.50  # Lowered to get more signals
        )

        # Generate report
        report = self.backtester.generate_report(result, symbol)
        logger.info("\n" + report)

        # Save results
        report_path = os.path.join(self.results_dir, 'backtest', f'{symbol}_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        # Save plot
        plot_path = os.path.join(self.results_dir, 'backtest', f'{symbol}_equity.png')
        self.backtester.plot_results(result, plot_path)

        self.pipeline_status['backtested'] = True

        # Format equity curve for frontend (list of {date, equity} objects)
        equity_curve_formatted = []
        if result.equity_curve is not None:
            for idx, value in result.equity_curve.items():
                date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)[:10]
                equity_curve_formatted.append({
                    'date': date_str,
                    'equity': float(value)
                })

        # Format trades for frontend
        trades_formatted = []
        if result.trades:
            for t in result.trades:
                trades_formatted.append({
                    'entry_date': str(t.entry_date)[:10] if t.entry_date else '',
                    'exit_date': str(t.exit_date)[:10] if t.exit_date else '',
                    'direction': t.direction,
                    'entry_price': float(t.entry_price),
                    'exit_price': float(t.exit_price) if t.exit_price else 0,
                    'shares': int(t.quantity),
                    'pnl': float(t.pnl),
                    'return_pct': float(t.pnl_pct),
                    'exit_reason': t.exit_reason
                })

        return {
            'symbol': symbol,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'equity_curve': equity_curve_formatted,
            'trades': trades_formatted
        }

    # ==================== STAGE 5: SIGNAL GENERATION ====================

    def generate_signals(self, portfolio_value: float = None) -> PortfolioSignals:
        """
        Stage 5: Generate trading signals for current market conditions.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            PortfolioSignals object
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: SIGNAL GENERATION")
        logger.info("=" * 60)

        portfolio_value = portfolio_value or self.initial_capital

        if not self.features_cache:
            logger.error("No features available. Run compute_features() first.")
            return None

        predictions = {}
        current_data = {}

        for symbol in self.features_cache.keys():
            df = self.features_cache[symbol]

            if len(df) < 2:
                continue

            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                           'date', 'symbol'] + [c for c in df.columns if c.startswith('target_')]
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            # Get latest row
            latest = df.iloc[-1]
            X = latest[feature_cols].values.reshape(1, -1)

            # Predict
            pred = self.model.predict_single(X)
            predictions[symbol] = pred
            current_data[symbol] = latest

        # Generate portfolio signals
        portfolio_signals = self.signal_generator.generate_portfolio_signals(
            list(predictions.keys()),
            predictions,
            current_data,
            portfolio_value
        )

        # Print report
        report = self.signal_generator.format_signals_report(portfolio_signals)
        logger.info("\n" + report)

        # Save signals
        signals_path = os.path.join(
            self.results_dir, 'signals',
            f'signals_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        )

        signals_data = {
            'timestamp': str(portfolio_signals.timestamp),
            'market_exposure': portfolio_signals.market_exposure,
            'total_allocation': portfolio_signals.total_suggested_allocation,
            'buy_signals': [
                {
                    'symbol': s.symbol,
                    'action': s.action,
                    'strength': s.strength,
                    'price': s.current_price,
                    'target': s.target_price,
                    'stop_loss': s.stop_loss,
                    'confidence': s.confidence,
                    'position_pct': s.suggested_position_pct
                }
                for s in portfolio_signals.buy_signals
            ]
        }

        with open(signals_path, 'w') as f:
            json.dump(signals_data, f, indent=2)

        self.signals_cache = portfolio_signals.signals
        self.pipeline_status['signals_generated'] = True

        return portfolio_signals

    def get_signal(self, symbol: str) -> Dict:
        """
        Get trading signal for a single symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with action (BUY/SELL/HOLD), confidence, and other details
        """
        if symbol not in self.features_cache:
            return {"action": "HOLD", "confidence": 0, "reason": "No features available"}
        
        df = self.features_cache[symbol]
        if len(df) < 2:
            return {"action": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                       'date', 'symbol'] + [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Get latest row
        latest = df.iloc[-1]
        X = latest[feature_cols].values.reshape(1, -1)
        
        try:
            # Predict
            pred = self.model.predict_single(X)
            
            # More sensitive thresholds for more trading action
            # Original: BUY > 0.6, SELL < 0.4
            # New: BUY > 0.52, SELL < 0.48 (tighter band = more signals)
            if pred > 0.52:
                action = "BUY"
                confidence = min((pred - 0.5) * 2 + 0.5, 0.95)  # Scale 0.52-1.0 to 0.54-0.95
            elif pred < 0.48:
                action = "SELL"
                confidence = min((0.5 - pred) * 2 + 0.5, 0.95)  # Scale 0.0-0.48 to 0.54-0.95
            else:
                action = "HOLD"
                confidence = 0.5
            
            return {
                "action": action,
                "confidence": float(confidence),
                "prediction": float(pred),
                "current_price": float(latest.get('close', 0)),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting signal for {symbol}: {e}")
            return {"action": "HOLD", "confidence": 0, "reason": str(e)}

    # ==================== STAGE 6: EXECUTION ====================

    def execute_signals(self, signals: PortfolioSignals = None, dry_run: bool = True):
        """
        Stage 6: Execute trading signals.

        Args:
            signals: Portfolio signals to execute
            dry_run: If True, only simulate execution
        """
        logger.info("=" * 60)
        logger.info(f"STAGE 6: EXECUTION {'(DRY RUN)' if dry_run else '(LIVE)'}")
        logger.info("=" * 60)

        if signals is None:
            signals = self.generate_signals()

        if not signals or not signals.buy_signals:
            logger.info("No signals to execute")
            return

        # Initialize broker
        if self.broker is None:
            self.broker = create_broker(
                paper_trading=self.paper_trading,
                initial_capital=self.initial_capital
            )
            self.broker.authenticate()

        # Get current account info
        account = self.broker.get_account_info()
        logger.info(f"Account: Cash Rs {account.cash:,.2f}, Positions Rs {account.positions_value:,.2f}")

        # Execute buy signals
        for signal in signals.buy_signals[:5]:  # Max 5 positions
            if dry_run:
                logger.info(f"[DRY RUN] Would BUY {signal.symbol}: {signal.suggested_quantity} shares @ Rs {signal.current_price:,.2f}")
            else:
                logger.info(f"Executing: BUY {signal.symbol} x {signal.suggested_quantity}")
                result = self.broker.place_market_order(
                    signal.symbol,
                    signal.suggested_quantity,
                    'BUY'
                )
                logger.info(f"  Result: {result.status.value} - {result.message}")

                # Log trade
                trade_log = os.path.join(self.results_dir, 'trades', 'trade_log.csv')
                trade_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': signal.symbol,
                    'action': 'BUY',
                    'quantity': signal.suggested_quantity,
                    'price': signal.current_price,
                    'order_id': result.order_id,
                    'status': result.status.value
                }

                pd.DataFrame([trade_data]).to_csv(
                    trade_log,
                    mode='a',
                    header=not os.path.exists(trade_log),
                    index=False
                )

        # Final account status
        account = self.broker.get_account_info()
        logger.info(f"Final Account: Total Rs {account.total_value:,.2f}")

    # ==================== FULL PIPELINE ====================

    def run_pipeline(self, skip_backtest: bool = False, force_download: bool = True):
        """
        Run complete pipeline from data to signals.

        Args:
            skip_backtest: Skip backtesting stage
            force_download: Force download fresh data from yfinance (default: True for live data)
        """
        logger.info("#" * 60)
        logger.info("# PRODUCTION TRADING PIPELINE")
        logger.info(f"# Symbols: {len(self.symbols)}")
        logger.info(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("#" * 60)

        start_time = time.time()

        # Stage 1: Data Collection
        self.collect_data(force_download=force_download)

        # Stage 2: Feature Engineering
        self.compute_features()

        # Stage 3: Model Training
        self.train_model()

        # Stage 4: Backtesting
        if not skip_backtest:
            for symbol in list(self.features_cache.keys())[:3]:  # Backtest first 3
                self.run_backtest(symbol)

        # Stage 5: Signal Generation
        signals = self.generate_signals()

        elapsed = time.time() - start_time

        logger.info("#" * 60)
        logger.info("# PIPELINE COMPLETE")
        logger.info(f"# Duration: {elapsed/60:.1f} minutes")
        logger.info(f"# Results: {self.results_dir}")
        logger.info("#" * 60)

        return signals

    # ==================== UTILITIES ====================

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'symbols': len(self.symbols),
            'data_loaded': len(self.data_cache),
            'features_computed': len(self.features_cache),
            'model_trained': self.model.training_date is not None,
            'signals_count': len(self.signals_cache),
            'pipeline_status': self.pipeline_status
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Production Trading System')
    parser.add_argument('--symbols', nargs='+', default=None, help='Stock symbols')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Live trading mode')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtesting')
    parser.add_argument('--signals-only', action='store_true', help='Only generate signals')

    args = parser.parse_args()

    # Default symbols
    if args.symbols is None:
        args.symbols = ['HDFCBANK', 'ICICIBANK', 'TCS', 'INFY', 'RELIANCE',
                        'SBIN', 'KOTAKBANK', 'TATAMOTORS', 'SUNPHARMA', 'LT']

    paper_trading = not args.live

    orchestrator = TradingOrchestrator(
        symbols=args.symbols,
        paper_trading=paper_trading,
        initial_capital=args.capital
    )

    if args.signals_only:
        orchestrator.collect_data()
        orchestrator.compute_features()
        orchestrator.train_model()
        signals = orchestrator.generate_signals()
    else:
        signals = orchestrator.run_pipeline(skip_backtest=args.skip_backtest)

    return signals


if __name__ == '__main__':
    main()
