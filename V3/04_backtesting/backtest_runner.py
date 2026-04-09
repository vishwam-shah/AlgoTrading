"""
Backtest Runner — Orchestrates full portfolio backtest from predictions.csv.

Pipeline:
  1. Load predictions from a run's {symbol}_predictions.csv files
  2. Build daily return matrix for HRP correlation clustering
  3. Run HRP portfolio optimizer → target weights
  4. Simulate portfolio with transaction costs (NSE model)
  5. Output equity_curve.csv + metrics.json

Usage:
    python backtest_runner.py --run-id 20260408_140735
    python backtest_runner.py --run-id 20260408_140735 --capital 500000
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
V3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(V3_ROOT / "00_config"))

import importlib.util


def _import(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tc_mod  = _import(str(V3_ROOT / "04_backtesting" / "transaction_costs.py"),  "transaction_costs")
_hrp_mod = _import(str(V3_ROOT / "04_backtesting" / "portfolio_optimizer.py"), "portfolio_optimizer")
_eng_mod = _import(str(V3_ROOT / "04_backtesting" / "backtest_engine.py"),     "backtest_engine")
_raw_mod = _import(str(V3_ROOT / "01_data" / "downloader.py"),                 "downloader")

TransactionCosts         = _tc_mod.TransactionCosts
HierarchicalRiskParity   = _hrp_mod.HierarchicalRiskParity
BacktestEngine           = _eng_mod.BacktestEngine

from config import RESULTS_RUNS_DIR, RAW_DATA_DIR  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────
NIFTY50_SYMBOLS = {
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    "RELIANCE", "ONGC", "COALINDIA", "NTPC", "POWERGRID",
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL",
    "BHARTIARTL", "LT", "ULTRACEMCO", "GRASIM", "ADANIPORTS",
    "TITAN", "ASIANPAINT", "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE",
    "ADANIENT", "BEL", "HAL", "TATAELXSI", "OFSS",
}


class BacktestRunner:
    """
    Orchestrates portfolio backtesting from V3 pipeline prediction outputs.

    Reads per-stock {symbol}_predictions.csv, builds daily signal matrix,
    applies HRP allocation, simulates execution with NSE transaction costs.
    """

    def __init__(
        self,
        run_id: str,
        initial_capital: float = 100_000,
        confidence_threshold: float = 0.55,
    ):
        self.run_id = run_id
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold

        self.run_dir = RESULTS_RUNS_DIR / run_id
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self.tc = TransactionCosts()
        self.hrp = HierarchicalRiskParity()
        self.engine = BacktestEngine(initial_cash=initial_capital)

    def run(self) -> Dict:
        """Full backtest pipeline. Returns summary metrics dict."""
        print(f"\n{'='*60}")
        print(f"BACKTEST — Run {self.run_id}")
        print(f"Capital: ₹{self.initial_capital:,.0f} | Confidence threshold: {self.confidence_threshold:.0%}")
        print(f"{'='*60}\n")

        # 1. Load all prediction CSVs
        signals_df = self._load_predictions()
        if signals_df.empty:
            raise ValueError("No prediction files found in run directory")

        symbols = signals_df["symbol"].unique().tolist()
        print(f"  Loaded {len(symbols)} stocks | {len(signals_df)} total predictions")

        # 2. Build return & price matrices from raw OHLCV parquets
        print("  Loading price & return data for HRP...")
        returns_matrix, self._prices_matrix = self._build_price_and_return_matrix(symbols)

        # 3. Get unique trading dates from predictions
        dates = sorted(signals_df["timestamp"].unique())
        print(f"  Trading dates: {len(dates)} ({dates[0]} → {dates[-1]})")

        # 4. Run day-by-day simulation
        # Strategy: rebalance only on signal days; hold current portfolio on quiet days
        print("\n  Simulating portfolio...")
        current_weights: Dict[str, float] = {}  # tracks active allocation

        for date in dates:
            day_signals = signals_df[signals_df["timestamp"] == date]

            # Filter to high-confidence BUY signals only
            buy_signals = day_signals[
                (day_signals["y_pred"] == 1) &
                (day_signals["y_pred_proba"] >= self.confidence_threshold)
            ]

            if buy_signals.empty:
                # No new signals: mark-to-market only, no trades
                all_syms = list(self.engine.holdings.keys())
                if not all_syms:
                    continue
                prices = self._get_prices_for_date(date, all_syms)
                self.engine.step(date, prices, {}, rebalance=False)
                continue

            active_symbols = buy_signals["symbol"].tolist()
            confidences = dict(zip(buy_signals["symbol"], buy_signals["y_pred_proba"]))

            # HRP weights using trailing 90-day returns for correlation
            lookback_end = pd.Timestamp(date).normalize()
            lookback_start = lookback_end - pd.Timedelta(days=90)
            ret_slice = returns_matrix.loc[
                (returns_matrix.index >= lookback_start) &
                (returns_matrix.index < lookback_end),
                [s for s in active_symbols if s in returns_matrix.columns]
            ].dropna(axis=1, how="all")

            valid_symbols = ret_slice.columns.tolist()
            if len(valid_symbols) >= 2:
                new_weights = self.hrp.allocate_with_confidence(
                    ret_slice, valid_symbols, confidences
                )
            elif len(valid_symbols) == 1:
                new_weights = {valid_symbols[0]: 1.0}
            else:
                # Fallback: equal weight
                new_weights = {s: 1.0 / len(active_symbols) for s in active_symbols}
                valid_symbols = active_symbols

            current_weights = new_weights

            # Get actual close prices for all positions (held + new)
            all_syms = list(set(valid_symbols + list(self.engine.holdings.keys())))
            prices = self._get_prices_for_date(date, all_syms)

            self.engine.step(date, prices, current_weights, self.tc)

        # 5. Calculate final metrics
        metrics = self.engine.calculate_metrics()
        metrics["run_id"] = self.run_id
        metrics["initial_capital"] = self.initial_capital
        metrics["confidence_threshold"] = self.confidence_threshold
        metrics["n_stocks"] = len(symbols)
        metrics["n_trading_days"] = len(dates)

        # 6. Save outputs
        self._save_outputs(metrics)

        # 7. Print summary
        self._print_summary(metrics)

        return metrics

    def _load_predictions(self) -> pd.DataFrame:
        """Load all {symbol}_predictions.csv files from run directory."""
        pred_files = list(self.run_dir.glob("*_predictions.csv"))
        dfs = []
        for f in pred_files:
            symbol = f.stem.replace("_predictions", "")
            try:
                df = pd.read_csv(f)
                df["symbol"] = symbol
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
            except Exception:
                pass

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        # Use only the final window (highest train ratio) if walk-forward
        if "window_train_ratio" in combined.columns:
            combined = combined[combined["window_train_ratio"] == combined["window_train_ratio"].max()]

        return combined.sort_values("timestamp").reset_index(drop=True)

    def _build_price_and_return_matrix(self, symbols: List[str]):
        """
        Load raw OHLCV from parquet cache.
        Returns:
          - returns_matrix: DataFrame of daily log returns (for HRP)
          - prices_matrix:  DataFrame of close prices (for backtest execution)
        """
        ret_frames = {}
        px_frames = {}

        for symbol in symbols:
            cache = Path(RAW_DATA_DIR) / f"{symbol}.parquet"
            if not cache.exists():
                continue
            try:
                df = pd.read_parquet(cache)
                date_col = "timestamp" if "timestamp" in df.columns else "date"
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                df.index = df.index.normalize()  # strip time component
                df["log_return"] = np.log(df["close"] / df["close"].shift(1))
                ret_frames[symbol] = df["log_return"].dropna()
                px_frames[symbol]  = df["close"]
            except Exception:
                pass

        returns_matrix = pd.DataFrame(ret_frames).fillna(0) if ret_frames else pd.DataFrame()
        prices_matrix  = pd.DataFrame(px_frames)             if px_frames  else pd.DataFrame()
        return returns_matrix, prices_matrix

    def _get_prices_for_date(
        self,
        date,
        symbols: List[str],
    ) -> Dict[str, float]:
        """Get actual close prices for a given date from the prices matrix."""
        ts = pd.Timestamp(date).normalize()
        prices = {}
        for sym in symbols:
            if sym in self._prices_matrix.columns:
                # Use exact date, or nearest prior trading day
                col = self._prices_matrix[sym].dropna()
                if ts in col.index:
                    prices[sym] = float(col[ts])
                else:
                    prior = col[col.index <= ts]
                    if not prior.empty:
                        prices[sym] = float(prior.iloc[-1])
                    else:
                        prices[sym] = 100.0
            else:
                prices[sym] = 100.0
        return prices

    def _save_outputs(self, metrics: Dict):
        """Save equity curve CSV and metrics JSON."""
        # Equity curve
        equity_data = [
            {"date": str(snap.date), "portfolio_value": snap.portfolio_value, "daily_return": snap.daily_return}
            for snap in self.engine.snapshots
        ]
        if equity_data:
            equity_df = pd.DataFrame(equity_data)
            equity_path = self.run_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            print(f"\n  ✓ Equity curve: {equity_path.name}")

        # Metrics JSON
        metrics_path = self.run_dir / "backtest_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  ✓ Metrics: {metrics_path.name}")

    def _print_summary(self, metrics: Dict):
        """Print formatted metrics to console."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>8.3f}")
        print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>8.3f}")
        print(f"  CAGR:                {metrics.get('cagr', 0):>8.2%}")
        print(f"  Max Drawdown:        {metrics.get('max_drawdown', 0):>8.2%}")
        print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):>8.3f}")
        print(f"  Win Rate:            {metrics.get('win_rate', 0):>8.2%}")
        print(f"  Profit Factor:       {metrics.get('profit_factor', 0):>8.2f}")
        print(f"  Total Return:        {metrics.get('total_return', 0):>8.2%}")
        print(f"  Trading Days:        {metrics.get('n_trading_days', 0):>8}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="V3 Backtest Runner")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., 20260408_140735)")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital in ₹")
    parser.add_argument(
        "--confidence", type=float, default=0.55,
        help="Min prediction confidence to trade (default 0.55)"
    )
    args = parser.parse_args()

    runner = BacktestRunner(
        run_id=args.run_id,
        initial_capital=args.capital,
        confidence_threshold=args.confidence,
    )
    runner.run()


if __name__ == "__main__":
    main()
