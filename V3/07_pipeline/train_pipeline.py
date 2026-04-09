"""
V3 Training Pipeline — Download, Feature, Train, Evaluate on 100 stocks.
Calculates next-day directional accuracy using XGBoost + LightGBM.

Usage:
    python train_pipeline.py --symbols SBIN HDFCBANK AXISBANK --fresh
    python train_pipeline.py --all-symbols
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Setup paths
V3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(V3_ROOT / "02_models"))

# Dynamic imports to avoid Python naming issues with numbered folders
import importlib.util

def _import_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

downloader_mod = _import_from_path(str(V3_ROOT / "01_data" / "downloader.py"), "data_downloader")
features_mod = _import_from_path(str(V3_ROOT / "01_data" / "features.py"), "data_features")
targets_mod = _import_from_path(str(V3_ROOT / "01_data" / "targets.py"), "data_targets")
metrics_mod = _import_from_path(str(V3_ROOT / "03_training" / "metrics.py"), "train_metrics")
reporting_mod = _import_from_path(str(V3_ROOT / "03_training" / "reporting.py"), "train_reporting")
sklearn_mod = _import_from_path(str(V3_ROOT / "02_models" / "traditional" / "sklearn_classifier.py"), "sklearn_clf")

# Import config and logging
from config_v3 import RAW_DATA_DIR, FEAT_SCALED_DIR, RESULTS_RUNS_DIR, SYMBOLS_100

# Import logging setup
logging_config_module = _import_from_path(str(V3_ROOT / "00_config" / "logging_config.py"), "logging_config_module")
setup_logging = logging_config_module.setup_logging

DataDownloader = downloader_mod.DataDownloader
FeatureEngineer = features_mod.FeatureEngineer
TargetComputer = targets_mod.TargetComputer
AccuracyMetrics = metrics_mod.AccuracyMetrics
TrainingReporter = reporting_mod.TrainingReporter

# Use sklearn classifiers (no libomp required)
GradientBoostingClassifier = sklearn_mod.SKLearnGradientBoostingClassifier
RandomForestClassifier = sklearn_mod.SKLearnRandomForestClassifier

warnings.filterwarnings("ignore")


class V3TrainingPipeline:
    """Main training orchestrator for 100 stocks."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = RESULTS_RUNS_DIR / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = DataDownloader(RAW_DATA_DIR)
        self.feature_engineer = FeatureEngineer()

        self.results: List[Dict] = []
        self.metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "symbols_trained": [],
            "symbols_failed": [],
        }

    def run_full_pipeline(
        self,
        symbols: List[str],
        fresh: bool = False,
        test_size: int = 100,
        verbose: bool = True,
    ):
        """
        Run full pipeline: download → features → train → evaluate.

        Args:
            symbols: List of stock symbols to process
            fresh: Force download new data (ignore cache)
            test_size: Number of recent samples for testing
            verbose: Print progress
        """
        logger.info(f"V3 TRAINING PIPELINE — {self.run_id}")
        logger.info(f"Symbols: {len(symbols)} | Fresh: {fresh} | Test size: {test_size}")
        print(f"\n{'='*70}\nV3 TRAINING PIPELINE — {self.run_id}\nSymbols: {len(symbols)}\n{'='*70}\n")

        successful = 0
        failed = 0

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol:12s} ... ", end="", flush=True)

            try:
                self._process_stock(symbol, fresh=fresh, test_size=test_size)
                successful += 1
                self.metadata["symbols_trained"].append(symbol)
                print("✓")
                logger.info(f"✓ {symbol}")
            except Exception as e:
                failed += 1
                self.metadata["symbols_failed"].append({"symbol": symbol, "error": str(e)})
                print(f"✗ {e}")
                logger.error(f"✗ {symbol}: {e}")

        # Summary
        logger.info(f"COMPLETED: {successful} successful, {failed} failed out of {len(symbols)}")
        logger.info(f"Results saved to: {self.results_dir}")
        print(f"\n{'='*70}\nCOMPLETED: {successful} successful, {failed} failed\nResults: {self.results_dir}\n{'='*70}\n")

        self._save_results()

    def _process_stock(self, symbol: str, fresh: bool = False, test_size: int = 100):
        """
        Process single stock with walk-forward expanding-window validation.

        Walk-forward protocol (7 windows):
          Window 1: Train on 70% → test on next 100 days
          Window 2: Train on 75% → test on next 100 days
          ...
          Window 7: Train on 95% → test on next 100 days
          Final metrics = average across all windows (robust OOS)
        """

        # 1. Download data (incremental — only fetches new data since last download)
        df = self.downloader.download_incremental(
            symbol,
            data_start_date="2018-01-01",
        )
        if df is None or len(df) < 400:
            raise ValueError(f"Insufficient data: {len(df) if df is not None else 0} rows")

        # 2. Compute targets
        df = TargetComputer.compute_direction_target(df)
        df = TargetComputer.compute_next_day_return(df)

        # 3. Compute features
        df = self.feature_engineer.compute_features(df)
        if len(df) < 400:
            raise ValueError(f"Insufficient data after features: {len(df)} rows")

        X = df[self.feature_engineer.feature_names].values
        y = df["direction_target"].values
        returns = df["next_day_log_return"].values
        timestamps = df["timestamp"].values
        n = len(df)

        # 4. Walk-forward expanding window setup
        # 7 windows: train ratios 70%, 75%, 80%, 85%, 90%, 92.5%, 95%
        train_ratios = [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95]
        test_size = min(test_size, max(30, n // 20))  # cap at 5% of data, min 30 samples

        window_metrics: List[Dict] = []
        all_preds = []  # collect predictions from all windows

        for ratio in train_ratios:
            train_end = int(n * ratio)
            test_start = train_end
            test_end = min(test_start + test_size, n)

            # Skip window if test set too small
            if test_end - test_start < 20:
                continue

            X_train, X_test = X[:train_end], X[test_start:test_end]
            y_train, y_test = y[:train_end], y[test_start:test_end]
            returns_test = returns[test_start:test_end]
            ts_test = timestamps[test_start:test_end]

            # Scale — fit ONLY on train, transform both (leakage-safe)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble
            gb_model = GradientBoostingClassifier()
            rf_model = RandomForestClassifier()
            gb_model.train(X_train_scaled, y_train, verbose=False)
            rf_model.train(X_train_scaled, y_train, verbose=False)

            # Ensemble predictions
            gb_proba = gb_model.predict_proba(X_test_scaled)
            rf_proba = rf_model.predict_proba(X_test_scaled)
            ensemble_proba = (gb_proba + rf_proba) / 2
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)

            # Per-window metrics
            w_metrics = AccuracyMetrics.generate_report(
                symbol, y_test, ensemble_pred, ensemble_proba, returns_test
            )
            window_metrics.append(w_metrics)

            # Collect predictions for output CSV
            for j in range(len(y_test)):
                all_preds.append({
                    "timestamp": ts_test[j],
                    "window_train_ratio": ratio,
                    "y_true": int(y_test[j]),
                    "y_pred": int(ensemble_pred[j]),
                    "y_pred_proba": float(ensemble_proba[j]),
                    "log_return": float(returns_test[j]),
                })

        if not window_metrics:
            raise ValueError("No valid walk-forward windows — too little data")

        # 5. Aggregate metrics across all windows (mean and std)
        metric_keys = [k for k in window_metrics[0].keys() if k != "symbol"]
        avg_metrics: Dict = {"symbol": symbol}
        for k in metric_keys:
            vals = [w[k] for w in window_metrics if isinstance(w.get(k), (int, float))]
            if vals:
                avg_metrics[k] = float(np.mean(vals))
                avg_metrics[f"{k}_std"] = float(np.std(vals))

        avg_metrics["n_windows"] = len(window_metrics)
        logger.info(
            f"{symbol}: {len(window_metrics)} windows | "
            f"acc={avg_metrics.get('accuracy', 0):.2f}% ± {avg_metrics.get('accuracy_std', 0):.2f}% | "
            f"pf={avg_metrics.get('profit_factor', 0):.2f}"
        )

        self.results.append(avg_metrics)

        # 6. Save all predictions (all windows combined)
        pred_df = pd.DataFrame(all_preds)
        pred_file = self.results_dir / f"{symbol}_predictions.csv"
        pred_df.to_csv(pred_file, index=False)

        # 7. Save per-window breakdown
        window_df = pd.DataFrame(window_metrics)
        window_file = self.results_dir / f"{symbol}_windows.csv"
        window_df.to_csv(window_file, index=False)

    def _save_results(self):
        """Save results and metadata with comprehensive reporting."""
        if len(self.results) == 0:
            print(f"⚠ No results to save")
            return

        results_df = pd.DataFrame(self.results)

        # Calculate summary statistics (walk-forward averaged)
        summary = {
            "total_stocks": len(self.results),
            "validation": "walk_forward_expanding_window",
            "n_windows_per_stock": int(results_df["n_windows"].mean()) if "n_windows" in results_df else 1,
            "avg_accuracy": float(results_df["accuracy"].mean()),
            "median_accuracy": float(results_df["accuracy"].median()),
            "min_accuracy": float(results_df["accuracy"].min()),
            "max_accuracy": float(results_df["accuracy"].max()),
            "avg_accuracy_std": float(results_df["accuracy_std"].mean()) if "accuracy_std" in results_df else 0.0,
            "avg_profit_factor": float(results_df["profit_factor"].replace([np.inf, -np.inf], 0).mean()),
            "avg_win_rate": float(results_df["win_rate"].mean()),
            "above_50_percent": int((results_df["accuracy"] >= 50).sum()),
            "above_52_percent": int((results_df["accuracy"] >= 52).sum()),
        }

        # Generate reports using reporter
        reporter = TrainingReporter(self.results_dir, self.run_id)

        # Save detailed CSV
        summary_csv = self.results_dir / "results_detailed.csv"
        reporter.generate_summary_csv(self.results, summary_csv)

        # Generate Markdown report
        markdown_file = self.results_dir / "REPORT.md"
        reporter.generate_markdown_report(self.results, summary, markdown_file)

        # Try to generate Excel (requires openpyxl)
        try:
            excel_file = self.results_dir / "results_report.xlsx"
            reporter.generate_excel_report(self.results, summary, excel_file)
            print(f"  ✓ Excel report: {excel_file.name}")
        except Exception as e:
            print(f"  ⚠ Excel report skipped: {e}")

        # Save metadata
        self.metadata["summary"] = summary
        metadata_file = self.results_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Log summary statistics
        logger.info(f"📊 SUMMARY STATISTICS — Avg Accuracy: {summary['avg_accuracy']:.2f}%")
        logger.info(f"  Median: {summary['median_accuracy']:.2f}% | Range: {summary['min_accuracy']:.2f}%-{summary['max_accuracy']:.2f}%")
        logger.info(f"  Stocks >50%: {summary['above_50_percent']}/{summary['total_stocks']} | >52%: {summary['above_52_percent']}/{summary['total_stocks']}")
        logger.info(f"  Profit Factor: {summary['avg_profit_factor']:.2f} | Win Rate: {summary['avg_win_rate']:.2f}%")

        # Print summary to console
        n_win = summary.get("n_windows_per_stock", 1)
        print("\n📊 WALK-FORWARD SUMMARY")
        print(f"  Validation: {n_win}-window expanding walk-forward")
        print(f"  Avg Accuracy: {summary['avg_accuracy']:.2f}% ± {summary.get('avg_accuracy_std', 0):.2f}%")
        print(f"  Median Accuracy: {summary['median_accuracy']:.2f}%")
        print(f"  Range: {summary['min_accuracy']:.2f}% - {summary['max_accuracy']:.2f}%")
        print(f"  Stocks with >50%: {summary['above_50_percent']}/{summary['total_stocks']}")
        print(f"  Stocks with >52%: {summary['above_52_percent']}/{summary['total_stocks']}")
        print(f"  Avg Profit Factor: {summary['avg_profit_factor']:.2f}")
        print(f"  Avg Win Rate: {summary['avg_win_rate']:.2f}%")

        print(f"\n✅ RESULTS SAVED")
        print(f"  Location: {self.results_dir}")
        print(f"  Detailed CSV: results_detailed.csv")
        print(f"  Markdown Report: REPORT.md")
        print(f"  Metadata: metadata.json")

        logger.info(f"✅ RESULTS SAVED | Location: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description="V3 Training Pipeline")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to train (e.g., SBIN HDFCBANK AXISBANK)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Train on all 100 stocks",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force download new data (ignore cache)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of samples for test set",
    )

    args = parser.parse_args()

    # Determine symbols
    if args.all_symbols:
        symbols = SYMBOLS_100
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default: first 10 stocks
        symbols = SYMBOLS_100[:10]

    # Run pipeline (setup_logging happens inside)
    pipeline = V3TrainingPipeline()
    setup_logging(pipeline.run_id, pipeline.results_dir)
    pipeline.run_full_pipeline(
        symbols=symbols,
        fresh=args.fresh,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
