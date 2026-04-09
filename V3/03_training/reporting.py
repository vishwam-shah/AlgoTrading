"""
Comprehensive reporting for training runs.
Generates Excel reports, plots, and detailed metrics for each stock.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np


class TrainingReporter:
    """Generate comprehensive reports for training runs."""

    def __init__(self, results_dir: Path, run_id: str):
        self.results_dir = Path(results_dir)
        self.run_id = run_id

    def generate_stock_report(
        self,
        symbol: str,
        results: Dict,
        predictions_df: pd.DataFrame,
    ) -> Dict:
        """Generate detailed report for a single stock."""
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "directional_accuracy": results.get("accuracy", 0.0),
            "win_rate": results.get("win_rate", 0.0),
            "precision_bullish": results.get("precision_bullish", 0.0),
            "recall_bullish": results.get("recall_bullish", 0.0),
            "profit_factor": results.get("profit_factor", 0.0),
            "total_trades": results.get("sample_count", 0),
            "bullish_signals": results.get("bullish_pred", 0),
            "actual_bullish": results.get("bullish_true", 0),
            "metrics_summary": {
                "accuracy_vs_random": round(results.get("accuracy", 0.0) - 50, 2),
                "profitable": "Yes" if results.get("profit_factor", 0.0) > 1.0 else "No",
                "consistency": "Positive bias" if results.get("bullish_pred", 0) > results.get("bullish_true", 0) else "Negative bias",
            },
        }

        # Add predictions stats
        if len(predictions_df) > 0:
            correct = (predictions_df["y_true"] == predictions_df["y_pred"]).sum()
            report["correct_predictions"] = int(correct)
            report["total_predictions"] = len(predictions_df)

        return report

    def generate_summary_csv(self, all_results: List[Dict], output_file: Path):
        """Generate summary CSV with all stocks."""
        df = pd.DataFrame(all_results)
        sort_col = "directional_accuracy" if "directional_accuracy" in df.columns else "accuracy"
        df = df.sort_values(sort_col, ascending=False)
        df.to_csv(output_file, index=False)
        return df

    def generate_excel_report(
        self,
        all_results: List[Dict],
        summary_stats: Dict,
        output_file: Path,
    ):
        """Generate comprehensive Excel report."""
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Summary sheet
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Detailed results
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values("directional_accuracy", ascending=False)
            results_df.to_excel(writer, sheet_name="All Results", index=False)

            # Top performers
            top_5 = results_df.head(5)
            top_5.to_excel(writer, sheet_name="Top 5", index=False)

            # Bottom performers
            bottom_5 = results_df.tail(5)
            bottom_5.to_excel(writer, sheet_name="Bottom 5", index=False)

    def generate_markdown_report(
        self,
        all_results: List[Dict],
        summary_stats: Dict,
        output_file: Path,
    ):
        """Generate Markdown report for easy reading."""
        lines = [
            "# V3 Training Pipeline Report",
            f"\nRun ID: {self.run_id}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary Statistics",
        ]

        # Summary
        for key, val in summary_stats.items():
            if isinstance(val, float) and not key.startswith("above"):
                lines.append(f"- **{key}**: {val:.2f}")
            elif key.startswith("above"):
                lines.append(f"- **{key}**: {val}")
            else:
                lines.append(f"- **{key}**: {val}")

        # Detailed results
        lines.append("\n## Stock-by-Stock Results")
        results_df = pd.DataFrame(all_results)
        sort_col = "directional_accuracy" if "directional_accuracy" in results_df.columns else "accuracy"
        results_df = results_df.sort_values(sort_col, ascending=False)

        for _, row in results_df.iterrows():
            lines.append(f"\n### {row['symbol']}")
            acc_col = "directional_accuracy" if "directional_accuracy" in row else "accuracy"
            lines.append(f"- Directional Accuracy: {row[acc_col]:.2f}%")
            lines.append(f"- Win Rate: {row['win_rate']:.2f}%")
            lines.append(f"- Profit Factor: {row['profit_factor']:.2f}")
            lines.append(f"- Precision (Bullish): {row['precision_bullish']:.2f}%")
            lines.append(f"- Recall (Bullish): {row['recall_bullish']:.2f}%")
            lines.append(f"- Total Trades: {row['sample_count']}")

        # Recommendations
        lines.append("\n## Recommendations")
        acc_col = "directional_accuracy" if "directional_accuracy" in results_df.columns else "accuracy"
        above_52 = len(results_df[results_df[acc_col] > 52])
        lines.append(f"- Stocks with >52% accuracy: {above_52}/{len(results_df)}")
        lines.append(f"- Average edge: {summary_stats.get('avg_accuracy', 0) - 50:.2f}%")

        report_text = "\n".join(lines)
        output_file.write_text(report_text)
        return report_text
