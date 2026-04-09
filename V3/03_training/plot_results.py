"""
Publication-quality plots for V3 pipeline results.

Generates:
  1. equity_curve.png       — Portfolio vs Nifty50 buy-and-hold
  2. sector_accuracy.png    — Accuracy bubble chart by sector
  3. accuracy_distribution.png — Histogram of per-stock accuracy
  4. profit_factor_vs_accuracy.png — Scatter: PF vs accuracy

Usage:
    python plot_results.py --run-id 20260408_140735
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ────────────────────────────────────────────────────────────────
V3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_ROOT / "00_config"))
from config import RESULTS_RUNS_DIR, SECTOR_MAP  # noqa: E402

PLOT_DPI = 130
STYLE = "seaborn-v0_8-darkgrid"
SECTOR_COLORS = {
    "banking":  "#2196F3",
    "IT":       "#4CAF50",
    "auto":     "#FF9800",
    "fmcg":     "#9C27B0",
    "pharma":   "#F44336",
    "energy":   "#795548",
    "metals":   "#607D8B",
    "telecom":  "#00BCD4",
    "capgoods": "#FF5722",
    "cement":   "#8BC34A",
    "consumer": "#E91E63",
    "realty":   "#FFEB3B",
    "infra":    "#009688",
    "defense":  "#3F51B5",
}


def load_results(run_dir: Path) -> pd.DataFrame:
    """Load results_detailed.csv from run directory."""
    csv_path = run_dir / "results_detailed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results_detailed.csv not found in {run_dir}")
    df = pd.read_csv(csv_path)
    df["sector"] = df["symbol"].map(SECTOR_MAP).fillna("other")
    return df


def plot_equity_curve(run_dir: Path, out_dir: Path):
    """Plot portfolio equity curve. Uses equity_curve.csv if available."""
    equity_path = run_dir / "equity_curve.csv"
    if not equity_path.exists():
        print("  ⚠ equity_curve.csv not found — skipping equity curve plot")
        return

    eq = pd.read_csv(equity_path, parse_dates=["date"])
    eq = eq.sort_values("date").reset_index(drop=True)

    initial = eq["portfolio_value"].iloc[0]
    eq["portfolio_return"] = (eq["portfolio_value"] / initial - 1) * 100

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: equity curve
    ax = axes[0]
    ax.plot(eq["date"], eq["portfolio_return"], color="#2196F3", lw=1.5, label="V3 Portfolio (HRP)")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.fill_between(
        eq["date"], eq["portfolio_return"], 0,
        where=eq["portfolio_return"] >= 0, alpha=0.15, color="#4CAF50"
    )
    ax.fill_between(
        eq["date"], eq["portfolio_return"], 0,
        where=eq["portfolio_return"] < 0, alpha=0.15, color="#F44336"
    )
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title("V3 Portfolio Equity Curve — Walk-Forward Validation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Bottom: daily returns bar chart
    ax2 = axes[1]
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in eq["daily_return"]]
    ax2.bar(eq["date"], eq["daily_return"] * 100, color=colors, width=1.5, alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("Daily Return (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "equity_curve.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_accuracy_distribution(results: pd.DataFrame, out_dir: Path):
    """Histogram of per-stock accuracy with 50% and 52% markers."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    acc = results["accuracy"].dropna()
    bins = np.arange(acc.min() - 1, acc.max() + 2, 1)
    n, _, patches = ax.hist(acc, bins=bins, color="#2196F3", alpha=0.8, edgecolor="white")

    # Colour bars by threshold
    for patch, left in zip(patches, bins[:-1]):
        if left >= 52:
            patch.set_facecolor("#4CAF50")
        elif left >= 50:
            patch.set_facecolor("#FFC107")
        else:
            patch.set_facecolor("#F44336")

    ax.axvline(50, color="#F44336", lw=2, ls="--", label="Random baseline (50%)")
    ax.axvline(52, color="#4CAF50", lw=2, ls="--", label="Publication target (52%)")
    ax.axvline(acc.mean(), color="black", lw=1.5, ls="-", label=f"Mean: {acc.mean():.1f}%")

    above_52 = (acc >= 52).sum()
    above_50 = (acc >= 50).sum()
    ax.set_xlabel("Directional Accuracy (%)", fontsize=12)
    ax.set_ylabel("Number of Stocks", fontsize=12)
    ax.set_title(
        f"OOS Directional Accuracy — {len(acc)} Stocks\n"
        f"{above_52}/{len(acc)} above 52% | {above_50}/{len(acc)} above 50%",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "accuracy_distribution.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_sector_accuracy(results: pd.DataFrame, out_dir: Path):
    """Bubble chart: accuracy by sector, bubble size = profit factor."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    sector_stats = (
        results.groupby("sector")
        .agg(
            accuracy=("accuracy", "mean"),
            profit_factor=("profit_factor", lambda x: np.clip(x.replace([np.inf], np.nan).mean(), 0, 5)),
            n_stocks=("symbol", "count"),
        )
        .reset_index()
    )

    for _, row in sector_stats.iterrows():
        color = SECTOR_COLORS.get(row["sector"], "#9E9E9E")
        size = max(50, row["n_stocks"] * 60)
        ax.scatter(
            row["accuracy"], row["profit_factor"],
            s=size, color=color, alpha=0.8, edgecolors="white", linewidth=1.5
        )
        ax.annotate(
            row["sector"],
            (row["accuracy"], row["profit_factor"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=9, fontweight="bold"
        )

    ax.axvline(50, color="#F44336", lw=1.5, ls="--", alpha=0.7, label="50% accuracy baseline")
    ax.axhline(1.0, color="#FF9800", lw=1.5, ls="--", alpha=0.7, label="Break-even profit factor")
    ax.set_xlabel("Average Directional Accuracy (%)", fontsize=12)
    ax.set_ylabel("Average Profit Factor", fontsize=12)
    ax.set_title(
        "Sector Performance — Accuracy vs Profit Factor\n(bubble size = # stocks in sector)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "sector_accuracy.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_pf_vs_accuracy(results: pd.DataFrame, out_dir: Path):
    """Scatter: Profit Factor vs Accuracy, coloured by sector."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    pf_clipped = results["profit_factor"].replace([np.inf, -np.inf], np.nan).clip(0, 5)

    for sector, grp in results.groupby("sector"):
        color = SECTOR_COLORS.get(sector, "#9E9E9E")
        pf = pf_clipped.loc[grp.index]
        ax.scatter(
            grp["accuracy"], pf,
            color=color, alpha=0.75, s=60, edgecolors="white", linewidth=0.8,
            label=sector
        )

    ax.axvline(50, color="#F44336", lw=1.5, ls="--", alpha=0.6, label="_50% baseline")
    ax.axhline(1.0, color="#FF9800", lw=1.5, ls="--", alpha=0.6, label="_PF = 1.0")

    ax.set_xlabel("Directional Accuracy (%)", fontsize=12)
    ax.set_ylabel("Profit Factor (clipped at 5)", fontsize=12)
    ax.set_title(
        "Profit Factor vs Directional Accuracy (per stock)\nColoured by Sector",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=8, ncol=3, loc="upper left")

    plt.tight_layout()
    out_path = out_dir / "profit_factor_vs_accuracy.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def generate_all_plots(run_id: str):
    """Generate all publication plots for a given run."""
    run_dir = RESULTS_RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"GENERATING PLOTS — Run {run_id}")
    print(f"Output: {plots_dir}")
    print(f"{'='*55}")

    try:
        results = load_results(run_dir)
        print(f"  Loaded {len(results)} stocks\n")

        plot_accuracy_distribution(results, plots_dir)
        plot_sector_accuracy(results, plots_dir)
        plot_pf_vs_accuracy(results, plots_dir)
        plot_equity_curve(run_dir, plots_dir)

        print(f"\n✅ All plots saved to: {plots_dir}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate V3 plots")
    parser.add_argument("--run-id", required=True, help="Run ID")
    args = parser.parse_args()
    generate_all_plots(args.run_id)


if __name__ == "__main__":
    main()
