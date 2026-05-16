"""
generate_plots.py — Comprehensive result visualization for the 100-stock pipeline.
Reads from the latest run directory and saves PNG plots to 06_results/plots/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_RUNS  = Path(__file__).parent / "runs"
_OUT   = Path(__file__).parent / "plots"
_OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "text.color":       "#111111",
    "axes.labelcolor":  "#111111",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "axes.edgecolor":   "#cccccc",
    "grid.color":       "#e5e5e5",
    "grid.linestyle":   "--",
    "grid.alpha":       0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)
ACCENT   = "#4f46e5"   # indigo
GREEN    = "#16a34a"
RED      = "#dc2626"
AMBER    = "#d97706"
BLUE     = "#2563eb"


def _latest_run() -> Path:
    runs = sorted(_RUNS.glob("20*"), reverse=True)
    if not runs:
        raise FileNotFoundError("No run directories found")
    return runs[0]


def load_data(run_dir: Path):
    summary  = pd.read_csv(run_dir / "summary.csv")
    detail   = pd.read_csv(run_dir / "all_windows_detail.csv")

    sym_rows = []
    for f in sorted(run_dir.glob("*/summary_row.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["symbol"] = f.parent.name
            sym_rows.append(d)
        except Exception:
            pass
    sym_df = pd.DataFrame(sym_rows) if sym_rows else pd.DataFrame()

    backtest = {}
    bt_path = run_dir / "backtest_summary.json"
    if bt_path.exists():
        with open(bt_path) as fh:
            backtest = json.load(fh)

    pred = None
    pred_path = run_dir / "next_day_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)

    return summary, detail, sym_df, backtest, pred


# ── Plot 1: OOS accuracy distribution ─────────────────────────────────────────

def plot_accuracy_distribution(summary: pd.DataFrame, out: Path) -> None:
    df = summary[summary["status"] == "ok"].dropna(subset=["oos_accuracy"])
    acc = df["oos_accuracy"].values * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Out-of-Sample Direction Accuracy — 100 NSE Stocks", fontsize=14, fontweight="bold", y=1.02)

    # Histogram
    ax = axes[0]
    bins = np.linspace(40, 65, 20)
    colors = [GREEN if v >= 54 else AMBER if v >= 50 else RED for v in acc]
    ax.hist(acc, bins=bins, color=ACCENT, alpha=0.8, edgecolor="#222")
    ax.axvline(50, color=RED,   linestyle="--", lw=1.5, label="50% (random)")
    ax.axvline(54, color=GREEN, linestyle="--", lw=1.5, label="54% (target)")
    ax.axvline(acc.mean(), color=AMBER, linestyle="-", lw=2, label=f"Mean {acc.mean():.1f}%")
    ax.set_xlabel("OOS Accuracy (%)")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Accuracy Histogram")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Sorted bar
    ax2 = axes[1]
    df_s = df.sort_values("oos_accuracy", ascending=False).reset_index(drop=True)
    bar_colors = [GREEN if v >= 0.54 else AMBER if v >= 0.50 else RED for v in df_s["oos_accuracy"]]
    ax2.bar(range(len(df_s)), df_s["oos_accuracy"] * 100, color=bar_colors, alpha=0.85)
    ax2.axhline(50, color=RED,   linestyle="--", lw=1.2)
    ax2.axhline(54, color=GREEN, linestyle="--", lw=1.2)
    ax2.set_xlabel("Stock rank (highest -> lowest)")
    ax2.set_ylabel("OOS Accuracy (%)")
    ax2.set_title("Sorted Accuracy by Stock")
    n_above = (df["oos_accuracy"] >= 0.54).sum()
    n_mid   = ((df["oos_accuracy"] >= 0.5) & (df["oos_accuracy"] < 0.54)).sum()
    n_below = (df["oos_accuracy"] < 0.5).sum()
    patches = [mpatches.Patch(color=GREEN, label=f">=54% ({n_above} stocks)"),
               mpatches.Patch(color=AMBER, label=f"50-54% ({n_mid} stocks)"),
               mpatches.Patch(color=RED,   label=f"<50% ({n_below} stocks)")]
    ax2.legend(handles=patches, fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(out / "01_accuracy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 01_accuracy_distribution.png")


# ── Plot 2: Portfolio backtest metrics ────────────────────────────────────────

def plot_backtest_metrics(backtest: dict, out: Path) -> None:
    if not backtest:
        return

    metrics = {
        "Portfolio\nReturn":   backtest.get("portfolio_total_return", 0) * 100,
        "Nifty 50\nReturn":    backtest.get("nifty_return", 0) * 100,
        "Avg/Stock\nReturn":   backtest.get("avg_per_stock_return", 0) * 100,
    }
    sharpe   = backtest.get("portfolio_sharpe", 0)
    max_dd   = backtest.get("portfolio_max_dd", 0) * 100
    boot_acc = backtest.get("bootstrap_acc_mean", 0) * 100
    ci_lo    = backtest.get("bootstrap_ci_lower", 0) * 100
    ci_hi    = backtest.get("bootstrap_ci_upper", 0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Backtest Performance Summary", fontsize=14, fontweight="bold")

    # Return bars
    ax = axes[0]
    labels = list(metrics.keys())
    vals   = list(metrics.values())
    bar_c  = [GREEN, BLUE, AMBER]
    bars   = ax.bar(labels, vals, color=bar_c, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Return Comparison")
    ax.grid(True, axis="y")

    # Risk gauges
    ax2 = axes[1]
    gauges = {"Sharpe\nRatio": sharpe, "Max\nDrawdown %": max_dd}
    g_c = [GREEN if sharpe >= 1.5 else AMBER, RED if max_dd > 20 else AMBER if max_dd > 15 else GREEN]
    b2 = ax2.bar(list(gauges.keys()), list(gauges.values()), color=g_c, alpha=0.85, width=0.4)
    for bar, v in zip(b2, gauges.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_title("Risk Metrics")
    ax2.grid(True, axis="y")

    # Bootstrap accuracy
    ax3 = axes[2]
    ax3.bar(["Bootstrap\nAccuracy"], [boot_acc], color=GREEN if boot_acc > 55 else AMBER, alpha=0.85, width=0.4)
    ax3.errorbar(["Bootstrap\nAccuracy"], [boot_acc],
                 yerr=[[boot_acc - ci_lo], [ci_hi - boot_acc]],
                 fmt="none", color="white", capsize=8, linewidth=2)
    ax3.axhline(50, color=RED, linestyle="--", lw=1.5, label="Random (50%)")
    ax3.text(0, boot_acc + (ci_hi - boot_acc) + 1,
             f"{boot_acc:.1f}%\n[{ci_lo:.1f}-{ci_hi:.1f}%]",
             ha="center", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Accuracy on UP signals (%)")
    ax3.set_title("Bootstrap Signal Accuracy")
    ax3.set_ylim(45, 70)
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(out / "02_backtest_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 02_backtest_metrics.png")


# ── Plot 3: Walk-forward window accuracy trend ────────────────────────────────

def plot_window_trend(detail: pd.DataFrame, out: Path) -> None:
    by_w = detail.groupby("window_id")["oos_accuracy"].agg(["mean", "std", "min", "max"])

    fig, ax = plt.subplots(figsize=(10, 5))
    w = by_w.index.values
    m = by_w["mean"].values * 100
    s = by_w["std"].values * 100

    ax.fill_between(w, m - s, m + s, alpha=0.2, color=ACCENT, label="±1 std dev")
    ax.plot(w, m, "o-", color=ACCENT, linewidth=2.5, markersize=7, label="Mean OOS accuracy")
    ax.plot(w, by_w["min"].values * 100, "v--", color=RED,   alpha=0.6, linewidth=1, label="Min")
    ax.plot(w, by_w["max"].values * 100, "^--", color=GREEN, alpha=0.6, linewidth=1, label="Max")
    ax.axhline(50, color=RED,   linestyle=":", lw=1.5, label="Random baseline (50%)")
    ax.axhline(54, color=GREEN, linestyle=":", lw=1.5, label="Target (54%)")

    for xi, yi in zip(w, m):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=ACCENT)

    ax.set_xlabel("Walk-forward Window ID (expanding train set)")
    ax.set_ylabel("OOS Accuracy (%)")
    ax.set_title("Walk-forward Validation — Accuracy Across 6 Windows (100 stocks)")
    ax.set_xticks(w)
    ax.set_xticklabels([f"W{i}\n(70-95% train)" for i in w], fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)
    ax.set_ylim(35, 80)

    plt.tight_layout()
    fig.savefig(out / "03_window_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 03_window_trend.png")


# ── Plot 4: Top stocks heatmap ────────────────────────────────────────────────

def plot_top_stocks_heatmap(sym_df: pd.DataFrame, out: Path) -> None:
    df = sym_df[sym_df["status"] == "ok"].dropna(subset=["oos_accuracy"])
    df = df.sort_values("oos_accuracy", ascending=False).head(30).reset_index(drop=True)

    metrics = ["oos_accuracy", "avg_dir_acc_up", "avg_dir_acc_down", "oos_f1"]
    col_labels = ["OOS Acc", "UP Acc", "DOWN Acc", "F1"]
    data = df[metrics].values * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=40, vmax=70)
    plt.colorbar(im, ax=ax, label="Accuracy / Score (%)")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["symbol"].values, fontsize=8)

    for i in range(len(df)):
        for j in range(len(metrics)):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="black" if 45 < v < 65 else "white")

    ax.set_title("Top 30 Stocks — Performance Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "04_top_stocks_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 04_top_stocks_heatmap.png")


# ── Plot 5: Sector breakdown ──────────────────────────────────────────────────

def plot_sector_breakdown(sym_df: pd.DataFrame, out: Path) -> None:
    SECTOR_MAP = {
        "ICICIBANK":"banking","SBIN":"banking","HDFCBANK":"banking","AXISBANK":"banking","KOTAK":"banking",
        "INDUSINDBK":"banking","BANDHANBNK":"banking","AUBANK":"banking","FEDERALBNK":"banking","IDFCFIRSTB":"banking",
        "TCS":"it","INFY":"it","WIPRO":"it","HCLTECH":"it","LTIM":"it","TECHM":"it","MPHASIS":"it","COFORGE":"it","PERSISTENT":"it","OFSS":"it",
        "RELIANCE":"energy","BPCL":"energy","ONGC":"energy","NTPC":"energy","POWERGRID":"energy","COAL":"energy",
        "SBILIFE":"insurance","HDFCLIFE":"insurance","ICICIPRULI":"insurance","BAJAJFINSV":"insurance",
        "BAJFINANCE":"fintech","CHOLAFIN":"fintech","MUTHOOTFIN":"fintech","MANAPPURAM":"fintech","PGHH":"fintech",
        "HDFCAMC":"fintech",
        "ASIANPAINT":"consumer","BRITANNIA":"consumer","ITC":"consumer","HINDUNILVR":"consumer","NESTLEIND":"consumer",
        "COLPAL":"consumer","MARICO":"consumer","TATACONSUM":"consumer","JUBLFOOD":"consumer",
        "SUNPHARMA":"pharma","DRREDDY":"pharma","CIPLA":"pharma","DIVISLAB":"pharma","AUROPHARMA":"pharma",
        "ALKEM":"pharma","GLAND":"pharma",
        "MARUTI":"auto","TATAMOTORS":"auto","BAJAJ-AUTO":"auto","EICHERMOT":"auto","HEROMOTOCO":"auto",
        "TVSMOTOR":"auto","BOSCHLTD":"auto","EXIDEIND":"auto","MOTHERSON":"auto",
        "HINDALCO":"metals","TATASTEEL":"metals","JSWSTEEL":"metals","SAIL":"metals","NMDC":"metals",
        "VEDL":"metals","COALINDIA":"metals",
        "LT":"infra","ULTRACEMCO":"infra","AMBUJACEM":"infra","ACC":"infra","BHEL":"infra","BEL":"infra",
        "ABB":"infra","HAVELLS":"infra","POLYCAB":"infra","CUMMINSIND":"infra","SIEMENS":"infra",
        "TITAN":"luxury","PIDILITIND":"luxury","BERGEPAINT":"luxury","GODREJCP":"luxury","VBL":"luxury",
        "ADANIENT":"conglomerate","ADANIPORTS":"conglomerate","ADANIGREEN":"conglomerate",
        "TATACHEM":"conglomerate","TATAPOWER":"conglomerate","LICHSGFIN":"conglomerate",
        "HDFCAMC":"fintech","RECLTD":"fintech","PFC":"fintech",
        "GODREJPROP":"realestate","DLF":"realestate","PRESTIGE":"realestate",
        "ATGL":"gas","GAIL":"gas","MGL":"gas",
        "IRFC":"infra","ETERNAL":"consumer",
    }
    df = sym_df[sym_df["status"] == "ok"].dropna(subset=["oos_accuracy"])
    df = df.copy()
    df["sector"] = df["symbol"].map(SECTOR_MAP).fillna("other")

    sector_perf = df.groupby("sector")["oos_accuracy"].agg(["mean", "count", "std"]).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [GREEN if v >= 0.54 else AMBER if v >= 0.50 else RED for v in sector_perf["mean"]]
    bars = ax.barh(sector_perf.index, sector_perf["mean"] * 100, color=colors, alpha=0.85)
    ax.errorbar(sector_perf["mean"] * 100, range(len(sector_perf)),
                xerr=sector_perf["std"] * 100, fmt="none", color="white", capsize=3, linewidth=1)
    ax.axvline(50, color=RED,   linestyle="--", lw=1.5, label="Random baseline (50%)")
    ax.axvline(54, color=GREEN, linestyle="--", lw=1.5, label="Target (54%)")

    for bar, (_, row) in zip(bars, sector_perf.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{row['mean']*100:.1f}% (n={int(row['count'])})",
                va="center", fontsize=8)

    ax.set_xlabel("Mean OOS Accuracy (%)")
    ax.set_title("OOS Accuracy by Sector")
    ax.legend(fontsize=8)
    ax.set_xlim(40, 65)
    ax.grid(True, axis="x")

    plt.tight_layout()
    fig.savefig(out / "05_sector_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 05_sector_breakdown.png")


# ── Plot 6: Signal confidence vs accuracy ─────────────────────────────────────

def plot_confidence_calibration(run_dir: Path, out: Path) -> None:
    """Bin UP predictions by confidence, check actual accuracy in each bin."""
    all_preds = []
    for fp in sorted(run_dir.glob("*/predictions.csv")):
        try:
            df = pd.read_csv(fp)
            df["symbol"] = fp.parent.name
            all_preds.append(df)
        except Exception:
            pass
    if not all_preds:
        return
    preds = pd.concat(all_preds, ignore_index=True)

    prob_col = "avg_prob" if "avg_prob" in preds.columns else "prob_up" if "prob_up" in preds.columns else None
    if prob_col is None or "actual" not in preds.columns:
        return

    preds = preds.dropna(subset=[prob_col, "actual"])
    bins = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 1.01]
    labels = ["50-52", "52-54", "54-56", "56-58", "58-60", "60-65", "65+"]
    preds["conf_bin"] = pd.cut(preds[prob_col], bins=bins, labels=labels)
    actual_col = "actual"

    cal = preds.groupby("conf_bin", observed=True)[actual_col].agg(["mean", "count"]).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_c = [GREEN if v >= 0.54 else AMBER if v >= 0.50 else RED for v in cal["mean"]]
    bars = ax.bar(cal.index, cal["mean"] * 100, color=bar_c, alpha=0.85)
    ax.axhline(50, color=RED,   linestyle="--", lw=1.5, label="Random baseline")
    ax.axhline(54, color=GREEN, linestyle="--", lw=1.5, label="Target 54%")

    for bar, (idx, row) in zip(bars, cal.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{row['mean']*100:.1f}%\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model Confidence Range")
    ax.set_ylabel("Actual UP direction accuracy (%)")
    ax.set_title("Confidence Calibration — Higher Confidence -> Higher Accuracy?")
    ax.legend(fontsize=9)
    ax.set_ylim(40, 75)
    ax.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(out / "06_confidence_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 06_confidence_calibration.png")


# ── Plot 7: Today's signal dashboard ─────────────────────────────────────────

def plot_todays_signals(pred: pd.DataFrame, out: Path) -> None:
    if pred is None or len(pred) == 0:
        return

    prob_col = "avg_prob" if "avg_prob" in pred.columns else "confidence"
    up = pred[pred["direction"] == "UP"].copy() if "direction" in pred.columns else pred.copy()
    up = up.sort_values(prob_col, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_c = [GREEN if v >= 0.58 else AMBER if v >= 0.54 else BLUE for v in up[prob_col]]
    bars  = ax.bar(range(len(up)), up[prob_col] * 100, color=bar_c, alpha=0.85)
    ax.set_xticks(range(len(up)))
    ax.set_xticklabels(up["symbol"].values, rotation=45, ha="right", fontsize=9)
    ax.axhline(52, color=AMBER, linestyle="--", lw=1, label="52% threshold")
    ax.axhline(58, color=GREEN, linestyle="--", lw=1, label="58% high confidence")

    for bar, (_, row) in zip(bars, up.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{row[prob_col]*100:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("P(UP) — Model Confidence (%)")
    ax.set_title("Today's Buy Signals (Top 20 by Confidence)")
    ax.legend(fontsize=8)
    ax.set_ylim(45, 75)
    ax.grid(True, axis="y")

    patches = [mpatches.Patch(color=GREEN, label="High (>=58%)"),
               mpatches.Patch(color=AMBER, label="Medium (54-58%)"),
               mpatches.Patch(color=BLUE,  label="Low (52-54%)")]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out / "07_todays_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 07_todays_signals.png")


# ── Plot 8: LightGBM vs XGBoost comparison ────────────────────────────────────

def plot_model_comparison(sym_df: pd.DataFrame, out: Path) -> None:
    df = sym_df[sym_df["status"] == "ok"].dropna(subset=["avg_lgbm_acc", "avg_xgb_acc"])
    df = df[(df["avg_lgbm_acc"] > 0) & (df["avg_xgb_acc"] > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("LightGBM vs XGBoost Model Comparison", fontsize=13, fontweight="bold")

    # Scatter
    ax = axes[0]
    colors = [GREEN if row["avg_lgbm_acc"] > row["avg_xgb_acc"] else RED
              for _, row in df.iterrows()]
    ax.scatter(df["avg_xgb_acc"] * 100, df["avg_lgbm_acc"] * 100,
               c=colors, alpha=0.7, s=60, edgecolors="none")
    lo, hi = 44, 63
    ax.plot([lo, hi], [lo, hi], "w--", lw=1, alpha=0.4, label="Equal performance")
    ax.set_xlabel("XGBoost OOS Accuracy (%)")
    ax.set_ylabel("LightGBM OOS Accuracy (%)")
    ax.set_title("LightGBM vs XGBoost per Stock")
    lgbm_wins = (df["avg_lgbm_acc"] > df["avg_xgb_acc"]).sum()
    ax.legend([f"LightGBM better: {lgbm_wins}/{len(df)} stocks"], fontsize=9)
    ax.grid(True)

    # Best model bar
    ax2 = axes[1]
    best = df["best_model"].value_counts()
    bar_c = [ACCENT if k == "lgbm" else BLUE for k in best.index]
    ax2.bar(best.index, best.values, color=bar_c, alpha=0.85)
    for i, (k, v) in enumerate(best.items()):
        ax2.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of stocks where model is best")
    ax2.set_title("Best Model by Stock Count")
    ax2.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(out / "08_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 08_model_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_dir = _latest_run()
    print(f"Generating plots for run: {run_dir.name}")
    summary, detail, sym_df, backtest, pred = load_data(run_dir)

    plot_accuracy_distribution(summary, _OUT)
    plot_backtest_metrics(backtest, _OUT)
    plot_window_trend(detail, _OUT)
    if not sym_df.empty:
        plot_top_stocks_heatmap(sym_df, _OUT)
        plot_sector_breakdown(sym_df, _OUT)
        plot_model_comparison(sym_df, _OUT)
    plot_confidence_calibration(run_dir, _OUT)
    plot_todays_signals(pred, _OUT)

    print(f"\nAll plots saved to: {_OUT}")


if __name__ == "__main__":
    main()
