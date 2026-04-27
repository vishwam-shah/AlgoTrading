"""
step 4 — Per-Symbol Evaluation + Plots
========================================
Orchestrates per-stock walk-forward run:
  features → feature selection → train_window × N → collect results → save CSVs → plots → production models

Also contains:
  - All per-stock and cross-stock plot functions
  - _flush_aggregate_csvs (crash-safe CSV rebuild after each symbol)
  - Worker wrappers for ProcessPoolExecutor
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_V3_ROOT   = _STEPS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "02_models"))

from config_v3 import (  # type: ignore  # noqa: E402
    MODELS_PROD_DIR, RESULTS_RUNS_DIR,
    MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES,
    N_TOP_FEATURES, CONFIDENCE_THRESHOLD,
    DL_SEQ_LEN,
    PLOT_DPI, PLOT_STYLE, PLOT_FIGSIZE_WIDE, PLOT_FIGSIZE_TALL, PLOT_FIGSIZE_SQUARE,
)
from steps.features import get_or_compute_features, feature_cols, select_top_features  # type: ignore
from steps.train    import build_windows, train_window, apply_temperature, _DL_CLASSES  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  CRASH-SAFE AGGREGATE CSV FLUSH
# ══════════════════════════════════════════════════════════════════════════════

def flush_aggregate_csvs(result_run_path: Path) -> None:
    """
    Rebuild summary.csv / all_windows_detail.csv / model_comparison.csv
    from per-symbol JSON/CSV files. Called after every symbol so results
    survive mid-run crashes.
    """
    rows: list = []
    win_rows: list = []
    for sym_dir in sorted(result_run_path.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name == "plots":
            continue
        sr_path = sym_dir / "summary_row.json"
        wr_path = sym_dir / "window_results.csv"
        if sr_path.exists():
            try:
                with open(sr_path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
        if wr_path.exists():
            try:
                win_rows.extend(pd.read_csv(wr_path).to_dict("records"))
            except Exception:
                pass

    if rows:
        summary_df = pd.DataFrame(rows)
        num_cols   = [
            "oos_accuracy", "oos_f1", "n_windows", "n_predictions", "n_features", "n_rows",
            "avg_lgbm_acc", "avg_xgb_acc", "avg_lstm_acc", "avg_bilstm_acc",
            "avg_gru_acc", "avg_cnn_lstm_acc", "avg_cnn_gru_acc",
            "avg_tcn_gru_acc", "avg_tcn_transformer_acc", "avg_nbeats_acc",
            "best_model_acc", "avg_dir_acc_up", "avg_dir_acc_down", "avg_pct_neutral",
        ]
        existing = [c for c in num_cols if c in summary_df.columns]
        avg: dict = {c: (summary_df[c].sum() if c == "n_predictions" else summary_df[c].mean())
                     for c in existing}
        avg["symbol"] = "AVERAGE"; avg["status"] = "ok"
        model_avg_cols = [c for c in existing if c.startswith("avg_") and c.endswith("_acc")
                          and not c.startswith("avg_dir_")]
        if model_avg_cols:
            best_col = max(model_avg_cols, key=lambda c: avg.get(c, 0))
            avg["best_model"] = best_col.replace("avg_", "").replace("_acc", "")
        pd.concat([summary_df, pd.DataFrame([avg])], ignore_index=True).to_csv(
            result_run_path / "summary.csv", index=False
        )
        if model_avg_cols:
            model_comp = []
            for col in model_avg_cols:
                model_name = col.replace("avg_", "").replace("_acc", "").upper()
                acc_vals = summary_df[col].dropna()
                model_comp.append({
                    "model": model_name,
                    "avg_accuracy":    float(acc_vals.mean())   if len(acc_vals) > 0 else 0.0,
                    "median_accuracy": float(acc_vals.median()) if len(acc_vals) > 0 else 0.0,
                    "max_accuracy":    float(acc_vals.max())    if len(acc_vals) > 0 else 0.0,
                    "min_accuracy":    float(acc_vals.min())    if len(acc_vals) > 0 else 0.0,
                    "std_accuracy":    float(acc_vals.std())    if len(acc_vals) > 1 else 0.0,
                    "n_stocks": int(len(acc_vals)),
                })
            (pd.DataFrame(model_comp).sort_values("avg_accuracy", ascending=False)
             .to_csv(result_run_path / "model_comparison.csv", index=False))

    if win_rows:
        pd.DataFrame(win_rows).to_csv(result_run_path / "all_windows_detail.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try: plt.style.use(PLOT_STYLE)
    except Exception: pass
    return plt


def plot_oos_accuracy(window_rows: List[Dict], symbol: str, out_dir: Path) -> None:
    plt  = _mpl()
    ids  = [w["window_id"]    for w in window_rows]
    accs = [w["oos_accuracy"] for w in window_rows]
    lgbs = [w.get("lgbm_acc", 0) for w in window_rows]
    xgbs = [w.get("xgb_acc",  0) for w in window_rows]
    temps = [w.get("temperature", 1.0) for w in window_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_TALL, sharex=True)
    x = np.arange(len(ids)); w = 0.25
    ax1.bar(x - w,     accs, w*2, label="Ensemble", color="#2196F3", alpha=0.85)
    ax1.bar(x + w*0.5, lgbs, w,   label="LightGBM", color="#4CAF50", alpha=0.7)
    ax1.bar(x + w*1.5, xgbs, w,   label="XGBoost",  color="#FF9800", alpha=0.7)
    ax1.axhline(0.58, color="red",    ls="--", lw=1.2, label="58% target")
    ax1.axhline(0.50, color="orange", ls=":",  lw=1.0, label="50% baseline")
    ax1.set_ylabel("OOS Accuracy"); ax1.set_title(f"{symbol} — OOS Accuracy by Window")
    ax1.set_ylim(0.35, 0.80); ax1.legend(loc="upper left", fontsize=8)
    ax2.bar(x, temps, color="#9C27B0", alpha=0.75)
    ax2.axhline(1.0, color="grey", ls="--", lw=1, label="T=1 (no scaling)")
    ax2.set_xticks(x); ax2.set_xticklabels([f"W{i}" for i in ids])
    ax2.set_ylabel("Temperature T"); ax2.set_title("Calibration Temperature")
    ax2.legend(fontsize=8)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "oos_accuracy.png", dpi=PLOT_DPI); plt.close(fig)


def plot_confusion(window_rows: List[Dict], symbol: str, out_dir: Path) -> None:
    plt = _mpl()
    TP = sum(w.get("tp", 0) for w in window_rows)
    TN = sum(w.get("tn", 0) for w in window_rows)
    FP = sum(w.get("fp", 0) for w in window_rows)
    FN = sum(w.get("fn", 0) for w in window_rows)
    cm  = np.array([[TN, FP], [FN, TP]]); pct = cm / (cm.sum() + 1e-10)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQUARE)
    ax.imshow(pct, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({pct[i,j]:.1%})", ha="center", va="center",
                    fontsize=12, color="white" if pct[i,j] > 0.5 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred DOWN","Pred UP"]); ax.set_yticklabels(["Act DOWN","Act UP"])
    ax.set_title(f"{symbol} — Confusion Matrix (all windows)")
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=PLOT_DPI); plt.close(fig)


def plot_confidence_timeline(pred_df: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    """
    prob_up over time coloured by correct/incorrect + rolling accuracy.
    Replaces prediction_history.png and window_accuracy.png.
    Journal-quality: shows calibration quality and temporal stability.
    """
    try:
        plt = _mpl()
        df  = pred_df.copy(); df["date"] = pd.to_datetime(df["date"])
        df  = df.sort_values("date").reset_index(drop=True)
        df["roll_acc"] = (df["actual"] == df["ensemble_pred"]).astype(int).rolling(30, min_periods=10).mean()

        has_prob = "prob_up" in df.columns and df["prob_up"].notna().any()

        fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_TALL, sharex=True)

        # Top panel: prob_up or rolling accuracy
        if has_prob:
            c_ok  = df["actual"] == df["ensemble_pred"]
            axes[0].scatter(df.loc[c_ok,  "date"], df.loc[c_ok,  "prob_up"],
                            s=6, alpha=0.5, color="#4CAF50", label="Correct")
            axes[0].scatter(df.loc[~c_ok, "date"], df.loc[~c_ok, "prob_up"],
                            s=6, alpha=0.4, color="#F44336", label="Incorrect")
            axes[0].axhline(0.5, color="grey", ls="--", lw=0.8)
            axes[0].set_ylabel("P(UP)"); axes[0].set_ylim(0, 1)
            axes[0].set_title(f"{symbol} — Calibrated P(UP) vs Outcome")
        else:
            axes[0].plot(df["date"], df["roll_acc"], color="#2196F3", lw=1.5, label="30-day rolling acc")
            axes[0].axhline(0.58, color="red",    ls="--", lw=1, label="58% target")
            axes[0].axhline(0.50, color="orange", ls=":",  lw=1, label="50% baseline")
            axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0.2, 0.95)
            axes[0].set_title(f"{symbol} — Rolling OOS Accuracy")
        axes[0].legend(fontsize=7)

        # Bottom panel: 30-day rolling accuracy
        axes[1].plot(df["date"], df["roll_acc"], color="#2196F3", lw=1.5)
        axes[1].axhline(0.58, color="red",    ls="--", lw=1, label="58% target")
        axes[1].axhline(0.50, color="orange", ls=":",  lw=1, label="50% baseline")
        axes[1].set_ylabel("30d Rolling Acc"); axes[1].set_ylim(0.25, 0.85)
        axes[1].legend(fontsize=7)

        fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "confidence_timeline.png", dpi=PLOT_DPI); plt.close(fig)
    except Exception:
        pass


def plot_cross_stock_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _mpl()
    df  = summary_df[summary_df["symbol"] != "AVERAGE"].copy().sort_values("oos_accuracy", ascending=True)
    colors = ["#4CAF50" if v >= 0.58 else "#FF9800" if v >= 0.50 else "#F44336"
              for v in df["oos_accuracy"]]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    bars = ax.barh(df["symbol"], df["oos_accuracy"], color=colors, alpha=0.85)
    ax.axvline(0.58, color="red",    ls="--", lw=1.2, label="58% target")
    ax.axvline(0.50, color="orange", ls=":",  lw=1.0, label="50% baseline")
    for bar, val in zip(bars, df["oos_accuracy"]):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9)
    ax.set_xlabel("OOS Accuracy"); ax.set_title("Cross-Stock OOS Accuracy")
    ax.legend(fontsize=8); ax.set_xlim(0.30, 0.80)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cross_stock_comparison.png", dpi=PLOT_DPI); plt.close(fig)


def plot_model_comparison_heatmap(run_result_path: Path, out_dir: Path) -> None:
    """
    Per-model OOS accuracy matrix across all stocks.
    Rows = stocks, columns = models (LightGBM, XGBoost, BiLSTM, TCN-Transformer, NBeats).
    Journal-quality: reveals which models work on which stocks/regimes.
    """
    plt = _mpl()
    model_cols = ["lgbm_pred", "xgb_pred", "bilstm_pred", "tcn_transformer_pred", "nbeats_pred"]
    model_names = ["LightGBM", "XGBoost", "BiLSTM", "TCN-Transformer", "N-BEATS"]
    records = []
    for sym_dir in sorted(run_result_path.glob("*/predictions.csv")):
        sym = sym_dir.parent.name
        try:
            df = pd.read_csv(sym_dir)
            if "actual" not in df.columns: continue
            row = {"symbol": sym}
            for col, name in zip(model_cols, model_names):
                if col in df.columns:
                    row[name] = (df[col] == df["actual"]).mean()
            row["Ensemble"] = (df["ensemble_pred"] == df["actual"]).mean()
            records.append(row)
        except Exception:
            continue
    if not records: return
    mat = pd.DataFrame(records).set_index("symbol")
    fig, ax = plt.subplots(figsize=(max(8, len(mat.columns)*1.4), max(6, len(mat)*0.35)))
    import numpy as np
    data = mat.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.40, vmax=0.65)
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat.index)));  ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if 0.45 < data[i, j] < 0.60 else "white")
    fig.colorbar(im, ax=ax, label="OOS Accuracy", fraction=0.02, pad=0.02)
    ax.set_title("Per-Model OOS Accuracy Heatmap (All Stocks)")
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "model_comparison_heatmap.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_top20(run_result_path: Path, out_dir: Path) -> None:
    """
    Aggregate LightGBM feature importance (gain) across all stocks.
    Shows top-20 features by mean gain — journal-quality feature analysis.
    """
    plt = _mpl()
    import pickle, numpy as np
    gain_agg: dict = {}
    for pkl_path in sorted(run_result_path.glob("*/lgbm_*.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                mdl = pickle.load(f)
            imp = mdl.feature_importances_ if hasattr(mdl, "feature_importances_") else None
            if imp is None:
                try: imp = np.array(list(mdl.get_score(importance_type="gain").values()))
                except Exception: continue
            names = (mdl.feature_name_ if hasattr(mdl, "feature_name_") else
                     [f"f{i}" for i in range(len(imp))])
            for n, v in zip(names, imp):
                gain_agg[n] = gain_agg.get(n, 0.0) + float(v)
        except Exception:
            continue
    # Also try production models dir
    prod_dir = run_result_path.parent.parent / "models" / "production"
    if prod_dir.exists():
        for pkl_path in sorted(prod_dir.glob("*/lightgbm.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    mdl = pickle.load(f)
                imp   = mdl.feature_importances_ if hasattr(mdl, "feature_importances_") else None
                names = (mdl.feature_name_ if hasattr(mdl, "feature_name_") else
                         [f"f{i}" for i in range(len(imp))])
                if imp is not None:
                    for n, v in zip(names, imp):
                        gain_agg[n] = gain_agg.get(n, 0.0) + float(v)
            except Exception:
                continue
    if not gain_agg: return
    series = pd.Series(gain_agg).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    colors = ["#1565C0" if "sentiment" in n else "#4CAF50" if any(x in n for x in ["rsi","macd","ema","sma"]) else "#FF9800"
              for n in series.index]
    ax.barh(series.index[::-1], series.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Aggregate Gain Importance"); ax.set_title("Top-20 LightGBM Features (All Stocks)")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#1565C0", label="Sentiment"),
                       Patch(facecolor="#4CAF50", label="Technical"),
                       Patch(facecolor="#FF9800", label="Other")]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "feature_importance_top20.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SYMBOL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    run_id: str,
    model_run_path: Path,
    result_run_path: Path,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    force_recompute_features: bool = False,
) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score
    from datetime import datetime

    print(f"\n{'─'*60}")
    print(f"  {symbol}")
    print(f"{'─'*60}")

    feat_df  = get_or_compute_features(
        symbol, raw_df, peer_returns=peer_returns, market_df=market_df,
        usdinr_df=usdinr_df, global_cues_df=global_cues_df,
        sentiment_df=sentiment_df,
        force_recompute=force_recompute_features,
    )
    fcols  = feature_cols(feat_df)
    n_rows = len(feat_df)

    if n_rows < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES:
        print(f"  ⚠ Not enough rows ({n_rows}), skipping")
        return {"symbol": symbol, "status": "skipped", "reason": "too_few_rows"}

    X       = feat_df[fcols].values.astype(float)
    y       = feat_df["target"].values.astype(int)
    dates   = feat_df["date"].values
    regimes = feat_df["market_regime"].values.astype(int) if "market_regime" in feat_df.columns else None
    # next_ret is populated by features.add_target (v2+) and is used by the
    # meta-labeling secondary classifier. Pipeline still works with v1 features
    # (next_ret column absent) — train_window just skips meta training.
    next_ret = feat_df["next_ret"].values.astype(float) if "next_ret" in feat_df.columns else None

    windows = build_windows(n_rows)
    print(f"  [1/3] Walk-forward: {len(windows)} windows  |  {n_rows} rows  |  {len(fcols)} features")
    for w in windows:
        print(f"        Win {w['id']}: train[0:{w['train_end']}]  "
              f"val[{w['val_start']}:{w['val_end']}]  "
              f"test[{w['test_start']}:{w['test_end']}]  ({w['train_ratio']:.0%})")

    # Feature selection
    n_feat = len(fcols)
    if n_feat > N_TOP_FEATURES and len(windows) >= 1:
        fs_win = windows[-2] if len(windows) >= 2 else windows[0]
        print(f"  [1b] Feature selection: top {N_TOP_FEATURES} of {n_feat} ...", end=" ", flush=True)
        selected = select_top_features(X, y, fcols, fs_win, symbol=symbol)
        if len(selected) < n_feat:
            sel_idx = [fcols.index(f) for f in selected]
            X = X[:, sel_idx]; fcols = selected; n_feat = len(fcols)
            print(f"✓ {n_feat} features kept")
        else:
            print("✓ (all features kept)")

    print(f"  [2/3] Training...")
    sym_model_path  = model_run_path  / symbol
    sym_result_path = result_run_path / symbol
    sym_plot_path   = sym_result_path / "plots"
    sym_model_path.mkdir(parents=True, exist_ok=True)
    sym_result_path.mkdir(parents=True, exist_ok=True)

    window_rows         = []
    all_preds           = []; all_actuals     = []; all_dates        = []
    all_window_ids      = []; all_lgbm_preds  = []; all_xgb_preds    = []
    all_lstm_preds      = []; all_bilstm_preds= []; all_gru_preds    = []
    all_cnnlstm_preds   = []; all_cnngru_preds= []
    all_tcngru_preds    = []; all_tcntransf_preds = []; all_nbeats_preds = []
    all_closes          = []; all_next_closes = []
    all_probs_up        = []   # calibrated P(UP) — the key for confidence filtering
    all_meta_probs      = []   # NEW — trade-selection meta prob (López de Prado)
    last_result         = None

    for idx, win in enumerate(windows):
        _is_last_win = (idx == len(windows) - 1)
        res = train_window(X, y, win, fcols, symbol, sym_model_path,
                           regimes=regimes, save_dl_keras=_is_last_win,
                           next_ret=next_ret)
        if res is None:
            continue
        if last_result is not None and last_result is not res:
            last_result.pop("models", None)
            last_result.pop("regime_lgb_models", None)
            last_result.pop("meta_model", None)
            gc.collect()
        last_result = res
        ts, te = win["test_start"], win["test_end"]
        test_dates = dates[ts:te]; n_test = len(res["y_test"])

        all_preds.extend(res["ens_pred"]); all_actuals.extend(res["y_test"])
        all_probs_up.extend(res["avg_prob"].tolist())
        # NEW — meta-prob (fallback 0.5 if meta wasn't trained in this window)
        _mp = res.get("meta_prob")
        if _mp is None:
            _mp = np.full(n_test, 0.5, dtype=float)
        all_meta_probs.extend(np.asarray(_mp).tolist())
        all_dates.extend(test_dates); all_window_ids.extend([win["id"]] * n_test)
        tp_ = res.get("test_preds", {})
        all_lgbm_preds.extend(     tp_.get("LightGBM",       [None]*n_test))
        all_xgb_preds.extend(      tp_.get("XGBoost",         [None]*n_test))
        all_lstm_preds.extend(     tp_.get("LSTM",             [None]*n_test))
        all_bilstm_preds.extend(   tp_.get("BiLSTM",           [None]*n_test))
        all_gru_preds.extend(      tp_.get("GRU",              [None]*n_test))
        all_cnnlstm_preds.extend(  tp_.get("CNN_LSTM",         [None]*n_test))
        all_cnngru_preds.extend(   tp_.get("CNN_GRU",          [None]*n_test))
        all_tcngru_preds.extend(   tp_.get("TCN_GRU",          [None]*n_test))
        all_tcntransf_preds.extend(tp_.get("TCN_Transformer",  [None]*n_test))
        all_nbeats_preds.extend(   tp_.get("NBEATS",           [None]*n_test))
        all_closes.extend(     feat_df["close"].iloc[ts:te].values)
        all_next_closes.extend(feat_df["close"].shift(-1).iloc[ts:te].values)

        tag = "✓" if res["accuracy"] >= 0.58 else ("~" if res["accuracy"] >= 0.50 else "✗")
        _pm = res["per_model"]
        _dl_line = "  ".join(
            f"{dn}={_pm.get(dn,0):.2%}"
            for dn in ["LSTM","BiLSTM","GRU","CNN_LSTM","TCN_GRU","TCN_Transformer","NBEATS"]
            if dn in _pm
        )
        print(f"    {tag} Win {win['id']} | {res['window']['train_ratio']:.0%} train"
              f" | test_n={te-ts}"
              f" | OOS={res['accuracy']:.2%} | AUC={res['auc']:.3f} | F1={res['f1']:.3f}"
              f" | LGB={_pm.get('LightGBM',0):.2%}"
              f" | XGB={_pm.get('XGBoost', 0):.2%}"
              f" | T={res.get('temperature',1.0):.2f}")
        print(f"         Dir → UP={res.get('dir_acc_up',0):.2%}  "
              f"DOWN={res.get('dir_acc_down',0):.2%}  "
              f"NEUTRAL={res.get('pct_neutral',0):.1%}")
        if _dl_line:
            print(f"         DL  → {_dl_line}")

        window_rows.append({
            "symbol": symbol, "window_id": win["id"], "train_ratio": win["train_ratio"],
            "train_size": win["train_end"], "val_size": win["val_end"]-win["val_start"],
            "test_size": te-ts,
            "test_start": str(test_dates[0])[:10] if len(test_dates) else "",
            "test_end":   str(test_dates[-1])[:10] if len(test_dates) else "",
            "oos_accuracy": res["accuracy"], "auc": res["auc"], "f1": res["f1"],
            "precision": res["precision"], "recall": res["recall"],
            "tp": int(res["tp"]), "fp": int(res["fp"]),
            "tn": int(res["tn"]), "fn": int(res["fn"]),
            "lgbm_acc":            _pm.get("LightGBM",       0.0),
            "xgb_acc":             _pm.get("XGBoost",         0.0),
            "lstm_acc":            _pm.get("LSTM",             0.0),
            "bilstm_acc":          _pm.get("BiLSTM",           0.0),
            "gru_acc":             _pm.get("GRU",              0.0),
            "cnn_lstm_acc":        _pm.get("CNN_LSTM",         0.0),
            "cnn_gru_acc":         _pm.get("CNN_GRU",          0.0),
            "tcn_gru_acc":         _pm.get("TCN_GRU",          0.0),
            "tcn_transformer_acc": _pm.get("TCN_Transformer",  0.0),
            "nbeats_acc":          _pm.get("NBEATS",           0.0),
            "dir_acc_up":    res.get("dir_acc_up",   0.0),
            "dir_acc_down":  res.get("dir_acc_down",  0.0),
            "pct_neutral":   res.get("pct_neutral",   0.0),
            "pct_up":        res.get("pct_up",        0.0),
            "pct_down":      res.get("pct_down",      0.0),
            "temperature":   res.get("temperature",   1.0),
            "dl_models_trained": len(res.get("dl_meta", {}).get("dl_models", [])),
        })

    if all_preds:
        oos_acc = accuracy_score(all_actuals, all_preds)
        oos_f1  = f1_score(all_actuals, all_preds, zero_division=0)
    else:
        oos_acc = 0.0; oos_f1 = 0.0

    avg_t = np.mean([w.get("temperature", 1.0) for w in window_rows]) if window_rows else 1.0
    print(f"  [3/3] OOS Overall → Accuracy={oos_acc:.2%}  F1={oos_f1:.4f}"
          f"  ({len(all_preds)} preds, {len(windows)} wins, avg_T={avg_t:.2f})")

    # Save window results (resume marker)
    pd.DataFrame(window_rows).to_csv(sym_result_path / "window_results.csv", index=False)

    # Per-model averages
    _model_accs = {}
    _model_cols = {
        "lgbm_acc": "LightGBM", "xgb_acc": "XGBoost",
        "lstm_acc": "LSTM", "bilstm_acc": "BiLSTM", "gru_acc": "GRU",
        "cnn_lstm_acc": "CNN_LSTM", "cnn_gru_acc": "CNN_GRU",
        "tcn_gru_acc": "TCN_GRU", "tcn_transformer_acc": "TCN_Transformer",
        "nbeats_acc": "NBEATS",
    }
    for col in _model_cols:
        vals = [w[col] for w in window_rows if w.get(col, 0) > 0]
        _model_accs[f"avg_{col}"] = float(np.mean(vals)) if vals else 0.0

    _dir_metrics = {}
    for dkey in ["dir_acc_up", "dir_acc_down", "pct_neutral"]:
        vals = [w.get(dkey, 0.0) for w in window_rows]
        _dir_metrics[f"avg_{dkey}"] = float(np.mean(vals)) if vals else 0.0

    _best_model     = max(_model_accs, key=_model_accs.get, default="") if _model_accs else ""
    _best_model_name = _best_model.replace("avg_", "").replace("_acc", "") if _best_model else "none"
    _best_model_acc  = _model_accs.get(_best_model, 0.0)

    with open(sym_result_path / "summary_row.json", "w") as _f:
        json.dump({
            "symbol": symbol, "status": "ok",
            "oos_accuracy": oos_acc, "oos_f1": oos_f1,
            "n_windows": len(windows), "n_predictions": len(all_preds),
            "n_features": n_feat, "n_rows": n_rows,
            **_model_accs, **_dir_metrics,
            "best_model": _best_model_name,
            "best_model_acc": _best_model_acc,
        }, _f, indent=2)

    pred_df = pd.DataFrame({
        "date": pd.to_datetime(all_dates), "window_id": all_window_ids,
        "close_price": all_closes, "next_close_price": all_next_closes,
        "actual": all_actuals,
        "prob_up":              np.round(all_probs_up, 4),   # calibrated P(UP) — for confidence filtering
        "meta_prob":            np.round(all_meta_probs, 4),  # NEW — meta-labeling filter (López de Prado)
        "lgbm_pred":            all_lgbm_preds,
        "xgb_pred":             all_xgb_preds,
        "lstm_pred":            all_lstm_preds,
        "bilstm_pred":          all_bilstm_preds,
        "gru_pred":             all_gru_preds,
        "cnn_lstm_pred":        all_cnnlstm_preds,
        "cnn_gru_pred":         all_cnngru_preds,
        "tcn_gru_pred":         all_tcngru_preds,
        "tcn_transformer_pred": all_tcntransf_preds,
        "nbeats_pred":          all_nbeats_preds,
        "ensemble_pred": all_preds,
        "direction": ["UP" if p == 1 else "DOWN" for p in all_preds],
        "correct": np.array(all_actuals) == np.array(all_preds),
    })
    pred_df.to_csv(sym_result_path / "predictions.csv", index=False)

    # Plots
    try:
        plot_oos_accuracy(window_rows, symbol, sym_plot_path)
        plot_confusion(window_rows, symbol, sym_plot_path)
        plot_confidence_timeline(pred_df, symbol, sym_plot_path)
        print(f"  Plots → {sym_plot_path.relative_to(_V3_ROOT)}")
    except Exception as exc:
        print(f"  ⚠ Plot error: {exc}")

    # Save production models (last window)
    if last_result:
        prod_path = MODELS_PROD_DIR / symbol
        prod_path.mkdir(parents=True, exist_ok=True)
        for name in ("LightGBM", "XGBoost"):
            if name in last_result["models"]:
                with open(prod_path / f"{name.lower()}.pkl", "wb") as f:
                    pickle.dump(last_result["models"][name], f)
        _last_dl_meta  = last_result.get("dl_meta", {})
        _last_win_path = last_result.get("win_path")
        for dl_name in _last_dl_meta.get("dl_models", []):
            src = _last_win_path / f"{dl_name.lower()}.keras" if _last_win_path else None
            if src and src.exists():
                try:
                    shutil.copy2(str(src), str(prod_path / f"{dl_name.lower()}.keras"))
                except Exception as _se:
                    print(f"  [prod save] {dl_name}: {_se}")
        with open(prod_path / "scaler.pkl",        "wb") as f: pickle.dump(last_result["scaler"],        f)
        with open(prod_path / "pca.pkl",           "wb") as f: pickle.dump(last_result["pca"],           f)
        with open(prod_path / "winsor_bounds.pkl", "wb") as f: pickle.dump(last_result["winsor_bounds"], f)
        if last_result.get("meta_model"):
            with open(prod_path / "meta_model.pkl", "wb") as f: pickle.dump(last_result["meta_model"], f)
        # NEW — trade-selection secondary (López de Prado meta-labeling). Copy to
        # production so predict.py can apply the meta gate at inference time.
        if last_result.get("secondary_model") is not None:
            with open(prod_path / "secondary.pkl", "wb") as f: pickle.dump(last_result["secondary_model"], f)
        elif (prod_path / "secondary.pkl").exists():
            # If a previous run had a secondary but this one doesn't, remove stale copy
            try: (prod_path / "secondary.pkl").unlink()
            except Exception: pass
        with open(prod_path / "calibration.json", "w") as f:
            json.dump({
                "temperature": last_result.get("temperature", 1.0),
                "meta_info":   last_result.get("meta_info", {}),
            }, f)
        with open(prod_path / "dl_meta.json", "w") as f:
            json.dump(_last_dl_meta, f, indent=2)
        for rv, rmdl in last_result.get("regime_lgb_models", {}).items():
            rtag = {2: "bull", 1: "sideways", 0: "bear"}.get(rv, str(rv))
            with open(prod_path / f"lgb_{rtag}.pkl", "wb") as f:
                pickle.dump(rmdl, f)
        with open(prod_path / "metadata.json", "w") as f:
            json.dump({
                "symbol": symbol, "run_id": run_id, "feature_names": fcols,
                "oos_accuracy": oos_acc, "oos_f1": oos_f1,
                "n_features": n_feat, "n_train_rows": n_rows,
                "last_window": windows[-1]["id"] if windows else 0,
                "dl_models": _last_dl_meta.get("dl_models", []),
                "dl_seq_len": _last_dl_meta.get("seq_len", DL_SEQ_LEN),
                "trained_at": datetime.now().isoformat(),
            }, f, indent=2)

    del last_result
    try:
        import keras; keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return {
        "symbol": symbol, "status": "ok",
        "oos_accuracy": oos_acc, "oos_f1": oos_f1,
        "n_windows": len(windows), "n_predictions": len(all_preds),
        "n_features": n_feat, "n_rows": n_rows, "window_rows": window_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL WORKER WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def worker_init(n_jobs: int, fast_mode: bool = False) -> None:
    """Called once per ProcessPoolExecutor worker. Sets thread limits and GPU backend."""
    # ── Backend: TensorFlow CPU ──────────────────────────────────────────────
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import steps.train as _train_mod  # type: ignore
    _train_mod._N_JOBS = n_jobs
    if fast_mode:
        _train_mod.set_fast_mode(True)
    s = str(n_jobs)
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS","TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS"]:
        os.environ[k] = s
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    try:
        import logging as _log; _log.getLogger("tensorflow").setLevel(_log.ERROR)
    except Exception:
        pass


def run_symbol_worker(
    symbol: str,
    raw_df: pd.DataFrame,
    run_id: str,
    model_run_path: Path,
    result_run_path: Path,
    peer_returns: dict,
    market_df,
    usdinr_df,
    global_cues_df,
    sentiment_df=None,
) -> dict:
    """Top-level picklable function for ProcessPoolExecutor. Redirects per-symbol stdout to file."""
    log_dir  = result_run_path / symbol
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    try:
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
            with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                return run_symbol(
                    symbol=symbol, raw_df=raw_df, run_id=run_id,
                    model_run_path=model_run_path, result_run_path=result_run_path,
                    peer_returns=peer_returns, market_df=market_df,
                    usdinr_df=usdinr_df, global_cues_df=global_cues_df,
                    sentiment_df=sentiment_df,
                )
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "reason": str(exc)}
