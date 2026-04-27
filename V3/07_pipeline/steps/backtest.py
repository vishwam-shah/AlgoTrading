"""
Step 6 — Trade Simulation Backtest
====================================
Reads per-stock OOS predictions.csv files and simulates a realistic
long-only strategy. Zero external dependencies beyond numpy/pandas.

Outputs:
  <run_dir>/backtest_results.csv   — per-stock P&L metrics
  <run_dir>/backtest_portfolio.csv — daily equity curve (top tradeable stocks)

Realistic NSE cost model (delivery trades):
  STT       : 0.1% buy + 0.1% sell = 0.20% round trip
  Brokerage : ~0.05% round trip (flat ₹20 @ avg ₹40K position)
  Exchange  : 0.004% round trip
  Total     : ~0.25% per round trip
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROUND_TRIP_COST = 0.0025   # 0.25% total (STT 0.20% + brokerage 0.05%)
ANNUAL_FACTOR   = 252

# v2 strategy parameters (match features.py horizon + Exp5 winner)
HOLD_DAYS_V2    = 10       # hold for 10 trading days to match the target horizon
META_THRESHOLD  = 0.60     # trade only when meta_prob >= this      # trading days


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, rf_annual: float = 0.065) -> float:
    """Annualised Sharpe ratio. RF = 6.5% (India 1yr T-bill approx)."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1 / ANNUAL_FACTOR) - 1
    excess   = returns - rf_daily
    return float(excess.mean() / excess.std() * math.sqrt(ANNUAL_FACTOR))


def _read_last_window_meta_auc(run_dir: Path, symbol: str) -> Optional[float]:
    """Read the validation AUC of the meta-labeller from the most recent window's
    calibration.json. Returns None if not available (meta skipped that window)."""
    # run_dir layout: <repo>/V3/06_results/runs/<run_id>
    # models layout : <repo>/V3/02_models/runs/<run_id>/<symbol>/window_NN/
    v3_root = run_dir.parents[2]   # → <repo>/V3
    models_dir = v3_root / "02_models" / "runs" / run_dir.name / symbol
    if not models_dir.exists():
        return None
    win_dirs = sorted(models_dir.glob("window_*"))
    for wd in reversed(win_dirs):
        cal = wd / "calibration.json"
        if not cal.exists():
            continue
        try:
            with open(cal) as _f:
                d = json.load(_f)
            mi = d.get("meta_info", {})
            if mi.get("trained"):
                return float(mi.get("val_auc", 0.0))
        except Exception:
            continue
    return None


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (0–1 scale, positive number)."""
    if len(equity) == 0:
        return 0.0
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def _simulate_stock(
    preds_df: pd.DataFrame,
    min_confidence: float = 0.58,
    hold_days: int = HOLD_DAYS_V2,
    meta_threshold: float = META_THRESHOLD,
) -> Optional[Dict]:
    """
    Simulate long-only trading for a single stock with a fixed holding period.

    v2 changes vs v1 (one-day hold):
      - Hold exactly `hold_days` trading days (matches the 5-day target horizon
        plus a 5-day trend-continuation buffer — see Exp5 sensitivity).
      - Gate by meta_prob >= meta_threshold (López de Prado meta-labeling).
        If meta_prob column is absent (legacy predictions.csv), skip that check.
      - No overlapping positions per stock: after entering at day D, skip any
        further signals until D + hold_days.

    Returns a dict of metrics, or None if too few trades.
    """
    df = preds_df.copy()

    # Deduplicate dates — keep highest window_id prediction per date
    # (most recently trained model is most relevant)
    if "window_id" in df.columns:
        df = df.sort_values("window_id").drop_duplicates("date", keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    # Build the price series available for exit. We need close at (D + hold_days)
    # — take from this df by shifting close_price by -hold_days.
    df["exit_price"] = df["close_price"].shift(-hold_days)

    # Gate: direction == UP, prob_up >= min_confidence, meta_prob >= threshold
    mask = (df["direction"] == "UP") & (df["prob_up"] >= min_confidence)
    if "meta_prob" in df.columns:
        # If meta column exists and is all 0.5 (legacy neutral), don't filter on it
        if df["meta_prob"].notna().any() and float(df["meta_prob"].std()) > 1e-6:
            mask &= (df["meta_prob"] >= meta_threshold)
    mask &= df["exit_price"].notna()

    candidate_idx = df.index[mask].tolist()
    if len(candidate_idx) < 5:
        return None

    # Skip overlapping trades — once entered, cooldown `hold_days` before next entry
    chosen_idx: list = []
    last_exit = -1
    for i in candidate_idx:
        if i < last_exit:
            continue
        chosen_idx.append(i)
        last_exit = i + hold_days

    if len(chosen_idx) < 5:
        return None

    trades_df = df.loc[chosen_idx].copy()
    trades_df["raw_return"] = (trades_df["exit_price"] - trades_df["close_price"]) / trades_df["close_price"]
    trades_df["net_return"] = trades_df["raw_return"] - ROUND_TRIP_COST

    returns = trades_df["net_return"].values

    # Equity curve — start at 1.0, apply each trade in sequence
    equity = np.cumprod(1 + np.concatenate([[0], returns]))

    # Core metrics
    n_trades   = len(returns)
    total_ret  = float(equity[-1] - 1)
    win_mask   = returns > 0
    win_rate   = float(win_mask.mean())
    avg_win    = float(returns[win_mask].mean()) if win_mask.any()  else 0.0
    avg_loss   = float(returns[~win_mask].mean()) if (~win_mask).any() else 0.0

    gross_wins  = returns[win_mask].sum()
    gross_loss  = abs(returns[~win_mask].sum())
    profit_factor = (gross_wins / gross_loss) if gross_loss > 0 else float("inf")

    mdd        = _max_drawdown(equity)
    sharpe     = _sharpe(returns)
    calmar     = total_ret / mdd if mdd > 0 else float("inf")

    # Binary direction accuracy (ignoring cost — just directional, hold-horizon)
    ref_exit = df["exit_price"].dropna()
    ref_entry = df["close_price"].loc[ref_exit.index]
    binary_acc = float((ref_exit.values >= ref_entry.values).mean()) if len(ref_exit) else 0.0
    # Among only UP-signal (executed) trades
    up_dir_acc = float((trades_df["exit_price"] >= trades_df["close_price"]).mean())

    # Hold-out period
    try:
        date_range = f"{df['date'].iloc[0][:10]} → {df['date'].iloc[-1][:10]}"
    except Exception:
        date_range = ""

    # Annualised return (assuming one trade per trading day on average)
    trading_days_per_trade = ANNUAL_FACTOR / max(n_trades, 1)
    holding_fraction = 1 / max(trading_days_per_trade, 1)
    ann_return = (1 + total_ret) ** (ANNUAL_FACTOR / max(n_trades, 1)) - 1

    return {
        "n_trades":       n_trades,
        "total_return":   round(total_ret, 4),
        "ann_return":     round(ann_return, 4),
        "win_rate":       round(win_rate, 4),
        "avg_win_pct":    round(avg_win * 100, 3),
        "avg_loss_pct":   round(avg_loss * 100, 3),
        "profit_factor":  round(min(profit_factor, 99.0), 3),
        "sharpe":         round(sharpe, 3),
        "max_drawdown":   round(mdd, 4),
        "calmar":         round(min(calmar, 99.0), 3),
        "binary_dir_acc": round(binary_acc, 4),
        "up_signal_acc":  round(up_dir_acc, 4),
        "date_range":     date_range,
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _bootstrap_accuracy_ci(
    outcomes: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for binary accuracy. Returns (mean, lower, upper)."""
    if len(outcomes) < 20:
        return float(outcomes.mean()) if len(outcomes) > 0 else 0.0, 0.0, 1.0
    rng  = np.random.default_rng(42)
    boot = np.array([
        rng.choice(outcomes, size=len(outcomes), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return float(outcomes.mean()), float(np.percentile(boot, alpha / 2 * 100)), float(np.percentile(boot, (1 - alpha / 2) * 100))


def _fetch_nifty_return(start_date: str, end_date: str) -> Optional[float]:
    """Fetch NIFTY50 buy-and-hold return for the period. Returns None on failure."""
    try:
        import yfinance as yf  # type: ignore
        nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if nifty.empty:
            return None
        # yfinance ≥0.2.x returns MultiIndex columns: ('Close', '^NSEI')
        if isinstance(nifty.columns, pd.MultiIndex):
            close = nifty["Close"].iloc[:, 0].dropna()
        else:
            close = nifty["Close"].dropna()
        if len(close) < 2:
            return None
        return round(float((close.iloc[-1] - close.iloc[0]) / close.iloc[0]), 4)
    except Exception:
        return None


# ── Portfolio equity curve ────────────────────────────────────────────────────

def _build_portfolio_curve(
    run_dir: Path,
    tradeable_symbols: List[str],
    min_confidence: float = 0.58,
    hold_days: int = HOLD_DAYS_V2,
    meta_threshold: float = META_THRESHOLD,
    n_max: int = 3,
) -> pd.DataFrame:
    """
    Equal-weight Top-N portfolio with fixed holding horizon.

    v2 mechanics (matches Exp5 winner: hold=10, t1=0.58, t2=0.60, n_max=3):
      - Each trading day, pick top-`n_max` signals by (prob_up * meta_prob)
        among tradeable symbols that pass both gates, not already held.
      - Enter at close, hold exactly `hold_days`, exit at close.
      - Equal weight 1/n_max; idle cash earns 0.
      - Entry cost applied once on entry day.
    """
    from collections import defaultdict

    # Collect price series per symbol (from predictions.csv close_price)
    sym_df: Dict[str, pd.DataFrame] = {}
    for sym in tradeable_symbols:
        p = run_dir / sym / "predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "window_id" in df.columns:
            df = df.sort_values("window_id").drop_duplicates("date", keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        sym_df[sym] = df

    if not sym_df:
        return pd.DataFrame(columns=["date", "equity", "daily_return"])

    # Union of dates
    all_dates = sorted({d for df in sym_df.values() for d in df["date"].tolist()})
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Per-symbol: map date → (idx, close, prob_up, meta_prob)
    stock_idx: Dict[str, Dict[str, int]] = {}
    for sym, df in sym_df.items():
        stock_idx[sym] = {row["date"]: i for i, row in df.iterrows()}

    # Slot-based simulation — up to n_max concurrent positions
    slots: List[Optional[Dict]] = [None] * n_max
    daily_ret = np.zeros(len(all_dates))

    # Build candidate signals per date
    per_day: Dict[str, list] = defaultdict(list)
    for sym, df in sym_df.items():
        mask = (df["direction"] == "UP") & (df["prob_up"] >= min_confidence)
        if "meta_prob" in df.columns and df["meta_prob"].notna().any() and float(df["meta_prob"].std()) > 1e-6:
            mask &= (df["meta_prob"] >= meta_threshold)
        for _, row in df[mask].iterrows():
            score = float(row["prob_up"] * row.get("meta_prob", 1.0))
            per_day[row["date"]].append((sym, score))

    for day_idx, d in enumerate(all_dates):
        # MtM open slots: use that symbol's close on day d vs close on day d-1
        for s_i, slot in enumerate(slots):
            if slot is None:
                continue
            sym = slot["symbol"]; df = sym_df[sym]
            idx_map = stock_idx[sym]
            if d in idx_map and idx_map[d] > 0:
                cur = float(df["close_price"].iloc[idx_map[d]])
                prev = float(df["close_price"].iloc[idx_map[d]-1])
                if prev > 0:
                    daily_ret[day_idx] += (cur / prev - 1) / n_max
            # Close if held long enough
            if day_idx >= slot["exit_day_idx"]:
                slots[s_i] = None

        # Consider new entries
        free = [i for i, s in enumerate(slots) if s is None]
        open_syms = {s["symbol"] for s in slots if s is not None}
        if free:
            cands = [c for c in sorted(per_day.get(d, []), key=lambda x: -x[1])
                     if c[0] not in open_syms]
            for sym, score in cands:
                if not free:
                    break
                if sym not in stock_idx or d not in stock_idx[sym]:
                    continue
                s_i = free.pop(0)
                df = sym_df[sym]
                entry_idx = stock_idx[sym][d]
                # Exit = hold_days later in this stock's own calendar (fallback to last row)
                exit_idx_in_stock = min(entry_idx + hold_days, len(df) - 1)
                exit_d = df["date"].iloc[exit_idx_in_stock]
                exit_day_idx = date_to_idx.get(exit_d, day_idx + hold_days * 2)
                slots[s_i] = {"symbol": sym, "entry_day_idx": day_idx,
                              "exit_day_idx": exit_day_idx}
                daily_ret[day_idx] -= ROUND_TRIP_COST / n_max

    daily = pd.DataFrame({"date": all_dates, "daily_return": daily_ret})
    daily["equity"] = (1 + daily["daily_return"]).cumprod()
    return daily


# ── Main entry point ──────────────────────────────────────────────────────────

def run_trade_backtest(
    run_dir: Path,
    min_confidence: float = 0.58,
    min_acc_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Run backtest for all symbols in run_dir that have predictions.csv.

    Returns a DataFrame with one row per stock containing P&L metrics.
    Also writes:
      - run_dir/backtest_results.csv
      - run_dir/backtest_portfolio.csv  (tradeable stocks only)
    """
    summary_path = run_dir / "summary.csv"
    oos_map: Dict[str, float] = {}
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        sdf = sdf[sdf["symbol"] != "AVERAGE"]
        oos_map = dict(zip(sdf["symbol"], sdf["oos_accuracy"]))

    rows: List[Dict] = []

    for pred_path in sorted(run_dir.glob("*/predictions.csv")):
        symbol = pred_path.parent.name
        try:
            df = pd.read_csv(pred_path)
            if len(df) < 10:
                continue
            metrics = _simulate_stock(df, min_confidence=min_confidence)
            if metrics is None:
                continue
            metrics["symbol"]       = symbol
            metrics["oos_accuracy"] = round(oos_map.get(symbol, 0.0), 4)
            # Surfaces meta validation AUC for analysis / dashboard ranking.
            meta_val_auc = _read_last_window_meta_auc(run_dir, symbol)
            metrics["meta_val_auc"] = round(meta_val_auc, 3) if meta_val_auc is not None else None
            # tradeable = directionally valid (>50%) AND actually profitable (sharpe>0).
            # We tested two relaxations (drop sharpe gate; replace with meta_val_auc>=0.50);
            # both *reduced* portfolio return on the production probabilities (production
            # ensemble has noisier UP-mass than Exp5's separately-trained primary). Keep
            # the proven gate; chase headroom via better probabilities, not gate tuning.
            metrics["tradeable"]    = (
                oos_map.get(symbol, 0.0) >= min_acc_threshold
                and metrics.get("sharpe", -999) > 0
            )
            rows.append(metrics)
        except Exception as exc:
            print(f"  [backtest] {symbol}: {exc}")

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # Cross-sectional rank: top 15 by Sharpe among stocks with OOS >= min_acc_threshold
    # AND Sharpe >= floor. Floor prevents padding the universe with negative-Sharpe losers
    # when only a handful of profitable stocks exist. Yields ≤15 symbols.
    CROSS_SECTIONAL_SHARPE_FLOOR = -0.1
    valid_for_rank = (
        (result_df["oos_accuracy"] >= min_acc_threshold)
        & (result_df["sharpe"] >= CROSS_SECTIONAL_SHARPE_FLOOR)
    )
    rank_df = result_df[valid_for_rank].sort_values("sharpe", ascending=False).head(15)
    cross_sectional_syms = set(rank_df["symbol"].tolist())
    result_df["cross_sectional_top15"] = result_df["symbol"].isin(cross_sectional_syms)
    result_df["sharpe_rank"] = (
        result_df["sharpe"].rank(method="dense", ascending=False).astype("Int64")
    )

    col_order = [
        "symbol", "oos_accuracy", "meta_val_auc", "tradeable", "cross_sectional_top15", "sharpe_rank",
        "n_trades", "total_return", "ann_return",
        "win_rate", "avg_win_pct", "avg_loss_pct", "profit_factor",
        "sharpe", "max_drawdown", "calmar",
        "binary_dir_acc", "up_signal_acc", "date_range",
    ]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    result_df = result_df.sort_values("sharpe", ascending=False)
    result_df.to_csv(run_dir / "backtest_results.csv", index=False)

    # Portfolio curve: only stocks marked tradeable (oos_acc>=floor AND sharpe>0).
    tradeable_syms = result_df[
        (result_df["tradeable"] == True) & (result_df["sharpe"] > 0)
    ]["symbol"].tolist()
    curve = pd.DataFrame()
    if tradeable_syms:
        curve = _build_portfolio_curve(run_dir, tradeable_syms, min_confidence)
        if not curve.empty:
            curve.to_csv(run_dir / "backtest_portfolio.csv", index=False)

    # ── Bootstrap CI across all tradeable UP-signal predictions ──────────────
    all_outcomes: List[int] = []
    for sym in tradeable_syms:
        p = run_dir / sym / "predictions.csv"
        if not p.exists():
            continue
        try:
            pdf = pd.read_csv(p)
            if "window_id" in pdf.columns:
                pdf = pdf.sort_values("window_id").drop_duplicates("date", keep="last")
            mask = (pdf["direction"] == "UP") & (pdf["prob_up"] >= min_confidence)
            pdf  = pdf[mask]
            # Bootstrap outcomes = "did the trade make money over the hold horizon"
            if "close_price" in pdf.columns:
                pdf_sorted = pdf.sort_values("date").reset_index(drop=True)
                pdf_sorted["exit_px"] = pdf_sorted["close_price"].shift(-HOLD_DAYS_V2)
                ok = pdf_sorted.dropna(subset=["exit_px"])
                outcomes = (ok["exit_px"] >= ok["close_price"]).astype(int).tolist()
                all_outcomes.extend(outcomes)
        except Exception:
            pass

    outcomes_arr = np.array(all_outcomes)
    acc_mean, ci_lower, ci_upper = _bootstrap_accuracy_ci(outcomes_arr)
    bootstrap_significant = bool(ci_lower > 0.50) if len(all_outcomes) >= 20 else False

    # ── NIFTY buy-and-hold comparison ─────────────────────────────────────────
    nifty_return: Optional[float] = None
    nifty_start = nifty_end = ""
    if not curve.empty and "date" in curve.columns:
        nifty_start = str(curve["date"].iloc[0])
        nifty_end   = str(curve["date"].iloc[-1])
        nifty_return = _fetch_nifty_return(nifty_start, nifty_end)

    # Real portfolio total return from the equity curve (slot-based top-3 simulation).
    portfolio_total_return: float = 0.0
    portfolio_sharpe:       float = 0.0
    portfolio_max_dd:       float = 0.0
    if not curve.empty and "equity" in curve.columns and len(curve) > 1:
        portfolio_total_return = float(curve["equity"].iloc[-1] - 1.0)
        _r = curve["daily_return"].values
        if _r.std() > 0:
            portfolio_sharpe = float(_r.mean() / _r.std() * math.sqrt(ANNUAL_FACTOR))
        _eq = curve["equity"].values
        _peak = np.maximum.accumulate(_eq)
        _dd = (_eq - _peak) / _peak
        portfolio_max_dd = float(-_dd.min()) if _dd.min() < 0 else 0.0

    avg_per_stock_return = (
        round(float(result_df[result_df["tradeable"]==True]["total_return"].mean()), 4)
        if not result_df[result_df["tradeable"]==True].empty else 0.0
    )

    # Save supplementary summary JSON
    bt_summary = {
        "bootstrap_acc_mean":    round(acc_mean, 4),
        "bootstrap_ci_lower":    round(ci_lower, 4),
        "bootstrap_ci_upper":    round(ci_upper, 4),
        "bootstrap_significant": bootstrap_significant,
        "bootstrap_n_signals":   len(all_outcomes),
        "nifty_return":          nifty_return,
        "nifty_start_date":      nifty_start,
        "nifty_end_date":        nifty_end,
        # Real equity-curve total return from _build_portfolio_curve (top-3 slot sim).
        "portfolio_total_return": round(portfolio_total_return, 4),
        "portfolio_sharpe":       round(portfolio_sharpe, 3),
        "portfolio_max_dd":       round(portfolio_max_dd, 4),
        # Mean of per-stock total returns among tradeable picks (legacy field).
        "avg_per_stock_return":  avg_per_stock_return,
        # Alias kept for backward-compat dashboards (= avg_per_stock_return).
        "portfolio_return":      avg_per_stock_return,
    }
    with open(run_dir / "backtest_summary.json", "w") as _f:
        json.dump(bt_summary, _f, indent=2)

    print(f"\n  Bootstrap CI (n={len(all_outcomes)} signals): "
          f"acc={acc_mean:.3f} 95%CI [{ci_lower:.3f}, {ci_upper:.3f}] "
          f"{'✅ significant' if bootstrap_significant else '○ not yet significant'}")
    if nifty_return is not None:
        print(f"  NIFTY buy-hold ({nifty_start[:10]} → {nifty_end[:10]}): {nifty_return:+.2%}")

    # Print summary
    td = result_df[result_df["tradeable"] == True]
    print(f"\n  ── Backtest Results (conf≥{min_confidence:.0%}, cost={ROUND_TRIP_COST:.2%} RT) ──")
    print(f"  {'Symbol':<14} {'OOS':>6} {'Trades':>6} {'TotalRet':>9} "
          f"{'WinRate':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*9} {'─'*8} {'─'*7} {'─'*7}")
    for _, r in result_df.head(20).iterrows():
        tag = "✅" if r["tradeable"] else "  "
        print(f"  {tag}{r['symbol']:<12} {r['oos_accuracy']:>6.1%} "
              f"{int(r['n_trades']):>6} {r['total_return']:>+9.2%} "
              f"{r['win_rate']:>8.1%} {r['sharpe']:>7.2f} {r['max_drawdown']:>7.2%}")

    pos_td = result_df[(result_df["tradeable"] == True) & (result_df["sharpe"] > 0)]
    if not pos_td.empty:
        print(f"\n  ── Profitable Tradeable Portfolio ({len(pos_td)} stocks, Sharpe>0) ──")
        print(f"  Avg Sharpe  : {pos_td['sharpe'].mean():.2f}")
        print(f"  Avg WinRate : {pos_td['win_rate'].mean():.1%}")
        print(f"  Avg TotalRet: {pos_td['total_return'].mean():+.2%}")
        print(f"  Avg MaxDD   : {pos_td['max_drawdown'].mean():.2%}")

    return result_df
