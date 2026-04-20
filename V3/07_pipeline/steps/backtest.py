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
ANNUAL_FACTOR   = 252      # trading days


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, rf_annual: float = 0.065) -> float:
    """Annualised Sharpe ratio. RF = 6.5% (India 1yr T-bill approx)."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1 / ANNUAL_FACTOR) - 1
    excess   = returns - rf_daily
    return float(excess.mean() / excess.std() * math.sqrt(ANNUAL_FACTOR))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (0–1 scale, positive number)."""
    if len(equity) == 0:
        return 0.0
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def _simulate_stock(
    preds_df: pd.DataFrame,
    min_confidence: float = 0.52,
) -> Optional[Dict]:
    """
    Simulate long-only trading for a single stock.

    Returns a dict of metrics, or None if too few trades.
    """
    df = preds_df.copy()

    # Deduplicate dates — keep highest window_id prediction per date
    # (most recently trained model is most relevant)
    if "window_id" in df.columns:
        df = df.sort_values("window_id").drop_duplicates("date", keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    # Filter to long signals above confidence threshold
    mask = (df["direction"] == "UP") & (df["prob_up"] >= min_confidence)
    trades_df = df[mask].copy()

    if len(trades_df) < 5:
        return None

    # Daily net return per trade (after round-trip costs)
    trades_df["raw_return"] = (
        (trades_df["next_close_price"] - trades_df["close_price"])
        / trades_df["close_price"]
    )
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

    # Binary direction accuracy (ignoring cost — just directional)
    binary_acc = float((df["next_close_price"] >= df["close_price"]).mean())
    # Among only UP-signal trades
    up_dir_acc = float(
        (trades_df["next_close_price"] >= trades_df["close_price"]).mean()
    )

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
    min_confidence: float = 0.52,
) -> pd.DataFrame:
    """
    Equal-weight portfolio of tradeable symbols.
    Returns daily equity curve as a DataFrame with columns: date, equity.
    """
    # Collect all (date, net_return) rows across tradeable symbols
    all_rows: List[Dict] = []
    for sym in tradeable_symbols:
        p = run_dir / sym / "predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "window_id" in df.columns:
            df = df.sort_values("window_id").drop_duplicates("date", keep="last")
        mask = (df["direction"] == "UP") & (df["prob_up"] >= min_confidence)
        df   = df[mask].copy()
        df["net_return"] = (
            (df["next_close_price"] - df["close_price"]) / df["close_price"]
        ) - ROUND_TRIP_COST
        for _, row in df.iterrows():
            all_rows.append({"date": str(row["date"])[:10], "symbol": sym,
                             "net_return": row["net_return"]})

    if not all_rows:
        return pd.DataFrame(columns=["date", "equity", "daily_return"])

    portfolio = pd.DataFrame(all_rows)
    # Daily portfolio return = mean return across all active signals that day
    daily = portfolio.groupby("date")["net_return"].mean().reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily.columns = ["date", "daily_return"]
    # Forward-fill equity on days with no trades (NaN daily_return → 0)
    daily["daily_return"] = daily["daily_return"].fillna(0.0)
    daily["equity"] = (1 + daily["daily_return"]).cumprod()
    return daily


# ── Main entry point ──────────────────────────────────────────────────────────

def run_trade_backtest(
    run_dir: Path,
    min_confidence: float = 0.52,
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
            # tradeable = directionally valid (>50%) AND actually profitable (sharpe>0)
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
    col_order = [
        "symbol", "oos_accuracy", "tradeable",
        "n_trades", "total_return", "ann_return",
        "win_rate", "avg_win_pct", "avg_loss_pct", "profit_factor",
        "sharpe", "max_drawdown", "calmar",
        "binary_dir_acc", "up_signal_acc", "date_range",
    ]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    result_df = result_df.sort_values("sharpe", ascending=False)
    result_df.to_csv(run_dir / "backtest_results.csv", index=False)

    # Portfolio curve: ONLY stocks with positive Sharpe (actually profitable)
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
            if "next_close_price" in pdf.columns and "close_price" in pdf.columns:
                outcomes = (pdf["next_close_price"] >= pdf["close_price"]).astype(int).tolist()
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
        "portfolio_return":      round(float(result_df[result_df["tradeable"]==True]["total_return"].mean()), 4)
                                 if not result_df[result_df["tradeable"]==True].empty else 0.0,
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
