"""
portfolio_optimizer.py — Risk-aware allocation across approved candidates
==========================================================================
Replaces "top-3 by score, equal weight" with a four-factor allocation:

  1. Volatility budgeting    — inverse-variance baseline weights
  2. Correlation penalty     — discount weight by avg correlation to picks
  3. Marginal contribution   — clip names whose MCR_i > MCR_cap
  4. Sector concentration    — enforce limits.max_sector_exposure_pct

Returns weights that:
  • sum to ≤ 1
  • respect per-stock and per-sector caps from risk_config.yaml
  • are deterministic given the same inputs (no random seeds)

Designed to be cheap (n ≤ 20 candidates), pure-numpy, no SciPy required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_config"))
from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

_LIVE_DIR = Path(__file__).resolve().parent
_RAW_DIR  = _LIVE_DIR.parent / "01_data" / "raw"

MAX_PER_STOCK   = float(_RC["MAX_POSITION_PCT"])
MAX_PER_SECTOR  = float(_RC["MAX_SECTOR_PCT"])
MAX_HOLDINGS    = int(_RC["MAX_HOLDINGS"])
WEIGHTING       = str(_rcget("sizing", "weighting", default="inv_vol"))


# Lazy import to avoid circulars
def _sector_map() -> Dict[str, str]:
    try:
        sys.path.insert(0, str(_LIVE_DIR))
        from risk_guard import SECTOR_MAP  # type: ignore
        return dict(SECTOR_MAP)
    except Exception:
        return {}


def _load_returns(symbol: str, lookback_days: int = 60) -> pd.Series:
    p = _RAW_DIR / f"{symbol}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    df = df.sort_values("date").tail(lookback_days + 1)
    rets = df["close"].pct_change().dropna()
    rets.index = pd.to_datetime(df["date"]).iloc[-len(rets):]
    return rets


def _build_panel(symbols: List[str], lookback_days: int = 60) -> pd.DataFrame:
    series = {s: _load_returns(s, lookback_days=lookback_days) for s in symbols}
    series = {s: r for s, r in series.items() if len(r) >= 20}
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).dropna()
    return df


def _inverse_vol(panel: pd.DataFrame) -> Dict[str, float]:
    sig = panel.std()
    if (sig == 0).any():
        sig = sig.replace(0, sig.median() or 1e-6)
    inv = 1.0 / sig
    w = inv / inv.sum()
    return w.to_dict()


def _correlation_penalty(panel: pd.DataFrame) -> Dict[str, float]:
    """Return scaling factor in [0.4, 1.0] per symbol based on avg corr to others."""
    if panel.shape[1] <= 1:
        return {c: 1.0 for c in panel.columns}
    C = panel.corr().fillna(0).copy()
    arr = C.values.copy()
    np.fill_diagonal(arr, 0.0)
    avg_corr = pd.Series(arr.mean(axis=1), index=C.index)
    # Map avg_corr ∈ [-1, 1] → scaling ∈ [1.0, 0.4] (high corr → smaller weight)
    scaled = (1.0 - 0.6 * np.clip(avg_corr, 0, 1))
    return scaled.to_dict()


def _apply_caps(weights: Dict[str, float],
                sector_of: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Project weights onto the feasible set (per-stock + per-sector caps)."""
    sector_of = sector_of or {}
    if not weights:
        return {}
    # Per-stock cap
    capped = {s: min(w, MAX_PER_STOCK) for s, w in weights.items()}
    # Per-sector cap (iterative scaling)
    for _ in range(8):  # 8 iterations is more than enough
        sec_total: Dict[str, float] = {}
        for s, w in capped.items():
            sec = sector_of.get(s, "Other")
            sec_total[sec] = sec_total.get(sec, 0.0) + w
        scale_needed = False
        for sec, total in sec_total.items():
            if total > MAX_PER_SECTOR:
                scale = MAX_PER_SECTOR / total
                for s in capped:
                    if sector_of.get(s, "Other") == sec:
                        capped[s] *= scale
                scale_needed = True
        if not scale_needed:
            break
    # Re-cap per-stock after sector scaling
    capped = {s: min(w, MAX_PER_STOCK) for s, w in capped.items()}
    return capped


def _normalise(weights: Dict[str, float], target_sum: float = 1.0) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        return weights
    return {k: v * target_sum / s for k, v in weights.items()}


def _marginal_risk_clip(weights: Dict[str, float], panel: pd.DataFrame,
                        mcr_cap: float = 0.45) -> Dict[str, float]:
    """
    Clip weights whose marginal contribution to portfolio variance exceeds
    `mcr_cap` of total variance. Prevents one name dominating risk.
    """
    if panel.empty or len(weights) < 2:
        return weights
    syms = [s for s in weights.keys() if s in panel.columns]
    if not syms:
        return weights
    Σ = panel[syms].cov().values
    w = np.array([weights[s] for s in syms])
    port_var = float(w @ Σ @ w)
    if port_var <= 0:
        return weights
    mcr = (Σ @ w) * w / port_var
    out = dict(weights)
    for sym, m in zip(syms, mcr):
        if m > mcr_cap:
            out[sym] *= mcr_cap / m
    return out


def allocate(
    candidates: List[Dict],
    capital: float,
    *,
    lookback_days: int = 60,
) -> List[Dict]:
    """
    Allocate `capital` across `candidates` using risk-aware weighting.

    Each candidate dict must include `symbol` and `price`; optional `score`
    (used as a tilt) and `atr_pct` (used by inverse-vol baseline if returns
    panel is unavailable).

    Returns the same dict shape with `weight`, `target_inr`, `qty`, `target_pct`,
    `weight_components` populated.
    """
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))[:MAX_HOLDINGS]
    syms = [c["symbol"] for c in candidates]
    panel = _build_panel(syms, lookback_days=lookback_days)
    sector_of = {s: _sector_map().get(s, "Other") for s in syms}

    if WEIGHTING == "equal" or panel.empty:
        # Equal weight fallback (also when historical returns are missing).
        base = {s: 1.0 / len(syms) for s in syms}
        components = {s: {"baseline": "equal"} for s in syms}
    else:
        ivw = _inverse_vol(panel)
        corr = _correlation_penalty(panel)
        # Score tilt: scores already sorted; convert to a soft tilt in [0.85, 1.15]
        scores = np.array([float(c.get("score", 0.5)) for c in candidates])
        if scores.std() > 0:
            tilt_arr = 1 + 0.3 * (scores - scores.mean()) / scores.std()
            tilt_arr = np.clip(tilt_arr, 0.85, 1.15)
        else:
            tilt_arr = np.ones_like(scores)
        tilt = {s: float(t) for s, t in zip(syms, tilt_arr)}
        base = {s: ivw.get(s, 1.0/len(syms)) * corr.get(s, 1.0) * tilt[s] for s in syms}
        components = {s: {"inv_vol": round(ivw.get(s, 0), 4),
                          "corr_penalty": round(corr.get(s, 1.0), 4),
                          "tilt": round(tilt[s], 4)} for s in syms}

    base = _normalise(base)
    base = _marginal_risk_clip(base, panel)
    base = _apply_caps(base, sector_of=sector_of)
    base = _normalise(base, target_sum=min(1.0, sum(base.values()) or 1.0))

    out: List[Dict] = []
    for c in candidates:
        sym = c["symbol"]
        w = float(base.get(sym, 0.0))
        if w <= 0:
            continue
        price = float(c["price"])
        target_inr = capital * w
        qty = max(1, int(target_inr / price))
        out.append({
            **c,
            "weight":            round(w, 4),
            "target_inr":        round(target_inr, 0),
            "qty":               qty,
            "order_value":       round(qty * price, 0),
            "target_pct":        round(w * 100, 2),
            "sector":            sector_of.get(sym, "Other"),
            "weight_components": components.get(sym, {}),
        })
    return out
