"""
step 5 — Next-Day Prediction
==============================
Two entry points:

  predict_next_day()   — lightweight, no network calls, used by pipeline Step 5
                         for all 100 symbols in parallel. Returns a compact dict.

  predict_with_news()  — full inference with FinBERT news sentiment, earnings
                         proximity, key signal display, and price range estimate.
                         Used by `orchestrator.py --predict SYMBOL`.

Both share the same model-loading and preprocessing stack.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_V3_ROOT   = _STEPS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "02_models"))
sys.path.insert(0, str(_V3_ROOT / "01_data" / "news"))
sys.path.insert(0, str(_V3_ROOT / "01_data"))

from config_v3 import (  # type: ignore  # noqa: E402
    MODELS_PROD_DIR, CONFIDENCE_THRESHOLD, DL_SEQ_LEN, RAW_DATA_DIR,
)
from steps.features import compute_features  # type: ignore
from steps.train    import apply_temperature, _DL_AVAILABLE, _DL_CLASSES  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_artifacts(symbol: str) -> Optional[Dict]:
    """Load all saved artefacts for symbol. Returns None if no model exists."""
    prod_path = MODELS_PROD_DIR / symbol
    if not (prod_path / "metadata.json").exists():
        return None

    try:
        with open(prod_path / "metadata.json") as f:
            meta = json.load(f)
    except Exception:
        return None

    scaler_path = prod_path / "scaler.pkl"
    if not scaler_path.exists():
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    pca = None
    if (prod_path / "pca.pkl").exists():
        with open(prod_path / "pca.pkl", "rb") as f:
            pca = pickle.load(f)

    wb = None
    if (prod_path / "winsor_bounds.pkl").exists():
        with open(prod_path / "winsor_bounds.pkl", "rb") as f:
            wb = pickle.load(f)

    temperature = 1.0
    if (prod_path / "calibration.json").exists():
        try:
            with open(prod_path / "calibration.json") as f:
                temperature = float(json.load(f).get("temperature", 1.0))
        except Exception:
            pass

    dl_meta = {}
    if (prod_path / "dl_meta.json").exists():
        try:
            with open(prod_path / "dl_meta.json") as f:
                dl_meta = json.load(f)
        except Exception:
            pass

    tree_models = {}
    for pkl in prod_path.glob("*.pkl"):
        if pkl.stem in ("scaler", "meta_model", "pca", "winsor_bounds", "secondary"):
            continue
        try:
            with open(pkl, "rb") as f:
                tree_models[pkl.stem] = pickle.load(f)
        except Exception:
            pass

    # Trade-selection secondary (López de Prado meta-labeling). Optional — only
    # present once a v2-pipeline run has completed for this symbol.
    secondary_model = None
    sec_path = prod_path / "secondary.pkl"
    if sec_path.exists():
        try:
            with open(sec_path, "rb") as f:
                secondary_model = pickle.load(f)
        except Exception:
            secondary_model = None

    return {
        "prod_path":       prod_path,
        "feat_cols":       meta["feature_names"],
        "scaler":          scaler,
        "pca":             pca,
        "wb":              wb,
        "temperature":     temperature,
        "dl_meta":         dl_meta,
        "tree_models":     tree_models,
        "secondary_model": secondary_model,
        "meta":            meta,
    }


def _run_inference(arts: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """
    Run ensemble inference on the preprocessed feature DataFrame.
    Returns dict with probs_dict, avg_prob, raw_avg_prob, regime_val/lbl.
    """
    feat_cols  = arts["feat_cols"]
    scaler     = arts["scaler"]
    pca        = arts["pca"]
    wb         = arts["wb"]
    temperature = arts["temperature"]
    dl_meta    = arts["dl_meta"]
    tree_models = arts["tree_models"]
    prod_path  = arts["prod_path"]

    dl_seq_len = int(dl_meta.get("seq_len",    DL_SEQ_LEN))
    dl_n_feat  = int(dl_meta.get("n_features", len(feat_cols)))
    dl_models  = dl_meta.get("dl_models",    [])
    meta_cols  = dl_meta.get("meta_columns", [])

    if len(df) < max(1, dl_seq_len):
        return None

    X_raw_N = df[feat_cols].iloc[-dl_seq_len:].values.astype(float)
    X_raw_N = np.nan_to_num(X_raw_N, nan=0.0)
    if wb is not None:
        X_raw_N = np.clip(X_raw_N, wb[0], wb[1])
    X_sc_N = np.clip(scaler.transform(X_raw_N), -5, 5)
    X_sc_1 = X_sc_N[[-1]]
    X_tree = pca.transform(X_sc_1) if pca is not None else X_sc_1
    X_seq  = (pca.transform(X_sc_N) if pca is not None else X_sc_N)[np.newaxis, :, :]

    probs_dict: Dict[str, float] = {}

    for name, mdl in tree_models.items():
        try:
            p = mdl.predict_proba(X_tree)
            probs_dict[name] = float(
                p[:, 1][0] if (hasattr(p, "ndim") and p.ndim == 2 and p.shape[1] == 2)
                else np.ravel(p)[0]
            )
        except Exception:
            pass

    if _DL_AVAILABLE and dl_models:
        _name_to_cls = {dn: cls for cls, dn in _DL_CLASSES}
        for dl_name in dl_models:
            keras_path = prod_path / f"{dl_name.lower()}.keras"
            if not keras_path.exists() or dl_name not in _name_to_cls:
                continue
            try:
                mdl = _name_to_cls[dl_name](seq_len=dl_seq_len, n_features=dl_n_feat)
                mdl.load(str(keras_path))
                p = mdl.predict_proba(X_seq)
                probs_dict[dl_name] = float(np.ravel(p)[0])
            except Exception:
                pass

    if not probs_dict:
        return None

    raw_avg_prob = float(np.mean(list(probs_dict.values())))

    # Meta-stacker
    meta_path = prod_path / "meta_model.pkl"
    if meta_path.exists() and meta_cols:
        try:
            with open(meta_path, "rb") as f:
                mm = pickle.load(f)
            mx = np.array([probs_dict[c] for c in meta_cols if c in probs_dict]).reshape(1, -1)
            if mx.shape[1] == mm.n_features_in_:
                raw_avg_prob = float(mm.predict_proba(mx)[0, 1])
        except Exception:
            pass

    cal_prob = float(apply_temperature(np.array([raw_avg_prob]), temperature)[0])

    # Regime detection + regime-specific LightGBM blend
    regime_val = int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 1
    regime_lbl = {0: "bear", 1: "sideways", 2: "bull"}.get(regime_val, "sideways")

    rlgb_path = prod_path / f"lgb_{regime_lbl}.pkl"
    if rlgb_path.exists():
        try:
            with open(rlgb_path, "rb") as f:
                rlgb = pickle.load(f)
            r_prob   = float(apply_temperature(
                np.array([float(rlgb.predict_proba(X_tree)[:, 1][0])]), temperature)[0])
            cal_prob = 0.6 * r_prob + 0.4 * cal_prob
        except Exception:
            pass

    # ── Meta-labeling secondary — "should we act on this UP signal?" ────────
    meta_prob = 0.5   # neutral when secondary is absent (no trade filter applied)
    sec = arts.get("secondary_model")
    if sec is not None:
        try:
            X_meta = np.column_stack([X_tree, np.array([[cal_prob]])])
            if hasattr(sec, "n_features_in_") and X_meta.shape[1] == sec.n_features_in_:
                meta_prob = float(sec.predict_proba(X_meta)[0, 1])
        except Exception:
            meta_prob = 0.5

    return {
        "probs_dict":   probs_dict,
        "raw_avg_prob": raw_avg_prob,
        "cal_prob":     cal_prob,
        "meta_prob":    meta_prob,
        "regime_val":   regime_val,
        "regime_lbl":   regime_lbl,
        "X_tree":       X_tree,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — PIPELINE PREDICT (lightweight, no news)
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_day(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    Load production model for `symbol` and predict tomorrow's direction.
    Lightweight — no network calls. Used by orchestrator Step 5 for all 100 symbols.

    Returns dict: symbol, last_date, last_close, direction, action,
                  confidence, avg_prob, signal_active, regime, regime_label, temperature.
    """
    arts = _load_artifacts(symbol)
    if arts is None:
        return None

    feat_cols = arts["feat_cols"]
    scaler    = arts["scaler"]

    df = compute_features(raw_df, symbol=symbol, peer_returns=peer_returns,
                          market_df=market_df, usdinr_df=usdinr_df,
                          global_cues_df=global_cues_df)
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 10)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df = df.reset_index(drop=True)

    if df.empty or len(feat_cols) != scaler.n_features_in_:
        return None

    inf = _run_inference(arts, df)
    if inf is None:
        return None

    avg_prob      = inf["cal_prob"]
    meta_prob     = inf.get("meta_prob", 0.5)
    direction     = 1 if avg_prob >= 0.5 else 0
    confidence    = avg_prob if direction == 1 else 1 - avg_prob
    # v2 gate: primary must pass AND (meta passes OR secondary absent → meta_prob=0.5)
    META_THRESHOLD = 0.60
    primary_ok    = confidence >= CONFIDENCE_THRESHOLD
    meta_ok       = (meta_prob >= META_THRESHOLD) or (arts.get("secondary_model") is None)
    signal_active = bool(primary_ok and meta_ok and direction == 1)
    price_info    = _compute_price_range(raw_df, avg_prob)
    last_date_str = str(df["date"].iloc[-1])[:10]
    try:
        last_dt = pd.Timestamp(last_date_str)
        next_trading_day = (last_dt + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    except Exception:
        next_trading_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    prev_close = float(price_info["prev_close"])
    range_low = float(price_info["low_est"])
    range_high = float(price_info["high_est"])
    predicted_price = float(price_info["point_estimate"])

    def _pct(target: float) -> float:
        if prev_close <= 0:
            return 0.0
        return ((target - prev_close) / prev_close) * 100.0

    return {
        "symbol":        symbol,
        "last_date":     last_date_str,
        "prediction_date": datetime.now().strftime("%Y-%m-%d"),
        "prediction_for": next_trading_day,
        "last_close":    float(raw_df["close"].iloc[-1]),
        "direction":     "UP" if direction == 1 else "DOWN",
        "action":        ("BUY" if direction == 1 else "SELL") if signal_active else "HOLD",
        "confidence":    round(confidence, 4),
        "avg_prob":      round(avg_prob, 4),
        "meta_prob":     round(meta_prob, 4),
        "predicted_price": round(predicted_price, 2),
        "range_low":     round(range_low, 2),
        "range_high":    round(range_high, 2),
        "predicted_move_pct": round(float(price_info["expected_move_pct"]), 2),
        "range_down_pct": round(_pct(range_low), 2),
        "range_up_pct":  round(_pct(range_high), 2),
        "atr_14":        round(float(price_info["atr"]), 2),
        "signal_active": signal_active,
        "regime":        inf["regime_val"],
        "regime_label":  inf["regime_lbl"],
        "temperature":   round(arts["temperature"], 3),
    }


def predict_worker(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns,
    market_df,
    usdinr_df,
    global_cues_df,
) -> Optional[Dict]:
    """Thread-safe wrapper for ThreadPoolExecutor in orchestrator Step 5."""
    try:
        return predict_next_day(
            symbol, raw_df, peer_returns=peer_returns,
            market_df=market_df, usdinr_df=usdinr_df,
            global_cues_df=global_cues_df,
        )
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: NEWS SENTIMENT (FinBERT India model)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_news_sentiment(symbol: str, verbose: bool = False) -> Dict:
    """Fetch Google News RSS for symbol and score with FinBERT India model."""
    _neutral = {
        "raw_score": 0.0, "positive_ratio": 0.0, "negative_ratio": 0.0,
        "n_articles": 0, "headlines": [], "prob_adjustment": 0.0,
    }
    try:
        from news_fetcher import NewsFeaturizer  # type: ignore
        nf     = NewsFeaturizer()
        result = nf.fetch_and_score(symbol, use_finbert=True)
        heads  = nf.fetch_google_news(symbol, max_articles=10)
        score  = float(result.get("raw_score", 0.0))

        if   score >  0.5: adj = +0.08
        elif score >  0.2: adj = +0.04
        elif score < -0.5: adj = -0.08
        elif score < -0.2: adj = -0.04
        else:              adj =  0.0

        if verbose:
            print(f"  [news] {result['n_articles']} articles  score={score:+.3f}  adj={adj:+.3f}")

        return {
            "raw_score":       score,
            "positive_ratio":  float(result.get("positive_ratio", 0.0)),
            "negative_ratio":  float(result.get("negative_ratio", 0.0)),
            "n_articles":      int(result.get("n_articles", 0)),
            "headlines":       [str(h) for h in heads[:5]],
            "prob_adjustment": adj,
        }
    except Exception as e:
        if verbose:
            print(f"  [news] skipped: {e}")
        return _neutral


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: EARNINGS PROXIMITY
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_earnings_info(symbol: str) -> Dict:
    """Live yfinance earnings date lookup + proximity flags."""
    _none = {"days_to_earnings": None, "pre_results_drift": 0,
             "post_results_day": 0, "earnings_date": None}
    try:
        from earnings_calendar import days_to_next_earnings, fetch_next_earnings_yf  # type: ignore
        today = _date.today()
        dte   = days_to_next_earnings(today, symbol=symbol, use_live=True)
        ed    = fetch_next_earnings_yf(symbol)
        return {
            "days_to_earnings":  dte,
            "pre_results_drift": 1 if 1 <= dte <= 5 else 0,
            "post_results_day":  1 if -2 <= dte <= 0 else 0,
            "earnings_date":     str(ed) if ed else None,
        }
    except Exception:
        return _none


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: KEY SIGNAL EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_key_signals(df: pd.DataFrame) -> Dict:
    """Pull most interpretable feature values from the last row."""
    last = df.iloc[-1]

    def _get(col, default=None):
        return float(last[col]) if col in df.columns and pd.notna(last[col]) else default

    regime_int = int(_get("market_regime", 1))
    return {
        "rsi_14":             _get("rsi_14"),
        "vol_ratio_20":       _get("vol_ratio_20"),
        "sp500_ret_prev":     _get("sp500_ret_prev"),
        "nasdaq_ret_prev":    _get("nasdaq_ret_prev"),
        "vix_close":          _get("us_vix_level"),
        "vix_zscore":         _get("us_vix_zscore"),
        "nifty50_ret_prev":   _get("nifty50_ret_prev"),
        "niftybank_ret_prev": _get("niftybank_ret_prev"),
        "niftybank_ret_5d":   _get("niftybank_ret_5d"),
        "alpha_vs_niftybank": _get("alpha_vs_niftybank_1d"),
        "bb_position":        _get("bb_pct"),
        "regime":             {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}.get(regime_int, "SIDEWAYS"),
        "regime_int":         regime_int,
        "sma20_signal":       "ABOVE" if _get("close_to_sma20", 0) > 0 else "BELOW",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: PRICE RANGE ESTIMATE (ATR-based)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_price_range(raw_df: pd.DataFrame, prob_up: float,
                         sentiment_score: float = 0.0, n: int = 14) -> Dict:
    """ATR-based price range estimate, sentiment-nudged."""
    if len(raw_df) < n + 1:
        pc = float(raw_df["close"].iloc[-1])
        return {"prev_close": pc, "point_estimate": pc,
                "low_est": pc, "high_est": pc, "atr": 0.0, "expected_move_pct": 0.0}

    h  = raw_df["high"].values[-n - 1:]
    lo = raw_df["low"].values[-n - 1:]
    c  = raw_df["close"].values[-n - 1:]
    tr = np.maximum(h[1:] - lo[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(lo[1:] - c[:-1])))
    atr = float(np.mean(tr))

    prev_close   = float(raw_df["close"].iloc[-1])
    edge         = (prob_up - 0.5) * 2
    dir_move     = edge * atr
    sent_nudge   = float(np.clip(sentiment_score, -1, 1)) * 0.25 * atr
    point_est    = prev_close + dir_move + sent_nudge
    exp_move_pct = ((point_est - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

    return {
        "prev_close":        round(prev_close, 2),
        "point_estimate":    round(point_est, 2),
        "low_est":           round(prev_close - atr * 0.65, 2),
        "high_est":          round(prev_close + atr * 0.65, 2),
        "atr":               round(atr, 2),
        "expected_move_pct": round(exp_move_pct, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def _print_prediction(r: Dict) -> None:
    """Pretty-print a full predict_with_news() result."""
    sym   = r["symbol"]
    sig   = r["signal"]
    conf  = r["confidence"] * 100
    dirn  = r["direction"]
    prev  = r["prev_close"]
    est   = r["estimated_close"]
    lo    = r["range_low"]
    hi    = r["range_high"]
    atr   = r["atr_14"]
    exp_  = r["expected_move_pct"]
    T     = r["temperature"]
    today = r["prediction_for"]
    thru  = r["data_through"]
    news  = r["news"]
    sigs  = r["key_signals"]
    earn  = r.get("earnings", {})

    sig_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
    d_icon   = "▲" if dirn == "UP" else "▼"
    LINE     = "═" * 64

    print(f"\n{LINE}")
    print(f"  {sig_icon}  {sym}  —  PREDICTION FOR {today}")
    print(LINE)
    print(f"  Data through    : {thru}\n")

    # Direction
    print(f"  DIRECTION       : {d_icon} {dirn}   ({conf:.1f}% confidence)")
    print(f"  SIGNAL          : {sig}")
    print(f"  Market Regime   : {sigs['regime']}")
    print(f"  Temperature T   : {T:.3f}\n")

    # Ensemble probabilities
    print(f"  ── Ensemble Probabilities (P(UP)) ──────────────────────")
    for name, prob in sorted(r["model_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {name:<22} {prob:.4f}  {bar}")
    print()
    print(f"  Raw ensemble    : {r['raw_avg_prob']:.4f}")
    print(f"  After temp cal  : {r['ensemble_prob_pre_news']:.4f}  (T={T:.3f})")
    print(f"  After sentiment : {r['ensemble_prob_final']:.4f}  "
          f"(news adj {news['prob_adjustment']:+.3f})\n")

    # News sentiment
    sc       = news["raw_score"]
    bar_str  = ("+" * max(0, int(sc * 10))) or ("-" * max(0, int(-sc * 10))) or "~"
    sent_lbl = "POSITIVE" if sc > 0.2 else ("NEGATIVE" if sc < -0.2 else "NEUTRAL")
    print(f"  ── News Sentiment ({news['n_articles']} articles) ──────────────────────")
    print(f"  Score           : {sc:+.3f}  [{bar_str}]  {sent_lbl}")
    print(f"  Bullish / Bearish: {news['positive_ratio']*100:.0f}% / {news['negative_ratio']*100:.0f}%")
    if news["headlines"]:
        print(f"  Top headlines:")
        for i, h in enumerate(news["headlines"][:5], 1):
            trunc = h[:72] + "..." if len(h) > 75 else h
            print(f"    {i}. {trunc}")
    else:
        print(f"  No headlines fetched.")
    print()

    # Earnings proximity
    dte = earn.get("days_to_earnings")
    if dte is not None:
        ed_str = earn.get("earnings_date", "")
        if   dte == 0:      lbl = "TODAY  ← results day"
        elif dte > 0:
            lbl = f"in {dte} days  ({ed_str})"
            if earn.get("pre_results_drift"): lbl += "  [PRE-RESULTS DRIFT +boost]"
        else:               lbl = f"{abs(dte)} days ago  ({ed_str})  [POST-RESULTS]"
        print(f"  ── Earnings Proximity ──────────────────────────────────")
        print(f"  Next results    : {lbl}\n")

    # Key signals
    print(f"  ── Key Signals (what drove the prediction) ─────────────")
    rsi = sigs.get("rsi_14")
    rsi_cmt = "OVERBOUGHT" if (rsi or 50) > 70 else ("OVERSOLD" if (rsi or 50) < 30 else "NEUTRAL")
    print(f"  RSI (14d)       : {f'{rsi:.1f}' if rsi is not None else 'N/A'}  [{rsi_cmt}]")

    vr = sigs.get("vol_ratio_20")
    vr_cmt = "HIGH" if (vr or 1.0) > 1.3 else ("LOW" if (vr or 1.0) < 0.8 else "AVERAGE")
    print(f"  Volume vs 20d   : {f'{vr:.2f}x' if vr is not None else 'N/A'}  [{vr_cmt}]")

    n5 = sigs.get("nifty50_ret_prev")
    nb = sigs.get("niftybank_ret_prev")
    nb5= sigs.get("niftybank_ret_5d")
    al = sigs.get("alpha_vs_niftybank")
    sp = sigs.get("sp500_ret_prev")
    nd = sigs.get("nasdaq_ret_prev")
    vx = sigs.get("vix_close")
    vz = sigs.get("vix_zscore")
    bb = sigs.get("bb_position")

    if n5 is not None: print(f"  Nifty50 prev    : {n5*100:+.2f}%")
    if nb is not None:
        nb_cmt = "STRONG" if abs(nb) > 0.015 else ("MODERATE" if abs(nb) > 0.007 else "FLAT")
        print(f"  NiftyBank prev  : {nb*100:+.2f}%  [{nb_cmt}]")
    if nb5 is not None: print(f"  NiftyBank 5d    : {nb5*100:+.2f}%")
    if al is not None:
        al_cmt = "outperforming" if al > 0.003 else ("underperforming" if al < -0.003 else "inline")
        print(f"  Alpha vs NBankIdx: {al*100:+.2f}%  [{al_cmt}]")
    if sp is not None: print(f"  S&P500 prev ret : {sp*100:+.2f}%")
    if nd is not None: print(f"  Nasdaq prev ret : {nd*100:+.2f}%")
    if vx is not None:
        vx_cmt = "HIGH FEAR" if (vx or 0) > 25 else ("ELEVATED" if (vx or 0) > 18 else "CALM")
        vz_str = f"  z={vz:.1f}" if vz is not None else ""
        print(f"  VIX             : {vx:.1f}{vz_str}  [{vx_cmt}]")
    if bb is not None:
        bb_cmt = "near upper band" if bb > 0.8 else ("near lower band" if bb < 0.2 else "mid range")
        print(f"  Bollinger %B    : {bb:.2f}  [{bb_cmt}]")

    sma_sig = sigs.get("sma20_signal", "N/A")
    print(f"  Price vs SMA20  : {sma_sig}\n")

    # Price range
    print(f"  ── Price Range Estimate (secondary, ATR-based) ─────────")
    print(f"  Prev Close      : ₹{prev:,.2f}")
    print(f"  Expected Move   : {exp_:+.2f}%  →  est. ₹{est:,.2f}")
    print(f"  Likely Range    : ₹{lo:,.2f}  –  ₹{hi:,.2f}")
    print(f"  ATR (14d)       : ₹{atr:,.2f}")
    print()
    print(f"  Note: Primary output is DIRECTION ({dirn}) not close price.")
    if prev > 0:
        print(f"  Range estimate has ~±{atr/prev*100:.1f}% uncertainty.")
    print(LINE)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL INFERENCE WITH NEWS + EARNINGS (replaces predict_today.py)
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_news(
    symbol: str,
    raw_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Full prediction for a single symbol with:
      - FinBERT news sentiment
      - Earnings proximity boost
      - Key signal extraction
      - ATR price range estimate

    If raw_df is None, downloads fresh data automatically.
    Returns the full result dict (same structure as predict_today.py did).
    """
    # --- Load artefacts ---
    arts = _load_artifacts(symbol)
    if arts is None:
        print(f"  [{symbol}] ✗ no production model at {MODELS_PROD_DIR / symbol}")
        print(f"             Run: python V3/07_pipeline/orchestrator.py --symbols {symbol}")
        return None

    feat_cols = arts["feat_cols"]
    scaler    = arts["scaler"]

    # --- Fresh data if not provided ---
    if raw_df is None:
        raw_df = _download_fresh(symbol, verbose=verbose)
    if raw_df is None or raw_df.empty:
        print(f"  [{symbol}] ✗ could not load data")
        return None

    # --- Load auxiliary data if not provided ---
    if global_cues_df is None:
        try:
            p = RAW_DATA_DIR / "global_cues.parquet"
            if p.exists():
                global_cues_df = pd.read_parquet(p)
                global_cues_df["date"] = pd.to_datetime(global_cues_df["date"]).astype("datetime64[us]")
        except Exception:
            pass

    if usdinr_df is None:
        try:
            p = RAW_DATA_DIR / "usdinr.parquet"
            if p.exists():
                usdinr_df = pd.read_parquet(p)
                usdinr_df["date"] = pd.to_datetime(usdinr_df["date"]).astype("datetime64[us]")
        except Exception:
            pass

    # --- Compute features ---
    if verbose:
        print(f"  [{symbol}] computing features ...")
    df = compute_features(raw_df, symbol=symbol,
                          usdinr_df=usdinr_df, global_cues_df=global_cues_df)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 10)
    df[feat_cols] = df[feat_cols].fillna(0.0)
    df = df.reset_index(drop=True)

    if df.empty or len(feat_cols) != scaler.n_features_in_:
        print(f"  [{symbol}] ✗ feature mismatch: "
              f"expected {scaler.n_features_in_}, got {len(feat_cols)}")
        return None

    # --- Key signals (before scaling — human-readable values) ---
    key_signals = _extract_key_signals(df)

    # --- Ensemble inference ---
    inf = _run_inference(arts, df)
    if inf is None:
        print(f"  [{symbol}] ✗ inference failed (not enough rows or model error)")
        return None

    ensemble_prob = inf["cal_prob"]    # pre-news calibrated probability

    # --- Earnings proximity ---
    earnings_info = _fetch_earnings_info(symbol)

    # --- News sentiment ---
    print(f"  [{symbol}] fetching news sentiment ...", end=" ", flush=True)
    news = _fetch_news_sentiment(symbol, verbose=verbose)
    print(f"✓ {news['n_articles']} articles  score={news['raw_score']:+.3f}")

    # --- Probability blending: 80% ensemble + 15% sentiment + earnings boost ---
    sentiment_prob = 0.5 + float(np.clip(news["raw_score"], -1, 1)) * 0.25
    final_prob     = 0.80 * ensemble_prob + 0.15 * sentiment_prob
    final_prob     = float(np.clip(final_prob + news["prob_adjustment"] * 0.5, 0.01, 0.99))

    if earnings_info["pre_results_drift"]:
        dte        = earnings_info["days_to_earnings"]
        boost      = 0.04 * (1.0 - (dte - 1) / 5.0)   # day 1=+4%, day 5≈0%
        final_prob = float(np.clip(final_prob + boost, 0.01, 0.99))
    elif earnings_info["post_results_day"]:
        final_prob = float(np.clip(final_prob - 0.03, 0.01, 0.99))

    direction     = "UP"   if final_prob >= 0.5 else "DOWN"
    confidence    = final_prob if final_prob >= 0.5 else 1.0 - final_prob
    signal_active = confidence >= CONFIDENCE_THRESHOLD

    # --- Price range ---
    price_info = _compute_price_range(raw_df, final_prob, news["raw_score"])

    from datetime import datetime as _dt
    last_date  = str(df["date"].iloc[-1])[:10]
    today_date = _dt.now().strftime("%Y-%m-%d")

    return {
        # Primary direction
        "symbol":          symbol,
        "prediction_for":  today_date,
        "data_through":    last_date,
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "signal":          ("BUY" if direction == "UP" else "SELL") if signal_active else "HOLD",
        "signal_active":   signal_active,

        # Ensemble internals
        "ensemble_prob_pre_news": round(ensemble_prob, 4),
        "ensemble_prob_final":    round(final_prob, 4),
        "raw_avg_prob":           round(inf["raw_avg_prob"], 4),
        "temperature":            round(arts["temperature"], 3),
        "regime":                 key_signals["regime"],
        "model_probs":            {k: round(v, 4) for k, v in inf["probs_dict"].items()},

        # Earnings
        "earnings": earnings_info,

        # News
        "news": {
            "raw_score":       round(news["raw_score"], 4),
            "positive_ratio":  round(news["positive_ratio"], 3),
            "negative_ratio":  round(news["negative_ratio"], 3),
            "n_articles":      news["n_articles"],
            "prob_adjustment": round(news["prob_adjustment"], 4),
            "headlines":       news["headlines"],
        },

        # Key signals
        "key_signals": key_signals,

        # Price range (secondary)
        "prev_close":        price_info["prev_close"],
        "estimated_close":   price_info["point_estimate"],
        "expected_move_pct": price_info["expected_move_pct"],
        "range_low":         price_info["low_est"],
        "range_high":        price_info["high_est"],
        "atr_14":            price_info["atr"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOADER (for --predict mode without prior pipeline run data)
# ══════════════════════════════════════════════════════════════════════════════

def _download_fresh(symbol: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Download latest OHLCV, appending to cached parquet if present."""
    import threading
    _lock = threading.Lock()

    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        from config_v3 import DATA_START_DATE  # type: ignore

        ticker    = f"{symbol}.NS"
        today     = datetime.now().strftime("%Y-%m-%d")
        save_path = RAW_DATA_DIR / f"{symbol}.parquet"
        existing  = None

        if save_path.exists():
            try:
                existing = pd.read_parquet(save_path)
                if "timestamp" in existing.columns:
                    existing = existing.rename(columns={"timestamp": "date"})
                existing["date"] = pd.to_datetime(existing["date"]).astype("datetime64[us]")
                last        = existing["date"].max()
                fetch_start = (last - timedelta(days=4)).strftime("%Y-%m-%d")
                next_day    = (last + timedelta(days=1)).strftime("%Y-%m-%d")
                if next_day >= today:
                    if verbose:
                        print(f"  [{symbol}] data up-to-date through {last.date()}")
                    return existing
            except Exception:
                existing    = None
                fetch_start = DATA_START_DATE
        else:
            fetch_start = DATA_START_DATE

        print(f"  [{symbol}] downloading {fetch_start} → {today} ...", end=" ", flush=True)
        with _lock:
            raw = yf.download(ticker, start=fetch_start, end=today,
                              progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            print("✗ no data")
            return existing

        df = raw.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        for dc in ["date", "datetime", "index", "price"]:
            if dc in df.columns:
                df = df.rename(columns={dc: "date"}); break
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"]
                if c in df.columns]
        df   = df[keep].sort_values("date").reset_index(drop=True)
        df   = df[df["close"] > 0].reset_index(drop=True)

        combined = (pd.concat([existing, df], ignore_index=True)
                    .drop_duplicates("date").sort_values("date").reset_index(drop=True)
                    if existing is not None and not existing.empty else df)
        combined.to_parquet(save_path, index=False)
        print(f"✓ {len(combined)} rows through {combined['date'].iloc[-1].date()}")
        return combined

    except Exception as e:
        print(f"✗ error: {e}")
        return None
