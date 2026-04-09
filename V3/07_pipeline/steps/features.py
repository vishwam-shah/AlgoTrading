"""
step 2 — Feature Engineering
=============================
260+ features from OHLCV + global cues + NSE calendar.
All rolling/EWM windows are backward-looking — zero look-ahead.

Cached at V3/01_data/features/raw/{symbol}_features.parquet.
Invalidated when raw data or global cues are newer than cache.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_V3_ROOT   = _STEPS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))

from config_v3 import (  # type: ignore  # noqa: E402
    RAW_DATA_DIR, FEAT_RAW_DIR, FEAT_SCALED_DIR,
    SECTOR_MAP, IT_SYMBOLS, BANKING_SYMBOLS,
    GLOBAL_CUES_FEATURES, USDINR_FEATURES, BANKING_CUES_FEATURES,
    RBI_MPC_DATES, BUDGET_DATES, RESULT_SEASON_MONTHS,
    MIN_MOVE, N_TOP_FEATURES,
    LGBM_FS_PARAMS, RANDOM_SEED,
)
from steps.download import load_parquet, save_parquet, last_thursday_of_month  # type: ignore

# Pre-parse calendar dates once (module-level — cheap)
_RBI_DATES    = pd.to_datetime(RBI_MPC_DATES)
_BUDGET_DATES = pd.to_datetime(BUDGET_DATES)


# ══════════════════════════════════════════════════════════════════════════════
#  NSE CALENDAR FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_nse_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    NSE-specific calendar features — pure date arithmetic, zero leakage.
    F&O expiry proximity, RBI MPC window, Budget window, result season.
    """
    from datetime import timedelta
    dates = pd.to_datetime(d["date"])

    def _days_to_expiry(dt: pd.Timestamp) -> int:
        yr, mo = dt.year, dt.month
        exp = last_thursday_of_month(yr, mo)
        if dt.date() > exp:
            mo2 = mo + 1 if mo < 12 else 1
            yr2 = yr if mo < 12 else yr + 1
            exp = last_thursday_of_month(yr2, mo2)
        return max(0, (exp - dt.date()).days)

    dte = dates.apply(_days_to_expiry)
    d["days_to_expiry"] = dte.values
    d["is_expiry_week"] = (dte <= 4).astype(int).values
    d["is_expiry_day"]  = (dte == 0).astype(int).values

    def _days_to_rbi(dt: pd.Timestamp) -> int:
        diffs = abs((_RBI_DATES - dt).days)
        return int(min(diffs.min(), 30))

    d["days_to_rbi"] = dates.apply(_days_to_rbi).values
    d["is_rbi_week"] = (d["days_to_rbi"] <= 3).astype(int)

    def _days_to_budget(dt: pd.Timestamp) -> int:
        diffs = abs((_BUDGET_DATES - dt).days)
        return int(min(diffs.min(), 60))

    d["days_to_budget"] = dates.apply(_days_to_budget).values
    d["is_budget_week"] = (d["days_to_budget"] <= 3).astype(int)
    d["is_result_season"] = dates.dt.month.isin(RESULT_SEASON_MONTHS).astype(int).values

    return d


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL MARKET CUES FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_global_cues_features(d: pd.DataFrame, global_cues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge global cues (S&P500, Nasdaq, VIX, DXY, Crude, Nikkei) into stock df.
    LEAKAGE SAFETY: US close T is available in India only on T+1 morning.
    We shift cue dates +1 day so merge_asof always uses prior-day US data.
    """
    cues = global_cues_df.copy()
    cues["date"] = pd.to_datetime(cues["date"]) + pd.Timedelta(days=1)
    cues = cues.sort_values("date").reset_index(drop=True)

    indian = pd.DataFrame({"date": pd.to_datetime(d["date"]), "_idx": np.arange(len(d))})
    indian = indian.sort_values("date")

    merged = pd.merge_asof(
        indian, cues, on="date", direction="backward",
        tolerance=pd.Timedelta("7 days"),
    ).sort_values("_idx").reset_index(drop=True)

    if "sp500_ret" in merged.columns:
        d["sp500_ret_prev"] = merged["sp500_ret"].values
        if "sp500_close" in merged.columns:
            sc = pd.Series(merged["sp500_close"].values)
            d["sp500_ret_5d"]  = sc.pct_change(5).values
            d["sp500_ret_20d"] = sc.pct_change(20).values

    if "nasdaq_ret" in merged.columns:
        d["nasdaq_ret_prev"] = merged["nasdaq_ret"].values
        if "nasdaq_close" in merged.columns:
            nc = pd.Series(merged["nasdaq_close"].values)
            d["nasdaq_ret_5d"] = nc.pct_change(5).values

    if "us_vix_close" in merged.columns:
        vix = pd.Series(merged["us_vix_close"].values)
        d["us_vix_level"]  = vix.values
        d["us_vix_ret_1d"] = vix.pct_change(1).values
        vix_ma = vix.rolling(20, min_periods=10).mean()
        vix_sd = vix.rolling(20, min_periods=10).std().replace(0, np.nan)
        vix_z  = (vix - vix_ma) / (vix_sd + 1e-10)
        d["us_vix_zscore"] = vix_z.values
        d["us_vix_spike"]  = (vix_z > 1.5).astype(float).values
        d["us_vix_regime"] = pd.cut(vix, bins=[-np.inf, 15, 25, np.inf],
                                    labels=[0, 1, 2]).astype(float).values

    if "dxy_ret" in merged.columns:
        d["dxy_ret_prev"] = merged["dxy_ret"].values
        if "dxy_close" in merged.columns:
            dx = pd.Series(merged["dxy_close"].values)
            d["dxy_ret_5d"]  = dx.pct_change(5).values
            d["dxy_ret_20d"] = dx.pct_change(20).values

    if "crude_ret" in merged.columns:
        d["crude_ret_prev"] = merged["crude_ret"].values
        if "crude_close" in merged.columns:
            cr = pd.Series(merged["crude_close"].values)
            d["crude_ret_5d"]  = cr.pct_change(5).values
            d["crude_ret_20d"] = cr.pct_change(20).values

    if "nikkei_ret" in merged.columns:
        d["nikkei_ret_prev"] = merged["nikkei_ret"].values
        if "nikkei_close" in merged.columns:
            nk = pd.Series(merged["nikkei_close"].values)
            d["nikkei_ret_5d"] = nk.pct_change(5).values

    return d


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build 260+ features from OHLCV + global cues + NSE calendar. Zero look-ahead."""
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]

    # 1. Returns
    d["ret_1d"]  = c.pct_change(1);  d["ret_2d"]  = c.pct_change(2)
    d["ret_5d"]  = c.pct_change(5);  d["ret_10d"] = c.pct_change(10)
    d["ret_20d"] = c.pct_change(20); d["log_ret"]  = np.log(c / c.shift(1))

    # 2. SMA / EMA ratios
    for p in [5, 10, 20, 50, 100, 200]:
        d[f"price_sma_{p}"] = c / (c.rolling(p).mean() + 1e-10)
        d[f"price_ema_{p}"] = c / (c.ewm(span=p, adjust=False).mean() + 1e-10)

    # 3. MACD (normalised)
    for fast, slow, sig in [(12, 26, 9), (5, 35, 5)]:
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        macd  = ema_f - ema_s
        msig  = macd.ewm(span=sig, adjust=False).mean()
        tag   = f"{fast}_{slow}"
        d[f"macd_pct_{tag}"]        = macd          / (c + 1e-10)
        d[f"macd_signal_pct_{tag}"] = msig          / (c + 1e-10)
        d[f"macd_hist_pct_{tag}"]   = (macd - msig) / (c + 1e-10)

    # 4. RSI
    for p in [7, 14, 21, 28]:
        delta = c.diff()
        up = delta.clip(lower=0).rolling(p).mean()
        dn = (-delta).clip(lower=0).rolling(p).mean()
        d[f"rsi_{p}"] = 100 - 100 / (1 + up / (dn + 1e-10))

    # 5. Bollinger Bands
    for p in [10, 20, 50]:
        sma = c.rolling(p).mean(); std = c.rolling(p).std()
        upper = sma + 2*std; lower = sma - 2*std
        d[f"bb_width_{p}"] = (upper - lower) / (sma + 1e-10)
        d[f"bb_pos_{p}"]   = (c - lower) / (upper - lower + 1e-10)

    # 6. ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    for p in [7, 14, 21]:
        d[f"atr_ratio_{p}"] = tr.rolling(p).mean() / (c + 1e-10)

    # 7. ADX / DI
    hd = h.diff(); ld = -l.diff()
    plus_dm  = np.where((hd > ld) & (hd > 0), hd, 0.0)
    minus_dm = np.where((ld > hd) & (ld > 0), ld, 0.0)
    for p in [14, 21]:
        tr_s = tr.rolling(p).sum()
        pdi  = 100 * pd.Series(plus_dm,  index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        mdi  = 100 * pd.Series(minus_dm, index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        d[f"adx_{p}"]     = dx.rolling(p).mean()
        d[f"plus_di_{p}"] = pdi; d[f"minus_di_{p}"] = mdi
        d[f"di_diff_{p}"] = pdi - mdi

    # 8. Stochastic
    for p in [9, 14, 21]:
        lo = l.rolling(p).min(); hi = h.rolling(p).max()
        k  = 100 * (c - lo) / (hi - lo + 1e-10)
        d[f"stoch_k_{p}"] = k; d[f"stoch_d_{p}"] = k.rolling(3).mean()

    # 9. CCI
    for p in [14, 20]:
        tp  = (h + l + c) / 3; stp = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d[f"cci_{p}"] = (tp - stp) / (0.015 * mad + 1e-10)

    # 10. Williams %R
    for p in [14, 21]:
        hi = h.rolling(p).max(); lo = l.rolling(p).min()
        d[f"willr_{p}"] = -100 * (hi - c) / (hi - lo + 1e-10)

    # 11. OBV ratio
    obv = (np.sign(c.diff()) * v).cumsum()
    d["obv_ratio"] = obv / (obv.rolling(20).mean() + 1e-10)

    # 12. Volume ratios
    for p in [5, 10, 20, 50]:
        d[f"vol_ratio_{p}"] = v / (v.rolling(p).mean() + 1e-10)
    d["vol_change"]    = v.pct_change()
    d["vol_change_5d"] = v.pct_change(5)

    # 13. Volatility
    log_r = np.log(c / c.shift(1))
    for p in [5, 10, 20, 50]:
        d[f"hist_vol_{p}"] = log_r.rolling(p).std() * np.sqrt(252)
    pk = (np.log(h / l))**2 / (4 * np.log(2))
    gk = 0.5*(np.log(h/l))**2 - (2*np.log(2)-1)*(np.log(c/o))**2
    for p in [10, 20]:
        d[f"parkinson_{p}"] = np.sqrt(pk.rolling(p).mean() * 252)
        d[f"gk_vol_{p}"]    = np.sqrt(gk.rolling(p).mean() * 252)

    # 14. Momentum / ROC
    for p in [3, 5, 10, 20, 50]:
        d[f"roc_{p}"] = c.pct_change(p)

    # 15. Candlestick
    body = (c - o).abs(); rng = (h - l).abs() + 1e-10
    d["body_size"]    = body / rng
    d["upper_shadow"] = (h - c.where(c >= o, o)) / rng
    d["lower_shadow"] = (c.where(c <= o, o) - l) / rng
    d["hl_range"]     = rng / (c + 1e-10)
    d["oc_return"]    = (c - o) / (o + 1e-10)
    d["gap"]          = (o - c.shift(1)) / (c.shift(1) + 1e-10)

    # 16. Statistical
    for p in [10, 20, 50]:
        d[f"skew_{p}"]   = log_r.rolling(p).skew()
        d[f"kurt_{p}"]   = log_r.rolling(p).kurt()
        d[f"zscore_{p}"] = (c - c.rolling(p).mean()) / (c.rolling(p).std() + 1e-10)

    # 17. Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f"ret_lag_{lag}"] = log_r.shift(lag)
        d[f"vol_lag_{lag}"] = (v / (v.rolling(20).mean() + 1e-10)).shift(lag)

    # 18. High/Low position
    for p in [20, 52, 126, 252]:
        hi = h.rolling(p).max(); lo = l.rolling(p).min()
        d[f"pos_hi_{p}"]  = (c - lo) / (hi - lo + 1e-10)
        d[f"dist_hi_{p}"] = (hi - c) / (c + 1e-10)
        d[f"dist_lo_{p}"] = (c - lo) / (c + 1e-10)

    # 19. Trend
    d["trend_20"]   = (c - c.shift(20)) / (c.shift(20) + 1e-10)
    d["trend_cons"] = log_r.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    sma50  = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    d["cross_ratio"] = sma50 / (sma200 + 1e-10)

    # 20. Calendar (cyclic encoding)
    d["dow_sin"] = np.sin(2*np.pi*d["date"].dt.dayofweek/5)
    d["dow_cos"] = np.cos(2*np.pi*d["date"].dt.dayofweek/5)
    d["mon_sin"] = np.sin(2*np.pi*d["date"].dt.month/12)
    d["mon_cos"] = np.cos(2*np.pi*d["date"].dt.month/12)
    d["week_of_year_sin"] = np.sin(2*np.pi*d["date"].dt.isocalendar().week.astype(int)/52)
    d["week_of_year_cos"] = np.cos(2*np.pi*d["date"].dt.isocalendar().week.astype(int)/52)

    # 21. Market regime (rule-based)
    vol_20r = log_r.rolling(20).std()
    vol_q33 = vol_20r.rolling(252, min_periods=60).quantile(0.33)
    vol_q66 = vol_20r.rolling(252, min_periods=60).quantile(0.66)
    regime  = pd.Series(1, index=d.index)
    regime[(sma50 > sma200) & (vol_20r < vol_q66)] = 2
    regime[(sma50 < sma200) & (vol_20r > vol_q33)] = 0
    d["market_regime"]  = regime
    d["regime_bull"]    = (regime == 2).astype(int)
    d["regime_bear"]    = (regime == 0).astype(int)
    d["trend_strength"] = (sma50 / (sma200 + 1e-10)) - 1

    # 22. Cross-sectional / sector
    if peer_returns is not None and symbol is not None:
        sector = SECTOR_MAP.get(symbol)
        if sector:
            peers = [s for s, sec in SECTOR_MAP.items()
                     if sec == sector and s != symbol and s in peer_returns]
            if peers:
                pf = pd.DataFrame({"date": d["date"].values}, index=d.index)
                for p in peers:
                    pf[p] = d["date"].map(peer_returns[p]).values
                sector_avg = pf[peers].mean(axis=1)
                all_rets   = pf[peers].copy(); all_rets["_own"] = d["ret_1d"].values
                d["cs_rank"]         = all_rets.rank(axis=1, pct=True)["_own"].values
                d["sector_avg_ret"]  = sector_avg.values
                d["rel_strength_1d"] = d["ret_1d"].values - sector_avg.values
                d["sector_mom_5d"]   = sector_avg.rolling(5).mean().values
                d["sector_mom_20d"]  = sector_avg.rolling(20).mean().values
                d["sector_vol_10d"]  = sector_avg.rolling(10).std().values
                _own_ret = pd.Series(d["ret_1d"].values, index=d.index)
                d["sector_corr_30d"] = _own_ret.rolling(30, min_periods=20).corr(sector_avg)
                d["sector_corr_60d"] = _own_ret.rolling(60, min_periods=30).corr(sector_avg)

    # 23. Nifty50 market context (disabled by default)
    if market_df is not None and not market_df.empty:
        mkt    = market_df.set_index("date")["close"]
        mkt_al = mkt.reindex(mkt.index.union(d["date"])).ffill()
        nifty_c = d["date"].map(mkt_al); nifty_c.index = d.index
        nifty_lr = np.log(nifty_c / nifty_c.shift(1))
        d["nifty_ret_1d"]       = nifty_c.pct_change(1)
        d["nifty_ret_5d"]       = nifty_c.pct_change(5)
        d["nifty_ret_20d"]      = nifty_c.pct_change(20)
        d["nifty_ma200_ratio"]  = nifty_c / (nifty_c.rolling(200, min_periods=60).mean() + 1e-10)
        nu = nifty_c.diff().clip(lower=0).rolling(14).mean()
        nd = (-nifty_c.diff()).clip(lower=0).rolling(14).mean()
        d["nifty_rsi_14"]       = 100 - 100 / (1 + nu / (nd + 1e-10))
        d["nifty_vol_20d"]      = nifty_lr.rolling(20).std() * np.sqrt(252)
        d["alpha_vs_nifty_1d"]  = d["ret_1d"]  - d["nifty_ret_1d"]
        d["alpha_vs_nifty_5d"]  = d["ret_5d"]  - d["nifty_ret_5d"]
        d["alpha_vs_nifty_20d"] = d["ret_20d"] - d["nifty_ret_20d"]

    # 24. USD/INR (IT sector)
    if usdinr_df is not None and (symbol is None or symbol in IT_SYMBOLS):
        fx    = usdinr_df.set_index("date")["usdinr_close"]
        fx_al = fx.reindex(fx.index.union(d["date"])).ffill()
        fx_c  = d["date"].map(fx_al); fx_c.index = d.index
        fx_r1 = fx_c.pct_change(1); fx_r5 = fx_c.pct_change(5)
        d["usdinr_ret_1d"]     = fx_r1; d["usdinr_ret_5d"]  = fx_r5
        d["usdinr_ret_20d"]    = fx_c.pct_change(20)
        fu = fx_c.diff().clip(lower=0).rolling(14).mean()
        fd = (-fx_c.diff()).clip(lower=0).rolling(14).mean()
        d["usdinr_rsi_14"]     = 100 - 100 / (1 + fu / (fd + 1e-10))
        d["usdinr_ma20_ratio"] = fx_c / (fx_c.rolling(20, min_periods=10).mean() + 1e-10)
        d["alpha_vs_usdinr_1d"] = d["ret_1d"] - fx_r1
        d["alpha_vs_usdinr_5d"] = d["ret_5d"] - fx_r5

    # 25. Global market cues (Phase 1)
    if global_cues_df is not None and not global_cues_df.empty:
        d = _add_global_cues_features(d, global_cues_df)

    # 26. NSE calendar features (Phase 1)
    d = _add_nse_calendar_features(d)

    # 27. GARCH(1,1) conditional volatility
    try:
        from arch import arch_model as _arch_model
        _lr_g = np.log(c / c.shift(1)).fillna(0).values * 100
        _am   = _arch_model(_lr_g, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        _res  = _am.fit(disp="off", show_warning=False, update_freq=0)
        _cond_vol  = _res.conditional_volatility
        _std_resid = _res.std_resid
        d["garch_vol"]      = _cond_vol
        d["garch_surprise"] = _std_resid
        _gv  = pd.Series(_cond_vol, index=d.index)
        _q33 = _gv.rolling(252, min_periods=60).quantile(0.33)
        _q67 = _gv.rolling(252, min_periods=60).quantile(0.67)
        d["garch_vol_regime"] = np.where(_gv > _q67, 2.0,
                                np.where(_gv < _q33, 0.0, 1.0))
    except Exception:
        pass

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def feature_cols(df: pd.DataFrame) -> List[str]:
    """Return names of feature columns (excludes OHLCV, date, symbol, target)."""
    exclude = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1=up>MIN_MOVE, 0=down<-MIN_MOVE, NaN=neutral (dropped)."""
    df = df.copy()
    next_ret = (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)
    df["target"] = np.where(next_ret >  MIN_MOVE, 1.0,
                   np.where(next_ret < -MIN_MOVE, 0.0, np.nan))
    return df


def get_or_compute_features(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return features from cache when fresh, else recompute.
    Cache stale if raw data or global_cues.parquet is newer than feature cache.
    """
    raw_path    = RAW_DATA_DIR    / f"{symbol}.parquet"
    feat_path   = FEAT_RAW_DIR   / f"{symbol}_features.parquet"
    scaled_path = FEAT_SCALED_DIR / f"{symbol}_scaled.parquet"
    gcues_path  = RAW_DATA_DIR   / "global_cues.parquet"

    if not force_recompute and feat_path.exists() and raw_path.exists():
        raw_mtime   = raw_path.stat().st_mtime
        feat_mtime  = feat_path.stat().st_mtime
        gcues_mtime = gcues_path.stat().st_mtime if gcues_path.exists() else 0.0
        if feat_mtime >= raw_mtime and feat_mtime >= gcues_mtime:
            cached = load_parquet(feat_path)
            if cached is not None and not cached.empty:
                print(f"  {symbol:<12} ✓ features cached  ({len(cached)} rows × {len(feature_cols(cached))} feat)")
                return cached

    print(f"  {symbol:<12} ↻ computing features ...", end=" ", flush=True)
    df    = compute_features(raw_df, symbol=symbol, peer_returns=peer_returns,
                             market_df=market_df, usdinr_df=usdinr_df,
                             global_cues_df=global_cues_df)
    df    = add_target(df)
    fcols = feature_cols(df)
    df    = df.dropna(subset=["target"])
    df    = df.dropna(subset=fcols, thresh=len(fcols) - 5)
    df[fcols] = df[fcols].fillna(df[fcols].median())
    df    = df.reset_index(drop=True)

    save_parquet(df, feat_path)

    try:
        from sklearn.preprocessing import RobustScaler
        X_sc  = RobustScaler().fit_transform(np.nan_to_num(df[fcols].values.astype(float), nan=0.0))
        sc_df = pd.DataFrame(X_sc, columns=fcols)
        sc_df.insert(0, "date",   df["date"].values)
        sc_df.insert(1, "target", df["target"].values)
        save_parquet(sc_df, scaled_path)
    except Exception:
        pass

    print(f"✓ {len(df)} rows × {len(fcols)} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_top_features(
    X: "np.ndarray",
    y: "np.ndarray",
    feature_names: List[str],
    fs_window: dict,
    symbol: str = "",
    top_n: int = N_TOP_FEATURES,
) -> List[str]:
    """
    Select top-N features by LightGBM split importance.
    Zero leakage: uses only train+val data from fs_window.
    Force-includes global cues, IT/banking sector features.
    """
    from sklearn.preprocessing import RobustScaler
    from lightgbm import LGBMClassifier, early_stopping as lgb_es, log_evaluation as lgb_log

    ws, we = fs_window["train_start"], fs_window["train_end"]
    vs, ve = fs_window["val_start"],   fs_window["val_end"]

    sc   = RobustScaler()
    X_tr = np.nan_to_num(sc.fit_transform(X[ws:we]), nan=0.0)
    X_va = np.nan_to_num(sc.transform(X[vs:ve]),     nan=0.0)

    try:
        fs_mdl = LGBMClassifier(**LGBM_FS_PARAMS)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fs_mdl.fit(X_tr, y[ws:we],
                       eval_set=[(X_va, y[vs:ve])],
                       callbacks=[lgb_es(40, verbose=False), lgb_log(period=-1)])
        importances = fs_mdl.feature_importances_

        forced_feats: List[str] = list(GLOBAL_CUES_FEATURES)
        if symbol in IT_SYMBOLS:
            forced_feats += USDINR_FEATURES
        if symbol in BANKING_SYMBOLS:
            forced_feats += BANKING_CUES_FEATURES

        forced     = [f for f in forced_feats if f in feature_names]
        forced_idx = {feature_names.index(f) for f in forced}
        slots      = max(top_n - len(forced), 1)
        others     = [i for i in np.argsort(importances)[::-1] if i not in forced_idx]
        combined   = sorted(forced_idx | set(others[:slots]))
        return [feature_names[i] for i in combined]
    except Exception:
        return feature_names
