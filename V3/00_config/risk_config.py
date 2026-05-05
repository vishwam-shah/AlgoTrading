"""
risk_config.py — Lazy loader for risk_config.yaml
==================================================
Single accessor used by every live-trading and backtest module so config drift
is impossible. Falls back to embedded defaults if PyYAML or the YAML are missing
(so unit tests don't have to ship YAML).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_YAML_PATH = _HERE / "risk_config.yaml"

_DEFAULTS: Dict[str, Any] = {
    "strategy": {"hold_days": 10, "min_confidence": 0.58, "meta_threshold": 0.60,
                 "entry_timing": "next_open", "cost_round_trip": 0.0025,
                 "slippage_one_way_bps": 5},
    "sizing": {"capital_default_inr": 500000, "max_position_pct": 0.34,
               "kelly_cap_full": 0.25, "kelly_haircut": 0.5,
               "vol_target_daily": 0.015, "min_lot": 1, "weighting": "inv_vol"},
    "limits": {"max_stock_exposure_pct": 0.34, "max_sector_exposure_pct": 0.40,
               "max_holdings": 3, "max_daily_loss_pct": 0.02,
               "max_drawdown_pct": 0.20, "max_slippage_pct": 0.003,
               "min_prob_up_floor": 0.52},
    "exits": {"time_stop_days": 10, "vol_stop_atr_mult": 2.5,
              "trailing_stop_pct": 0.04, "trailing_arm_pct": 0.03,
              "signal_decay_threshold": 0.50, "signal_decay_lookback": 2,
              "partial_take_profit_pct": 0.06, "partial_take_profit_size": 0.5},
    "universe": {"min_oos_accuracy": 0.50, "min_sharpe_for_tradeable": 0.0,
                 "cross_sectional_top_k": 15, "expand_when_strict_below": 3},
    "execution": {"market_open": "09:15", "market_close": "15:25",
                  "default_order_type": "LIMIT", "default_product": "CNC",
                  "default_validity": "DAY", "default_exchange": "NSE",
                  "rate_limit_per_sec": 16},
    "promotion": {"min_paper_trades": 40, "min_paper_days": 20,
                  "min_rolling_sharpe": 1.0, "max_rolling_drawdown": 0.10,
                  "max_slippage_drift_bps": 25, "max_calibration_brier_drift": 0.05,
                  "min_fill_rate": 0.90, "cooldown_after_breach_days": 5},
    "mode": "paper",
}


@lru_cache(maxsize=1)
def load() -> Dict[str, Any]:
    """Read risk_config.yaml once per process. Env TRADING_MODE wins over YAML."""
    cfg = dict(_DEFAULTS)
    if _YAML_PATH.exists():
        try:
            import yaml  # type: ignore
            with open(_YAML_PATH) as f:
                yam = yaml.safe_load(f) or {}
            for k, v in yam.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**cfg[k], **v}
                else:
                    cfg[k] = v
        except ImportError:
            pass  # PyYAML not installed — use defaults
        except Exception as e:
            print(f"  [risk_config] WARN — falling back to defaults: {e}")
    env_mode = os.getenv("TRADING_MODE", "").strip().lower()
    if env_mode in ("paper", "live"):
        cfg["mode"] = env_mode
    return cfg


def get(*path: str, default: Any = None) -> Any:
    """Dot-style accessor: get('strategy', 'hold_days')."""
    node: Any = load()
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


# Convenience constants for hot paths (re-evaluated on import)
def _hot():
    c = load()
    return {
        "HOLD_DAYS":           int(c["strategy"]["hold_days"]),
        "MIN_CONFIDENCE":      float(c["strategy"]["min_confidence"]),
        "META_THRESHOLD":      float(c["strategy"]["meta_threshold"]),
        "ENTRY_TIMING":        str(c["strategy"]["entry_timing"]),
        "COST_RT":             float(c["strategy"]["cost_round_trip"]),
        "MAX_HOLDINGS":        int(c["limits"]["max_holdings"]),
        "MAX_POSITION_PCT":    float(c["limits"]["max_stock_exposure_pct"]),
        "MAX_SECTOR_PCT":      float(c["limits"]["max_sector_exposure_pct"]),
        "MAX_SLIPPAGE_PCT":    float(c["limits"]["max_slippage_pct"]),
        "MAX_DAILY_LOSS_PCT":  float(c["limits"]["max_daily_loss_pct"]),
        "MAX_DD_PCT":          float(c["limits"]["max_drawdown_pct"]),
        "MIN_PROB_FLOOR":      float(c["limits"]["min_prob_up_floor"]),
        "MIN_OOS_ACC":         float(c["universe"]["min_oos_accuracy"]),
        "TOPK":                int(c["universe"]["cross_sectional_top_k"]),
        "EXPAND_BELOW":        int(c["universe"]["expand_when_strict_below"]),
        "TRAIL_PCT":           float(c["exits"]["trailing_stop_pct"]),
        "TRAIL_ARM":           float(c["exits"]["trailing_arm_pct"]),
        "VOL_STOP_ATR_MULT":   float(c["exits"]["vol_stop_atr_mult"]),
        "DECAY_THR":           float(c["exits"]["signal_decay_threshold"]),
        "DECAY_LOOKBACK":      int(c["exits"]["signal_decay_lookback"]),
        "PARTIAL_PROFIT_PCT":  float(c["exits"]["partial_take_profit_pct"]),
        "PARTIAL_PROFIT_SIZE": float(c["exits"]["partial_take_profit_size"]),
        "MODE":                str(c["mode"]),
    }


HOT = _hot()
