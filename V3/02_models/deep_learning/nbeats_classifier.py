"""
nbeats_classifier.py
================================================================================
NBEATSClassifier — N-BEATS generic architecture adapted for binary classification.

Original N-BEATS (Oreshkin et al. 2019, ICLR)
----------------------------------------------
N-BEATS was designed for univariate time series forecasting:
  - Input: lookback window of length L
  - Each block: 4 FC layers → split into backcast (size L) + forecast (size H)
  - Residual: next block input = current input - backcast
  - Final forecast = sum of all block forecasts → regression output

Adaptation for Multivariate Binary Classification
--------------------------------------------------
The core insight from N-BEATS is the "residual backcast" mechanism:
  each block explains away the portion of the input it can model,
  passing only the unexplained residual to the next block.
This hierarchical residual decomposition is preserved here.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features) → Flatten → (batch, SEQ_LEN * n_features)

For each of N_BLOCKS generic blocks:
  FC(256, relu) → FC(256, relu) → FC(256, relu) → FC(256, relu)
  Backcast  : FC(seq_len * n_features, linear)  ← reconstructs block input
  Forecast  : FC(64, relu)                       ← classification representation
  Residual  : next_input = block_input - backcast

Aggregate all block forecasts → Add
  Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)

Key Differences from Original N-BEATS
--------------------------------------
1. Multivariate: flattens the (seq_len × features) tensor instead of univariate L
2. Classification: sigmoid output instead of multi-step regression
3. Generic only: no interpretable (trend/seasonality) stack since we're not
   forecasting a time series — the "basis expansion" concept doesn't apply directly
4. Forecast representations are summed (not as a time-series horizon) and
   fed into a classification head

Performance Context
-------------------
N-BEATS achieved:
  - 11% improvement over statistical ensembles on M3/M4 benchmarks
  - 3% improvement over the M4 competition winner (Smyl's ES-RNN hybrid)
  - Pure deep learning — no domain-specific time series components required

The residual backcast mechanism allows each block to focus on patterns the
previous blocks could not model, similar to boosting but end-to-end trainable.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, NBEATS_PARAMS               # noqa: E402
from deep_learning.base_deep import BaseDLClassifier        # noqa: E402


class NBEATSClassifier(BaseDLClassifier):
    """N-BEATS generic residual blocks adapted for binary classification."""

    model_name = "NBEATS"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = NBEATS_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p         = NBEATS_PARAMS
        L2        = regularizers.L2(p["l2"])
        proj_dim  = p.get("proj_dim", 256)   # bottleneck projection dimension

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        # Flatten sequence into 1D vector
        flat = layers.Flatten(name="flatten")(inp)   # (batch, seq_len * n_features)

        # ── Input projection: (seq*feat) → proj_dim  ──────────────────────────
        # Fixes the core bug: raw input could be 1000-dim (20×50), making each
        # block's backcast a 1000-dim linear layer that can't reconstruct the
        # 256-dim FC stack's output. Projecting to 256 first means blocks operate
        # in a compressed space where backcast capacity matches FC capacity.
        block_input = layers.Dense(
            proj_dim, activation="relu",
            kernel_regularizer=L2, name="input_proj",
        )(flat)

        forecast_outputs = []

        for b in range(p["n_blocks"]):
            # ── FC stack (4 layers, matching N-BEATS paper generic config) ──────
            h = block_input
            for layer_idx in range(p["n_layers"]):
                h = layers.Dense(
                    p["fc_dim"], activation="relu",
                    kernel_regularizer=L2,
                    name=f"b{b}_fc{layer_idx}",
                )(h)

            # ── Backcast: reconstruct the projected block input ────────────────
            backcast = layers.Dense(
                proj_dim, activation="linear",   # matches proj_dim, not raw input_size
                name=f"b{b}_backcast",
            )(h)

            # ── Forecast: classification representation from this block ─────────
            forecast = layers.Dense(
                p["forecast_dim"], activation="relu",
                name=f"b{b}_forecast",
            )(h)
            forecast_outputs.append(forecast)

            # ── Residual: next block sees only what this block couldn't explain ─
            block_input = layers.Subtract(name=f"b{b}_residual")([block_input, backcast])

        # Aggregate all block forecasts (sum — matches N-BEATS paper)
        if len(forecast_outputs) > 1:
            agg = layers.Add(name="forecast_agg")(forecast_outputs)
        else:
            agg = forecast_outputs[0]

        # Classification head
        agg = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(agg)
        agg = layers.Dropout(p["dropout"], name="dense_drop")(agg)
        out = layers.Dense(1, activation="sigmoid", name="output")(agg)

        return keras.Model(inp, out, name="nbeats_classifier")
