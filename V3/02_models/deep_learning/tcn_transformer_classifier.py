"""
tcn_transformer_classifier.py
================================================================================
TCNTransformerClassifier — Dilated causal TCN + Transformer encoder.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features)
  Dense(64)  ← project features to d_model=64
  TCN Block dilation=1 : Conv1D(64,3,causal,d=1) → LN → Dropout → Add(residual)
  TCN Block dilation=2 : same, dilation_rate=2
  TCN Block dilation=4 : same, dilation_rate=4
  MultiHeadAttention(4 heads, key_dim=16, dropout=0.1) + Add + LayerNorm
  FFN: Dense(128, relu) → Dropout → Dense(64) + Add + LayerNorm
  GlobalAveragePooling1D
  Dense(32, relu)  →  Dropout(0.2)  →  Dense(1, sigmoid)

Design Rationale
----------------
- TCN layers compress local temporal patterns (3-5 day candlestick formations)
  into a d_model-dimensional representation at each time step.
- MultiHeadAttention then learns which past time steps are most informative
  for the current prediction, regardless of distance (global receptive field).
- This is the key advantage over pure LSTM: attention weights are interpretable
  and global, while TCN residuals provide fast local feature extraction.
- 4 attention heads × key_dim=16 = d_model=64 (standard configuration).
- GlobalAveragePooling aggregates across the time dimension before the final
  classification head — more stable than [CLS] token for short sequences.
- No causal mask in attention: all features are already lagged (backward-only),
  so attending across the full window is leakage-free.

Paper Reference Context
-----------------------
Temporal Fusion Transformer (Lim et al. 2021) combines LSTM local processing
with self-attention for long-range dependency capture and achieves "significant
performance improvements over existing benchmarks" on multi-horizon forecasting.
This model adopts the same TCN+Attention hybrid philosophy but adapts it for
binary classification on OHLCV sequences with pre-computed technical features.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, TCN_TRANSFORMER_PARAMS       # noqa: E402
from deep_learning.base_deep import BaseDLClassifier         # noqa: E402


class TCNTransformerClassifier(BaseDLClassifier):
    """Dilated causal TCN + Transformer encoder binary classifier."""

    model_name = "TCN_Transformer"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = TCN_TRANSFORMER_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = TCN_TRANSFORMER_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        # Project n_features → d_model (uniform channel size for residuals)
        x = layers.Dense(p["d_model"], kernel_regularizer=L2, name="proj")(inp)

        # TCN residual blocks
        for i, dilation in enumerate(p["dilations"]):
            res = x
            x = layers.Conv1D(
                p["d_model"], p["kernel_size"],
                padding="causal", dilation_rate=dilation,
                activation="relu", kernel_regularizer=L2,
                name=f"tcn_{i}_conv",
            )(x)
            x = layers.LayerNormalization(name=f"tcn_{i}_ln")(x)
            x = layers.Dropout(p["dropout"], name=f"tcn_{i}_drop")(x)
            x = layers.Add(name=f"tcn_{i}_res")([x, res])

        # Transformer encoder layer
        # MultiHeadAttention: attends across time steps (full window, no causal mask)
        attn = layers.MultiHeadAttention(
            num_heads=p["num_heads"],
            key_dim=p["d_model"] // p["num_heads"],
            dropout=p["dropout"],
            name="mha",
        )(x, x)
        x = layers.LayerNormalization(name="ln_attn")(layers.Add(name="res_attn")([x, attn]))

        # Feed-Forward Network (2× expansion)
        ffn = layers.Dense(p["d_model"] * 2, activation="relu",
                           kernel_regularizer=L2, name="ffn_1")(x)
        ffn = layers.Dropout(p["dropout"], name="ffn_drop")(ffn)
        ffn = layers.Dense(p["d_model"], kernel_regularizer=L2, name="ffn_2")(ffn)
        x   = layers.LayerNormalization(name="ln_ffn")(layers.Add(name="res_ffn")([x, ffn]))

        # Aggregate across time dimension
        x   = layers.GlobalAveragePooling1D(name="gap")(x)
        x   = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(x)
        x   = layers.Dropout(p["dropout"], name="dense_drop")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        return keras.Model(inp, out, name="tcn_transformer_classifier")
