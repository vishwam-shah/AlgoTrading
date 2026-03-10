"""
tcn_gru_classifier.py
================================================================================
TCNGRUClassifier — Temporal Convolutional Network (dilated causal) + GRU.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features)
  [Channel projection: Conv1D(64, 1, causal) if n_features != 64]
  TCN Block dilation=1 : Conv1D(64,3,causal,d=1) → LN → Conv1D(64,3,causal,d=1) → LN → Dropout → Add(residual)
  TCN Block dilation=2 : same pattern, dilation_rate=2
  TCN Block dilation=4 : same pattern, dilation_rate=4
  TCN Block dilation=8 : same pattern, dilation_rate=8
  [Receptive field = 1 + (kernel-1)*2*(1+2+4+8) = 1 + 4*30 = 121 > SEQ_LEN — covers full window]
  GRU(32, dropout=0.3, recurrent_dropout=0.2)
  Dense(32, relu)  →  Dropout(0.3)  →  Dense(1, sigmoid)

Design Rationale
----------------
- TCN (dilated causal convolutions) captures multi-scale local patterns without
  recurrent state overhead. Dilation rates [1,2,4,8] give exponential receptive
  field growth with O(log N) depth instead of O(N) for vanilla LSTM.
- Residual connections (ResNet-style) stabilise gradients across TCN depth.
- LayerNorm (not BatchNorm): more stable with small financial mini-batches.
- GRU at the end provides sequential inductive bias after the multi-scale CNN
  extraction, giving the model both local-pattern and temporal-dependency
  representations.
- Architecture inspired by Bai et al. (2018) "An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modelling".

Paper Reference Context
-----------------------
N-BEATS (Oreshkin et al. 2019) demonstrated that deep fully-connected
residual networks can match or beat ARIMA+ML hybrids on M3/M4 forecasting.
TCN provides a convolutional analogue: causal dilated residual blocks outperform
standard RNNs on many seq2seq tasks (Bai 2018). This model fuses the local
TCN feature extractor with GRU's sequential bias for stock direction prediction.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, TCN_GRU_PARAMS              # noqa: E402
from deep_learning.base_deep import BaseDLClassifier        # noqa: E402


class TCNGRUClassifier(BaseDLClassifier):
    """Dilated causal TCN + GRU binary classifier."""

    model_name = "TCN_GRU"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = TCN_GRU_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = TCN_GRU_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        # Channel projection: map n_features → filters via pointwise conv
        if self.n_features != p["filters"]:
            x = layers.Conv1D(
                p["filters"], 1, padding="causal",
                kernel_regularizer=L2, name="proj",
            )(inp)
        else:
            x = inp

        # TCN residual blocks with exponentially growing dilation rates
        for i, dilation in enumerate(p["dilations"]):
            res = x  # skip connection

            x = layers.Conv1D(
                p["filters"], p["kernel_size"],
                padding="causal", dilation_rate=dilation,
                activation="relu", kernel_regularizer=L2,
                name=f"tcn_{i}_conv1",
            )(x)
            x = layers.LayerNormalization(name=f"tcn_{i}_ln1")(x)

            x = layers.Conv1D(
                p["filters"], p["kernel_size"],
                padding="causal", dilation_rate=dilation,
                activation="relu", kernel_regularizer=L2,
                name=f"tcn_{i}_conv2",
            )(x)
            x = layers.LayerNormalization(name=f"tcn_{i}_ln2")(x)

            x = layers.Dropout(p["dropout"], name=f"tcn_{i}_drop")(x)
            x = layers.Add(name=f"tcn_{i}_res")([x, res])

        # GRU temporal compression (operates on TCN-extracted features)
        x = layers.GRU(
            p["gru_units"],
            return_sequences=False,
            dropout=p["dropout"],
            recurrent_dropout=p["recurrent_dropout"],
            kernel_regularizer=L2,
            name="gru",
        )(x)

        x   = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(x)
        x   = layers.Dropout(p["dropout"], name="dense_drop")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        return keras.Model(inp, out, name="tcn_gru_classifier")
