"""
lstm_classifier.py
================================================================================
LSTMClassifier — 2-layer stacked LSTM for V3 stock direction prediction.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features=50)
  LSTM(64, return_sequences=True,  dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
  LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
  Dense(32, relu, L2=1e-4)
  Dropout(0.3)
  Dense(1, sigmoid)

Design Rationale
----------------
- Units 64 → 32: halving prevents layer-2 from memorising layer-1 output.
- recurrent_dropout=0.2: dropout on recurrent connections injects noise at each
  time step, preventing temporal co-adaptation inside BPTT.
- L2=1e-4 on kernels only (not recurrent kernels): light weight-decay without
  squashing gradients through time.
- Dense(32) + Dropout(0.3): single projection before sigmoid; deeper dense
  blocks overfit on sub-1000 sample financial datasets.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, LSTM_PARAMS              # noqa: E402
from deep_learning.base_deep import BaseDLClassifier     # noqa: E402


class LSTMClassifier(BaseDLClassifier):
    """2-layer stacked LSTM binary classifier."""

    model_name = "LSTM"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = LSTM_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = LSTM_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        x = layers.LSTM(
            p["units_1"],
            return_sequences=True,
            dropout=p["dropout"],
            recurrent_dropout=p["recurrent_dropout"],
            kernel_regularizer=L2,
            name="lstm_1",
        )(inp)

        x = layers.LSTM(
            p["units_2"],
            return_sequences=False,
            dropout=p["dropout"],
            recurrent_dropout=p["recurrent_dropout"],
            kernel_regularizer=L2,
            name="lstm_2",
        )(x)

        x   = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(x)
        x   = layers.Dropout(p["dropout"], name="drop_out")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        return keras.Model(inp, out, name="lstm_classifier")
