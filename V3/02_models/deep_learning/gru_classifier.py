"""
gru_classifier.py
================================================================================
GRUClassifier — 2-layer stacked GRU for V3 stock direction prediction.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features=50)
  GRU(64, return_sequences=True,  dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
  GRU(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
  Dense(32, relu, L2=1e-4)
  Dropout(0.3)
  Dense(1, sigmoid)

Design Rationale
----------------
- Identical topology to LSTM. GRU has fewer parameters (2 gates vs 3, no cell
  state) — trains faster and generalises comparably on sub-1000 sample sets.
- Acts as a diversity partner: when LSTM slightly overfits to a training window,
  GRU's simpler gating tends not to, improving ensemble robustness.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, GRU_PARAMS               # noqa: E402
from deep_learning.base_deep import BaseDLClassifier     # noqa: E402


class GRUClassifier(BaseDLClassifier):
    """2-layer stacked GRU binary classifier."""

    model_name = "GRU"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = GRU_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = GRU_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        x = layers.GRU(
            p["units_1"],
            return_sequences=True,
            dropout=p["dropout"],
            recurrent_dropout=p["recurrent_dropout"],
            kernel_regularizer=L2,
            name="gru_1",
        )(inp)

        x = layers.GRU(
            p["units_2"],
            return_sequences=False,
            dropout=p["dropout"],
            recurrent_dropout=p["recurrent_dropout"],
            kernel_regularizer=L2,
            name="gru_2",
        )(x)

        x   = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(x)
        x   = layers.Dropout(p["dropout"], name="drop_out")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        return keras.Model(inp, out, name="gru_classifier")
