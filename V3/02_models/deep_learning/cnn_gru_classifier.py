"""
cnn_gru_classifier.py
================================================================================
CNNGRUClassifier — Causal 1D-CNN feature extractor + GRU temporal model.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features=50)
  Conv1D(64, kernel=3, padding='causal', relu, L2=1e-4)
  BatchNormalization()
  Conv1D(64, kernel=3, padding='causal', relu, L2=1e-4)
  BatchNormalization()
  MaxPooling1D(pool_size=2)           → (batch, 10, 64)
  Dropout(0.3)
  GRU(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
  Dense(32, relu, L2=1e-4)
  Dropout(0.3)
  Dense(1, sigmoid)

Design Rationale
----------------
- Identical CNN backbone to CNNLSTMClassifier; GRU replaces LSTM at the
  recurrent stage. Provides a distinct inductive bias (GRU's reset/update
  gates vs LSTM's cell state) at minimal incremental design cost, increasing
  ensemble diversity without adding architectural complexity.
- See cnn_lstm_classifier.py for detailed CNN rationale.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, CNN_GRU_PARAMS            # noqa: E402
from deep_learning.base_deep import BaseDLClassifier     # noqa: E402


class CNNGRUClassifier(BaseDLClassifier):
    """Causal 1D-CNN + GRU binary classifier."""

    model_name = "CNN_GRU"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = CNN_GRU_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = CNN_GRU_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        # ── Causal CNN block ──────────────────────────────────────────────────
        x = layers.Conv1D(
            p["cnn_filters"], p["cnn_kernel_size"],
            padding="causal", activation="relu",
            kernel_regularizer=L2, name="conv_1",
        )(inp)
        x = layers.BatchNormalization(name="bn_1")(x)

        x = layers.Conv1D(
            p["cnn_filters"], p["cnn_kernel_size"],
            padding="causal", activation="relu",
            kernel_regularizer=L2, name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="bn_2")(x)

        x = layers.MaxPooling1D(pool_size=p["pool_size"], name="pool")(x)
        x = layers.Dropout(p["dropout"], name="cnn_drop")(x)

        # ── GRU temporal compression ──────────────────────────────────────────
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

        return keras.Model(inp, out, name="cnn_gru_classifier")
