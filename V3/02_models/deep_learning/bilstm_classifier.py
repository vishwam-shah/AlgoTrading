"""
bilstm_classifier.py
================================================================================
BiLSTMClassifier — Bidirectional 2-layer LSTM for V3 stock direction prediction.

Architecture
------------
Input  (batch, SEQ_LEN=20, n_features=50)
  Bidirectional(LSTM(32, return_sequences=True,  dropout=0.3, recurrent_dropout=0.2, L2=1e-4))
      → merged output: 64 units (32 forward + 32 backward, concatenated)
  Bidirectional(LSTM(16, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, L2=1e-4))
      → merged output: 32 units
  Dense(32, relu, L2=1e-4)
  Dropout(0.3)
  Dense(1, sigmoid)

Design Rationale
----------------
- Per-direction units halved (32/16) so merged output (64/32) matches the LSTM
  model's parameter budget. This keeps ensemble diversity without extra capacity.
- Bidirectional is safe here: ALL V3 features are look-back indicators (lagged
  returns, rolling RSI, etc.). The backward LSTM pass reads feature values at
  t+1 … t+19, none of which encode the target (next-day return). No leakage.
- Same dropout / L2 as LSTM for regularisation consistency across the ensemble.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_cfg_dir = Path(__file__).resolve().parent.parent.parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

from config import DL_SEQ_LEN, BILSTM_PARAMS            # noqa: E402
from deep_learning.base_deep import BaseDLClassifier     # noqa: E402


class BiLSTMClassifier(BaseDLClassifier):
    """2-layer bidirectional LSTM binary classifier."""

    model_name = "BiLSTM"

    def __init__(
        self,
        seq_len:    int = DL_SEQ_LEN,
        n_features: int = 50,
    ) -> None:
        super().__init__(seq_len=seq_len, n_features=n_features)
        self._lr = BILSTM_PARAMS["learning_rate"]

    def _build(self):
        import keras
        from keras import layers, regularizers

        p  = BILSTM_PARAMS
        L2 = regularizers.L2(p["l2"])

        inp = keras.Input(shape=(self.seq_len, self.n_features), name="input")

        x = layers.Bidirectional(
            layers.LSTM(
                p["units_1"],
                return_sequences=True,
                dropout=p["dropout"],
                recurrent_dropout=p["recurrent_dropout"],
                kernel_regularizer=L2,
            ),
            name="bilstm_1",
        )(inp)

        x = layers.Bidirectional(
            layers.LSTM(
                p["units_2"],
                return_sequences=False,
                dropout=p["dropout"],
                recurrent_dropout=p["recurrent_dropout"],
                kernel_regularizer=L2,
            ),
            name="bilstm_2",
        )(x)

        x   = layers.Dense(p["dense_units"], activation="relu",
                           kernel_regularizer=L2, name="dense_1")(x)
        x   = layers.Dropout(p["dropout"], name="drop_out")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        return keras.Model(inp, out, name="bilstm_classifier")
