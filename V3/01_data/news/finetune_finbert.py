"""
finetune_finbert.py
===================
Fine-tune ProsusAI/finbert on Indian stock market sentences.

Base model : ProsusAI/finbert  (trained on financial English text)
Fine-tune  : Indian market dataset (NSE/BSE/RBI/FII vocabulary)
Output     : V3/02_models/finbert_india/

Label mapping (matches FinBERT original order):
  LABEL_0 = positive
  LABEL_1 = negative
  LABEL_2 = neutral

Our dataset labels are remapped before training to match.

Usage:
    python V3/01_data/news/finetune_finbert.py
    python V3/01_data/news/finetune_finbert.py --epochs 8 --lr 1e-5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import sys
from pathlib import Path

_NEWS_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _NEWS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_NEWS_DIR))

SAVE_DIR   = _V3_ROOT / "02_models" / "finbert_india"
BASE_MODEL = "ProsusAI/finbert"

# FinBERT original label order: 0=positive, 1=negative, 2=neutral
# Our dataset order:             0=negative, 1=neutral,  2=positive
# Remap our labels to FinBERT order so we don't reinitialise the head
_OUR_TO_FINBERT = {0: 1, 1: 2, 2: 0}   # neg→1, neu→2, pos→0

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}


def finetune(epochs: int = 8, lr: float = 1e-5, batch_size: int = 16, max_samples: int = 0) -> None:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    from indian_finance_dataset import get_dataset

    texts, raw_labels = get_dataset()

    if max_samples and max_samples < len(texts):
        texts      = texts[:max_samples]
        raw_labels = raw_labels[:max_samples]
        print(f"  Truncated to first {max_samples} samples")

    # Remap labels to match FinBERT's original label order
    labels = [_OUR_TO_FINBERT[l] for l in raw_labels]

    counts = {v: labels.count(v) for v in [0, 1, 2]}
    print(f"\n  Dataset  : {len(texts)} samples")
    print(f"  Positive : {counts[0]}  Negative : {counts[1]}  Neutral : {counts[2]}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"  Train    : {len(X_tr)}   Val : {len(X_val)}")

    # ── Tokeniser ────────────────────────────────────────────────────────────
    print(f"\n  Loading tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding=False)

    ds_train = Dataset.from_dict({"text": X_tr,  "label": y_tr})
    ds_val   = Dataset.from_dict({"text": X_val, "label": y_val})
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tokenize,   batched=True, remove_columns=["text"])

    # ── Model — keep FinBERT's original classifier head, NO reinit ───────────
    print(f"  Loading base model {BASE_MODEL} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels   = 3,
        id2label     = ID2LABEL,
        label2id     = LABEL2ID,
        # ignore_mismatched_sizes NOT set — reuse pre-trained classifier head
    )

    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device   : {device}")
    model.to(device)

    # ── Training arguments ────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = SAVE_DIR / "checkpoints"

    warmup_steps = max(10, int(0.1 * (len(X_tr) // batch_size) * epochs))

    args = TrainingArguments(
        output_dir                  = str(tmp_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = lr,
        weight_decay                = 0.01,
        warmup_steps                = warmup_steps,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 5,
        report_to                   = "none",
        seed                        = 42,
        dataloader_pin_memory       = False,
        gradient_accumulation_steps = 2,   # simulate batch_size*2 = 32
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1":       float(f1_score(labels, preds, average="macro")),
        }

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Fine-tuning {epochs} epochs  lr={lr}  warmup={warmup_steps}steps ...")
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_train,
        eval_dataset    = ds_val,
        data_collator   = DataCollatorWithPadding(tokenizer),
        compute_metrics = compute_metrics,
        processing_class= tokenizer,
    )
    trainer.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    acc = metrics["eval_accuracy"] * 100
    f1  = metrics["eval_f1"]
    print(f"\n  Validation accuracy : {acc:.1f}%")
    print(f"  Validation F1 macro : {f1:.3f}")

    # Per-class breakdown
    preds_out = trainer.predict(ds_val)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print("\n  Per-class report:")
    print(classification_report(
        preds_out.label_ids, preds,
        target_names=["positive", "negative", "neutral"]
    ))

    if acc < 70:
        print(f"\n  [!] Accuracy {acc:.1f}% below 70% -- consider adding more training data")
        print(f"    Run: python V3/01_data/news/finetune_finbert.py --epochs 10")
    elif acc < 85:
        print(f"\n  [>] Accuracy {acc:.1f}% -- good, run more epochs to push toward 90%")
    else:
        print(f"\n  [ok] Accuracy {acc:.1f}% -- target reached!")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n  Saving model to {SAVE_DIR} ...")
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))

    import json
    with open(SAVE_DIR / "label_map.json", "w") as f:
        json.dump({
            "id2label":          ID2LABEL,
            "label2id":          LABEL2ID,
            "our_to_finbert":    _OUR_TO_FINBERT,
            "val_accuracy":      acc / 100,
            "val_f1":            f1,
            "n_train":           len(X_tr),
            "n_val":             len(X_val),
        }, f, indent=2)

    print(f"  Done. Model ready for use in news_fetcher.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--max-samples",  type=int,   default=0, help="Use only first N samples (0=all)")
    args = parser.parse_args()
    finetune(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_samples=args.max_samples)
