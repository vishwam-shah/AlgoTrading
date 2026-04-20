"""
news_fetcher.py — Google News RSS + FinBERT (India fine-tuned)
==============================================================
Three-tier sentiment pipeline:
  Tier 1: Fine-tuned FinBERT (V3/02_models/finbert_india/) — best accuracy
  Tier 2: Base ProsusAI/finbert from HuggingFace              — good accuracy
  Tier 3: VADER keyword scorer                                — fallback

Usage:
    nf = NewsFeaturizer()
    result = nf.fetch_and_score('HDFCBANK')
    # returns: {raw_score, positive_ratio, negative_ratio, n_articles, ...}

Fine-tune the India model:
    python V3/01_data/news/finetune_finbert.py
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


_NEWS_DIR  = Path(__file__).resolve().parent
_V3_ROOT   = _NEWS_DIR.parent.parent
_MODEL_DIR = _V3_ROOT / "02_models" / "finbert_india"


# ══════════════════════════════════════════════════════════════════════════════
#  VADER FALLBACK  (keyword-based, no ML model needed)
# ══════════════════════════════════════════════════════════════════════════════

_POSITIVE_WORDS = {
    # Generic financial
    "gain", "up", "rise", "jump", "bullish", "strong", "beat", "beat",
    "profit", "rally", "surge", "boom", "growth", "positive", "outperform",
    "upgrade", "recovery", "high", "record", "soar", "climb",
    # Indian market specific
    "inflows", "buying", "accumulate", "overweight", "outperform",
    "target raised", "upgrade", "beat", "expansion", "improvement",
    "improvement", "narrowing", "declined npa", "credit growth",
    "buyback", "dividend", "casa", "nim expansion", "deal win",
    "guidance raised", "fii buying", "dii buying", "rate cut",
    "repo cut", "accommodative", "ipo listing gains", "all time high",
}

_NEGATIVE_WORDS = {
    # Generic financial
    "loss", "down", "fall", "drop", "bearish", "weak", "miss",
    "decline", "crash", "plunge", "fear", "contraction", "negative",
    "downgrade", "underperform", "recession", "sell", "low", "slump",
    # Indian market specific
    "outflows", "selling", "npa", "npa rises", "provision",
    "fraud", "scam", "default", "restructured", "stressed",
    "ban", "penalty", "notice", "sebi", "enforcement", "violation",
    "guidance cut", "margin pressure", "attrition", "fii selling",
    "rate hike", "hawkish", "repo hike", "inflation spike",
    "circuit breaker", "52 week low", "crash", "rout",
}


def _vader_score(text: str) -> float:
    """Simple word-count sentiment [-1, 1]."""
    t = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in t)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in t)
    total = pos + neg
    if total == 0:
        return 0.0
    return float(np.clip((pos - neg) / total, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  FINBERT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _FinBERTScorer:
    """
    Lazy-loaded FinBERT scorer.
    Loads fine-tuned India model if available, else base ProsusAI/finbert.
    """

    _instance: Optional["_FinBERTScorer"] = None
    _pipe = None
    _source: str = "none"

    @classmethod
    def get(cls) -> Optional["_FinBERTScorer"]:
        if cls._instance is None:
            inst = cls()
            inst._load()
            cls._instance = inst
        return cls._instance if cls._instance._pipe is not None else None

    def _load(self) -> None:
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else \
                     ("mps" if torch.backends.mps.is_available() else -1)

            # Tier 1: local fine-tuned India model
            if (_MODEL_DIR / "config.json").exists():
                self._pipe = hf_pipeline(
                    "text-classification",
                    model=str(_MODEL_DIR),
                    tokenizer=str(_MODEL_DIR),
                    device=device,
                    top_k=None,
                )
                self._source = "finbert-india"
                print(f"  [FinBERT] loaded fine-tuned India model from {_MODEL_DIR}")
                return

            # Tier 2: base FinBERT from HuggingFace
            self._pipe = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
                top_k=None,
            )
            self._source = "finbert-base"
            print("  [FinBERT] loaded base ProsusAI/finbert")

        except Exception as e:
            print(f"  [FinBERT] load failed ({e}), falling back to VADER")
            self._pipe = None

    def score_batch(self, texts: List[str]) -> List[Dict]:
        """
        Score a list of headlines. Returns list of dicts:
          {positive: float, neutral: float, negative: float}
        """
        if self._pipe is None or not texts:
            return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33}] * len(texts)

        results = []
        try:
            # Run in batches of 32 to avoid OOM
            batch_size = 32
            all_outputs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                out   = self._pipe(batch, truncation=True, max_length=128)
                all_outputs.extend(out)

            for out in all_outputs:
                scores = {item["label"].lower(): item["score"] for item in out}
                # Normalise label names (finbert uses "positive"/"negative"/"neutral")
                d = {
                    "positive": scores.get("positive", scores.get("pos", 0.33)),
                    "negative": scores.get("negative", scores.get("neg", 0.33)),
                    "neutral":  scores.get("neutral",  scores.get("neu", 0.34)),
                }
                results.append(d)
        except Exception as e:
            print(f"  [FinBERT] inference error: {e}")
            results = [{"positive": 0.33, "neutral": 0.34, "negative": 0.33}] * len(texts)

        return results

    @property
    def source(self) -> str:
        return self._source


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NewsFeaturizer:
    """Fetch Google News RSS + compute FinBERT sentiment for NSE stocks."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "nse_news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch headlines ───────────────────────────────────────────────────────

    def fetch_google_news(self, symbol: str, max_articles: int = 20) -> List[str]:
        """Fetch headlines from Google News RSS. Requires internet access."""
        try:
            import feedparser
            import requests as _req
            from urllib.parse import quote_plus

            query   = f"{symbol} NSE stock India"
            url     = (f"https://news.google.com/rss/search?q={quote_plus(query)}"
                       f"&hl=en-IN&gl=IN&ceid=IN:en")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            resp = _req.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            return [e.get("title", "") for e in feed.entries[:max_articles] if e.get("title")]

        except Exception as e:
            print(f"  [news] fetch failed for {symbol}: {e}")
            return []

    # ── Score headlines ───────────────────────────────────────────────────────

    def score_headlines(self, headlines: List[str]) -> Dict:
        """
        Score headlines using FinBERT (India fine-tuned if available) or VADER fallback.

        Returns:
          {
            raw_score       : float [-1, 1]
            positive_ratio  : float [0, 1]
            negative_ratio  : float [0, 1]
            neutral_ratio   : float [0, 1]
            n_articles      : int
            model_used      : str
          }
        """
        if not headlines:
            return {
                "raw_score": 0.0, "positive_ratio": 0.0,
                "negative_ratio": 0.0, "neutral_ratio": 1.0,
                "n_articles": 0, "model_used": "none",
            }

        scorer = _FinBERTScorer.get()

        if scorer is not None:
            # ── FinBERT path ──────────────────────────────────────────────
            batch   = scorer.score_batch(headlines)
            pos_arr = np.array([d["positive"] for d in batch])
            neg_arr = np.array([d["negative"] for d in batch])
            neu_arr = np.array([d["neutral"]  for d in batch])

            pos_ratio = float(np.mean(pos_arr))
            neg_ratio = float(np.mean(neg_arr))
            neu_ratio = float(np.mean(neu_arr))
            # raw_score: +1 = all positive, -1 = all negative
            raw_score = float(pos_ratio - neg_ratio)

            return {
                "raw_score":       float(np.clip(raw_score, -1.0, 1.0)),
                "positive_ratio":  pos_ratio,
                "negative_ratio":  neg_ratio,
                "neutral_ratio":   neu_ratio,
                "n_articles":      len(headlines),
                "model_used":      scorer.source,
            }
        else:
            # ── VADER fallback path ───────────────────────────────────────
            scores    = [_vader_score(h) for h in headlines]
            raw_score = float(np.mean(scores))
            pos_ratio = sum(1 for s in scores if s > 0.2)  / len(scores)
            neg_ratio = sum(1 for s in scores if s < -0.2) / len(scores)
            neu_ratio = 1.0 - pos_ratio - neg_ratio

            return {
                "raw_score":       float(np.clip(raw_score, -1.0, 1.0)),
                "positive_ratio":  pos_ratio,
                "negative_ratio":  neg_ratio,
                "neutral_ratio":   neu_ratio,
                "n_articles":      len(headlines),
                "model_used":      "vader",
            }

    # ── Main public API ───────────────────────────────────────────────────────

    def fetch_and_score(
        self,
        symbol: str,
        date: Optional[str] = None,
        max_articles: int = 20,
        use_finbert: bool = True,   # kept for backwards compat, always True now
    ) -> Dict:
        """
        Fetch news and compute sentiment for a symbol.

        Returns:
          {
            raw_score, positive_ratio, negative_ratio, neutral_ratio,
            n_articles, spike_flag, timestamp, model_used
          }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        headlines = self.fetch_google_news(symbol, max_articles=max_articles)
        scores    = self.score_headlines(headlines)

        # Spike: score extreme enough to warrant hard signal adjustment
        is_spike = abs(scores["raw_score"]) > 0.6

        return {
            **scores,
            "spike_flag": is_spike,
            "timestamp":  date,
        }

    def adjust_confidence_threshold(
        self,
        base_threshold: float,
        sentiment: Dict,
    ) -> float:
        """Adjust signal confidence threshold based on sentiment score."""
        score = sentiment["raw_score"]
        spike = sentiment["spike_flag"]

        if spike and score < 0:
            return min(base_threshold + 0.05, 0.75)
        elif score > 0.5:
            return max(base_threshold - 0.02, 0.50)
        elif score < -0.5:
            return base_threshold + 0.02
        return base_threshold
