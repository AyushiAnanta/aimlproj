"""
predict.py — Inference utilities for the sentiment + engagement models.

Loads trained sklearn Pipeline objects (standard Pipelines, no custom
classes) and provides majority-vote sentiment prediction plus
engagement estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import build_engagement_features_for_inference


# ── Model directory ──────────────────────────────────────────────
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_sentiment_models(
    model_dir: Path = _DEFAULT_MODEL_DIR,
) -> List:
    """Load the three TF-IDF sentiment pipelines."""
    names = ["pipe_lr.pkl", "pipe_svm.pkl", "pipe_sgd.pkl"]
    models = []
    for name in names:
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Sentiment model not found: {path}")
        models.append(joblib.load(path))
    return models


def load_engagement_model(
    model_dir: Path = _DEFAULT_MODEL_DIR,
    name: str = "engagement_regressor_likes.pkl",
):
    """Load one engagement regressor. Returns None if file is absent."""
    path = model_dir / name
    if not path.exists():
        return None
    return joblib.load(path)


# ── Sentiment prediction ────────────────────────────────────────
def sentiment_majority_vote(
    text: str, models: List
) -> Tuple[str, List[str]]:
    """Return (majority_label, [individual_predictions])."""
    preds = [m.predict([text])[0] for m in models]
    vals, counts = np.unique(preds, return_counts=True)
    return vals[np.argmax(counts)], preds


# ── Engagement prediction ───────────────────────────────────────
def predict_engagement(
    text: str,
    follower_count: int,
    engagement_model,
    has_hashtag: int = 0,
    has_media: int = 0,
    is_verified: int = 0,
    sentiment_probs: Optional[dict] = None,
) -> float:
    """Predict likes (back-transformed from log scale).

    If the engagement model was trained without sentiment features,
    pass ``sentiment_probs=None``.
    """
    X = build_engagement_features_for_inference(
        text=text,
        follower_count=follower_count,
        has_hashtag=has_hashtag,
        has_media=has_media,
        is_verified=is_verified,
        sentiment_probs=sentiment_probs,
    )
    log_likes = engagement_model.predict(X)[0]
    return float(np.expm1(log_likes))
