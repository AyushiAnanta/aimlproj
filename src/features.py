"""
features.py — Shared feature-construction functions.

Used by BOTH training scripts and the Streamlit app so that
train-time and inference-time features are guaranteed identical.

NOTE: Sentiment labels in this project are RoBERTa pseudo-labels,
not human annotations.  Any metric should be called
"agreement with RoBERTa labels", never "accuracy".
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Text cleaning (sentiment pipeline) ──────────────────────────
def clean_text(s: str) -> str:
    """Light normalisation that preserves sentiment cues."""
    s = s or ""
    s = re.sub(r"http\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"@\w+", " <USER> ", s)
    s = re.sub(r"#(\w+)", r" #\1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Engagement features ─────────────────────────────────────────
def build_engagement_features(
    df: pd.DataFrame,
    sentiment_probs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the feature matrix for the engagement regressor.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: ``followers``, ``description``,
        ``hashtags``, ``photos``, ``is_verified``.
    sentiment_probs : pd.DataFrame or None
        If provided, must contain columns ``prob_positive``,
        ``prob_negative``, ``prob_neutral``, ``sentiment_confidence``.
        When *None*, sentiment columns are omitted (baseline model).

    Returns
    -------
    pd.DataFrame  — one row per tweet, ready for the regressor.
    """
    X = pd.DataFrame(index=df.index)

    # Audience size (log-scaled to reduce viral-outlier effect)
    X["log_followers"] = np.log1p(df["followers"].astype(float))

    # Text features
    X["text_length"] = df["description"].astype(str).str.len()
    X["word_count"] = df["description"].astype(str).str.split().str.len()

    # Media / hashtag / verification signals
    X["has_hashtag"] = df["hashtags"].apply(
        lambda x: 1 if str(x) not in ("null", "None", "nan", "[]") else 0
    )
    X["has_media"] = df["photos"].apply(
        lambda x: 1 if str(x) not in ("null", "None", "nan", "[]") else 0
    )
    X["is_verified"] = df["is_verified"].apply(
        lambda x: 1 if str(x).lower() in ("true", "1") else 0
    )

    # Sentiment features (only when available)
    if sentiment_probs is not None:
        X["prob_positive"] = sentiment_probs["prob_positive"].values
        X["prob_negative"] = sentiment_probs["prob_negative"].values
        X["prob_neutral"] = sentiment_probs["prob_neutral"].values
        X["sentiment_confidence"] = sentiment_probs["sentiment_confidence"].values
        X["sentiment_score"] = (
            sentiment_probs["prob_positive"].values
            - sentiment_probs["prob_negative"].values
        )

    return X


# ── Engagement-feature builder for inference (single row) ───────
def build_engagement_features_for_inference(
    text: str,
    follower_count: int,
    has_hashtag: int = 0,
    has_media: int = 0,
    is_verified: int = 0,
    sentiment_probs: Optional[dict] = None,
) -> pd.DataFrame:
    """Construct the same feature vector used at training time,
    but from single-row inputs (the Streamlit form).

    Parameters
    ----------
    sentiment_probs : dict or None
        Keys: prob_positive, prob_negative, prob_neutral, sentiment_confidence.
        Pass *None* when using a model trained without sentiment features.
    """
    row: dict = {
        "log_followers": float(np.log1p(follower_count)),
        "text_length": len(text),
        "word_count": len(text.split()),
        "has_hashtag": has_hashtag,
        "has_media": has_media,
        "is_verified": is_verified,
    }

    if sentiment_probs is not None:
        row["prob_positive"] = sentiment_probs["prob_positive"]
        row["prob_negative"] = sentiment_probs["prob_negative"]
        row["prob_neutral"] = sentiment_probs["prob_neutral"]
        row["sentiment_confidence"] = sentiment_probs["sentiment_confidence"]
        row["sentiment_score"] = (
            sentiment_probs["prob_positive"] - sentiment_probs["prob_negative"]
        )

    return pd.DataFrame([row])
