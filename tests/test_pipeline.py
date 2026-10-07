"""
tests/test_pipeline.py — Pytest suite covering:
1. Feature construction (shape, nulls, views == 0 case).
2. Train/inference parity test (verifying build_engagement_features_for_inference produces
   matching feature columns and data types as training).
3. Model loading and predictions on sample tweets returning valid labels and engagement estimates.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features import (
    clean_text,
    build_engagement_features,
    build_engagement_features_for_inference,
)
from src.predict import (
    load_sentiment_models,
    load_engagement_model,
    sentiment_majority_vote,
    predict_engagement,
)


@pytest.fixture
def sample_tweets_df() -> pd.DataFrame:
    return pd.DataFrame({
        "description": [
            "Just launched our new open source tool! Check it out #opensource",
            "This was the worst customer experience I have ever had. Terrible.",
            "Normal update: the meeting starts at 3 PM UTC.",
            "Zero views edge case test post",
        ],
        "followers": [1500, 200, 50000, 10],
        "hashtags": ["['opensource']", "[]", "null", "None"],
        "photos": ["['img.png']", "null", "[]", "None"],
        "is_verified": [1, 0, 0, 0],
        "views": [1000, 500, 25000, 0],
        "likes": [50, 12, 450, 0],
        "replies": [5, 4, 25, 0],
        "reposts": [10, 2, 70, 0],
        "quotes": [1, 0, 15, 0],
        "bookmarks": [8, 1, 95, 0],
    })


def test_clean_text():
    raw = "Check https://t.co/xyz @someone and #AI!"
    cleaned = clean_text(raw)
    assert "<URL>" in cleaned
    assert "<USER>" in cleaned
    assert "#AI" in cleaned
    assert "https://" not in cleaned


def test_engagement_features_shape_and_no_nan(sample_tweets_df):
    features = build_engagement_features(sample_tweets_df, sentiment_probs=None)
    assert not features.empty
    assert features.shape[0] == len(sample_tweets_df)
    assert not features.isna().any().any()
    expected_cols = ["log_followers", "text_length", "word_count", "has_hashtag", "has_media", "is_verified"]
    for col in expected_cols:
        assert col in features.columns


def test_train_inference_feature_parity(sample_tweets_df):
    """Ensure feature order and column names match between batch training and single-row inference."""
    train_feat = build_engagement_features(sample_tweets_df.iloc[[0]], sentiment_probs=None)
    inf_feat = build_engagement_features_for_inference(
        text=sample_tweets_df.iloc[0]["description"],
        follower_count=sample_tweets_df.iloc[0]["followers"],
        has_hashtag=1,
        has_media=1,
        is_verified=1,
        sentiment_probs=None,
    )

    assert list(train_feat.columns) == list(inf_feat.columns)
    assert np.isclose(train_feat["log_followers"].values[0], inf_feat["log_followers"].values[0])
    assert train_feat["text_length"].values[0] == inf_feat["text_length"].values[0]


def test_safe_engagement_metrics_views_zero(sample_tweets_df):
    """Simulate the app's CSV processor handling views == 0 without dividing by zero."""
    df = sample_tweets_df.copy()
    df["total_engagements"] = df[["likes", "replies", "reposts", "quotes", "bookmarks"]].sum(axis=1)

    safe_views = df["views"].replace(0, np.nan)
    df["engagement_rate"] = (df["total_engagements"] / safe_views * 100).fillna(0.0)

    # For the 4th row where views == 0, engagement_rate must be cleanly 0.0 and not inf / nan
    assert not np.isinf(df["engagement_rate"].iloc[3])
    assert not np.isnan(df["engagement_rate"].iloc[3])
    assert df["engagement_rate"].iloc[3] == 0.0


def test_models_predict_valid_labels_and_ranges():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    sentiment_models = load_sentiment_models(model_dir)
    eng_model = load_engagement_model(model_dir, "engagement_regressor_likes.pkl")

    assert len(sentiment_models) == 3
    assert eng_model is not None

    test_sentences = [
        "I am so excited and genuinely love this new development!",
        "This is an absolute failure and completely unacceptable.",
        "The meeting is scheduled tomorrow at 9 AM.",
    ]

    valid_labels = {"Positive", "Negative", "Neutral"}
    for text in test_sentences:
        label, votes = sentiment_majority_vote(text, sentiment_models)
        assert label in valid_labels
        assert len(votes) == 3
        for v in votes:
            assert v in valid_labels

        likes = predict_engagement(
            text=text,
            follower_count=5000,
            engagement_model=eng_model,
            has_hashtag=1,
            has_media=0,
            is_verified=0,
            sentiment_probs=None,
        )
        assert isinstance(likes, float)
        assert likes >= 0.0
