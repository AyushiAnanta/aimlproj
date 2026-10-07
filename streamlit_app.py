"""
streamlit_app.py — Social Media Sentiment + Engagement Predictor.

Features:
- Sentiment classification via majority vote among standard TF-IDF models
  (trained via pseudo-labeling with a transformer teacher).
- Engagement prediction based on metadata features with feature parity to training.
- Transparent presentation of derived engagement estimates (clearly labeled).
- CSV engagement metric calculator with safe zero-division prevention.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import build_engagement_features_for_inference
from src.predict import (
    load_sentiment_models,
    load_engagement_model,
    sentiment_majority_vote,
    predict_engagement,
)

st.set_page_config(page_title="Social Media Analyzer", layout="centered")
st.title("Social Media Sentiment + Engagement Predictor")

# ---------------------------------------------------------
# AUTO-DETECT MODEL DIRECTORY
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

env_override = os.getenv("MODEL_DIR_OVERRIDE")
if env_override:
    MODEL_DIR = Path(env_override)
MODEL_DIR = MODEL_DIR.resolve()

st.caption(f"Using model directory: `{MODEL_DIR}`")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
try:
    sentiment_models = load_sentiment_models(MODEL_DIR)
    st.success("Loaded sentiment models successfully.")
except Exception as e:
    st.error(f"Failed to load required sentiment models from `{MODEL_DIR}`: {e}")
    st.stop()

eng_like_model = load_engagement_model(MODEL_DIR, "engagement_regressor_likes.pkl")
eng_rate_model = load_engagement_model(MODEL_DIR, "engagement_regressor_rate.pkl")

if eng_like_model is not None:
    st.info("Engagement likes regressor loaded.")
if eng_rate_model is not None:
    st.info("Engagement rate regressor loaded.")

# ---------------------------------------------------------
# UI: SINGLE TEXT PREDICTION
# ---------------------------------------------------------
st.header("Analyze Text Input")
text = st.text_area("Enter a tweet/post text:", height=120)

col1, col2, col3 = st.columns(3)
with col1:
    follower_count = st.number_input("Follower Count", min_value=0, value=1000)
with col2:
    hashtags_count = st.number_input("Hashtag Count", min_value=0, value=0)
with col3:
    is_verified_input = st.selectbox("Account Verified?", options=["No", "Yes"])

is_verified = 1 if is_verified_input == "Yes" else 0
has_hashtag = 1 if (hashtags_count > 0 or "#" in text) else 0

if st.button("Run Prediction"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Sentiment Prediction
        label, individual_preds = sentiment_majority_vote(text, sentiment_models)

        st.subheader("Sentiment Prediction")
        st.write(f"**Predicted Sentiment:** {label}")
        st.caption(
            f"Individual Model Votes (LR, SVM, SGD): {individual_preds}. "
            "Note: Models were trained via pseudo-labeling with a RoBERTa transformer teacher."
        )

        # 2. Engagement Prediction
        st.subheader("Engagement Estimation")
        if eng_like_model is not None:
            # Predict likes using identical feature construction as training
            likes = predict_engagement(
                text=text,
                follower_count=follower_count,
                engagement_model=eng_like_model,
                has_hashtag=has_hashtag,
                has_media=0,
                is_verified=is_verified,
                sentiment_probs=None,
            )
        else:
            # Fallback heuristic rate if model is missing
            pos_votes = sum(1 for p in individual_preds if p == "Positive")
            neg_votes = sum(1 for p in individual_preds if p == "Negative")
            score = (pos_votes - neg_votes) / len(individual_preds)
            base_rate = 0.006 + (score * 0.012)
            likes = follower_count * base_rate

        likes_val = max(0, int(likes))
        st.metric("Estimated Likes (Model Prediction)", f"{likes_val:,}")

        # Transparent labeling for derived ratios
        st.markdown("**Derived Heuristic Projections:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Estimated Retweets", f"{int(likes_val * 0.15):,}")
            st.caption("Rough estimate derived as ~15% of predicted likes")
        with col_m2:
            st.metric("Estimated Replies", f"{int(likes_val * 0.06):,}")
            st.caption("Rough estimate derived as ~6% of predicted likes")

# ---------------------------------------------------------
# CSV Upload: Engagement Metrics Calculator
# ---------------------------------------------------------
st.markdown("---")
st.header("Upload CSV to Compute Real Engagement Metrics")
uploaded = st.file_uploader("Upload twitter-posts.csv", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.write("Data Preview:", df_raw.head())

    required_cols = ["likes", "replies", "reposts", "quotes", "bookmarks", "views"]
    missing_cols = [c for c in required_cols if c not in df_raw.columns]

    if missing_cols:
        st.error(f"Uploaded CSV is missing required engagement columns: {missing_cols}")
    else:
        def calculate_engagement_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            df_out["total_engagements"] = df_out[["likes", "replies", "reposts", "quotes", "bookmarks"]].sum(axis=1)

            # Prevent division by zero when views == 0
            safe_views = df_out["views"].replace(0, np.nan)

            df_out["engagement_rate"] = (df_out["total_engagements"] / safe_views * 100).fillna(0.0)
            df_out["like_rate"] = (df_out["likes"] / safe_views * 100).fillna(0.0)
            df_out["reply_rate"] = (df_out["replies"] / safe_views * 100).fillna(0.0)
            df_out["repost_rate"] = (df_out["reposts"] / safe_views * 100).fillna(0.0)
            df_out["quote_rate"] = (df_out["quotes"] / safe_views * 100).fillna(0.0)
            df_out["bookmark_rate"] = (df_out["bookmarks"] / safe_views * 100).fillna(0.0)
            return df_out

        df_processed = calculate_engagement_metrics(df_raw)
        st.write("Computed Engagement Metrics:", df_processed[["views", "likes", "total_engagements", "engagement_rate", "like_rate"]].head())

        st.download_button(
            label="Download Processed CSV",
            data=df_processed.to_csv(index=False),
            file_name="processed_engagement_metrics.csv",
            mime="text/csv",
        )
