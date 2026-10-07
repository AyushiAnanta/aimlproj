import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import glob
from sklearn.pipeline import FeatureUnion
if not hasattr(FeatureUnion, 'verbose_feature_names_out'):
    FeatureUnion.verbose_feature_names_out = True

# --- Needed to load old models: recreate WeightedPipeline class ---
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class WeightedPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, features=None, estimator=None):
        self.features = features
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self._vect_ = clone(self.features)
        Xv = self._vect_.fit_transform(X)
        self._clf_ = clone(self.estimator)
        try:
            self._clf_.fit(Xv, y, sample_weight=sample_weight)
        except TypeError:
            self._clf_.fit(Xv, y)
        return self

    def predict(self, X):
        Xv = self._vect_.transform(X)
        return self._clf_.predict(Xv)

st.set_page_config(page_title="Social Media Analyzer", layout="centered")
st.title("Social Media Sentiment + Engagement Predictor")

st.write("Models will be auto-detected on your system. No need to configure paths manually.")

# ---------------------------------------------------------
# AUTO-DETECT MODEL DIRECTORY
# ---------------------------------------------------------
# --- robust local-models loader (put this at top of streamlit_app.py) ---
import os, joblib, glob, sys
from pathlib import Path

# Prefer models folder next to this script. Fallback to current working dir.
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent
else:
    # if running interactively, use cwd (useful when running inside notebooks/VSCode)
    PROJECT_ROOT = Path.cwd()

MODEL_DIR = PROJECT_ROOT / "models"

# quick sanity: allow an env override if you want to point elsewhere temporarily
# e.g. set MODEL_DIR_OVERRIDE="C:\\Users\\theay\\aimlproj\\models" in your env
env_override = os.getenv("MODEL_DIR_OVERRIDE")
if env_override:
    MODEL_DIR = Path(env_override)

MODEL_DIR = MODEL_DIR.resolve()

# show location in Streamlit UI (and allow early failure handling)
import streamlit as st
st.write(f"Using model folder: `{MODEL_DIR}`")

# make sure required files exist before continuing
required = ["pipe_lr.pkl", "pipe_svm.pkl", "pipe_sgd.pkl"]
missing = [f for f in required if not (MODEL_DIR / f).exists()]

if missing:
    st.error(f"No sentiment classifier models found in `{MODEL_DIR}`. Missing: {missing}\n"
             "Place pipe_lr.pkl, pipe_svm.pkl, pipe_sgd.pkl inside that folder or set MODEL_DIR_OVERRIDE to the correct path.")
    st.stop()

# helper to load safely
def safe_load(name, required=True):
    path = MODEL_DIR / name
    if not path.exists():
        if required:
            st.error(f"Required model not found: {path}")
            st.stop()
        return None  # optional model, just skip
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        if required:
            st.stop()
        return None


# ---------------------------------------------------------
# LOAD SENTIMENT MODELS
# ---------------------------------------------------------
sentiment_models = []



pipe_lr = safe_load("pipe_lr.pkl")
pipe_svm = safe_load("pipe_svm.pkl")
pipe_sgd = safe_load("pipe_sgd.pkl")

for model in [pipe_lr, pipe_svm, pipe_sgd]:
    if model is not None:
        sentiment_models.append(model)

if not sentiment_models:
    st.error("❌ No sentiment classifier models could be loaded.")
    st.stop()

st.success("✔ Loaded sentiment models successfully.")

# ---------------------------------------------------------
# Optional Engagement Models
# ---------------------------------------------------------
eng_like_model = safe_load("engagement_regressor_likes.pkl", required=False)
eng_rate_model = safe_load("engagement_regressor_rate.pkl", required=False)

if eng_like_model:
    st.success("Engagement likes regressor loaded.")
if eng_rate_model:
    st.success("Engagement rate regressor loaded.")

# ---------------------------------------------------------
# Majority Voting Function
# ---------------------------------------------------------
def sentiment_majority_vote(text):
    preds = [model.predict([text])[0] for model in sentiment_models]
    vals, counts = np.unique(preds, return_counts=True)
    return vals[np.argmax(counts)], preds

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.header("Analyze Text Input")
text = st.text_area("Enter a tweet/post text:", height=140)

col1, col2, col3 = st.columns(3)
with col1: follower_count = st.number_input("Follower Count", min_value=0, value=1000)
with col2: hashtags = st.number_input("Hashtags", min_value=0, value=0)
with col3: mentions = st.number_input("Mentions", min_value=0, value=0)

if st.button("Run Prediction"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        label, individual_preds = sentiment_majority_vote(text)

        st.subheader("Sentiment Prediction")
        st.write(f"**Final Sentiment:** {label}")
        st.caption(f"Votes: {individual_preds}")

        st.subheader("Engagement Prediction")
        feature_row = {
            "text_for_model": text,
            "followers": follower_count,
            "views": follower_count,
            "reposts": 0,
            "quotes": 0,
            "bookmarks": 0,
            "replies": 0,
            "text_length": len(text)
        }
        Xnew = pd.DataFrame([feature_row])

        if eng_like_model:
            Xnew = pd.DataFrame([{
                "log_followers":        np.log1p(follower_count),
                "prob_positive":        0.33,   # placeholder — no RoBERTa at inference time
                "prob_negative":        0.33,
                "prob_neutral":         0.34,
                "sentiment_confidence": 0.5,
                "sentiment_score":      0.0,
                "text_length":          len(text),
                "word_count":           len(text.split()),
                "has_hashtag":          1 if "#" in text else 0,
                "has_media":            0,
                "is_verified":          0,
            }])
            likes = float(np.expm1(eng_like_model.predict(Xnew)[0]))
        else:
            # Use sentiment probabilities + followers for a data-informed estimate
            # Get individual model predictions to extract confidence
            pos_votes = sum(1 for p in individual_preds if p == "Positive")
            neg_votes = sum(1 for p in individual_preds if p == "Negative")
            sentiment_score = (pos_votes - neg_votes) / len(individual_preds)  # -1 to +1
            base_rate = 0.006 + (sentiment_score * 0.012)  # ranges 0.006 to 0.018 continuously
            likes = follower_count * base_rate

        st.metric("Estimated Likes", f"{int(likes):,}")
        st.metric("Estimated Retweets", f"{int(likes * 0.15):,}")
        st.metric("Estimated Replies", f"{int(likes * 0.06):,}")

# ---------------------------------------------------------
# CSV Upload: Engagement Metrics Calculator
# ---------------------------------------------------------
st.markdown("---")
st.header("Upload CSV to Compute Real Engagement Metrics")
uploaded = st.file_uploader("Upload twitter-posts.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    required = ["likes","replies","reposts","quotes","bookmarks","views"]
    if not all(c in df.columns for c in required):
        st.error("CSV missing required engagement columns.")
    else:
        def calculate_engagement_metrics(df):
            df["total_engagements"] = df[["likes","replies","reposts","quotes","bookmarks"]].sum(axis=1)
            df["engagement_rate"] = df["total_engagements"] / df["views"] * 100
            df["like_rate"] = df["likes"] / df["views"] * 100
            df["reply_rate"] = df["replies"] / df["views"] * 100
            df["repost_rate"] = df["reposts"] / df["views"] * 100
            df["quote_rate"] = df["quotes"] / df["views"] * 100
            df["bookmark_rate"] = df["bookmarks"] / df["views"] * 100
            df.fillna(0, inplace=True)
            return df

        df2 = calculate_engagement_metrics(df)
        st.write("Computed engagement metrics:", df2.head())

        st.download_button("Download Processed CSV", df2.to_csv(index=False), "processed.csv")
