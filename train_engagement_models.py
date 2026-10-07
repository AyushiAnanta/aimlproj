"""
train_engagement_models.py
==========================
Run this ONCE from your project root (C:\\Users\\Public\\Projects\\aimlproj)
to generate the two engagement model files the Streamlit app expects.

Usage:
    cd C:\\Users\\Public\\Projects\\aimlproj
    python train_engagement_models.py

Output:
    models/engagement_regressor_likes.pkl
    models/engagement_regressor_rate.pkl

Requirements (already installed if you ran the Streamlit app):
    pip install scikit-learn pandas joblib numpy
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
POSTS_CSV   = os.path.join(SCRIPT_DIR, "twitter-posts.csv")
ROBERTA_CSV = os.path.join(SCRIPT_DIR, "twitter_with_roberta_labels.csv")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 2. LOAD & MERGE
# ──────────────────────────────────────────────
print("Loading data...")
posts   = pd.read_csv(POSTS_CSV)
roberta = pd.read_csv(ROBERTA_CSV)

# Merge on tweet text — 998/1000 rows match
df = pd.merge(posts, roberta, left_on="description", right_on="Content", how="inner")
df = df.dropna(subset=["views", "likes", "followers"])
df = df[df["views"] > 0]
print(f"  Rows available for training: {len(df)}")

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def build_features(df):
    X = pd.DataFrame()

    # Audience size — log-scaled to reduce viral outlier effect
    X["log_followers"]        = np.log1p(df["followers"])

    # RoBERTa sentiment probabilities (continuous 0-1, not just a label bucket)
    X["prob_positive"]        = df["prob_positive"]
    X["prob_negative"]        = df["prob_negative"]
    X["prob_neutral"]         = df["prob_neutral"]

    # How confident the sentiment model was
    X["sentiment_confidence"] = df["teacher_conf"]

    # Single polarity score: +1 = fully positive, -1 = fully negative
    X["sentiment_score"]      = df["prob_positive"] - df["prob_negative"]

    # Text features
    X["text_length"]          = df["description"].str.len()
    X["word_count"]           = df["description"].str.split().str.len()

    # Media / hashtag signals
    X["has_hashtag"]          = df["hashtags"].apply(
        lambda x: 1 if str(x) not in ["null", "None", "nan", "[]"] else 0
    )
    X["has_media"]            = df["photos"].apply(
        lambda x: 1 if str(x) not in ["null", "None", "nan", "[]"] else 0
    )
    X["is_verified"]          = df["is_verified"].apply(
        lambda x: 1 if str(x).lower() in ["true", "1"] else 0
    )

    return X

X = build_features(df)

# ──────────────────────────────────────────────
# 4. TARGETS
# ──────────────────────────────────────────────
# Log-transform likes: range is 1–200k, log scale stops outliers dominating
y_likes = np.log1p(df["likes"])
y_rate  = df["likes"] / df["views"] * 100   # engagement rate %

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Training rows: {len(X)}")

# ──────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, yl_train, yl_test, yr_train, yr_test = train_test_split(
    X, y_likes, y_rate, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 6. TRAIN
# ──────────────────────────────────────────────
print("\nTraining likes model...")
model_likes = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8, random_state=42
)
model_likes.fit(X_train, yl_train)

pred_likes = np.expm1(model_likes.predict(X_test))
true_likes = np.expm1(yl_test)
print(f"  MAE:  {mean_absolute_error(true_likes, pred_likes):.0f} likes")
print(f"  R²:   {r2_score(yl_test, model_likes.predict(X_test)):.3f}")

print("\nTraining engagement rate model...")
model_rate = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8, random_state=42
)
model_rate.fit(X_train, yr_train)
print(f"  MAE:  {mean_absolute_error(yr_test, model_rate.predict(X_test)):.3f}%")
print(f"  R²:   {r2_score(yr_test, model_rate.predict(X_test)):.3f}")

# ──────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
print("\nTop features for likes prediction:")
for feat, imp in sorted(zip(X.columns, model_likes.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat:30s} {imp:.3f}")

# ──────────────────────────────────────────────
# 8. SAVE
# ──────────────────────────────────────────────
joblib.dump(model_likes, os.path.join(MODEL_DIR, "engagement_regressor_likes.pkl"))
joblib.dump(model_rate,  os.path.join(MODEL_DIR, "engagement_regressor_rate.pkl"))

print(f"\n✅  Saved to: {MODEL_DIR}")
print("Restart your Streamlit app — it will auto-load both models.")