"""
train_engagement.py
===================
Train and honestly evaluate engagement prediction models (Likes and Engagement Rate).

Implements rigorous baseline comparisons:
  (a) Predict median / mean
  (b) Log(followers) only
  (c) Followers + non-sentiment metadata features
  (d) Full model with sentiment features (probabilities)

Compares whether sentiment features actually improve over (c).
Saves results to results/engagement_comparison.csv and updates results/metrics.json.
Also saves the production engagement regressors to models/.

Usage:
    python -m src.train_engagement
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.features import build_engagement_features

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTS_CSV = PROJECT_ROOT / "twitter-posts.csv"
ROBERTA_CSV = PROJECT_ROOT / "twitter_with_roberta_labels.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading data for engagement modeling...")
    posts = pd.read_csv(POSTS_CSV)
    roberta = pd.read_csv(ROBERTA_CSV)

    # Merge on tweet text
    df = pd.merge(posts, roberta, left_on="description", right_on="Content", how="inner")
    df = df.dropna(subset=["views", "likes", "followers"])
    df = df[df["views"] > 0]
    print(f"  Rows available for training: {len(df)}")

    # Sentiment probabilities dataframe
    sentiment_df = pd.DataFrame({
        "prob_positive": df["prob_positive"],
        "prob_negative": df["prob_negative"],
        "prob_neutral": df["prob_neutral"],
        "sentiment_confidence": df["teacher_conf"],
    }, index=df.index)

    # Targets
    y_likes_log = np.log1p(df["likes"])
    y_likes_raw = df["likes"]
    y_rate = df["likes"] / df["views"] * 100

    # Build feature sets for comparison
    # (b) log(followers) only
    X_b = pd.DataFrame({"log_followers": np.log1p(df["followers"].astype(float))}, index=df.index)

    # (c) followers + non-sentiment features
    X_c = build_engagement_features(df, sentiment_probs=None)

    # (d) full model (with sentiment features)
    X_d = build_engagement_features(df, sentiment_probs=sentiment_df)

    # Train / test split (fixed random state, 80/20)
    split = train_test_split(
        df.index, y_likes_log, y_likes_raw, y_rate, test_size=0.2, random_state=42
    )
    idx_tr, idx_te, yl_log_tr, yl_log_te, yl_raw_tr, yl_raw_te, yr_tr, yr_te = split

    print(f"  Train samples: {len(idx_tr)}, Test samples: {len(idx_te)}")

    # Evaluation function
    def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, y_raw_te):
        model.fit(X_tr, y_tr)
        pred_log = model.predict(X_te)
        pred_raw = np.expm1(pred_log)

        r2_log = r2_score(y_te, pred_log)
        mae_log = mean_absolute_error(y_te, pred_log)
        mae_raw = mean_absolute_error(y_raw_te, pred_raw)

        # 5-fold CV on full data for r2_log
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        X_full = pd.concat([X_tr, X_te]).loc[df.index]
        y_full = pd.concat([y_tr, y_te]).loc[df.index]
        cv_r2 = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2")

        return {
            "Model Variant": name,
            "Test R² (log)": round(float(r2_log), 4),
            "Test MAE (log)": round(float(mae_log), 4),
            "Test MAE (raw likes)": round(float(mae_raw), 1),
            "5-Fold CV R² (mean)": round(float(cv_r2.mean()), 4),
            "5-Fold CV R² (std)": round(float(cv_r2.std()), 4),
        }

    records = []

    # Baseline (a): Predict Median
    dummy_median = DummyRegressor(strategy="median")
    dummy_median.fit(X_b.loc[idx_tr], yl_log_tr)
    pred_log_a = dummy_median.predict(X_b.loc[idx_te])
    pred_raw_a = np.expm1(pred_log_a)
    records.append({
        "Model Variant": "(a) Baseline: Median Likes",
        "Test R² (log)": round(float(r2_score(yl_log_te, pred_log_a)), 4),
        "Test MAE (log)": round(float(mean_absolute_error(yl_log_te, pred_log_a)), 4),
        "Test MAE (raw likes)": round(float(mean_absolute_error(yl_raw_te, pred_raw_a)), 1),
        "5-Fold CV R² (mean)": 0.0,
        "5-Fold CV R² (std)": 0.0,
    })

    # Baseline (b): log(followers) only
    gbr_b = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    records.append(evaluate_model(
        "(b) Follower Count Only", gbr_b,
        X_b.loc[idx_tr], X_b.loc[idx_te], yl_log_tr, yl_log_te, yl_raw_te
    ))

    # Baseline (c): followers + non-sentiment metadata
    gbr_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    records.append(evaluate_model(
        "(c) Metadata (Followers, Text Len, Media, Verified, Hashtags)", gbr_c,
        X_c.loc[idx_tr], X_c.loc[idx_te], yl_log_tr, yl_log_te, yl_raw_te
    ))

    # Variant (d): full model with sentiment features
    gbr_d = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    records.append(evaluate_model(
        "(d) Full Model (Metadata + Sentiment Probs)", gbr_d,
        X_d.loc[idx_tr], X_d.loc[idx_te], yl_log_tr, yl_log_te, yl_raw_te
    ))

    comp_df = pd.DataFrame(records)
    print("\n--- Engagement Model Comparison Table ---")
    print(comp_df.to_string(index=False))

    comp_csv_path = RESULTS_DIR / "engagement_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved {comp_csv_path}")

    # Check whether (d) beats (c)
    r2_c = comp_df.loc[comp_df["Model Variant"].str.startswith("(c)"), "Test R² (log)"].values[0]
    r2_d = comp_df.loc[comp_df["Model Variant"].str.startswith("(d)"), "Test R² (log)"].values[0]
    cv_c = comp_df.loc[comp_df["Model Variant"].str.startswith("(c)"), "5-Fold CV R² (mean)"].values[0]
    cv_d = comp_df.loc[comp_df["Model Variant"].str.startswith("(d)"), "5-Fold CV R² (mean)"].values[0]

    sentiment_helps = (r2_d > r2_c) and (cv_d > cv_c)
    print(f"\nAnalysis: Does sentiment improve prediction?")
    print(f"  Test R²: (c)={r2_c} vs (d)={r2_d}")
    print(f"  CV R²:   (c)={cv_c} vs (d)={cv_d}")
    if sentiment_helps:
        print("  -> YES: Sentiment features provide incremental predictive power.")
    else:
        print("  -> NO / NEGLIGIBLE: Sentiment features do not meaningfully improve over audience metadata.")

    # Save the best production models:
    # If sentiment helps, production model uses X_d; if not or for inference parity without RoBERTa,
    # Variant (c) has 100% clean parity at inference time without needing RoBERTa probabilities or placeholder values!
    # Let's save model (c) as production if sentiment doesn't help or if inference parity is prioritized.
    # To be fully aligned with Option A/Phase 3:
    # If we use Variant (c) (clean metadata), there are NO placeholder values required!
    # Let's save model trained on X_c to models/engagement_regressor_likes.pkl
    # and also rate model on X_c to models/engagement_regressor_rate.pkl
    prod_model_likes = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    prod_model_likes.fit(X_c.loc[idx_tr], yl_log_tr)
    joblib.dump(prod_model_likes, MODEL_DIR / "engagement_regressor_likes.pkl")

    prod_model_rate = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    prod_model_rate.fit(X_c.loc[idx_tr], yr_tr)
    joblib.dump(prod_model_rate, MODEL_DIR / "engagement_regressor_rate.pkl")
    print(f"Saved production models (trained on metadata features) to {MODEL_DIR}")

    # Update metrics.json
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    metrics_data["engagement_comparison"] = records
    metrics_data["sentiment_improves_engagement"] = bool(sentiment_helps)
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print("Updated results/metrics.json with engagement metrics.")


if __name__ == "__main__":
    main()
