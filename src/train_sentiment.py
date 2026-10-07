"""
train_sentiment.py
==================
Train three TF-IDF sentiment classifiers using RoBERTa pseudo-labels
(NOT human annotations) via a pseudo-labeling-with-a-transformer-teacher
approach.

Saves standard sklearn Pipeline objects to ./models/ — no custom classes
required at load time.

Usage
-----
    cd <project root>
    python -m src.train_sentiment

Outputs
-------
    models/pipe_lr.pkl
    models/pipe_svm.pkl
    models/pipe_sgd.pkl
    results/classification_report.txt
    results/confusion_matrix.png
    results/metrics.json
    results/cv_results.csv
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.features import clean_text

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV     = PROJECT_ROOT / "twitter_with_roberta_labels.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def make_features() -> FeatureUnion:
    """Create a *new* (unfitted) blended TF-IDF feature extractor."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=3, max_features=60_000,
            strip_accents="unicode", sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            min_df=2, max_features=120_000,
            strip_accents="unicode", sublinear_tf=True,
        )),
    ])


def main() -> None:
    # ── 1. Load data ─────────────────────────────────────────────
    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    df["Content_clean"] = df["Content"].fillna("").map(clean_text)

    X_all = df["Content_clean"]
    y_all = df["transformer_sentiment"]
    w_all = df["teacher_conf"].astype(float).clip(0.3, 1.0)

    print(f"  Total rows: {len(df)}")
    print(f"  Label distribution (RoBERTa pseudo-labels):")
    print(y_all.value_counts().to_string())

    # ── 2. Stratified split (BEFORE any fitting) ─────────────────
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X_all, y_all, w_all,
        test_size=0.2, stratify=y_all, random_state=42,
    )
    print(f"\n  Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # ── 3. Build pipelines (TF-IDF fitted INSIDE the Pipeline) ──
    # Each pipeline gets its own FeatureUnion → no shared-state bug.
    classifiers = {
        "LR":  LogisticRegression(
            C=3, solver="liblinear",
            class_weight="balanced", max_iter=5000,
        ),
        "SVM": LinearSVC(
            C=1.5, class_weight="balanced", dual=False,
        ),
        "SGD": SGDClassifier(
            loss="modified_huber", alpha=1e-4,
            max_iter=2000, early_stopping=True,
            n_iter_no_change=5, class_weight="balanced",
            random_state=42,
        ),
    }

    pipelines: dict[str, Pipeline] = {}
    for name, clf in classifiers.items():
        pipelines[name] = Pipeline([
            ("feat", make_features()),   # fresh, unfitted copy
            ("clf",  clf),
        ])

    # ── 4. Train with and without sample_weight ──────────────────
    results_all: dict = {}

    for use_weights in [True, False]:
        tag = "weighted" if use_weights else "unweighted"
        print(f"\n{'='*60}")
        print(f"Training ({tag}) ...")
        print(f"{'='*60}")

        fitted: dict[str, Pipeline] = {}
        preds: dict[str, np.ndarray] = {}

        for name, pipe_template in pipelines.items():
            # Must re-create fresh pipeline for each run
            pipe = Pipeline([
                ("feat", make_features()),
                ("clf",  type(pipe_template.named_steps["clf"])(
                    **pipe_template.named_steps["clf"].get_params()
                )),
            ])

            fit_params = {}
            if use_weights:
                fit_params["clf__sample_weight"] = w_tr.values

            pipe.fit(X_tr, y_tr, **fit_params)
            fitted[name] = pipe
            preds[name] = pipe.predict(X_te)

        # Majority vote
        stack = np.vstack([preds[n] for n in ["LR", "SVM", "SGD"]]).T
        vote_pred = np.array([
            pd.Series(row).value_counts().index[0] for row in stack
        ])

        preds["Ensemble"] = vote_pred

        # ── Metrics ──────────────────────────────────────────────
        model_metrics = {}
        for name, pred in preds.items():
            # NOTE: these measure agreement with RoBERTa pseudo-labels,
            # NOT accuracy against human ground truth.
            agreement = float(accuracy_score(y_te, pred))
            macro_f1 = float(f1_score(y_te, pred, average="macro"))
            report = classification_report(y_te, pred)
            cm = confusion_matrix(y_te, pred, labels=["Negative", "Neutral", "Positive"])

            model_metrics[name] = {
                "agreement_with_roberta": round(agreement, 4),
                "macro_f1": round(macro_f1, 4),
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            }

            print(f"\n--- {name} ({tag}) ---")
            print(f"  Agreement with RoBERTa labels: {agreement:.4f}")
            print(f"  Macro-F1: {macro_f1:.4f}")
            print(report)

        results_all[tag] = model_metrics

        # Save the weighted versions as the production models
        if use_weights:
            for name, pipe in fitted.items():
                fname = {
                    "LR": "pipe_lr.pkl",
                    "SVM": "pipe_svm.pkl",
                    "SGD": "pipe_sgd.pkl",
                }.get(name)
                if fname:
                    joblib.dump(pipe, MODEL_DIR / fname)
                    print(f"  Saved {fname}")

    # ── 5. Majority-class baseline ───────────────────────────────
    print(f"\n{'='*60}")
    print("Baseline: majority class")
    print(f"{'='*60}")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    dummy_pred = dummy.predict(X_te)
    dummy_agreement = float(accuracy_score(y_te, dummy_pred))
    dummy_f1 = float(f1_score(y_te, dummy_pred, average="macro"))
    print(f"  Agreement with RoBERTa labels: {dummy_agreement:.4f}")
    print(f"  Macro-F1: {dummy_f1:.4f}")

    results_all["baseline_majority_class"] = {
        "agreement_with_roberta": round(dummy_agreement, 4),
        "macro_f1": round(dummy_f1, 4),
    }

    # ── 6. Stratified 5-fold CV (weighted and unweighted) ────────
    print(f"\n{'='*60}")
    print("Stratified 5-fold CV on full dataset")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for name in ["LR", "SVM", "SGD"]:
        for use_w in [True, False]:
            tag_w = "weighted" if use_w else "unweighted"
            pipe = Pipeline([
                ("feat", make_features()),
                ("clf",  type(pipelines[name].named_steps["clf"])(
                    **pipelines[name].named_steps["clf"].get_params()
                )),
            ])

            fit_params = {}
            if use_w:
                fit_params["clf__sample_weight"] = w_all.values

            scores = cross_val_score(
                pipe, X_all, y_all, cv=cv,
                scoring="f1_macro",
                fit_params=fit_params,
            )
            mean_s, std_s = scores.mean(), scores.std()
            print(f"  {name} ({tag_w}): macro-F1 = {mean_s:.4f} ± {std_s:.4f}")
            cv_records.append({
                "model": name,
                "weighting": tag_w,
                "cv_mean_macro_f1": round(float(mean_s), 4),
                "cv_std_macro_f1": round(float(std_s), 4),
                "folds": scores.tolist(),
            })

    results_all["cv_results"] = cv_records

    # ── 7. Hyperparameter search (train split only) ──────────────
    print(f"\n{'='*60}")
    print("GridSearchCV on LR (train split only)")
    print(f"{'='*60}")

    param_grid = {
        "feat__word__max_features": [40_000, 80_000],
        "feat__word__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [1, 3, 5],
    }
    gs_pipe = Pipeline([
        ("feat", make_features()),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=5000,
        )),
    ])
    gs = GridSearchCV(
        gs_pipe, param_grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    gs.fit(X_tr, y_tr, clf__sample_weight=w_tr.values)
    gs_test_pred = gs.best_estimator_.predict(X_te)
    gs_test_f1 = float(f1_score(y_te, gs_test_pred, average="macro"))

    print(f"  Best CV macro-F1: {gs.best_score_:.4f}")
    print(f"  Best params: {gs.best_params_}")
    print(f"  Test macro-F1 (single evaluation): {gs_test_f1:.4f}")

    results_all["gridsearch"] = {
        "best_cv_macro_f1": round(float(gs.best_score_), 4),
        "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "test_macro_f1": round(gs_test_f1, 4),
    }

    # ── 8. Save results ─────────────────────────────────────────
    # metrics.json
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nSaved results/metrics.json")

    # classification_report.txt  (weighted ensemble on test set)
    weighted_ensemble_report = results_all["weighted"]["Ensemble"]["classification_report"]
    report_txt = (
        "Sentiment Classification Report\n"
        "================================\n"
        "NOTE: Labels are RoBERTa pseudo-labels, NOT human annotations.\n"
        "Metrics measure agreement with RoBERTa, not true accuracy.\n\n"
        f"Label distribution (full dataset, n={len(df)}):\n"
        f"{y_all.value_counts().to_string()}\n\n"
        f"Test set (n={len(X_te)}):\n"
        f"{weighted_ensemble_report}\n"
        f"Majority-class baseline agreement: {dummy_agreement:.4f}\n"
        f"Majority-class baseline macro-F1:  {dummy_f1:.4f}\n"
    )
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_txt)
    print("Saved results/classification_report.txt")

    # confusion_matrix.png
    cm = confusion_matrix(y_te, results_all["weighted"]["Ensemble"]["confusion_matrix"]
                          if False else vote_pred,  # use the last computed vote_pred
                          labels=["Negative", "Neutral", "Positive"])
    # Re-compute to be safe: use weighted preds
    # Load saved models and re-predict
    loaded = [joblib.load(MODEL_DIR / f) for f in ["pipe_lr.pkl", "pipe_svm.pkl", "pipe_sgd.pkl"]]
    loaded_preds = [m.predict(X_te) for m in loaded]
    loaded_stack = np.vstack(loaded_preds).T
    loaded_vote = np.array([pd.Series(row).value_counts().index[0] for row in loaded_stack])
    cm = confusion_matrix(y_te, loaded_vote, labels=["Negative", "Neutral", "Positive"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (agreement with RoBERTa pseudo-labels)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved results/confusion_matrix.png")

    # cv_results.csv
    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    print("Saved results/cv_results.csv")

    print(f"\n[OK] Sentiment training complete. Models in {MODEL_DIR}")


if __name__ == "__main__":
    main()
