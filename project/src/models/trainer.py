"""
src/models/trainer.py
----------------------
Model training, cross-validation, evaluation, and saving for the
Movie Genre Classification System.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODELS_DIR  = os.path.join("artifacts", "models")
REPORTS_DIR = os.path.join("artifacts", "reports")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── TF-IDF configuration ────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features=30_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

# ── Classifier configurations ───────────────────────────────────────────────
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        random_state=42,
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Linear SVM": LinearSVC(
        max_iter=2000,
        C=1.0,
        random_state=42,
    ),
}


# ── Training pipeline ────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    pipelines = {}
    for name, clf in CLASSIFIERS.items():
        pipelines[name] = Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",   clf),
        ])
    return pipelines


def train_and_evaluate(
    df: pd.DataFrame,
    text_col: str = "cleaned_plot",
    label_col: str = "genre",
    test_size: float = 0.20,
    cv_folds: int = 5,
) -> dict:
    """
    Train all classifiers, evaluate with CV and hold-out test set,
    and return a results dictionary.

    Returns
    -------
    dict with keys:
        results          – per-model metrics dict
        best_model_name  – name of the best model by F1
        best_pipeline    – fitted sklearn Pipeline (best model)
        X_test           – test raw text
        y_test           – test true labels
        label_classes    – list of genre class names
    """
    X = df[text_col].fillna("").astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[Train] Train: {len(X_train)} | Test: {len(X_test)} samples")

    pipelines = build_pipelines()
    results   = {}
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, pipeline in pipelines.items():
        print(f"\n[Train] Training: {name} …")

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_splitter, scoring="f1_weighted", n_jobs=-1
        )
        print(f"         CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full train set
        pipeline.fit(X_train, y_train)

        # Hold-out evaluation
        y_pred = pipeline.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy":          round(acc,  4),
            "precision":         round(prec, 4),
            "recall":            round(rec,  4),
            "f1_weighted":       round(f1,   4),
            "cv_f1_mean":        round(cv_scores.mean(), 4),
            "cv_f1_std":         round(cv_scores.std(),  4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
            "pipeline":          pipeline,
        }
        print(f"         Test  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    # Best model
    best_name = max(
        results,
        key=lambda n: results[n]["f1_weighted"]
    )
    print(f"\n[Train] ✅  Best model: {best_name}  (F1={results[best_name]['f1_weighted']:.4f})")

    return {
        "results":         results,
        "best_model_name": best_name,
        "best_pipeline":   results[best_name]["pipeline"],
        "X_test":          X_test,
        "y_test":          y_test,
        "label_classes":   sorted(y.unique().tolist()),
    }


# ── Saving artefacts ─────────────────────────────────────────────────────────

def save_artifacts(train_output: dict) -> None:
    best_pipeline = train_output["best_pipeline"]
    best_name     = train_output["best_model_name"]

    # Save full pipeline (includes fitted TF-IDF + classifier)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"[Save] Model  → {model_path}")

    # Save TF-IDF separately for convenience
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(best_pipeline.named_steps["tfidf"], tfidf_path)
    print(f"[Save] TF-IDF → {tfidf_path}")

    # Save metrics JSON
    metrics = {}
    for name, res in train_output["results"].items():
        metrics[name] = {k: v for k, v in res.items() if k not in ("pipeline",)}
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Save] Metrics → {metrics_path}")

    # Save model metadata
    meta = {
        "best_model": best_name,
        "label_classes": train_output["label_classes"],
        "best_f1": train_output["results"][best_name]["f1_weighted"],
    }
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Save] Meta   → {meta_path}")


# ── Visualisations ───────────────────────────────────────────────────────────

def plot_model_comparison(results: dict, save: bool = True) -> plt.Figure:
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_weighted"]
    data    = {m: [results[n][m] for n in names] for m in metrics}

    x      = np.arange(len(names))
    width  = 0.18
    colors = ["#6C63FF", "#FF6584", "#43B89C", "#F5A623"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, data[metric], width,
                      label=metric.replace("_", " ").title(),
                      color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison Dashboard", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {path}")
    return fig


def plot_confusion_matrix(results: dict, best_name: str,
                           label_classes: list, save: bool = True) -> plt.Figure:
    cm = np.array(results[best_name]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Confusion Matrix — {best_name}", fontsize=15, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (Recall)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_classes, yticklabels=label_classes,
            linewidths=0.5, ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {path}")
    return fig


def print_comparison_table(results: dict) -> None:
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":          name,
            "Accuracy":       f"{res['accuracy']:.4f}",
            "Precision":      f"{res['precision']:.4f}",
            "Recall":         f"{res['recall']:.4f}",
            "F1 (Weighted)":  f"{res['f1_weighted']:.4f}",
            "CV F1 (Mean)":   f"{res['cv_f1_mean']:.4f}",
            "CV F1 (±Std)":   f"{res['cv_f1_std']:.4f}",
        })
    table = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 72)
    print("MODEL COMPARISON TABLE")
    print("=" * 72)
    print(table.to_string())
    print("=" * 72 + "\n")
