"""
src/models/predictor.py
------------------------
Prediction utilities for the Movie Genre Classification System.
Loads the saved best_model.pkl and exposes clean prediction functions.
"""

import os
import json
import numpy as np
import joblib

MODELS_DIR = os.path.join("artifacts", "models")


def _load_pipeline(model_path: str = None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train.py first to train and save the model."
        )
    return joblib.load(model_path)


def _load_meta(meta_path: str = None) -> dict:
    if meta_path is None:
        meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


# ── Public API ────────────────────────────────────────────────────────────────

class GenrePredictor:
    """
    Convenience wrapper around the saved genre-classification pipeline.

    Usage
    -----
    predictor = GenrePredictor()
    genre      = predictor.predict_genre("A soldier races to stop a nuclear threat.")
    top3       = predictor.predict_top3_genres("...")
    probs      = predictor.predict_probability("...")
    """

    def __init__(self, model_path: str = None):
        self._pipeline = _load_pipeline(model_path)
        self._meta     = _load_meta()

        # Retrieve class labels from the classifier
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "classes_"):
            self._classes = list(clf.classes_)
        else:
            # LinearSVC doesn't expose .predict_proba; use meta
            self._classes = self._meta.get("label_classes", [])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_proba(self, text: str) -> np.ndarray:
        """Return probability vector (or decision-function softmax for SVM)."""
        from src.preprocessing.text_cleaner import clean_text
        cleaned = clean_text(text)

        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self._pipeline.predict_proba([cleaned])[0]
        else:
            # Decision function → softmax
            dec = self._pipeline.decision_function([cleaned])[0]
            e   = np.exp(dec - dec.max())
            proba = e / e.sum()
        return proba

    # ── Public methods ───────────────────────────────────────────────────────

    def predict_genre(self, plot: str) -> str:
        """
        Predict a single genre label.

        Parameters
        ----------
        plot : str  — raw plot / description text

        Returns
        -------
        str  — predicted genre label
        """
        from src.preprocessing.text_cleaner import clean_text
        cleaned = clean_text(plot)
        return self._pipeline.predict([cleaned])[0]

    def predict_top3_genres(self, plot: str) -> list[dict]:
        """
        Return the top-3 most likely genres with confidence scores.

        Returns
        -------
        list of dicts: [{"genre": str, "confidence": float}, ...]
        """
        proba   = self._get_proba(plot)
        top_idx = np.argsort(proba)[::-1][:3]
        return [
            {"genre": self._classes[i], "confidence": round(float(proba[i]), 4)}
            for i in top_idx
        ]

    def predict_probability(self, plot: str) -> dict:
        """
        Return a {genre: probability} dictionary for all genres.

        Returns
        -------
        dict[str, float]
        """
        proba = self._get_proba(plot)
        return {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)}

    def batch_predict(self, plots: list[str]) -> list[str]:
        """
        Predict genres for a list of plot texts.

        Returns
        -------
        list[str]
        """
        from src.preprocessing.text_cleaner import clean_text
        cleaned = [clean_text(p) for p in plots]
        return self._pipeline.predict(cleaned).tolist()


# ── Module-level convenience functions ──────────────────────────────────────
_default_predictor: GenrePredictor | None = None


def _get_predictor() -> GenrePredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = GenrePredictor()
    return _default_predictor


def predict_genre(plot: str) -> str:
    return _get_predictor().predict_genre(plot)


def predict_top3_genres(plot: str) -> list[dict]:
    return _get_predictor().predict_top3_genres(plot)


def predict_probability(plot: str) -> dict:
    return _get_predictor().predict_probability(plot)
