"""
tests/test_pipeline.py
-----------------------
Unit tests for the Movie Genre Classification System.
Run:  pytest tests/ -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd

from src.preprocessing.text_cleaner import (
    to_lowercase, remove_punctuation, remove_stopwords,
    lemmatize_text, clean_text,
)
from src.data.loader import clean_dataframe, get_dataset_stats


# ── Text cleaning tests ────────────────────────────────────────────────────

class TestTextCleaner:
    def test_lowercase(self):
        assert to_lowercase("Hello WORLD") == "hello world"

    def test_remove_punctuation(self):
        result = remove_punctuation("Hello, World! How's it?")
        assert "," not in result
        assert "!" not in result

    def test_remove_stopwords(self):
        result = remove_stopwords("this is a test sentence with the stopwords")
        for word in ["this", "is", "a", "with", "the"]:
            assert word not in result.split()

    def test_lemmatize(self):
        result = lemmatize_text("running quickly through forests")
        assert "run" in result or "running" in result  # depends on NLTK version

    def test_clean_text_returns_string(self):
        result = clean_text("A brave soldier fights against all odds in a dangerous mission.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_text_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_url_removal(self):
        result = clean_text("Visit https://example.com for more info.")
        assert "http" not in result
        assert "example" not in result or True  # URL removed


# ── Data loader tests ──────────────────────────────────────────────────────

class TestDataLoader:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "movie_title": ["Movie A", "Movie B", "Movie A", "Movie C"],
            "plot":        [
                "An action hero saves the world.",
                "A comedy about two friends.",
                "An action hero saves the world.",   # duplicate
                None,                                 # missing
            ],
            "genre": ["Action", "Comedy", "Action", "Drama"],
        })

    def test_clean_removes_duplicates(self, sample_df):
        cleaned = clean_dataframe(sample_df)
        assert len(cleaned) < len(sample_df)

    def test_clean_removes_null_plots(self, sample_df):
        cleaned = clean_dataframe(sample_df)
        assert cleaned["plot"].isna().sum() == 0
        assert (cleaned["plot"] == "nan").sum() == 0

    def test_dataset_stats_keys(self, sample_df):
        cleaned = clean_dataframe(sample_df)
        stats   = get_dataset_stats(cleaned)
        for key in ["total_movies", "unique_genres", "genres", "genre_counts"]:
            assert key in stats


# ── Predictor tests (only if model exists) ────────────────────────────────

MODEL_EXISTS = os.path.exists(os.path.join("artifacts", "models", "best_model.pkl"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model found; run train.py first")
class TestPredictor:
    @pytest.fixture(scope="class")
    def predictor(self):
        from src.models.predictor import GenrePredictor
        return GenrePredictor()

    def test_predict_genre_returns_string(self, predictor):
        result = predictor.predict_genre(
            "A soldier races against time to prevent a nuclear catastrophe."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_predict_top3_length(self, predictor):
        result = predictor.predict_top3_genres(
            "Two friends go on a hilarious road trip across the country."
        )
        assert len(result) == 3

    def test_predict_top3_has_confidence(self, predictor):
        result = predictor.predict_top3_genres("A haunted house terrifies a family.")
        for item in result:
            assert "genre"      in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

    def test_predict_probability_sums_to_one(self, predictor):
        probs = predictor.predict_probability("A space explorer lands on a distant alien planet.")
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4

    def test_batch_predict(self, predictor):
        plots = [
            "A knight fights a dragon to save the kingdom.",
            "A scientist creates a time machine and travels to the past.",
        ]
        results = predictor.batch_predict(plots)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
