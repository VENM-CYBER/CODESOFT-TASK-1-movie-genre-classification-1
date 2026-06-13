"""
src/data/loader.py
-------------------
Data loading for the Movie Genre Classification System.
Supports both:
  - The real IMDB dataset (.txt files with  :::  separator)
  - Generic CSV files with columns: movie_title, plot, genre
"""

import os
import pandas as pd


# ── Top genres to keep (drop very rare / NSFW ones) ──────────────────────────
TOP_GENRES = [
    "drama", "documentary", "comedy", "horror", "thriller",
    "action", "romance", "sci-fi", "animation", "fantasy",
    "adventure", "family", "mystery", "crime", "western",
    "biography", "history", "musical", "sport", "war",
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load movie dataset from a .txt (IMDB :::) or .csv file.

    For .txt files the expected format is:
        ID ::: TITLE ::: GENRE ::: DESCRIPTION

    For .csv files the expected columns are:
        movie_title, plot, genre

    Returns a dataframe with columns: movie_title, plot, genre
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place train_data.txt in data/raw/ or run python train.py."
        )

    if path.endswith(".txt"):
        return _load_txt(path)
    else:
        return _load_csv(path)


def _load_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=" ::: ", engine="python", header=None,
        names=["id", "movie_title", "genre", "plot"],
        on_bad_lines="skip",
    )
    df = df[["movie_title", "plot", "genre"]].copy()
    return df


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"movie_title", "plot", "genre"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_dataframe(df: pd.DataFrame, top_n_genres: int = 20,
                    min_words: int = 10) -> pd.DataFrame:
    """
    Clean the dataframe:
      1. Strip whitespace & normalise genre to title-case
      2. Drop missing/empty plots and genres
      3. Drop duplicate plots
      4. Optionally keep only top N genres (by frequency)
      5. Drop very short plots (< min_words)
    """
    original_len = len(df)

    for col in ["movie_title", "plot", "genre"]:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with missing content
    df = df[df["plot"].notna() & (df["plot"] != "") & (df["plot"] != "nan")]
    df = df[df["genre"].notna() & (df["genre"] != "") & (df["genre"] != "nan")]

    # Normalise genre casing
    df["genre"] = df["genre"].str.strip().str.lower()

    # Keep top N genres only
    top_genres = df["genre"].value_counts().head(top_n_genres).index.tolist()
    df = df[df["genre"].isin(top_genres)]

    # Drop very short descriptions
    df = df[df["plot"].str.split().apply(len) >= min_words]

    # Drop duplicates
    df = df.drop_duplicates(subset=["plot"]).reset_index(drop=True)

    # Title-case genre for display
    df["genre"] = df["genre"].str.title()

    dropped = original_len - len(df)
    print(f"[Loader] {original_len:,} → {len(df):,} rows  ({dropped:,} removed)")
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    wc = df["plot"].str.split().apply(len)
    return {
        "total_movies":    len(df),
        "unique_genres":   df["genre"].nunique(),
        "genres":          sorted(df["genre"].unique().tolist()),
        "genre_counts":    df["genre"].value_counts().to_dict(),
        "avg_plot_length": round(wc.mean(), 1),
        "max_plot_length": int(wc.max()),
        "min_plot_length": int(wc.min()),
        "missing_plots":   int(df["plot"].isna().sum()),
        "missing_genres":  int(df["genre"].isna().sum()),
    }
