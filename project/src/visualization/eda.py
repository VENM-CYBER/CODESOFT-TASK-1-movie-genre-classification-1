"""
src/visualization/eda.py
-------------------------
Exploratory Data Analysis visualizations for the Movie Genre Classification System.
All functions save their figures to the artifacts/reports/ directory.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

REPORTS_DIR = os.path.join("artifacts", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = [
    "#6C63FF", "#FF6584", "#43B89C", "#F5A623", "#4A90E2",
    "#E57373", "#64B5F6", "#81C784", "#FFD54F", "#CE93D8",
]

sns.set_theme(style="darkgrid", palette=PALETTE)


# ── 1. Genre Distribution ───────────────────────────────────────────────────

def plot_genre_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = df["genre"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Genre Distribution", fontsize=18, fontweight="bold", y=1.02)

    # Bar chart
    ax1 = axes[0]
    bars = ax1.barh(counts.index, counts.values,
                    color=PALETTE[:len(counts)], edgecolor="white", linewidth=0.6)
    ax1.set_xlabel("Number of Movies", fontsize=12)
    ax1.set_title("Horizontal Bar Chart", fontsize=13)
    ax1.bar_label(bars, padding=4, fontsize=10)
    ax1.invert_yaxis()

    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(counts)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Proportion Pie Chart", fontsize=13)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "genre_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    return fig


# ── 2. Plot Length Distribution ─────────────────────────────────────────────

def plot_plot_length_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    df = df.copy()
    df["word_count"] = df["plot"].str.split().apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Plot Length Analysis", fontsize=18, fontweight="bold")

    # Overall histogram
    ax1 = axes[0]
    ax1.hist(df["word_count"], bins=40, color=PALETTE[0], edgecolor="white")
    ax1.set_xlabel("Word Count", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Overall Word-Count Distribution", fontsize=13)
    ax1.axvline(df["word_count"].mean(), color=PALETTE[1], linestyle="--",
                linewidth=1.8, label=f'Mean: {df["word_count"].mean():.0f}')
    ax1.legend()

    # Box plot per genre
    ax2 = axes[1]
    genre_order = df.groupby("genre")["word_count"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="genre", y="word_count", order=genre_order,
                palette=PALETTE[:df["genre"].nunique()], ax=ax2)
    ax2.set_xlabel("Genre", fontsize=12)
    ax2.set_ylabel("Word Count", fontsize=12)
    ax2.set_title("Word Count by Genre", fontsize=13)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "plot_length_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    return fig


# ── 3. Top Words per Genre ──────────────────────────────────────────────────

def plot_top_words_per_genre(df: pd.DataFrame, cleaned_col: str = "cleaned_plot",
                              top_n: int = 10, save: bool = True) -> plt.Figure:
    if cleaned_col not in df.columns:
        print(f"[EDA] Column '{cleaned_col}' not found; skipping top-words chart.")
        return None

    genres = sorted(df["genre"].unique())
    n_cols = 4
    n_rows = (len(genres) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    fig.suptitle(f"Top {top_n} Words per Genre", fontsize=18, fontweight="bold")
    axes = axes.flatten()

    for i, genre in enumerate(genres):
        texts = " ".join(df[df["genre"] == genre][cleaned_col].dropna())
        word_counts = Counter(texts.split()).most_common(top_n)
        if not word_counts:
            continue
        words, counts = zip(*word_counts)
        ax = axes[i]
        ax.barh(words[::-1], counts[::-1], color=PALETTE[i % len(PALETTE)])
        ax.set_title(genre, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)

    for j in range(len(genres), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "top_words_per_genre.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    return fig


# ── 4. Word Cloud ───────────────────────────────────────────────────────────

def plot_word_cloud(df: pd.DataFrame, cleaned_col: str = "cleaned_plot",
                    save: bool = True) -> plt.Figure:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[EDA] wordcloud not installed; skipping word cloud.")
        return None

    if cleaned_col not in df.columns:
        print(f"[EDA] Column '{cleaned_col}' not found; skipping word cloud.")
        return None

    all_text = " ".join(df[cleaned_col].dropna())
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — All Movie Plots", fontsize=18, fontweight="bold", pad=16)
    plt.tight_layout()

    if save:
        path = os.path.join(REPORTS_DIR, "word_cloud.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    return fig


# ── 5. Genre Word Clouds ────────────────────────────────────────────────────

def plot_genre_word_clouds(df: pd.DataFrame, cleaned_col: str = "cleaned_plot",
                            save: bool = True) -> plt.Figure:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[EDA] wordcloud not installed; skipping genre word clouds.")
        return None

    if cleaned_col not in df.columns:
        return None

    genres = sorted(df["genre"].unique())
    n_cols = 3
    n_rows = (len(genres) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    fig.suptitle("Word Clouds per Genre", fontsize=18, fontweight="bold")
    axes = axes.flatten()
    cmaps = ["viridis", "plasma", "magma", "inferno", "cividis",
             "cool", "hot", "spring", "summer", "autumn", "winter"]

    for i, genre in enumerate(genres):
        text = " ".join(df[df["genre"] == genre][cleaned_col].dropna())
        if not text.strip():
            continue
        wc = WordCloud(width=500, height=300, background_color="white",
                       colormap=cmaps[i % len(cmaps)], max_words=100,
                       collocations=False).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(genre, fontsize=12, fontweight="bold")

    for j in range(len(genres), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "genre_word_clouds.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
    return fig


# ── 6. Run all EDA ──────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame, cleaned_col: str = "cleaned_plot") -> None:
    print("\n[EDA] Generating all visualizations …")
    plot_genre_distribution(df)
    plot_plot_length_distribution(df)
    plot_top_words_per_genre(df, cleaned_col=cleaned_col)
    plot_word_cloud(df, cleaned_col=cleaned_col)
    plot_genre_word_clouds(df, cleaned_col=cleaned_col)
    print("[EDA] All visualizations saved to artifacts/reports/\n")
