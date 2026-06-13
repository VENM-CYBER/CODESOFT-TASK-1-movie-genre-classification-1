"""
train.py  —  Movie Genre Classification System
------------------------------------------------
Usage:
    python train.py                          # IMDB dataset (data/raw/train_data.txt)
    python train.py --data path/to/file      # custom CSV or TXT
    python train.py --no-eda                 # skip charts (faster)
    python train.py --genres 15              # keep top N genres (default: 15)
    python train.py --lemmatize              # enable lemmatization (slower but higher quality)
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader                import load_data, clean_dataframe, get_dataset_stats
from src.preprocessing.text_cleaner import preprocess_series
from src.visualization.eda          import run_full_eda
from src.models.trainer              import (
    train_and_evaluate, save_artifacts,
    plot_model_comparison, plot_confusion_matrix, print_comparison_table,
)


def main(data_path: str, run_eda: bool = True,
         top_n_genres: int = 15, lemmatize: bool = False):
    print("=" * 65)
    print("  MOVIE GENRE CLASSIFICATION — TRAINING PIPELINE")
    print("=" * 65)

    print(f"\n[Setup] Dataset : {data_path}")
    df = load_data(data_path)
    df = clean_dataframe(df, top_n_genres=top_n_genres)

    stats = get_dataset_stats(df)
    print(f"\n[Stats] Total movies     : {stats['total_movies']:,}")
    print(f"[Stats] Unique genres    : {stats['unique_genres']}")
    print(f"[Stats] Genres           : {', '.join(stats['genres'])}")
    print(f"[Stats] Avg description  : {stats['avg_plot_length']:.0f} words")

    print(f"\n[Preprocess] Cleaning text (lemmatize={lemmatize}) …")
    df["cleaned_plot"] = preprocess_series(df["plot"], lemmatize=lemmatize)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/movies_processed.csv", index=False)
    print("[Preprocess] Saved → data/processed/movies_processed.csv")

    if run_eda:
        run_full_eda(df, cleaned_col="cleaned_plot")

    print("\n[Train] Training models …")
    output = train_and_evaluate(df, text_col="cleaned_plot", label_col="genre")

    print_comparison_table(output["results"])
    plot_model_comparison(output["results"])
    plot_confusion_matrix(output["results"], output["best_model_name"],
                          output["label_classes"])
    save_artifacts(output)

    best = output["results"][output["best_model_name"]]
    print("\n" + "=" * 65)
    print(f"  Training complete!")
    print(f"  Best model  : {output['best_model_name']}")
    print(f"  Accuracy    : {best['accuracy']:.4f}")
    print(f"  F1 Score    : {best['f1_weighted']:.4f}")
    print(f"  Artifacts   : artifacts/models/")
    print(f"  Reports     : artifacts/reports/")
    print("=" * 65 + "\n")
    print("  Run the app:  streamlit run streamlit_app\\main.py")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/raw/train_data.txt")
    parser.add_argument("--no-eda",    action="store_true")
    parser.add_argument("--genres",    type=int, default=15)
    parser.add_argument("--lemmatize", action="store_true")
    args = parser.parse_args()
    main(args.data, run_eda=not args.no_eda,
         top_n_genres=args.genres, lemmatize=args.lemmatize)
