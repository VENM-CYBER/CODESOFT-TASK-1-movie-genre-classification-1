"""
predict.py
-----------
Command-line prediction interface for the Movie Genre Classification System.

Usage:
    # Interactive mode
    python predict.py

    # Direct prediction
    python predict.py --plot "A soldier races to stop a nuclear threat in Tokyo."

    # Batch prediction from a file (one plot per line)
    python predict.py --file plots.txt

    # Show top-3 predictions
    python predict.py --plot "..." --top3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.predictor import GenrePredictor
from src.utils.helpers    import generate_prediction_report_text


def predict_single(predictor: GenrePredictor, plot: str, top3: bool = True) -> None:
    genre    = predictor.predict_genre(plot)
    top3_res = predictor.predict_top3_genres(plot)
    probs    = predictor.predict_probability(plot)

    print("\n" + "=" * 55)
    print(f"  Predicted Genre : {genre}")
    print("=" * 55)

    if top3:
        print("\n  Top 3 Predictions:")
        for i, item in enumerate(top3_res, 1):
            bar = "█" * int(item["confidence"] * 25)
            print(f"  {i}. {item['genre']:<22} {item['confidence']*100:5.1f}%  {bar}")

    print("\n  All Genre Probabilities:")
    for g, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 25)
        print(f"     {g:<22} {p*100:5.1f}%  {bar}")

    print()


def interactive_mode(predictor: GenrePredictor) -> None:
    print("\n🎬  Movie Genre Classifier — Interactive Mode")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        plot = input("Enter a movie plot description:\n> ").strip()
        if plot.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not plot:
            print("Please enter a non-empty plot description.\n")
            continue
        predict_single(predictor, plot)


def batch_mode(predictor: GenrePredictor, file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    with open(file_path) as f:
        plots = [line.strip() for line in f if line.strip()]

    print(f"\nProcessing {len(plots)} plots …\n")
    predictions = predictor.batch_predict(plots)

    for i, (plot, pred) in enumerate(zip(plots, predictions), 1):
        snippet = plot[:80] + ("…" if len(plot) > 80 else "")
        print(f"  [{i:03d}] {pred:<22} | {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict movie genres from plot descriptions")
    parser.add_argument("--plot", type=str, default=None, help="Single plot text")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file (one plot per line)")
    parser.add_argument("--top3", action="store_true", help="Show top-3 predictions")
    args = parser.parse_args()

    try:
        predictor = GenrePredictor()
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    if args.file:
        batch_mode(predictor, args.file)
    elif args.plot:
        predict_single(predictor, args.plot, top3=args.top3)
    else:
        interactive_mode(predictor)
