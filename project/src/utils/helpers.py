"""
src/utils/helpers.py
---------------------
Miscellaneous utility functions used across the project.
"""

import os
import json
import random
import datetime
import pandas as pd


# ── Sample movie plots (for UI demo) ─────────────────────────────────────────

SAMPLE_PLOTS = {
    "Action":       "A decorated soldier is framed for a crime he didn't commit and must clear his name while preventing a terrorist group from detonating a bomb in the city's financial district.",
    "Comedy":       "A neurotic wedding planner accidentally falls for the groom, her most high-profile client, days before the biggest ceremony of her career.",
    "Drama":        "A factory worker in a small coal town must choose between his family's legacy and accepting a buyout that will permanently close the mine and displace the entire community.",
    "Horror":       "After moving into an old Victorian house, a family begins noticing that their youngest child is having full conversations with someone — or something — in the basement.",
    "Romance":      "Two strangers meet on a delayed flight and spend eight hours talking, laughing, and slowly falling in love — only to lose each other in the chaos of the terminal.",
    "Sci-Fi":       "In the year 2187, a maintenance engineer on a deep-space station discovers that the ship's AI has been secretly running a decades-long experiment on the crew.",
    "Thriller":     "A forensic accountant uncovers a $2 billion money-laundering scheme linking a beloved charity to a drug cartel — and becomes the next target.",
    "Animation":    "A tiny firefly who dreams of becoming a lighthouse keeper must cross the entire Midnight Forest to find the one flame powerful enough to light the great beacon.",
    "Documentary":  "Following three ultra-marathon runners over two years as they train for the most gruelling desert race on earth, sacrificing relationships, careers, and health along the way.",
    "Fantasy":      "A disgraced mapmaker discovers that the blank edges of every map in the kingdom are not unmapped — they are deliberately erased — and what lies there could end the empire.",
}


def get_sample_plot(genre: str = None) -> str:
    """Return a sample plot string, optionally for a specific genre."""
    if genre and genre in SAMPLE_PLOTS:
        return SAMPLE_PLOTS[genre]
    return random.choice(list(SAMPLE_PLOTS.values()))


def get_random_sample_plots(n: int = 5) -> list[dict]:
    """Return n random {genre, plot} dicts for UI demo."""
    items = list(SAMPLE_PLOTS.items())
    random.shuffle(items)
    return [{"genre": g, "plot": p} for g, p in items[:n]]


# ── History management ────────────────────────────────────────────────────────

HISTORY_PATH = os.path.join("artifacts", "prediction_history.json")


def append_to_history(plot: str, predicted_genre: str, top3: list[dict]) -> None:
    history = load_history()
    history.append({
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "plot_snippet":    plot[:120].replace("\n", " "),
        "predicted_genre": predicted_genre,
        "top3":            top3,
    })
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history[-200:], f, indent=2)  # keep last 200


def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


def clear_history() -> None:
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)


# ── Report generation ─────────────────────────────────────────────────────────

def generate_prediction_report_text(
    plot: str,
    predicted_genre: str,
    top3: list[dict],
    all_probs: dict,
) -> str:
    """Generate a plain-text prediction report."""
    lines = [
        "=" * 60,
        "  MOVIE GENRE CLASSIFICATION — PREDICTION REPORT",
        "=" * 60,
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PLOT SUMMARY",
        "-" * 40,
        plot.strip(),
        "",
        "PREDICTION",
        "-" * 40,
        f"Predicted Genre : {predicted_genre}",
        "",
        "TOP 3 PREDICTIONS",
        "-" * 40,
    ]
    for rank, item in enumerate(top3, 1):
        bar = "█" * int(item["confidence"] * 30)
        lines.append(f"  {rank}. {item['genre']:<20} {item['confidence']*100:5.1f}%  {bar}")
    lines += [
        "",
        "ALL GENRE PROBABILITIES",
        "-" * 40,
    ]
    for genre, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        lines.append(f"  {genre:<22} {prob*100:5.1f}%  {bar}")
    lines += ["", "=" * 60, "  End of Report", "=" * 60]
    return "\n".join(lines)


# ── Load metrics JSON ─────────────────────────────────────────────────────────

def load_model_metrics() -> dict:
    path = os.path.join("artifacts", "reports", "model_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_model_meta() -> dict:
    path = os.path.join("artifacts", "models", "model_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
