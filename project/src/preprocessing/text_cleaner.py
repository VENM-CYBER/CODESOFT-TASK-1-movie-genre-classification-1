"""
src/preprocessing/text_cleaner.py
----------------------------------
Text preprocessing pipeline for the Movie Genre Classification System.
Supports a 'fast' mode (no lemmatization) suitable for 50k+ row datasets.
"""

import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _download_nltk():
    resources = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
        ("corpora/omw-1.4",      "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_download_nltk()

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def clean_text(text: str, lemmatize: bool = False) -> str:
    """
    Full NLP preprocessing pipeline.

    Parameters
    ----------
    text     : raw plot string
    lemmatize: apply lemmatization (slower; default False for large datasets)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(series, lemmatize: bool = False):
    """
    Apply clean_text to a pandas Series with a progress bar.
    lemmatize defaults to False for performance on large datasets.
    """
    from tqdm import tqdm
    tqdm.pandas(desc="Cleaning text")
    return series.progress_apply(lambda t: clean_text(t, lemmatize=lemmatize))
