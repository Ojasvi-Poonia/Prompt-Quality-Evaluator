"""Shared helpers for the Prompt Quality Evaluator.

Provides NLP model loading, text preprocessing, and common utility functions
used across all three metric pillars.
"""

import re
import math
from typing import List, Tuple

import spacy
import nltk


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_nlp = None
_stopwords = None
_common_words = None


def get_nlp():
    """Return a cached spaCy English model (en_core_web_sm).

    Downloads the model automatically if it is not installed.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_stopwords() -> set:
    """Return NLTK English stopwords, downloading if necessary."""
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords = set(nltk.corpus.stopwords.words("english"))
    return _stopwords


def ensure_punkt():
    """Ensure the NLTK punkt tokenizer data is available."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Common-word frequency list (top ~5000)
# ---------------------------------------------------------------------------

def get_common_words() -> set:
    """Return a set of the ~5000 most common English words.

    Built from NLTK's Brown corpus frequency distribution.  Falls back to a
    small manually-curated set if the corpus is unavailable.
    """
    global _common_words
    if _common_words is not None:
        return _common_words

    try:
        from nltk.corpus import brown
        brown.words()
    except LookupError:
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown

    freq = nltk.FreqDist(w.lower() for w in brown.words())
    _common_words = {word for word, _ in freq.most_common(5000)}
    return _common_words


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using spaCy's sentence boundary detector.

    Returns a list of non-empty, stripped sentence strings.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def word_count(text: str) -> int:
    """Count whitespace-delimited words in *text*."""
    return len(text.split())


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the [lo, hi] interval."""
    return max(lo, min(hi, value))


def gaussian_score(value: float, mean: float, std: float) -> float:
    """Return a Gaussian-shaped score peaking at *mean* with spread *std*.

    Output is in [0, 1] — exactly 1.0 at the mean, decaying towards 0.
    """
    return math.exp(-0.5 * ((value - mean) / std) ** 2)


def is_code_like(token: str) -> bool:
    """Heuristic check for code-like tokens.

    Matches camelCase, snake_case, file extensions, dotted paths, etc.
    """
    patterns = [
        r"[a-z]+[A-Z]",           # camelCase
        r"[a-z]+_[a-z]+",         # snake_case
        r"\w+\.\w{1,5}",          # file.ext or module.attr
        r"[A-Z]{2,}",             # CONSTANT
        r"\w+\(\)",               # func()
        r"<\w+>",                 # <tag>
        r"\{[\w:]+\}",            # {placeholder}
    ]
    return any(re.search(p, token) for p in patterns)
