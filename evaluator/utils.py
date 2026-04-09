import re
import math
from typing import List, Tuple

import spacy
import nltk


_nlp = None
_stopwords = None
_common_words = None


def get_nlp():
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
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords = set(nltk.corpus.stopwords.words("english"))
    return _stopwords


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def get_common_words() -> set:
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


def split_sentences(text: str) -> List[str]:
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def gaussian_score(value: float, mean: float, std: float) -> float:
    return math.exp(-0.5 * ((value - mean) / std) ** 2)


def is_code_like(token: str) -> bool:
    patterns = [
        r"[a-z]+[A-Z]",
        r"[a-z]+_[a-z]+",
        r"\w+\.\w{1,5}",
        r"[A-Z]{2,}",
        r"\w+\(\)",
        r"<\w+>",
        r"\{[\w:]+\}",
    ]
    return any(re.search(p, token) for p in patterns)
