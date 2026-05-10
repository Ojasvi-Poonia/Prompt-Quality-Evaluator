import re
import math
from typing import List, Tuple

import spacy
import nltk


_nlp = None
_stopwords = None
_common_words = None
_modern_tech_vocab = None


MODERN_TECH_VOCAB = {
    "api", "apis", "endpoint", "endpoints", "rest", "restful", "graphql",
    "json", "yaml", "xml", "csv", "sql", "nosql", "database", "db",
    "backend", "frontend", "fullstack", "middleware", "microservice",
    "microservices", "serverless", "kubernetes", "docker", "container",
    "containers", "deploy", "deployment", "ci", "cd", "pipeline",
    "webhook", "webhooks", "oauth", "jwt", "token", "tokens", "auth",
    "authentication", "authorization", "cors", "csrf", "xss",
    "http", "https", "tcp", "udp", "websocket", "websockets",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt",
    "django", "flask", "fastapi", "express", "spring", "rails",
    "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "lambda", "s3", "ec2", "rds", "dynamodb",
    "python", "javascript", "typescript", "rust", "golang", "java",
    "kotlin", "swift", "scala", "elixir", "ruby", "php",
    "git", "github", "gitlab", "bitbucket", "devops", "sre",
    "async", "await", "promise", "callback", "closure", "lambda",
    "regex", "regexp", "tokenize", "tokenization", "embedding",
    "embeddings", "llm", "llms", "gpt", "claude", "transformer",
    "transformers", "nlp", "ml", "ai", "dl", "tensor", "tensors",
    "cnn", "rnn", "lstm", "gan", "pytorch", "tensorflow",
    "numpy", "pandas", "sklearn", "scikit", "matplotlib", "jupyter",
    "ssr", "csr", "spa", "pwa", "seo", "cdn", "dns", "ssl", "tls",
    "saas", "paas", "iaas", "crud", "orm", "mvc", "dto", "dao",
    "sdk", "cli", "gui", "ui", "ux", "ide", "repl", "ast", "dom",
    "iot", "ar", "vr", "xr", "3d", "cli", "gpu", "cpu", "ram",
    "yaml", "toml", "ini", "env", "dotenv", "config", "webpack",
    "vite", "babel", "eslint", "prettier", "npm", "yarn", "pnpm",
    "pytest", "jest", "mocha", "cypress", "playwright", "selenium",
}


_PREFERRED_SPACY_MODEL = "en_core_web_md"
_FALLBACK_SPACY_MODEL = "en_core_web_sm"


def get_nlp():
    global _nlp
    if _nlp is None:
        for model_name in (_PREFERRED_SPACY_MODEL, _FALLBACK_SPACY_MODEL):
            try:
                _nlp = spacy.load(model_name)
                return _nlp
            except OSError:
                continue
        from spacy.cli import download
        download(_PREFERRED_SPACY_MODEL)
        _nlp = spacy.load(_PREFERRED_SPACY_MODEL)
    return _nlp


_spell_checker = None


def _get_spellchecker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker()
    return _spell_checker


def word_zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word.lower(), "en")


KNOWN_WORD_ZIPF_THRESHOLD = 1.0


def is_known_word(word: str) -> bool:
    if not word or not word.isalpha() or len(word) < 2:
        return True
    w = word.lower()
    if w in MODERN_TECH_VOCAB:
        return True
    if word_zipf(w) >= KNOWN_WORD_ZIPF_THRESHOLD:
        return True
    try:
        nlp = get_nlp()
        if nlp.vocab[w].has_vector:
            return True
    except Exception:
        pass
    return False


def is_typo(word: str) -> bool:
    if not word or not word.isalpha() or len(word) < 3:
        return False
    return not is_known_word(word)


def suggest_correction(word: str) -> str:
    try:
        spell = _get_spellchecker()
        suggestion = spell.correction(word.lower())
        return suggestion if suggestion and suggestion != word.lower() else ""
    except Exception:
        return ""


def detect_typos(text: str) -> List[Tuple[str, str]]:
    nlp = get_nlp()
    doc = nlp(text)
    typos: List[Tuple[str, str]] = []
    seen = set()
    for tok in doc:
        if not tok.is_alpha or len(tok.text) < 3:
            continue
        lower = tok.text.lower()
        if lower in seen:
            continue
        seen.add(lower)
        if is_typo(tok.text):
            typos.append((tok.text, suggest_correction(tok.text)))
    return typos


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
    _common_words |= MODERN_TECH_VOCAB
    return _common_words


def get_modern_tech_vocab() -> set:
    return MODERN_TECH_VOCAB


def is_likely_english(text: str, threshold: float = 0.6) -> Tuple[bool, float]:
    nlp = get_nlp()
    doc = nlp(text)

    tokens = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 1]
    if not tokens:
        return True, 1.0

    english_count = 0
    for t in tokens:
        if nlp.vocab[t].is_stop:
            english_count += 1
            continue
        if is_known_word(t):
            english_count += 1

    ratio = english_count / len(tokens)
    return ratio >= threshold, ratio


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


MIN_WORDS_FOR_RELIABLE = 30
MIN_WORDS_FOR_PARTIAL = 10
NEUTRAL_SCORE = 0.5
SHORT_TEXT_FLOOR = 0.3


def short_text_factor(text: str) -> float:
    wc = word_count(text)
    if wc >= MIN_WORDS_FOR_RELIABLE:
        return 1.0
    if wc <= MIN_WORDS_FOR_PARTIAL:
        return 0.3
    span = MIN_WORDS_FOR_RELIABLE - MIN_WORDS_FOR_PARTIAL
    progress = (wc - MIN_WORDS_FOR_PARTIAL) / span
    return 0.3 + 0.7 * progress


def neutralise_for_short_text(score: float, text: str, neutral: float = NEUTRAL_SCORE) -> float:
    factor = short_text_factor(text)
    if factor >= 1.0:
        return score
    return neutral + (score - neutral) * factor


def strip_code_blocks(text: str) -> str:
    fenced = re.sub(r"```[\s\S]*?```", " ", text)
    inline = re.sub(r"`[^`\n]+`", " ", fenced)
    indented = re.sub(r"(?m)^(?: {4,}|\t).+$", " ", inline)
    return re.sub(r"\s+", " ", indented).strip()


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
