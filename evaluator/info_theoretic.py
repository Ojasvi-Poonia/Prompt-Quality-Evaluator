"""Pillar 3: Information-Theoretic Metrics.

Six metrics rooted in information theory — entropy, redundancy,
density, vocabulary richness, n-gram repetition, and burstiness.

Uses ``tiktoken`` (cl100k_base) for tokenisation and numpy/scipy
for computation.  No transformer models.

Each public function returns ``(score: float, findings: list[str])``.
"""

import math
from typing import Tuple, List
from collections import Counter

import numpy as np
import tiktoken

from evaluator.utils import clamp


# ---------------------------------------------------------------------------
# Tokeniser singleton
# ---------------------------------------------------------------------------

_encoding = None


def _get_encoding():
    """Return a cached tiktoken cl100k_base encoding."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _tokenise(text: str) -> List[int]:
    """Tokenise *text* into a list of integer token IDs."""
    return _get_encoding().encode(text)


# ===================================================================
# 1. Shannon Entropy
# ===================================================================

def shannon_entropy(text: str) -> Tuple[float, List[str]]:
    """Compute normalised Shannon entropy of the token distribution.

    H = -sum(p(x) * log2(p(x)))  normalised by log2(vocab_size).
    Higher entropy = more diverse token usage.

    Range: 0-1.
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 2:
        return 0.0, ["Text too short for entropy calculation."]

    freq = Counter(tokens)
    total = len(tokens)
    vocab_size = len(freq)

    entropy = -sum(
        (c / total) * math.log2(c / total) for c in freq.values()
    )

    max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0

    score = clamp(normalised)

    if normalised > 0.8:
        findings.append(f"High token diversity (normalised entropy {normalised:.2f}).")
    elif normalised > 0.5:
        findings.append(f"Moderate token diversity ({normalised:.2f}).")
    else:
        findings.append(f"Low token diversity ({normalised:.2f}). Repetitive language.")

    return score, findings


# ===================================================================
# 2. Redundancy Score
# ===================================================================

def redundancy_score(text: str) -> Tuple[float, List[str]]:
    """Score based on token-level redundancy.

    redundancy = 1 - (unique_tokens / total_tokens).
    Returns ``1 - redundancy`` so low repetition scores high.
    Penalises heavily if redundancy > 0.6.

    Range: 0-1.
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    if not tokens:
        return 0.0, ["No tokens."]

    unique = len(set(tokens))
    total = len(tokens)
    redundancy = 1.0 - (unique / total)

    score = 1.0 - redundancy
    if redundancy > 0.6:
        score *= 0.5  # heavy penalty

    score = clamp(score)

    if redundancy < 0.3:
        findings.append(f"Low redundancy ({redundancy:.2f}). Diverse token usage.")
    elif redundancy < 0.6:
        findings.append(f"Moderate redundancy ({redundancy:.2f}).")
    else:
        findings.append(f"High redundancy ({redundancy:.2f}). Prompt is very repetitive.")

    return score, findings


# ===================================================================
# 3. Information Density
# ===================================================================

def information_density(text: str) -> Tuple[float, List[str]]:
    """Entropy per token — how much information each token carries.

    Range: 0-1 (normalised).
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 2:
        return 0.0, ["Text too short for density analysis."]

    freq = Counter(tokens)
    total = len(tokens)

    entropy = -sum(
        (c / total) * math.log2(c / total) for c in freq.values()
    )

    density = entropy / total if total > 0 else 0.0
    # Typical values are 0.01-0.15; normalise
    score = clamp(density * 10.0)

    if density > 0.08:
        findings.append(f"High information density ({density:.4f} bits/token).")
    elif density > 0.03:
        findings.append(f"Moderate information density ({density:.4f} bits/token).")
    else:
        findings.append(f"Low information density ({density:.4f} bits/token).")

    return score, findings


# ===================================================================
# 4. Vocabulary Richness (TTR variants)
# ===================================================================

def vocabulary_richness(text: str) -> Tuple[float, List[str]]:
    """Compute three Type-Token Ratio variants and average them.

    - Basic TTR: unique / total
    - Root TTR: unique / sqrt(total)
    - Log TTR: log(unique) / log(total)

    Averaging handles length sensitivity of basic TTR.
    Range: 0-1.
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    total = len(tokens)
    if total < 2:
        return 0.0, ["Text too short for vocabulary analysis."]

    unique = len(set(tokens))

    basic_ttr = unique / total
    root_ttr = unique / math.sqrt(total)
    log_ttr = math.log(unique) / math.log(total) if total > 1 else 0.0

    # Normalise root_ttr (typically 3-20) and log_ttr (typically 0.7-1.0)
    norm_root = clamp(root_ttr / 15.0)
    norm_log = clamp(log_ttr)

    avg = (basic_ttr + norm_root + norm_log) / 3.0
    score = clamp(avg)

    findings.append(
        f"TTR variants — basic: {basic_ttr:.2f}, root: {root_ttr:.1f}, log: {log_ttr:.2f}"
    )

    if score > 0.6:
        findings.append("Rich vocabulary.")
    elif score > 0.35:
        findings.append("Moderate vocabulary richness.")
    else:
        findings.append("Limited vocabulary. Vary word choice.")

    return score, findings


# ===================================================================
# 5. N-gram Repetition Penalty
# ===================================================================

def ngram_repetition_penalty(text: str) -> Tuple[float, List[str]]:
    """Penalise repeated bigrams and trigrams.

    Score = 1 - (repeated_ngrams / total_ngrams).
    Range: 0-1.  1 = no repeated n-grams.
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 4:
        return 0.8, ["Text too short for n-gram analysis."]

    def count_repeats(n: int) -> Tuple[int, int]:
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        freq = Counter(ngrams)
        repeated = sum(1 for c in freq.values() if c > 1)
        return repeated, len(ngrams)

    rep_bi, total_bi = count_repeats(2)
    rep_tri, total_tri = count_repeats(3)

    total_ngrams = total_bi + total_tri
    total_repeats = rep_bi + rep_tri

    if total_ngrams == 0:
        return 0.8, ["No n-grams to analyse."]

    ratio = total_repeats / total_ngrams
    score = clamp(1.0 - ratio * 3.0)  # amplify penalty

    if ratio < 0.05:
        findings.append("Minimal n-gram repetition.")
    elif ratio < 0.15:
        findings.append(f"Some repeated phrases ({total_repeats} repeated n-grams).")
    else:
        findings.append(f"High phrase repetition ({total_repeats} repeated n-grams). Rephrase.")

    return score, findings


# ===================================================================
# 6. Burstiness
# ===================================================================

def burstiness(text: str) -> Tuple[float, List[str]]:
    """Measure uniformity of information flow across chunks.

    Splits tokens into chunks, computes per-chunk entropy, then
    measures variance.  Low variance = consistent flow (scores high).

    Range: 0-1.
    """
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 10:
        return 0.5, ["Text too short for burstiness analysis."]

    # Split into ~5 chunks
    chunk_size = max(len(tokens) // 5, 3)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    chunks = [c for c in chunks if len(c) >= 3]

    if len(chunks) < 2:
        return 0.5, ["Not enough chunks for burstiness analysis."]

    def chunk_entropy(chunk: List[int]) -> float:
        freq = Counter(chunk)
        total = len(chunk)
        return -sum(
            (c / total) * math.log2(c / total) for c in freq.values()
        )

    entropies = [chunk_entropy(c) for c in chunks]
    variance = float(np.var(entropies))

    # Low variance = consistent = good; normalise
    score = clamp(1.0 - variance * 2.0)

    if variance < 0.2:
        findings.append(f"Consistent information flow (entropy variance {variance:.3f}).")
    elif variance < 0.5:
        findings.append(f"Somewhat uneven information flow ({variance:.3f}).")
    else:
        findings.append(f"Bursty information distribution ({variance:.3f}). Redistribute content.")

    return score, findings


# ===================================================================
# Aggregate helper
# ===================================================================

def compute_all(text: str) -> dict:
    """Run all Pillar-3 metrics and return a dict of results."""
    metrics = {
        "shannon_entropy": shannon_entropy,
        "redundancy": redundancy_score,
        "information_density": information_density,
        "vocabulary_richness": vocabulary_richness,
        "ngram_repetition": ngram_repetition_penalty,
        "burstiness": burstiness,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
