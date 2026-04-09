import math
from typing import Tuple, List
from collections import Counter

import numpy as np
import tiktoken

from evaluator.utils import clamp


_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _tokenise(text: str) -> List[int]:
    return _get_encoding().encode(text)


def shannon_entropy(text: str) -> Tuple[float, List[str]]:
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


def redundancy_score(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenise(text)
    findings: List[str] = []

    if not tokens:
        return 0.0, ["No tokens."]

    unique = len(set(tokens))
    total = len(tokens)
    redundancy = 1.0 - (unique / total)

    score = 1.0 - redundancy
    if redundancy > 0.6:
        score *= 0.5

    score = clamp(score)

    if redundancy < 0.3:
        findings.append(f"Low redundancy ({redundancy:.2f}). Diverse token usage.")
    elif redundancy < 0.6:
        findings.append(f"Moderate redundancy ({redundancy:.2f}).")
    else:
        findings.append(f"High redundancy ({redundancy:.2f}). Prompt is very repetitive.")

    return score, findings


def information_density(text: str) -> Tuple[float, List[str]]:
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
    score = clamp(density * 10.0)

    if density > 0.08:
        findings.append(f"High information density ({density:.4f} bits/token).")
    elif density > 0.03:
        findings.append(f"Moderate information density ({density:.4f} bits/token).")
    else:
        findings.append(f"Low information density ({density:.4f} bits/token).")

    return score, findings


def vocabulary_richness(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenise(text)
    findings: List[str] = []

    total = len(tokens)
    if total < 2:
        return 0.0, ["Text too short for vocabulary analysis."]

    unique = len(set(tokens))

    basic_ttr = unique / total
    root_ttr = unique / math.sqrt(total)
    log_ttr = math.log(unique) / math.log(total) if total > 1 else 0.0

    norm_root = clamp(root_ttr / 15.0)
    norm_log = clamp(log_ttr)

    avg = (basic_ttr + norm_root + norm_log) / 3.0
    score = clamp(avg)

    findings.append(
        f"TTR variants -- basic: {basic_ttr:.2f}, root: {root_ttr:.1f}, log: {log_ttr:.2f}"
    )

    if score > 0.6:
        findings.append("Rich vocabulary.")
    elif score > 0.35:
        findings.append("Moderate vocabulary richness.")
    else:
        findings.append("Limited vocabulary. Vary word choice.")

    return score, findings


def ngram_repetition_penalty(text: str) -> Tuple[float, List[str]]:
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
    score = clamp(1.0 - ratio * 3.0)

    if ratio < 0.05:
        findings.append("Minimal n-gram repetition.")
    elif ratio < 0.15:
        findings.append(f"Some repeated phrases ({total_repeats} repeated n-grams).")
    else:
        findings.append(f"High phrase repetition ({total_repeats} repeated n-grams). Rephrase.")

    return score, findings


def burstiness(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 10:
        return 0.5, ["Text too short for burstiness analysis."]

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

    score = clamp(1.0 - variance * 2.0)

    if variance < 0.2:
        findings.append(f"Consistent information flow (entropy variance {variance:.3f}).")
    elif variance < 0.5:
        findings.append(f"Somewhat uneven information flow ({variance:.3f}).")
    else:
        findings.append(f"Bursty information distribution ({variance:.3f}). Redistribute content.")

    return score, findings


def compute_all(text: str) -> dict:
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
