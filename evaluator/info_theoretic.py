import math
from typing import Tuple, List
from collections import Counter

import numpy as np
import tiktoken

from evaluator.utils import clamp


_encoding = None

MIN_TOKENS_FOR_RELIABLE_SCORE = 30
SHORT_TEXT_CAP = 0.3


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _tokenise(text: str) -> List[int]:
    return _get_encoding().encode(text)


def _apply_short_text_cap(score: float, tokens: List[int], findings: List[str]) -> float:
    if len(tokens) < MIN_TOKENS_FOR_RELIABLE_SCORE:
        capped = min(score, SHORT_TEXT_CAP)
        if capped < score:
            findings.append(
                f"Short prompt ({len(tokens)} tokens) -- score capped at {SHORT_TEXT_CAP} "
                f"(information-theoretic metrics need >= {MIN_TOKENS_FOR_RELIABLE_SCORE} tokens)."
            )
        return capped
    return score


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

    score = _apply_short_text_cap(score, tokens, findings)
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

    score = _apply_short_text_cap(score, tokens, findings)
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

    score = _apply_short_text_cap(score, tokens, findings)
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

    mtld_norm: float = basic_ttr
    hdd_norm: float = basic_ttr
    mattr_norm: float = basic_ttr
    advanced_used = False

    try:
        from lexicalrichness import LexicalRichness
        lex = LexicalRichness(text)
        if lex.words >= 5:
            advanced_used = True
            try:
                mtld_raw = lex.mtld(threshold=0.72)
                mtld_norm = clamp(mtld_raw / 100.0)
            except Exception:
                pass
            try:
                hdd_norm = clamp(lex.hdd(draws=min(42, lex.words - 1)))
            except Exception:
                pass
            try:
                mattr_norm = clamp(lex.mattr(window_size=min(10, lex.words)))
            except Exception:
                pass
    except Exception:
        pass

    norm_root = clamp(root_ttr / 15.0)
    norm_log = clamp(log_ttr)

    if advanced_used:
        avg = (basic_ttr + mtld_norm + hdd_norm + mattr_norm) / 4.0
        findings.append(
            f"Lexical diversity -- TTR: {basic_ttr:.2f}, "
            f"MTLD: {mtld_norm:.2f}, HD-D: {hdd_norm:.2f}, MATTR: {mattr_norm:.2f}"
        )
    else:
        avg = (basic_ttr + norm_root + norm_log) / 3.0
        findings.append(
            f"TTR variants -- basic: {basic_ttr:.2f}, root: {root_ttr:.1f}, log: {log_ttr:.2f}"
        )

    score = clamp(avg)

    if score > 0.6:
        findings.append("Rich vocabulary.")
    elif score > 0.35:
        findings.append("Moderate vocabulary richness.")
    else:
        findings.append("Limited vocabulary. Vary word choice.")

    score = _apply_short_text_cap(score, tokens, findings)
    return score, findings


def ngram_repetition_penalty(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 4:
        return 0.3, ["Text too short for n-gram analysis."]

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
        return 0.3, ["No n-grams to analyse."]

    ratio = total_repeats / total_ngrams
    score = clamp(1.0 - ratio * 3.0)

    if ratio < 0.05:
        findings.append("Minimal n-gram repetition.")
    elif ratio < 0.15:
        findings.append(f"Some repeated phrases ({total_repeats} repeated n-grams).")
    else:
        findings.append(f"High phrase repetition ({total_repeats} repeated n-grams). Rephrase.")

    score = _apply_short_text_cap(score, tokens, findings)
    return score, findings


def burstiness(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenise(text)
    findings: List[str] = []

    if len(tokens) < 10:
        return 0.3, ["Text too short for burstiness analysis."]

    chunk_size = max(len(tokens) // 5, 3)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    chunks = [c for c in chunks if len(c) >= 3]

    if len(chunks) < 2:
        return 0.3, ["Not enough chunks for burstiness analysis."]

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

    score = _apply_short_text_cap(score, tokens, findings)
    return score, findings


def sentence_repetition_penalty(text: str) -> Tuple[float, List[str]]:
    from evaluator.utils import split_sentences, NEUTRAL_SCORE
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) < 2:
        return NEUTRAL_SCORE, ["Single sentence -- repetition analysis neutral."]

    repeats = 0
    pairs_checked = 0

    for i in range(len(sentences)):
        words_i = set(w.lower().strip(".,!?;:") for w in sentences[i].split() if len(w) > 2)
        if not words_i:
            continue
        for j in range(i + 1, len(sentences)):
            words_j = set(w.lower().strip(".,!?;:") for w in sentences[j].split() if len(w) > 2)
            if not words_j:
                continue
            pairs_checked += 1
            smaller = min(len(words_i), len(words_j))
            if smaller == 0:
                continue
            overlap = len(words_i & words_j) / smaller
            if overlap > 0.7:
                repeats += 1

    if pairs_checked == 0:
        return NEUTRAL_SCORE, ["No sentence pairs to compare (neutral)."]

    repeat_ratio = repeats / pairs_checked
    score = clamp(1.0 - repeat_ratio * 2.0)

    if repeat_ratio > 0.3:
        findings.append(f"High sentence repetition detected ({repeats} overlapping pairs). "
                        f"Avoid restating the same idea.")
    elif repeat_ratio > 0.1:
        findings.append(f"Some sentences overlap significantly ({repeats} pairs).")
    else:
        findings.append("Sentences are distinct and non-repetitive.")

    return score, findings


def compute_all(text: str) -> dict:
    metrics = {
        "shannon_entropy": shannon_entropy,
        "redundancy": redundancy_score,
        "information_density": information_density,
        "vocabulary_richness": vocabulary_richness,
        "ngram_repetition": ngram_repetition_penalty,
        "burstiness": burstiness,
        "sentence_repetition": sentence_repetition_penalty,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
