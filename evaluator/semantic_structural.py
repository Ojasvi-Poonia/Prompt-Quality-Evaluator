"""Pillar 2: Semantic-Structural Metrics.

Six metrics that capture semantic relationships and textual organisation
using TF-IDF (scikit-learn) and spaCy.  No transformer models.

Each public function returns ``(score: float, findings: list[str])``.
"""

import math
from typing import Tuple, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evaluator.utils import (
    get_nlp,
    get_stopwords,
    split_sentences,
    word_count,
    clamp,
    ensure_punkt,
)


# ===================================================================
# 1. Coherence Score (TF-IDF cosine similarity)
# ===================================================================

def coherence_score(text: str) -> Tuple[float, List[str]]:
    """Measure semantic coherence via consecutive-sentence TF-IDF similarity.

    Splits the prompt into sentences, builds TF-IDF vectors, then
    averages the cosine similarity between each consecutive pair.

    Range: 0-1.  Higher = more coherent flow between sentences.
    Single-sentence prompts return 0.5 (neutral).
    """
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence — coherence is neutral.")
        return 0.5, findings

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        findings.append("Could not build TF-IDF vectors (vocabulary too small).")
        return 0.3, findings

    sims = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf[i], tfidf[i + 1])[0][0]
        sims.append(float(sim))

    avg_sim = float(np.mean(sims)) if sims else 0.0
    score = clamp(avg_sim * 1.5)  # scale up; raw cosine is usually < 0.7

    if avg_sim > 0.4:
        findings.append(f"Good sentence-to-sentence coherence (avg similarity {avg_sim:.2f}).")
    elif avg_sim > 0.15:
        findings.append(f"Moderate coherence ({avg_sim:.2f}). Some sentences may be loosely connected.")
    else:
        findings.append(f"Low coherence ({avg_sim:.2f}). Sentences seem disconnected.")

    return score, findings


# ===================================================================
# 2. Semantic Density
# ===================================================================

def semantic_density(text: str) -> Tuple[float, List[str]]:
    """Ratio of unique meaningful lemmas to total tokens.

    Meaningful POS: nouns, verbs, adjectives, adverbs.
    High ratio = dense, information-rich.  Low ratio = filler-heavy.

    Range: 0-1.
    """
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []

    total_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
    if total_tokens == 0:
        return 0.0, ["No tokens to analyse."]

    meaningful_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    meaningful_lemmas = {
        t.lemma_.lower() for t in doc
        if t.pos_ in meaningful_pos and not t.is_stop
    }

    ratio = len(meaningful_lemmas) / total_tokens
    score = clamp(ratio * 2.5)  # typical ratio is 0.2-0.5

    if ratio > 0.35:
        findings.append(f"High semantic density ({ratio:.2f}). Information-rich text.")
    elif ratio > 0.2:
        findings.append(f"Moderate semantic density ({ratio:.2f}).")
    else:
        findings.append(f"Low semantic density ({ratio:.2f}). Text may be filler-heavy.")

    return score, findings


# ===================================================================
# 3. Topic Focus (TF-IDF centroid distance)
# ===================================================================

def topic_focus(text: str) -> Tuple[float, List[str]]:
    """Measure how focused the prompt is on a single topic.

    Builds TF-IDF vectors for all sentences, computes a centroid, then
    measures average cosine distance from the centroid.  Low distance =
    focused.  High distance = scattered.

    Range: 0-1.  1 = perfectly focused.
    """
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence — topic focus is maximal by default.")
        return 0.9, findings

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return 0.5, ["Could not compute topic focus (vocabulary too small)."]

    centroid = tfidf.mean(axis=0)
    centroid = np.asarray(centroid)

    distances = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray()
        sim = cosine_similarity(row, centroid)[0][0]
        distances.append(1.0 - sim)

    avg_dist = float(np.mean(distances))
    score = clamp(1.0 - avg_dist * 2.0)

    if avg_dist < 0.3:
        findings.append(f"Well-focused prompt (avg topic distance {avg_dist:.2f}).")
    elif avg_dist < 0.6:
        findings.append(f"Moderate focus ({avg_dist:.2f}). Some topic drift detected.")
    else:
        findings.append(f"Scattered prompt ({avg_dist:.2f}). Multiple unrelated topics.")

    return score, findings


# ===================================================================
# 4. Lexical Sophistication
# ===================================================================

def lexical_sophistication(text: str) -> Tuple[float, List[str]]:
    """Score vocabulary sophistication via word frequency rank.

    Uses NLTK Brown corpus frequencies.  Prompts using more precise,
    less common words score higher.

    Range: 0-1.  Everyday language ~0.4, technical ~0.8-1.0.
    """
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []

    # Get frequency distribution from Brown corpus
    try:
        from nltk.corpus import brown
        brown.words()
    except LookupError:
        import nltk
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown

    import nltk as _nltk
    freq_dist = _nltk.FreqDist(w.lower() for w in brown.words())
    total_words_in_corpus = freq_dist.N()

    words = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 2 and not t.is_stop]
    if not words:
        return 0.3, ["No substantive words found."]

    # For each word, compute a rarity score
    rarity_scores = []
    for w in words:
        freq = freq_dist.get(w, 0)
        if freq == 0:
            rarity_scores.append(1.0)  # unknown = rare/technical
        else:
            # Normalise: common words get low rarity
            norm_freq = freq / total_words_in_corpus
            rarity = 1.0 - min(norm_freq * 5000, 1.0)
            rarity_scores.append(max(rarity, 0.0))

    avg_rarity = float(np.mean(rarity_scores))
    # Scale: 0.3 baseline, up to 1.0
    score = clamp(0.3 + avg_rarity * 0.7)

    if avg_rarity > 0.7:
        findings.append("Highly sophisticated vocabulary.")
    elif avg_rarity > 0.4:
        findings.append("Moderately sophisticated vocabulary.")
    else:
        findings.append("Simple, everyday vocabulary. Consider using more precise terms.")

    return score, findings


# ===================================================================
# 5. Sentence Flow
# ===================================================================

def sentence_flow(text: str) -> Tuple[float, List[str]]:
    """Score sentence length variance for natural flow.

    Very uniform lengths suggest mechanical writing.  Very high variance
    suggests disorganised writing.  Moderate variance is ideal.

    Range: 0-1.
    """
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence — flow analysis not applicable.")
        return 0.5, findings

    lengths = [word_count(s) for s in sentences]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    if mean_len == 0:
        return 0.3, ["Sentences are empty."]

    cv = std_len / mean_len  # coefficient of variation

    # Ideal CV is around 0.3-0.6
    from evaluator.utils import gaussian_score
    score = gaussian_score(cv, mean=0.45, std=0.3)

    if cv < 0.15:
        findings.append(f"Very uniform sentence lengths (CV={cv:.2f}). Feels mechanical.")
    elif cv < 0.3:
        findings.append(f"Slightly uniform sentence flow (CV={cv:.2f}).")
    elif cv <= 0.6:
        findings.append(f"Good sentence flow variation (CV={cv:.2f}).")
    else:
        findings.append(f"High sentence length variance (CV={cv:.2f}). May feel disorganised.")

    return clamp(score), findings


# ===================================================================
# 6. Reference Resolution
# ===================================================================

def reference_resolution(text: str) -> Tuple[float, List[str]]:
    """Score pronoun-antecedent clarity.

    Counts pronouns without clear antecedents using spaCy dependency
    parsing.  Ambiguous references reduce the score.

    Range: 0-1.  1 = all references are clear.
    """
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []

    ambiguous_pronouns = {"it", "this", "that", "they", "these", "those", "them"}
    sentences = list(doc.sents)

    if not sentences:
        return 0.5, ["No sentences to analyse."]

    total_pronouns = 0
    ambiguous_count = 0

    for i, sent in enumerate(sentences):
        for token in sent:
            if token.text.lower() in ambiguous_pronouns and token.dep_ in ("nsubj", "dobj", "nsubjpass", "pobj"):
                total_pronouns += 1

                # Check if there is a noun in this or the previous sentence
                has_antecedent = False
                search_range = list(sent)
                if i > 0:
                    search_range = list(sentences[i - 1]) + search_range

                for candidate in search_range:
                    if candidate.pos_ in ("NOUN", "PROPN") and candidate != token:
                        has_antecedent = True
                        break

                if not has_antecedent:
                    ambiguous_count += 1

    if total_pronouns == 0:
        findings.append("No ambiguous pronouns found.")
        return 0.9, findings

    clarity_ratio = 1.0 - (ambiguous_count / total_pronouns)
    score = clamp(clarity_ratio)

    if ambiguous_count == 0:
        findings.append("All pronoun references are clear.")
    elif ambiguous_count <= 2:
        findings.append(f"{ambiguous_count} potentially ambiguous pronoun(s).")
    else:
        findings.append(f"{ambiguous_count} ambiguous pronouns. Replace with explicit nouns.")

    return score, findings


# ===================================================================
# Aggregate helper
# ===================================================================

def compute_all(text: str) -> dict:
    """Run all Pillar-2 metrics and return a dict of results."""
    metrics = {
        "coherence": coherence_score,
        "semantic_density": semantic_density,
        "topic_focus": topic_focus,
        "lexical_sophistication": lexical_sophistication,
        "sentence_flow": sentence_flow,
        "reference_resolution": reference_resolution,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
