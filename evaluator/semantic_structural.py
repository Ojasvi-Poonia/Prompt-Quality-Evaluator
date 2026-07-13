import math
import re
from typing import Tuple, List, Dict

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
    is_likely_english,
    short_text_factor,
    neutralise_for_short_text,
    NEUTRAL_SCORE,
)


def _vector_similarity(doc_a, doc_b) -> float:
    if not (doc_a.has_vector and doc_b.has_vector):
        return -1.0
    if doc_a.vector_norm == 0 or doc_b.vector_norm == 0:
        return -1.0
    return float(doc_a.similarity(doc_b))


def coherence_score(text: str) -> Tuple[float, List[str]]:
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence -- coherence is neutral.")
        return 0.5, findings

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        findings.append("Could not build TF-IDF vectors (vocabulary too small).")
        return 0.3, findings

    tfidf_sims = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf[i], tfidf[i + 1])[0][0]
        tfidf_sims.append(float(sim))
    avg_tfidf_sim = float(np.mean(tfidf_sims)) if tfidf_sims else 0.0

    nlp = get_nlp()
    sent_docs = [nlp(s) for s in sentences]
    vec_sims: List[float] = []
    for i in range(len(sent_docs) - 1):
        sim = _vector_similarity(sent_docs[i], sent_docs[i + 1])
        if sim >= 0:
            vec_sims.append(sim)

    if vec_sims:
        avg_vec_sim = float(np.mean(vec_sims))
        avg_sim = 0.4 * avg_tfidf_sim + 0.6 * avg_vec_sim
        score = clamp(avg_sim * 1.3)
        findings.append(
            f"Coherence (TF-IDF: {avg_tfidf_sim:.2f}, vector: {avg_vec_sim:.2f}, blended: {avg_sim:.2f})"
        )
    else:
        avg_sim = avg_tfidf_sim
        score = clamp(avg_sim * 1.5)
        findings.append(f"Coherence (TF-IDF only: {avg_sim:.2f}; install spaCy md for vector blend)")

    if avg_sim > 0.5:
        findings.append("Strong sentence-to-sentence coherence.")
    elif avg_sim > 0.3:
        findings.append("Moderate coherence. Some sentences may be loosely connected.")
    elif avg_sim > 0.15:
        findings.append("Weak coherence. Many sentences feel disconnected.")
    else:
        findings.append("Very low coherence. Sentences appear unrelated.")

    return score, findings


def semantic_density(text: str) -> Tuple[float, List[str]]:
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
    score = clamp(ratio * 2.5)

    if ratio > 0.35:
        findings.append(f"High semantic density ({ratio:.2f}). Information-rich text.")
    elif ratio > 0.2:
        findings.append(f"Moderate semantic density ({ratio:.2f}).")
    else:
        findings.append(f"Low semantic density ({ratio:.2f}). Text may be filler-heavy.")

    capped = neutralise_for_short_text(score, text)
    if capped < score:
        findings.append(
            f"Short prompt ({word_count(text)} words) -- semantic density score "
            f"smoothed toward neutral (sample too small to be reliable)."
        )
    return capped, findings


def topic_focus(text: str) -> Tuple[float, List[str]]:
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence -- topic focus is neutral (cannot be measured).")
        return NEUTRAL_SCORE, findings

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return 0.5, ["Could not compute topic focus (vocabulary too small)."]

    centroid = tfidf.mean(axis=0)
    centroid = np.asarray(centroid)

    tfidf_distances = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray()
        sim = cosine_similarity(row, centroid)[0][0]
        tfidf_distances.append(1.0 - sim)
    avg_tfidf_dist = float(np.mean(tfidf_distances))

    nlp = get_nlp()
    sent_docs = [nlp(s) for s in sentences]
    valid_vectors = [d.vector for d in sent_docs if d.has_vector and d.vector_norm > 0]

    vec_distances: List[float] = []
    if len(valid_vectors) >= 2:
        vec_centroid = np.mean(valid_vectors, axis=0)
        norm = np.linalg.norm(vec_centroid)
        if norm > 0:
            vec_centroid_unit = vec_centroid / norm
            for v in valid_vectors:
                v_norm = np.linalg.norm(v)
                if v_norm > 0:
                    cos_sim = float(np.dot(v / v_norm, vec_centroid_unit))
                    vec_distances.append(1.0 - cos_sim)

    if vec_distances:
        avg_vec_dist = float(np.mean(vec_distances))
        avg_dist = 0.4 * avg_tfidf_dist + 0.6 * avg_vec_dist
        score = clamp(1.0 - avg_dist * 1.6)
        findings.append(
            f"Topic focus (TF-IDF dist: {avg_tfidf_dist:.2f}, vector dist: {avg_vec_dist:.2f})"
        )
    else:
        avg_dist = avg_tfidf_dist
        score = clamp(1.0 - avg_dist * 2.0)
        findings.append(f"Topic focus (TF-IDF only, dist: {avg_dist:.2f})")

    if avg_dist < 0.3:
        findings.append("Well-focused prompt.")
    elif avg_dist < 0.6:
        findings.append("Moderate focus -- some topic drift detected.")
    else:
        findings.append("Scattered prompt -- multiple unrelated topics.")

    return score, findings


def _zipf_to_rarity(zipf: float, in_tech_vocab: bool, has_vector: bool) -> float:
    if zipf >= 6.5:
        return 0.10
    if zipf >= 5.0:
        return 0.30
    if zipf >= 3.5:
        return 0.55
    if zipf >= 2.0:
        return 0.75
    if zipf >= 1.0:
        return 0.85
    if in_tech_vocab or has_vector:
        return 0.80
    return 0.0


def lexical_sophistication(text: str) -> Tuple[float, List[str]]:
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []

    is_english, english_ratio = is_likely_english(text, threshold=0.5)
    if not is_english:
        findings.append(
            f"Text appears non-English ({english_ratio:.0%} recognised). "
            "Lexical sophistication requires English vocabulary."
        )
        return clamp(0.3 * english_ratio), findings

    from evaluator.utils import get_modern_tech_vocab, word_zipf
    modern_tech = get_modern_tech_vocab()

    words = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 2 and not t.is_stop]
    if not words:
        return 0.3, ["No substantive words found."]

    rarity_scores = []
    typo_count = 0
    for w in words:
        zipf = word_zipf(w)
        in_tech = w in modern_tech
        has_vec = nlp.vocab[w].has_vector
        rarity = _zipf_to_rarity(zipf, in_tech, has_vec)
        if rarity == 0.0:
            typo_count += 1
            continue
        rarity_scores.append(rarity)

    if not rarity_scores:
        return 0.2, ["All substantive words appear to be typos or unknown."]

    avg_rarity = float(np.mean(rarity_scores))
    score = clamp(0.3 + avg_rarity * 0.7)

    if typo_count > 0:
        score = max(0.1, score - typo_count * 0.1)
        findings.append(
            f"Excluded {typo_count} likely typo(s) from sophistication calculation."
        )

    if avg_rarity > 0.7:
        findings.append("Highly sophisticated vocabulary.")
    elif avg_rarity > 0.4:
        findings.append("Moderately sophisticated vocabulary.")
    else:
        findings.append("Simple, everyday vocabulary. Consider using more precise terms.")

    return score, findings


def sentence_flow(text: str) -> Tuple[float, List[str]]:
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence -- flow analysis not applicable.")
        return 0.5, findings

    lengths = [word_count(s) for s in sentences]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    if mean_len == 0:
        return 0.3, ["Sentences are empty."]

    cv = std_len / mean_len

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


def reference_resolution(text: str) -> Tuple[float, List[str]]:
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
        wc = word_count(text)
        if wc < 10:
            findings.append("Too short to assess reference clarity (neutral).")
            return NEUTRAL_SCORE, findings
        findings.append("No ambiguous pronouns found.")
        return 0.85, findings

    clarity_ratio = 1.0 - (ambiguous_count / total_pronouns)
    score = clamp(clarity_ratio)

    if ambiguous_count == 0:
        findings.append("All pronoun references are clear.")
    elif ambiguous_count <= 2:
        findings.append(f"{ambiguous_count} potentially ambiguous pronoun(s).")
    else:
        findings.append(f"{ambiguous_count} ambiguous pronouns. Replace with explicit nouns.")

    return score, findings


_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            _vader_analyzer = False
    return _vader_analyzer


def tone_consistency(text: str) -> Tuple[float, List[str]]:
    sentences = split_sentences(text)
    findings: List[str] = []

    if len(sentences) <= 1:
        findings.append("Single sentence -- tone consistency not measurable.")
        return NEUTRAL_SCORE, findings

    vader = _get_vader()
    if not vader:
        return NEUTRAL_SCORE, ["VADER sentiment analyzer not installed -- tone analysis skipped."]

    compounds = [vader.polarity_scores(s)["compound"] for s in sentences]
    if not compounds:
        return NEUTRAL_SCORE, ["No sentiment data."]

    variance = float(np.var(compounds))
    score = clamp(1.0 - variance * 3.0)

    pos_count = sum(1 for c in compounds if c > 0.5)
    neg_count = sum(1 for c in compounds if c < -0.5)

    if pos_count > 0 and neg_count > 0:
        score = min(score, 0.5)
        findings.append(
            f"Mixed tone detected ({pos_count} strongly positive, {neg_count} strongly negative sentences). "
            "Consider unifying tone."
        )
    elif variance < 0.05:
        findings.append("Highly consistent tone across the prompt.")
    elif variance < 0.15:
        findings.append("Mostly consistent tone with some variation.")
    else:
        findings.append(f"Variable tone (variance {variance:.2f}). Some sentences feel out of register.")

    return score, findings


def _max_dep_depth(token) -> int:
    if not list(token.children):
        return 1
    return 1 + max(_max_dep_depth(c) for c in token.children)


def syntactic_complexity_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    nlp = get_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    if not sentences:
        return NEUTRAL_SCORE, ["No sentences to analyse."]

    depths = []
    clause_counts = []
    for sent in sentences:
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if not roots:
            continue
        depth = max(_max_dep_depth(r) for r in roots)
        depths.append(depth)
        clauses = sum(1 for t in sent if t.dep_ in {"ccomp", "advcl", "relcl", "xcomp"})
        clause_counts.append(clauses + 1)

    if not depths:
        return NEUTRAL_SCORE, ["Could not parse sentences."]

    avg_depth = float(np.mean(depths))
    avg_clauses = float(np.mean(clause_counts))

    from evaluator.utils import gaussian_score
    depth_score = gaussian_score(avg_depth, mean=5.5, std=2.5)
    clause_score = gaussian_score(avg_clauses, mean=2.0, std=1.5)

    score = 0.6 * depth_score + 0.4 * clause_score

    if avg_depth < 3:
        findings.append(f"Shallow syntactic structure (avg depth {avg_depth:.1f}). Add subordinate clauses.")
    elif avg_depth > 9:
        findings.append(f"Very deep syntactic structure (avg depth {avg_depth:.1f}). Simplify.")
    else:
        findings.append(f"Balanced syntactic complexity (avg depth {avg_depth:.1f}, avg clauses/sent {avg_clauses:.1f}).")

    return clamp(score), findings


_DISCOURSE_MARKERS = {
    "addition": ["additionally", "furthermore", "moreover", "also", "besides", "in addition"],
    "contrast": ["however", "but", "although", "though", "nevertheless", "on the other hand", "conversely", "yet", "whereas"],
    "cause": ["because", "since", "therefore", "thus", "hence", "consequently", "as a result", "due to"],
    "sequence": ["first", "second", "third", "next", "then", "finally", "subsequently", "afterwards"],
    "example": ["for example", "for instance", "such as", "specifically", "namely"],
    "conclusion": ["in conclusion", "in summary", "to summarize", "overall", "in short"],
}


def discourse_markers_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    text_lower = text.lower()
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return NEUTRAL_SCORE, ["Single sentence -- discourse markers not applicable."]

    by_category: Dict[str, int] = {}
    total_markers = 0
    for category, markers in _DISCOURSE_MARKERS.items():
        count = sum(len(re.findall(rf"\b{re.escape(m)}\b", text_lower)) for m in markers)
        if count > 0:
            by_category[category] = count
            total_markers += count

    sent_count = len(sentences)
    density = total_markers / sent_count

    if total_markers == 0:
        return 0.45, [
            "No discourse markers (however, therefore, additionally...). "
            "Adding them improves logical flow."
        ]

    score = clamp(0.5 + density * 1.2)
    variety = len(by_category)
    score = clamp(score + (variety - 1) * 0.05)

    findings.append(
        f"{total_markers} discourse marker(s) across {variety} categor(ies): "
        f"{', '.join(f'{k}({v})' for k, v in list(by_category.items())[:3])}"
    )

    if density >= 0.3:
        findings.append("Strong logical flow signaling.")
    elif density >= 0.15:
        findings.append("Moderate discourse signaling.")
    else:
        findings.append("Sparse discourse markers.")

    return score, findings


def objectivity_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []

    if word_count(text) < 5:
        return NEUTRAL_SCORE, ["Too short for objectivity analysis."]

    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        subjectivity = float(blob.sentiment.subjectivity)
    except Exception:
        return NEUTRAL_SCORE, ["TextBlob unavailable -- objectivity analysis skipped."]

    score = clamp(1.0 - subjectivity * 0.7)

    if subjectivity < 0.3:
        findings.append(f"Highly objective language (subjectivity {subjectivity:.2f}). Good for prompts.")
    elif subjectivity < 0.6:
        findings.append(f"Mostly objective ({subjectivity:.2f}). Acceptable.")
    else:
        findings.append(
            f"Subjective language ({subjectivity:.2f}). "
            "Replace 'I think/feel/want' with direct instructions."
        )

    return score, findings


def compute_all(text: str) -> dict:
    metrics = {
        "coherence": coherence_score,
        "semantic_density": semantic_density,
        "topic_focus": topic_focus,
        "lexical_sophistication": lexical_sophistication,
        "sentence_flow": sentence_flow,
        "reference_resolution": reference_resolution,
        "tone_consistency": tone_consistency,
        "objectivity": objectivity_score,
        "syntactic_complexity": syntactic_complexity_score,
        "discourse_markers": discourse_markers_score,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
