"""Pillar 1: Rule-Based Metrics.

Seven heuristic functions that score a prompt on surface-level quality
signals.  Each returns ``(score: float, findings: list[str])`` where
*score* is in [0, 1] and *findings* explains what was detected.

No transformer models are used — only regex, spaCy NER, textstat, and
language_tool_python.
"""

import re
import math
from typing import Tuple, List

import textstat
import language_tool_python

from evaluator.utils import (
    get_nlp,
    get_common_words,
    split_sentences,
    word_count,
    clamp,
    gaussian_score,
    is_code_like,
)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_lang_tool = None


def _get_lang_tool():
    """Return a cached LanguageTool instance."""
    global _lang_tool
    if _lang_tool is None:
        _lang_tool = language_tool_python.LanguageTool("en-US")
    return _lang_tool


# ===================================================================
# 1. Length Score
# ===================================================================

def length_score(text: str) -> Tuple[float, List[str]]:
    """Score prompt length on a Gaussian curve centred at 150 words.

    Range: 0-1.  Sweet spot is 50-500 words.
    Penalises <10 words (too short) and >2000 words (too long).
    """
    wc = word_count(text)
    findings: List[str] = []

    if wc == 0:
        return 0.0, ["Prompt is empty."]

    # Gaussian centred at 150 words, std=200
    score = gaussian_score(wc, mean=150, std=200)

    # Hard penalties for extremes
    if wc < 10:
        score *= 0.3
        findings.append(f"Very short prompt ({wc} words). Aim for 50-500 words.")
    elif wc < 50:
        findings.append(f"Short prompt ({wc} words). More detail may improve quality.")
    elif wc > 2000:
        score *= 0.5
        findings.append(f"Very long prompt ({wc} words). Consider trimming to essentials.")
    elif wc > 500:
        findings.append(f"Long prompt ({wc} words). Ensure all content is necessary.")
    else:
        findings.append(f"Good prompt length ({wc} words).")

    return clamp(score), findings


# ===================================================================
# 2. Grammar Quality
# ===================================================================

def grammar_quality(text: str) -> Tuple[float, List[str]]:
    """Score grammatical correctness using LanguageTool.

    Score = max(0, 1 - (error_count / sentence_count) * 0.3).
    Each grammar error is returned as a finding.
    """
    findings: List[str] = []
    sentences = split_sentences(text)
    sent_count = max(len(sentences), 1)

    if not text.strip():
        return 0.0, ["No text to check grammar."]

    tool = _get_lang_tool()
    matches = tool.check(text)

    error_count = len(matches)
    score = max(0.0, 1.0 - (error_count / sent_count) * 0.3)

    if error_count == 0:
        findings.append("No grammar issues detected.")
    else:
        for m in matches[:5]:  # cap reported findings
            findings.append(f"Grammar: '{m.message}' near '...{m.context}...'")
        if error_count > 5:
            findings.append(f"...and {error_count - 5} more grammar issues.")

    return clamp(score), findings


# ===================================================================
# 3. Readability Score
# ===================================================================

def readability_score(text: str) -> Tuple[float, List[str]]:
    """Score readability via Flesch-Kincaid grade level.

    Ideal range is grade 8-12 (clear but not oversimplified).
    Too simple (<5) or too complex (>16) is penalised.
    """
    findings: List[str] = []

    if word_count(text) < 3:
        return 0.1, ["Text too short for readability analysis."]

    grade = textstat.flesch_kincaid_grade(text)

    # Bell curve peaking at grade 10
    score = gaussian_score(grade, mean=10, std=4)

    if grade < 5:
        findings.append(f"Very simple language (grade {grade:.1f}). May lack precision.")
    elif grade < 8:
        findings.append(f"Simple language (grade {grade:.1f}).")
    elif grade <= 12:
        findings.append(f"Good readability (grade {grade:.1f}).")
    elif grade <= 16:
        findings.append(f"Complex language (grade {grade:.1f}). May be hard to parse.")
    else:
        findings.append(f"Very complex language (grade {grade:.1f}). Simplify for clarity.")

    return clamp(score), findings


# ===================================================================
# 4. Specificity Score
# ===================================================================

def specificity_score(text: str) -> Tuple[float, List[str]]:
    """Score how specific the prompt is.

    Counts named entities, numbers, technical terms, quoted strings,
    and code-like tokens.  Normalises by sentence count.
    """
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []
    sentences = split_sentences(text)
    sent_count = max(len(sentences), 1)
    common = get_common_words()

    # Named entities
    entities = [ent.text for ent in doc.ents]
    # Numbers / quantities
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|px|ms|mb|gb|kb)?\b", text, re.IGNORECASE)
    # Technical terms (not in top 5000)
    tech_terms = [
        t.text for t in doc
        if t.is_alpha and len(t.text) > 2 and t.text.lower() not in common and not t.is_stop
    ]
    # Quoted strings
    quoted = re.findall(r'"[^"]{2,}"', text) + re.findall(r"'[^']{2,}'", text)
    # Code-like tokens
    code_tokens = [t.text for t in doc if is_code_like(t.text)]

    total_specifics = len(entities) + len(numbers) + len(tech_terms) + len(quoted) + len(code_tokens)
    density = total_specifics / sent_count

    # Map density to 0-1 (roughly 0-10 specifics per sentence)
    score = clamp(density / 8.0)

    if entities:
        findings.append(f"Named entities found: {', '.join(entities[:5])}")
    if numbers:
        findings.append(f"Numeric values: {', '.join(numbers[:5])}")
    if tech_terms:
        findings.append(f"Technical terms: {', '.join(tech_terms[:5])}")
    if not entities and not numbers and not tech_terms:
        findings.append("Prompt lacks specific details (no entities, numbers, or technical terms).")

    return score, findings


# ===================================================================
# 5. Structure Score
# ===================================================================

def structure_score(text: str) -> Tuple[float, List[str]]:
    """Score formatting and structural organisation.

    Detects bullet points, numbered lists, headers, delimiters,
    section labels, and line-break separation.  Scores based on
    count and variety of elements.
    """
    findings: List[str] = []
    detected: List[str] = []

    patterns = {
        "bullet_points": r"^[\s]*[-*\u2022]\s",
        "numbered_lists": r"^[\s]*\d+[.)]\s",
        "markdown_headers": r"^#{1,6}\s",
        "code_blocks": r"```",
        "horizontal_rules": r"^-{3,}$",
        "section_labels": r"(?i)^(context|task|format|constraints|instructions|requirements|input|output|example|note|goal|objective|background|rules)\s*:",
    }

    variety = 0
    total_elements = 0

    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            variety += 1
            total_elements += len(matches)
            label = name.replace("_", " ")
            detected.append(f"{label} ({len(matches)})")

    # Line-break separation (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        variety += 1
        total_elements += len(paragraphs)
        detected.append(f"paragraphs ({len(paragraphs)})")

    # Score: variety (0-6 types) contributes 60%, count contributes 40%
    variety_score = clamp(variety / 4.0)
    count_score = clamp(total_elements / 10.0)
    score = 0.6 * variety_score + 0.4 * count_score

    # Floor for plain text
    if variety == 0:
        score = 0.2
        findings.append("Plain text with no structural elements. Add sections, lists, or headers.")
    else:
        findings.append(f"Structural elements: {', '.join(detected)}.")

    return clamp(score), findings


# ===================================================================
# 6. Task Signal Score
# ===================================================================

def task_signal_score(text: str) -> Tuple[float, List[str]]:
    """Score clarity of the prompt's task/instruction.

    Detects imperative verbs, question patterns, and explicit
    deliverable mentions.  Score 0 if no task signal found.
    """
    findings: List[str] = []
    text_lower = text.lower()

    imperative_verbs = [
        "write", "explain", "list", "compare", "analyze", "analyse",
        "summarize", "summarise", "create", "generate", "design",
        "build", "implement", "evaluate", "describe", "provide",
        "calculate", "translate", "convert", "refactor", "debug",
        "review", "optimize", "optimise", "classify", "extract",
        "suggest", "recommend", "outline", "define", "develop",
    ]

    question_patterns = [
        r"\b(what|how|why|when|where|which|who|can you|could you)\b.*\?",
        r"\?$",
    ]

    deliverables = [
        "code", "script", "function", "class", "api", "table",
        "essay", "list", "json", "csv", "report", "summary",
        "diagram", "plan", "outline", "algorithm", "query",
        "template", "html", "css", "sql", "yaml", "xml",
    ]

    found_imperatives = [v for v in imperative_verbs if re.search(rf"\b{v}\b", text_lower)]
    found_questions = any(re.search(p, text_lower, re.MULTILINE) for p in question_patterns)
    found_deliverables = [d for d in deliverables if re.search(rf"\b{d}\b", text_lower)]

    signals = 0
    if found_imperatives:
        signals += min(len(found_imperatives), 3)
        findings.append(f"Task verbs: {', '.join(found_imperatives[:5])}")
    if found_questions:
        signals += 1
        findings.append("Contains question pattern.")
    if found_deliverables:
        signals += min(len(found_deliverables), 2)
        findings.append(f"Deliverables mentioned: {', '.join(found_deliverables[:5])}")

    if signals == 0:
        findings.append("No clear task signal. Add an imperative verb or question.")

    score = clamp(signals / 5.0)
    return score, findings


# ===================================================================
# 7. Constraint Score
# ===================================================================

def constraint_score(text: str) -> Tuple[float, List[str]]:
    """Score the presence of constraints and guardrails.

    Detects negations, limits, format constraints, audience specs,
    and tone directives.  Scores on count and variety.
    """
    findings: List[str] = []
    text_lower = text.lower()

    categories = {
        "negations": [
            r"\bdo not\b", r"\bdon'?t\b", r"\bavoid\b", r"\bnever\b",
            r"\bexclude\b", r"\bwithout\b", r"\bno\s+\w+",
        ],
        "limits": [
            r"\bmaximum\b", r"\bat most\b", r"\bno more than\b",
            r"\bwithin\b", r"\bunder\b", r"\blimit\b", r"\bup to\b",
            r"\bat least\b", r"\bminimum\b", r"\bexactly\b",
        ],
        "format_constraints": [
            r"\bin json\b", r"\bas a table\b", r"\bin bullet points?\b",
            r"\bin \d+ paragraphs?\b", r"\bas markdown\b", r"\bin csv\b",
            r"\bas a list\b", r"\bin yaml\b", r"\bformatted as\b",
        ],
        "audience": [
            r"\bfor beginners?\b", r"\bfor a (\d[- ]year[- ]old|child)\b",
            r"\bfor experts?\b", r"\bfor (a )?technical audience\b",
            r"\bfor (a )?non[- ]technical\b", r"\bfor (a )?developer\b",
        ],
        "tone": [
            r"\bformal(ly)?\b", r"\bcasual(ly)?\b", r"\bprofessional(ly)?\b",
            r"\bfriendly\b", r"\bconcise(ly)?\b", r"\bdetailed\b",
            r"\bbrief(ly)?\b", r"\btechnical(ly)?\b",
        ],
    }

    variety = 0
    total = 0

    for cat_name, patterns in categories.items():
        matches = sum(1 for p in patterns if re.search(p, text_lower))
        if matches:
            variety += 1
            total += matches
            findings.append(f"{cat_name.replace('_', ' ').title()}: {matches} constraint(s).")

    if variety == 0:
        findings.append("No constraints specified. Adding constraints improves output quality.")

    score = clamp((variety * 0.15) + (total * 0.05))
    return score, findings


# ===================================================================
# Aggregate helper
# ===================================================================

def compute_all(text: str) -> dict:
    """Run all Pillar-1 metrics and return a dict of results.

    Each key maps to ``{"score": float, "findings": list[str]}``.
    """
    metrics = {
        "length": length_score,
        "grammar": grammar_quality,
        "readability": readability_score,
        "specificity": specificity_score,
        "structure": structure_score,
        "task_signal": task_signal_score,
        "constraint": constraint_score,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
