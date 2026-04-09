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


_lang_tool = None


def _get_lang_tool():
    global _lang_tool
    if _lang_tool is None:
        _lang_tool = language_tool_python.LanguageTool("en-US")
    return _lang_tool


def length_score(text: str) -> Tuple[float, List[str]]:
    wc = word_count(text)
    findings: List[str] = []

    if wc == 0:
        return 0.0, ["Prompt is empty."]

    score = gaussian_score(wc, mean=150, std=200)

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


def grammar_quality(text: str) -> Tuple[float, List[str]]:
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
        for m in matches[:5]:
            findings.append(f"Grammar: '{m.message}' near '...{m.context}...'")
        if error_count > 5:
            findings.append(f"...and {error_count - 5} more grammar issues.")

    return clamp(score), findings


def readability_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []

    if word_count(text) < 3:
        return 0.1, ["Text too short for readability analysis."]

    grade = textstat.flesch_kincaid_grade(text)
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


def specificity_score(text: str) -> Tuple[float, List[str]]:
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []
    sentences = split_sentences(text)
    sent_count = max(len(sentences), 1)
    common = get_common_words()

    entities = [ent.text for ent in doc.ents]
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|px|ms|mb|gb|kb)?\b", text, re.IGNORECASE)
    tech_terms = [
        t.text for t in doc
        if t.is_alpha and len(t.text) > 2 and t.text.lower() not in common and not t.is_stop
    ]
    quoted = re.findall(r'"[^"]{2,}"', text) + re.findall(r"'[^']{2,}'", text)
    code_tokens = [t.text for t in doc if is_code_like(t.text)]

    total_specifics = len(entities) + len(numbers) + len(tech_terms) + len(quoted) + len(code_tokens)
    density = total_specifics / sent_count
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


def structure_score(text: str) -> Tuple[float, List[str]]:
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

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        variety += 1
        total_elements += len(paragraphs)
        detected.append(f"paragraphs ({len(paragraphs)})")

    variety_score = clamp(variety / 4.0)
    count_score = clamp(total_elements / 10.0)
    score = 0.6 * variety_score + 0.4 * count_score

    if variety == 0:
        score = 0.2
        findings.append("Plain text with no structural elements. Add sections, lists, or headers.")
    else:
        findings.append(f"Structural elements: {', '.join(detected)}.")

    return clamp(score), findings


def task_signal_score(text: str) -> Tuple[float, List[str]]:
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


def constraint_score(text: str) -> Tuple[float, List[str]]:
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


def compute_all(text: str) -> dict:
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
