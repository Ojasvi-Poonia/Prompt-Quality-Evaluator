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
    is_likely_english,
    is_known_word,
    is_typo,
    detect_typos,
    word_zipf,
    strip_code_blocks,
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

    if not text.strip():
        return 0.0, ["No text to check grammar."]

    prose = strip_code_blocks(text)
    if not prose:
        return 0.5, ["Prompt is mostly code -- grammar check not applicable."]

    sentences = split_sentences(prose)
    sent_count = max(len(sentences), 1)

    tool = _get_lang_tool()
    matches = tool.check(prose)
    grammar_errors = len(matches)

    typos = detect_typos(prose)
    typo_count = len(typos)

    total_errors = grammar_errors + typo_count
    score = max(0.0, 1.0 - (total_errors / sent_count) * 0.3)

    if grammar_errors == 0 and typo_count == 0:
        findings.append("No grammar or spelling issues detected.")
    else:
        for m in matches[:3]:
            findings.append(f"Grammar: '{m.message}' near '...{m.context}...'")
        if grammar_errors > 3:
            findings.append(f"...and {grammar_errors - 3} more grammar issues.")
        if typo_count > 0:
            preview = ", ".join(
                f"'{t}'" + (f" -> '{c}'" if c else "")
                for t, c in typos[:5]
            )
            findings.append(f"Likely typos ({typo_count}): {preview}")
            if typo_count > 5:
                findings.append(f"...and {typo_count - 5} more spelling issues.")

    return clamp(score), findings


def readability_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []

    prose = strip_code_blocks(text)
    if word_count(prose) < 3:
        return 0.1, ["Text too short for readability analysis."]
    text = prose

    grades: List[Tuple[str, float]] = []

    def _safe(label: str, fn) -> None:
        try:
            value = fn(text)
            if value is not None and not (isinstance(value, float) and (value != value)):
                grades.append((label, float(value)))
        except Exception:
            pass

    _safe("Flesch-Kincaid", textstat.flesch_kincaid_grade)
    _safe("Coleman-Liau", textstat.coleman_liau_index)
    _safe("ARI", textstat.automated_readability_index)
    _safe("Gunning Fog", textstat.gunning_fog)
    _safe("SMOG", textstat.smog_index)
    _safe("Linsear Write", textstat.linsear_write_formula)

    if not grades:
        return 0.5, ["Could not compute any readability metric (text too short or unusual)."]

    valid_grades = [g for _, g in grades if 0 <= g <= 30]
    if not valid_grades:
        return 0.3, ["Readability scores out of expected range."]

    avg_grade = sum(valid_grades) / len(valid_grades)
    score = gaussian_score(avg_grade, mean=10, std=4)

    summary = ", ".join(f"{label}={grade:.1f}" for label, grade in grades[:3])
    findings.append(f"Readability ensemble ({len(grades)} formulas) avg grade {avg_grade:.1f} -- {summary}")

    if avg_grade < 5:
        findings.append(f"Very simple language (avg grade {avg_grade:.1f}). May lack precision.")
    elif avg_grade < 8:
        findings.append(f"Simple language (avg grade {avg_grade:.1f}).")
    elif avg_grade <= 12:
        findings.append(f"Good readability (avg grade {avg_grade:.1f}).")
    elif avg_grade <= 16:
        findings.append(f"Complex language (avg grade {avg_grade:.1f}). May be hard to parse.")
    else:
        findings.append(f"Very complex language (avg grade {avg_grade:.1f}). Simplify for clarity.")

    return clamp(score), findings


def specificity_score(text: str) -> Tuple[float, List[str]]:
    nlp = get_nlp()
    doc = nlp(text)
    findings: List[str] = []
    sentences = split_sentences(text)
    sent_count = max(len(sentences), 1)
    common = get_common_words()

    is_english, english_ratio = is_likely_english(text, threshold=0.5)
    if not is_english:
        findings.append(
            f"Text appears non-English ({english_ratio:.0%} recognised words). "
            "Specificity metrics assume English input."
        )

    entities = [ent.text for ent in doc.ents]
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|px|ms|mb|gb|kb)?\b", text, re.IGNORECASE)

    tech_terms = []
    typo_count = 0
    for t in doc:
        if not t.is_alpha or len(t.text) <= 2 or t.is_stop:
            continue
        lower = t.text.lower()
        if lower in common:
            continue
        if is_typo(t.text):
            typo_count += 1
            continue
        if not is_known_word(t.text):
            continue
        if word_zipf(lower) > 6.0:
            continue
        tech_terms.append(t.text)

    quoted = re.findall(r'"[^"]{2,}"', text) + re.findall(r"'[^']{2,}'", text)
    code_tokens = [t.text for t in doc if is_code_like(t.text)]

    total_specifics = len(entities) + len(numbers) + len(tech_terms) + len(quoted) + len(code_tokens)
    density = total_specifics / sent_count
    score = clamp(density / 8.0)

    if typo_count > 0:
        penalty = min(0.5, typo_count * 0.2)
        score = max(0.0, score - penalty)
        findings.append(
            f"Detected {typo_count} likely typo(s) -- specificity reduced "
            f"(typos are not real technical terms)."
        )

    if not is_english:
        score *= english_ratio

    if entities:
        findings.append(f"Named entities found: {', '.join(entities[:5])}")
    if numbers:
        findings.append(f"Numeric values: {', '.join(numbers[:5])}")
    if tech_terms:
        findings.append(f"Technical terms: {', '.join(tech_terms[:5])}")
    if not entities and not numbers and not tech_terms:
        findings.append("Prompt lacks specific details (no entities, numbers, or technical terms).")

    return score, findings


SECTION_LABEL_PATTERN = (
    r"(?im)^(context|task|format|constraints|instructions|requirements|"
    r"input|output|example|note|goal|objective|background|rules)\s*:(.*)$"
)


def _section_has_substance(content: str) -> bool:
    words = [w for w in content.strip().split() if len(w) > 2]
    substantive = [w for w in words if w.lower() not in {"stuff", "things", "thing", "stuff", "item", "items"}]
    return len(substantive) >= 3


def _markdown_quality_bonus(text: str) -> Tuple[float, List[str]]:
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return 0.0, []

    findings: List[str] = []
    md = MarkdownIt()
    try:
        tokens = md.parse(text)
    except Exception:
        return 0.0, []

    well_formed_lists = 0
    headers_with_content = 0
    code_blocks = 0
    nesting_depth = 0
    current_depth = 0

    for tok in tokens:
        if tok.type in ("bullet_list_open", "ordered_list_open"):
            current_depth += 1
            nesting_depth = max(nesting_depth, current_depth)
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            well_formed_lists += 1
            current_depth -= 1
        elif tok.type == "heading_open":
            pass
        elif tok.type == "inline" and tok.content:
            pass
        elif tok.type in ("code_block", "fence"):
            code_blocks += 1

    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok.type == "inline" and next_tok.content.strip():
                headers_with_content += 1

    bonus = 0.0
    if well_formed_lists > 0:
        bonus += min(0.15, well_formed_lists * 0.05)
        findings.append(f"Well-formed lists: {well_formed_lists}")
    if headers_with_content > 0:
        bonus += min(0.10, headers_with_content * 0.03)
        findings.append(f"Headers with content: {headers_with_content}")
    if code_blocks > 0:
        bonus += min(0.10, code_blocks * 0.05)
        findings.append(f"Code blocks: {code_blocks}")
    if nesting_depth >= 2:
        bonus += 0.05
        findings.append("Nested structure detected.")

    return bonus, findings


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

    md_bonus, md_findings = _markdown_quality_bonus(text)

    section_matches = re.findall(SECTION_LABEL_PATTERN, text)
    hollow_sections = 0
    real_sections = 0
    for label, content in section_matches:
        if _section_has_substance(content):
            real_sections += 1
        else:
            hollow_sections += 1

    variety_score = clamp(variety / 4.0)
    count_score = clamp(total_elements / 10.0)
    score = 0.6 * variety_score + 0.4 * count_score

    if variety == 0:
        score = 0.2
        findings.append("Plain text with no structural elements. Add sections, lists, or headers.")
    else:
        findings.append(f"Structural elements: {', '.join(detected)}.")
        if md_bonus > 0:
            score = clamp(score + md_bonus)
            findings.append(f"Markdown quality bonus +{md_bonus:.2f} ({'; '.join(md_findings)}).")

    if hollow_sections > 0 and hollow_sections >= real_sections:
        score *= 0.5
        findings.append(
            f"{hollow_sections} section label(s) have vague/empty content. "
            "Labels like 'Task: things' do not count as real structure."
        )

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


_OPPOSING_TONE_PAIRS = [
    ({"formal", "formally"}, {"casual", "casually"}),
    ({"concise", "concisely", "brief", "briefly"}, {"detailed", "comprehensive", "thorough"}),
    ({"professional", "professionally"}, {"friendly", "playful", "casual"}),
]


def _has_constraint_signals(text_lower: str) -> bool:
    constraint_patterns = [
        r"\bmust\b", r"\bshould\b", r"\bdo not\b", r"\bdon'?t\b",
        r"\bmaximum\b", r"\bat most\b", r"\bno more than\b", r"\blimit\b",
        r"\bformat\b", r"\bin json\b", r"\bas a table\b", r"\brequire\b",
        r"\bensure\b", r"\bmake sure\b", r"\bavoid\b", r"\bnever\b",
        r"\bin \d+", r"\bunder\b", r"\bwithin\b", r"\bexactly\b",
        r"\bat least\b", r"\bminimum\b", r"\bup to\b",
        r"\bformal(ly)?\b", r"\bcasual(ly)?\b", r"\bconcise(ly)?\b",
        r"\bdetailed\b", r"\bbrief(ly)?\b",
    ]
    return any(re.search(p, text_lower) for p in constraint_patterns)


def contradiction_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    text_lower = text.lower()
    contradictions = 0

    if not _has_constraint_signals(text_lower):
        findings.append("No constraints detected -- contradiction check vacuous (neutral score).")
        return 0.5, findings

    words_match = re.search(r"(\d+)\s*words?", text_lower)
    sentences_match = re.search(r"(\d+)\s*sentences?", text_lower)
    if words_match and sentences_match:
        w = int(words_match.group(1))
        s = int(sentences_match.group(1))
        if s > 0 and w / s > 80:
            contradictions += 1
            findings.append(
                f"Conflict: {w} words in {s} sentences implies {w//s} words/sentence (unrealistic)."
            )
        elif s > 0 and w / s < 3:
            contradictions += 1
            findings.append(
                f"Conflict: {w} words across {s} sentences is impossibly terse."
            )

    pages_match = re.search(r"(\d+)\s*pages?", text_lower)
    if pages_match and re.search(r"\b(one|single)\s+sentence\b", text_lower):
        contradictions += 1
        findings.append("Conflict: multiple pages but only one sentence requested.")

    for set_a, set_b in _OPPOSING_TONE_PAIRS:
        has_a = any(re.search(rf"\b{w}\b", text_lower) for w in set_a)
        has_b = any(re.search(rf"\b{w}\b", text_lower) for w in set_b)
        if has_a and has_b:
            contradictions += 1
            findings.append(
                f"Conflicting tone directives: {'/'.join(sorted(set_a))} vs "
                f"{'/'.join(sorted(set_b))}."
            )

    if re.search(r"\b(do not|don'?t|avoid|never)\s+include\s+examples?\b", text_lower) and \
       re.search(r"\b(include|provide|give|show)\s+examples?\b", text_lower):
        contradictions += 1
        findings.append("Conflict: both 'include examples' and 'do not include examples' detected.")

    if contradictions == 0:
        findings.append("No logical contradictions detected.")
        return 1.0, findings

    score = clamp(1.0 - contradictions * 0.35)
    return score, findings


_PERSONA_PATTERNS = [
    r"\byou are (?:a |an |the )?[\w\s\-]{2,40}",
    r"\byou'?re (?:a |an |the )?[\w\s\-]{2,40}",
    r"\bact (?:as|like) (?:a |an |the )?[\w\s\-]{2,40}",
    r"\bpretend (?:to be|you are|you'?re) [\w\s\-]{2,40}",
    r"\bimagine (?:you are|you'?re|that you) [\w\s\-]{2,40}",
    r"\btake the role of [\w\s\-]{2,40}",
    r"\bassume the role of [\w\s\-]{2,40}",
    r"\byou'?ll (?:act|be|play) (?:as |the role of )?[\w\s\-]{2,40}",
    r"\byour role is [\w\s\-]{2,40}",
    r"\bas (?:a |an )?(?:senior|expert|professional|experienced) [\w\s\-]{2,40}",
    r"\bspeak as (?:a |an |the )?[\w\s\-]{2,40}",
]


def persona_signal_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    text_lower = text.lower()

    matches = []
    for pattern in _PERSONA_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            matches.append(m.group(0)[:60])

    if not matches:
        return 0.5, ["No explicit persona/role set (neutral)."]

    nlp = get_nlp()
    doc = nlp(text)
    role_specificity = 0
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "NORP", "GPE"}:
            role_specificity += 1
    role_specificity += sum(1 for tok in doc if tok.pos_ in {"PROPN", "NOUN"} and len(tok.text) > 2 and tok.is_alpha)

    persona_count = len(matches)
    base = min(0.6 + persona_count * 0.15, 0.9)
    if role_specificity > 5:
        base = min(1.0, base + 0.1)

    findings.append(f"Persona/role detected: '{matches[0]}'")
    if persona_count > 1:
        findings.append(f"{persona_count} role-setting phrases found.")

    return clamp(base), findings


_FEW_SHOT_PATTERNS = [
    (r"(?im)^\s*(?:input|in|q|question|user)\s*:\s*.+\n+\s*(?:output|out|a|answer|assistant|response)\s*:\s*.+", "io_pair"),
    (r"(?im)^\s*example\s*\d*\s*:\s*.+", "example_label"),
    (r"(?im)^\s*\d+\.\s+input\s*:.+\n.*\d+\.\s+output\s*:", "numbered_io"),
    (r"\->", "arrow_mapping"),
]


def few_shot_quality_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []

    io_pairs = re.findall(_FEW_SHOT_PATTERNS[0][0], text)
    example_labels = re.findall(_FEW_SHOT_PATTERNS[1][0], text)
    arrow_mappings = re.findall(r"^\s*[\w\s'\".,]{1,40}\s*->\s*[\w\s'\".,\[\]\{\}]{1,80}\s*$",
                                 text, re.MULTILINE)

    pair_count = len(io_pairs)
    example_count = len(example_labels)
    arrow_count = len(arrow_mappings)

    total_examples = pair_count + max(0, example_count - 1)
    if arrow_count >= 2:
        total_examples = max(total_examples, arrow_count)

    if total_examples == 0:
        return 0.5, ["No few-shot examples detected (neutral)."]

    if total_examples == 1:
        score = 0.55
        findings.append(f"Single example detected (consider 2+ for true few-shot).")
    elif total_examples == 2:
        score = 0.75
        findings.append(f"2 examples detected -- minimal few-shot.")
    elif total_examples <= 5:
        score = 0.90
        findings.append(f"{total_examples} examples detected -- good few-shot pattern.")
    else:
        score = 0.95
        findings.append(f"{total_examples} examples detected -- comprehensive few-shot.")

    if pair_count > 0:
        findings.append(f"Input/Output pairs: {pair_count}")
    if arrow_count >= 2:
        findings.append(f"Arrow mappings (->): {arrow_count}")

    return clamp(score), findings


_HEDGING_TERMS = {
    "maybe", "perhaps", "possibly", "kind of", "kinda", "sort of", "sorta",
    "i think", "i guess", "i suppose", "i feel", "i believe",
    "if possible", "if you can", "if it works", "if that's okay",
    "would it be okay", "could you maybe", "would you mind",
    "somewhat", "rather", "fairly", "quite", "pretty much",
    "more or less", "i'd like", "i would like", "would be nice",
    "hopefully", "ideally", "preferably",
}


def hedging_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    text_lower = text.lower()
    wc = max(word_count(text), 1)

    found = []
    for term in _HEDGING_TERMS:
        count = len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
        if count > 0:
            found.append((term, count))

    total_hedges = sum(c for _, c in found)

    if total_hedges == 0:
        return 0.95, ["No hedging language detected (clear, direct prompt)."]

    density = total_hedges / wc
    score = clamp(1.0 - density * 30.0)

    if density > 0.1:
        findings.append(f"Heavy hedging: {total_hedges} hedge term(s) in {wc} words. Be direct.")
    elif density > 0.04:
        findings.append(f"Moderate hedging: {total_hedges} hedge term(s).")
    else:
        findings.append(f"Light hedging: {total_hedges} term(s) detected.")

    if found:
        examples = [t for t, _ in found[:3]]
        findings.append(f"Examples: {', '.join(examples)}")

    return score, findings


_INJECTION_PATTERNS = [
    r"\bignore (?:previous|all|the|any) (?:instructions?|prompts?|rules?|context)\b",
    r"\bdisregard (?:previous|all|the|any) (?:instructions?|prompts?|rules?)\b",
    r"\bforget (?:everything|what i said|previous|all|prior)\b",
    r"\bnow (?:instead|act as)\b.*\b(?:ignore|forget)\b",
    r"\bpretend (?:that )?(?:you are|to be) (?:a |an )?(?:different|new|other)\b",
    r"\bsystem\s*:\s*",
    r"\b(?:override|bypass|circumvent) (?:safety|filter|guard)",
    r"\byou are no longer\b",
    r"\byour new instructions are\b",
    r"\bend of (?:previous|prior) (?:instructions?|context)\b",
]


def safety_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    text_lower = text.lower()

    matches = []
    for pattern in _INJECTION_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            matches.append(m.group(0)[:50])

    if not matches:
        return 1.0, ["No prompt-injection patterns detected."]

    score = max(0.0, 1.0 - len(matches) * 0.3)
    findings.append(f"WARNING: {len(matches)} prompt-injection pattern(s) detected.")
    for m in matches[:3]:
        findings.append(f"  - '{m}...'")
    findings.append("These patterns suggest adversarial intent or sloppy chaining.")

    return clamp(score), findings


def output_schema_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    score_components = []

    json_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if json_block:
        try:
            import json as _json
            content = json_block.group(1)
            normalised = re.sub(r":\s*(string|int(?:eger)?|number|float|bool(?:ean)?|null|any|array|\[?\w+\??\]?)(?=[,}])", r': "\1"', content)
            normalised = re.sub(r":\s*(string|int(?:eger)?|number|float|bool(?:ean)?)\s*\]", r': "\1"]', normalised)
            try:
                parsed = _json.loads(normalised)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    score_components.append(0.85)
                    findings.append(f"JSON schema-like structure with {len(parsed)} fields.")
            except _json.JSONDecodeError:
                if any(t in content for t in ["string", "int", "number", "boolean", "null"]):
                    score_components.append(0.7)
                    findings.append("JSON-like schema detected (with type annotations).")
        except Exception:
            pass

    if re.search(r"```(?:typescript|ts)\s*(?:type|interface)\s+\w+\s*=?\s*\{", text):
        score_components.append(0.85)
        findings.append("TypeScript type/interface definition detected.")

    yaml_schema = re.search(r"```(?:yaml|yml)\s*[\s\S]*?:\s*(?:string|integer|number|boolean|array|object)", text)
    if yaml_schema:
        score_components.append(0.80)
        findings.append("YAML schema with type annotations detected.")

    if re.search(r"```\s*(?:def|class|function|interface)\s+\w+\s*\(", text):
        score_components.append(0.70)
        findings.append("Function/class signature template detected.")

    fields_pattern = re.findall(r"^\s*[-*]\s*\*?\*?(\w+)\*?\*?\s*[:\(]\s*(?:string|str|int|integer|number|float|bool|boolean|date|array|list|object|dict)\b", text, re.MULTILINE | re.IGNORECASE)
    if len(fields_pattern) >= 3:
        score_components.append(0.75)
        findings.append(f"Field-list schema with {len(fields_pattern)} typed fields.")

    if not score_components:
        return 0.5, ["No explicit output schema detected (neutral)."]

    score = max(score_components)
    if len(score_components) > 1:
        score = min(1.0, score + 0.05 * (len(score_components) - 1))
        findings.append(f"Multiple schema styles detected (+bonus).")

    return clamp(score), findings


_TASK_VERB_LEMMAS = {
    "write", "explain", "list", "compare", "analyze", "analyse",
    "summarize", "summarise", "create", "generate", "design",
    "build", "implement", "evaluate", "describe", "provide",
    "calculate", "translate", "convert", "refactor", "debug",
    "review", "optimize", "optimise", "classify", "extract",
    "suggest", "recommend", "outline", "define", "develop",
    "filter", "sort", "compute", "export", "import", "parse",
    "validate", "verify", "transform", "format", "render",
    "read", "fetch", "load", "save", "delete", "update",
}


def compound_task_count_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    nlp = get_nlp()
    doc = nlp(text)

    task_verbs_found = []
    for sent in doc.sents:
        for tok in sent:
            if tok.lemma_.lower() in _TASK_VERB_LEMMAS and tok.pos_ in {"VERB", "AUX"}:
                task_verbs_found.append(tok.lemma_.lower())
                for child in tok.head.children:
                    if child.dep_ == "conj" and child.lemma_.lower() in _TASK_VERB_LEMMAS:
                        task_verbs_found.append(child.lemma_.lower())
                if tok.dep_ == "conj" and tok.head.lemma_.lower() in _TASK_VERB_LEMMAS:
                    pass

    distinct_tasks = list(dict.fromkeys(task_verbs_found))
    n = len(distinct_tasks)

    if n == 0:
        return 0.5, ["No task verbs detected (neutral)."]

    if n == 1:
        score = 0.65
        findings.append(f"Single task verb: {distinct_tasks[0]}")
    elif n <= 3:
        score = 0.85
        findings.append(f"{n} chained tasks: {', '.join(distinct_tasks)} -- good compound prompt.")
    elif n <= 6:
        score = 0.95
        findings.append(f"{n} chained tasks: {', '.join(distinct_tasks[:5])}{'...' if n > 5 else ''} -- complex multi-step.")
    else:
        score = 0.6
        findings.append(
            f"{n} task verbs detected -- prompt may be over-stuffed. Consider splitting."
        )

    return clamp(score), findings


def imperative_strength_score(text: str) -> Tuple[float, List[str]]:
    findings: List[str] = []
    nlp = get_nlp()
    doc = nlp(text)

    imperative_count = 0
    polite_softener_count = 0
    capital_emphasis = 0
    exclamation_count = text.count("!")

    softeners = {"please", "kindly", "could", "would", "may", "might", "perhaps"}

    for sent in doc.sents:
        first_alpha = next((t for t in sent if t.is_alpha), None)
        if not first_alpha:
            continue
        if first_alpha.lemma_.lower() in _TASK_VERB_LEMMAS and first_alpha.pos_ in {"VERB", "AUX"}:
            imperative_count += 1
            if first_alpha.text.isupper() and len(first_alpha.text) > 2:
                capital_emphasis += 1
        for tok in sent:
            if tok.text.lower() in softeners:
                polite_softener_count += 1

    if imperative_count == 0:
        return 0.4, ["No imperative-mood verbs detected at sentence start."]

    base = 0.7
    if imperative_count >= 2:
        base += 0.1
    if polite_softener_count == 0:
        base += 0.1
    elif polite_softener_count > imperative_count:
        base -= 0.15
    if capital_emphasis > 0 and capital_emphasis < imperative_count:
        base += 0.05

    score = clamp(base)

    parts = [f"{imperative_count} imperative(s)"]
    if polite_softener_count > 0:
        parts.append(f"{polite_softener_count} softener(s)")
    if capital_emphasis > 0:
        parts.append(f"{capital_emphasis} all-caps")
    findings.append(" / ".join(parts))

    if polite_softener_count > imperative_count:
        findings.append("Heavy politeness markers may dilute the imperative.")
    elif imperative_count >= 2:
        findings.append("Multiple direct imperatives -- strong instruction signal.")

    return score, findings


def reference_similarity_score(text: str) -> Tuple[float, List[str]]:
    from evaluator.reference_templates import reference_similarity
    from evaluator.rubric import detect_domain

    findings: List[str] = []
    domain, _ = detect_domain(text)

    sim = reference_similarity(text, domain)
    if sim is None:
        return 0.5, ["Could not compute reference similarity (no word vectors available)."]

    score = clamp((sim - 0.4) / 0.5)

    if score >= 0.8:
        findings.append(
            f"Strongly matches canonical {domain} prompt patterns (similarity {sim:.2f})."
        )
    elif score >= 0.5:
        findings.append(
            f"Moderate match to canonical {domain} prompt patterns (similarity {sim:.2f})."
        )
    elif score >= 0.3:
        findings.append(
            f"Weak match to canonical {domain} patterns (similarity {sim:.2f}) -- consider adding context/task/constraints sections."
        )
    else:
        findings.append(
            f"Far from canonical {domain} prompt patterns (similarity {sim:.2f})."
        )

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
        "contradiction": contradiction_score,
        "reference_similarity": reference_similarity_score,
        "persona_signal": persona_signal_score,
        "few_shot_quality": few_shot_quality_score,
        "hedging": hedging_score,
        "safety": safety_score,
        "output_schema": output_schema_score,
        "compound_task_count": compound_task_count_score,
        "imperative_strength": imperative_strength_score,
    }
    results = {}
    for name, fn in metrics.items():
        score, findings = fn(text)
        results[name] = {"score": score, "findings": findings}
    return results
