import re
from typing import Dict, List, Tuple


def _label_for_score(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Strong"
    if score >= 0.50:
        return "Good"
    if score >= 0.30:
        return "Fair"
    return "Weak"


DIMENSIONS = {
    "Clarity": [
        ("rule_based.grammar", 0.18),
        ("rule_based.readability", 0.15),
        ("semantic.reference_resolution", 0.10),
        ("semantic.sentence_flow", 0.10),
        ("semantic.tone_consistency", 0.08),
        ("semantic.objectivity", 0.10),
        ("rule_based.hedging", 0.14),
        ("semantic.syntactic_complexity", 0.15),
    ],
    "Specificity": [
        ("rule_based.specificity", 0.30),
        ("rule_based.reference_similarity", 0.10),
        ("rule_based.output_schema", 0.15),
        ("semantic.lexical_sophistication", 0.20),
        ("semantic.semantic_density", 0.15),
        ("rule_based.persona_signal", 0.10),
    ],
    "Structure": [
        ("rule_based.structure", 0.20),
        ("graph.completeness", 0.10),
        ("graph.information_flow", 0.10),
        ("graph.centrality", 0.10),
        ("semantic.coherence", 0.10),
        ("semantic.discourse_markers", 0.10),
        ("rule_based.reference_similarity", 0.10),
        ("rule_based.few_shot_quality", 0.10),
        ("rule_based.output_schema", 0.10),
    ],
    "Task Definition": [
        ("rule_based.task_signal", 0.30),
        ("rule_based.imperative_strength", 0.15),
        ("rule_based.compound_task_count", 0.10),
        ("graph.has_task", 0.15),
        ("rule_based.contradiction", 0.10),
        ("rule_based.persona_signal", 0.05),
        ("rule_based.safety", 0.15),
    ],
    "Context Completeness": [
        ("graph.has_context", 0.30),
        ("semantic.topic_focus", 0.20),
        ("rule_based.length", 0.20),
        ("rule_based.persona_signal", 0.15),
        ("rule_based.few_shot_quality", 0.15),
    ],
    "Constraint Specification": [
        ("rule_based.constraint", 0.35),
        ("graph.has_constraint", 0.20),
        ("rule_based.contradiction", 0.15),
        ("rule_based.output_schema", 0.20),
        ("rule_based.safety", 0.10),
    ],
    "Information Richness": [
        ("info.shannon_entropy", 0.25),
        ("info.vocabulary_richness", 0.25),
        ("info.information_density", 0.25),
        ("semantic.semantic_density", 0.25),
    ],
    "Conciseness": [
        ("info.redundancy", 0.25),
        ("info.ngram_repetition", 0.25),
        ("info.sentence_repetition", 0.25),
        ("info.burstiness", 0.25),
    ],
}


CODING_KEYWORDS = {
    "function", "code", "script", "class", "api", "database", "sql", "json",
    "python", "javascript", "typescript", "rust", "golang", "java", "kotlin",
    "django", "flask", "fastapi", "react", "vue", "angular", "node", "deploy",
    "endpoint", "backend", "frontend", "algorithm", "refactor", "debug",
    "postgres", "mongodb", "redis", "docker", "kubernetes", "regex",
}

CREATIVE_KEYWORDS = {
    "story", "poem", "narrative", "character", "tone", "style", "creative",
    "write about", "imagine", "describe a scene", "essay", "blog", "article",
    "novel", "short story", "screenplay", "dialogue", "metaphor",
}

ANALYSIS_KEYWORDS = {
    "analyze", "analyse", "compare", "evaluate", "assess", "data", "trends",
    "report", "insights", "findings", "statistics", "metrics", "benchmark",
    "pros and cons", "advantages", "disadvantages", "summarize", "summarise",
    "critique", "review",
}

INSTRUCTIONAL_KEYWORDS = {
    "explain", "teach", "describe", "how does", "what is", "why does",
    "tutorial", "guide", "walkthrough", "step by step", "beginner",
    "introduction",
}


DOMAIN_WEIGHT_OVERRIDES = {
    "coding": {
        "Specificity": 1.3,
        "Constraint Specification": 1.3,
        "Task Definition": 1.2,
        "Structure": 1.1,
        "Clarity": 0.9,
    },
    "creative": {
        "Context Completeness": 1.3,
        "Information Richness": 1.2,
        "Clarity": 1.1,
        "Constraint Specification": 0.7,
        "Structure": 0.8,
    },
    "analysis": {
        "Context Completeness": 1.2,
        "Task Definition": 1.2,
        "Constraint Specification": 1.1,
        "Specificity": 1.1,
    },
    "instructional": {
        "Task Definition": 1.2,
        "Clarity": 1.2,
        "Context Completeness": 1.1,
        "Constraint Specification": 0.8,
    },
    "general": {},
}


def detect_domain(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    scores = {
        "coding": sum(1 for kw in CODING_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text_lower)),
        "creative": sum(1 for kw in CREATIVE_KEYWORDS if kw in text_lower),
        "analysis": sum(1 for kw in ANALYSIS_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text_lower)),
        "instructional": sum(1 for kw in INSTRUCTIONAL_KEYWORDS if kw in text_lower),
    }

    best_domain, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score < 2:
        return "general", 0.0

    confidence = best_score / (sum(scores.values()) + 1)
    return best_domain, confidence


def _resolve_metric(key: str, flat_metrics: Dict[str, dict]) -> float:
    entry = flat_metrics.get(key)
    if entry is None:
        return 0.0
    return entry.get("score", 0.0)


def _collect_findings(keys: List[str], flat_metrics: Dict[str, dict]) -> List[str]:
    findings: List[str] = []
    for key in keys:
        entry = flat_metrics.get(key)
        if entry and entry.get("findings"):
            findings.extend(entry["findings"])
    return findings


def map_to_rubric(flat_metrics: Dict[str, dict]) -> List[dict]:
    dimensions = []

    for dim_name, components in DIMENSIONS.items():
        weighted_sum = 0.0
        total_weight = 0.0
        metric_keys = []

        for key, weight in components:
            score = _resolve_metric(key, flat_metrics)
            weighted_sum += score * weight
            total_weight += weight
            metric_keys.append(key)

        dim_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        findings = _collect_findings(metric_keys, flat_metrics)

        dimensions.append({
            "name": dim_name,
            "score": round(dim_score, 3),
            "label": _label_for_score(dim_score),
            "findings": findings,
        })

    return dimensions


def apply_domain_weights(dimensions: List[dict], domain: str) -> Dict[str, float]:
    overrides = DOMAIN_WEIGHT_OVERRIDES.get(domain, {})
    return {d["name"]: overrides.get(d["name"], 1.0) for d in dimensions}
