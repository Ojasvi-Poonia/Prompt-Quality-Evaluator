from typing import Dict, List


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
        ("rule_based.grammar", 0.3),
        ("rule_based.readability", 0.3),
        ("semantic.reference_resolution", 0.2),
        ("semantic.sentence_flow", 0.2),
    ],
    "Specificity": [
        ("rule_based.specificity", 0.4),
        ("semantic.lexical_sophistication", 0.3),
        ("semantic.semantic_density", 0.3),
    ],
    "Structure": [
        ("rule_based.structure", 0.3),
        ("graph.completeness", 0.3),
        ("graph.information_flow", 0.2),
        ("semantic.coherence", 0.2),
    ],
    "Task Definition": [
        ("rule_based.task_signal", 0.5),
        ("graph.completeness", 0.3),
        ("rule_based.constraint", 0.2),
    ],
    "Context Completeness": [
        ("graph.completeness", 0.4),
        ("semantic.topic_focus", 0.3),
        ("rule_based.length", 0.3),
    ],
    "Constraint Specification": [
        ("rule_based.constraint", 0.5),
        ("graph.completeness", 0.3),
        ("rule_based.specificity", 0.2),
    ],
    "Information Richness": [
        ("info.shannon_entropy", 0.25),
        ("info.vocabulary_richness", 0.25),
        ("info.information_density", 0.25),
        ("semantic.semantic_density", 0.25),
    ],
    "Conciseness": [
        ("info.redundancy", 0.35),
        ("info.ngram_repetition", 0.35),
        ("info.burstiness", 0.30),
    ],
}


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
