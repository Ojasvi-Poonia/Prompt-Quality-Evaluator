from typing import Dict, List, Optional

from evaluator import rule_based, semantic_structural, info_theoretic, graph_analysis
from evaluator.rubric import map_to_rubric


def _overall_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 75:
        return "Very Good"
    if score >= 60:
        return "Good"
    if score >= 45:
        return "Average"
    if score >= 25:
        return "Below Average"
    return "Poor"


def _flatten_pillar(prefix: str, results: dict) -> dict:
    flat = {}
    for key, val in results.items():
        if isinstance(val, dict) and "score" in val:
            flat[f"{prefix}.{key}"] = val
    return flat


def evaluate(
    text: str,
    dimension_weights: Optional[Dict[str, float]] = None,
) -> dict:
    p1 = rule_based.compute_all(text)
    p2 = semantic_structural.compute_all(text)
    p3 = info_theoretic.compute_all(text)
    pg = graph_analysis.compute_all(text)

    flat: Dict[str, dict] = {}
    flat.update(_flatten_pillar("rule_based", p1))
    flat.update(_flatten_pillar("semantic", p2))
    flat.update(_flatten_pillar("info", p3))
    flat.update(_flatten_pillar("graph", pg))

    rubric_dims = map_to_rubric(flat)

    def avg_scores(results: dict) -> float:
        scores = [v["score"] for v in results.values() if isinstance(v, dict) and "score" in v]
        return sum(scores) / len(scores) if scores else 0.0

    pillar_scores = {
        "rule_based": round(avg_scores(p1), 3),
        "semantic_structural": round(avg_scores(p2), 3),
        "info_theoretic": round(avg_scores(p3), 3),
        "graph_analysis": round(avg_scores(pg), 3),
    }

    if dimension_weights is None:
        dim_scores = [d["score"] for d in rubric_dims]
        final_01 = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0
    else:
        weighted_sum = 0.0
        total_weight = 0.0
        for d in rubric_dims:
            w = dimension_weights.get(d["name"], 1.0)
            weighted_sum += d["score"] * w
            total_weight += w
        final_01 = weighted_sum / total_weight if total_weight > 0 else 0.0

    final_score = round(final_01 * 100, 1)

    raw_metrics = {}
    for pillar_name, pillar_data in [
        ("rule_based", p1),
        ("semantic_structural", p2),
        ("info_theoretic", p3),
        ("graph_analysis", pg),
    ]:
        for metric_name, metric_data in pillar_data.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                raw_metrics[f"{pillar_name}.{metric_name}"] = {
                    "score": metric_data["score"],
                    "findings": metric_data.get("findings", []),
                }

    sorted_dims = sorted(rubric_dims, key=lambda d: d["score"])
    suggestions = []
    for d in sorted_dims[:3]:
        suggestions.append(f"Improve **{d['name']}** (score: {d['score']:.2f}): {d['findings'][0] if d['findings'] else 'No specific finding.'}")

    return {
        "final_score": final_score,
        "label": _overall_label(final_score),
        "rubric_dimensions": rubric_dims,
        "raw_metrics": raw_metrics,
        "pillar_scores": pillar_scores,
        "suggestions": suggestions,
    }
