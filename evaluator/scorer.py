import random
from typing import Dict, List, Optional

from evaluator import rule_based, semantic_structural, info_theoretic, graph_analysis
from evaluator.rubric import map_to_rubric, detect_domain, apply_domain_weights
from evaluator.utils import word_count, split_sentences


TASK_PENALTY_THRESHOLD = 0.2
TASK_PENALTY_CAP_TIERS = [
    (5, 12.0),
    (10, 22.0),
    (20, 32.0),
    (float("inf"), 40.0),
]


def _task_penalty_cap(text: str) -> float:
    wc = word_count(text)
    for limit, cap in TASK_PENALTY_CAP_TIERS:
        if wc <= limit:
            return cap
    return 40.0


def _length_quality_factor(text: str) -> float:
    wc = word_count(text)
    if wc >= 30:
        return 1.0
    if wc <= 3:
        return 0.4
    if wc <= 10:
        return 0.6 + (wc - 3) * 0.04
    return 0.85 + (wc - 10) * 0.0075


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
    domain_override: Optional[str] = None,
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

    domain, domain_confidence = (domain_override, 1.0) if domain_override else detect_domain(text)
    if domain != "general":
        domain_weights = apply_domain_weights(rubric_dims, domain)
    else:
        domain_weights = {d["name"]: 1.0 for d in rubric_dims}

    def avg_scores(results: dict) -> float:
        scores = [v["score"] for v in results.values() if isinstance(v, dict) and "score" in v]
        return sum(scores) / len(scores) if scores else 0.0

    pillar_scores = {
        "rule_based": round(avg_scores(p1), 3),
        "semantic_structural": round(avg_scores(p2), 3),
        "info_theoretic": round(avg_scores(p3), 3),
        "graph_analysis": round(avg_scores(pg), 3),
    }

    effective_weights = dimension_weights if dimension_weights else domain_weights

    weighted_sum = 0.0
    total_weight = 0.0
    for d in rubric_dims:
        w = effective_weights.get(d["name"], 1.0)
        weighted_sum += d["score"] * w
        total_weight += w
    final_01 = weighted_sum / total_weight if total_weight > 0 else 0.0

    length_factor = _length_quality_factor(text)
    if length_factor < 1.0:
        final_01 *= length_factor

    pillar_values = list(pillar_scores.values())
    excellence_bonus = 0.0
    if pillar_values and min(pillar_values) >= 0.6:
        avg_pillar = sum(pillar_values) / len(pillar_values)
        excellence_bonus = (avg_pillar - 0.6) * 0.25
        final_01 = min(1.0, final_01 + excellence_bonus)

    final_score = round(final_01 * 100, 1)

    task_signal_score = flat.get("rule_based.task_signal", {}).get("score", 0.0)
    has_task_score = flat.get("graph.has_task", {}).get("score", 0.0)
    task_penalty_applied = False
    task_penalty_cap = _task_penalty_cap(text)

    if task_signal_score < TASK_PENALTY_THRESHOLD and has_task_score < 0.5:
        if final_score > task_penalty_cap:
            final_score = task_penalty_cap
            task_penalty_applied = True

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

    from evaluator.improvements import actionable_suggestions
    suggestions = actionable_suggestions(raw_metrics, rubric_dims, max_suggestions=5)
    if not suggestions:
        sorted_dims = sorted(rubric_dims, key=lambda d: d["score"])
        for d in sorted_dims[:3]:
            suggestions.append(
                f"Improve **{d['name']}** (score: {d['score']:.2f}): "
                f"{d['findings'][0] if d['findings'] else 'No specific finding.'}"
            )

    if task_penalty_applied:
        suggestions.insert(
            0,
            f"No clear task instruction detected -- final score capped at {task_penalty_cap}. "
            "Add an imperative verb (write/explain/analyze) or a direct question."
        )

    if length_factor < 1.0:
        wc = word_count(text)
        suggestions.insert(
            0,
            f"Prompt is very short ({wc} words) -- score reduced by length factor {length_factor:.2f}. "
            "Short prompts cannot demonstrate clarity, structure, or context."
        )

    return {
        "final_score": final_score,
        "label": _overall_label(final_score),
        "rubric_dimensions": rubric_dims,
        "raw_metrics": raw_metrics,
        "pillar_scores": pillar_scores,
        "suggestions": suggestions,
        "detected_domain": domain,
        "domain_confidence": round(domain_confidence, 3),
        "task_penalty_applied": task_penalty_applied,
        "task_penalty_cap": task_penalty_cap,
        "length_factor": round(length_factor, 3),
        "excellence_bonus": round(excellence_bonus, 3),
    }


def evaluate_with_confidence(
    text: str,
    n_bootstrap: int = 30,
    seed: int = 42,
) -> dict:
    main_result = evaluate(text)
    sentences = split_sentences(text)

    if len(sentences) < 3:
        main_result["confidence_interval_95"] = [main_result["final_score"], main_result["final_score"]]
        main_result["score_std"] = 0.0
        main_result["bootstrap_samples"] = 0
        return main_result

    rng = random.Random(seed)
    n = len(sentences)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        sample_idxs = sorted({rng.randint(0, n - 1) for _ in range(n)})
        if len(sample_idxs) < 2:
            continue
        resampled = " ".join(sentences[i] for i in sample_idxs)
        try:
            r = evaluate(resampled)
            bootstrap_scores.append(r["final_score"])
        except Exception:
            continue

    if len(bootstrap_scores) >= 5:
        bootstrap_scores.sort()
        mean = sum(bootstrap_scores) / len(bootstrap_scores)
        variance = sum((s - mean) ** 2 for s in bootstrap_scores) / len(bootstrap_scores)
        std = variance ** 0.5

        lo_idx = max(0, int(0.025 * len(bootstrap_scores)))
        hi_idx = min(len(bootstrap_scores) - 1, int(0.975 * len(bootstrap_scores)))
        ci = [round(bootstrap_scores[lo_idx], 1), round(bootstrap_scores[hi_idx], 1)]

        main_result["confidence_interval_95"] = ci
        main_result["score_std"] = round(std, 2)
        main_result["bootstrap_samples"] = len(bootstrap_scores)
    else:
        main_result["confidence_interval_95"] = [main_result["final_score"], main_result["final_score"]]
        main_result["score_std"] = 0.0
        main_result["bootstrap_samples"] = len(bootstrap_scores)

    return main_result


def ablation_analysis(text: str) -> dict:
    full_result = evaluate(text)
    full_score = full_result["final_score"]

    contributions = {}
    rubric_dims = full_result["rubric_dimensions"]
    for d in rubric_dims:
        weighted = d["score"] / 8 * 100
        contributions[d["name"]] = {
            "score": d["score"],
            "weighted_contribution_to_final": round(weighted, 1),
        }

    return {
        "final_score": full_score,
        "dimension_contributions": contributions,
        "lowest_dimensions": sorted(rubric_dims, key=lambda x: x["score"])[:3],
        "highest_dimensions": sorted(rubric_dims, key=lambda x: -x["score"])[:3],
    }
