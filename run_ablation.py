#!/usr/bin/env python3
"""Pillar ablation: neutralise each pillar's metrics to 0.5 and rescore.

Reads the stored Gemini judge scores from benchmark_real_results.json and
recomputes the Pearson correlation with each pillar knocked out in turn.
Runs entirely offline: no API key, no network.

The aggregation logic here is imported from evaluator.scorer rather than
reimplemented, so the "full pipeline" baseline is by construction the same
scorer the paper describes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "benchmark_real_results.json"

sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import graph_analysis, info_theoretic, rule_based, semantic_structural  # noqa: E402
from evaluator.rubric import apply_domain_weights, detect_domain, map_to_rubric  # noqa: E402
from evaluator.scorer import (  # noqa: E402
    TASK_PENALTY_THRESHOLD,
    _length_quality_factor,
    _task_penalty_cap,
)

PILLARS = ["rule_based", "semantic", "info", "graph"]


def _neutralise(pillar: dict) -> dict:
    """Return a copy of the pillar with every metric score set to 0.5."""
    return {
        k: ({**v, "score": 0.5} if isinstance(v, dict) and "score" in v else v)
        for k, v in pillar.items()
    }


def ablated_evaluate(text: str, ablate: str | None = None) -> float:
    """Score a prompt with one pillar neutralised.

    With ablate=None this delegates straight to the shipped scorer, so the
    baseline row is the real pipeline by construction and cannot drift from it.
    """
    if ablate is None:
        from evaluator.scorer import evaluate

        return float(evaluate(text)["final_score"])

    pillars = {
        "rule_based": rule_based.compute_all(text),
        "semantic": semantic_structural.compute_all(text),
        "info": info_theoretic.compute_all(text),
        "graph": graph_analysis.compute_all(text),
    }
    if ablate is not None:
        pillars[ablate] = _neutralise(pillars[ablate])

    flat = {
        f"{name}.{k}": v
        for name, pillar in pillars.items()
        for k, v in pillar.items()
        if isinstance(v, dict) and "score" in v
    }

    rubric_dims = map_to_rubric(flat)
    domain, _ = detect_domain(text)
    weights = (
        apply_domain_weights(rubric_dims, domain)
        if domain != "general"
        else {d["name"]: 1.0 for d in rubric_dims}
    )

    weighted_sum = sum(d["score"] * weights.get(d["name"], 1.0) for d in rubric_dims)
    total_weight = sum(weights.get(d["name"], 1.0) for d in rubric_dims)
    final_01 = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Modifier 1: length factor
    length_factor = _length_quality_factor(text)
    if length_factor < 1.0:
        final_01 *= length_factor

    # Modifier 3: excellence bonus.
    # Pillar means are averaged with sum/len and rounded to 3dp, exactly as
    # evaluator.scorer does. Using np.mean here tips the rounding on some
    # prompts and shifts the final score by 0.1.
    def _mean(pillar: dict) -> float:
        scores = [v["score"] for v in pillar.values() if isinstance(v, dict) and "score" in v]
        return sum(scores) / len(scores) if scores else 0.0

    pillar_means = [round(_mean(p), 3) for p in pillars.values()]
    if pillar_means and min(pillar_means) >= 0.6:
        final_01 = min(1.0, final_01 + (sum(pillar_means) / len(pillar_means) - 0.6) * 0.25)

    final_score = round(final_01 * 100, 1)

    # Modifier 2: task penalty cap
    task_signal = flat.get("rule_based.task_signal", {}).get("score", 0.0)
    has_task = flat.get("graph.has_task", {}).get("score", 0.0)
    if task_signal < TASK_PENALTY_THRESHOLD and has_task < 0.5:
        final_score = min(final_score, _task_penalty_cap(text))

    return final_score


def main() -> None:
    if not RESULTS_PATH.exists():
        sys.exit(f"{RESULTS_PATH} not found. Run: python run_benchmark.py")

    with open(RESULTS_PATH) as fh:
        results = json.load(fh)

    llm_scores = np.array([r["llm_score"] for r in results])
    prompts = [r["prompt"] for r in results]

    baseline = np.array([ablated_evaluate(p) for p in prompts])
    r_base, _ = pearsonr(baseline, llm_scores)

    print("PILLAR ABLATION (Pearson r with the Gemini judge)")
    print("=" * 62)
    print(f"{'Full pipeline (baseline)':<34}: r = {r_base:.4f}")

    for pillar in PILLARS:
        scores = np.array([ablated_evaluate(p, pillar) for p in prompts])
        r, _ = pearsonr(scores, llm_scores)
        print(f"{'Without Pillar [' + pillar + ']':<34}: r = {r:.4f}   delta = {r - r_base:+.4f}")

    # Cross-check: the baseline above must equal the shipped scorer exactly.
    from evaluator.scorer import evaluate as full_evaluate

    shipped = np.array([full_evaluate(p)["final_score"] for p in prompts])
    r_shipped, _ = pearsonr(shipped, llm_scores)
    print()
    print(f"Cross-check, evaluator.scorer.evaluate : r = {r_shipped:.4f}")
    if np.allclose(shipped, baseline):
        print("Baseline matches the shipped scorer exactly.")
    else:
        n = int(np.sum(~np.isclose(shipped, baseline)))
        print(f"WARNING: baseline differs from the shipped scorer on {n} prompt(s).")


if __name__ == "__main__":
    main()
