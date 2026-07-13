#!/usr/bin/env python3
"""Benchmark the classical scorer against the Gemini LLM judge.

Two modes:

  --offline   Recompute classical scores with the current code and reuse the
              Gemini scores already stored in benchmark_real_results.json.
              Needs no API key and no network. This is the mode that
              reproduces every table in the paper.

  (default)   Full live run: score all prompts with both the classical
              pipeline and the Gemini judge, then overwrite
              benchmark_real_results.json.

The live run needs a Gemini API key. Put it in .env (copy .env.example), or
export it:

    export GEMINI_API_KEY=...            # required
    export GEMINI_API_KEY_BACKUP=...     # optional, used if the first hits quota

It also needs the judge's own dependencies, which are deliberately kept out of
requirements.txt so that installing the scorer never pulls in an LLM SDK:

    pip install -r requirements-benchmark.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "benchmark_real_results.json"
PROMPTS_PATH = PROJECT_ROOT / "data" / "sample_prompts.json"

DIMENSIONS = [
    "Clarity",
    "Specificity",
    "Structure",
    "Task Definition",
    "Context Completeness",
    "Constraint Specification",
    "Information Richness",
    "Conciseness",
]
TIERS = ["terrible", "poor", "average", "good", "excellent"]
LABEL_ORDER = ["Poor", "Below Average", "Average", "Good", "Very Good", "Excellent"]

sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.scorer import evaluate as classical_evaluate  # noqa: E402


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ. No dependency needed.

    Existing environment variables win, so an explicit `export` still overrides
    the file. Missing .env is fine: the offline path needs no keys at all.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def gemini_keys() -> list[str]:
    """Read Gemini API keys from the environment. Never hardcode keys here."""
    load_dotenv()
    keys = [
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_BACKUP"),
    ]
    keys = [k for k in keys if k and not k.startswith("your-")]
    if not keys:
        sys.exit(
            "No Gemini API key found.\n"
            "Copy .env.example to .env and fill in your key, or export GEMINI_API_KEY.\n"
            "To reproduce the paper's tables you need no key at all:\n"
            "    python run_benchmark.py --offline"
        )
    return keys


def run_llm_with_fallback(text: str, keys: list[str], retries: int = 3) -> dict[str, Any]:
    """Score one prompt with the Gemini judge, rotating keys on quota errors."""
    from llm_evaluator.judge import LLMAPIError, LLMQuotaError
    from llm_evaluator.judge import evaluate as llm_evaluate

    last_error: Exception | None = None
    for key in keys:
        os.environ["GEMINI_API_KEY"] = key
        for attempt in range(retries):
            try:
                return llm_evaluate(text, provider="gemini")
            except LLMQuotaError as exc:
                last_error = exc
                print(f"    Quota exhausted on this key, attempt {attempt + 1}")
                time.sleep(2)
                break
            except LLMAPIError as exc:
                last_error = exc
                print(f"    API error, retrying in 5s: {exc}")
                time.sleep(5)
    raise RuntimeError(f"All Gemini keys exhausted: {last_error}")


def run_live() -> list[dict[str, Any]]:
    """Score every prompt with both systems. Requires an API key and network."""
    keys = gemini_keys()
    with open(PROMPTS_PATH) as fh:
        prompts = json.load(fh)
    print(f"Loaded {len(prompts)} prompts")

    results: list[dict[str, Any]] = []
    for i, entry in enumerate(prompts, 1):
        text, tier = entry["prompt"], entry["tier"]
        preview = text[:60] + ("..." if len(text) > 60 else "")
        print(f"[{i:2d}/{len(prompts)}] {tier:10s}: {preview}")

        classical = classical_evaluate(text)
        print(
            f"    classical: {classical['final_score']:.1f} ({classical['label']}) "
            f"domain={classical.get('detected_domain', 'general')}"
        )

        try:
            llm = run_llm_with_fallback(text, keys)
            print(f"    llm:       {llm['final_score']:.1f} ({llm['label']})")
        except Exception as exc:  # noqa: BLE001 - one bad prompt should not kill the run
            print(f"    LLM FAILED: {exc}")
            continue

        results.append(
            {
                "id": i,
                "tier": tier,
                "prompt": text,
                "preview": text[:80] + ("..." if len(text) > 80 else ""),
                "classical_score": classical["final_score"],
                "classical_label": classical["label"],
                "classical_domain": classical.get("detected_domain", "general"),
                "llm_score": llm["final_score"],
                "llm_label": llm["label"],
                "classical_dims": {d["name"]: d["score"] for d in classical["rubric_dimensions"]},
                "llm_dims": {d["name"]: d["score"] for d in llm["rubric_dimensions"]},
            }
        )
        time.sleep(0.5)

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {len(results)} results to {RESULTS_PATH}\n")
    return results


def run_offline(write: bool = False) -> list[dict[str, Any]]:
    """Rescore prompts with the current classical code, reusing stored judge scores."""
    if not RESULTS_PATH.exists():
        sys.exit(f"{RESULTS_PATH} not found. A live run is needed to create it.")

    with open(RESULTS_PATH) as fh:
        stored = json.load(fh)
    print(f"Loaded {len(stored)} stored results (reusing their Gemini scores)\n")

    results: list[dict[str, Any]] = []
    drifted = 0
    for entry in stored:
        classical = classical_evaluate(entry["prompt"])
        if abs(classical["final_score"] - entry["classical_score"]) > 1e-9:
            drifted += 1
            print(
                f"  #{entry['id']:>2} {entry['tier']:<9} "
                f"stored={entry['classical_score']:>5.1f} -> now={classical['final_score']:>5.1f}"
            )
        results.append(
            {
                **entry,
                "classical_score": classical["final_score"],
                "classical_label": classical["label"],
                "classical_domain": classical.get("detected_domain", "general"),
                "classical_dims": {d["name"]: d["score"] for d in classical["rubric_dimensions"]},
            }
        )

    if drifted:
        print(
            f"\n{drifted}/{len(stored)} classical scores differ from the stored run.\n"
            "The stored file was produced by a different code or dependency state.\n"
        )
        if write:
            with open(RESULTS_PATH, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"Rewrote {RESULTS_PATH} with the current classical scores.\n")
    else:
        print("All classical scores reproduce the stored run exactly.\n")
    return results


def report(results: list[dict[str, Any]]) -> None:
    """Print every statistic the paper reports."""
    classical = np.array([r["classical_score"] for r in results])
    llm = np.array([r["llm_score"] for r in results])

    print("=" * 70)
    print("OVERALL CORRELATIONS")
    print("=" * 70)
    r_p, p_p = pearsonr(classical, llm)
    r_s, p_s = spearmanr(classical, llm)
    print(f"Pearson r  = {r_p:.4f}  (p = {p_p:.2e})")
    print(f"Spearman r = {r_s:.4f}  (p = {p_s:.2e})")
    print(f"Mean absolute difference: {np.mean(np.abs(classical - llm)):.1f}")

    print("\n" + "=" * 70)
    print("PER-TIER MEANS")
    print("=" * 70)
    for tier in TIERS:
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        c = [float(r["classical_score"]) for r in rows]
        l = [float(r["llm_score"]) for r in rows]
        print(f"{tier:10s}: classical = {np.mean(c):5.1f}  llm = {np.mean(l):5.1f}  (n={len(c)})")
        print(f"          : classical scores = {[round(x, 1) for x in c]}")
        print(f"          : llm scores       = {[round(x, 1) for x in l]}")

    print("\n" + "=" * 70)
    print("PER-DIMENSION CORRELATIONS")
    print("=" * 70)
    for dim in DIMENSIONS:
        c_vals = np.array([r["classical_dims"].get(dim, 0) for r in results])
        l_vals = np.array([r["llm_dims"].get(dim, 0) for r in results])
        if len(set(c_vals.tolist())) > 1 and len(set(l_vals.tolist())) > 1:
            r, p = pearsonr(c_vals, l_vals)
            strength = "strong" if abs(r) >= 0.6 else ("moderate" if abs(r) >= 0.3 else "weak")
            print(f"{dim:25s}: r = {r:6.3f}  (p = {p:.4f})  {strength}")

    print("\n" + "=" * 70)
    print("LABEL AGREEMENT")
    print("=" * 70)
    exact = sum(1 for r in results if r["classical_label"] == r["llm_label"])
    print(f"Exact label match: {exact}/{len(results)} ({100 * exact / len(results):.0f}%)")

    idx = {label: i for i, label in enumerate(LABEL_ORDER)}
    within_one = sum(
        1
        for r in results
        if abs(idx.get(r["classical_label"], -1) - idx.get(r["llm_label"], -1)) <= 1
    )
    print(f"Within one band: {within_one}/{len(results)} ({100 * within_one / len(results):.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Reuse stored Gemini scores. No API key or network needed.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="With --offline, persist the recomputed classical scores back to the results file.",
    )
    args = parser.parse_args()
    report(run_offline(write=args.write) if args.offline else run_live())


if __name__ == "__main__":
    main()
