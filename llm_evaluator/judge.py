import os
import json
import hashlib
import re
from pathlib import Path
from typing import Optional

from llm_evaluator.template import JUDGE_INSTRUCTIONS, RUBRIC_DIMENSIONS, RAW_METRICS


CACHE_DIR = Path.home() / ".cache" / "prompt-llm-evaluator"


class LLMQuotaError(RuntimeError):
    pass


class LLMAPIError(RuntimeError):
    pass


def _cache_key(provider: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{provider}|{model}|{text}".encode()).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cached(key: str) -> Optional[dict]:
    path = _cache_path(key)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_cached(key: str, result: dict) -> None:
    _cache_path(key).write_text(json.dumps(result, indent=2))


def _extract_json(text: str) -> Optional[dict]:
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _call_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return response.text


def _call_grok(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set. Get a key at https://console.x.ai")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _call_provider(provider: str, prompt: str, model: Optional[str]) -> str:
    try:
        if provider == "gemini":
            return _call_gemini(prompt, model or "gemini-2.5-flash-lite")
        if provider == "grok":
            return _call_grok(prompt, model or "grok-2-1212")
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'grok'.")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            raise LLMQuotaError(
                f"{provider} API quota exceeded. "
                "Free tier has rate/daily limits. Wait or switch model with --model."
            ) from e
        if "401" in msg or "403" in msg or ("invalid" in msg.lower() and "key" in msg.lower()):
            raise LLMAPIError(
                f"{provider} API rejected the key. Verify the key and that the API is enabled."
            ) from e
        if isinstance(e, RuntimeError) and "API_KEY" in msg:
            raise
        raise LLMAPIError(f"{provider} API call failed: {msg[:200]}") from e


def _label_for_overall(score: float) -> str:
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


def _label_for_dimension(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Strong"
    if score >= 0.50:
        return "Good"
    if score >= 0.30:
        return "Fair"
    return "Weak"


def _clamp(v, lo=0.0, hi=1.0):
    try:
        v = float(v)
    except (ValueError, TypeError):
        v = 0.0
    return max(lo, min(hi, v))


def _normalise_response(raw: dict, model: str, provider: str) -> dict:
    raw_metrics = {}
    for pillar, names in RAW_METRICS.items():
        for name in names:
            key = f"{pillar}.{name}"
            entry = raw.get("raw_metrics", {}).get(key, {})
            score = _clamp(entry.get("score", 0.0))
            findings = entry.get("findings", [])
            if isinstance(findings, str):
                findings = [findings]
            raw_metrics[key] = {"score": round(score, 3), "findings": findings}

    rubric_dims = []
    raw_rubric = raw.get("rubric_dimensions", [])
    rubric_lookup = {d.get("name"): d for d in raw_rubric if isinstance(d, dict)}
    for name in RUBRIC_DIMENSIONS:
        d = rubric_lookup.get(name, {})
        score = _clamp(d.get("score", 0.0))
        findings = d.get("findings", [])
        if isinstance(findings, str):
            findings = [findings]
        rubric_dims.append({
            "name": name,
            "score": round(score, 3),
            "label": _label_for_dimension(score),
            "findings": findings,
        })

    pillar_scores = {}
    for pillar, names in RAW_METRICS.items():
        scores = [raw_metrics.get(f"{pillar}.{n}", {}).get("score", 0.0) for n in names]
        pillar_scores[
            "rule_based" if pillar == "rule_based"
            else "semantic_structural" if pillar == "semantic"
            else "info_theoretic" if pillar == "info"
            else "graph_analysis"
        ] = round(sum(scores) / len(scores), 3) if scores else 0.0

    final_score = raw.get("final_score")
    if final_score is None:
        avg = sum(d["score"] for d in rubric_dims) / max(len(rubric_dims), 1)
        final_score = avg * 100
    try:
        final_score = float(final_score)
    except (ValueError, TypeError):
        avg = sum(d["score"] for d in rubric_dims) / max(len(rubric_dims), 1)
        final_score = avg * 100
    final_score = max(0.0, min(100.0, final_score))

    label = raw.get("label") or _label_for_overall(final_score)

    suggestions = raw.get("suggestions") or []
    if not suggestions:
        sorted_dims = sorted(rubric_dims, key=lambda d: d["score"])
        for d in sorted_dims[:3]:
            finding = d["findings"][0] if d["findings"] else "No specific finding."
            suggestions.append(f"Improve **{d['name']}** (score: {d['score']:.2f}): {finding}")

    detected_domain = raw.get("detected_domain") or "general"

    return {
        "final_score": round(final_score, 1),
        "label": label,
        "rubric_dimensions": rubric_dims,
        "raw_metrics": raw_metrics,
        "pillar_scores": pillar_scores,
        "suggestions": suggestions,
        "detected_domain": detected_domain,
        "engine": f"llm:{provider}:{model}",
    }


def evaluate(
    text: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    use_cache: bool = True,
) -> dict:
    full_prompt = JUDGE_INSTRUCTIONS + text

    cache_key = _cache_key(provider, model or "default", text)
    if use_cache:
        cached = _load_cached(cache_key)
        if cached is not None:
            cached["from_cache"] = True
            return cached

    raw_response = _call_provider(provider, full_prompt, model)
    parsed = _extract_json(raw_response)

    if parsed is None:
        return {
            "final_score": 0.0,
            "label": "Error",
            "rubric_dimensions": [
                {"name": d, "score": 0.0, "label": "Weak", "findings": []}
                for d in RUBRIC_DIMENSIONS
            ],
            "raw_metrics": {},
            "pillar_scores": {},
            "suggestions": [],
            "detected_domain": "general",
            "engine": f"llm:{provider}:{model or 'default'}",
            "error": f"Failed to parse JSON from LLM response: {raw_response[:300]}",
            "from_cache": False,
        }

    result = _normalise_response(parsed, model or "default", provider)
    result["from_cache"] = False

    if use_cache:
        _save_cached(cache_key, result)

    return result
