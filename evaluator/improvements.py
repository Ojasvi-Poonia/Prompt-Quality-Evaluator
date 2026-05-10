from typing import Dict, List, Tuple


SUGGESTION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "rule_based.length": {
        "low": (
            "Your prompt is very short. Add at minimum:\n"
            "  - one sentence of context (what you're working on)\n"
            "  - one sentence describing the desired output\n"
            "  - one constraint (format, length, or audience)"
        ),
        "high": (
            "Your prompt may be too long. Try splitting it into two sequential prompts, "
            "or removing redundant context."
        ),
    },
    "rule_based.grammar": {
        "low": "Fix flagged grammar/spelling issues (see findings). Even one typo can confuse the model.",
    },
    "rule_based.specificity": {
        "low": (
            "Add concrete details. Examples to include:\n"
            "  - specific tool/library versions (e.g., 'Python 3.11', 'Django 4.2')\n"
            "  - exact data shapes (field names, types, sample values)\n"
            "  - quantitative thresholds ('must complete in <100ms')"
        ),
    },
    "rule_based.structure": {
        "low": (
            "Restructure your prompt with clear sections. Suggested template:\n\n"
            "  Context:\n  <background>\n\n"
            "  Task:\n  <imperative instruction>\n\n"
            "  Requirements:\n  - <requirement 1>\n  - <requirement 2>\n\n"
            "  Constraints:\n  - <thing to avoid>\n  - <format spec>"
        ),
    },
    "rule_based.task_signal": {
        "low": (
            "Open with a strong imperative verb. Replace 'I want' / 'Could you' with one of:\n"
            "  Write, Implement, Generate, Analyze, Design, Compare, Translate, Summarize, "
            "Evaluate, Refactor, Debug, Classify, Extract."
        ),
    },
    "rule_based.constraint": {
        "low": (
            "Add explicit constraints. Common patterns:\n"
            "  - Negations: 'Do not use external libraries.'\n"
            "  - Limits: 'Maximum 200 lines.' / 'Within 500 words.'\n"
            "  - Format: 'Output as JSON.' / 'As a markdown table.'\n"
            "  - Audience: 'For an experienced backend engineer.'\n"
            "  - Tone: 'Use a formal, technical tone.'"
        ),
    },
    "rule_based.contradiction": {
        "low": "Resolve detected contradictions in your constraints (see findings).",
    },
    "rule_based.persona_signal": {
        "low": (
            "Add an explicit persona at the start. Examples:\n"
            "  'You are a senior backend engineer with 10 years of Django experience.'\n"
            "  'Act as a professional technical writer.'\n"
            "  'Imagine you are explaining this to a curious 12-year-old.'"
        ),
    },
    "rule_based.few_shot_quality": {
        "low": (
            "Add 2-3 few-shot examples using this pattern:\n\n"
            "  Input: <example input 1>\n  Output: <example output 1>\n\n"
            "  Input: <example input 2>\n  Output: <example output 2>"
        ),
    },
    "rule_based.hedging": {
        "low": (
            "Remove hedging language. Replace:\n"
            "  'I kind of want maybe a function...' -> 'Write a function...'\n"
            "  'Could you possibly...' -> 'Do this:'\n"
            "  'I think it should...' -> 'It must:'"
        ),
    },
    "rule_based.safety": {
        "low": (
            "WARNING: Your prompt contains injection patterns ('Ignore previous instructions', etc). "
            "Remove these unless intentional -- they often indicate sloppy prompt chaining."
        ),
    },
    "rule_based.output_schema": {
        "low": (
            "Specify the output schema explicitly. Add a fenced code block like:\n\n"
            "  ```json\n"
            "  {\n"
            "    \"id\": <integer>,\n"
            "    \"name\": \"<string>\",\n"
            "    \"tags\": [\"<string>\"]\n"
            "  }\n"
            "  ```"
        ),
    },
    "rule_based.imperative_strength": {
        "low": (
            "Strengthen the imperative. Remove 'please', 'could you', 'would you mind'. "
            "Start sentences with the verb directly."
        ),
    },
    "rule_based.compound_task_count": {
        "high": "Your prompt asks for too many things at once. Split into 2-3 smaller prompts for cleaner output.",
    },
    "semantic.coherence": {
        "low": (
            "Sentences feel disconnected. Add transitional phrases ('First...', 'Then...', "
            "'However...') and ensure consecutive sentences share concrete subjects."
        ),
    },
    "semantic.semantic_density": {
        "low": (
            "Cut filler. Remove phrases like 'in order to', 'the fact that', 'it is important to note'. "
            "Use direct nouns and verbs."
        ),
    },
    "semantic.topic_focus": {
        "low": (
            "Your prompt covers too many topics. Either focus on one main task, "
            "or explicitly section the prompt with headers per topic."
        ),
    },
    "semantic.lexical_sophistication": {
        "low": (
            "Use more precise terminology. Replace generic words ('thing', 'stuff', 'do') with "
            "domain-specific terms ('record', 'transaction', 'compute')."
        ),
    },
    "semantic.sentence_flow": {
        "low": (
            "Vary sentence lengths. A mix of short and medium sentences reads more naturally "
            "than uniform-length text."
        ),
    },
    "semantic.reference_resolution": {
        "low": (
            "Replace ambiguous pronouns ('it', 'this', 'that') with the explicit nouns they refer to. "
            "The model can't always trace pronoun chains."
        ),
    },
    "semantic.tone_consistency": {
        "low": "Pick one tone (formal OR casual) and stay consistent throughout the prompt.",
    },
    "semantic.objectivity": {
        "low": (
            "Reduce subjective language. Replace 'I think this should be cool' with "
            "'The output must include X, Y, Z'."
        ),
    },
    "semantic.syntactic_complexity": {
        "low": "Sentences are too simple. Add subordinate clauses to express dependencies and conditions.",
        "high": "Sentences are too nested. Break complex sentences into shorter independent clauses.",
    },
    "semantic.discourse_markers": {
        "low": (
            "Add logical connectors: 'First', 'Then', 'However', 'Therefore', 'For example', "
            "'Finally'. They signal flow to the model."
        ),
    },
    "info.redundancy": {
        "low": "Tokens repeat too much. Vary your word choice and remove duplicate phrases.",
    },
    "info.ngram_repetition": {
        "low": "Same phrases repeat. Rephrase repeated chunks of 2-3 words.",
    },
    "info.sentence_repetition": {
        "low": (
            "You have sentences that repeat the same idea. Pick the strongest one and "
            "delete the duplicates."
        ),
    },
    "info.shannon_entropy": {
        "low": (
            "Token diversity is low. The prompt may be repetitive or use a narrow vocabulary. "
            "Vary word choice and add context."
        ),
    },
    "info.vocabulary_richness": {
        "low": (
            "Your vocabulary is limited. Use a wider range of nouns, verbs, and adjectives -- "
            "even if synonyms, variety helps."
        ),
    },
    "graph.has_context": {
        "low": (
            "Add a Context section explaining the situation, system, or background.\n\n"
            "  Context:\n"
            "  I am building <X>. Currently we have <Y>. We need to <Z>."
        ),
    },
    "graph.has_task": {
        "low": (
            "Add a clear Task section with an imperative.\n\n"
            "  Task:\n"
            "  Write/Implement/Analyze <specific deliverable>."
        ),
    },
    "graph.has_constraint": {
        "low": (
            "Add a Constraints section.\n\n"
            "  Constraints:\n"
            "  - Do not use <X>\n"
            "  - Maximum <Y>\n"
            "  - Format as <Z>"
        ),
    },
    "graph.completeness": {
        "low": "Ensure your prompt has all four sections: Context, Task, Constraints, Examples.",
    },
    "graph.information_flow": {
        "low": "Reorder so that Context comes first, then Task, then Constraints, then Examples.",
    },
    "graph.connectivity": {
        "low": "Sentences are disconnected. Repeat key entities (project name, target, format) across sentences to tie them together.",
    },
    "graph.centrality": {
        "low": (
            "No clear focal sentence. Make one sentence the obvious 'main task' by "
            "leading with it or making it standalone."
        ),
    },
}


LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.85


def actionable_suggestions(
    raw_metrics: Dict[str, Dict],
    rubric_dimensions: List[Dict],
    max_suggestions: int = 5,
) -> List[str]:
    suggestions: List[str] = []
    seen_keys = set()

    sorted_metrics = sorted(
        raw_metrics.items(),
        key=lambda kv: kv[1].get("score", 0.5)
    )

    for metric_key, metric in sorted_metrics:
        if len(suggestions) >= max_suggestions:
            break
        score = metric.get("score", 0.5)
        templates = SUGGESTION_TEMPLATES.get(metric_key, {})
        if not templates:
            continue

        if score < LOW_THRESHOLD and "low" in templates:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            display_name = metric_key.split(".")[-1].replace("_", " ").title()
            suggestions.append(
                f"[{display_name}] (score {score:.2f})\n  {templates['low']}"
            )
        elif score > HIGH_THRESHOLD and "high" in templates:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            display_name = metric_key.split(".")[-1].replace("_", " ").title()
            suggestions.append(
                f"[{display_name}] (score {score:.2f})\n  {templates['high']}"
            )

    return suggestions


def build_rewrite_template(
    raw_metrics: Dict[str, Dict],
    rubric_dimensions: List[Dict],
    detected_domain: str,
    original_text: str,
) -> str:
    sections: List[Tuple[str, str]] = []

    has_context = raw_metrics.get("graph.has_context", {}).get("score", 0.0) >= 0.5
    has_task = raw_metrics.get("graph.has_task", {}).get("score", 0.0) >= 0.5
    has_constraint = raw_metrics.get("graph.has_constraint", {}).get("score", 0.0) >= 0.5
    has_persona = raw_metrics.get("rule_based.persona_signal", {}).get("score", 0.0) >= 0.6
    has_schema = raw_metrics.get("rule_based.output_schema", {}).get("score", 0.0) >= 0.6
    has_few_shot = raw_metrics.get("rule_based.few_shot_quality", {}).get("score", 0.0) >= 0.6

    if not has_persona:
        if detected_domain == "coding":
            sections.append(("Persona", "You are a senior software engineer with deep expertise in production systems."))
        elif detected_domain == "creative":
            sections.append(("Persona", "You are an experienced creative writer with a distinctive voice."))
        elif detected_domain == "analysis":
            sections.append(("Persona", "You are a meticulous data analyst skilled at identifying patterns."))
        elif detected_domain == "instructional":
            sections.append(("Persona", "You are a patient teacher who explains concepts clearly."))
        else:
            sections.append(("Persona", "You are an expert in this domain."))

    if not has_context:
        sections.append((
            "Context",
            "<Describe the situation, the system you're working in, and any relevant background.>"
        ))

    if not has_task:
        sections.append((
            "Task",
            "<State the specific instruction with an imperative verb (Write/Implement/Analyze/etc).>"
        ))
    else:
        sections.append(("Task (from your prompt)", original_text.strip()[:300]))

    if not has_constraint:
        sections.append((
            "Constraints",
            "- Do not use <forbidden libraries/approaches>\n"
            "- Maximum <length/complexity limit>\n"
            "- Format as <JSON/markdown/code>\n"
            "- Audience: <who will read this>"
        ))

    if not has_schema and detected_domain == "coding":
        sections.append((
            "Output Schema",
            "```json\n{\n  \"<field>\": \"<type>\"\n}\n```"
        ))

    if not has_few_shot:
        sections.append((
            "Examples (optional but recommended)",
            "Input: <example input>\nOutput: <expected output>\n\n"
            "Input: <another example>\nOutput: <expected output>"
        ))

    if not sections:
        return (
            "Your prompt already has the major sections.\n"
            "Tighten weakest dimensions: see actionable suggestions."
        )

    output = "REWRITE TEMPLATE\n" + "=" * 60 + "\n\n"
    for header, body in sections:
        output += f"### {header}\n{body}\n\n"
    output += "=" * 60 + "\n"
    output += "Replace each <placeholder> with your specifics.\n"
    return output
