import json
import os

import pytest

from evaluator.scorer import evaluate
from evaluator import rule_based, semantic_structural, info_theoretic, graph_analysis


SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prompts.json")


@pytest.fixture(scope="module")
def sample_prompts():
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tier_scores(sample_prompts):
    tiers = {}
    for entry in sample_prompts:
        tier = entry["tier"]
        result = evaluate(entry["prompt"])
        tiers.setdefault(tier, []).append(result["final_score"])
    return {tier: sum(scores) / len(scores) for tier, scores in tiers.items()}


GOOD_PROMPT = (
    "Write a Python function called `calculate_statistics` that takes a list of numbers "
    "and returns a dictionary with the mean, median, standard deviation, and mode.\n\n"
    "Requirements:\n"
    "- Handle edge cases: empty list, single element, all same values\n"
    "- Use only the standard library (no numpy or pandas)\n"
    "- Add type hints and a docstring\n"
    "- Time complexity should be O(n log n) or better\n\n"
    "Example:\n"
    "  calculate_statistics([1, 2, 3, 4, 5]) -> {'mean': 3.0, 'median': 3, 'std': 1.41, 'mode': None}"
)


class TestMonotonicity:

    def test_tier_ordering(self, tier_scores):
        tier_order = ["terrible", "poor", "average", "good", "excellent"]
        available = [t for t in tier_order if t in tier_scores]

        for i in range(len(available) - 1):
            lower = tier_scores[available[i]]
            upper = tier_scores[available[i + 1]]
            assert lower < upper, (
                f"Tier '{available[i]}' (avg={lower:.1f}) should score lower "
                f"than '{available[i + 1]}' (avg={upper:.1f})"
            )


class TestBoundaries:

    def test_empty_string(self):
        result = evaluate("")
        assert 0 <= result["final_score"] <= 20, f"Empty prompt scored {result['final_score']}"

    def test_single_word(self):
        result = evaluate("help")
        assert 0 <= result["final_score"] <= 40, f"Single word scored {result['final_score']}"

    def test_good_prompt_above_threshold(self):
        result = evaluate(GOOD_PROMPT)
        assert result["final_score"] > 55, f"Good prompt scored only {result['final_score']}"


class TestRuleBasedMetrics:

    def test_length_short(self):
        score, findings = rule_based.length_score("hi")
        assert score < 0.3

    def test_length_optimal(self):
        text = " ".join(["word"] * 150)
        score, _ = rule_based.length_score(text)
        assert score > 0.7

    def test_readability_returns_score(self):
        score, findings = rule_based.readability_score(
            "The quick brown fox jumps over the lazy dog. This is a simple sentence."
        )
        assert 0 <= score <= 1
        assert len(findings) > 0

    def test_specificity_with_entities(self):
        text = "Use PostgreSQL 14 on Ubuntu 22.04 with Django 4.2 to build the API."
        score, _ = rule_based.specificity_score(text)
        assert score > 0.1

    def test_structure_plain_text(self):
        score, _ = rule_based.structure_score("Just a plain sentence with nothing special.")
        assert score <= 0.3

    def test_structure_with_formatting(self):
        text = (
            "# Title\n\n"
            "- Point one with meaningful content describing requirements\n"
            "- Point two with specific details about the task\n\n"
            "Context: we are building a REST API with Django and PostgreSQL\n"
            "Task: write a serializer with CRUD operations and validation"
        )
        score, _ = rule_based.structure_score(text)
        assert score > 0.5

    def test_task_signal_with_imperatives(self):
        score, _ = rule_based.task_signal_score("Write a function that calculates the sum.")
        assert score > 0.3

    def test_task_signal_missing(self):
        score, _ = rule_based.task_signal_score("The sky is blue and water is wet.")
        assert score < 0.3

    def test_constraint_detection(self):
        text = "Do not use external libraries. Maximum 100 lines. Output in JSON format."
        score, _ = rule_based.constraint_score(text)
        assert score > 0.3


class TestSemanticMetrics:

    def test_coherence_single_sentence(self):
        score, _ = semantic_structural.coherence_score("Just one sentence here.")
        assert score == 0.5

    def test_coherence_related_sentences(self):
        text = (
            "Python is a programming language. "
            "Python supports multiple paradigms. "
            "Python is widely used in data science."
        )
        score, _ = semantic_structural.coherence_score(text)
        assert score > 0.1

    def test_semantic_density(self):
        score, _ = semantic_structural.semantic_density(
            "Implement a binary search tree with insert, delete, and traverse operations."
        )
        assert 0 <= score <= 1

    def test_topic_focus_single(self):
        score, _ = semantic_structural.topic_focus("One sentence only.")
        assert score == 0.5

    def test_sentence_flow_single(self):
        score, _ = semantic_structural.sentence_flow("Just one.")
        assert score == 0.5

    def test_reference_resolution(self):
        score, _ = semantic_structural.reference_resolution(
            "The database stores user records. It processes them quickly."
        )
        assert 0 <= score <= 1


class TestInfoTheoreticMetrics:

    def test_shannon_entropy_short(self):
        score, _ = info_theoretic.shannon_entropy("a")
        assert score == 0.0

    def test_shannon_entropy_diverse(self):
        text = (
            "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
            "Meanwhile the sun rose slowly above the distant mountains casting golden light. "
            "Birds chirped merrily while squirrels scurried through the ancient oak trees. "
            "A gentle breeze carried the scent of wildflowers across the peaceful meadow. "
            "Far away a church bell rang announcing the start of another beautiful day."
        )
        score, _ = info_theoretic.shannon_entropy(text)
        assert score > 0.5

    def test_redundancy_repetitive(self):
        text = "word word word word word word word word word word"
        score, _ = info_theoretic.redundancy_score(text)
        assert score < 0.5

    def test_vocabulary_richness(self):
        text = "Implement concurrent asynchronous processing with middleware orchestration."
        score, _ = info_theoretic.vocabulary_richness(text)
        assert 0 <= score <= 1

    def test_ngram_repetition(self):
        text = "the cat sat on the mat. the cat sat on the mat."
        score, _ = info_theoretic.ngram_repetition_penalty(text)
        assert score < 0.7

    def test_burstiness_short(self):
        score, _ = info_theoretic.burstiness("short")
        assert score == 0.3


class TestGraphAnalysis:

    def test_build_graph(self):
        text = (
            "I have a Django application. "
            "Write an authentication module. "
            "Do not use third-party libraries. "
            "For example, use JWT tokens."
        )
        G, nodes = graph_analysis.build_discourse_graph(text)
        assert G.number_of_nodes() == 4
        types = {n["type"] for n in nodes}
        assert "TASK" in types
        assert "CONSTRAINT" in types

    def test_completeness_all_present(self):
        nodes = [
            {"type": "CONTEXT", "id": 0},
            {"type": "TASK", "id": 1},
            {"type": "CONSTRAINT", "id": 2},
            {"type": "EXAMPLE", "id": 3},
        ]
        score, _ = graph_analysis.completeness_score(nodes)
        assert score >= 0.99

    def test_completeness_missing(self):
        nodes = [{"type": "TASK", "id": 0}]
        score, findings = graph_analysis.completeness_score(nodes)
        assert score < 0.5
        assert any("Missing" in f for f in findings)


class TestDeterminism:

    def test_deterministic(self):
        results = [evaluate(GOOD_PROMPT)["final_score"] for _ in range(3)]
        assert results[0] == results[1] == results[2], (
            f"Non-deterministic scores: {results}"
        )


class TestEdgeCases:

    def test_all_caps(self):
        result = evaluate("WRITE A FUNCTION THAT SORTS NUMBERS IN ASCENDING ORDER")
        assert 0 <= result["final_score"] <= 100

    def test_code_only(self):
        result = evaluate("def foo(x): return x * 2")
        assert 0 <= result["final_score"] <= 100

    def test_url_only(self):
        result = evaluate("https://example.com/api/v1/users?page=1&limit=10")
        assert 0 <= result["final_score"] <= 100

    def test_long_prompt(self):
        text = " ".join(["Write a detailed analysis."] * 500)
        result = evaluate(text)
        assert 0 <= result["final_score"] <= 100

    def test_special_characters(self):
        result = evaluate("!@#$%^&*() {} [] <> ??? ///")
        assert 0 <= result["final_score"] <= 100

    def test_result_structure(self):
        result = evaluate(GOOD_PROMPT)
        assert "final_score" in result
        assert "label" in result
        assert "rubric_dimensions" in result
        assert "raw_metrics" in result
        assert "pillar_scores" in result
        assert len(result["rubric_dimensions"]) == 8
