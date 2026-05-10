<p align="center">
  <h1 align="center">Prompt Quality Evaluator</h1>
  <p align="center">
    A deterministic, transformer-free scoring engine for LLM prompts.<br/>
    Analyzes prompt quality across 19 metrics and 8 rubric dimensions using classical NLP only.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#usage">Usage</a> |
  <a href="#how-it-works">How It Works</a> |
  <a href="#api-reference">API Reference</a> |
  <a href="#contributing">Contributing</a> |
  <a href="#license">License</a>
</p>

---

## Why This Exists

Prompt engineering is critical but subjective. Teams spend hours debating whether a prompt is "good enough" with no objective baseline. Existing solutions either require sending your prompt to an LLM (defeating the purpose) or offer only surface-level checks like word count.

**Prompt Quality Evaluator** solves this by providing a deterministic, locally-run scoring engine that evaluates prompts across 19 metrics derived from three complementary analytical approaches -- without ever calling an LLM or loading a transformer model.

---

## Features

- **19 individual metrics** across three analytical pillars (rule-based, semantic-structural, information-theoretic)
- **Discourse graph analysis** using NetworkX -- classifies sentences by function (context, task, constraint, example) and evaluates structural flow
- **8 rubric dimensions** that map raw metrics into human-readable quality categories
- **Radar chart visualization** saved as PNG for reports and documentation
- **Rich terminal output** with colored score bars, findings, and improvement suggestions
- **JSON output mode** for CI/CD pipeline integration and programmatic consumption
- **Fully deterministic** -- same input always produces same output
- **Zero external API calls** -- runs entirely offline after initial setup
- **No transformer models** -- uses only spaCy (en_core_web_md, with sm fallback), NLTK, scikit-learn TF-IDF, textstat (Flesch-Kincaid + Coleman-Liau + ARI + Gunning Fog + SMOG + Linsear Write ensemble), wordfreq (modern multi-source word frequencies), pyspellchecker, lexicalrichness (MTLD/HD-D/MATTR), and mathematical computation

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/prompt-quality-evaluator.git
cd prompt-quality-evaluator

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_md   # preferred (40 MB, includes word vectors)
# python -m spacy download en_core_web_sm # fallback (12 MB, no vectors -- still works)
```

NLTK data (`punkt_tab`, `stopwords`, `brown` corpus) is downloaded automatically on first run.

The system prefers `en_core_web_md` (with word vectors for better semantic analysis and out-of-vocabulary detection). It falls back to `en_core_web_sm` if `md` is not installed.

### First Evaluation

```bash
python main.py "Write a Python function that sorts a list of integers"
```

---

## Usage

### CLI

```bash
# Evaluate an inline prompt
python main.py "Your prompt text here"

# Evaluate from a file
python main.py --file prompt.txt

# JSON output (for scripts and pipelines)
python main.py "Your prompt" --json

# Show all 19 individual metric scores
python main.py "Your prompt" --verbose

# Skip radar chart generation
python main.py "Your prompt" --no-chart

# Custom output directory for the radar chart
python main.py "Your prompt" --output reports/
```

### Python API

```python
from evaluator.scorer import evaluate

result = evaluate("Your prompt text here")

print(result["final_score"])        # 0-100
print(result["label"])              # "Good", "Excellent", etc.
print(result["rubric_dimensions"])  # List of 8 dimension dicts
print(result["pillar_scores"])      # Per-pillar aggregates
print(result["suggestions"])        # Top 3 improvement suggestions
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Check prompt quality
  run: |
    score=$(python main.py --json --no-chart "$(cat prompts/system_prompt.txt)" | jq '.final_score')
    if (( $(echo "$score < 60" | bc -l) )); then
      echo "Prompt quality below threshold: $score"
      exit 1
    fi
```

---

## How It Works

The evaluator operates in four stages:

```
Input Prompt
    |
    v
+-------------------+    +------------------------+    +------------------------+
| Pillar 1          |    | Pillar 2               |    | Pillar 3               |
| Rule-Based        |    | Semantic-Structural    |    | Information-Theoretic  |
| (7 metrics)       |    | (6 metrics)            |    | (6 metrics)            |
+-------------------+    +------------------------+    +------------------------+
    |                         |                             |
    v                         v                             v
+------------------------------------------------------------------+
| Discourse Graph Analysis (NetworkX)                              |
| Sentence classification + structural flow metrics                |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
| Rubric Mapping                                                   |
| 19 raw metrics -> 8 weighted rubric dimensions                   |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
| Final Score (0-100) + Label + Radar Chart + Suggestions          |
+------------------------------------------------------------------+
```

### Pillar 1: Rule-Based Metrics

| Metric | What It Measures | Method |
|---|---|---|
| Length | Prompt word count against ideal range | Gaussian curve centered at 150 words |
| Grammar | Grammatical correctness | LanguageTool error detection |
| Readability | Flesch-Kincaid grade level | textstat library, ideal grade 8-12 |
| Specificity | Named entities, numbers, technical terms, code tokens | spaCy NER + regex patterns |
| Structure | Formatting elements (headers, lists, sections, delimiters) | Regex detection + variety scoring |
| Task Signal | Imperative verbs, question patterns, deliverable mentions | Keyword matching |
| Constraints | Negations, limits, format specs, audience, tone directives | Pattern matching across 5 categories |

### Pillar 2: Semantic-Structural Metrics

| Metric | What It Measures | Method |
|---|---|---|
| Coherence | Sentence-to-sentence topic flow | TF-IDF cosine similarity (scikit-learn) |
| Semantic Density | Ratio of meaningful lemmas to total tokens | spaCy POS tagging |
| Topic Focus | Single-topic vs scattered content | TF-IDF centroid distance |
| Lexical Sophistication | Vocabulary precision and rarity | NLTK Brown corpus frequency ranks |
| Sentence Flow | Natural length variation | Coefficient of variation scoring |
| Reference Resolution | Pronoun-antecedent clarity | spaCy dependency parsing |

### Pillar 3: Information-Theoretic Metrics

| Metric | What It Measures | Method |
|---|---|---|
| Shannon Entropy | Token diversity | Normalized entropy of tiktoken distribution |
| Redundancy | Token-level repetition | Unique/total token ratio |
| Information Density | Information per token | Entropy divided by token count |
| Vocabulary Richness | Type-token ratio robustness | Average of basic, root, and log TTR |
| N-gram Repetition | Repeated bigrams and trigrams | Frequency counting with penalty scaling |
| Burstiness | Uniformity of information distribution | Per-chunk entropy variance |

### Discourse Graph Analysis

Each sentence is classified as one of four discourse types using keyword heuristics:

- **CONTEXT** -- background information ("I have", "currently", "given")
- **TASK** -- the core instruction ("write", "explain", "how")
- **CONSTRAINT** -- guardrails ("must", "do not", "maximum", "in JSON")
- **EXAMPLE** -- illustrations ("for example", "e.g.", "such as")

The graph connects sentences via sequential adjacency and shared noun chunks/entities (spaCy). Four metrics are extracted: completeness (are all discourse types present?), connectivity (graph density), information flow (correct CONTEXT -> TASK -> CONSTRAINT ordering), and complexity (edge/node ratio).

### Rubric Dimensions

Raw metrics are combined into 8 interpretable dimensions using weighted composites:

| Dimension | Contributing Metrics | What It Tells You |
|---|---|---|
| **Clarity** | Grammar, readability, reference resolution, sentence flow | Can the LLM easily parse this prompt? |
| **Specificity** | Specificity, lexical sophistication, semantic density | Does the prompt contain concrete details? |
| **Structure** | Formatting, graph completeness, information flow, coherence | Is the prompt well-organized? |
| **Task Definition** | Task signal, graph task nodes, constraints | Is there a clear instruction? |
| **Context Completeness** | Graph context nodes, topic focus, length | Is enough background provided? |
| **Constraint Specification** | Constraints, graph constraint nodes, specificity | Are output requirements defined? |
| **Information Richness** | Entropy, vocabulary richness, info density, semantic density | Is the content substantive? |
| **Conciseness** | Redundancy, n-gram repetition, burstiness | Is the prompt free of filler and repetition? |

### Scoring Scale

| Score Range | Label |
|---|---|
| 85 -- 100 | Excellent |
| 75 -- 84 | Very Good |
| 60 -- 74 | Good |
| 45 -- 59 | Average |
| 25 -- 44 | Below Average |
| 0 -- 24 | Poor |

---

## Project Structure

```
prompt-quality-evaluator/
├── main.py                          # CLI entry point
├── evaluator/
│   ├── __init__.py
│   ├── rule_based.py                # Pillar 1: 7 rule-based metrics
│   ├── semantic_structural.py       # Pillar 2: 6 TF-IDF/spaCy metrics
│   ├── info_theoretic.py            # Pillar 3: 6 entropy-based metrics
│   ├── graph_analysis.py            # Discourse graph (NetworkX)
│   ├── rubric.py                    # Metric-to-dimension mapping
│   ├── scorer.py                    # Final scoring and aggregation
│   ├── visualizer.py                # Radar chart + terminal report
│   └── utils.py                     # Shared NLP helpers
├── data/
│   └── sample_prompts.json          # 20 prompts across 5 quality tiers
├── tests/
│   └── test_evaluator.py            # 35 tests (pytest)
├── requirements.txt
└── .gitignore
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test class
pytest tests/test_evaluator.py::TestMonotonicity -v

# Run with coverage (requires pytest-cov)
pytest tests/ --cov=evaluator --cov-report=term-missing
```

### Test Categories

| Category | Tests | What It Validates |
|---|---|---|
| Monotonicity | 1 | Average tier scores are strictly increasing across all 5 tiers |
| Boundaries | 3 | Empty string, single word, and good prompt produce expected score ranges |
| Rule-Based | 9 | Each Pillar 1 metric in isolation with known inputs |
| Semantic | 6 | Each Pillar 2 metric in isolation |
| Info-Theoretic | 6 | Each Pillar 3 metric in isolation |
| Graph Analysis | 3 | Graph construction, completeness scoring |
| Determinism | 1 | Same prompt produces identical scores across 3 runs |
| Edge Cases | 6 | All-caps, code-only, URL-only, special characters, long prompts |

---

## Requirements

| Package | Purpose |
|---|---|
| spacy (en_core_web_sm) | NER, POS tagging, dependency parsing, sentence splitting |
| nltk | Brown corpus frequencies, stopwords, tokenization |
| scikit-learn | TF-IDF vectorization, cosine similarity |
| textstat | Readability scoring (Flesch-Kincaid) |
| language-tool-python | Grammar checking |
| tiktoken | Token-level analysis (cl100k_base encoding) |
| networkx | Discourse graph construction and metrics |
| numpy / scipy | Mathematical computation |
| matplotlib | Radar chart generation |
| rich | Terminal formatting and colored output |
| pytest | Test framework |

---

## API Reference

### `evaluator.scorer.evaluate(text, dimension_weights=None)`

Main entry point. Returns a dictionary with:

```python
{
    "final_score": 72.5,                # float, 0-100
    "label": "Good",                    # str
    "rubric_dimensions": [              # list of 8 dicts
        {
            "name": "Clarity",
            "score": 0.73,              # float, 0-1
            "label": "Strong",
            "findings": ["No grammar issues detected.", ...]
        },
        ...
    ],
    "raw_metrics": {                    # all 19 individual metrics
        "rule_based.grammar": {"score": 0.95, "findings": [...]},
        ...
    },
    "pillar_scores": {                  # per-pillar averages
        "rule_based": 0.65,
        "semantic_structural": 0.58,
        "info_theoretic": 0.72,
        "graph_analysis": 0.55
    },
    "suggestions": [                    # top 3 improvement tips
        "Improve **Constraint Specification** (score: 0.14): ..."
    ]
}
```

### Individual Pillar APIs

Each pillar module exposes a `compute_all(text)` function and individual metric functions:

```python
from evaluator.rule_based import grammar_quality, compute_all

score, findings = grammar_quality("Your text here")
all_metrics = compute_all("Your text here")
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-metric`)
3. Ensure all 35 tests pass (`pytest tests/ -v`)
4. Verify the monotonicity test still passes (tier ordering must hold)
5. Submit a pull request

### Adding a New Metric

1. Add the scoring function to the appropriate pillar module (`rule_based.py`, `semantic_structural.py`, or `info_theoretic.py`)
2. Include it in the module's `compute_all()` function
3. Wire it into the relevant rubric dimension(s) in `rubric.py`
4. Add unit tests in `test_evaluator.py`
5. Verify monotonicity still holds

### Design Constraints

- No transformer models, no LLM API calls, no HuggingFace models
- All computation must be local and deterministic
- Every metric function must return `(score: float, findings: list[str])`
- Scores must be in the 0-1 range

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), [NetworkX](https://networkx.org/), [textstat](https://github.com/textstat/textstat), [language-tool-python](https://github.com/jxmorris12/language_tool_python), [tiktoken](https://github.com/openai/tiktoken), [matplotlib](https://matplotlib.org/), and [Rich](https://github.com/Textualize/rich).
