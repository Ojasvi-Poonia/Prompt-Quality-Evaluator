# Prompt Quality Evaluator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](tests/test_evaluator.py)
[![Deterministic](https://img.shields.io/badge/scoring-deterministic-blueviolet.svg)](#determinism)

Reference implementation for the paper **"A Multi-Pillar Prompt Quality Evaluation System Using Rule-Based, Semantic-Structural, Information-Theoretic, and Discourse Graph Metrics Without LLM Judgment."**

Scoring an LLM prompt usually means paying a second LLM to grade it. The cost grows with every call, and the verdict drifts between runs even when the input does not change. This project scores the prompt text directly, using classical NLP and no neural inference at evaluation time.

It computes **41 metrics** across **4 pillars**, folds them into **8 rubric dimensions**, and returns a single score in `[0, 100]` along with a named finding for every metric. It is deterministic, runs offline, and costs nothing per evaluation.

Against Gemini 2.5 Flash Lite on a 20-prompt benchmark it reaches **Pearson r = 0.932** and **Spearman ρ = 0.893**.

---

## Contents

- [Results](#results)
- [Install](#install)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [How it works](#how-it-works)
- [The 41 metrics](#the-41-metrics)
- [Rubric dimensions](#rubric-dimensions)
- [Python API](#python-api)
- [CI integration](#ci-integration)
- [Testing](#testing)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Results

Benchmarked against Gemini 2.5 Flash Lite over 20 prompts spanning five quality tiers. Raw measurements are checked in at [`benchmark_real_results.json`](benchmark_real_results.json).

| Statistic | Value |
|---|---|
| Pearson r (final score) | **0.932** (p = 2.30 × 10⁻⁹) |
| Spearman ρ (final score) | **0.893** (p = 1.14 × 10⁻⁷) |
| Mean absolute difference | 14.4 points |
| Exact verdict-label agreement | 40% (8/20) |
| Agreement within one band | 80% (16/20) |

Reproduce with `python run_benchmark.py --offline` (no API key needed).

### Per-dimension correlation

| Dimension | Pearson r | Strength |
|---|---|---|
| Structure | 0.909 | strong |
| Information Richness | 0.860 | strong |
| Specificity | 0.825 | strong |
| Task Definition | 0.799 | strong |
| Constraint Specification | 0.695 | strong |
| Clarity | 0.576 | moderate |
| Context Completeness | 0.509 | moderate |
| Conciseness | 0.240 | weak |

Conciseness is the weakest dimension, and the benchmark cannot cleanly adjudicate it. The judge's Conciseness scores are near-constant (mean 0.89, and it rates the bare prompt "fix it" a perfect 1.00), which caps the achievable correlation before our metrics get a vote. It appears to read conciseness as brevity. We read it structurally, with a short-text floor. Neither is really the construct a prompt engineer wants. See [Limitations](#limitations).

### Cost and latency

| Property | This project | Gemini judge |
|---|---|---|
| Mean latency per evaluation | 1.2 s | 3.4 s |
| Cost per evaluation | $0 | ~$0.005 |
| Internet required | no | yes |
| Deterministic | yes | no |
| Explanations | 41 named metrics | 1 paragraph |

---

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Ojasvi-Poonia/Prompt-Quality-Evaluator.git
cd Prompt-Quality-Evaluator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_md
```

`en_core_web_md` (40 MB) carries the word vectors that Pillar 2 needs. The system falls back to `en_core_web_sm` if `md` is absent, but semantic scores degrade.

NLTK data (`punkt_tab`, `stopwords`, `brown`) downloads automatically on first run.

> **Install every dependency in `requirements.txt`.** Several metrics fall back to a neutral 0.5 when their library is missing, so a partial install silently changes scores rather than failing loudly. `objectivity` needs TextBlob, `tone_consistency` needs vaderSentiment, and `vocabulary_richness` needs lexicalrichness.

Disk footprint at evaluation time is roughly 700 MB, almost all of it the spaCy model and the Java grammar tables bundled with language-tool-python. Peak memory is about 1.5 GB. No GPU, no network.

---

## Quick start

```bash
python main.py "Write a Python function that sorts a list of integers"
```

### CLI

```bash
python main.py "Your prompt text here"        # evaluate an inline prompt
python main.py --file prompt.txt              # evaluate from a file
python main.py "Your prompt" --json           # machine-readable output
python main.py "Your prompt" --verbose        # show all 41 metric scores
python main.py "Your prompt" --no-chart       # skip radar chart generation
python main.py "Your prompt" --output reports/  # chart output directory

python main.py "Your prompt" --rewrite        # emit an improved prompt template
python main.py "Your prompt" --confidence     # bootstrap confidence interval
python main.py "Your prompt" --ablation       # per-pillar contribution breakdown
python main.py "Your prompt" --compare other.txt   # score two prompts side by side
python main.py --batch prompts.json --export-csv out.csv   # batch mode
```

---

## Reproducing the paper

Every table in the paper regenerates from this repository. The measured Gemini scores are checked in, so the correlation figures reproduce **offline, with no API key**.

```bash
# Per-prompt comparison, correlations, tier means, label agreement.
# Reuses the stored Gemini scores. No network, no key.
python run_benchmark.py --offline

# Pillar ablation. Neutralises each pillar in turn and recomputes Pearson r.
python run_ablation.py

# Test suite: 35 cases, roughly one per metric plus aggregation logic.
pytest tests/ -v
```

### Regenerating the judge scores

The LLM judge is part of this repository, in [`llm_evaluator/`](llm_evaluator/). Every number we report about Gemini depends on how Gemini was asked, so the exact prompt template, the `temperature=0.0` setting, and the response parsing are all readable rather than described. If you disagree with how the question was posed, you can change it and re-run.

```bash
pip install -r requirements-benchmark.txt   # google-genai, openai
cp .env.example .env                        # then paste your key into .env
python run_benchmark.py                     # live run, overwrites the results file
```

`.env` is gitignored. Never commit it.

The judge's dependencies are deliberately kept **out** of `requirements.txt`. Installing the scorer must never pull in an LLM SDK, so the paper's central claim (nothing neural runs at evaluation time) is checkable by reading what `pip` installs, not just by trusting the prose.

### Determinism

The scorer returns the same value on every call for the same input. There is no sampling, no temperature, and no network. The test suite asserts this directly, and the paper reports one prompt scored 100 times with an identical result each time.

---

## How it works

```
                          Prompt text
                               |
                     spaCy preprocessing
           (tokens, POS, NER, dependency parse)
                               |
        +--------------+-------+-------+--------------+
        |              |               |              |
    Pillar 1       Pillar 2        Pillar 3       Pillar 4
   Rule-Based      Semantic       Info-Theory     Discourse
  (16 metrics)   (10 metrics)     (7 metrics)   Graph (8 metrics)
        |              |               |              |
        +--------------+-------+-------+--------------+
                               |
                  41 raw scores in [0, 1]
                               |
              Rubric weighting matrix (8 x 41)
                               |
       length factor  x  task penalty cap  x  excellence bonus
                               |
        Final score [0, 100] + 8 dimensions + 41 findings
```

Three modifiers sit on top of the weighted dimensions:

1. **Length factor.** Prompts under 30 words are scaled down. A four-word prompt has no room to demonstrate structure or context.
2. **Task penalty cap.** If a prompt carries no clear instruction, the score is clipped (12 to 40 depending on length). No amount of clean formatting lifts a prompt with no task above this ceiling.
3. **Excellence bonus.** Applied only when *all four* pillar means clear 0.6, so a prompt earns it by being good across the board rather than by gaming one pillar.

---

## The 41 metrics

### Pillar 1: Rule-based (16)

| Metric | What it measures |
|---|---|
| `length` | Word count, Gaussian centred at 150 words |
| `grammar` | LanguageTool matches plus spellchecker typos, per sentence |
| `readability` | Mean of six formulas (Flesch-Kincaid, Coleman-Liau, ARI, Gunning Fog, SMOG, Linsear Write) |
| `specificity` | Density of entities, numbers, technical terms, quoted strings, code tokens |
| `structure` | Bullets, numbered lists, headers, fenced code, section labels |
| `task_signal` | Imperative verbs, question patterns, deliverable nouns |
| `constraint` | Negations, limits, format directives, audience specs, tone directives |
| `contradiction` | Conflicting instructions and incompatible tone pairs |
| `reference_similarity` | Cosine similarity to canonical prompts in the detected domain |
| `persona_signal` | Role-setting phrases ("you are a senior X", "act as Y") |
| `few_shot_quality` | In-context examples: input/output pairs, arrow mappings |
| `hedging` | Density of hedge phrases from a 33-item lexicon |
| `safety` | Prompt-injection patterns ("ignore previous instructions") |
| `output_schema` | Explicit output format specs (JSON, types, signatures) |
| `compound_task_count` | Distinct task verbs chained via dependency conjunctions |
| `imperative_strength` | Sentence-initial verbs minus politeness softeners |

### Pillar 2: Semantic-structural (10)

| Metric | What it measures |
|---|---|
| `coherence` | Cosine similarity between consecutive sentences (TF-IDF 0.4, word vectors 0.6) |
| `semantic_density` | Unique meaningful lemmas over total tokens |
| `topic_focus` | Mean cosine distance of each sentence to the document centroid |
| `lexical_sophistication` | Word rarity from `wordfreq` zipf scores |
| `sentence_flow` | Coefficient of variation of sentence lengths |
| `reference_resolution` | Ambiguous pronouns without a nearby antecedent |
| `tone_consistency` | Variance of VADER sentiment across sentences |
| `objectivity` | Inverse of TextBlob subjectivity |
| `syntactic_complexity` | Mean dependency-tree depth and clause count |
| `discourse_markers` | Logical connectives across six PDTB categories |

### Pillar 3: Information-theoretic (7)

Tokenised with tiktoken `cl100k_base`. Every metric here applies a short-text cap of 0.3 below 30 tokens, because entropy measures are biased upward on short text.

| Metric | What it measures |
|---|---|
| `shannon_entropy` | Normalised entropy of the token distribution |
| `redundancy` | Token repetition, with a heavy penalty past 60% repeats |
| `information_density` | Entropy per token |
| `vocabulary_richness` | Mean of TTR, MTLD, HD-D, and MATTR |
| `ngram_repetition` | Repeated bigrams and trigrams |
| `burstiness` | Entropy variance across five equal token chunks |
| `sentence_repetition` | Sentence pairs with Jaccard overlap above 0.7 |

### Pillar 4: Discourse graph (8)

Each sentence is classified as CONTEXT, TASK, CONSTRAINT, EXAMPLE, or OTHER. A directed graph links consecutive sentences and any pair sharing noun chunks or named entities. Standard graph algorithms then turn the structure into numbers.

| Metric | What it measures |
|---|---|
| `completeness` | Are CONTEXT, TASK, and CONSTRAINT all present? |
| `has_context` / `has_task` / `has_constraint` | Binary presence indicators |
| `connectivity` | Graph density over the number of weakly connected components |
| `information_flow` | Whether CONTEXT precedes TASK precedes CONSTRAINT |
| `complexity` | Edge-to-node ratio, Gaussian-scored |
| `centrality` | PageRank spread. A flat distribution means no sentence is the focus; a spiked one means a single sentence dominates. Both are penalised. |

---

## Rubric dimensions

The 41 metrics collapse into 8 dimensions through a fixed weighting matrix. Each row sums to 1.0.

| Dimension | Contributing metrics (weight) |
|---|---|
| **Clarity** | grammar (.18), readability (.15), syntactic_complexity (.15), hedging (.14), reference_resolution (.10), sentence_flow (.10), objectivity (.10), tone_consistency (.08) |
| **Specificity** | specificity (.30), lexical_sophistication (.20), output_schema (.15), semantic_density (.15), reference_similarity (.10), persona_signal (.10) |
| **Structure** | structure (.20), completeness (.10), information_flow (.10), centrality (.10), coherence (.10), discourse_markers (.10), reference_similarity (.10), few_shot_quality (.10), output_schema (.10) |
| **Task Definition** | task_signal (.30), imperative_strength (.15), has_task (.15), safety (.15), compound_task_count (.10), contradiction (.10), persona_signal (.05) |
| **Context Completeness** | has_context (.30), topic_focus (.20), length (.20), persona_signal (.15), few_shot_quality (.15) |
| **Constraint Specification** | constraint (.35), has_constraint (.20), output_schema (.20), contradiction (.15), safety (.10) |
| **Information Richness** | shannon_entropy (.25), vocabulary_richness (.25), information_density (.25), semantic_density (.25) |
| **Conciseness** | redundancy (.25), ngram_repetition (.25), sentence_repetition (.25), burstiness (.25) |

### Scoring scale

| Range | Label |
|---|---|
| 85 to 100 | Excellent |
| 75 to 84 | Very Good |
| 60 to 74 | Good |
| 45 to 59 | Average |
| 25 to 44 | Below Average |
| 0 to 24 | Poor |

---

## Python API

```python
from evaluator.scorer import evaluate

result = evaluate("Your prompt text here")

result["final_score"]        # float, 0 to 100
result["label"]              # "Good", "Excellent", ...
result["rubric_dimensions"]  # 8 dicts: name, score, label, findings
result["raw_metrics"]        # all 41 metrics, keyed "<pillar>.<metric>"
result["pillar_scores"]      # per-pillar means
result["suggestions"]        # ranked improvement suggestions
```

Each pillar module also exposes its metrics individually:

```python
from evaluator.rule_based import compute_all, grammar_quality

score, findings = grammar_quality("Your text here")
all_metrics = compute_all("Your text here")
```

Every metric function returns `(score: float, findings: list[str])` with the score in `[0, 1]`.

---

## CI integration

Because scoring is deterministic and free, it works as a regression gate on prompts held in version control.

```yaml
- name: Check prompt quality
  run: |
    score=$(python main.py --json --no-chart "$(cat prompts/system_prompt.txt)" | jq '.final_score')
    if (( $(echo "$score < 60" | bc -l) )); then
      echo "Prompt quality below threshold: $score"
      exit 1
    fi
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=evaluator --cov-report=term-missing
```

35 tests covering tier monotonicity, boundary cases, each pillar in isolation, graph construction, determinism, and edge cases (all-caps, code-only, URL-only, very long prompts).

---

## Limitations

Reported honestly, because they bound what this tool is good for.

- **The benchmark is 20 prompts and entirely engineering.** The domain detector labels 11 of them coding and 9 general. There is not one creative or natural-science prompt in the corpus. Every correlation here is a statement about engineering prompts and should not be read as a claim about prompt quality in general. The 95% confidence interval on r = 0.932 at n = 20 is roughly [0.83, 0.97].
- **Context Completeness is only moderate** (r = 0.509). We count discourse markers; the LLM judges whether the context actually fits the task. A short prompt with exactly the right context scores low here and high with the judge. This is the largest genuine hole in what the pipeline can see.
- **Conciseness does not correlate** (r = 0.240), and the benchmark cannot settle who is right. See the note under [Results](#results).
- **Creative and literary prompts are underserved.** "noir-style" and "unreliable narrator" constrain an output tightly, but we score such a prompt only 48.4 (Average). We expect a transformer judge to do better here, though with no creative prompts in the corpus we have not measured the gap.
- **Code-heavy prompts** with unfenced inline code confuse the grammar and readability metrics. `Refactor this function: def f(x): return x*2` scores 32.6 despite being a perfectly clear instruction.
- **English only.** Other languages are flagged and not scored.
- **Rubric weights are hand-tuned**, nudged until tier monotonicity held. The `creative` and `instructional` domain weights have never been validated against a judge. A regression fitted on human-labelled prompts would very likely beat hand-tuning.

---

## Project structure

```
prompt-quality-evaluator/
├── main.py                      # CLI entry point
├── evaluator/
│   ├── rule_based.py            # Pillar 1: 16 metrics
│   ├── semantic_structural.py   # Pillar 2: 10 metrics
│   ├── info_theoretic.py        # Pillar 3: 7 metrics
│   ├── graph_analysis.py        # Pillar 4: discourse graph, 8 metrics
│   ├── rubric.py                # 41 metrics -> 8 dimensions, domain weighting
│   ├── scorer.py                # Aggregation and the three modifiers
│   ├── improvements.py          # Actionable suggestions
│   ├── reference_templates.py   # Canonical prompts per domain
│   ├── visualizer.py            # Radar chart and terminal report
│   └── utils.py                 # Shared NLP helpers
├── llm_evaluator/               # The Gemini judge baseline (NOT used by the scorer)
│   ├── judge.py                 #   API calls, temperature=0.0, response parsing
│   ├── template.py              #   the exact judge prompt, 8 dimensions
│   └── visualizer.py
├── data/sample_prompts.json     # 20-prompt benchmark corpus, 5 tiers
├── tests/test_evaluator.py      # 35 tests
├── run_benchmark.py             # Benchmark vs the Gemini judge (--offline supported)
├── run_ablation.py              # Pillar ablation study
├── benchmark_real_results.json  # Measured scores for both systems
├── requirements.txt             # Scorer deps. Contains no LLM SDK, by design.
├── requirements-benchmark.txt   # Judge deps, only needed for a live run
├── requirements.lock.txt        # Exact versions behind every number in the paper
└── .env.example                 # Copy to .env for a live run. .env is gitignored.
```

`evaluator/` and `llm_evaluator/` never import each other. The scorer does not know the judge exists.

---

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{poonia2026promptquality,
  title     = {A Multi-Pillar Prompt Quality Evaluation System Using Rule-Based,
               Semantic-Structural, Information-Theoretic, and Discourse Graph
               Metrics Without {LLM} Judgment},
  author    = {Poonia, Ojasvi and G, Brunda},
  booktitle = {},
  year      = {2026},
  note      = {Code: \url{https://github.com/Ojasvi-Poonia/Prompt-Quality-Evaluator}}
}
```

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built with [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), [NetworkX](https://networkx.org/), [textstat](https://github.com/textstat/textstat), [wordfreq](https://github.com/rspeer/wordfreq), [lexicalrichness](https://github.com/LSYS/LexicalRichness), [language-tool-python](https://github.com/jxmorris12/language_tool_python), [tiktoken](https://github.com/openai/tiktoken), [TextBlob](https://textblob.readthedocs.io/), [VADER](https://github.com/cjhutto/vaderSentiment), [matplotlib](https://matplotlib.org/), and [Rich](https://github.com/Textualize/rich).
