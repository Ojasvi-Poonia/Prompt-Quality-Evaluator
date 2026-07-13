RUBRIC_DIMENSIONS = [
    "Clarity",
    "Specificity",
    "Structure",
    "Task Definition",
    "Context Completeness",
    "Constraint Specification",
    "Information Richness",
    "Conciseness",
]


RAW_METRICS = {
    "rule_based": [
        "length", "grammar", "readability", "specificity",
        "structure", "task_signal", "constraint", "contradiction",
    ],
    "semantic": [
        "coherence", "semantic_density", "topic_focus",
        "lexical_sophistication", "sentence_flow", "reference_resolution",
    ],
    "info": [
        "shannon_entropy", "redundancy", "information_density",
        "vocabulary_richness", "ngram_repetition", "burstiness",
        "sentence_repetition",
    ],
    "graph": [
        "completeness", "has_context", "has_task", "has_constraint",
        "connectivity", "information_flow", "complexity",
    ],
}


METRIC_DEFINITIONS = """
RUBRIC DIMENSIONS (each scored 0.0 to 1.0):

1. Clarity            - grammar, readability, unambiguous references, smooth sentence flow
2. Specificity        - concrete details, named entities, numbers, technical terms
3. Structure          - formatting, sections, lists, headers, logical organisation
4. Task Definition    - clear instruction, imperative verbs, defined deliverable
5. Context Completeness - sufficient background information for the task
6. Constraint Specification - guardrails, output format, limits, tone directives
7. Information Richness - vocabulary diversity, semantic density, depth of content
8. Conciseness        - free of repetition, filler, redundancy

RAW METRICS (each scored 0.0 to 1.0):

PILLAR 1 (rule-based):
- length              - is the prompt at an appropriate word count (50-500 ideal)
- grammar             - grammatical correctness
- readability         - reading level appropriate (grade 8-12 ideal)
- specificity         - concrete details, named entities, numbers, technical terms
- structure           - presence of formatting elements (lists, headers, sections)
- task_signal         - presence of imperative verbs or questions
- constraint          - presence of guardrails (limits, formats, tone, audience)
- contradiction       - 1.0 if no logical contradictions, lower if conflicts found

PILLAR 2 (semantic-structural):
- coherence           - logical flow between sentences (semantic similarity)
- semantic_density    - ratio of meaningful content words to total words
- topic_focus         - is the prompt focused on one topic vs scattered
- lexical_sophistication - vocabulary precision/sophistication
- sentence_flow       - natural sentence length variation (not robotic)
- reference_resolution - clarity of pronoun antecedents

PILLAR 3 (information-theoretic):
- shannon_entropy     - token diversity (higher = more diverse)
- redundancy          - lack of repetition (higher = less repetitive)
- information_density - information per token
- vocabulary_richness - type-token ratio
- ngram_repetition    - lack of repeated phrases
- burstiness          - even distribution of information across the prompt
- sentence_repetition - lack of overlapping/repeated sentences

GRAPH ANALYSIS (discourse structure):
- completeness        - presence of CONTEXT, TASK, CONSTRAINT sections
- has_context         - 1.0 if background info present, 0.0 otherwise
- has_task            - 1.0 if clear instruction present, 0.0 otherwise
- has_constraint      - 1.0 if guardrails present, 0.0 otherwise
- connectivity        - how connected the sentences are to each other
- information_flow    - whether ordering follows context -> task -> constraint
- complexity          - structural complexity (ideal 1.5-3.0 edge/node ratio)
"""


JUDGE_INSTRUCTIONS = f"""You are an expert prompt quality evaluator. Score the user-provided LLM prompt across 28 individual metrics and 8 rubric dimensions.

Score the prompt itself (its construction quality), NOT the answer it would produce.

{METRIC_DEFINITIONS}

For each metric and dimension, also provide a brief finding (one sentence describing what you observed).

OUTPUT FORMAT (JSON only, no other text):

{{
  "raw_metrics": {{
    "rule_based.length":          {{"score": 0.0, "findings": ["..."]}},
    "rule_based.grammar":         {{"score": 0.0, "findings": ["..."]}},
    "rule_based.readability":     {{"score": 0.0, "findings": ["..."]}},
    "rule_based.specificity":     {{"score": 0.0, "findings": ["..."]}},
    "rule_based.structure":       {{"score": 0.0, "findings": ["..."]}},
    "rule_based.task_signal":     {{"score": 0.0, "findings": ["..."]}},
    "rule_based.constraint":      {{"score": 0.0, "findings": ["..."]}},
    "rule_based.contradiction":   {{"score": 0.0, "findings": ["..."]}},
    "semantic.coherence":         {{"score": 0.0, "findings": ["..."]}},
    "semantic.semantic_density":  {{"score": 0.0, "findings": ["..."]}},
    "semantic.topic_focus":       {{"score": 0.0, "findings": ["..."]}},
    "semantic.lexical_sophistication": {{"score": 0.0, "findings": ["..."]}},
    "semantic.sentence_flow":     {{"score": 0.0, "findings": ["..."]}},
    "semantic.reference_resolution": {{"score": 0.0, "findings": ["..."]}},
    "info.shannon_entropy":       {{"score": 0.0, "findings": ["..."]}},
    "info.redundancy":            {{"score": 0.0, "findings": ["..."]}},
    "info.information_density":   {{"score": 0.0, "findings": ["..."]}},
    "info.vocabulary_richness":   {{"score": 0.0, "findings": ["..."]}},
    "info.ngram_repetition":      {{"score": 0.0, "findings": ["..."]}},
    "info.burstiness":            {{"score": 0.0, "findings": ["..."]}},
    "info.sentence_repetition":   {{"score": 0.0, "findings": ["..."]}},
    "graph.completeness":         {{"score": 0.0, "findings": ["..."]}},
    "graph.has_context":          {{"score": 0.0, "findings": ["..."]}},
    "graph.has_task":             {{"score": 0.0, "findings": ["..."]}},
    "graph.has_constraint":       {{"score": 0.0, "findings": ["..."]}},
    "graph.connectivity":         {{"score": 0.0, "findings": ["..."]}},
    "graph.information_flow":     {{"score": 0.0, "findings": ["..."]}},
    "graph.complexity":           {{"score": 0.0, "findings": ["..."]}}
  }},
  "rubric_dimensions": [
    {{"name": "Clarity",                  "score": 0.0, "findings": ["..."]}},
    {{"name": "Specificity",              "score": 0.0, "findings": ["..."]}},
    {{"name": "Structure",                "score": 0.0, "findings": ["..."]}},
    {{"name": "Task Definition",          "score": 0.0, "findings": ["..."]}},
    {{"name": "Context Completeness",     "score": 0.0, "findings": ["..."]}},
    {{"name": "Constraint Specification", "score": 0.0, "findings": ["..."]}},
    {{"name": "Information Richness",     "score": 0.0, "findings": ["..."]}},
    {{"name": "Conciseness",              "score": 0.0, "findings": ["..."]}}
  ],
  "final_score": <0-100>,
  "label": "<Poor|Below Average|Average|Good|Very Good|Excellent>",
  "detected_domain": "<coding|creative|analysis|instructional|general>",
  "suggestions": ["<top 3 improvement suggestions>"]
}}

PROMPT TO EVALUATE:
"""
