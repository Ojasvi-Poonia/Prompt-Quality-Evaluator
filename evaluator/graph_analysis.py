import re
from typing import Tuple, List, Dict

import networkx as nx

from evaluator.utils import get_nlp, split_sentences, clamp


_CONTEXT_MARKERS = [
    r"\bcurrently\b", r"\bgiven\b", r"\bbackground\b", r"\bsituation\b",
    r"\bi have\b", r"\bi want\b", r"\bi need\b", r"\bi am\b",
    r"\bwe have\b", r"\bwe are\b", r"\bour\b", r"\bthe current\b",
    r"\bexisting\b", r"\bpreviously\b",
]

_TASK_MARKERS = [
    r"\bwrite\b", r"\bexplain\b", r"\blist\b", r"\bcompare\b",
    r"\banalyze\b", r"\banalyse\b", r"\bsummarize\b", r"\bsummarise\b",
    r"\bcreate\b", r"\bgenerate\b", r"\bdesign\b", r"\bbuild\b",
    r"\bimplement\b", r"\bevaluate\b", r"\bdescribe\b", r"\bprovide\b",
    r"\bcalculate\b", r"\btranslate\b", r"\bconvert\b",
    r"\bhow\b.*\?", r"\bwhat\b.*\?", r"\bwhy\b.*\?",
]

_CONSTRAINT_MARKERS = [
    r"\bmust\b", r"\bshould\b", r"\bdo not\b", r"\bdon'?t\b",
    r"\bmaximum\b", r"\bat most\b", r"\bno more than\b", r"\blimit\b",
    r"\bformat\b", r"\bin json\b", r"\bas a table\b", r"\brequire\b",
    r"\bensure\b", r"\bmake sure\b", r"\bavoid\b", r"\bnever\b",
]

_EXAMPLE_MARKERS = [
    r"\bfor example\b", r"\be\.g\.\b", r"\bsuch as\b",
    r"\blike this\b", r"\bsample\b", r"\bfor instance\b",
    r"\bhere is\b", r"\bhere's\b", r"\bexample\b",
]


def _classify_sentence(sentence: str) -> str:
    text = sentence.lower()

    if any(re.search(p, text) for p in _EXAMPLE_MARKERS):
        return "EXAMPLE"
    if any(re.search(p, text) for p in _CONSTRAINT_MARKERS):
        return "CONSTRAINT"
    if any(re.search(p, text) for p in _TASK_MARKERS):
        return "TASK"
    if any(re.search(p, text) for p in _CONTEXT_MARKERS):
        return "CONTEXT"

    return "OTHER"


def build_discourse_graph(text: str) -> Tuple[nx.DiGraph, List[Dict]]:
    nlp = get_nlp()
    sentences = split_sentences(text)

    G = nx.DiGraph()
    node_data: List[Dict] = []

    for i, sent_text in enumerate(sentences):
        doc = nlp(sent_text)
        node_type = _classify_sentence(sent_text)
        chunks = {chunk.root.lemma_.lower() for chunk in doc.noun_chunks}
        entities = {ent.text.lower() for ent in doc.ents}

        G.add_node(i, type=node_type, text=sent_text)
        node_data.append({
            "id": i,
            "text": sent_text,
            "type": node_type,
            "noun_chunks": chunks,
            "entities": entities,
        })

    for i in range(len(node_data)):
        if i + 1 < len(node_data):
            G.add_edge(i, i + 1, weight=1)

        for j in range(i + 2, len(node_data)):
            shared = (
                node_data[i]["noun_chunks"] & node_data[j]["noun_chunks"]
                | node_data[i]["entities"] & node_data[j]["entities"]
            )
            if shared:
                G.add_edge(i, j, weight=len(shared))

    return G, node_data


def completeness_score(node_data: List[Dict]) -> Tuple[float, List[str]]:
    types_present = {nd["type"] for nd in node_data}
    findings: List[str] = []

    score = 0.0
    required = {"CONTEXT", "TASK", "CONSTRAINT"}
    for t in required:
        if t in types_present:
            score += 0.33
        else:
            findings.append(f"Missing {t} section.")

    if "EXAMPLE" in types_present:
        score += 0.1
        findings.append("Includes examples (bonus).")

    if not findings:
        findings.append("All discourse sections present (context, task, constraint).")

    return clamp(score), findings


def has_context(node_data: List[Dict]) -> Tuple[float, List[str]]:
    types_present = {nd["type"] for nd in node_data}
    if "CONTEXT" in types_present:
        return 1.0, ["Context section detected in discourse graph."]
    return 0.0, ["No context section detected. Add background info."]


def has_task(node_data: List[Dict]) -> Tuple[float, List[str]]:
    types_present = {nd["type"] for nd in node_data}
    if "TASK" in types_present:
        return 1.0, ["Task section detected in discourse graph."]
    return 0.0, ["No task section detected. Add a clear instruction."]


def has_constraint(node_data: List[Dict]) -> Tuple[float, List[str]]:
    types_present = {nd["type"] for nd in node_data}
    if "CONSTRAINT" in types_present:
        return 1.0, ["Constraint section detected in discourse graph."]
    return 0.0, ["No constraint section detected. Add guardrails."]


def connectivity_score(G: nx.DiGraph) -> Tuple[float, List[str]]:
    findings: List[str] = []

    if G.number_of_nodes() <= 1:
        return 0.5, ["Too few nodes for connectivity analysis."]

    density = nx.density(G)
    undirected = G.to_undirected()
    n_components = nx.number_connected_components(undirected)

    connectivity_factor = 1.0 / n_components
    score = clamp(density * 2 * connectivity_factor)

    if n_components == 1:
        findings.append(f"Fully connected graph (density {density:.2f}).")
    else:
        findings.append(f"{n_components} disconnected sections in the prompt.")

    return score, findings


def information_flow_score(node_data: List[Dict]) -> Tuple[float, List[str]]:
    findings: List[str] = []

    type_positions: Dict[str, List[int]] = {}
    for nd in node_data:
        t = nd["type"]
        if t in ("CONTEXT", "TASK", "CONSTRAINT"):
            type_positions.setdefault(t, []).append(nd["id"])

    if len(type_positions) < 2:
        return 0.5, ["Not enough discourse types for flow analysis."]

    order_score = 0.0
    checks = 0

    def avg_pos(t: str) -> float:
        positions = type_positions.get(t, [])
        return sum(positions) / len(positions) if positions else float("inf")

    ctx_pos = avg_pos("CONTEXT")
    task_pos = avg_pos("TASK")
    con_pos = avg_pos("CONSTRAINT")

    if "CONTEXT" in type_positions and "TASK" in type_positions:
        checks += 1
        if ctx_pos < task_pos:
            order_score += 1.0
            findings.append("Context appears before task (good).")
        else:
            findings.append("Task appears before context (suboptimal).")

    if "TASK" in type_positions and "CONSTRAINT" in type_positions:
        checks += 1
        if task_pos < con_pos:
            order_score += 1.0
            findings.append("Task appears before constraints (good).")
        else:
            findings.append("Constraints appear before task (suboptimal).")

    score = order_score / max(checks, 1)
    return clamp(score), findings


def graph_centrality_score(G: nx.DiGraph) -> Tuple[float, List[str]]:
    findings: List[str] = []
    nodes = G.number_of_nodes()

    if nodes <= 1:
        return 0.4, ["Too few nodes for centrality analysis."]

    if G.number_of_edges() == 0:
        return 0.3, ["No edges -- sentences are completely disconnected."]

    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception:
        return 0.5, ["Could not compute PageRank."]

    pr_values = list(pagerank.values())
    if not pr_values:
        return 0.5, ["Empty PageRank result."]

    pr_max = max(pr_values)
    pr_min = min(pr_values)
    spread = pr_max - pr_min

    avg_pr = sum(pr_values) / len(pr_values)
    variance = sum((v - avg_pr) ** 2 for v in pr_values) / len(pr_values)
    std = variance ** 0.5

    cv = std / avg_pr if avg_pr > 0 else 0.0

    from evaluator.utils import gaussian_score
    score = gaussian_score(cv, mean=0.4, std=0.3)

    if cv < 0.1:
        findings.append(
            f"All sentences equally central (CV={cv:.2f}). "
            "Prompt lacks a 'core' sentence -- consider emphasising the main task."
        )
    elif cv > 0.8:
        findings.append(
            f"One sentence dominates (CV={cv:.2f}). Other sentences may be redundant."
        )
    else:
        findings.append(
            f"Good information hierarchy (CV={cv:.2f}, max PR={pr_max:.2f}, min PR={pr_min:.2f})."
        )

    return clamp(score), findings


def complexity_score(G: nx.DiGraph) -> Tuple[float, List[str]]:
    findings: List[str] = []
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    if nodes == 0:
        return 0.0, ["No nodes in graph."]

    if nodes <= 1:
        return 0.3, ["Single-sentence prompt -- structural complexity is minimal."]

    ratio = edges / nodes

    from evaluator.utils import gaussian_score
    score = gaussian_score(ratio, mean=2.25, std=1.5)

    if ratio < 1.0:
        findings.append(f"Sparse prompt structure (edge/node ratio {ratio:.2f}).")
    elif ratio <= 3.0:
        findings.append(f"Good structural complexity (ratio {ratio:.2f}).")
    else:
        findings.append(f"Complex/convoluted structure (ratio {ratio:.2f}). Consider simplifying.")

    return clamp(score), findings


def compute_all(text: str) -> dict:
    G, node_data = build_discourse_graph(text)

    type_counts: Dict[str, int] = {}
    for nd in node_data:
        t = nd["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    results = {}
    for name, fn in [
        ("completeness", lambda: completeness_score(node_data)),
        ("has_context", lambda: has_context(node_data)),
        ("has_task", lambda: has_task(node_data)),
        ("has_constraint", lambda: has_constraint(node_data)),
        ("connectivity", lambda: connectivity_score(G)),
        ("information_flow", lambda: information_flow_score(node_data)),
        ("complexity", lambda: complexity_score(G)),
        ("centrality", lambda: graph_centrality_score(G)),
    ]:
        score, findings = fn()
        results[name] = {"score": score, "findings": findings}

    results["graph_node_types"] = type_counts
    return results
