from typing import Dict, List, Optional


CANONICAL_PROMPTS: Dict[str, List[str]] = {
    "coding": [
        "Context: I am building a REST API using Django 4.2 with PostgreSQL 14. "
        "The system has 5000 daily active users and we need to add a Task model. "
        "Task: Write a Django REST Framework serializer and viewset for the Task model "
        "with full CRUD operations. "
        "Requirements: title (max 200 chars), status (todo/in_progress/done), priority (1-5), "
        "due_date validation, paginated results, role-based update permissions. "
        "Constraints: Do not use generic views, return proper HTTP status codes, "
        "format response as JSON with snake_case fields. "
        "Example: GET /api/tasks/1 returns {\"id\": 1, \"title\": \"Fix bug\", \"status\": \"todo\"}.",

        "Context: We have a high-traffic e-commerce backend on Node.js Express with Redis caching. "
        "Task: Implement a rate limiter middleware using the sliding window algorithm. "
        "Requirements: 100 requests per minute per IP, return 429 with Retry-After header, "
        "use Redis for cross-instance state, log violations. "
        "Constraints: No third-party rate limiter packages, must include unit tests. "
        "Output as a single JavaScript module with exported middleware function.",
    ],
    "creative": [
        "Context: I'm writing a short story collection about climate change set in 2050. "
        "Task: Write the opening paragraph of a story about a coastal city dealing with rising sea levels. "
        "Tone: literary, melancholic, hopeful. "
        "Constraints: 250-300 words, third-person omniscient narrator, no exposition dumps, "
        "introduce the protagonist through action not description. "
        "Style references: Kim Stanley Robinson, Margaret Atwood.",

        "Context: I'm developing a screenplay for a 30-minute psychological thriller. "
        "Task: Write a tense dialogue scene between a detective and a suspect in an interrogation room. "
        "Constraints: Maximum 400 words, no scene direction beyond character actions, "
        "the suspect must lie at least once but the lie should be subtle. "
        "Output in standard screenplay format (CHARACTER NAME in caps, dialogue centered). "
        "Style: Aaron Sorkin pacing.",
    ],
    "analysis": [
        "Context: I have sales data for the past 24 months across 5 product categories and 3 regions. "
        "Total annual revenue is $12M with declining margins. "
        "Task: Analyse the data to identify the top 3 factors driving margin decline. "
        "Requirements: Use Python with pandas, segment by category and region, "
        "compute YoY growth rates, surface seasonal patterns, "
        "compare against industry benchmarks. "
        "Constraints: Do not use machine learning, prefer interpretable statistics, "
        "produce visualisations as matplotlib charts. "
        "Output: numbered list of findings with supporting data.",

        "Context: We are evaluating cloud providers for a B2B SaaS migration. "
        "Current infrastructure: 50 servers on-prem, $200K annual IT spend, 99.9% SLA requirement. "
        "Task: Compare AWS, Azure, and Google Cloud across cost, reliability, and developer experience. "
        "Requirements: Three-column comparison table, weighted scoring (cost 40%, reliability 35%, DX 25%), "
        "factor in 3-year TCO, include migration risk assessment. "
        "Constraints: Use only publicly available pricing, cite sources, "
        "make recommendation with rationale. Maximum 1500 words.",
    ],
    "instructional": [
        "Context: I am teaching an introductory web development class to college students with basic HTML knowledge. "
        "Task: Explain how HTTP cookies work in a way that a beginner can understand. "
        "Requirements: Start with a real-world analogy, define key terms (session, persistent, secure flag), "
        "explain the request-response cycle, give a concrete example with code. "
        "Constraints: Maximum 600 words, no jargon without definition, include 2 code examples in JavaScript, "
        "tone: friendly and encouraging. End with three review questions.",

        "Context: I am writing a tutorial for engineers learning Kubernetes. "
        "Task: Explain the difference between Pods, Deployments, and Services. "
        "Requirements: Show the relationship between them with a diagram, "
        "include a working YAML example, explain when to use each, "
        "address common misconceptions. "
        "Constraints: 800-1200 words, intermediate audience (assume Docker knowledge), "
        "use a real-world scenario (e.g., deploying a web app), include 'gotchas' section.",
    ],
    "general": [
        "Context: <relevant background information>. "
        "Task: <clear instruction with imperative verb>. "
        "Requirements: <specific deliverables and details>. "
        "Constraints: <do-not list, format requirements, audience, tone>. "
        "Example: <concrete example of expected output>.",
    ],
}


def get_canonical_prompts(domain: str) -> List[str]:
    return CANONICAL_PROMPTS.get(domain, CANONICAL_PROMPTS["general"])


def reference_similarity(text: str, domain: str) -> Optional[float]:
    from evaluator.utils import get_nlp
    nlp = get_nlp()
    target = nlp(text)
    if not target.has_vector or target.vector_norm == 0:
        return None

    canonical = get_canonical_prompts(domain)
    sims: List[float] = []
    for cp in canonical:
        ref = nlp(cp)
        if ref.has_vector and ref.vector_norm > 0:
            sims.append(float(target.similarity(ref)))

    if not sims:
        return None
    return max(sims)
