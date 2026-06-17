from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .retrieval import RetrievalResult, Retriever
from .utils import compact_text


@dataclass
class Answer:
    query: str
    route: str
    answer: str
    reasoning: str
    citations: list[dict[str, Any]]
    evidence_chain: list[list[str]]
    confidence: float
    conflicts_or_missing_evidence: list[str]
    production_recommendation: str
    next_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerSynthesizer:
    def synthesize(self, result: RetrievalResult) -> Answer:
        candidates = result.candidates[:6]
        if not candidates:
            return Answer(
                query=result.query,
                route=result.route.route,
                answer="I do not have enough retrieved evidence to answer this query.",
                reasoning="No chunks, claims, or graph paths were retrieved above the local relevance threshold.",
                citations=[],
                evidence_chain=[],
                confidence=0.0,
                conflicts_or_missing_evidence=["No retrieved evidence."],
                production_recommendation="Do not make a production decision from this run; ingest more source data first.",
                next_checks=["Run `signalgraph ingest run --topic ...` for the target domain.", "Inspect raw source records and graph stats."],
            )
        best = candidates[0]
        route = result.mode if result.mode != "vector" else result.route.route
        citations = [candidate.citation for candidate in candidates]
        evidence_chain = [candidate.path for candidate in candidates if candidate.path]
        if result.mode == "drift":
            for branch in result.trace.get("answer_tree", [])[:3]:
                evidence_chain.extend(branch.get("evidence_chain", []))
        confidence = round(sum(candidate.score for candidate in candidates) / len(candidates), 3)
        labels = _labels_from_candidates(candidates)
        top_summary = "; ".join(compact_text(candidate.source_span, 180) for candidate in candidates[:3])
        answer = _answer_sentence(result.query, route, labels, top_summary)
        reasoning = (
            f"SignalGraph routed the query to {route} retrieval and ranked evidence by semantic/lexical relevance, "
            "graph path quality, source quality, freshness, confidence, and evidence strength."
        )
        conflicts = _conflicts_and_gaps(candidates, labels)
        recommendation = _production_recommendation(candidates, labels)
        next_checks = _next_checks(labels)
        return Answer(
            query=result.query,
            route=route,
            answer=answer,
            reasoning=reasoning,
            citations=citations,
            evidence_chain=evidence_chain,
            confidence=confidence,
            conflicts_or_missing_evidence=conflicts,
            production_recommendation=recommendation,
            next_checks=next_checks,
        )


def ask(paths, query: str) -> Answer:
    result = Retriever(paths).graph_aware(query)
    return AnswerSynthesizer().synthesize(result)


def compare(paths, query: str) -> dict[str, Any]:
    retriever = Retriever(paths)
    synthesizer = AnswerSynthesizer()
    vector_result = retriever.vector_only(query)
    graph_result = retriever.graph_aware(query)
    vector_answer = synthesizer.synthesize(vector_result)
    graph_answer = synthesizer.synthesize(graph_result)
    return {
        "query": query,
        "vector_only": vector_answer.to_dict(),
        "graph_rag": graph_answer.to_dict(),
        "retrieved_context_difference": {
            "vector_node_ids": [candidate.node_id for candidate in vector_result.candidates],
            "graph_node_ids": [candidate.node_id for candidate in graph_result.candidates],
            "graph_paths": [candidate.path for candidate in graph_result.candidates if candidate.path],
        },
        "faithfulness_estimate": {
            "vector_only": _faithfulness_estimate(vector_answer),
            "graph_rag": _faithfulness_estimate(graph_answer),
        },
        "evidence_chain_completeness": {
            "vector_only": _chain_completeness(vector_answer),
            "graph_rag": _chain_completeness(graph_answer),
        },
        "latency_ms": {
            "vector_only": vector_result.latency_ms,
            "graph_rag": graph_result.latency_ms,
        },
        "cost_usd_estimate": {
            "vector_only": 0.0,
            "graph_rag": 0.0,
        },
    }


def _answer_sentence(query: str, route: str, labels: set[str], top_summary: str) -> str:
    if route == "global":
        return f"The landscape answer is grounded in community reports and representative graph evidence: {top_summary}"
    if route == "drift":
        return f"The decision-grade answer combines community report breadth with local evidence chains: {top_summary}"
    if route == "structured_lookup":
        return f"The structured lookup found typed graph matches through the Cypher-template path: {top_summary}"
    if route == "hybrid":
        return f"The hybrid answer fuses vector, full-text, and graph traversal evidence before diversity reranking: {top_summary}"
    if "Repo" in labels and "production" in query.lower():
        return f"The strongest production-oriented signal is the repository evidence connected to the research graph: {top_summary}"
    if route in {"comparison"}:
        return f"The graph-aware view favors evidence with connected papers, methods, repos, and claims rather than isolated chunks: {top_summary}"
    return f"The best-supported answer is grounded in the retrieved claims and source chunks: {top_summary}"


def _conflicts_and_gaps(candidates, labels: set[str]) -> list[str]:
    notes: list[str] = []
    caution = [candidate for candidate in candidates if any(word in candidate.text.lower() for word in ["risk", "limitation", "challenge", "stale"])]
    if caution:
        notes.append("Retrieved evidence includes cautionary or limitation language; treat recommendation as conditional.")
    if "Benchmark" not in labels and "Dataset" not in labels:
        notes.append("No explicit benchmark or dataset node was retrieved for the top evidence chain.")
    if "Repo" not in labels:
        notes.append("No implementation repository was retrieved in the top evidence.")
    return notes or ["No direct conflicts detected in retrieved evidence."]


def _labels_from_candidates(candidates) -> set[str]:
    labels = {label for candidate in candidates for label in candidate.labels}
    for candidate in candidates:
        for part in candidate.path:
            if part.startswith("paper:"):
                labels.add("Paper")
            elif part.startswith("repo:"):
                labels.add("Repo")
            elif part.startswith("method:"):
                labels.add("Method")
            elif part.startswith("claim:"):
                labels.add("Claim")
            elif part.startswith("chunk:"):
                labels.add("DocumentChunk")
            elif part.startswith("benchmark:"):
                labels.add("Benchmark")
            elif part.startswith("dataset:"):
                labels.add("Dataset")
            elif part.startswith("model:"):
                labels.add("Model")
    return labels


def _production_recommendation(candidates, labels: set[str]) -> str:
    if "Repo" in labels and "Claim" in labels:
        return "Evaluate the linked implementation first, but require a small reproduction pass and repo-health review before adoption."
    if "Claim" in labels:
        return "Use the claim evidence for research prioritization, then ingest linked repos before an engineering decision."
    return "Treat the answer as exploratory until stronger claim and implementation evidence is available."


def _next_checks(labels: set[str]) -> list[str]:
    checks = ["Open the saved Cypher evidence-path query in Neo4j Browser.", "Inspect raw source records for the cited spans."]
    if "Benchmark" not in labels:
        checks.append("Add benchmark or dataset extraction for the target method.")
    if "Repo" not in labels:
        checks.append("Run GitHub ingestion for implementation evidence.")
    return checks


def _faithfulness_estimate(answer: Answer) -> float:
    if not answer.citations:
        return 0.0
    cited = min(1.0, len(answer.citations) / 4)
    chain = _chain_completeness(answer)
    return round((0.6 * cited) + (0.4 * chain), 3)


def _chain_completeness(answer: Answer) -> float:
    if not answer.evidence_chain:
        return 0.0
    best = max(len(path) for path in answer.evidence_chain)
    return round(min(1.0, best / 7), 3)
