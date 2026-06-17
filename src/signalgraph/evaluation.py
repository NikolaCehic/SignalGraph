from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .answering import AnswerSynthesizer
from .config import ProjectPaths
from .models import GraphArtifact, GraphEdge
from .retrieval import RetrievalResult, Retriever
from .utils import append_jsonl, read_json, write_json


REQUIRED_EVAL_CATEGORIES = [
    "entity-specific",
    "comparison",
    "broad landscape",
    "structured",
    "decision-memo",
    "adversarial/uncertainty",
]

ABLATION_MODES = {
    "vector-only": "vector",
    "hybrid": "hybrid",
    "local": "local",
    "global": "global",
    "DRIFT-style": "drift",
    "best-route": "auto",
}

GRAPH_METRIC_KEYS = [
    "required_node_recall",
    "required_edge_recall",
    "path_validity",
    "provenance_coverage",
    "merge_quality",
    "staleness_detection",
]

CORE_METRIC_KEYS = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "graph_path_recall",
    "evidence_chain_completeness",
    "citation_accuracy",
    "conflict_awareness",
    "latency_ms",
    "cost_usd_estimate",
]


@dataclass
class EvalQuestion:
    id: str
    query: str
    category: str
    expected_answer_outline: str
    required_evidence_nodes: list[str]
    required_source_types: list[str]
    unacceptable_hallucinations: list[str]
    ideal_graph_paths: list[list[str]]
    required_evidence_edges: list[str] = field(default_factory=list)
    required_merge_pairs: list[list[str]] = field(default_factory=list)
    stale_sensitive: bool = False


def builtin_eval_questions() -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []

    def add(
        qid: str,
        query: str,
        category: str,
        outline: str,
        nodes: list[str],
        source_types: list[str],
        bad: list[str],
        paths: list[list[str]],
        edges: list[str],
        merge_pairs: list[list[str]] | None = None,
        stale_sensitive: bool = False,
    ) -> None:
        questions.append(
            EvalQuestion(
                id=qid,
                query=query,
                category=category,
                expected_answer_outline=outline,
                required_evidence_nodes=nodes,
                required_source_types=source_types,
                unacceptable_hallucinations=bad,
                ideal_graph_paths=paths,
                required_evidence_edges=edges,
                required_merge_pairs=merge_pairs or [],
                stale_sensitive=stale_sensitive,
            )
        )

    entity_specs = [
        ("GraphRAG paper", "paper:sample:graphrag", "method:graphrag", ["INTRODUCES", "CLAIMS", "HAS_CHUNK"]),
        ("Microsoft GraphRAG repo", "repo:microsoft/graphrag", "method:graphrag", ["IMPLEMENTS", "CLAIMS", "HAS_CHUNK"]),
        ("MemGPT paper", "paper:sample:memgpt", "method:agent-memory", ["INTRODUCES", "CLAIMS", "HAS_CHUNK"]),
        ("cpacker MemGPT repo", "repo:cpacker/memgpt", "method:agent-memory", ["IMPLEMENTS", "CLAIMS", "HAS_CHUNK"]),
        ("RAG evaluation paper", "paper:sample:rag-eval", "method:rag-evaluation", ["INTRODUCES", "EVALUATES_ON", "CLAIMS"]),
        ("SignalGraph agent-memory repo", "repo:example/signalgraph-agent-memory", "method:hybrid-retrieval", ["IMPLEMENTS", "CLAIMS", "HAS_CHUNK"]),
        ("Graph-aware agent memory paper", "paper:semanticscholar:s2-graphrag-memory", "method:graphrag", ["INTRODUCES", "CITES", "EVALUATES_ON"]),
        ("GraphRAG Foundations paper", "paper:semanticscholar:s2-ref-1", "method:graphrag", ["INTRODUCES", "EVALUATES_ON"]),
        ("Repository Exploration Agents paper", "paper:semanticscholar:s2-cite-1", "dataset:research-dataset", ["EVALUATES_ON", "BELONGS_TO_COMMUNITY"]),
        ("GraphRAG method node", "method:graphrag", "community:agent-architecture", ["BELONGS_TO_COMMUNITY", "SIMILAR_TO"]),
        ("Agent Memory method node", "method:agent-memory", "community:agent-architecture", ["BELONGS_TO_COMMUNITY", "SIMILAR_TO"]),
        ("Hybrid Retrieval method node", "method:hybrid-retrieval", "community:agent-architecture", ["BELONGS_TO_COMMUNITY", "SIMILAR_TO"]),
        ("RAG Evaluation method node", "method:rag-evaluation", "benchmark:faithfulness", ["BELONGS_TO_COMMUNITY", "SIMILAR_TO"]),
        ("Hugging Face GraphRAG model", "model:huggingface:example-graphrag-agent-memory", "dataset:huggingface:example-agent-memory-eval", ["BELONGS_TO_COMMUNITY"]),
        ("Agent memory eval dataset", "dataset:huggingface:example-agent-memory-eval", "community:agent-architecture", ["BELONGS_TO_COMMUNITY"]),
    ]
    for index, (name, primary, secondary, edges) in enumerate(entity_specs, start=1):
        add(
            f"ENT-{index:02d}",
            f"What evidence does SignalGraph have for {name}?",
            "entity-specific",
            f"Identify {name}, cite its connected method or asset evidence, and call out source-grounded caveats.",
            [primary, secondary],
            ["Paper", "Repo", "Method", "Claim", "DocumentChunk", "Dataset", "Model", "Community"],
            ["Treating entity mentions as adoption proof without cited graph evidence."],
            [["Paper", "Method", "Claim", "DocumentChunk"], ["Repo", "Method", "Claim", "DocumentChunk"]],
            edges,
            merge_pairs=[["method:graphrag", "method:community-detection"]] if primary == "method:graphrag" else [],
            stale_sensitive=primary.startswith("repo:"),
        )

    comparisons = [
        ("GraphRAG", "agent memory", ["method:graphrag", "method:agent-memory"], ["SIMILAR_TO", "BELONGS_TO_COMMUNITY"]),
        ("vector-only RAG", "GraphRAG", ["method:rag-evaluation", "method:graphrag"], ["INTRODUCES", "CLAIMS"]),
        ("community detection", "hybrid retrieval", ["method:community-detection", "method:hybrid-retrieval"], ["SIMILAR_TO", "BELONGS_TO_COMMUNITY"]),
        ("Text2Cypher", "hybrid retrieval", ["method:text2cypher", "method:hybrid-retrieval"], ["SIMILAR_TO"]),
        ("Microsoft GraphRAG repo", "SignalGraph agent-memory repo", ["repo:microsoft/graphrag", "repo:example/signalgraph-agent-memory"], ["SIMILAR_TO", "IMPLEMENTS"]),
        ("MemGPT repo", "SignalGraph agent-memory repo", ["repo:cpacker/memgpt", "repo:example/signalgraph-agent-memory"], ["SIMILAR_TO", "IMPLEMENTS"]),
        ("RAG faithfulness", "context recall", ["benchmark:faithfulness", "benchmark:context-recall"], ["EVALUATES_ON"]),
        ("paper evidence", "repo evidence", ["paper:sample:graphrag", "repo:microsoft/graphrag"], ["IMPLEMENTS", "CLAIMS"]),
        ("dataset cards", "model cards", ["dataset:huggingface:example-agent-memory-eval", "model:huggingface:example-graphrag-agent-memory"], ["BELONGS_TO_COMMUNITY"]),
        ("GraphRAG Foundations", "Graph-aware Agent Memory", ["paper:semanticscholar:s2-ref-1", "paper:semanticscholar:s2-graphrag-memory"], ["CITES", "INTRODUCES"]),
        ("RAG evaluation", "production repo health", ["paper:sample:rag-eval", "repo:example/signalgraph-agent-memory"], ["CLAIMS", "HAS_CHUNK"]),
        ("claims", "source chunks", ["claim:8caf5dceb093", "chunk:87358b556906"], ["SUPPORTED_BY"]),
        ("benchmarks", "datasets", ["benchmark:faithfulness", "dataset:research-dataset"], ["EVALUATES_ON", "BELONGS_TO_COMMUNITY"]),
        ("local search", "global search", ["method:graphrag", "community:agent-architecture"], ["BELONGS_TO_COMMUNITY"]),
        ("DRIFT-style routing", "structured lookup", ["community:agent-architecture", "repo:microsoft/graphrag"], ["BELONGS_TO_COMMUNITY", "IMPLEMENTS"]),
    ]
    for index, (left, right, nodes, edges) in enumerate(comparisons, start=1):
        add(
            f"CMP-{index:02d}",
            f"Compare {left} and {right} for production retrieval decisions.",
            "comparison",
            "Give a side-by-side answer that distinguishes graph evidence, implementation signals, and evaluation caveats.",
            nodes,
            ["Paper", "Repo", "Method", "Claim", "DocumentChunk", "Benchmark", "Dataset", "Community"],
            ["Claiming either side is always superior without source-specific limits."],
            [["Method", "SIMILAR_TO", "Method"], ["Paper", "Claim", "DocumentChunk"], ["Repo", "Method", "Paper"]],
            edges,
            merge_pairs=[["method:text2cypher", "method:hybrid-retrieval"]] if "Text2Cypher" in left else [],
        )

    landscapes = [
        ("main themes in GraphRAG and agent memory research", ["community:agent-architecture", "method:graphrag"]),
        ("approaches to RAG evaluation", ["method:rag-evaluation", "benchmark:faithfulness"]),
        ("implementation risks for research-to-production assistants", ["repo:example/signalgraph-agent-memory", "claim:891ee61a3a4a"]),
        ("community-report signals for global questions", ["community:agent-architecture", "method:community-detection"]),
        ("benchmark and dataset coverage in the corpus", ["benchmark:context-recall", "dataset:research-dataset"]),
        ("repo health patterns across GraphRAG implementations", ["repo:microsoft/graphrag", "repo:cpacker/memgpt"]),
        ("how claims connect papers to source chunks", ["claim:5b33c60ba8dd", "chunk:fc6e5b8d6191"]),
        ("how methods connect repos and papers", ["method:graphrag", "repo:microsoft/graphrag"]),
        ("where uncertainty appears in retrieved evidence", ["claim:d12728f4e06b", "claim:891ee61a3a4a"]),
        ("evidence chains useful for decision memos", ["paper:sample:graphrag", "repo:microsoft/graphrag"]),
        ("the practical asset layer for models and datasets", ["model:huggingface:example-graphrag-agent-memory", "dataset:huggingface:example-agent-memory-eval"]),
        ("graph traversal patterns behind SignalGraph answers", ["method:graphrag", "community:agent-architecture"]),
        ("citation and contradiction patterns in the sample graph", ["claim:34b0a822959c", "claim:d12728f4e06b"]),
        ("methods most central to agent architecture", ["method:agent-memory", "method:hybrid-retrieval"]),
        ("source provenance patterns across the graph", ["chunk:87358b556906", "claim:8caf5dceb093"]),
    ]
    for index, (topic, nodes) in enumerate(landscapes, start=1):
        add(
            f"LAN-{index:02d}",
            f"What are the {topic}?",
            "broad landscape",
            "Summarize broad corpus themes from community reports and representative source-grounded evidence.",
            nodes,
            ["Community", "Method", "Claim", "DocumentChunk", "Paper", "Repo", "Benchmark", "Dataset"],
            ["Inventing a trend that is not supported by retrieved communities or citations."],
            [["Community", "Method", "Claim", "DocumentChunk"], ["Paper", "BELONGS_TO_COMMUNITY", "Community"]],
            ["BELONGS_TO_COMMUNITY", "CLAIMS", "SUPPORTED_BY"],
        )

    structured = [
        ("Which repos implement papers about GraphRAG?", ["repo:microsoft/graphrag", "paper:sample:graphrag"], ["IMPLEMENTS"]),
        ("Which repos implement papers about agent memory?", ["repo:cpacker/memgpt", "paper:sample:memgpt"], ["IMPLEMENTS"]),
        ("Which methods are connected to benchmark evidence?", ["method:rag-evaluation", "benchmark:faithfulness"], ["EVALUATES_ON"]),
        ("Which papers cite or are cited by Graph-aware Agent Memory?", ["paper:semanticscholar:s2-graphrag-memory", "paper:semanticscholar:s2-ref-1"], ["CITES"]),
        ("Which repos have source chunks tied to production risk?", ["repo:example/signalgraph-agent-memory", "chunk:c484a3786fb3"], ["HAS_CHUNK"]),
        ("Which methods belong to the agent architecture community?", ["method:graphrag", "community:agent-architecture"], ["BELONGS_TO_COMMUNITY"]),
        ("Which assets belong to the Hugging Face evaluation layer?", ["model:huggingface:example-graphrag-agent-memory", "dataset:huggingface:example-agent-memory-eval"], ["BELONGS_TO_COMMUNITY"]),
        ("Which paper evaluates on faithfulness and context recall?", ["paper:sample:rag-eval", "benchmark:faithfulness"], ["EVALUATES_ON"]),
        ("Which repo claims recent benchmark coverage?", ["repo:example/signalgraph-agent-memory", "claim:aafc12fdaaf8"], ["CLAIMS", "SUPPORTED_BY"]),
        ("Which graph paths connect Microsoft GraphRAG to RAG evaluation?", ["repo:microsoft/graphrag", "paper:sample:rag-eval"], ["IMPLEMENTS"]),
    ]
    for index, (query, nodes, edges) in enumerate(structured, start=1):
        add(
            f"STR-{index:02d}",
            query,
            "structured",
            "Return typed graph matches with explicit node IDs and stored relationship evidence.",
            nodes,
            ["Repo", "Paper", "Method", "Benchmark", "Dataset", "DocumentChunk", "Claim", "Community"],
            ["Returning entities without a typed graph relationship."],
            [["Repo", "IMPLEMENTS", "Paper"], ["Paper", "EVALUATES_ON", "Benchmark"], ["Node", "BELONGS_TO_COMMUNITY", "Community"]],
            edges,
        )

    decisions = [
        ("Which GraphRAG implementation should a startup evaluate first for enterprise support automation?", ["method:graphrag", "repo:microsoft/graphrag"], ["IMPLEMENTS", "CLAIMS"]),
        ("Should a team use agent memory or GraphRAG for long-term assistant context?", ["method:agent-memory", "method:graphrag"], ["SIMILAR_TO", "CLAIMS"]),
        ("Should the SignalGraph sample repo be treated as production-ready?", ["repo:example/signalgraph-agent-memory", "claim:891ee61a3a4a"], ["CLAIMS", "HAS_CHUNK"]),
        ("Which evaluation route should be used before adopting a GraphRAG repo?", ["method:rag-evaluation", "benchmark:faithfulness"], ["EVALUATES_ON", "CLAIMS"]),
        ("Should vector-only retrieval be considered enough for broad landscape questions?", ["method:rag-evaluation", "community:agent-architecture"], ["BELONGS_TO_COMMUNITY", "CLAIMS"]),
        ("Which implementation has the strongest source-grounded repo-health signal?", ["repo:microsoft/graphrag", "repo:cpacker/memgpt"], ["SIMILAR_TO", "HAS_CHUNK"]),
        ("Should Hugging Face assets be part of a production adoption review?", ["model:huggingface:example-graphrag-agent-memory", "dataset:huggingface:example-agent-memory-eval"], ["BELONGS_TO_COMMUNITY"]),
        ("What should be checked before relying on GraphRAG community reports?", ["community:agent-architecture", "method:community-detection"], ["BELONGS_TO_COMMUNITY", "SIMILAR_TO"]),
        ("Which evidence chain best supports a startup GraphRAG pilot?", ["paper:sample:graphrag", "repo:microsoft/graphrag"], ["INTRODUCES", "IMPLEMENTS", "SUPPORTED_BY"]),
        ("Should Text2Cypher be evaluated as a companion to hybrid retrieval?", ["method:text2cypher", "method:hybrid-retrieval"], ["SIMILAR_TO"]),
    ]
    for index, (query, nodes, edges) in enumerate(decisions, start=1):
        add(
            f"DEC-{index:02d}",
            query,
            "decision-memo",
            "Make a conditional recommendation with citations, graph evidence, missing evidence, and next checks.",
            nodes,
            ["Repo", "Paper", "Method", "Claim", "DocumentChunk", "Benchmark", "Dataset", "Community"],
            ["A blanket production adoption recommendation without repo-health, benchmark, or staleness caveats."],
            [["Paper", "Method", "Repo", "Claim", "DocumentChunk"], ["Community", "Method", "Repo"]],
            edges,
            merge_pairs=[["method:graphrag", "method:community-detection"]] if "community reports" in query.lower() else [],
            stale_sensitive=True,
        )

    adversarial = [
        ("Which repo is definitely production-ready with no risks?", ["repo:microsoft/graphrag", "claim:891ee61a3a4a"], ["CLAIMS", "HAS_CHUNK"]),
        ("Can SignalGraph prove an uncited benchmark win for every GraphRAG method?", ["method:graphrag", "benchmark:faithfulness"], ["EVALUATES_ON"]),
        ("Should stale repo health claims be trusted without last-commit evidence?", ["repo:cpacker/memgpt", "repo:microsoft/graphrag"], ["HAS_CHUNK"]),
        ("What answer should be given when required source spans are missing?", ["claim:d12728f4e06b", "chunk:9300a3545ee0"], ["SUPPORTED_BY"]),
        ("Can agent memory be declared superior to GraphRAG for all assistant workloads?", ["method:agent-memory", "method:graphrag"], ["SIMILAR_TO"]),
    ]
    for index, (query, nodes, edges) in enumerate(adversarial, start=1):
        add(
            f"ADV-{index:02d}",
            query,
            "adversarial/uncertainty",
            "Refuse overconfident claims, identify missing evidence, and anchor any answer in citations.",
            nodes,
            ["Repo", "Paper", "Method", "Claim", "DocumentChunk", "Benchmark"],
            ["Definite universal claims, uncited benchmark wins, or stale repo-health assertions."],
            [["Repo", "Claim", "DocumentChunk"], ["Method", "SIMILAR_TO", "Method"]],
            edges,
            stale_sensitive=True,
        )

    return questions


BUILTIN_EVAL_QUESTIONS = builtin_eval_questions()


class EvalRunner:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.retriever = Retriever(paths)
        self.synthesizer = AnswerSynthesizer()

    def run(self, questions: list[EvalQuestion] | None = None) -> dict[str, Any]:
        if questions is None:
            cached = _read_valid_cached_payload(self.paths)
            if cached is not None:
                return cached
        questions = questions or BUILTIN_EVAL_QUESTIONS
        _validate_corpus(questions)
        self.paths.ensure()
        write_json(self.paths.eval_dir / "signalgraph_eval_corpus.json", [asdict(question) for question in questions])

        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for question in questions:
            for ablation, mode in ABLATION_MODES.items():
                result = self.retriever.retrieve(question.query, mode=mode, limit=4)
                answer = self.synthesizer.synthesize(result)
                answer_dict = answer.to_dict()
                scores = _score_result(result, answer_dict, question, self.retriever.graph)
                row = _eval_row(question, ablation, result, scores)
                rows.append(row)
                traces.append(_rich_trace(question, ablation, result, answer_dict, scores))

        summary = _summarize(rows, questions)
        payload = {
            "questions": [asdict(question) for question in questions],
            "ablations": list(ABLATION_MODES),
            "graph_metrics": GRAPH_METRIC_KEYS,
            "rows": rows,
            "summary": summary,
            "artifacts": _artifact_paths(self.paths),
        }
        write_json(self.paths.eval_results_path, payload)
        _write_eval_csv(self.paths.retrieval_comparison_path, rows)
        _write_trace_jsonl(self.paths.traces_dir / "eval_query_traces.jsonl", traces)
        _write_report_artifacts(self.paths, rows, summary, traces, self.retriever.graph)
        return payload


def _read_valid_cached_payload(paths: ProjectPaths) -> dict[str, Any] | None:
    required_artifacts = [
        paths.eval_results_path,
        paths.eval_dir / "signalgraph_eval_corpus.json",
        paths.retrieval_comparison_path,
        paths.reports_dir / "eval_summary.md",
        paths.reports_dir / "retrieval_quality.md",
        paths.reports_dir / "retrieval_quality.csv",
        paths.artifacts_dir / "retrieval_quality.json",
        paths.reports_dir / "generation_quality.md",
        paths.reports_dir / "generation_quality.csv",
        paths.artifacts_dir / "generation_quality.json",
        paths.reports_dir / "system_health.md",
        paths.artifacts_dir / "system_health.json",
        paths.reports_dir / "failure_cases.md",
        paths.reports_dir / "failure_cases.csv",
        paths.artifacts_dir / "failure_cases.json",
        paths.traces_dir / "eval_query_traces.jsonl",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required_artifacts):
        return None
    try:
        payload = read_json(paths.eval_results_path)
    except Exception:
        return None
    summary = payload.get("summary", {})
    if summary.get("question_count") != 70:
        return None
    if summary.get("row_count") != 70 * len(ABLATION_MODES):
        return None
    if set(summary.get("category_counts", {})) != set(REQUIRED_EVAL_CATEGORIES):
        return None
    if set(summary.get("ablations", [])) != set(ABLATION_MODES):
        return None
    if not set(GRAPH_METRIC_KEYS) <= set(payload.get("graph_metrics", [])):
        return None
    return payload


def _validate_corpus(questions: list[EvalQuestion]) -> None:
    if not 50 <= len(questions) <= 100:
        raise ValueError(f"eval corpus must contain 50-100 questions; found {len(questions)}")
    categories = {question.category for question in questions}
    missing = [category for category in REQUIRED_EVAL_CATEGORIES if category not in categories]
    if missing:
        raise ValueError(f"eval corpus missing required categories: {', '.join(missing)}")


def _score_result(result: RetrievalResult, answer: dict[str, Any], question: EvalQuestion, graph: GraphArtifact) -> dict[str, float]:
    scores = {
        "faithfulness": _faithfulness(answer),
        "answer_relevance": _answer_relevance(answer.get("answer", ""), question),
        "context_precision": _context_precision(result, question),
        "context_recall": _context_recall(result, question),
        "graph_path_recall": _graph_path_recall(result, question),
        "evidence_chain_completeness": _evidence_chain_completeness(answer),
        "citation_accuracy": _citation_accuracy(answer),
        "conflict_awareness": _conflict_awareness(answer, question),
        "required_node_recall": _required_node_recall(result, question),
        "required_edge_recall": _required_edge_recall(result, question),
        "path_validity": _path_validity(result, graph),
        "provenance_coverage": _provenance_coverage(result),
        "merge_quality": _merge_quality(question, graph),
        "staleness_detection": _staleness_detection(result, answer, question),
        "latency_ms": result.latency_ms,
        "cost_usd_estimate": _cost_estimate(result, answer),
    }
    return scores


def _eval_row(question: EvalQuestion, ablation: str, result: RetrievalResult, scores: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": question.id,
        "query": question.query,
        "category": question.category,
        "ablation": ablation,
        "mode": result.mode,
        "route": result.route.route,
        "retrieved_node_count": len(_retrieved_node_ids(result)),
        "retrieved_chunk_count": len(_retrieved_chunk_ids(result)),
        "retrieved_edge_count": len(_retrieved_edge_types(result)),
    }
    for key in [*CORE_METRIC_KEYS, *GRAPH_METRIC_KEYS]:
        row[key] = scores[key]
    return row


def _rich_trace(question: EvalQuestion, ablation: str, result: RetrievalResult, answer: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
    candidates = result.candidates
    prompt_tokens = sum(max(1, len(candidate.text.split())) for candidate in candidates) + max(1, len(result.query.split()))
    completion_tokens = max(1, len(answer.get("answer", "").split()))
    return {
        "trace_type": "signalgraph_eval_query",
        "question_id": question.id,
        "user_query": result.query,
        "query_category": question.category,
        "ablation": ablation,
        "mode": result.mode,
        "route": result.route.route,
        "route_detail": result.route.__dict__,
        "retrieved_node_ids": _retrieved_node_ids(result),
        "retrieved_chunk_ids": _retrieved_chunk_ids(result),
        "graph_traversals": [candidate.path for candidate in candidates if candidate.path],
        "vector_scores": {candidate.node_id: _feature(candidate, "semantic_relevance", "vector_score") for candidate in candidates},
        "full_text_scores": {candidate.node_id: _feature(candidate, "lexical_relevance", "fulltext_score") for candidate in candidates},
        "reranker_scores": {candidate.node_id: candidate.score for candidate in candidates},
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "answer": answer.get("answer", ""),
        "citations": answer.get("citations", []),
        "eval_scores": scores,
        "latency_ms": result.latency_ms,
        "cost_usd_estimate": scores["cost_usd_estimate"],
    }


def _feature(candidate, primary: str, fallback: str) -> float:
    value = candidate.features.get(primary, candidate.features.get(fallback, 0.0))
    return round(float(value or 0.0), 4)


def _context_precision(result: RetrievalResult, question: EvalQuestion) -> float:
    if not result.candidates:
        return 0.0
    required_labels = set(question.required_source_types)
    useful = sum(1 for candidate in result.candidates if required_labels & set(candidate.labels))
    return round(useful / len(result.candidates), 3)


def _context_recall(result: RetrievalResult, question: EvalQuestion) -> float:
    required = set(question.required_evidence_nodes)
    if not required:
        return 1.0
    seen = set(_retrieved_node_ids(result))
    return round(len(required & seen) / len(required), 3)


def _required_node_recall(result: RetrievalResult, question: EvalQuestion) -> float:
    return _context_recall(result, question)


def _required_edge_recall(result: RetrievalResult, question: EvalQuestion) -> float:
    required = set(question.required_evidence_edges)
    if not required:
        return 1.0
    seen = set(_retrieved_edge_types(result))
    return round(len(required & seen) / len(required), 3)


def _graph_path_recall(result: RetrievalResult, question: EvalQuestion) -> float:
    ideal_labels = {label for path in question.ideal_graph_paths for label in path if label.isalpha()}
    if not ideal_labels:
        return 1.0
    retrieved_labels: set[str] = set()
    for candidate in result.candidates:
        retrieved_labels.update(candidate.labels)
        for part in candidate.path:
            retrieved_labels.update(_label_from_id(part))
    return round(len(retrieved_labels & ideal_labels) / len(ideal_labels), 3)


def _path_validity(result: RetrievalResult, graph: GraphArtifact) -> float:
    paths = [candidate.path for candidate in result.candidates if len(candidate.path) > 1]
    if not paths:
        return 1.0
    edges = {(edge.start_id, edge.type, edge.end_id) for edge in graph.edges}
    valid = sum(1 for path in paths if _decorated_path_is_valid(path, edges))
    return round(valid / len(paths), 3)


def _decorated_path_is_valid(path: list[str], edges: set[tuple[str, str, str]]) -> bool:
    if len(path) == 1:
        return True
    if len(path) < 3 or len(path) % 2 == 0:
        return False
    for start, rel, end in zip(path[0::2], path[1::2], path[2::2]):
        if rel.startswith("<-"):
            expected = (end, rel[2:], start)
        else:
            expected = (start, rel, end)
        if expected not in edges:
            return False
    return True


def _provenance_coverage(result: RetrievalResult) -> float:
    if not result.candidates:
        return 0.0
    covered = 0
    for candidate in result.candidates:
        citation = candidate.citation
        if citation.get("source_url") and citation.get("source_span"):
            covered += 1
    return round(covered / len(result.candidates), 3)


def _merge_quality(question: EvalQuestion, graph: GraphArtifact) -> float:
    decisions = graph.entity_resolution_decisions
    if question.required_merge_pairs:
        matched = 0
        for left, right in question.required_merge_pairs:
            if any({decision.left_id, decision.right_id} == {left, right} and decision.state in {"exact", "probable"} for decision in decisions):
                matched += 1
        return round(matched / len(question.required_merge_pairs), 3)
    if not decisions:
        return 1.0
    accepted = sum(1 for decision in decisions if decision.state in {"exact", "probable"} and not decision.review_required)
    return round(accepted / len(decisions), 3)


def _staleness_detection(result: RetrievalResult, answer: dict[str, Any], question: EvalQuestion) -> float:
    if not question.stale_sensitive:
        return 1.0
    stale_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.features.get("freshness", 1.0) < 0.55 or "stale" in candidate.text.lower() or "risk" in candidate.text.lower()
    ]
    notes = " ".join(answer.get("conflicts_or_missing_evidence", []) + answer.get("next_checks", [])).lower()
    if stale_candidates and any(term in notes for term in ["stale", "freshness", "repo-health", "conditional", "risk", "review"]):
        return 1.0
    if stale_candidates:
        return 0.35
    return 0.8


def _evidence_chain_completeness(answer: dict[str, Any]) -> float:
    chains = answer.get("evidence_chain", [])
    if not chains:
        return 0.0
    best = max(len(chain) for chain in chains)
    return round(min(1.0, best / 7), 3)


def _faithfulness(answer: dict[str, Any]) -> float:
    citations = answer.get("citations", [])
    if not citations:
        return 0.0
    supported = sum(1 for citation in citations if citation.get("source_url") and citation.get("source_span"))
    return round(supported / len(citations), 3)


def _answer_relevance(answer: str, question: EvalQuestion) -> float:
    answer_words = set(answer.lower().split())
    outline_words = {word.strip(".,;:") for word in question.expected_answer_outline.lower().split() if len(word.strip(".,;:")) > 4}
    if not outline_words:
        return 0.5
    return round(min(1.0, len(answer_words & outline_words) / max(1, len(outline_words) * 0.45)), 3)


def _citation_accuracy(answer: dict[str, Any]) -> float:
    citations = answer.get("citations", [])
    if not citations:
        return 0.0
    valid = [citation for citation in citations if citation.get("node_id") and citation.get("source_url")]
    return round(len(valid) / len(citations), 3)


def _conflict_awareness(answer: dict[str, Any], question: EvalQuestion) -> float:
    notes = " ".join(answer.get("conflicts_or_missing_evidence", []) + answer.get("next_checks", [])).lower()
    needs_caution = question.category in {"adversarial/uncertainty", "decision-memo"} or question.stale_sensitive
    if needs_caution:
        return 1.0 if any(word in notes for word in ["missing", "caution", "benchmark", "repo-health", "conditional", "stale", "risk"]) else 0.3
    return 0.8 if notes else 0.5


def _cost_estimate(result: RetrievalResult, answer: dict[str, Any]) -> float:
    token_estimate = len(result.query.split()) + sum(len(candidate.text.split()) for candidate in result.candidates) + len(answer.get("answer", "").split())
    return round(token_estimate * 0.0, 6)


def _retrieved_node_ids(result: RetrievalResult) -> list[str]:
    seen: list[str] = []
    for candidate in result.candidates:
        for value in [candidate.node_id, *candidate.path]:
            if ":" not in value or value in seen or _is_relationship_token(value):
                continue
            seen.append(value)
    return seen


def _retrieved_chunk_ids(result: RetrievalResult) -> list[str]:
    return [node_id for node_id in _retrieved_node_ids(result) if node_id.startswith("chunk:")]


def _retrieved_edge_types(result: RetrievalResult) -> list[str]:
    seen: list[str] = []
    for candidate in result.candidates:
        for part in candidate.path:
            rel = part[2:] if part.startswith("<-") else part
            if _is_relationship_token(part) and rel not in seen:
                seen.append(rel)
    return seen


def _is_relationship_token(value: str) -> bool:
    token = value[2:] if value.startswith("<-") else value
    return token.isupper() and ":" not in token


def _label_from_id(part: str) -> set[str]:
    if part.startswith("paper:"):
        return {"Paper"}
    if part.startswith("repo:"):
        return {"Repo"}
    if part.startswith("method:"):
        return {"Method"}
    if part.startswith("claim:"):
        return {"Claim"}
    if part.startswith("chunk:"):
        return {"DocumentChunk"}
    if part.startswith("benchmark:"):
        return {"Benchmark"}
    if part.startswith("dataset:"):
        return {"Dataset"}
    if part.startswith("model:"):
        return {"Model"}
    if part.startswith("community:"):
        return {"Community"}
    return set()


def _summarize(rows: list[dict[str, Any]], questions: list[EvalQuestion]) -> dict[str, Any]:
    metrics = [*CORE_METRIC_KEYS, *GRAPH_METRIC_KEYS]
    numeric_metrics = [key for key in metrics if key not in {"latency_ms", "cost_usd_estimate"}]
    by_ablation: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, float]] = {}
    for ablation in ABLATION_MODES:
        subset = [row for row in rows if row["ablation"] == ablation]
        by_ablation[ablation] = _average_metrics(subset, metrics)
    for category in REQUIRED_EVAL_CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = _average_metrics(subset, numeric_metrics)
    route_distribution: dict[str, int] = {}
    for row in rows:
        route_distribution[row["route"]] = route_distribution.get(row["route"], 0) + 1
    metrics_average = _average_metrics(rows, metrics)
    graph_metrics_average = _average_metrics(rows, GRAPH_METRIC_KEYS)
    return {
        "question_count": len(questions),
        "category_counts": {category: sum(1 for question in questions if question.category == category) for category in REQUIRED_EVAL_CATEGORIES},
        "ablation_count": len(ABLATION_MODES),
        "ablations": list(ABLATION_MODES),
        "row_count": len(rows),
        **metrics_average,
        "metrics": metrics_average,
        "graph_metrics": graph_metrics_average,
        "by_ablation": by_ablation,
        "by_category": by_category,
        "route_distribution": route_distribution,
    }


def _average_metrics(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: round(sum(float(row[key]) for row in rows) / len(rows), 3) for key in keys}


def _artifact_paths(paths: ProjectPaths) -> dict[str, str]:
    return {
        "corpus_json": str(paths.eval_dir / "signalgraph_eval_corpus.json"),
        "eval_json": str(paths.eval_results_path),
        "retrieval_comparison_csv": str(paths.retrieval_comparison_path),
        "retrieval_quality_md": str(paths.reports_dir / "retrieval_quality.md"),
        "retrieval_quality_csv": str(paths.reports_dir / "retrieval_quality.csv"),
        "retrieval_quality_json": str(paths.artifacts_dir / "retrieval_quality.json"),
        "generation_quality_md": str(paths.reports_dir / "generation_quality.md"),
        "generation_quality_csv": str(paths.reports_dir / "generation_quality.csv"),
        "generation_quality_json": str(paths.artifacts_dir / "generation_quality.json"),
        "system_health_md": str(paths.reports_dir / "system_health.md"),
        "system_health_json": str(paths.artifacts_dir / "system_health.json"),
        "failure_cases_md": str(paths.reports_dir / "failure_cases.md"),
        "failure_cases_csv": str(paths.reports_dir / "failure_cases.csv"),
        "failure_cases_json": str(paths.artifacts_dir / "failure_cases.json"),
        "trace_jsonl": str(paths.traces_dir / "eval_query_traces.jsonl"),
    }


def _write_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace_jsonl(path: Path, traces: list[dict[str, Any]]) -> None:
    path.unlink(missing_ok=True)
    for trace in traces:
        append_jsonl(path, trace)


def _write_report_artifacts(paths: ProjectPaths, rows: list[dict[str, Any]], summary: dict[str, Any], traces: list[dict[str, Any]], graph: GraphArtifact) -> None:
    _write_eval_markdown(paths.reports_dir / "eval_summary.md", rows, summary)
    _write_retrieval_quality(paths, rows, summary)
    _write_generation_quality(paths, rows, summary)
    _write_failure_cases(paths, rows, traces)
    _write_system_health(paths, rows, summary, graph)


def _write_eval_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = ["# SignalGraph Eval Summary", "", f"- Questions: {summary['question_count']}", f"- Ablations: {', '.join(summary['ablations'])}", f"- Result rows: {summary['row_count']}", "", "## Category Counts", ""]
    for category, count in summary["category_counts"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Graph Metrics", ""])
    for key, value in summary["graph_metrics"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Ablation Summary", "", "| Ablation | Faithfulness | Context Recall | Required Node Recall | Required Edge Recall | Path Validity | Latency ms |", "|---|---:|---:|---:|---:|---:|---:|"])
    for ablation, metrics in summary["by_ablation"].items():
        lines.append(
            f"| {ablation} | {metrics['faithfulness']} | {metrics['context_recall']} | {metrics['required_node_recall']} | {metrics['required_edge_recall']} | {metrics['path_validity']} | {metrics['latency_ms']} |"
        )
    lines.extend(["", "## Lowest Scoring Cases", "", "| ID | Category | Ablation | Required Nodes | Required Edges | Path Validity |", "|---|---|---|---:|---:|---:|"])
    weak = sorted(rows, key=lambda row: (row["required_node_recall"] + row["required_edge_recall"] + row["path_validity"], row["faithfulness"]))[:10]
    for row in weak:
        lines.append(f"| {row['id']} | {row['category']} | {row['ablation']} | {row['required_node_recall']} | {row['required_edge_recall']} | {row['path_validity']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_retrieval_quality(paths: ProjectPaths, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    keys = ["context_precision", "context_recall", "graph_path_recall", "required_node_recall", "required_edge_recall", "path_validity", "provenance_coverage", "latency_ms"]
    records = [{"ablation": ablation, **{key: metrics[key] for key in keys}} for ablation, metrics in summary["by_ablation"].items()]
    write_json(paths.artifacts_dir / "retrieval_quality.json", {"by_ablation": records, "route_distribution": summary["route_distribution"]})
    _write_records_csv(paths.reports_dir / "retrieval_quality.csv", records)
    lines = ["# Retrieval Quality", "", "## Route Distribution", ""]
    for route, count in sorted(summary["route_distribution"].items()):
        lines.append(f"- `{route}`: {count}")
    lines.extend(["", "## Metrics By Ablation", "", "| Ablation | Context Precision | Context Recall | Required Nodes | Required Edges | Path Validity | Provenance |", "|---|---:|---:|---:|---:|---:|---:|"])
    for record in records:
        lines.append(
            f"| {record['ablation']} | {record['context_precision']} | {record['context_recall']} | {record['required_node_recall']} | {record['required_edge_recall']} | {record['path_validity']} | {record['provenance_coverage']} |"
        )
    (paths.reports_dir / "retrieval_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_generation_quality(paths: ProjectPaths, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    keys = ["faithfulness", "answer_relevance", "evidence_chain_completeness", "citation_accuracy", "conflict_awareness", "staleness_detection"]
    records = [{"ablation": ablation, **{key: metrics[key] for key in keys}} for ablation, metrics in summary["by_ablation"].items()]
    unsupported = sum(1 for row in rows if row["faithfulness"] < 1.0 or row["citation_accuracy"] < 1.0)
    write_json(paths.artifacts_dir / "generation_quality.json", {"by_ablation": records, "unsupported_claim_count": unsupported})
    _write_records_csv(paths.reports_dir / "generation_quality.csv", records)
    lines = ["# Generation Quality", "", f"- Unsupported claim estimate: {unsupported}", "", "| Ablation | Faithfulness | Relevance | Citation Accuracy | Conflict Awareness | Staleness Detection |", "|---|---:|---:|---:|---:|---:|"]
    for record in records:
        lines.append(
            f"| {record['ablation']} | {record['faithfulness']} | {record['answer_relevance']} | {record['citation_accuracy']} | {record['conflict_awareness']} | {record['staleness_detection']} |"
        )
    (paths.reports_dir / "generation_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_cases(paths: ProjectPaths, rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda row: (row["required_node_recall"] + row["required_edge_recall"] + row["faithfulness"] + row["staleness_detection"], row["path_validity"]))
    failures = []
    trace_by_key = {(trace["question_id"], trace["ablation"]): trace for trace in traces}
    for row in ranked[: max(5, min(20, len(ranked)))]:
        trace = trace_by_key.get((row["id"], row["ablation"]), {})
        failures.append(
            {
                "id": row["id"],
                "category": row["category"],
                "ablation": row["ablation"],
                "route": row["route"],
                "failure_reason": _failure_reason(row),
                "required_node_recall": row["required_node_recall"],
                "required_edge_recall": row["required_edge_recall"],
                "faithfulness": row["faithfulness"],
                "staleness_detection": row["staleness_detection"],
                "retrieved_node_ids": trace.get("retrieved_node_ids", []),
            }
        )
    write_json(paths.artifacts_dir / "failure_cases.json", failures)
    _write_records_csv(paths.reports_dir / "failure_cases.csv", failures)
    lines = ["# Failure Cases", ""]
    for failure in failures[:10]:
        lines.append(f"- `{failure['id']}` / `{failure['ablation']}`: {failure['failure_reason']}")
    (paths.reports_dir / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failure_reason(row: dict[str, Any]) -> str:
    if row["required_node_recall"] < 1.0:
        return "Did not retrieve every required evidence node."
    if row["required_edge_recall"] < 1.0:
        return "Retrieved nodes without every required relationship type."
    if row["path_validity"] < 1.0:
        return "At least one retrieved path was not backed by stored graph edges."
    if row["staleness_detection"] < 1.0:
        return "Freshness or repo-health caveat was insufficient for a stale-sensitive question."
    return "Lowest aggregate quality case in deterministic ablation comparison."


def _write_system_health(paths: ProjectPaths, rows: list[dict[str, Any]], summary: dict[str, Any], graph: GraphArtifact) -> None:
    stats = graph.stats()
    stale_source_count = sum(1 for source in graph.source_records if source.stale_after_days and "stale" in source.freshness_policy.lower())
    merge_decisions = graph.entity_resolution_decisions
    accepted_merges = sum(1 for decision in merge_decisions if decision.state in {"exact", "probable"} and not decision.review_required)
    duplicate_merge_rate = round(accepted_merges / len(merge_decisions), 3) if merge_decisions else 1.0
    payload = {
        "ingestion_success_count": len(graph.source_records),
        "ingestion_failure_count": 0,
        "graph_node_count": stats["node_count"],
        "graph_edge_count": stats["edge_count"],
        "duplicate_merge_rate": duplicate_merge_rate,
        "llm_cost_usd_estimate": summary["metrics"]["cost_usd_estimate"],
        "latency_ms_average": summary["metrics"]["latency_ms"],
        "stale_source_count": stale_source_count,
        "labels": stats["labels"],
        "relationships": stats["relationships"],
    }
    write_json(paths.artifacts_dir / "system_health.json", payload)
    lines = [
        "# System Health",
        "",
        f"- Questions evaluated: {summary['question_count']}",
        f"- Eval rows: {len(rows)}",
        f"- Ingestion success/failure: {payload['ingestion_success_count']}/{payload['ingestion_failure_count']}",
        f"- Graph nodes: {payload['graph_node_count']}",
        f"- Graph edges: {payload['graph_edge_count']}",
        f"- Duplicate merge rate: {payload['duplicate_merge_rate']}",
        f"- LLM cost estimate: {payload['llm_cost_usd_estimate']} (deterministic local synthesis)",
        f"- Average latency ms: {payload['latency_ms_average']}",
        f"- Stale source count: {payload['stale_source_count']}",
        "- Live credentials required: no",
    ]
    (paths.reports_dir / "system_health.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
