from __future__ import annotations

import json

from signalgraph.answering import AnswerSynthesizer, compare
from signalgraph.evaluation import ABLATION_MODES, GRAPH_METRIC_KEYS, REQUIRED_EVAL_CATEGORIES, EvalRunner
from signalgraph.reports import ReportWriter
from signalgraph.retrieval import Retriever


def test_graph_build_creates_typed_nodes_edges_provenance_and_cypher(sample_project):
    paths, graph = sample_project
    labels = {label for node in graph.nodes for label in node.labels}
    relationships = {edge.type for edge in graph.edges}

    assert {"Paper", "Author", "Repo", "Method", "Claim", "DocumentChunk"} <= labels
    assert {"AUTHORED_BY", "INTRODUCES", "USES_METHOD", "IMPLEMENTS", "CLAIMS", "SUPPORTED_BY", "HAS_CHUNK"} <= relationships
    assert all("source_url" in node.properties for node in graph.nodes)
    assert all("confidence" in edge.properties for edge in graph.edges)
    assert paths.graph_artifact_path.exists()
    assert "CREATE CONSTRAINT paper_id" in paths.cypher_export_path.read_text(encoding="utf-8")
    assert "Paper/Repo -> Claim -> Source span" in paths.evidence_query_path.read_text(encoding="utf-8")


def test_retrieval_reranking_and_answer_contract(sample_project):
    paths, _ = sample_project
    retriever = Retriever(paths)
    vector = retriever.vector_only("GraphRAG production implementation")
    graph = retriever.graph_aware("Which GraphRAG implementation should a startup evaluate first for production?")

    assert vector.mode == "vector"
    assert graph.mode in {"local", "global", "drift", "hybrid", "structured_lookup"}
    assert graph.candidates
    top_features = graph.candidates[0].features
    assert {"semantic_relevance", "lexical_relevance", "graph_path_quality", "source_quality", "freshness", "confidence", "evidence_strength", "combined_score"} <= set(top_features)
    assert any("IMPLEMENTS" in part or "CLAIMS" in part or "SUPPORTED_BY" in part for candidate in graph.candidates for part in candidate.path)

    answer = AnswerSynthesizer().synthesize(graph)
    assert answer.answer
    assert answer.citations
    assert answer.evidence_chain
    assert isinstance(answer.confidence, float)
    assert answer.conflicts_or_missing_evidence
    assert answer.next_checks


def test_compare_eval_and_reports_generate_expected_artifacts(sample_project):
    paths, _ = sample_project
    comparison = compare(paths, "Compare GraphRAG and agent memory for enterprise support automation.")
    assert "vector_only" in comparison
    assert "graph_rag" in comparison
    assert comparison["evidence_chain_completeness"]["graph_rag"] >= comparison["evidence_chain_completeness"]["vector_only"]

    eval_payload = EvalRunner(paths).run()
    assert eval_payload["summary"]["graph_path_recall"] >= 0
    assert eval_payload["summary"]["evidence_chain_completeness"] >= 0
    assert paths.eval_results_path.exists()
    assert paths.retrieval_comparison_path.exists()
    assert (paths.reports_dir / "eval_summary.md").exists()

    outputs = ReportWriter(paths).decision_memo("Which GraphRAG implementation should a startup evaluate first?")
    assert outputs["markdown"].exists()
    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert "Vector-Only vs GraphRAG" in outputs["markdown"].read_text(encoding="utf-8")


def test_wu006_eval_corpus_ablations_graph_metrics_and_traces(sample_project):
    paths, _ = sample_project
    payload = EvalRunner(paths).run()

    assert 50 <= payload["summary"]["question_count"] <= 100
    assert payload["summary"]["question_count"] == 70
    assert set(payload["summary"]["category_counts"]) == set(REQUIRED_EVAL_CATEGORIES)
    assert payload["summary"]["category_counts"] == {
        "entity-specific": 15,
        "comparison": 15,
        "broad landscape": 15,
        "structured": 10,
        "decision-memo": 10,
        "adversarial/uncertainty": 5,
    }
    assert set(payload["summary"]["ablations"]) == set(ABLATION_MODES)
    assert payload["summary"]["row_count"] == payload["summary"]["question_count"] * len(ABLATION_MODES)
    assert set(GRAPH_METRIC_KEYS) <= set(payload["graph_metrics"])

    for row in payload["rows"]:
        assert row["ablation"] in ABLATION_MODES
        for metric in GRAPH_METRIC_KEYS:
            assert metric in row
            assert 0 <= row[metric] <= 1

    trace_path = paths.traces_dir / "eval_query_traces.jsonl"
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    required_trace_fields = {
        "user_query",
        "query_category",
        "route",
        "retrieved_node_ids",
        "retrieved_chunk_ids",
        "graph_traversals",
        "vector_scores",
        "full_text_scores",
        "reranker_scores",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "answer",
        "citations",
        "eval_scores",
        "latency_ms",
        "cost_usd_estimate",
    }
    assert required_trace_fields <= set(trace)
    assert set(GRAPH_METRIC_KEYS) <= set(trace["eval_scores"])


def test_wu006_report_artifacts_cover_required_formats(sample_project):
    paths, _ = sample_project
    outputs = ReportWriter(paths).eval_summary()

    required_paths = [
        paths.eval_dir / "signalgraph_eval_corpus.json",
        paths.eval_results_path,
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
    for path in required_paths:
        assert path.exists(), path
        assert path.read_text(encoding="utf-8").strip()

    assert outputs["trace_jsonl"] == paths.traces_dir / "eval_query_traces.jsonl"
    assert "Retrieval Quality" in (paths.reports_dir / "retrieval_quality.md").read_text(encoding="utf-8")
    assert "Generation Quality" in (paths.reports_dir / "generation_quality.md").read_text(encoding="utf-8")
    assert "System Health" in (paths.reports_dir / "system_health.md").read_text(encoding="utf-8")
    assert "Failure Cases" in (paths.reports_dir / "failure_cases.md").read_text(encoding="utf-8")
