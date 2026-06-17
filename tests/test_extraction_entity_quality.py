from __future__ import annotations

from signalgraph.answering import AnswerSynthesizer
from signalgraph.entity_resolution import EXACT_MATCH, POSSIBLE_DUPLICATE, PROBABLE_MATCH, resolve_entities
from signalgraph.graph import build_graph
from signalgraph.models import BenchmarkRecord, DatasetRecord, NormalizedCorpus, PaperRecord
from signalgraph.normalization import normalize_stored_documents, source_record_from_payload
from signalgraph.providers import LLMStructuredExtractionProvider
from signalgraph.quality import score_source_quality
from signalgraph.retrieval import Retriever
from signalgraph.sources import RawDocument


def _traceable_document():
    text = (
        "GraphRAG improves agent memory retrieval on a faithfulness benchmark using the Agent Memory Dataset. "
        "The approach has a production risk around stale memory."
    )
    payload = {
        "kind": "paper",
        "id": "paper:fixture:traceable",
        "title": "Traceable GraphRAG Extraction",
        "abstract": text,
        "published_at": "2026-01-01",
        "venue": "Fixture",
        "authors": ["Ada Extractor"],
    }
    document = RawDocument("sample", "https://example.test/paper/traceable", "traceable", {"topic": "GraphRAG"}, payload)
    record = source_record_from_payload(document.source_name, document.source_url, document.source_id, payload)
    return document, record, text


def test_llm_structured_extraction_validates_spans_and_quarantines_untraceable_claims():
    document, record, source_text = _traceable_document()
    provider = LLMStructuredExtractionProvider(
        lambda prompt, schema, context: {
            "methods": [
                {
                    "name": "GraphRAG",
                    "aliases": ["graph retrieval augmented generation"],
                    "description": "Graph-aware retrieval.",
                    "category": "graph-aware retrieval",
                    "source_span": "GraphRAG",
                    "confidence": 0.91,
                }
            ],
            "benchmarks": [{"name": "Faithfulness Benchmark", "task": "rag_evaluation", "metric": "faithfulness", "source_span": "faithfulness benchmark", "confidence": 0.87}],
            "datasets": [{"name": "Agent Memory Dataset", "domain": "agent_memory", "source_span": "Agent Memory Dataset", "confidence": 0.86}],
            "claims": [
                {
                    "text": "GraphRAG improves agent memory retrieval on a faithfulness benchmark using the Agent Memory Dataset.",
                    "claim_type": "benchmark",
                    "confidence": 0.84,
                    "polarity": "positive",
                    "source_span": "GraphRAG improves agent memory retrieval on a faithfulness benchmark using the Agent Memory Dataset.",
                },
                {
                    "text": "This claim was invented by the provider.",
                    "claim_type": "performance",
                    "confidence": 0.99,
                    "polarity": "positive",
                    "source_span": "invented span",
                },
            ],
        },
        name="fixture-llm",
        model="structured-output-test",
    )

    corpus = normalize_stored_documents([(document, record)], extraction_provider=provider)

    assert any(method.name == "GraphRAG" and method.extraction_method == "llm" for method in corpus.methods)
    assert any(benchmark.name == "Faithfulness Benchmark" and benchmark.source_span in source_text for benchmark in corpus.benchmarks)
    assert any(dataset.name == "Agent Memory Dataset" and dataset.source_span in source_text for dataset in corpus.datasets)
    assert any(claim.claim_type == "benchmark" and claim.source_span in source_text for claim in corpus.claims)
    assert not any("invented by the provider" in claim.text for claim in corpus.claims)
    assert any(row.reason == "untraceable_source_span" and row.attempted_record_type == "claim" for row in corpus.extraction_quarantine)

    graph = build_graph(corpus)
    claim_nodes = [node for node in graph.nodes if "Claim" in node.labels]
    assert claim_nodes
    assert all(node.properties["source_span"] in source_text for node in claim_nodes)


def test_invalid_llm_output_is_quarantined_and_deterministic_fallback_extracts_records():
    document, record, _ = _traceable_document()
    invalid_provider = LLMStructuredExtractionProvider(
        lambda prompt, schema, context: {
            "claims": [
                {
                    "text": "Untraceable claim.",
                    "claim_type": "performance",
                    "confidence": 0.9,
                    "polarity": "positive",
                    "source_span": "not present in the source",
                }
            ]
        },
        name="fixture-llm",
        model="invalid-output-test",
    )

    corpus = normalize_stored_documents([(document, record)], extraction_provider=invalid_provider)

    assert corpus.extraction_quarantine
    assert any(row.extraction_method == "llm" and row.reason == "untraceable_source_span" for row in corpus.extraction_quarantine)
    assert any(method.extraction_method == "deterministic" for method in corpus.methods)
    assert any(claim.extraction_method == "deterministic" and claim.source_span for claim in corpus.claims)


def test_entity_resolution_states_and_reviewable_possible_duplicate_edges():
    corpus = NormalizedCorpus(
        papers=[
            PaperRecord(id="paper:doi-a", title="GraphRAG Systems", abstract="", published_at="2025", doi="10.123/example", source_url="https://a.test"),
            PaperRecord(id="paper:doi-b", title="Graph RAG Systems", abstract="", published_at="2025", doi="10.123/example", source_url="https://b.test"),
        ],
        benchmarks=[
            BenchmarkRecord(id="benchmark:faithfulness-a", name="Faithfulness Benchmark", task="rag", metric="faithfulness", source_url="https://bench-a.test"),
            BenchmarkRecord(id="benchmark:faithfulness-b", name="Faithfulness Benchmark", task="rag", metric="faithfulness", source_url="https://bench-b.test"),
        ],
        datasets=[
            DatasetRecord(id="dataset:agent-memory-dataset", name="Agent Memory Dataset", domain="memory", source_url="https://dataset-a.test"),
            DatasetRecord(id="dataset:agent-memory-data", name="Agent Memory Data", domain="evaluation", source_url="https://dataset-b.test"),
        ],
    )

    decisions = resolve_entities(corpus)
    states = {decision.state for decision in decisions}
    assert {EXACT_MATCH, PROBABLE_MATCH, POSSIBLE_DUPLICATE} <= states

    graph = build_graph(corpus)
    possible_edges = [edge for edge in graph.edges if edge.type == "POSSIBLE_DUPLICATE"]
    assert possible_edges
    assert all(edge.properties["review_required"] is True for edge in possible_edges)
    assert any(decision.review_required and decision.state == POSSIBLE_DUPLICATE for decision in graph.entity_resolution_decisions)


def test_source_quality_scoring_flows_into_retrieval_reranking_and_answering(sample_project):
    paths, graph = sample_project
    readme = score_source_quality(source_name="github", extraction_method="deterministic", record_source_type="RepoDocument", section="readme", source_url="https://github.com/example/repo#readme")
    issue = score_source_quality(source_name="github", extraction_method="deterministic", record_source_type="Issue", section="issue", source_url="https://github.com/example/repo/issues/7", claim_type="repo_risk")
    inferred = score_source_quality(source_name="sample", extraction_method="llm", record_source_type="Claim", has_source_span=False)

    assert readme.score > issue.score > inferred.score
    assert any("source_quality_score" in node.properties for node in graph.nodes)

    result = Retriever(paths).graph_aware("Which GraphRAG implementation has production risk evidence?")
    assert result.candidates
    assert "source_quality" in result.candidates[0].features
    assert result.candidates[0].features["source_quality"] > 0

    answer = AnswerSynthesizer().synthesize(result)
    assert answer.confidence > 0
    assert answer.citations
