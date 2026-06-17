from __future__ import annotations

from signalgraph.answering import AnswerSynthesizer
from signalgraph.community import community_reports_from_graph
from signalgraph.config import ProjectPaths
from signalgraph.graph import GraphBuilder
from signalgraph.models import EntityResolutionDecision, GraphArtifact, GraphEdge, GraphNode
from signalgraph.rerank import maximal_marginal_relevance, reciprocal_rank_fusion
from signalgraph.retrieval import Retriever
from signalgraph.storage import NormalizedStore


def test_deterministic_community_detection_reports_embeddings_and_graph_links(sample_project):
    paths, graph = sample_project
    reports = community_reports_from_graph(graph)
    community_nodes = [node for node in graph.nodes if "Community" in node.labels]
    belongs_edges = [edge for edge in graph.edges if edge.type == "BELONGS_TO_COMMUNITY"]

    assert community_nodes
    assert reports
    assert belongs_edges
    assert paths.graph_dir.joinpath("community_reports.json").exists()
    assert all(node.properties["report"] for node in community_nodes)
    assert all(node.properties["member_ids"] for node in community_nodes)
    assert all(node.properties["embedding_provider"] == "deterministic" for node in community_nodes)
    assert all(len(node.properties["embedding"]) == 64 for node in community_nodes)

    corpus = NormalizedStore(paths).load()
    first = GraphBuilder(paths).build(use_sample_if_empty=False)
    second = GraphBuilder(paths).build(use_sample_if_empty=False)
    first_members = sorted((node.id, tuple(node.properties.get("member_ids", []))) for node in first.nodes if "Community" in node.labels)
    second_members = sorted((node.id, tuple(node.properties.get("member_ids", []))) for node in second.nodes if "Community" in node.labels)
    assert corpus.methods
    assert first_members == second_members


def test_retrieval_modes_use_distinct_graph_community_and_fusion_artifacts(sample_project):
    paths, _ = sample_project
    retriever = Retriever(paths)

    vector = retriever.retrieve("GraphRAG production implementation", mode="vector", limit=6)
    local = retriever.retrieve("GraphRAG production implementation", mode="local", limit=12)
    global_result = retriever.retrieve("What are the main themes in GraphRAG and agent memory research?", mode="global", limit=10)
    drift = retriever.retrieve("What should a startup use for customer support agents based on GraphRAG and memory research?", mode="drift", limit=12)
    hybrid = retriever.retrieve("Compare GraphRAG and agent memory for production support automation.", mode="hybrid", limit=10)
    structured = retriever.retrieve("Which repos implement papers after 2024?", mode="structured_lookup", limit=8)

    assert vector.mode == "vector"
    assert local.mode == "local"
    assert global_result.mode == "global"
    assert drift.mode == "drift"
    assert hybrid.mode == "hybrid"
    assert structured.mode == "structured_lookup"

    assert "typed_graph_neighborhood" in local.trace["components"]
    assert local.trace["neighborhood_path_count"] > 0
    assert "community_report_pull_in" in local.trace["components"]
    assert "map" in global_result.trace["components"]
    assert global_result.trace["map_results"]
    assert any("Community" in candidate.labels for candidate in global_result.candidates)
    assert "generated_subquestions" in drift.trace["components"]
    assert drift.trace["subquestions"]
    assert drift.trace["answer_tree"]
    assert "reciprocal_rank_fusion" in hybrid.trace["components"]
    assert hybrid.trace["rrf_scores"]
    assert any(candidate.features.get("rrf_score", 0) > 0 for candidate in hybrid.candidates)
    assert any(candidate.features.get("mmr_score", 0) > 0 for candidate in hybrid.candidates)
    assert "cypher_template" in structured.trace["components"]
    assert structured.trace["cypher_template"] == "structured_repo_lookup"
    assert any("Repo" in candidate.labels for candidate in structured.candidates)

    node_sets = {
        "vector": tuple(candidate.node_id for candidate in vector.candidates),
        "local": tuple(candidate.node_id for candidate in local.candidates),
        "global": tuple(candidate.node_id for candidate in global_result.candidates),
        "drift": tuple(candidate.node_id for candidate in drift.candidates),
        "hybrid": tuple(candidate.node_id for candidate in hybrid.candidates),
        "structured": tuple(candidate.node_id for candidate in structured.candidates),
    }
    assert len(set(node_sets.values())) > 3


def test_local_search_consumes_entity_resolution_and_adversarial_removal_changes_results(tmp_path):
    nodes = [
        GraphNode("method:query-alias", ["Method"], {"name": "GraphRAG production implementation"}),
        GraphNode("method:canonical", ["Method"], {"name": "Canonical retrieval method"}),
        GraphNode("method:probable", ["Method"], {"name": "Resolved deployment method"}),
        GraphNode("claim:canonical", ["Claim"], {"text": "Canonical evidence supports staged rollout with grounded citations.", "confidence": 0.91}),
        GraphNode("claim:probable", ["Claim"], {"text": "Resolved evidence highlights guardrails before scaling.", "confidence": 0.88}),
        GraphNode("repo:review-duplicate", ["Repo"], {"full_name": "review/neighbor", "description": "Review candidate for neighboring evidence."}),
    ]
    edges = [
        GraphEdge("edge:canonical-claim", "SUPPORTED_BY", "method:canonical", "claim:canonical", {"confidence": 0.9}),
        GraphEdge("edge:probable-claim", "SUPPORTED_BY", "method:probable", "claim:probable", {"confidence": 0.86}),
        GraphEdge(
            "edge:possible-duplicate",
            "POSSIBLE_DUPLICATE",
            "method:query-alias",
            "repo:review-duplicate",
            {
                "resolution_state": "possible_duplicate",
                "score": 0.78,
                "signals": ["fuzzy_name_below_merge_threshold"],
                "canonical_id": "method:query-alias",
                "review_required": True,
                "confidence": 0.78,
            },
        ),
    ]
    decisions = [
        EntityResolutionDecision(
            id="entity_resolution:exact-fixture",
            state="exact",
            left_id="method:query-alias",
            right_id="method:canonical",
            entity_label="Method",
            score=0.99,
            signals=["canonical_url"],
            canonical_id="method:canonical",
            canonical_url="https://example.test/canonical",
        ),
        EntityResolutionDecision(
            id="entity_resolution:probable-fixture",
            state="probable",
            left_id="method:query-alias",
            right_id="method:probable",
            entity_label="Method",
            score=0.91,
            signals=["normalized_name"],
            canonical_id="method:probable",
            canonical_url="https://example.test/probable",
        ),
        EntityResolutionDecision(
            id="entity_resolution:possible-fixture",
            state="possible_duplicate",
            left_id="method:query-alias",
            right_id="repo:review-duplicate",
            entity_label="Method",
            score=0.78,
            signals=["fuzzy_name_below_merge_threshold"],
            canonical_id="method:query-alias",
            review_required=True,
        ),
    ]
    graph = GraphArtifact("fixture", "2026-06-17T00:00:00+00:00", nodes, edges, entity_resolution_decisions=decisions)
    stripped = GraphArtifact(
        "fixture",
        "2026-06-17T00:00:00+00:00",
        nodes,
        [edge for edge in edges if edge.type != "POSSIBLE_DUPLICATE"],
        entity_resolution_decisions=[],
    )

    paths = ProjectPaths(tmp_path)
    full = Retriever(paths, graph=graph).local_search("GraphRAG production implementation", limit=6, trace=False)
    removed = Retriever(paths, graph=stripped).local_search("GraphRAG production implementation", limit=6, trace=False)

    entity_trace = full.trace["entity_resolution"]
    states = {decision["state"] for decision in entity_trace["consumed_decisions"]}
    full_ids = [candidate.node_id for candidate in full.candidates]
    removed_ids = [candidate.node_id for candidate in removed.candidates]
    full_paths = [candidate.path for candidate in full.candidates]
    removed_paths = [candidate.path for candidate in removed.candidates]

    assert states == {"exact", "probable", "possible_duplicate"}
    assert "method:canonical" in entity_trace["canonical_anchor_ids"]
    assert "method:probable" in entity_trace["probable_anchor_ids"]
    assert entity_trace["possible_duplicate_signals"]
    assert "method:canonical" in entity_trace["expansion_anchor_ids"]
    assert "method:probable" in entity_trace["expansion_anchor_ids"]
    assert "repo:review-duplicate" in entity_trace["expansion_anchor_ids"]
    assert "claim:canonical" in full_ids
    assert "claim:probable" in full_ids
    assert "repo:review-duplicate" in full_ids
    assert full.trace["entity_resolution"] != removed.trace["entity_resolution"]
    assert full_ids != removed_ids
    assert full_paths != removed_paths


def test_answer_synthesis_contract_covers_drift_evidence_and_next_checks(sample_project):
    paths, _ = sample_project
    result = Retriever(paths).retrieve(
        "What should a startup use for customer support agents based on GraphRAG and memory research?",
        mode="drift",
        limit=12,
    )
    answer = AnswerSynthesizer().synthesize(result)

    assert answer.route == "drift"
    assert answer.answer
    assert answer.reasoning
    assert answer.citations
    assert answer.evidence_chain
    assert answer.confidence > 0
    assert answer.conflicts_or_missing_evidence
    assert answer.production_recommendation
    assert answer.next_checks


def test_rrf_and_mmr_are_deterministic_and_promote_diversity():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    assert fused["b"] > fused["c"]
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]]) == fused

    class Candidate:
        def __init__(self, identifier: str, text: str, score: float):
            self.id = identifier
            self.text = text
            self.score = score

    selected = maximal_marginal_relevance(
        [
            Candidate("a", "GraphRAG community report retrieval", 0.9),
            Candidate("b", "GraphRAG community report retrieval duplicate", 0.88),
            Candidate("c", "agent memory production risk", 0.82),
        ],
        limit=2,
    )
    assert [candidate.id for candidate in selected] == ["a", "c"]
