from __future__ import annotations

from signalgraph.cypher_templates import inspect_template, list_templates
from signalgraph.embeddings import DeterministicEmbeddingProvider, embedding_records_for_nodes
from signalgraph.neo4j_loader import Neo4jLoader, fulltext_index_statements, vector_index_statements


def test_embedding_provider_covers_required_record_kinds_with_deterministic_fallback(sample_project):
    _, graph = sample_project
    records = embedding_records_for_nodes(graph.nodes)
    kinds = {record.kind for record in records}
    assert {"DocumentChunk", "Claim", "Method", "Repo", "Community"} <= kinds

    provider = DeterministicEmbeddingProvider(dimensions=64)
    first = provider.embed_texts(["GraphRAG community report"])[0]
    second = provider.embed_texts(["GraphRAG community report"])[0]
    assert first == second
    assert len(first) == 64

    for node in graph.nodes:
        labels = set(node.labels)
        if "Repo" in labels:
            assert len(node.properties["readme_embedding"]) == 64
        if labels & {"DocumentChunk", "Claim", "Method", "Community"}:
            assert len(node.properties["embedding"]) == 64


def test_neo4j_loader_dry_run_includes_schema_indexes_and_batches_without_env(sample_project, monkeypatch):
    for name in ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]:
        monkeypatch.delenv(name, raising=False)

    _, graph = sample_project
    plan = Neo4jLoader(batch_size=2).load(graph, dry_run=True)
    payload = plan.to_dict()

    assert payload["dry_run"] is True
    assert payload["config"]["uri"] == "bolt://localhost:7687"
    assert payload["config"]["password"] == "<unset>"
    assert any("CONSTRAINT community_id" in statement for statement in payload["constraints"])
    assert any("FULLTEXT INDEX signalgraph_community_text" in statement for statement in fulltext_index_statements())
    assert any("VECTOR INDEX signalgraph_chunk_embedding" in statement for statement in vector_index_statements(64))
    assert any(key.startswith("nodes:Community") for key in payload["batch_counts"])
    assert any(key.startswith("relationships:BELONGS_TO_COMMUNITY") for key in payload["batch_counts"])


def test_cypher_templates_cover_evidence_structured_counts_and_typed_inspection():
    names = {template.name for template in list_templates()}
    assert {"evidence_paths", "structured_repo_lookup", "counts", "community_evidence", "conflicts", "duplicate_review"} <= names

    evidence = inspect_template("evidence_paths")["cypher"]
    structured = inspect_template("structured_repo_lookup")["cypher"]
    counts = inspect_template("counts")["cypher"]

    assert "(source)-[:CLAIMS]->(claim:Claim)-[:SUPPORTED_BY]->(chunk:DocumentChunk)" in evidence
    assert "(repo:Repo)-[:IMPLEMENTS|USES_METHOD]->(target)" in structured
    assert "AFFILIATED_WITH" in structured
    assert "type(r)" in counts
    assert "POSSIBLE_DUPLICATE" in inspect_template("duplicate_review")["cypher"]


def test_graph_schema_expansion_has_community_and_required_relationship_types(sample_project):
    _, graph = sample_project
    labels = {label for node in graph.nodes for label in node.labels}
    relationships = {edge.type for edge in graph.edges}

    assert "Community" in labels
    assert {
        "AFFILIATED_WITH",
        "CITES",
        "CONTRADICTS",
        "BELONGS_TO_COMMUNITY",
        "SIMILAR_TO",
    } <= relationships
    assert "POSSIBLE_DUPLICATE" in inspect_template("duplicate_review")["cypher"]
    assert all(edge.type != "RELATED_TO" for edge in graph.edges)
