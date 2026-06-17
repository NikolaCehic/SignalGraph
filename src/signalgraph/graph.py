from __future__ import annotations

from dataclasses import asdict
from typing import Any

from . import cypher_templates
from .community import COMMUNITY_SOURCE_URL, detect_communities, write_community_report_artifacts
from .config import ProjectPaths
from .embeddings import apply_embeddings_to_nodes
from .entity_resolution import possible_duplicate_edges, resolve_entities
from .models import (
    ClaimRecord,
    CommunityRecord,
    DocumentChunk,
    GraphArtifact,
    GraphEdge,
    GraphNode,
    NormalizedCorpus,
    Provenance,
)
from .neo4j_loader import constraint_statements, fulltext_index_statements, vector_index_statements
from .quality import score_from_properties
from .sample_data import ensure_sample_corpus
from .storage import NormalizedStore
from .utils import lexical_overlap, read_json, short_hash, slugify, token_set, utc_now, write_json


SCHEMA_VERSION = "signalgraph.graph.v1"


class GraphBuilder:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.store = NormalizedStore(paths)

    def build(self, use_sample_if_empty: bool = True) -> GraphArtifact:
        corpus = self.store.load()
        if use_sample_if_empty and not (corpus.papers or corpus.repos or corpus.claims):
            ensure_sample_corpus(self.paths)
            corpus = self.store.load()
        artifact = build_graph(corpus)
        self.paths.ensure()
        write_json(self.paths.graph_artifact_path, artifact.to_dict())
        write_cypher_artifacts(artifact, self.paths)
        write_community_report_artifacts(artifact, self.paths.graph_dir / "community_reports.json")
        return artifact


def load_graph(paths: ProjectPaths, build_if_missing: bool = True) -> GraphArtifact:
    if not paths.graph_artifact_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(paths.graph_artifact_path)
        return GraphBuilder(paths).build(use_sample_if_empty=True)
    return GraphArtifact.from_dict(read_json(paths.graph_artifact_path))


def build_graph(corpus: NormalizedCorpus) -> GraphArtifact:
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    communities = detect_communities(corpus)

    def add_node(node_id: str, label: str, props: dict[str, Any]) -> None:
        props = {key: value for key, value in props.items() if value not in (None, "")}
        if node_id in nodes:
            nodes[node_id].properties.update(props)
            if label not in nodes[node_id].labels:
                nodes[node_id].labels.append(label)
            return
        nodes[node_id] = GraphNode(id=node_id, labels=[label], properties=props)

    def add_edge(start_id: str, rel_type: str, end_id: str, props: dict[str, Any]) -> None:
        if start_id not in nodes or end_id not in nodes:
            return
        edge_id = f"edge:{short_hash([start_id, rel_type, end_id, props.get('source_record_id', '')])}"
        if edge_id not in edges:
            edges[edge_id] = GraphEdge(edge_id, rel_type, start_id, end_id, props)

    for paper in corpus.papers:
        add_node(paper.id, "Paper", _node_props(asdict(paper), paper.source_url, "api", paper.source_record_id, labels=["Paper"]))
    for author in corpus.authors:
        add_node(author.id, "Author", _node_props(asdict(author), author.source_url, "api", author.source_record_id, labels=["Author"]))
    for org in corpus.organizations:
        add_node(org.id, "Organization", _node_props(asdict(org), org.source_url, "api", org.source_record_id, labels=["Organization"]))
    for repo in corpus.repos:
        add_node(repo.id, "Repo", _node_props(asdict(repo), repo.url, "api", repo.source_record_id, labels=["Repo"]))
    for benchmark in corpus.benchmarks:
        add_node(benchmark.id, "Benchmark", _node_props(asdict(benchmark), benchmark.source_url, benchmark.extraction_method, benchmark.source_record_id, benchmark.confidence, labels=["Benchmark"]))
    for dataset in corpus.datasets:
        add_node(dataset.id, "Dataset", _node_props(asdict(dataset), dataset.source_url, dataset.extraction_method, dataset.source_record_id, dataset.confidence, labels=["Dataset"]))
    for model in corpus.models:
        add_node(model.id, "Model", _node_props(asdict(model), model.source_url, "api", model.source_record_id, 0.82, labels=["Model"]))
    for method in corpus.methods:
        add_node(method.id, "Method", _node_props(asdict(method), method.source_url, method.extraction_method, method.source_record_id, method.confidence, labels=["Method"]))
    for chunk in corpus.chunks:
        add_node(chunk.id, "DocumentChunk", _node_props(asdict(chunk), chunk.source_url, "chunking", chunk.source_record_id, 1.0, labels=["DocumentChunk"]))
    for claim in corpus.claims:
        add_node(claim.id, "Claim", _node_props(asdict(claim), claim.source_url, claim.extraction_method, claim.source_record_id, claim.confidence, labels=["Claim"]))
    for community in communities:
        add_node(
            community.id,
            "Community",
            _node_props(
                asdict(community),
                community.source_url or "local://signalgraph/community",
                community.extraction_method,
                community.source_record_id or "source:signalgraph:community",
                community.confidence,
                labels=["Community"],
            ),
        )

    author_by_source = _group_by_source(corpus.authors)
    chunks_by_source_id = _group_chunks(corpus.chunks)
    claims_by_source_id = _group_claims(corpus.claims)
    benchmarks_by_source = _group_by_source(corpus.benchmarks)
    datasets_by_source = _group_by_source(corpus.datasets)
    models_by_source = _group_by_source(corpus.models)
    community_by_category = {_community_category(community): community for community in communities}
    community_by_method_id = {
        method.id: community_by_category.get(_method_category(method))
        for method in corpus.methods
        if community_by_category.get(_method_category(method))
    }
    community_by_member_id: dict[str, list[CommunityRecord]] = {}
    for community in communities:
        for member_id in community.member_ids:
            community_by_member_id.setdefault(member_id, []).append(community)

    for paper in corpus.papers:
        prov = _edge_props(paper.source_url, "api", paper.source_record_id, 1.0)
        for author in author_by_source.get(paper.source_record_id, []):
            add_edge(paper.id, "AUTHORED_BY", author.id, prov)
            if author.affiliation_text:
                for org in corpus.organizations:
                    if org.source_record_id == author.source_record_id and org.name in author.affiliation_text:
                        add_edge(author.id, "AFFILIATED_WITH", org.id, _edge_props(author.source_url, "api", author.source_record_id, 0.9))
        matched_methods = _matched_methods(f"{paper.title} {paper.abstract}", corpus.methods)
        _link_text_source(paper.id, "Paper", paper.source_record_id, paper.source_url, matched_methods, chunks_by_source_id, claims_by_source_id, benchmarks_by_source, datasets_by_source, models_by_source, add_edge)
        _link_to_communities(paper.id, matched_methods, community_by_method_id, paper.source_url, paper.source_record_id, add_edge)

    for repo in corpus.repos:
        matched_methods = _matched_methods(f"{repo.full_name} {repo.description} {' '.join(repo.topics)}", corpus.methods)
        _link_text_source(repo.id, "Repo", repo.source_record_id, repo.url, matched_methods, chunks_by_source_id, claims_by_source_id, benchmarks_by_source, datasets_by_source, models_by_source, add_edge)
        for method in matched_methods:
            add_edge(repo.id, "IMPLEMENTS", method.id, _edge_props(repo.url, "heuristic", repo.source_record_id, 0.72))
        _link_to_communities(repo.id, matched_methods, community_by_method_id, repo.url, repo.source_record_id, add_edge)

    for method in corpus.methods:
        community = community_by_method_id.get(method.id)
        if community:
            add_edge(method.id, "BELONGS_TO_COMMUNITY", community.id, _edge_props(method.source_url, "graph_analytics", method.source_record_id, 0.82))

    for claim in corpus.claims:
        matched_methods = _matched_methods(claim.text, corpus.methods)
        _link_to_communities(claim.id, matched_methods, community_by_method_id, claim.source_url, claim.source_record_id, add_edge)

    for member_id, member_communities in community_by_member_id.items():
        if member_id not in nodes:
            continue
        source_url = nodes[member_id].properties.get("source_url") or nodes[member_id].properties.get("url", COMMUNITY_SOURCE_URL)
        source_record_id = nodes[member_id].properties.get("source_record_id", "")
        for community in member_communities:
            add_edge(member_id, "BELONGS_TO_COMMUNITY", community.id, _edge_props(source_url, "deterministic_label_propagation", source_record_id, community.confidence))

    # Deterministic method-level implementation links connect repos to papers through shared methods.
    method_to_papers: dict[str, list[str]] = {}
    method_to_repos: dict[str, list[str]] = {}
    for edge in list(edges.values()):
        if edge.type in {"INTRODUCES", "USES_METHOD"} and edge.start_id.startswith("paper:"):
            method_to_papers.setdefault(edge.end_id, []).append(edge.start_id)
        if edge.type in {"USES_METHOD", "IMPLEMENTS"} and edge.start_id.startswith("repo:"):
            method_to_repos.setdefault(edge.end_id, []).append(edge.start_id)
    for method_id, repo_ids in method_to_repos.items():
        for repo_id in repo_ids:
            for paper_id in method_to_papers.get(method_id, []):
                add_edge(repo_id, "IMPLEMENTS", paper_id, _edge_props(nodes[repo_id].properties.get("url", ""), "heuristic", nodes[repo_id].properties.get("source_record_id", ""), 0.64))

    _add_citation_edges(corpus.papers, add_edge)
    _add_similarity_edges(corpus, add_edge)
    _add_contradiction_edges(corpus.claims, add_edge)

    entity_resolution_decisions = resolve_entities(corpus)
    for decision in possible_duplicate_edges(entity_resolution_decisions):
        add_edge(
            decision.left_id,
            "POSSIBLE_DUPLICATE",
            decision.right_id,
            {
                "resolution_state": decision.state,
                "score": decision.score,
                "signals": decision.signals,
                "canonical_id": decision.canonical_id,
                "canonical_url": decision.canonical_url,
                "review_required": decision.review_required,
                "created_at": decision.created_at,
                "extraction_method": "entity_resolution",
                "confidence": decision.score,
            },
        )

    artifact_nodes = sorted(nodes.values(), key=lambda node: node.id)
    apply_embeddings_to_nodes(artifact_nodes)
    return GraphArtifact(
        schema_version=SCHEMA_VERSION,
        created_at=utc_now(),
        nodes=artifact_nodes,
        edges=sorted(edges.values(), key=lambda edge: edge.id),
        source_records=corpus.source_records,
        entity_resolution_decisions=entity_resolution_decisions,
    )


def write_cypher_artifacts(artifact: GraphArtifact, paths: ProjectPaths) -> None:
    paths.ensure()
    paths.cypher_export_path.write_text(export_cypher(artifact), encoding="utf-8")
    paths.evidence_query_path.write_text(saved_evidence_queries(), encoding="utf-8")


def export_cypher(artifact: GraphArtifact) -> str:
    lines = [
        "// SignalGraph Neo4j-oriented export",
    ]
    lines.extend(f"{statement};" for statement in constraint_statements())
    lines.extend(f"{statement};" for statement in fulltext_index_statements())
    lines.extend(f"{statement};" for statement in vector_index_statements())
    lines.append("")
    for node in artifact.nodes:
        label_string = ":".join(_cypher_name(label) for label in node.labels)
        lines.append(f"MERGE (n:{label_string} {{id: {_cypher_literal(node.id)}}}) SET n += {_cypher_literal(node.properties)};")
    lines.append("")
    for edge in artifact.edges:
        lines.append(
            "MATCH (a {id: "
            + _cypher_literal(edge.start_id)
            + "}), (b {id: "
            + _cypher_literal(edge.end_id)
            + "}) MERGE (a)-[r:"
            + _cypher_name(edge.type)
            + " {id: "
            + _cypher_literal(edge.id)
            + "}]->(b) SET r += "
            + _cypher_literal(edge.properties)
            + ";"
        )
    return "\n".join(lines) + "\n"


def saved_evidence_queries() -> str:
    return cypher_templates.saved_evidence_queries()


def _node_props(
    props: dict[str, Any],
    source_url: str,
    extraction_method: str,
    source_record_id: str,
    confidence: float = 1.0,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    source_span = props.get("source_span", "")
    provenance = Provenance(
        source_url=source_url,
        source_name=_source_name(source_record_id),
        source_record_id=source_record_id,
        source_span=source_span,
        extraction_method=extraction_method,
        extractor_version=props.get("extractor_version", "signalgraph-0.1") or "signalgraph-0.1",
        created_at=utc_now(),
        confidence=confidence,
    )
    props.update(asdict(provenance))
    quality = score_from_properties(props, labels)
    props.update(
        {
            "source_quality_score": quality.score,
            "source_quality_type": quality.source_type,
            "source_quality_tier": quality.evidence_tier,
            "source_quality_reasons": quality.reasons,
        }
    )
    return props


def _edge_props(source_url: str, extraction_method: str, source_record_id: str, confidence: float) -> dict[str, Any]:
    props = asdict(
        Provenance(
            source_url=source_url,
            source_name=_source_name(source_record_id),
            source_record_id=source_record_id,
            extraction_method=extraction_method,
            created_at=utc_now(),
            confidence=confidence,
        )
    )
    quality = score_from_properties(props)
    props["source_quality_score"] = quality.score
    props["source_quality_type"] = quality.source_type
    props["source_quality_tier"] = quality.evidence_tier
    return props


def _link_text_source(
    source_id: str,
    source_type: str,
    source_record_id: str,
    source_url: str,
    matched_methods: list[Any],
    chunks_by_source_id: dict[str, list[DocumentChunk]],
    claims_by_source_id: dict[str, list[ClaimRecord]],
    benchmarks_by_source: dict[str, list[Any]],
    datasets_by_source: dict[str, list[Any]],
    models_by_source: dict[str, list[Any]],
    add_edge: Any,
) -> None:
    for method in matched_methods:
        rel_type = "INTRODUCES" if source_type == "Paper" else "USES_METHOD"
        add_edge(source_id, rel_type, method.id, _edge_props(source_url, getattr(method, "extraction_method", "deterministic"), source_record_id, getattr(method, "confidence", 0.78)))
    for chunk in chunks_by_source_id.get(source_id, []):
        add_edge(source_id, "HAS_CHUNK", chunk.id, _edge_props(source_url, "chunking", source_record_id, 1.0))
        for method in matched_methods:
            add_edge(chunk.id, "MENTIONS", method.id, _edge_props(source_url, getattr(method, "extraction_method", "deterministic"), source_record_id, min(getattr(method, "confidence", 0.7), 0.82)))
    for claim in claims_by_source_id.get(source_id, []):
        add_edge(source_id, "CLAIMS", claim.id, _edge_props(source_url, getattr(claim, "extraction_method", "deterministic"), source_record_id, claim.confidence))
        for chunk in chunks_by_source_id.get(source_id, []):
            add_edge(claim.id, "SUPPORTED_BY", chunk.id, _edge_props(source_url, getattr(claim, "extraction_method", "deterministic"), source_record_id, claim.confidence))
    for benchmark in benchmarks_by_source.get(source_record_id, []):
        add_edge(source_id, "EVALUATES_ON", benchmark.id, _edge_props(source_url, getattr(benchmark, "extraction_method", "deterministic"), source_record_id, getattr(benchmark, "confidence", 0.68)))
    for dataset in datasets_by_source.get(source_record_id, []):
        add_edge(source_id, "EVALUATES_ON", dataset.id, _edge_props(source_url, getattr(dataset, "extraction_method", "deterministic"), source_record_id, getattr(dataset, "confidence", 0.64)))
    for model in models_by_source.get(source_record_id, []):
        add_edge(source_id, "USES_METHOD", model.id, _edge_props(source_url, "heuristic", source_record_id, 0.62))


def _derived_communities(corpus: NormalizedCorpus) -> list[CommunityRecord]:
    categories: dict[str, list[Any]] = {}
    for method in corpus.methods:
        categories.setdefault(_method_category(method), []).append(method)
    communities: list[CommunityRecord] = []
    for category, methods in sorted(categories.items()):
        names = ", ".join(method.name for method in methods[:6])
        name = category.replace("_", " ").title()
        summary = f"{name} community covering {names}."
        communities.append(
            CommunityRecord(
                id=f"community:{slugify(category)}",
                level=0,
                name=name,
                summary=summary,
                report=f"Deterministic local community report for methods: {names}.",
                size=len(methods),
                generated_at=utc_now(),
                source_url="local://signalgraph/community",
                source_record_id="source:signalgraph:community",
                extraction_method="graph_analytics",
                confidence=0.72,
            )
        )
    return communities


def _link_to_communities(
    source_id: str,
    matched_methods: list[Any],
    community_by_method_id: dict[str, CommunityRecord | None],
    source_url: str,
    source_record_id: str,
    add_edge: Any,
) -> None:
    seen: set[str] = set()
    for method in matched_methods:
        community = community_by_method_id.get(method.id)
        if not community or community.id in seen:
            continue
        seen.add(community.id)
        add_edge(source_id, "BELONGS_TO_COMMUNITY", community.id, _edge_props(source_url, "graph_analytics", source_record_id, 0.74))


def _add_citation_edges(papers: list[Any], add_edge: Any) -> None:
    paper_by_source: dict[str, list[Any]] = _group_by_source(papers)
    for group in paper_by_source.values():
        primary = [paper for paper in group if paper.abstract or paper.published_at]
        related = [paper for paper in group if not paper.abstract and paper.semantic_scholar_id]
        for source in primary:
            for target in related:
                if source.id != target.id:
                    add_edge(source.id, "CITES", target.id, _edge_props(source.source_url, "api", source.source_record_id, 0.86))


def _add_similarity_edges(corpus: NormalizedCorpus, add_edge: Any) -> None:
    for index, left in enumerate(corpus.methods):
        for right in corpus.methods[index + 1 :]:
            if _method_category(left) == _method_category(right) or token_set(left.description) & token_set(right.description):
                add_edge(left.id, "SIMILAR_TO", right.id, _edge_props(left.source_url or right.source_url, "embedding_similarity", left.source_record_id or right.source_record_id, 0.68))
    for index, left in enumerate(corpus.repos):
        left_topics = set(left.topics)
        for right in corpus.repos[index + 1 :]:
            overlap = left_topics & set(right.topics)
            if overlap:
                add_edge(left.id, "SIMILAR_TO", right.id, _edge_props(left.url or right.url, "embedding_similarity", left.source_record_id or right.source_record_id, 0.62))


def _add_contradiction_edges(claims: list[ClaimRecord], add_edge: Any) -> None:
    positive = [claim for claim in claims if claim.polarity == "positive"]
    caution = [claim for claim in claims if claim.polarity == "caution"]
    for left in positive:
        for right in caution:
            if left.id == right.id:
                continue
            if lexical_overlap(left.text, right.text) <= 0 and not (token_set(left.text) & token_set(right.text) & {"retrieval", "memory", "graphrag", "production", "benchmark"}):
                continue
            props = _edge_props(left.source_url or right.source_url, "heuristic", left.source_record_id or right.source_record_id, min(left.confidence, right.confidence, 0.64))
            props["reason"] = "opposite_claim_polarity_with_shared_terms"
            add_edge(left.id, "CONTRADICTS", right.id, props)


def _method_category(method: Any) -> str:
    return slugify(getattr(method, "category", "") or "uncategorized", fallback="uncategorized")


def _community_category(community: CommunityRecord) -> str:
    value = community.id.split("community:", 1)[-1] if community.id.startswith("community:") else community.name
    return slugify(value, fallback="uncategorized")


def _group_by_source(items: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for item in items:
        grouped.setdefault(item.source_record_id, []).append(item)
    return grouped


def _group_chunks(chunks: list[DocumentChunk]) -> dict[str, list[DocumentChunk]]:
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)
    return grouped


def _group_claims(claims: list[ClaimRecord]) -> dict[str, list[ClaimRecord]]:
    grouped: dict[str, list[ClaimRecord]] = {}
    for claim in claims:
        grouped.setdefault(claim.source_id, []).append(claim)
    return grouped


def _matched_methods(text: str, methods: list[Any]) -> list[Any]:
    lowered = (text or "").lower()
    tokens = token_set(lowered)
    matches = []
    for method in methods:
        names = [method.name] + list(method.aliases)
        for name in names:
            normalized = name.lower()
            if normalized in lowered or token_set(normalized) <= tokens:
                matches.append(method)
                break
    seen = set()
    unique = []
    for method in matches:
        if method.id not in seen:
            seen.add(method.id)
            unique.append(method)
    return unique


def _source_name(source_record_id: str) -> str:
    parts = source_record_id.split(":")
    return parts[1] if len(parts) > 2 else ""


def _cypher_literal(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _cypher_name(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch == "_")
