from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .config import ProjectPaths
from .cypher_templates import get_template
from .graph import load_graph
from .models import GraphArtifact, GraphEdge, GraphNode
from .rerank import (
    RerankFeatures,
    diversity_against_selected,
    freshness_score,
    graph_path_quality,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
    source_quality,
)
from .utils import append_jsonl, compact_text, cosine_bow, lexical_overlap, parse_yearish, short_hash, token_set, utc_now

MERGED_ENTITY_RESOLUTION_STATES = {"exact", "probable"}
REVIEW_ENTITY_RESOLUTION_STATES = {"possible_duplicate"}


@dataclass
class QueryRoute:
    route: str
    reason: str
    retrieval_budget: int
    allowed_sources: list[str]
    required_evidence_types: list[str]
    fallback_route: str


@dataclass
class RetrievalCandidate:
    id: str
    node_id: str
    labels: list[str]
    text: str
    source_url: str
    source_name: str
    source_span: str
    path: list[str]
    features: dict[str, Any]
    score: float
    citation: dict[str, Any]


@dataclass
class RetrievalResult:
    query: str
    mode: str
    route: QueryRoute
    candidates: list[RetrievalCandidate]
    latency_ms: float
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryRouter:
    def route(self, query: str) -> QueryRoute:
        lowered = query.lower()
        broad = any(term in lowered for term in ["main approaches", "themes", "landscape", "what changed", "emerging"])
        decision = any(term in lowered for term in ["should", "recommend", "startup", "production", "decision", "evaluate first"])
        comparison = any(term in lowered for term in ["compare", "versus", " vs ", "tradeoff"])
        structured = any(term in lowered for term in ["which repos", "count", "after 20", "before 20", "from stanford"])
        if comparison:
            return QueryRoute("comparison", "Query asks for side-by-side contrast.", 10, ["arxiv", "openalex", "github", "sample"], ["Paper", "Method", "Repo", "Claim"], "local")
        if decision:
            return QueryRoute("drift", "Query asks for a production recommendation, so breadth plus local evidence is useful.", 12, ["arxiv", "openalex", "github", "sample"], ["Claim", "Repo", "Benchmark", "DocumentChunk"], "local")
        if broad:
            return QueryRoute("global", "Query asks for broad themes across the corpus.", 12, ["arxiv", "openalex", "sample"], ["Community", "Method", "Claim"], "vector")
        if structured:
            return QueryRoute("structured_lookup", "Query resembles a structured graph lookup.", 8, ["github", "arxiv", "openalex", "sample"], ["Repo", "Paper", "Method"], "local")
        if len(token_set(query)) <= 5:
            return QueryRoute("local", "Short entity-anchored query.", 8, ["arxiv", "openalex", "github", "sample"], ["Claim", "DocumentChunk"], "vector")
        return QueryRoute("hybrid", "Default mixed semantic, lexical, and graph retrieval.", 10, ["arxiv", "openalex", "github", "sample"], ["Claim", "DocumentChunk", "Method"], "vector")


class Retriever:
    def __init__(self, paths: ProjectPaths, graph: GraphArtifact | None = None):
        self.paths = paths
        self.graph = graph or load_graph(paths, build_if_missing=True)
        self.router = QueryRouter()
        self.nodes = self.graph.node_map()
        self.outgoing: dict[str, list[GraphEdge]] = {}
        self.incoming: dict[str, list[GraphEdge]] = {}
        for edge in self.graph.edges:
            self.outgoing.setdefault(edge.start_id, []).append(edge)
            self.incoming.setdefault(edge.end_id, []).append(edge)

    def vector_only(self, query: str, limit: int = 5) -> RetrievalResult:
        started = time.perf_counter()
        route = QueryRoute("vector", "Vector-only baseline over source-addressable chunks and claims.", limit, ["arxiv", "openalex", "github", "sample"], ["DocumentChunk", "Claim"], "none")
        candidates: list[RetrievalCandidate] = []
        for node in self.graph.nodes:
            if not ({"DocumentChunk", "Claim"} & set(node.labels)):
                continue
            text = _node_text(node)
            semantic = cosine_bow(query, text)
            lexical = lexical_overlap(query, text)
            features = RerankFeatures(
                semantic_relevance=semantic,
                lexical_relevance=lexical,
                graph_path_quality=0.0,
                source_quality=_source_quality(node),
                freshness=freshness_score(node.properties.get("published_at") or node.properties.get("created_at")),
                confidence=float(node.properties.get("confidence", 0.8)),
                evidence_strength=float(node.properties.get("confidence", 0.6)),
            )
            score = features.combined_score()
            if score > 0:
                candidates.append(_candidate(node, query, text, [node.id], features, score))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        result = RetrievalResult(query, "vector", route, candidates[:limit], _elapsed_ms(started), {"retrieved_node_ids": [c.node_id for c in candidates[:limit]], "created_at": utc_now(), "components": ["vector_similarity"]})
        self._trace(result)
        return result

    def retrieve(self, query: str, mode: str = "auto", limit: int | None = None) -> RetrievalResult:
        normalized = mode.replace("-", "_")
        if normalized == "auto":
            return self.graph_aware(query, limit=limit)
        if normalized == "vector":
            return self.vector_only(query, limit=limit or 5)
        if normalized == "local":
            return self.local_search(query, limit=limit)
        if normalized == "global":
            return self.global_search(query, limit=limit)
        if normalized == "drift":
            return self.drift_search(query, limit=limit)
        if normalized == "hybrid":
            return self.hybrid_search(query, limit=limit)
        if normalized in {"structured", "structured_lookup", "cypher_template"}:
            return self.structured_lookup(query, limit=limit)
        available = "auto, vector, local, global, drift, hybrid, structured_lookup"
        raise ValueError(f"unknown retrieval mode {mode!r}; available: {available}")

    def graph_aware(self, query: str, limit: int | None = None) -> RetrievalResult:
        route = self.router.route(query)
        if route.route == "global":
            return self.global_search(query, limit=limit, route=route)
        if route.route == "drift":
            return self.drift_search(query, limit=limit, route=route)
        if route.route == "structured_lookup":
            return self.structured_lookup(query, limit=limit, route=route)
        if route.route in {"comparison", "hybrid"}:
            return self.hybrid_search(query, limit=limit, route=route)
        return self.local_search(query, limit=limit, route=route)

    def local_search(self, query: str, limit: int | None = None, route: QueryRoute | None = None, trace: bool = True) -> RetrievalResult:
        started = time.perf_counter()
        route = route or QueryRoute("local", "Entity-anchored local graph search.", limit or 8, ["arxiv", "openalex", "github", "sample"], ["Claim", "DocumentChunk", "Community"], "vector")
        budget = limit or route.retrieval_budget
        anchors = self._anchors(query, budget=max(4, min(8, budget)))
        expansion_anchors, entity_resolution_trace, resolution_influence = self._entity_resolution_expansion(anchors)
        raw_candidates: dict[str, RetrievalCandidate] = {}
        neighborhood_paths: list[list[str]] = []
        for anchor in expansion_anchors:
            for path in self._expand_paths(anchor.id, "local"):
                neighborhood_paths.append(path)
                target = self.nodes[path[-1]]
                if not _eligible_target(target):
                    continue
                text = _path_text(path, self.nodes)
                features, score = self._features_for_path(query, path)
                score = self._apply_entity_resolution_influence(features, score, anchor.id, path, resolution_influence)
                candidate_id = f"retrieval:{target.id}:{len(path)}"
                candidate = _candidate(target, query, text, _path_with_relationships(path, self.outgoing, self.incoming), features, score)
                existing = raw_candidates.get(candidate_id)
                if existing is None or candidate.score > existing.score:
                    raw_candidates[candidate_id] = candidate
                for community_path in self._community_paths_for_path(path):
                    community = self.nodes[community_path[-1]]
                    features, score = self._features_for_path(query, community_path)
                    score = self._apply_entity_resolution_influence(features, score, anchor.id, community_path, resolution_influence)
                    community_text = _path_text(community_path, self.nodes)
                    candidate = _candidate(community, query, community_text, _path_with_relationships(community_path, self.outgoing, self.incoming), features, score)
                    existing = raw_candidates.get(candidate.id)
                    if existing is None or candidate.score > existing.score:
                        raw_candidates[candidate.id] = candidate
        candidates = list(raw_candidates.values())
        if not candidates:
            return self.vector_only(query, limit=budget)
        candidates = self._finalize_diverse(candidates, budget)
        result = RetrievalResult(
            query=query,
            mode="local",
            route=route,
            candidates=candidates,
            latency_ms=_elapsed_ms(started),
            trace={
                "anchors": [anchor.id for anchor in anchors],
                "resolved_anchors": [anchor.id for anchor in expansion_anchors],
                "entity_resolution": entity_resolution_trace,
                "neighborhood_path_count": len(neighborhood_paths),
                "community_report_ids": sorted({candidate.node_id for candidate in candidates if "Community" in candidate.labels}),
                "retrieved_node_ids": [candidate.node_id for candidate in candidates],
                "route": asdict(route),
                "created_at": utc_now(),
                "components": ["entity_resolution", "typed_graph_neighborhood", "chunks_claims", "community_report_pull_in", "reranking", "answer_synthesis_ready"],
            },
        )
        if trace:
            self._trace(result)
        return result

    def global_search(self, query: str, limit: int | None = None, route: QueryRoute | None = None) -> RetrievalResult:
        started = time.perf_counter()
        route = route or QueryRoute("global", "Community-report map/filter/reduce search.", limit or 12, ["arxiv", "openalex", "sample"], ["Community", "Claim", "Method"], "vector")
        budget = limit or route.retrieval_budget
        community_nodes = [node for node in self.graph.nodes if "Community" in node.labels]
        mapped: list[dict[str, Any]] = []
        candidates: list[RetrievalCandidate] = []
        for community in community_nodes:
            text = _node_text(community)
            semantic = cosine_bow(query, text)
            lexical = lexical_overlap(query, text)
            member_ids = self._community_member_ids(community.id)
            representative_text = " ".join(_node_text(self.nodes[node_id]) for node_id in member_ids[:6] if node_id in self.nodes)
            relevance = max(semantic, lexical, cosine_bow(query, representative_text))
            if relevance <= 0 and token_set(query) & set(community.properties.get("top_terms", [])):
                relevance = 0.2
            if relevance <= 0:
                continue
            features = RerankFeatures(
                semantic_relevance=semantic,
                lexical_relevance=max(lexical, lexical_overlap(query, representative_text)),
                graph_path_quality=0.72,
                source_quality=_source_quality(community),
                freshness=freshness_score(community.properties.get("generated_at")),
                confidence=float(community.properties.get("confidence", 0.72)),
                evidence_strength=min(1.0, 0.45 + (0.05 * len(member_ids))),
                graph_path_score=0.65,
            )
            score = features.combined_score()
            mapped.append({"community_id": community.id, "score": score, "member_count": len(member_ids), "point": compact_text(text, 220)})
            candidates.append(_candidate(community, query, f"{text} {representative_text}", [community.id], features, score))
            for member_id in member_ids[:4]:
                if member_id not in self.nodes:
                    continue
                member = self.nodes[member_id]
                if not _eligible_target(member):
                    continue
                path = [member_id, community.id]
                features, score = self._features_for_path(query, path)
                candidates.append(_candidate(member, query, _path_text(path, self.nodes), _path_with_relationships(path, self.outgoing, self.incoming), features, score))
        mapped.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            return self.vector_only(query, limit=budget)
        filtered = [item for item in mapped if item["score"] >= 0.12][:budget]
        candidates = self._finalize_diverse(candidates, budget)
        result = RetrievalResult(
            query=query,
            mode="global",
            route=route,
            candidates=candidates,
            latency_ms=_elapsed_ms(started),
            trace={
                "map_results": mapped,
                "filtered_community_ids": [item["community_id"] for item in filtered],
                "reduce_summary": compact_text("; ".join(item["point"] for item in filtered[:5]), 700),
                "retrieved_node_ids": [candidate.node_id for candidate in candidates],
                "route": asdict(route),
                "created_at": utc_now(),
                "components": ["community_reports", "map", "filter", "reduce", "representative_evidence"],
            },
        )
        self._trace(result)
        return result

    def drift_search(self, query: str, limit: int | None = None, route: QueryRoute | None = None) -> RetrievalResult:
        started = time.perf_counter()
        route = route or QueryRoute("drift", "DRIFT-style broad-to-local retrieval.", limit or 12, ["arxiv", "openalex", "github", "sample"], ["Community", "Claim", "Repo", "DocumentChunk"], "local")
        budget = limit or route.retrieval_budget
        primer = self.global_search(query, limit=max(3, min(6, budget)), route=route)
        community_candidates = [candidate for candidate in primer.candidates if "Community" in candidate.labels]
        subquestions = self._drift_subquestions(query, community_candidates)
        answer_tree: list[dict[str, Any]] = []
        combined: dict[str, RetrievalCandidate] = {candidate.id: candidate for candidate in primer.candidates}
        for subquestion in subquestions:
            local = self.local_search(subquestion, limit=4, trace=False)
            branch_score = round(sum(candidate.score for candidate in local.candidates[:3]) / max(1, len(local.candidates[:3])), 4)
            answer_tree.append(
                {
                    "subquestion": subquestion,
                    "route": local.route.route,
                    "score": branch_score,
                    "evidence_node_ids": [candidate.node_id for candidate in local.candidates[:4]],
                    "evidence_chain": [candidate.path for candidate in local.candidates[:2]],
                }
            )
            for candidate in local.candidates:
                candidate.features["drift_branch_score"] = branch_score
                candidate.score = round((candidate.score * 0.78) + (branch_score * 0.22), 4)
                combined[candidate.id] = candidate
        ranked_tree = sorted(answer_tree, key=lambda item: item["score"], reverse=True)
        candidates = self._finalize_diverse(list(combined.values()), budget)
        result = RetrievalResult(
            query=query,
            mode="drift",
            route=route,
            candidates=candidates,
            latency_ms=_elapsed_ms(started),
            trace={
                "broad_primer": primer.trace.get("reduce_summary", ""),
                "community_report_ids": [candidate.node_id for candidate in community_candidates],
                "subquestions": subquestions,
                "answer_tree": ranked_tree,
                "retrieved_node_ids": [candidate.node_id for candidate in candidates],
                "route": asdict(route),
                "created_at": utc_now(),
                "components": ["community_reports", "broad_primer", "generated_subquestions", "local_searches", "answer_tree_ranking", "evidence_chain_output"],
            },
        )
        self._trace(result)
        return result

    def hybrid_search(self, query: str, limit: int | None = None, route: QueryRoute | None = None) -> RetrievalResult:
        started = time.perf_counter()
        route = route or QueryRoute("hybrid", "Vector/full-text/graph fusion with RRF and MMR.", limit or 10, ["arxiv", "openalex", "github", "sample"], ["Claim", "DocumentChunk", "Method", "Community"], "vector")
        budget = limit or route.retrieval_budget
        vector_candidates = self._rank_nodes(query, "vector")
        fulltext_candidates = self._rank_nodes(query, "fulltext")
        graph_candidates = self._graph_component_candidates(query, budget)
        by_id: dict[str, RetrievalCandidate] = {}
        ranked_lists: list[list[str]] = []
        for ranked in [vector_candidates, fulltext_candidates, graph_candidates]:
            ranked_lists.append([candidate.id for candidate in ranked])
            for candidate in ranked:
                existing = by_id.get(candidate.id)
                if existing is None or candidate.score > existing.score:
                    by_id[candidate.id] = candidate
        rrf_scores = reciprocal_rank_fusion(ranked_lists)
        fused: list[RetrievalCandidate] = []
        for candidate_id, candidate in by_id.items():
            rrf = rrf_scores.get(candidate_id, 0.0)
            candidate.features["rrf_score"] = rrf
            candidate.features["fusion_components"] = _component_count(candidate_id, ranked_lists)
            candidate.score = round((candidate.score * 0.74) + (rrf * 0.26), 4)
            fused.append(candidate)
        fused.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = self._finalize_diverse(fused, budget)
        result = RetrievalResult(
            query=query,
            mode="hybrid",
            route=route,
            candidates=candidates,
            latency_ms=_elapsed_ms(started),
            trace={
                "vector_ranked_ids": [candidate.node_id for candidate in vector_candidates[:budget]],
                "fulltext_ranked_ids": [candidate.node_id for candidate in fulltext_candidates[:budget]],
                "graph_ranked_ids": [candidate.node_id for candidate in graph_candidates[:budget]],
                "rrf_scores": {candidate.node_id: candidate.features.get("rrf_score", 0.0) for candidate in candidates},
                "retrieved_node_ids": [candidate.node_id for candidate in candidates],
                "route": asdict(route),
                "created_at": utc_now(),
                "components": ["vector_similarity", "full_text_exact", "graph_traversal", "reciprocal_rank_fusion", "mmr_diversity", "graph_path_scoring"],
            },
        )
        self._trace(result)
        return result

    def structured_lookup(self, query: str, limit: int | None = None, route: QueryRoute | None = None) -> RetrievalResult:
        started = time.perf_counter()
        route = route or QueryRoute("structured_lookup", "Cypher-template structured lookup.", limit or 8, ["github", "arxiv", "openalex", "sample"], ["Repo", "Paper", "Method"], "local")
        budget = limit or route.retrieval_budget
        year = _query_year(query) or 0
        organization = "stanford" if "stanford" in query.lower() else ""
        template = get_template("structured_repo_lookup")
        candidates: list[RetrievalCandidate] = []
        for edge in self.graph.edges:
            if edge.type not in {"IMPLEMENTS", "USES_METHOD"}:
                continue
            start = self.nodes.get(edge.start_id)
            target = self.nodes.get(edge.end_id)
            if not start or not target or "Repo" not in start.labels or not ({"Paper", "Method"} & set(target.labels)):
                continue
            if year and "Paper" in target.labels and (parse_yearish(target.properties.get("published_at")) or 0) < year:
                continue
            if organization and not self._target_has_org(target.id, organization):
                continue
            path = [start.id, target.id]
            features, score = self._features_for_path(query, path)
            features.lexical_relevance = max(features.lexical_relevance, 0.82 if "which repos" in query.lower() else 0.55)
            features.graph_path_score = max(features.graph_path_score, 0.84)
            score = features.combined_score()
            candidates.append(_candidate(start, query, _path_text(path, self.nodes), _path_with_relationships(path, self.outgoing, self.incoming), features, score))
        if not candidates:
            return self.local_search(query, limit=budget, route=route)
        candidates = self._finalize_diverse(candidates, budget)
        result = RetrievalResult(
            query=query,
            mode="structured_lookup",
            route=route,
            candidates=candidates,
            latency_ms=_elapsed_ms(started),
            trace={
                "cypher_template": template.name,
                "cypher": template.cypher,
                "parameters": {"year": year or template.parameters["year"], "organization": organization, "limit": budget},
                "retrieved_node_ids": [candidate.node_id for candidate in candidates],
                "route": asdict(route),
                "created_at": utc_now(),
                "components": ["cypher_template", "structured_filters", "typed_graph_lookup", "graph_path_scoring"],
            },
        )
        self._trace(result)
        return result

    def _anchors(self, query: str, budget: int) -> list[GraphNode]:
        scored: list[tuple[float, GraphNode]] = []
        for node in self.graph.nodes:
            text = _node_text(node)
            score = max(cosine_bow(query, text), lexical_overlap(query, text))
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored[:budget]]

    def _entity_resolution_expansion(self, anchors: list[GraphNode]) -> tuple[list[GraphNode], dict[str, Any], dict[str, list[dict[str, Any]]]]:
        original_anchor_ids = [anchor.id for anchor in anchors]
        original_anchor_set = set(original_anchor_ids)
        expansion_ids = list(original_anchor_ids)
        influence: dict[str, list[dict[str, Any]]] = {anchor_id: [] for anchor_id in original_anchor_ids}
        consumed_decisions: list[dict[str, Any]] = []
        possible_duplicate_signals: list[dict[str, Any]] = []
        canonical_anchor_ids: list[str] = []
        probable_anchor_ids: list[str] = []

        def add_expansion(target_id: str, signal: dict[str, Any]) -> None:
            if target_id not in self.nodes:
                return
            if target_id not in expansion_ids:
                expansion_ids.append(target_id)
            influence.setdefault(target_id, []).append(signal)

        for decision in self.graph.entity_resolution_decisions:
            related_anchor_ids = [node_id for node_id in (decision.left_id, decision.right_id) if node_id in original_anchor_set]
            if not related_anchor_ids:
                continue
            state = decision.state
            counterpart_ids = [node_id for node_id in (decision.left_id, decision.right_id) if node_id not in related_anchor_ids and node_id in self.nodes]
            canonical_id = decision.canonical_id if decision.canonical_id in self.nodes else ""
            resolved_ids = []
            if state in MERGED_ENTITY_RESOLUTION_STATES:
                resolved_ids.extend([canonical_id, *counterpart_ids])
            elif state in REVIEW_ENTITY_RESOLUTION_STATES:
                resolved_ids.extend(counterpart_ids)
            resolved_ids = _unique_ids([node_id for node_id in resolved_ids if node_id])
            if not resolved_ids:
                continue
            decision_trace = {
                "decision_id": decision.id,
                "state": state,
                "anchor_ids": related_anchor_ids,
                "left_id": decision.left_id,
                "right_id": decision.right_id,
                "canonical_id": canonical_id,
                "resolved_ids": resolved_ids,
                "entity_label": decision.entity_label,
                "score": decision.score,
                "signals": decision.signals,
                "review_required": decision.review_required,
            }
            consumed_decisions.append(decision_trace)
            weight = 0.12 if state == "exact" else 0.08 if state == "probable" else 0.035
            signal = {
                "source": "entity_resolution_decision",
                "decision_id": decision.id,
                "state": state,
                "score": decision.score,
                "signals": decision.signals,
                "review_required": decision.review_required,
                "weight": weight,
            }
            for resolved_id in resolved_ids:
                add_expansion(resolved_id, signal)
            if state == "exact" and canonical_id:
                canonical_anchor_ids.append(canonical_id)
            if state == "probable":
                probable_anchor_ids.extend(resolved_ids)
            if state in REVIEW_ENTITY_RESOLUTION_STATES:
                possible_duplicate_signals.append(decision_trace)

        for anchor_id in original_anchor_ids:
            for edge in self.outgoing.get(anchor_id, []) + self.incoming.get(anchor_id, []):
                if edge.type != "POSSIBLE_DUPLICATE":
                    continue
                counterpart_id = edge.end_id if edge.start_id == anchor_id else edge.start_id
                if counterpart_id not in self.nodes:
                    continue
                edge_trace = {
                    "source": "possible_duplicate_edge",
                    "edge_id": edge.id,
                    "anchor_id": anchor_id,
                    "counterpart_id": counterpart_id,
                    "resolution_state": edge.properties.get("resolution_state", "possible_duplicate"),
                    "score": float(edge.properties.get("score", edge.properties.get("confidence", 0.35))),
                    "signals": edge.properties.get("signals", []),
                    "canonical_id": edge.properties.get("canonical_id", ""),
                    "review_required": bool(edge.properties.get("review_required", True)),
                }
                possible_duplicate_signals.append(edge_trace)
                add_expansion(
                    counterpart_id,
                    {
                        "source": "possible_duplicate_edge",
                        "edge_id": edge.id,
                        "state": "possible_duplicate",
                        "score": edge_trace["score"],
                        "signals": edge_trace["signals"],
                        "review_required": edge_trace["review_required"],
                        "weight": 0.035,
                    },
                )

        expansion_ids = _unique_ids(expansion_ids)
        trace = {
            "original_anchor_ids": original_anchor_ids,
            "canonical_anchor_ids": sorted(set(canonical_anchor_ids)),
            "probable_anchor_ids": sorted(set(probable_anchor_ids)),
            "expansion_anchor_ids": expansion_ids,
            "consumed_decisions": consumed_decisions,
            "possible_duplicate_signals": possible_duplicate_signals,
        }
        return [self.nodes[node_id] for node_id in expansion_ids if node_id in self.nodes], trace, influence

    def _apply_entity_resolution_influence(
        self,
        features: RerankFeatures,
        score: float,
        anchor_id: str,
        path: list[str],
        influence: dict[str, list[dict[str, Any]]],
    ) -> float:
        signals = list(influence.get(anchor_id, []))
        possible_duplicate_edges = [self._edge_between(start, end) for start, end in zip(path, path[1:])]
        possible_duplicate_edges = [edge for edge in possible_duplicate_edges if edge and edge.type == "POSSIBLE_DUPLICATE"]
        if possible_duplicate_edges:
            signals.extend(
                {
                    "source": "possible_duplicate_edge",
                    "edge_id": edge.id,
                    "state": "possible_duplicate",
                    "score": float(edge.properties.get("score", edge.properties.get("confidence", 0.35))),
                    "signals": edge.properties.get("signals", []),
                    "review_required": bool(edge.properties.get("review_required", True)),
                    "weight": 0.025,
                }
                for edge in possible_duplicate_edges
            )
        if not signals:
            return score
        boost = min(0.16, sum(float(signal.get("weight", 0.0)) for signal in signals))
        features.graph_path_score = min(1.0, features.graph_path_score + boost)
        features.evidence_strength = min(1.0, features.evidence_strength + (boost * 0.5))
        adjusted = round(min(1.0, score + boost), 4)
        features_dict = features.__dict__
        features_dict["entity_resolution_boost"] = boost
        features_dict["entity_resolution_states"] = sorted({str(signal.get("state", "")) for signal in signals if signal.get("state")})
        features_dict["entity_resolution_signal_ids"] = [
            str(signal.get("decision_id") or signal.get("edge_id"))
            for signal in signals
            if signal.get("decision_id") or signal.get("edge_id")
        ]
        return adjusted

    def _expand_paths(self, start_id: str, route: str) -> list[list[str]]:
        max_depth = 3 if route in {"local", "structured_lookup", "cypher-template", "hybrid"} else 4
        max_paths = 28 if route in {"local", "structured_lookup", "cypher-template", "hybrid"} else 56
        queue: list[list[str]] = [[start_id]]
        paths: list[list[str]] = []
        while queue:
            path = queue.pop(0)
            paths.append(path)
            if len(paths) >= max_paths:
                break
            if len(path) > max_depth:
                continue
            last = path[-1]
            edges = sorted(self.outgoing.get(last, []) + self.incoming.get(last, []), key=lambda edge: (edge.type, edge.start_id, edge.end_id))[:8]
            for edge in edges:
                nxt = edge.end_id if edge.start_id == last else edge.start_id
                if nxt in path:
                    continue
                queue.append(path + [nxt])
        return paths

    def _rank_nodes(self, query: str, component: str) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for node in self.graph.nodes:
            if not _eligible_target(node):
                continue
            text = _node_text(node)
            semantic = cosine_bow(query, text)
            lexical = lexical_overlap(query, text)
            relevance = semantic if component == "vector" else lexical
            if relevance <= 0:
                continue
            features = RerankFeatures(
                semantic_relevance=semantic,
                lexical_relevance=lexical,
                graph_path_quality=0.0,
                source_quality=_source_quality(node),
                freshness=freshness_score(node.properties.get("published_at") or node.properties.get("last_commit_at") or node.properties.get("generated_at") or node.properties.get("created_at")),
                confidence=float(node.properties.get("confidence", 0.75)),
                evidence_strength=_evidence_strength([node.id], self.nodes),
            )
            score = round((features.combined_score() * 0.62) + (relevance * 0.38), 4)
            candidate = _candidate(node, query, text, [node.id], features, score)
            candidate.features[f"{component}_score"] = relevance
            candidates.append(candidate)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    def _graph_component_candidates(self, query: str, budget: int) -> list[RetrievalCandidate]:
        candidates: dict[str, RetrievalCandidate] = {}
        for anchor in self._anchors(query, max(4, min(10, budget))):
            for path in self._expand_paths(anchor.id, "hybrid"):
                target = self.nodes[path[-1]]
                if not _eligible_target(target):
                    continue
                features, score = self._features_for_path(query, path)
                if score <= 0:
                    continue
                candidate = _candidate(target, query, _path_text(path, self.nodes), _path_with_relationships(path, self.outgoing, self.incoming), features, score)
                existing = candidates.get(candidate.id)
                if existing is None or candidate.score > existing.score:
                    candidates[candidate.id] = candidate
        ranked = list(candidates.values())
        ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return ranked

    def _features_for_path(self, query: str, path: list[str]) -> tuple[RerankFeatures, float]:
        target = self.nodes[path[-1]]
        text = _path_text(path, self.nodes)
        semantic = cosine_bow(query, text)
        lexical = lexical_overlap(query, text)
        typed_rels = max(0, (len(path) - 1))
        has_claim = any("Claim" in self.nodes[node_id].labels for node_id in path if node_id in self.nodes)
        has_span = any(bool(self.nodes[node_id].properties.get("source_span") or self.nodes[node_id].properties.get("text")) for node_id in path if node_id in self.nodes)
        graph_score = self._graph_path_score(path)
        features = RerankFeatures(
            semantic_relevance=semantic,
            lexical_relevance=lexical,
            graph_path_quality=graph_path_quality(len(path), typed_rels, has_claim, has_span),
            source_quality=_source_quality(target),
            freshness=freshness_score(target.properties.get("published_at") or target.properties.get("last_commit_at") or target.properties.get("generated_at") or target.properties.get("created_at")),
            confidence=_path_confidence(path, self.nodes),
            evidence_strength=_evidence_strength(path, self.nodes),
            graph_path_score=graph_score,
        )
        return features, features.combined_score()

    def _graph_path_score(self, path: list[str]) -> float:
        if len(path) <= 1:
            return 0.15
        rel_weights = {
            "CLAIMS": 0.95,
            "SUPPORTED_BY": 0.95,
            "IMPLEMENTS": 0.88,
            "INTRODUCES": 0.86,
            "USES_METHOD": 0.78,
            "BELONGS_TO_COMMUNITY": 0.74,
            "EVALUATES_ON": 0.82,
            "HAS_CHUNK": 0.72,
            "MENTIONS": 0.66,
            "CITES": 0.64,
            "SIMILAR_TO": 0.52,
            "CONTRADICTS": 0.58,
            "POSSIBLE_DUPLICATE": 0.25,
        }
        scores: list[float] = []
        for start, end in zip(path, path[1:]):
            edge = self._edge_between(start, end)
            if not edge:
                scores.append(0.25)
                continue
            confidence = float(edge.properties.get("confidence", 0.65))
            source_quality_score = float(edge.properties.get("source_quality_score", 0.55))
            rel_score = rel_weights.get(edge.type, 0.45)
            scores.append((0.45 * rel_score) + (0.35 * confidence) + (0.2 * source_quality_score))
        length_penalty = max(0.72, 1.0 - (0.04 * max(0, len(path) - 4)))
        return round((sum(scores) / len(scores)) * length_penalty, 4)

    def _edge_between(self, start: str, end: str) -> GraphEdge | None:
        for edge in self.outgoing.get(start, []):
            if edge.end_id == end:
                return edge
        for edge in self.incoming.get(start, []):
            if edge.start_id == end:
                return edge
        return None

    def _community_paths_for_path(self, path: list[str]) -> list[list[str]]:
        community_paths: list[list[str]] = []
        for node_id in path:
            for edge in self.outgoing.get(node_id, []):
                target = self.nodes.get(edge.end_id)
                if edge.type == "BELONGS_TO_COMMUNITY" and target and "Community" in target.labels:
                    community_paths.append(path + [edge.end_id] if path[-1] != edge.end_id else path)
        return community_paths

    def _community_member_ids(self, community_id: str) -> list[str]:
        linked = [edge.start_id for edge in self.incoming.get(community_id, []) if edge.type == "BELONGS_TO_COMMUNITY"]
        props = self.nodes[community_id].properties if community_id in self.nodes else {}
        stored = props.get("member_ids", [])
        values = linked + (stored if isinstance(stored, list) else [])
        return sorted({value for value in values if value in self.nodes})

    def _drift_subquestions(self, query: str, communities: list[RetrievalCandidate]) -> list[str]:
        selected = communities[:3] or self.global_search(query, limit=3).candidates[:3]
        questions: list[str] = []
        for candidate in selected:
            name = self.nodes[candidate.node_id].properties.get("name", candidate.node_id) if candidate.node_id in self.nodes else candidate.node_id
            questions.append(f"What claim and source evidence supports {name} for: {query}")
            questions.append(f"Which implementation or production risks are linked to {name}?")
        if not questions:
            questions = [f"What local graph evidence supports: {query}", f"What risks or missing evidence affect: {query}"]
        deduped: list[str] = []
        for question in questions:
            if question not in deduped:
                deduped.append(question)
        return deduped[:5]

    def _target_has_org(self, target_id: str, organization: str) -> bool:
        lowered = organization.lower()
        queue: list[tuple[str, int]] = [(target_id, 0)]
        seen: set[str] = set()
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in seen or depth > 3:
                continue
            seen.add(node_id)
            node = self.nodes.get(node_id)
            if node and "Organization" in node.labels and lowered in _node_text(node).lower():
                return True
            for edge in self.outgoing.get(node_id, []) + self.incoming.get(node_id, []):
                if edge.type not in {"AUTHORED_BY", "AFFILIATED_WITH"}:
                    continue
                nxt = edge.end_id if edge.start_id == node_id else edge.start_id
                queue.append((nxt, depth + 1))
        return False

    def _finalize_diverse(self, candidates: list[RetrievalCandidate], limit: int) -> list[RetrievalCandidate]:
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = maximal_marginal_relevance(candidates, limit)
        selected_texts: list[str] = []
        for candidate in selected:
            diversity = diversity_against_selected(candidate.text, selected_texts)
            candidate.features["diversity_score"] = diversity
            candidate.features["mmr_score"] = round((0.72 * candidate.score) + (0.28 * diversity), 6)
            selected_texts.append(candidate.text)
        selected.sort(key=lambda candidate: candidate.features.get("mmr_score", candidate.score), reverse=True)
        return selected

    def _trace(self, result: RetrievalResult) -> None:
        self.paths.ensure()
        append_jsonl(self.paths.query_trace_path, result.to_dict())


def _eligible_target(node: GraphNode) -> bool:
    useful = {"Claim", "DocumentChunk", "Repo", "Paper", "Method", "Benchmark", "Dataset", "Model", "Community"}
    return bool(useful & set(node.labels))


def _candidate(node: GraphNode, query: str, text: str, path: list[str], features: RerankFeatures, score: float) -> RetrievalCandidate:
    source_span = node.properties.get("source_span") or compact_text(node.properties.get("text") or text, 260)
    citation = {
        "node_id": node.id,
        "labels": node.labels,
        "source_url": node.properties.get("source_url") or node.properties.get("url", ""),
        "source_span": source_span,
        "score": score,
    }
    return RetrievalCandidate(
        id=f"candidate:{node.id}:{short_hash([query, path])}",
        node_id=node.id,
        labels=node.labels,
        text=compact_text(text, 900),
        source_url=citation["source_url"],
        source_name=node.properties.get("source_name", ""),
        source_span=source_span,
        path=path,
        features=features.to_dict(),
        score=score,
        citation=citation,
    )


def _node_text(node: GraphNode) -> str:
    props = node.properties
    pieces = [
        props.get("title", ""),
        props.get("name", ""),
        props.get("full_name", ""),
        props.get("description", ""),
        props.get("abstract", ""),
        props.get("summary", ""),
        props.get("report", ""),
        props.get("text", ""),
        props.get("source_span", ""),
        " ".join(props.get("aliases", []) if isinstance(props.get("aliases"), list) else []),
        " ".join(props.get("topics", []) if isinstance(props.get("topics"), list) else []),
        " ".join(props.get("top_terms", []) if isinstance(props.get("top_terms"), list) else []),
    ]
    return " ".join(piece for piece in pieces if piece)


def _source_quality(node: GraphNode) -> float:
    if "source_quality_score" in node.properties:
        return float(node.properties.get("source_quality_score") or 0.5)
    return source_quality(
        node.properties.get("source_name", ""),
        node.properties.get("extraction_method", ""),
        node.properties.get("source_type", ""),
        node.properties.get("section", ""),
        node.properties.get("source_url") or node.properties.get("url", ""),
        bool(node.properties.get("source_span") or node.properties.get("text") or node.properties.get("abstract")),
        node.properties.get("claim_type", ""),
    )


def _path_text(path: list[str], nodes: dict[str, GraphNode]) -> str:
    return " ".join(_node_text(nodes[node_id]) for node_id in path if node_id in nodes)


def _path_confidence(path: list[str], nodes: dict[str, GraphNode]) -> float:
    values = [float(nodes[node_id].properties.get("confidence", 0.75)) for node_id in path if node_id in nodes]
    return sum(values) / len(values) if values else 0.5


def _evidence_strength(path: list[str], nodes: dict[str, GraphNode]) -> float:
    labels = {label for node_id in path if node_id in nodes for label in nodes[node_id].labels}
    score = 0.35
    if "Claim" in labels:
        score += 0.22
    if "DocumentChunk" in labels:
        score += 0.2
    if "Repo" in labels:
        score += 0.12
    if "Benchmark" in labels or "Dataset" in labels:
        score += 0.11
    return min(1.0, score)


def _path_with_relationships(path: list[str], outgoing: dict[str, list[GraphEdge]], incoming: dict[str, list[GraphEdge]]) -> list[str]:
    if len(path) <= 1:
        return path
    decorated = [path[0]]
    for start, end in zip(path, path[1:]):
        rel = next((edge.type for edge in outgoing.get(start, []) if edge.end_id == end), None)
        if rel is None:
            rel = next((f"<-{edge.type}" for edge in incoming.get(start, []) if edge.start_id == end), None)
        decorated.append(rel or "RELATED")
        decorated.append(end)
    return decorated


def _unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _query_year(query: str) -> int | None:
    return parse_yearish(query)


def _component_count(candidate_id: str, ranked_lists: list[list[str]]) -> int:
    return sum(1 for ranked in ranked_lists if candidate_id in ranked)
