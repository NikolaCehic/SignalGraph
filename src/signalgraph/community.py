from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import CommunityRecord, GraphArtifact, NormalizedCorpus
from .utils import compact_text, slugify, token_set, utc_now, write_json


COMMUNITY_SOURCE_URL = "local://signalgraph/community"
COMMUNITY_SOURCE_RECORD_ID = "source:signalgraph:community"


def detect_communities(corpus: NormalizedCorpus) -> list[CommunityRecord]:
    """Deterministic local community detection over records, methods, and co-mentions."""
    if corpus.communities:
        return corpus.communities
    records = _community_records(corpus)
    if not records:
        return []
    method_signals = _method_signals(corpus)
    adjacency = _build_adjacency(records, method_signals)
    labels = _initial_labels(records, method_signals)
    for _ in range(8):
        changed = False
        for node_id in sorted(records):
            next_label = _best_neighbor_label(node_id, labels, adjacency)
            if next_label and next_label != labels[node_id]:
                labels[node_id] = next_label
                changed = True
        if not changed:
            break
    grouped: dict[str, list[str]] = {}
    for node_id, label in labels.items():
        grouped.setdefault(label, []).append(node_id)
    communities: list[CommunityRecord] = []
    for label, member_ids in sorted(grouped.items()):
        members = sorted(member_ids)
        if not members:
            continue
        methods = sorted({method for member_id in members for method in records[member_id]["method_ids"]})
        top_terms = _top_terms([records[member_id]["text"] for member_id in members])
        name = _community_name(label, methods, records)
        report = _community_report(name, members, methods, top_terms, records)
        confidence = min(0.94, 0.62 + (0.035 * len(members)) + (0.025 * len(methods)))
        communities.append(
            CommunityRecord(
                id=f"community:{slugify(label)}",
                level=0,
                name=name,
                summary=compact_text(report, 360),
                report=report,
                size=len(members),
                generated_at=utc_now(),
                source_url=COMMUNITY_SOURCE_URL,
                source_record_id=COMMUNITY_SOURCE_RECORD_ID,
                extraction_method="deterministic_label_propagation",
                confidence=round(confidence, 3),
                member_ids=members,
                method_ids=methods,
                top_terms=top_terms,
            )
        )
    return communities


def community_reports_from_graph(graph: GraphArtifact) -> list[dict[str, Any]]:
    nodes = graph.node_map()
    member_ids_by_community: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.type == "BELONGS_TO_COMMUNITY" and edge.end_id in nodes:
            member_ids_by_community.setdefault(edge.end_id, []).append(edge.start_id)
    reports: list[dict[str, Any]] = []
    for node in sorted(graph.nodes, key=lambda item: item.id):
        if "Community" not in node.labels:
            continue
        member_ids = sorted(set(member_ids_by_community.get(node.id, node.properties.get("member_ids", []))))
        reports.append(
            {
                "id": node.id,
                "name": node.properties.get("name", ""),
                "level": node.properties.get("level", 0),
                "summary": node.properties.get("summary", ""),
                "report": node.properties.get("report", ""),
                "size": node.properties.get("size", len(member_ids)),
                "confidence": node.properties.get("confidence", 0.0),
                "member_ids": member_ids,
                "method_ids": node.properties.get("method_ids", []),
                "top_terms": node.properties.get("top_terms", []),
                "embedding_provider": node.properties.get("embedding_provider", ""),
                "embedding_dimensions": node.properties.get("embedding_dimensions", 0),
            }
        )
    return reports


def write_community_report_artifacts(graph: GraphArtifact, path: Any) -> list[dict[str, Any]]:
    reports = community_reports_from_graph(graph)
    write_json(path, {"community_reports": reports})
    return reports


def _community_records(corpus: NormalizedCorpus) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for method in corpus.methods:
        records[method.id] = {
            "id": method.id,
            "kind": "Method",
            "name": method.name,
            "text": " ".join([method.name, method.description, method.category, " ".join(method.aliases)]),
            "method_ids": [method.id],
            "source_record_id": method.source_record_id,
            "source_id": "",
        }
    for paper in corpus.papers:
        records[paper.id] = {
            "id": paper.id,
            "kind": "Paper",
            "name": paper.title,
            "text": f"{paper.title} {paper.abstract}",
            "method_ids": [],
            "source_record_id": paper.source_record_id,
            "source_id": paper.id,
        }
    for repo in corpus.repos:
        records[repo.id] = {
            "id": repo.id,
            "kind": "Repo",
            "name": repo.full_name,
            "text": f"{repo.full_name} {repo.description} {' '.join(repo.topics)}",
            "method_ids": [],
            "source_record_id": repo.source_record_id,
            "source_id": repo.id,
        }
    for chunk in corpus.chunks:
        records[chunk.id] = {
            "id": chunk.id,
            "kind": "DocumentChunk",
            "name": chunk.section,
            "text": chunk.text,
            "method_ids": [],
            "source_record_id": chunk.source_record_id,
            "source_id": chunk.source_id,
        }
    for claim in corpus.claims:
        records[claim.id] = {
            "id": claim.id,
            "kind": "Claim",
            "name": claim.claim_type,
            "text": claim.text,
            "method_ids": [],
            "source_record_id": claim.source_record_id,
            "source_id": claim.source_id,
        }
    for benchmark in corpus.benchmarks:
        records[benchmark.id] = {
            "id": benchmark.id,
            "kind": "Benchmark",
            "name": benchmark.name,
            "text": f"{benchmark.name} {benchmark.task} {benchmark.metric}",
            "method_ids": [],
            "source_record_id": benchmark.source_record_id,
            "source_id": "",
        }
    for dataset in corpus.datasets:
        records[dataset.id] = {
            "id": dataset.id,
            "kind": "Dataset",
            "name": dataset.name,
            "text": f"{dataset.name} {dataset.domain} {' '.join(dataset.tags)}",
            "method_ids": [],
            "source_record_id": dataset.source_record_id,
            "source_id": "",
        }
    for model in corpus.models:
        records[model.id] = {
            "id": model.id,
            "kind": "Model",
            "name": model.name,
            "text": f"{model.name} {model.provider_or_org} {' '.join(model.tags)}",
            "method_ids": [],
            "source_record_id": model.source_record_id,
            "source_id": "",
        }
    return records


def _method_signals(corpus: NormalizedCorpus) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    for method in corpus.methods:
        names = [method.name] + list(method.aliases)
        category = slugify(method.category or method.name, fallback="uncategorized")
        signals[method.id] = {
            "id": method.id,
            "names": names,
            "tokens": set().union(*(token_set(name) for name in names), token_set(method.description), token_set(method.category)),
            "category": category,
        }
    return signals


def _build_adjacency(records: dict[str, dict[str, Any]], method_signals: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = {node_id: {} for node_id in records}
    source_groups: dict[str, list[str]] = {}
    for node_id, record in records.items():
        for method_id in _matched_method_ids(record["text"], method_signals):
            record["method_ids"].append(method_id)
            _connect(adjacency, node_id, method_id, 3.0)
        if record["source_record_id"]:
            source_groups.setdefault(f"record:{record['source_record_id']}", []).append(node_id)
        if record["source_id"]:
            source_groups.setdefault(f"source:{record['source_id']}", []).append(node_id)
    for node_ids in source_groups.values():
        for index, left in enumerate(sorted(set(node_ids))):
            for right in sorted(set(node_ids))[index + 1 :]:
                _connect(adjacency, left, right, 1.7)
    by_method: dict[str, list[str]] = {}
    for node_id, record in records.items():
        record["method_ids"] = sorted(set(record["method_ids"]))
        for method_id in record["method_ids"]:
            by_method.setdefault(method_id, []).append(node_id)
    for node_ids in by_method.values():
        for index, left in enumerate(sorted(set(node_ids))):
            for right in sorted(set(node_ids))[index + 1 :]:
                _connect(adjacency, left, right, 2.2)
    methods_by_category: dict[str, list[str]] = {}
    for method_id, signal in method_signals.items():
        if method_id in records:
            methods_by_category.setdefault(signal["category"], []).append(method_id)
    for node_ids in methods_by_category.values():
        for index, left in enumerate(sorted(node_ids)):
            for right in sorted(node_ids)[index + 1 :]:
                _connect(adjacency, left, right, 1.4)
    return adjacency


def _initial_labels(records: dict[str, dict[str, Any]], method_signals: dict[str, dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for node_id, record in records.items():
        method_ids = sorted(record["method_ids"])
        if method_ids:
            labels[node_id] = method_signals[method_ids[0]]["category"]
        elif record["kind"] == "Method" and node_id in method_signals:
            labels[node_id] = method_signals[node_id]["category"]
        else:
            labels[node_id] = slugify(record["kind"], fallback="uncategorized")
    return labels


def _best_neighbor_label(node_id: str, labels: dict[str, str], adjacency: dict[str, dict[str, float]]) -> str:
    votes: dict[str, float] = {}
    for neighbor_id, weight in adjacency.get(node_id, {}).items():
        votes[labels[neighbor_id]] = votes.get(labels[neighbor_id], 0.0) + weight
    if not votes:
        return labels[node_id]
    current = labels[node_id]
    votes[current] = votes.get(current, 0.0) + 0.25
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _matched_method_ids(text: str, method_signals: dict[str, dict[str, Any]]) -> list[str]:
    lowered = (text or "").lower()
    tokens = token_set(lowered)
    matches: list[str] = []
    for method_id, signal in sorted(method_signals.items()):
        if any(name.lower() and name.lower() in lowered for name in signal["names"]):
            matches.append(method_id)
        elif signal["tokens"] and len(tokens & signal["tokens"]) >= min(2, len(signal["tokens"])):
            matches.append(method_id)
    return matches


def _connect(adjacency: dict[str, dict[str, float]], left: str, right: str, weight: float) -> None:
    if left == right or left not in adjacency or right not in adjacency:
        return
    adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
    adjacency[right][left] = adjacency[right].get(left, 0.0) + weight


def _top_terms(texts: list[str], limit: int = 8) -> list[str]:
    stop = {"with", "from", "that", "this", "uses", "using", "into", "over", "when", "should", "source"}
    counts: dict[str, int] = {}
    for text in texts:
        for token in token_set(text):
            if len(token) < 4 or token in stop:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _community_name(label: str, method_ids: list[str], records: dict[str, dict[str, Any]]) -> str:
    if method_ids:
        names = [records[method_id]["name"] for method_id in method_ids if method_id in records]
        if names:
            return " / ".join(names[:2])
    return label.replace("-", " ").title()


def _community_report(
    name: str,
    member_ids: list[str],
    method_ids: list[str],
    top_terms: list[str],
    records: dict[str, dict[str, Any]],
) -> str:
    kinds: dict[str, int] = {}
    for member_id in member_ids:
        kind = records[member_id]["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
    representatives = [records[member_id]["name"] for member_id in member_ids[:6] if records[member_id]["name"]]
    method_names = [records[method_id]["name"] for method_id in method_ids if method_id in records]
    lines = [
        f"Community report: {name}.",
        f"Map signal: {len(member_ids)} graph nodes clustered by deterministic local label propagation over method mentions, source links, and co-mentions.",
        f"Composition: {', '.join(f'{kind}={count}' for kind, count in sorted(kinds.items()))}.",
        f"Representative methods: {', '.join(method_names[:6]) or 'No explicit method node.'}.",
        f"Top terms: {', '.join(top_terms) or 'n/a'}.",
        f"Representative evidence: {', '.join(representatives) or 'n/a'}.",
        "Production note: use this report for broad landscape retrieval, then verify recommendations with local claim and repo evidence.",
    ]
    return "\n".join(lines)
