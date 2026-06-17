from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar


@dataclass
class Provenance:
    source_url: str
    source_name: str
    source_record_id: str
    source_span: str = ""
    extraction_method: str = "api"
    extractor_version: str = "signalgraph-0.1"
    created_at: str = ""
    confidence: float = 1.0


@dataclass
class SourceRecord:
    id: str
    source_name: str
    source_url: str
    source_id: str
    fetched_at: str
    request_params: dict[str, Any]
    response_hash: str
    raw_payload_path: str
    license_or_terms_note: str
    freshness_policy: str = ""
    stale_after_days: int = 0
    rate_limit_note: str = ""
    cache_policy: str = ""
    cache_key: str = ""
    cache_status: str = ""
    quality_gate_status: str = "pass"
    quality_gate_reasons: list[str] = field(default_factory=list)


@dataclass
class PaperRecord:
    id: str
    title: str
    abstract: str
    published_at: str
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    semantic_scholar_id: str = ""
    citation_count: int = 0
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class AuthorRecord:
    id: str
    name: str
    orcid: str = ""
    openalex_id: str = ""
    semantic_scholar_id: str = ""
    affiliation_text: str = ""
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class OrganizationRecord:
    id: str
    name: str
    type: str = ""
    homepage: str = ""
    openalex_id: str = ""
    semantic_scholar_id: str = ""
    github_login: str = ""
    huggingface_id: str = ""
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class RepoRecord:
    id: str
    owner: str
    name: str
    full_name: str
    url: str
    stars: int = 0
    forks: int = 0
    license: str = ""
    default_branch: str = ""
    last_commit_at: str = ""
    latest_release_at: str = ""
    open_issues_count: int = 0
    health_score: float = 0.0
    risk_score: float = 0.0
    description: str = ""
    topics: list[str] = field(default_factory=list)
    readme_chars: int = 0
    docs_count: int = 0
    releases_count: int = 0
    selected_issues_count: int = 0
    changelog_present: bool = False
    latest_release_tag: str = ""
    repo_risk_signals: list[str] = field(default_factory=list)
    source_record_id: str = ""


@dataclass
class RepoDocumentRecord:
    id: str
    repo_id: str
    repo_full_name: str
    doc_type: str
    title: str
    path: str
    url: str
    text: str
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class RepoReleaseRecord:
    id: str
    repo_id: str
    repo_full_name: str
    tag_name: str
    name: str = ""
    published_at: str = ""
    url: str = ""
    body: str = ""
    prerelease: bool = False
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class RepoIssueRecord:
    id: str
    repo_id: str
    repo_full_name: str
    number: int
    title: str
    state: str = ""
    labels: list[str] = field(default_factory=list)
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    body: str = ""
    risk_signals: list[str] = field(default_factory=list)
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class BenchmarkRecord:
    id: str
    name: str
    task: str = ""
    metric: str = ""
    source_url: str = ""
    source_record_id: str = ""
    source_span: str = ""
    extracted_at: str = ""
    extractor_version: str = ""
    extraction_method: str = "deterministic"
    confidence: float = 0.75


@dataclass
class DatasetRecord:
    id: str
    name: str
    domain: str = ""
    license: str = ""
    provider_or_org: str = ""
    dataset_type: str = ""
    huggingface_id: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    source_url: str = ""
    source_record_id: str = ""
    source_span: str = ""
    extracted_at: str = ""
    extractor_version: str = ""
    extraction_method: str = "deterministic"
    confidence: float = 0.75


@dataclass
class ModelRecord:
    id: str
    name: str
    provider_or_org: str = ""
    model_type: str = ""
    huggingface_id: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    last_modified: str = ""
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class MethodRecord:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    category: str = ""
    source_url: str = ""
    source_record_id: str = ""
    source_span: str = ""
    extracted_at: str = ""
    extractor_version: str = ""
    extraction_method: str = "deterministic"
    confidence: float = 0.82


@dataclass
class DocumentChunk:
    id: str
    source_type: str
    source_id: str
    section: str
    text: str
    start_offset: int
    end_offset: int
    hash: str
    source_url: str = ""
    source_record_id: str = ""


@dataclass
class ClaimRecord:
    id: str
    text: str
    claim_type: str
    confidence: float
    polarity: str
    source_span: str
    source_url: str
    extracted_at: str
    extractor_version: str
    source_type: str
    source_id: str
    source_record_id: str
    extraction_method: str = "deterministic"


@dataclass
class CommunityRecord:
    id: str
    level: int
    name: str
    summary: str = ""
    report: str = ""
    size: int = 0
    generated_at: str = ""
    source_url: str = ""
    source_record_id: str = ""
    extraction_method: str = "graph_analytics"
    confidence: float = 0.72
    member_ids: list[str] = field(default_factory=list)
    method_ids: list[str] = field(default_factory=list)
    top_terms: list[str] = field(default_factory=list)


@dataclass
class ExtractionQuarantineRecord:
    id: str
    source_type: str
    source_id: str
    source_record_id: str
    source_url: str
    attempted_record_type: str
    reason: str
    payload: dict[str, Any]
    source_span: str = ""
    extractor_version: str = ""
    extraction_method: str = ""
    created_at: str = ""


@dataclass
class EntityResolutionDecision:
    id: str
    state: str
    left_id: str
    right_id: str
    entity_label: str
    score: float
    signals: list[str] = field(default_factory=list)
    canonical_id: str = ""
    canonical_url: str = ""
    review_required: bool = False
    created_at: str = ""


@dataclass
class NormalizedCorpus:
    source_records: list[SourceRecord] = field(default_factory=list)
    papers: list[PaperRecord] = field(default_factory=list)
    authors: list[AuthorRecord] = field(default_factory=list)
    organizations: list[OrganizationRecord] = field(default_factory=list)
    repos: list[RepoRecord] = field(default_factory=list)
    repo_documents: list[RepoDocumentRecord] = field(default_factory=list)
    repo_releases: list[RepoReleaseRecord] = field(default_factory=list)
    repo_issues: list[RepoIssueRecord] = field(default_factory=list)
    benchmarks: list[BenchmarkRecord] = field(default_factory=list)
    datasets: list[DatasetRecord] = field(default_factory=list)
    models: list[ModelRecord] = field(default_factory=list)
    methods: list[MethodRecord] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    communities: list[CommunityRecord] = field(default_factory=list)
    extraction_quarantine: list[ExtractionQuarantineRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedCorpus":
        return cls(
            source_records=_load_list(SourceRecord, data.get("source_records", [])),
            papers=_load_list(PaperRecord, data.get("papers", [])),
            authors=_load_list(AuthorRecord, data.get("authors", [])),
            organizations=_load_list(OrganizationRecord, data.get("organizations", [])),
            repos=_load_list(RepoRecord, data.get("repos", [])),
            repo_documents=_load_list(RepoDocumentRecord, data.get("repo_documents", [])),
            repo_releases=_load_list(RepoReleaseRecord, data.get("repo_releases", [])),
            repo_issues=_load_list(RepoIssueRecord, data.get("repo_issues", [])),
            benchmarks=_load_list(BenchmarkRecord, data.get("benchmarks", [])),
            datasets=_load_list(DatasetRecord, data.get("datasets", [])),
            models=_load_list(ModelRecord, data.get("models", [])),
            methods=_load_list(MethodRecord, data.get("methods", [])),
            chunks=_load_list(DocumentChunk, data.get("chunks", [])),
            claims=_load_list(ClaimRecord, data.get("claims", [])),
            communities=_load_list(CommunityRecord, data.get("communities", [])),
            extraction_quarantine=_load_list(ExtractionQuarantineRecord, data.get("extraction_quarantine", [])),
        )

    def merge(self, other: "NormalizedCorpus") -> "NormalizedCorpus":
        return NormalizedCorpus(
            source_records=_dedupe(self.source_records + other.source_records),
            papers=_dedupe(self.papers + other.papers),
            authors=_dedupe(self.authors + other.authors),
            organizations=_dedupe(self.organizations + other.organizations),
            repos=_dedupe(self.repos + other.repos),
            repo_documents=_dedupe(self.repo_documents + other.repo_documents),
            repo_releases=_dedupe(self.repo_releases + other.repo_releases),
            repo_issues=_dedupe(self.repo_issues + other.repo_issues),
            benchmarks=_dedupe(self.benchmarks + other.benchmarks),
            datasets=_dedupe(self.datasets + other.datasets),
            models=_dedupe(self.models + other.models),
            methods=_dedupe(self.methods + other.methods),
            chunks=_dedupe(self.chunks + other.chunks),
            claims=_dedupe(self.claims + other.claims),
            communities=_dedupe(self.communities + other.communities),
            extraction_quarantine=_dedupe(self.extraction_quarantine + other.extraction_quarantine),
        )


@dataclass
class GraphNode:
    id: str
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class GraphEdge:
    id: str
    type: str
    start_id: str
    end_id: str
    properties: dict[str, Any]


@dataclass
class GraphArtifact:
    schema_version: str
    created_at: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    source_records: list[SourceRecord] = field(default_factory=list)
    entity_resolution_decisions: list[EntityResolutionDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphArtifact":
        return cls(
            schema_version=data["schema_version"],
            created_at=data["created_at"],
            nodes=_load_list(GraphNode, data.get("nodes", [])),
            edges=_load_list(GraphEdge, data.get("edges", [])),
            source_records=_load_list(SourceRecord, data.get("source_records", [])),
            entity_resolution_decisions=_load_list(EntityResolutionDecision, data.get("entity_resolution_decisions", [])),
        )

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}

    def stats(self) -> dict[str, Any]:
        labels: dict[str, int] = {}
        rels: dict[str, int] = {}
        for node in self.nodes:
            for label in node.labels:
                labels[label] = labels.get(label, 0) + 1
        for edge in self.edges:
            rels[edge.type] = rels.get(edge.type, 0) + 1
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "labels": labels,
            "relationships": rels,
        }


T = TypeVar("T")


def _load_list(cls: type[T], rows: list[dict[str, Any]]) -> list[T]:
    names = {field.name for field in fields(cls)}
    return [cls(**{key: value for key, value in row.items() if key in names}) for row in rows]


def _dedupe(items: list[T]) -> list[T]:
    seen: set[str] = set()
    result: list[T] = []
    for item in items:
        item_id = getattr(item, "id")
        if item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result
