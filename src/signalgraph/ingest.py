from __future__ import annotations

from dataclasses import dataclass

from .config import ProjectPaths
from .models import NormalizedCorpus
from .normalization import normalize_stored_documents
from .sources import RawDocument, SourceSearchService
from .storage import NormalizedStore, RawStorage


@dataclass
class CorpusSizeControls:
    max_source_records: int | None = None
    max_papers: int | None = None
    max_authors: int | None = None
    max_organizations: int | None = None
    max_repos: int | None = None
    max_repo_documents: int | None = None
    max_repo_releases: int | None = None
    max_repo_issues: int | None = None
    max_benchmarks: int | None = None
    max_datasets: int | None = None
    max_models: int | None = None
    max_methods: int | None = None
    max_chunks: int | None = None
    max_claims: int | None = None

    @classmethod
    def starter_targets(cls) -> "CorpusSizeControls":
        return cls(
            max_papers=250,
            max_repos=100,
            max_methods=40,
            max_benchmarks=50,
            max_datasets=50,
            max_models=50,
            max_claims=800,
        )

    def apply(self, corpus: NormalizedCorpus) -> NormalizedCorpus:
        return NormalizedCorpus(
            source_records=_limit(corpus.source_records, self.max_source_records),
            papers=_limit(corpus.papers, self.max_papers),
            authors=_limit(corpus.authors, self.max_authors),
            organizations=_limit(corpus.organizations, self.max_organizations),
            repos=_limit(corpus.repos, self.max_repos),
            repo_documents=_limit(corpus.repo_documents, self.max_repo_documents),
            repo_releases=_limit(corpus.repo_releases, self.max_repo_releases),
            repo_issues=_limit(corpus.repo_issues, self.max_repo_issues),
            benchmarks=_limit(corpus.benchmarks, self.max_benchmarks),
            datasets=_limit(corpus.datasets, self.max_datasets),
            models=_limit(corpus.models, self.max_models),
            methods=_limit(corpus.methods, self.max_methods),
            chunks=_limit(corpus.chunks, self.max_chunks),
            claims=_limit(corpus.claims, self.max_claims),
        )


class Ingestor:
    def __init__(
        self,
        paths: ProjectPaths,
        search_service: SourceSearchService | None = None,
        raw_storage: RawStorage | None = None,
        normalized_store: NormalizedStore | None = None,
    ):
        self.paths = paths
        self.search_service = search_service or SourceSearchService()
        self.raw_storage = raw_storage or RawStorage(paths)
        self.normalized_store = normalized_store or NormalizedStore(paths)

    def search(self, topic: str, limit: int = 25, source_names: list[str] | None = None, per_source_limit: int | None = None) -> list[RawDocument]:
        return self.search_service.search(topic, limit=limit, source_names=source_names, per_source_limit=per_source_limit)

    def run(
        self,
        topic: str,
        limit: int = 25,
        source_names: list[str] | None = None,
        per_source_limit: int | None = None,
        size_controls: CorpusSizeControls | None = None,
    ) -> dict[str, int]:
        documents = self.search(topic, limit=limit, source_names=source_names, per_source_limit=per_source_limit)
        pairs = []
        for document in documents:
            source_record = self.raw_storage.store(
                source_name=document.source_name,
                source_url=document.source_url,
                source_id=document.source_id,
                request_params=document.request_params,
                raw_payload=document.raw_payload,
                source_metadata=document.source_metadata,
            )
            pairs.append((document, source_record))
        corpus = normalize_stored_documents(pairs)
        if size_controls:
            corpus = size_controls.apply(corpus)
        merged = self.normalized_store.merge_and_save(corpus)
        return {
            "raw_documents": len(documents),
            "source_records": len(merged.source_records),
            "papers": len(merged.papers),
            "authors": len(merged.authors),
            "organizations": len(merged.organizations),
            "repos": len(merged.repos),
            "repo_documents": len(merged.repo_documents),
            "repo_releases": len(merged.repo_releases),
            "repo_issues": len(merged.repo_issues),
            "benchmarks": len(merged.benchmarks),
            "datasets": len(merged.datasets),
            "models": len(merged.models),
            "methods": len(merged.methods),
            "claims": len(merged.claims),
            "chunks": len(merged.chunks),
        }


def _limit(items: list, max_items: int | None) -> list:
    if max_items is None:
        return list(items)
    return list(items)[: max(0, max_items)]
