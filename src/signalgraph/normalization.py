from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from .extraction import extract_traceable_records
from .models import (
    AuthorRecord,
    BenchmarkRecord,
    ClaimRecord,
    DatasetRecord,
    DocumentChunk,
    MethodRecord,
    ModelRecord,
    NormalizedCorpus,
    OrganizationRecord,
    PaperRecord,
    RepoDocumentRecord,
    RepoIssueRecord,
    RepoReleaseRecord,
    RepoRecord,
    SourceRecord,
)
from .providers import METHOD_PATTERNS, StructuredExtractionProvider
from .sources import RawDocument, source_policy_metadata
from .utils import first_sentence, short_hash, stable_hash, tokenize, utc_now


BENCHMARK_KEYWORDS = {
    "benchmark": "Benchmark",
    "leaderboard": "Leaderboard",
    "faithfulness": "Faithfulness",
    "context recall": "Context Recall",
    "answer relevance": "Answer Relevance",
}

DATASET_KEYWORDS = {
    "dataset": "Research Dataset",
    "corpus": "Research Corpus",
    "knowledge base": "Knowledge Base",
}

MODEL_KEYWORDS = {
    "gpt": "GPT-family Model",
    "llama": "Llama-family Model",
    "embedding": "Embedding Model",
}


def normalize_stored_documents(
    pairs: list[tuple[RawDocument, SourceRecord]],
    extraction_provider: StructuredExtractionProvider | None = None,
) -> NormalizedCorpus:
    corpus = NormalizedCorpus(source_records=[record for _, record in pairs])
    for document, record in pairs:
        if getattr(record, "quality_gate_status", "pass") != "pass":
            continue
        if document.source_name == "arxiv":
            _normalize_arxiv(document.raw_payload, record, corpus, extraction_provider)
        elif document.source_name == "openalex":
            _normalize_openalex(document.raw_payload, record, corpus, extraction_provider)
        elif document.source_name == "semantic_scholar":
            _normalize_semantic_scholar(document.raw_payload, record, corpus, extraction_provider)
        elif document.source_name == "github":
            _normalize_github(document.raw_payload, record, corpus, extraction_provider)
        elif document.source_name == "huggingface":
            _normalize_huggingface(document.raw_payload, record, corpus, extraction_provider)
        elif document.source_name == "sample":
            _normalize_sample(document.raw_payload, record, corpus, extraction_provider)
        else:
            _normalize_generic(document.raw_payload, record, corpus, extraction_provider)
    return _dedupe_corpus(corpus)


def _normalize_arxiv(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    arxiv_id = record.source_id
    source_url = payload.get("id") or record.source_url
    paper = PaperRecord(
        id=f"paper:arxiv:{_safe_id(arxiv_id)}",
        title=_clean(payload.get("title", "")),
        abstract=_clean(payload.get("summary", "")),
        published_at=payload.get("published", ""),
        venue="arXiv",
        doi=payload.get("doi", ""),
        arxiv_id=arxiv_id,
        source_url=source_url,
        source_record_id=record.id,
    )
    corpus.papers.append(paper)
    for author_name in payload.get("authors", []):
        author = AuthorRecord(
            id=f"author:{_safe_id(author_name)}",
            name=author_name,
            source_url=source_url,
            source_record_id=record.id,
        )
        corpus.authors.append(author)
    _append_text_derivatives("Paper", paper.id, paper.title, paper.abstract, source_url, record, corpus, extraction_provider=extraction_provider)


def _normalize_openalex(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    source_url = payload.get("id") or record.source_url
    abstract = _openalex_abstract(payload.get("abstract_inverted_index") or {})
    title = _clean(payload.get("display_name", ""))
    paper = PaperRecord(
        id=f"paper:openalex:{_safe_id(payload.get('id') or record.source_id)}",
        title=title,
        abstract=abstract,
        published_at=str(payload.get("publication_date") or payload.get("publication_year") or ""),
        venue=(payload.get("primary_location") or {}).get("source", {}).get("display_name", "") or "OpenAlex",
        doi=payload.get("doi") or "",
        openalex_id=payload.get("id") or "",
        citation_count=int(payload.get("cited_by_count") or 0),
        source_url=source_url,
        source_record_id=record.id,
    )
    corpus.papers.append(paper)
    for authorship in payload.get("authorships", []) or []:
        author_data = authorship.get("author") or {}
        name = author_data.get("display_name", "")
        if name:
            corpus.authors.append(
                AuthorRecord(
                    id=f"author:{_safe_id(author_data.get('id') or name)}",
                    name=name,
                    openalex_id=author_data.get("id", ""),
                    affiliation_text=", ".join(i.get("display_name", "") for i in authorship.get("institutions", []) if i.get("display_name")),
                    source_url=source_url,
                    source_record_id=record.id,
                )
            )
        for institution in authorship.get("institutions", []) or []:
            if institution.get("display_name"):
                corpus.organizations.append(
                    OrganizationRecord(
                        id=f"org:{_safe_id(institution.get('id') or institution.get('display_name'))}",
                        name=institution.get("display_name", ""),
                        type=institution.get("type", ""),
                        openalex_id=institution.get("id", ""),
                        source_url=source_url,
                        source_record_id=record.id,
                    )
                )
    _append_text_derivatives("Paper", paper.id, paper.title, paper.abstract, source_url, record, corpus, extraction_provider=extraction_provider)


def _normalize_semantic_scholar(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    source_url = payload.get("url") or record.source_url
    external = payload.get("externalIds") or {}
    semantic_id = payload.get("paperId") or record.source_id
    title = _clean(payload.get("title", ""))
    abstract = _clean(payload.get("abstract", ""))
    paper = PaperRecord(
        id=f"paper:semanticscholar:{_safe_id(semantic_id)}",
        title=title,
        abstract=abstract,
        published_at=str(payload.get("publicationDate") or payload.get("year") or ""),
        venue=payload.get("venue") or "Semantic Scholar",
        doi=external.get("DOI") or external.get("doi") or "",
        arxiv_id=external.get("ArXiv") or external.get("arXiv") or "",
        semantic_scholar_id=semantic_id,
        citation_count=int(payload.get("citationCount") or 0),
        source_url=source_url,
        source_record_id=record.id,
    )
    corpus.papers.append(paper)
    for author_data in payload.get("authors", []) or []:
        name = author_data.get("name", "")
        if not name:
            continue
        author_id = author_data.get("authorId") or name
        corpus.authors.append(
            AuthorRecord(
                id=f"author:{_safe_id(author_id)}",
                name=name,
                semantic_scholar_id=author_data.get("authorId", ""),
                affiliation_text=", ".join(author_data.get("affiliations", []) or []),
                source_url=source_url,
                source_record_id=record.id,
            )
        )
        for affiliation in author_data.get("affiliations", []) or []:
            corpus.organizations.append(
                OrganizationRecord(
                    id=f"org:{_safe_id(affiliation)}",
                    name=affiliation,
                    type="affiliation",
                    source_url=source_url,
                    source_record_id=record.id,
                )
            )
    for relation_name in ["references", "citations"]:
        for related in payload.get(relation_name, []) or []:
            related_id = related.get("paperId")
            related_title = related.get("title", "")
            if not related_id or not related_title:
                continue
            corpus.papers.append(
                PaperRecord(
                    id=f"paper:semanticscholar:{_safe_id(related_id)}",
                    title=_clean(related_title),
                    abstract="",
                    published_at="",
                    venue="Semantic Scholar",
                    semantic_scholar_id=related_id,
                    source_url=related.get("url") or f"https://www.semanticscholar.org/paper/{related_id}",
                    source_record_id=record.id,
                )
            )
    _append_text_derivatives("Paper", paper.id, paper.title, paper.abstract, source_url, record, corpus, extraction_provider=extraction_provider)


def _normalize_github(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    owner = (payload.get("owner") or {}).get("login") or payload.get("full_name", "/").split("/")[0]
    name = payload.get("name") or payload.get("full_name", "/").split("/")[-1]
    full_name = payload.get("full_name") or f"{owner}/{name}"
    pushed_at = payload.get("pushed_at") or payload.get("updated_at") or ""
    stars = int(payload.get("stargazers_count") or payload.get("stars") or 0)
    forks = int(payload.get("forks_count") or payload.get("forks") or 0)
    open_issues = int(payload.get("open_issues_count") or 0)
    releases = payload.get("releases", []) or []
    issues = payload.get("issues", []) or []
    readme = _repo_document_payload(payload.get("readme"), "readme", "README.md")
    docs = [_repo_document_payload(doc, "docs", f"docs/{idx}") for idx, doc in enumerate(payload.get("docs", []) or [], start=1)]
    changelog = _repo_document_payload(payload.get("changelog"), "changelog", "CHANGELOG.md")
    doc_payloads = [doc for doc in [readme, *docs, changelog] if doc.get("text")]
    latest_release = releases[0] if releases else {}
    has_license = bool(payload.get("license"))
    health_score = _repo_health_score(stars, forks, has_license, pushed_at, open_issues)
    repo_risk_signals = payload.get("risk_signals") or _repo_risk_signals(open_issues, issues, releases, bool(changelog.get("text")), has_license)
    repo = RepoRecord(
        id=f"repo:{full_name.lower()}",
        owner=owner,
        name=name,
        full_name=full_name,
        url=payload.get("html_url") or record.source_url,
        stars=stars,
        forks=forks,
        license=(payload.get("license") or {}).get("spdx_id", "") if isinstance(payload.get("license"), dict) else (payload.get("license") or ""),
        default_branch=payload.get("default_branch", ""),
        last_commit_at=pushed_at,
        latest_release_at=payload.get("latest_release_at", "") or latest_release.get("published_at", ""),
        open_issues_count=open_issues,
        health_score=health_score,
        risk_score=_repo_risk_score(health_score, repo_risk_signals, issues),
        description=_clean(payload.get("description", "")),
        topics=payload.get("topics", []) or [],
        readme_chars=len(readme.get("text", "")),
        docs_count=len([doc for doc in docs if doc.get("text")]),
        releases_count=len(releases),
        selected_issues_count=len(issues),
        changelog_present=bool(changelog.get("text")),
        latest_release_tag=latest_release.get("tag_name", ""),
        repo_risk_signals=repo_risk_signals,
        source_record_id=record.id,
    )
    corpus.repos.append(repo)
    corpus.organizations.append(
        OrganizationRecord(
            id=f"org:{_safe_id(owner)}",
            name=owner,
            type="github_owner",
            homepage=payload.get("owner", {}).get("html_url", "") if isinstance(payload.get("owner"), dict) else "",
            github_login=owner,
            source_url=repo.url,
            source_record_id=record.id,
        )
    )
    text = " ".join([repo.full_name, repo.description, " ".join(repo.topics)])
    _append_text_derivatives("Repo", repo.id, repo.full_name, text, repo.url, record, corpus, section="repo_metadata", extraction_provider=extraction_provider)
    for doc in doc_payloads:
        corpus.repo_documents.append(
            RepoDocumentRecord(
                id=f"repo_doc:{repo.id}:{_safe_id(doc.get('doc_type', 'doc'))}:{_safe_id(doc.get('path', 'root'))}",
                repo_id=repo.id,
                repo_full_name=repo.full_name,
                doc_type=doc.get("doc_type", "docs"),
                title=doc.get("title", ""),
                path=doc.get("path", ""),
                url=doc.get("url") or repo.url,
                text=doc.get("text", ""),
                source_url=doc.get("url") or repo.url,
                source_record_id=record.id,
            )
        )
        _append_text_derivatives("RepoDocument", repo.id, doc.get("title", ""), doc.get("text", ""), doc.get("url") or repo.url, record, corpus, section=doc.get("doc_type", "docs"), extraction_provider=extraction_provider)
    for release in releases:
        release_record = RepoReleaseRecord(
            id=f"release:{repo.id}:{_safe_id(release.get('tag_name') or release.get('name') or release.get('published_at') or 'release')}",
            repo_id=repo.id,
            repo_full_name=repo.full_name,
            tag_name=release.get("tag_name", ""),
            name=release.get("name", ""),
            published_at=release.get("published_at", ""),
            url=release.get("html_url", ""),
            body=_clean(release.get("body", "")),
            prerelease=bool(release.get("prerelease", False)),
            source_url=release.get("html_url") or repo.url,
            source_record_id=record.id,
        )
        corpus.repo_releases.append(release_record)
        _append_text_derivatives("Release", repo.id, release_record.name or release_record.tag_name, release_record.body, release_record.source_url, record, corpus, section="release_notes", extraction_provider=extraction_provider)
    for issue in issues:
        risk_signals = _issue_risk_signals(issue)
        issue_record = RepoIssueRecord(
            id=f"issue:{repo.id}:{int(issue.get('number') or 0)}",
            repo_id=repo.id,
            repo_full_name=repo.full_name,
            number=int(issue.get("number") or 0),
            title=issue.get("title", ""),
            state=issue.get("state", ""),
            labels=issue.get("labels", []) or [],
            url=issue.get("html_url", ""),
            created_at=issue.get("created_at", ""),
            updated_at=issue.get("updated_at", ""),
            body=_clean(issue.get("body", "")),
            risk_signals=risk_signals,
            source_url=issue.get("html_url") or repo.url,
            source_record_id=record.id,
        )
        corpus.repo_issues.append(issue_record)
        issue_text = " ".join([issue_record.title, issue_record.body, " ".join(issue_record.labels), " ".join(risk_signals)])
        _append_text_derivatives("Issue", repo.id, issue_record.title, issue_text, issue_record.source_url, record, corpus, section="issue", extraction_provider=extraction_provider)


def _normalize_huggingface(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    asset_type = payload.get("asset_type") or ("dataset" if str(record.source_url).startswith("https://huggingface.co/datasets/") else "model")
    asset_id = payload.get("modelId") or payload.get("id") or payload.get("_id") or record.source_id
    source_url = record.source_url or f"https://huggingface.co/{'datasets/' if asset_type == 'dataset' else ''}{asset_id}"
    owner = str(asset_id).split("/")[0] if "/" in str(asset_id) else payload.get("author", "")
    tags = [str(tag) for tag in payload.get("tags", []) or []]
    card_text = _huggingface_card_text(payload)
    if owner:
        corpus.organizations.append(
            OrganizationRecord(
                id=f"org:{_safe_id(owner)}",
                name=owner,
                type="huggingface_owner",
                huggingface_id=owner,
                source_url=source_url,
                source_record_id=record.id,
            )
        )
    if asset_type == "dataset":
        dataset = DatasetRecord(
            id=f"dataset:huggingface:{_safe_id(asset_id)}",
            name=str(asset_id),
            domain=_clean(str(payload.get("pipeline_tag") or payload.get("task_categories") or payload.get("task_ids") or "ai_research")),
            license=_huggingface_license(payload),
            provider_or_org=owner,
            dataset_type=_clean(str(payload.get("pretty_name") or payload.get("config_name") or "")),
            huggingface_id=str(asset_id),
            downloads=int(payload.get("downloads") or 0),
            likes=int(payload.get("likes") or 0),
            tags=tags,
            source_url=source_url,
            source_record_id=record.id,
        )
        corpus.datasets.append(dataset)
        text = " ".join([dataset.name, dataset.domain, dataset.license, " ".join(tags), card_text])
        _append_text_derivatives("Dataset", dataset.id, dataset.name, text, source_url, record, corpus, section="huggingface_dataset_card", extraction_provider=extraction_provider)
    else:
        model = ModelRecord(
            id=f"model:huggingface:{_safe_id(asset_id)}",
            name=str(asset_id),
            provider_or_org=owner,
            model_type=_clean(str(payload.get("pipeline_tag") or payload.get("library_name") or "model")),
            huggingface_id=str(asset_id),
            downloads=int(payload.get("downloads") or 0),
            likes=int(payload.get("likes") or 0),
            tags=tags,
            last_modified=payload.get("lastModified") or payload.get("last_modified") or "",
            source_url=source_url,
            source_record_id=record.id,
        )
        corpus.models.append(model)
        text = " ".join([model.name, model.model_type, " ".join(tags), card_text])
        _append_text_derivatives("Model", model.id, model.name, text, source_url, record, corpus, section="huggingface_model_card", extraction_provider=extraction_provider)


def _normalize_sample(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    kind = payload.get("kind")
    if kind == "paper":
        paper = PaperRecord(
            id=payload["id"],
            title=payload["title"],
            abstract=payload["abstract"],
            published_at=payload.get("published_at", ""),
            venue=payload.get("venue", ""),
            arxiv_id=payload.get("arxiv_id", ""),
            citation_count=int(payload.get("citation_count", 0)),
            source_url=record.source_url,
            source_record_id=record.id,
        )
        corpus.papers.append(paper)
        for author in payload.get("authors", []):
            corpus.authors.append(AuthorRecord(id=f"author:{_safe_id(author)}", name=author, source_url=record.source_url, source_record_id=record.id))
        _append_text_derivatives("Paper", paper.id, paper.title, paper.abstract, record.source_url, record, corpus, extraction_provider=extraction_provider)
    elif kind == "repo":
        repo = RepoRecord(
            id=payload["id"],
            owner=payload["owner"],
            name=payload["name"],
            full_name=payload["full_name"],
            url=record.source_url,
            stars=int(payload.get("stars", 0)),
            forks=int(payload.get("forks", 0)),
            license=payload.get("license", ""),
            default_branch=payload.get("default_branch", "main"),
            last_commit_at=payload.get("last_commit_at", ""),
            open_issues_count=int(payload.get("open_issues_count", 0)),
            health_score=float(payload.get("health_score", 0.75)),
            risk_score=float(payload.get("risk_score", 0.25)),
            description=payload.get("description", ""),
            topics=payload.get("topics", []),
            readme_chars=len(payload.get("readme", {}).get("text", "")) if isinstance(payload.get("readme"), dict) else 0,
            docs_count=len(payload.get("docs", []) or []),
            releases_count=len(payload.get("releases", []) or []),
            selected_issues_count=len(payload.get("issues", []) or []),
            changelog_present=bool(payload.get("changelog")),
            latest_release_tag=(payload.get("releases", [{}]) or [{}])[0].get("tag_name", ""),
            repo_risk_signals=payload.get("risk_signals", []),
            source_record_id=record.id,
        )
        corpus.repos.append(repo)
        corpus.organizations.append(OrganizationRecord(id=f"org:{_safe_id(repo.owner)}", name=repo.owner, type="github_owner", source_url=repo.url, source_record_id=record.id))
        _append_text_derivatives("Repo", repo.id, repo.full_name, repo.description + " " + " ".join(repo.topics), repo.url, record, corpus, extraction_provider=extraction_provider)
        if payload.get("readme"):
            _normalize_github(
                {
                    "owner": {"login": repo.owner},
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "html_url": repo.url,
                    "description": repo.description,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "license": repo.license,
                    "default_branch": repo.default_branch,
                    "pushed_at": repo.last_commit_at,
                    "open_issues_count": repo.open_issues_count,
                    "topics": repo.topics,
                    "readme": payload.get("readme"),
                    "docs": payload.get("docs", []),
                    "changelog": payload.get("changelog", {}),
                    "releases": payload.get("releases", []),
                    "issues": payload.get("issues", []),
                    "risk_signals": payload.get("risk_signals", []),
                },
                record,
                corpus,
                extraction_provider,
            )


def _normalize_generic(payload: dict[str, Any], record: SourceRecord, corpus: NormalizedCorpus, extraction_provider: StructuredExtractionProvider | None = None) -> None:
    title = _clean(str(payload.get("title") or payload.get("name") or record.source_id))
    text = _clean(str(payload.get("summary") or payload.get("description") or payload))
    paper = PaperRecord(
        id=f"paper:{record.source_name}:{_safe_id(record.source_id)}",
        title=title,
        abstract=text,
        published_at="",
        source_url=record.source_url,
        source_record_id=record.id,
    )
    corpus.papers.append(paper)
    _append_text_derivatives("Paper", paper.id, title, text, record.source_url, record, corpus, extraction_provider=extraction_provider)


def _append_text_derivatives(
    source_type: str,
    source_id: str,
    title: str,
    text: str,
    source_url: str,
    record: SourceRecord,
    corpus: NormalizedCorpus,
    section: str | None = None,
    extraction_provider: StructuredExtractionProvider | None = None,
) -> None:
    clean_text = _clean(text)
    if not clean_text:
        return
    chunk_id = f"chunk:{short_hash([source_id, clean_text])}"
    corpus.chunks.append(
        DocumentChunk(
            id=chunk_id,
            source_type=source_type,
            source_id=source_id,
            section=section or ("abstract" if source_type == "Paper" else f"{source_type.lower()}_metadata"),
            text=clean_text,
            start_offset=0,
            end_offset=len(clean_text),
            hash=stable_hash(clean_text),
            source_url=source_url,
            source_record_id=record.id,
        )
    )
    bundle = extract_traceable_records(
        source_type=source_type,
        source_id=source_id,
        title=title,
        text=clean_text,
        source_url=source_url,
        source_record=record,
        provider=extraction_provider,
    )
    corpus.methods.extend(bundle.methods)
    corpus.benchmarks.extend(bundle.benchmarks)
    corpus.datasets.extend(bundle.datasets)
    corpus.claims.extend(bundle.claims)
    corpus.extraction_quarantine.extend(bundle.quarantine)


def extract_methods(text: str) -> list[dict[str, Any]]:
    lowered = (text or "").lower()
    found: list[dict[str, Any]] = []
    for pattern, method in METHOD_PATTERNS.items():
        aliases = [pattern] + [alias.lower() for alias in method["aliases"]]
        if any(alias in lowered for alias in aliases):
            found.append(method)
    if not found:
        tokens = set(tokenize(lowered))
        if {"rag", "retrieval"} & tokens:
            found.append(METHOD_PATTERNS["hybrid retrieval"])
    return found


def _append_keyword_records(source_url: str, record: SourceRecord, corpus: NormalizedCorpus, text: str) -> None:
    lowered = text.lower()
    for keyword, name in BENCHMARK_KEYWORDS.items():
        if keyword in lowered:
            corpus.benchmarks.append(BenchmarkRecord(id=f"benchmark:{_safe_id(name)}", name=name, task="retrieval_quality", metric=name, source_url=source_url, source_record_id=record.id))
    for keyword, name in DATASET_KEYWORDS.items():
        if keyword in lowered:
            corpus.datasets.append(DatasetRecord(id=f"dataset:{_safe_id(name)}", name=name, domain="ai_research", source_url=source_url, source_record_id=record.id))
    for keyword, name in MODEL_KEYWORDS.items():
        if keyword in lowered:
            corpus.models.append(ModelRecord(id=f"model:{_safe_id(name)}", name=name, provider_or_org=name.split("-")[0], model_type="language_model", source_url=source_url, source_record_id=record.id))


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, offsets in index.items():
        for offset in offsets:
            positions.append((int(offset), word))
    return " ".join(word for _, word in sorted(positions))


def _repo_health_score(stars: int, forks: int, has_license: bool, last_commit_at: str, open_issues: int) -> float:
    score = 0.25
    score += min(0.25, stars / 5000)
    score += min(0.15, forks / 1000)
    score += 0.15 if has_license else 0.0
    score += 0.15 if re.search(r"20(2[4-9]|3[0-9])", last_commit_at or "") else 0.05
    score -= min(0.15, open_issues / 2000)
    return round(max(0.0, min(1.0, score)), 3)


def _repo_risk_score(health_score: float, risk_signals: list[str], issues: list[dict[str, Any]]) -> float:
    score = 1.0 - health_score
    score += min(0.25, 0.04 * len(risk_signals))
    score += min(0.2, 0.025 * len([issue for issue in issues if _issue_risk_signals(issue)]))
    return round(max(0.0, min(1.0, score)), 3)


def _repo_risk_signals(open_issues: int, issues: list[dict[str, Any]], releases: list[dict[str, Any]], has_changelog: bool, has_license: bool) -> list[str]:
    signals: list[str] = []
    if not has_license:
        signals.append("missing_license")
    if open_issues > 100:
        signals.append("high_open_issue_count")
    if not releases:
        signals.append("no_recent_release_metadata")
    if not has_changelog:
        signals.append("missing_changelog")
    if any(_issue_risk_signals(issue) for issue in issues):
        signals.append("recent_issue_risk")
    return signals


def _issue_risk_signals(issue: dict[str, Any]) -> list[str]:
    labels = " ".join(issue.get("labels", []) or []).lower()
    text = f"{issue.get('title', '')} {issue.get('body', '')}".lower()
    signals: list[str] = []
    if any(word in labels or word in text for word in ["bug", "regression", "broken", "failure"]):
        signals.append("bug_or_regression")
    if any(word in labels or word in text for word in ["install", "setup", "dependency"]):
        signals.append("installation_risk")
    if "security" in labels or "security" in text:
        signals.append("security_risk")
    return signals


def _repo_document_payload(value: Any, doc_type: str, fallback_path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    text = _clean(value.get("text", ""))
    path = value.get("path") or fallback_path
    return {
        "doc_type": value.get("doc_type") or doc_type,
        "title": value.get("name") or value.get("title") or path.rsplit("/", 1)[-1],
        "path": path,
        "url": value.get("html_url") or value.get("url") or value.get("download_url") or "",
        "text": text,
    }


def _huggingface_card_text(payload: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ["description", "summary", "README", "readme", "card_text"]:
        if payload.get(key):
            pieces.append(str(payload[key]))
    card_data = payload.get("cardData") or payload.get("card_data") or {}
    if isinstance(card_data, dict):
        for key in ["language", "license", "library_name", "pipeline_tag", "tags", "datasets", "metrics", "model_name", "pretty_name"]:
            if card_data.get(key):
                pieces.append(f"{key}: {card_data[key]}")
    elif card_data:
        pieces.append(str(card_data))
    return _clean(" ".join(pieces))


def _huggingface_license(payload: dict[str, Any]) -> str:
    card_data = payload.get("cardData") or payload.get("card_data") or {}
    if isinstance(card_data, dict) and card_data.get("license"):
        return str(card_data.get("license"))
    for tag in payload.get("tags", []) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return ""


def _claim_confidence(source_type: str) -> float:
    return {
        "Paper": 0.78,
        "Repo": 0.72,
        "RepoDocument": 0.7,
        "Release": 0.68,
        "Issue": 0.58,
        "Model": 0.7,
        "Dataset": 0.7,
    }.get(source_type, 0.65)


def _claim_type(text: str, source_type: str) -> str:
    lowered = text.lower()
    if source_type == "Issue":
        return "repo_risk"
    if source_type == "Release":
        return "release_signal"
    if source_type in {"Model", "Dataset"}:
        return "asset_metadata"
    if source_type == "RepoDocument":
        return "documentation"
    if "benchmark" in lowered or "evaluation" in lowered or "faithfulness" in lowered:
        return "benchmark"
    if "limitation" in lowered or "risk" in lowered or "challenge" in lowered:
        return "limitation"
    if source_type == "Repo":
        return "adoption"
    return "architecture"


def _claim_polarity(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["risk", "limitation", "challenge", "fails", "stale"]):
        return "caution"
    if any(word in lowered for word in ["improve", "outperform", "strong", "effective", "production"]):
        return "positive"
    return "neutral"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).lower()).strip("-")
    return cleaned or short_hash(value)


def _dedupe_corpus(corpus: NormalizedCorpus) -> NormalizedCorpus:
    # Reuse the central merge path so every model list dedupes by stable id.
    return NormalizedCorpus().merge(corpus)


def source_record_from_payload(source_name: str, source_url: str, source_id: str, payload: dict[str, Any]) -> SourceRecord:
    payload_hash = stable_hash(payload)
    metadata = source_policy_metadata(source_name)
    return SourceRecord(
        id=f"source:{source_name}:{payload_hash[:16]}",
        source_name=source_name,
        source_url=source_url,
        source_id=source_id,
        fetched_at=utc_now(),
        request_params={},
        response_hash=payload_hash,
        raw_payload_path="",
        license_or_terms_note=metadata.get("license_or_terms_note", ""),
        freshness_policy=metadata.get("freshness_policy", ""),
        stale_after_days=int(metadata.get("stale_after_days") or 0),
        rate_limit_note=metadata.get("rate_limit_note", ""),
        cache_policy=metadata.get("cache_policy", ""),
        cache_key=stable_hash([source_name, source_id, {}]),
        cache_status="memory",
        quality_gate_status="pass",
        quality_gate_reasons=[],
    )


def corpus_to_records(corpus: NormalizedCorpus) -> dict[str, list[dict[str, Any]]]:
    return {key: [asdict(item) for item in value] for key, value in asdict(corpus).items()}
