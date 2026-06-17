from __future__ import annotations

import base64

from signalgraph.config import ProjectPaths
from signalgraph.ingest import CorpusSizeControls
from signalgraph.normalization import normalize_stored_documents
from signalgraph.sources import ArxivClient, GitHubClient, HuggingFaceClient, OpenAlexClient, RawDocument, SemanticScholarClient
from signalgraph.storage import RawStorage


class FakeFetcher:
    def get_text(self, url, headers=None):
        return """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00001</id>
            <title>GraphRAG for Agent Memory</title>
            <summary>GraphRAG improves agent memory retrieval with graph paths and evidence-chain evaluation.</summary>
            <published>2024-01-01T00:00:00Z</published>
            <updated>2024-01-02T00:00:00Z</updated>
            <author><name>Ada Researcher</name></author>
            <category term="cs.AI"/>
            <arxiv:doi>10.0000/example</arxiv:doi>
          </entry>
        </feed>"""

    def get_json(self, url, headers=None):
        if "openalex" in url:
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Text2Cypher Retrieval Evaluation",
                        "publication_year": 2025,
                        "cited_by_count": 9,
                        "abstract_inverted_index": {"Text2Cypher": [0], "supports": [1], "structured": [2], "graph": [3], "lookup": [4]},
                        "authorships": [
                            {
                                "author": {"id": "https://openalex.org/A1", "display_name": "Grace Graph"},
                                "institutions": [{"id": "https://openalex.org/I1", "display_name": "Graph Lab", "type": "education"}],
                            }
                        ],
                    }
                ]
            }
        if "semanticscholar" in url:
            return {
                "data": [
                    {
                        "paperId": "S2-123",
                        "title": "Semantic Scholar GraphRAG Agent Memory",
                        "abstract": "GraphRAG links agent memory papers with datasets, benchmarks, and repository evidence.",
                        "year": 2025,
                        "publicationDate": "2025-01-05",
                        "venue": "Semantic Scholar fixture",
                        "externalIds": {"ArXiv": "2501.00001", "DOI": "10.0000/s2"},
                        "url": "https://www.semanticscholar.org/paper/S2-123",
                        "citationCount": 11,
                        "authors": [{"authorId": "A-S2", "name": "Sasha Scholar", "affiliations": ["Fixture AI Lab"]}],
                        "references": [{"paperId": "S2-REF", "title": "GraphRAG Foundations", "url": "https://www.semanticscholar.org/paper/S2-REF"}],
                        "citations": [{"paperId": "S2-CITE", "title": "Production Agent Memory", "url": "https://www.semanticscholar.org/paper/S2-CITE"}],
                    }
                ]
            }
        if "huggingface.co/api/models" in url:
            return [
                {
                    "modelId": "example/graphrag-agent-model",
                    "pipeline_tag": "text-generation",
                    "downloads": 1000,
                    "likes": 25,
                    "tags": ["graphrag", "agent-memory", "benchmark"],
                    "lastModified": "2026-01-01T00:00:00Z",
                    "cardData": {"license": "apache-2.0", "datasets": ["example/agent-memory-eval"], "metrics": ["faithfulness"]},
                    "description": "Model card for GraphRAG agent memory benchmark evaluation.",
                }
            ]
        if "huggingface.co/api/datasets" in url:
            return [
                {
                    "id": "example/agent-memory-eval",
                    "downloads": 500,
                    "likes": 12,
                    "tags": ["dataset", "benchmark", "rag-evaluation"],
                    "cardData": {"license": "cc-by-4.0", "task_categories": ["question-answering"], "metrics": ["context_recall"]},
                    "description": "Dataset card for GraphRAG agent memory benchmark questions.",
                }
            ]
        if "search/repositories" in url:
            return {
                "items": [
                    {
                        "full_name": "example/graphrag-agent-memory",
                        "name": "graphrag-agent-memory",
                        "owner": {"login": "example", "html_url": "https://github.com/example"},
                        "html_url": "https://github.com/example/graphrag-agent-memory",
                        "description": "GraphRAG and agent memory implementation with RAG evaluation.",
                        "stargazers_count": 50,
                        "forks_count": 4,
                        "license": {"spdx_id": "MIT"},
                        "default_branch": "main",
                        "pushed_at": "2026-01-01T00:00:00Z",
                        "open_issues_count": 3,
                        "topics": ["graphrag", "agent-memory"],
                    }
                ]
            }
        if url.endswith("/readme"):
            return _content("README.md", "GraphRAG agent memory README with benchmark and dataset evidence.")
        if "contents/docs/README.md" in url:
            return _content("docs/README.md", "Docs describe hybrid retrieval and Text2Cypher production setup.")
        if "contents/CHANGELOG.md" in url:
            return _content("CHANGELOG.md", "Changelog release notes fix an installation regression and improve benchmark coverage.")
        if "releases" in url:
            return [
                {
                    "tag_name": "v1.2.0",
                    "name": "GraphRAG benchmark refresh",
                    "published_at": "2026-01-15T00:00:00Z",
                    "html_url": "https://github.com/example/graphrag-agent-memory/releases/tag/v1.2.0",
                    "body": "Adds release notes for datasets and benchmark scripts.",
                    "prerelease": False,
                }
            ]
        if "issues" in url:
            return [
                {
                    "number": 42,
                    "title": "Installation regression in graph backend",
                    "state": "open",
                    "labels": [{"name": "bug"}, {"name": "install"}],
                    "html_url": "https://github.com/example/graphrag-agent-memory/issues/42",
                    "created_at": "2026-01-10T00:00:00Z",
                    "updated_at": "2026-01-20T00:00:00Z",
                    "body": "The optional graph backend has a broken dependency path.",
                }
            ]
        return {}


def _content(path, text):
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {
        "path": path,
        "name": path.rsplit("/", 1)[-1],
        "html_url": f"https://github.com/example/graphrag-agent-memory/blob/main/{path}",
        "download_url": f"https://raw.githubusercontent.com/example/graphrag-agent-memory/main/{path}",
        "encoding": "base64",
        "content": encoded,
    }


def test_public_source_clients_parse_fixture_payloads_without_network():
    fetcher = FakeFetcher()
    arxiv = ArxivClient(fetcher).search("agent memory", limit=1)
    openalex = OpenAlexClient(fetcher).search("Text2Cypher", limit=1)
    semantic = SemanticScholarClient(fetcher, api_key="").search("GraphRAG", limit=1)
    github = GitHubClient(fetcher, token="").search("GraphRAG", limit=1)
    huggingface = HuggingFaceClient(fetcher, token="").search("GraphRAG", limit=2)

    assert arxiv[0].source_name == "arxiv"
    assert arxiv[0].raw_payload["title"] == "GraphRAG for Agent Memory"
    assert openalex[0].source_name == "openalex"
    assert openalex[0].raw_payload["display_name"] == "Text2Cypher Retrieval Evaluation"
    assert semantic[0].source_name == "semantic_scholar"
    assert semantic[0].raw_payload["paperId"] == "S2-123"
    assert github[0].source_name == "github"
    assert github[0].raw_payload["full_name"] == "example/graphrag-agent-memory"
    assert github[0].raw_payload["readme"]["text"].startswith("GraphRAG agent memory README")
    assert github[0].raw_payload["releases"][0]["tag_name"] == "v1.2.0"
    assert github[0].raw_payload["issues"][0]["number"] == 42
    assert {doc.raw_payload["asset_type"] for doc in huggingface} == {"model", "dataset"}


def test_raw_storage_records_required_metadata_and_normalization(tmp_path):
    paths = ProjectPaths(tmp_path)
    raw = RawStorage(paths)
    document = RawDocument(
        "github",
        "https://github.com/example/graphrag",
        "example/graphrag",
        {"q": "GraphRAG"},
        {
            "full_name": "example/graphrag",
            "name": "graphrag",
            "owner": {"login": "example"},
            "html_url": "https://github.com/example/graphrag",
            "description": "GraphRAG repository with benchmark and dataset evidence.",
            "stargazers_count": 100,
            "forks_count": 10,
            "license": {"spdx_id": "MIT"},
            "pushed_at": "2026-01-01T00:00:00Z",
        },
    )
    record = raw.store(
        source_name=document.source_name,
        source_url=document.source_url,
        source_id=document.source_id,
        request_params=document.request_params,
        raw_payload=document.raw_payload,
    )

    assert record.source_name == "github"
    assert record.source_url == "https://github.com/example/graphrag"
    assert record.source_id == "example/graphrag"
    assert record.fetched_at
    assert record.request_params == {"q": "GraphRAG"}
    assert len(record.response_hash) == 64
    assert (tmp_path / record.raw_payload_path).exists()
    assert record.license_or_terms_note
    assert record.freshness_policy
    assert record.rate_limit_note
    assert record.cache_policy
    assert record.cache_key
    assert record.cache_status == "miss"
    assert record.quality_gate_status == "pass"
    assert record.quality_gate_reasons == []

    corpus = normalize_stored_documents([(document, record)])
    assert corpus.source_records
    assert corpus.repos
    assert corpus.methods
    assert corpus.claims
    assert corpus.chunks
    assert corpus.benchmarks
    assert corpus.datasets


def test_expanded_normalization_for_semantic_scholar_huggingface_and_github_evidence(tmp_path):
    paths = ProjectPaths(tmp_path)
    raw = RawStorage(paths)
    fetcher = FakeFetcher()
    documents = []
    documents.extend(SemanticScholarClient(fetcher, api_key="").search("GraphRAG", limit=1))
    documents.extend(GitHubClient(fetcher, token="").search("GraphRAG", limit=1))
    documents.extend(HuggingFaceClient(fetcher, token="").search("GraphRAG", limit=2))
    pairs = []
    for document in documents:
        record = raw.store(
            source_name=document.source_name,
            source_url=document.source_url,
            source_id=document.source_id,
            request_params=document.request_params,
            raw_payload=document.raw_payload,
            source_metadata=document.source_metadata,
        )
        pairs.append((document, record))

    corpus = normalize_stored_documents(pairs)

    assert any(paper.semantic_scholar_id == "S2-123" for paper in corpus.papers)
    assert any(author.semantic_scholar_id == "A-S2" for author in corpus.authors)
    assert any(org.name == "Fixture AI Lab" for org in corpus.organizations)
    assert any(doc.doc_type == "readme" and "benchmark" in doc.text.lower() for doc in corpus.repo_documents)
    assert any(doc.doc_type == "changelog" for doc in corpus.repo_documents)
    assert any(release.tag_name == "v1.2.0" for release in corpus.repo_releases)
    assert any("installation_risk" in issue.risk_signals for issue in corpus.repo_issues)
    assert any(model.huggingface_id == "example/graphrag-agent-model" for model in corpus.models)
    assert any(dataset.huggingface_id == "example/agent-memory-eval" for dataset in corpus.datasets)
    assert {"semantic_scholar", "github", "huggingface"} <= {record.source_name for record in corpus.source_records}
    sections = {chunk.section for chunk in corpus.chunks}
    assert {"readme", "changelog", "release_notes", "issue", "huggingface_model_card", "huggingface_dataset_card"} <= sections


def test_source_quality_gates_quarantine_bad_payloads_and_size_controls_limit_corpus(tmp_path):
    paths = ProjectPaths(tmp_path)
    raw = RawStorage(paths)
    bad_document = RawDocument("semantic_scholar", "", "", {}, {})
    bad_record = raw.store(
        source_name=bad_document.source_name,
        source_url=bad_document.source_url,
        source_id=bad_document.source_id,
        request_params=bad_document.request_params,
        raw_payload=bad_document.raw_payload,
    )
    bad_corpus = normalize_stored_documents([(bad_document, bad_record)])
    assert bad_record.quality_gate_status == "quarantine"
    assert {"missing_source_url", "missing_stable_source_id", "empty_payload"} <= set(bad_record.quality_gate_reasons)
    assert bad_corpus.source_records
    assert not bad_corpus.papers

    fetcher = FakeFetcher()
    documents = SemanticScholarClient(fetcher, api_key="").search("GraphRAG", limit=1) + GitHubClient(fetcher, token="").search("GraphRAG", limit=1)
    pairs = []
    for document in documents:
        record = raw.store(
            source_name=document.source_name,
            source_url=document.source_url,
            source_id=document.source_id,
            request_params=document.request_params,
            raw_payload=document.raw_payload,
            source_metadata=document.source_metadata,
        )
        pairs.append((document, record))
    corpus = normalize_stored_documents(pairs)
    limited = CorpusSizeControls(max_papers=1, max_repos=0, max_claims=1, max_repo_issues=0).apply(corpus)
    assert len(limited.papers) == 1
    assert len(limited.repos) == 0
    assert len(limited.claims) == 1
    assert len(limited.repo_issues) == 0
