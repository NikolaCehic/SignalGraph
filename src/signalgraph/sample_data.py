from __future__ import annotations

from .config import ProjectPaths
from .normalization import normalize_stored_documents
from .sources import RawDocument
from .storage import NormalizedStore, RawStorage


SAMPLE_DOCUMENTS = [
    RawDocument(
        "sample",
        "https://arxiv.org/abs/2404.16130",
        "2404.16130",
        {"topic": "GraphRAG agent memory sample"},
        {
            "kind": "paper",
            "id": "paper:sample:graphrag",
            "title": "From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
            "abstract": "GraphRAG uses entity graphs, community detection, and community reports to improve answers for global questions over a corpus. The method is useful when evidence requires multi-hop relationships across papers, claims, and source chunks.",
            "published_at": "2024-04-24",
            "venue": "arXiv",
            "arxiv_id": "2404.16130",
            "citation_count": 128,
            "authors": ["Darren Edge", "Ha Trinh"],
        },
    ),
    RawDocument(
        "sample",
        "https://github.com/microsoft/graphrag",
        "microsoft/graphrag",
        {"topic": "GraphRAG agent memory sample"},
        {
            "kind": "repo",
            "id": "repo:microsoft/graphrag",
            "owner": "microsoft",
            "name": "graphrag",
            "full_name": "microsoft/graphrag",
            "description": "A data pipeline and transformation suite for GraphRAG with community detection, reports, indexing, and retrieval workflows.",
            "stars": 26000,
            "forks": 2400,
            "license": "MIT",
            "default_branch": "main",
            "last_commit_at": "2026-03-15T10:00:00Z",
            "open_issues_count": 150,
            "health_score": 0.87,
            "risk_score": 0.13,
            "topics": ["graphrag", "retrieval", "knowledge-graph", "community-detection"],
        },
    ),
    RawDocument(
        "sample",
        "https://arxiv.org/abs/2310.08560",
        "2310.08560",
        {"topic": "agent memory sample"},
        {
            "kind": "paper",
            "id": "paper:sample:memgpt",
            "title": "MemGPT: Towards LLMs as Operating Systems",
            "abstract": "Agent memory systems manage short-term and long-term context for language agents. The approach highlights production risks around memory freshness, retrieval policy, and evaluation of persisted context.",
            "published_at": "2023-10-12",
            "venue": "arXiv",
            "arxiv_id": "2310.08560",
            "citation_count": 340,
            "authors": ["Charles Packer", "Sarah Wooders"],
        },
    ),
    RawDocument(
        "sample",
        "https://github.com/cpacker/MemGPT",
        "cpacker/MemGPT",
        {"topic": "agent memory sample"},
        {
            "kind": "repo",
            "id": "repo:cpacker/memgpt",
            "owner": "cpacker",
            "name": "MemGPT",
            "full_name": "cpacker/MemGPT",
            "description": "Reference implementation for agent memory and long-term memory agents with examples and evaluation scripts.",
            "stars": 13000,
            "forks": 1400,
            "license": "Apache-2.0",
            "default_branch": "main",
            "last_commit_at": "2025-11-02T12:00:00Z",
            "open_issues_count": 220,
            "health_score": 0.74,
            "risk_score": 0.26,
            "topics": ["agent-memory", "llm-agents", "retrieval", "evaluation"],
        },
    ),
    RawDocument(
        "sample",
        "https://openalex.org/W999999",
        "W999999",
        {"topic": "RAG evaluation sample"},
        {
            "kind": "paper",
            "id": "paper:sample:rag-eval",
            "title": "RAG Evaluation with Faithfulness and Context Recall",
            "abstract": "RAG evaluation should include faithfulness, answer relevance, context precision, context recall, graph path recall, and evidence-chain completeness. Benchmarks are strongest when they compare vector-only retrieval against graph-aware retrieval.",
            "published_at": "2024-08-01",
            "venue": "OpenAlex sample",
            "citation_count": 42,
            "authors": ["Alex Evaluator"],
        },
    ),
    RawDocument(
        "semantic_scholar",
        "https://www.semanticscholar.org/paper/S2-GRAPHRAG-MEMORY",
        "S2-GRAPHRAG-MEMORY",
        {"topic": "GraphRAG agent memory sample"},
        {
            "paperId": "S2-GRAPHRAG-MEMORY",
            "title": "Graph-Aware Agent Memory for Production Assistants",
            "abstract": "GraphRAG and agent memory systems can connect papers, repositories, datasets, and evaluation benchmarks for production assistants.",
            "year": 2025,
            "publicationDate": "2025-02-15",
            "venue": "Semantic Scholar sample",
            "externalIds": {"ArXiv": "2502.00001", "DOI": "10.0000/signalgraph.sample"},
            "url": "https://www.semanticscholar.org/paper/S2-GRAPHRAG-MEMORY",
            "citationCount": 17,
            "authors": [{"authorId": "S2-AUTHOR-1", "name": "Sam Scholar", "affiliations": ["SignalGraph Lab"]}],
            "references": [{"paperId": "S2-REF-1", "title": "GraphRAG Foundations", "url": "https://www.semanticscholar.org/paper/S2-REF-1"}],
            "citations": [{"paperId": "S2-CITE-1", "title": "Repository Exploration Agents", "url": "https://www.semanticscholar.org/paper/S2-CITE-1"}],
        },
    ),
    RawDocument(
        "github",
        "https://github.com/example/signalgraph-agent-memory",
        "example/signalgraph-agent-memory",
        {"topic": "GraphRAG agent memory sample"},
        {
            "full_name": "example/signalgraph-agent-memory",
            "name": "signalgraph-agent-memory",
            "owner": {"login": "example", "html_url": "https://github.com/example"},
            "html_url": "https://github.com/example/signalgraph-agent-memory",
            "description": "GraphRAG agent memory implementation with benchmark, dataset, and Hugging Face model integration.",
            "stargazers_count": 420,
            "forks_count": 38,
            "license": {"spdx_id": "Apache-2.0"},
            "default_branch": "main",
            "pushed_at": "2026-02-01T00:00:00Z",
            "open_issues_count": 12,
            "topics": ["graphrag", "agent-memory", "rag-evaluation"],
            "readme": {
                "doc_type": "readme",
                "path": "README.md",
                "name": "README.md",
                "html_url": "https://github.com/example/signalgraph-agent-memory#readme",
                "text": "SignalGraph Agent Memory uses GraphRAG, hybrid retrieval, and Text2Cypher to connect long-term memory datasets with benchmark evidence.",
            },
            "docs": [
                {
                    "doc_type": "docs",
                    "path": "docs/architecture.md",
                    "name": "architecture.md",
                    "html_url": "https://github.com/example/signalgraph-agent-memory/blob/main/docs/architecture.md",
                    "text": "The architecture links repo documents, releases, and issue risk signals to production readiness claims.",
                }
            ],
            "changelog": {
                "doc_type": "changelog",
                "path": "CHANGELOG.md",
                "name": "CHANGELOG.md",
                "html_url": "https://github.com/example/signalgraph-agent-memory/blob/main/CHANGELOG.md",
                "text": "The February 2026 refresh improves benchmark coverage and fixes stale retrieval cache handling.",
            },
            "releases": [
                {
                    "tag_name": "release-2026-02",
                    "name": "Benchmark refresh",
                    "published_at": "2026-02-01T00:00:00Z",
                    "html_url": "https://github.com/example/signalgraph-agent-memory/releases/tag/release-2026-02",
                    "body": "Adds dataset cards, release notes, and benchmark evaluation scripts.",
                    "prerelease": False,
                }
            ],
            "issues": [
                {
                    "number": 7,
                    "title": "Installation regression with optional graph backend",
                    "state": "open",
                    "labels": ["bug", "install"],
                    "html_url": "https://github.com/example/signalgraph-agent-memory/issues/7",
                    "created_at": "2026-01-15T00:00:00Z",
                    "updated_at": "2026-02-02T00:00:00Z",
                    "body": "Users report a broken optional dependency path when enabling graph-aware retrieval.",
                }
            ],
            "risk_signals": ["recent_issue_risk"],
        },
    ),
    RawDocument(
        "huggingface",
        "https://huggingface.co/example/graphrag-agent-memory",
        "example/graphrag-agent-memory",
        {"topic": "GraphRAG agent memory sample", "asset_type": "model"},
        {
            "asset_type": "model",
            "modelId": "example/graphrag-agent-memory",
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "downloads": 1200,
            "likes": 45,
            "tags": ["graphrag", "agent-memory", "benchmark"],
            "lastModified": "2026-01-20T00:00:00Z",
            "cardData": {"license": "apache-2.0", "datasets": ["example/agent-memory-eval"], "metrics": ["faithfulness"]},
            "description": "Model card for a GraphRAG agent memory assistant evaluated with faithfulness and context recall benchmarks.",
        },
    ),
    RawDocument(
        "huggingface",
        "https://huggingface.co/datasets/example/agent-memory-eval",
        "example/agent-memory-eval",
        {"topic": "GraphRAG agent memory sample", "asset_type": "dataset"},
        {
            "asset_type": "dataset",
            "id": "example/agent-memory-eval",
            "downloads": 850,
            "likes": 32,
            "tags": ["dataset", "benchmark", "rag-evaluation"],
            "cardData": {"license": "cc-by-4.0", "task_categories": ["question-answering"], "metrics": ["context_recall"]},
            "description": "Dataset card for agent memory and GraphRAG evaluation with benchmark questions and source-record provenance.",
        },
    ),
]


def ensure_sample_corpus(paths: ProjectPaths, force: bool = False) -> None:
    store = NormalizedStore(paths)
    existing = store.load()
    if not force and (existing.papers or existing.repos or existing.claims):
        return
    raw = RawStorage(paths)
    pairs = []
    for document in SAMPLE_DOCUMENTS:
        record = raw.store(
            source_name=document.source_name,
            source_url=document.source_url,
            source_id=document.source_id,
            request_params=document.request_params,
            raw_payload=document.raw_payload,
        )
        pairs.append((document, record))
    store.save(normalize_stored_documents(pairs))
