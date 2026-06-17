from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED_TOPICS = [
    "GraphRAG",
    "agent memory",
    "long-term memory agents",
    "Text2Cypher",
    "knowledge graph agents",
    "RAG evaluation",
    "contextual retrieval",
    "repository exploration agents",
]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout for the engine-first local project."""

    root: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        configured = os.environ.get("SIGNALGRAPH_HOME")
        return cls(Path(configured).expanduser().resolve() if configured else Path.cwd().resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def graph_dir(self) -> Path:
        return self.artifacts_dir / "graph"

    @property
    def cypher_dir(self) -> Path:
        return self.artifacts_dir / "cypher"

    @property
    def traces_dir(self) -> Path:
        return self.artifacts_dir / "traces"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def decision_memos_dir(self) -> Path:
        return self.reports_dir / "decision_memos"

    @property
    def raw_manifest_path(self) -> Path:
        return self.raw_dir / "source_records.jsonl"

    @property
    def normalized_corpus_path(self) -> Path:
        return self.normalized_dir / "corpus.json"

    @property
    def graph_artifact_path(self) -> Path:
        return self.graph_dir / "signalgraph_graph.json"

    @property
    def cypher_export_path(self) -> Path:
        return self.cypher_dir / "signalgraph_export.cypher"

    @property
    def evidence_query_path(self) -> Path:
        return self.cypher_dir / "evidence_path_queries.cypher"

    @property
    def eval_results_path(self) -> Path:
        return self.artifacts_dir / "eval_results.json"

    @property
    def retrieval_comparison_path(self) -> Path:
        return self.artifacts_dir / "retrieval_comparison.csv"

    @property
    def query_trace_path(self) -> Path:
        return self.traces_dir / "query_traces.jsonl"

    def ensure(self) -> None:
        for path in [
            self.raw_dir,
            self.normalized_dir,
            self.eval_dir,
            self.graph_dir,
            self.cypher_dir,
            self.traces_dir,
            self.reports_dir,
            self.decision_memos_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


SOURCE_TERMS = {
    "arxiv": "arXiv public API metadata; acknowledge arXiv when presenting public results.",
    "openalex": "OpenAlex public API metadata under OpenAlex terms.",
    "semantic_scholar": "Semantic Scholar Graph API metadata; optional API key improves rate limits.",
    "github": "GitHub REST API metadata; optional token improves rate limits.",
    "huggingface": "Hugging Face Hub public model and dataset metadata/cards; optional token improves rate limits.",
    "sample": "Bundled deterministic sample corpus for local tests and evaluation runs.",
}
