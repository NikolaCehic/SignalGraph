from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .utils import first_sentence, sentence_split


EXTRACTION_SCHEMA_VERSION = "signalgraph.extraction.v1"


@dataclass(frozen=True)
class ExtractionContext:
    source_type: str
    source_id: str
    title: str
    text: str
    source_url: str
    source_record_id: str


class StructuredExtractionProvider(Protocol):
    name: str
    version: str
    extraction_method: str

    def extract(self, context: ExtractionContext) -> dict[str, Any]:
        """Return provider-native structured extraction output."""


METHOD_PATTERNS: dict[str, dict[str, Any]] = {
    "graphrag": {
        "name": "GraphRAG",
        "aliases": ["graph retrieval augmented generation", "graph rag"],
        "category": "graph-aware retrieval",
        "description": "Retrieval augmented generation that uses a knowledge graph, graph neighborhoods, or community summaries.",
    },
    "text2cypher": {
        "name": "Text2Cypher",
        "aliases": ["text-to-cypher", "natural language to cypher"],
        "category": "structured graph query",
        "description": "Natural-language-to-Cypher query generation over property graphs.",
    },
    "agent memory": {
        "name": "Agent Memory",
        "aliases": ["long-term memory agents", "memory augmented agents", "agentic memory"],
        "category": "agent architecture",
        "description": "Persistent or retrieved memory patterns for AI agents.",
    },
    "hybrid retrieval": {
        "name": "Hybrid Retrieval",
        "aliases": ["vector plus keyword", "vector and full-text"],
        "category": "retrieval",
        "description": "Combines semantic vector search with exact lexical or full-text retrieval.",
    },
    "community detection": {
        "name": "Community Detection",
        "aliases": ["louvain", "leiden", "graph clustering"],
        "category": "graph analytics",
        "description": "Finds clusters in a graph to support global summaries and landscape analysis.",
    },
    "rag evaluation": {
        "name": "RAG Evaluation",
        "aliases": ["ragas", "faithfulness", "context recall"],
        "category": "evaluation",
        "description": "Metrics and workflows for measuring retrieval and answer quality.",
    },
    "query decomposition": {
        "name": "Query Decomposition",
        "aliases": ["subquestions", "drift search", "broad-to-local search"],
        "category": "retrieval orchestration",
        "description": "Breaks complex user questions into retrievable subquestions.",
    },
}

BENCHMARK_KEYWORDS = {
    "benchmark": ("Benchmark", "retrieval_quality", "benchmark"),
    "leaderboard": ("Leaderboard", "leaderboard", "ranking"),
    "faithfulness": ("Faithfulness", "rag_evaluation", "faithfulness"),
    "context recall": ("Context Recall", "rag_evaluation", "context_recall"),
    "answer relevance": ("Answer Relevance", "rag_evaluation", "answer_relevance"),
}

DATASET_KEYWORDS = {
    "dataset": ("Research Dataset", "ai_research"),
    "corpus": ("Research Corpus", "ai_research"),
    "knowledge base": ("Knowledge Base", "knowledge_graph"),
}


class DeterministicStructuredExtractionProvider:
    name = "deterministic"
    version = "deterministic-extraction-0.3"
    extraction_method = "deterministic"

    def extract(self, context: ExtractionContext) -> dict[str, Any]:
        text = _clean(context.text)
        full_text = _clean(" ".join([context.title, text]))
        return {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "methods": self._methods(full_text, text),
            "benchmarks": self._benchmarks(text),
            "datasets": self._datasets(text),
            "claims": self._claims(context.source_type, text),
        }

    def _methods(self, full_text: str, source_text: str) -> list[dict[str, Any]]:
        lowered = full_text.lower()
        found: list[dict[str, Any]] = []
        for pattern, method in METHOD_PATTERNS.items():
            aliases = [pattern] + [alias.lower() for alias in method["aliases"]]
            span = _first_matching_span(source_text, aliases) or _first_matching_span(full_text, aliases)
            if span and any(alias in lowered for alias in aliases):
                found.append(
                    {
                        "name": method["name"],
                        "aliases": method["aliases"],
                        "description": method["description"],
                        "category": method["category"],
                        "source_span": span,
                        "confidence": 0.84,
                    }
                )
        if not found and {"rag", "retrieval"} & set(re.findall(r"[a-z0-9]+", lowered)):
            method = METHOD_PATTERNS["hybrid retrieval"]
            span = _sentence_with_any(source_text, ["retrieval", "rag"]) or first_sentence(source_text)
            if span:
                found.append({**method, "source_span": span, "confidence": 0.68})
        return _dedupe_by_name(found)

    def _benchmarks(self, text: str) -> list[dict[str, Any]]:
        lowered = text.lower()
        records: list[dict[str, Any]] = []
        for keyword, (name, task, metric) in BENCHMARK_KEYWORDS.items():
            if keyword in lowered:
                records.append(
                    {
                        "name": name,
                        "task": task,
                        "metric": metric,
                        "source_span": _sentence_with_any(text, [keyword]) or keyword,
                        "confidence": 0.76,
                    }
                )
        return _dedupe_by_name(records)

    def _datasets(self, text: str) -> list[dict[str, Any]]:
        lowered = text.lower()
        records: list[dict[str, Any]] = []
        for keyword, (name, domain) in DATASET_KEYWORDS.items():
            if keyword in lowered:
                records.append(
                    {
                        "name": name,
                        "domain": domain,
                        "source_span": _sentence_with_any(text, [keyword]) or keyword,
                        "confidence": 0.74,
                    }
                )
        return _dedupe_by_name(records)

    def _claims(self, source_type: str, text: str) -> list[dict[str, Any]]:
        sentences = sentence_split(text)
        candidates = [
            sentence
            for sentence in sentences
            if any(
                keyword in sentence.lower()
                for keyword in [
                    "benchmark",
                    "dataset",
                    "evaluation",
                    "improve",
                    "outperform",
                    "risk",
                    "limitation",
                    "challenge",
                    "production",
                    "evidence",
                    "retrieval",
                    "memory",
                ]
            )
        ]
        if not candidates and text:
            candidates = [first_sentence(text)]
        claims: list[dict[str, Any]] = []
        for sentence in candidates[:3]:
            if not sentence:
                continue
            claims.append(
                {
                    "text": sentence,
                    "claim_type": _claim_type(sentence, source_type),
                    "confidence": _claim_confidence(source_type),
                    "polarity": _claim_polarity(sentence),
                    "source_span": sentence,
                }
            )
        return claims


class LLMStructuredExtractionProvider:
    """Provider-neutral adapter for LLMs that can return JSON structured output."""

    def __init__(
        self,
        generate_json: Callable[[str, dict[str, Any], ExtractionContext], str | dict[str, Any]],
        *,
        name: str = "llm",
        model: str = "provider-configured",
    ):
        self._generate_json = generate_json
        self.name = name
        self.version = f"{name}:{model}:{EXTRACTION_SCHEMA_VERSION}"
        self.extraction_method = "llm"

    def extract(self, context: ExtractionContext) -> dict[str, Any]:
        payload = self._generate_json(build_extraction_prompt(context), extraction_schema(), context)
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"LLM provider returned unsupported payload type: {type(payload).__name__}")


def extraction_schema() -> dict[str, Any]:
    item_fields = {
        "methods": ["name", "aliases", "description", "category", "source_span", "confidence"],
        "benchmarks": ["name", "task", "metric", "source_span", "confidence"],
        "datasets": ["name", "domain", "license", "source_span", "confidence"],
        "claims": ["text", "claim_type", "confidence", "polarity", "source_span"],
    }
    return {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "required_top_level_keys": list(item_fields),
        "required_item_fields": item_fields,
        "source_span_rule": "Every extracted item must include a non-empty source_span copied exactly from the input text.",
    }


def build_extraction_prompt(context: ExtractionContext) -> str:
    return (
        "Extract methods, benchmarks, datasets, and first-class claims from the supplied source text. "
        "Return JSON matching the provided schema. Only include items grounded in exact source_span strings. "
        f"source_type={context.source_type}; source_id={context.source_id}; title={context.title}\n\n{context.text}"
    )


def default_structured_extraction_provider() -> StructuredExtractionProvider:
    configured = os.environ.get("SIGNALGRAPH_EXTRACTION_PROVIDER", "deterministic").strip().lower()
    if configured in {"", "deterministic", "local", "fallback"}:
        return DeterministicStructuredExtractionProvider()
    raise RuntimeError(
        "Only the deterministic extraction provider is auto-configured locally. "
        "Pass an LLMStructuredExtractionProvider with a provider-specific JSON callable for live LLM extraction."
    )


def _claim_confidence(source_type: str) -> float:
    return {
        "Paper": 0.8,
        "Repo": 0.72,
        "RepoDocument": 0.72,
        "Release": 0.68,
        "Issue": 0.58,
        "Model": 0.7,
        "Dataset": 0.72,
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
    if any(word in lowered for word in ["risk", "limitation", "challenge", "fails", "stale", "broken"]):
        return "caution"
    if any(word in lowered for word in ["improve", "outperform", "strong", "effective", "production", "useful"]):
        return "positive"
    return "neutral"


def _first_matching_span(text: str, needles: list[str]) -> str:
    lowered = text.lower()
    for needle in needles:
        index = lowered.find(needle.lower())
        if index >= 0:
            return text[index : index + len(needle)]
    return ""


def _sentence_with_any(text: str, needles: list[str]) -> str:
    lowered_needles = [needle.lower() for needle in needles]
    for sentence in sentence_split(text):
        lowered = sentence.lower()
        if any(needle in lowered for needle in lowered_needles):
            return sentence
    return ""


def _dedupe_by_name(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("name") or row.get("text") or "").lower()
        if key and key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
