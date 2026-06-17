from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

from .utils import tokenize


DEFAULT_EMBEDDING_DIMENSIONS = 64
DEFAULT_EMBEDDING_MODEL = "deterministic-hash-64"


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    kind: str
    text: str
    property_name: str = "embedding"


class DeterministicEmbeddingProvider:
    """Credential-free embedding fallback with stable dimensions and values."""

    name = "deterministic"

    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions
        self.model = f"deterministic-hash-{dimensions}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            tokens = ["<empty>"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            magnitude = 1.0 + (digest[5] / 255.0)
            vector[index] += sign * magnitude
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 8) for value in vector]


class OpenAIEmbeddingProvider:
    """Lazy OpenAI adapter; constructed only when credentials are intentionally present."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]


def provider_from_env() -> EmbeddingProvider:
    provider_name = os.environ.get("SIGNALGRAPH_EMBEDDING_PROVIDER", "deterministic").strip().lower()
    if provider_name == "openai" and os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("SIGNALGRAPH_EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = int(os.environ.get("SIGNALGRAPH_EMBEDDING_DIMENSIONS", "1536"))
        return OpenAIEmbeddingProvider(model=model, dimensions=dimensions)
    dimensions = int(os.environ.get("SIGNALGRAPH_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)))
    return DeterministicEmbeddingProvider(dimensions=dimensions)


def embedding_text_for_kind(kind: str, value: Any) -> str:
    props = value if isinstance(value, dict) else getattr(value, "properties", {})
    if kind == "DocumentChunk":
        return str(props.get("text", ""))
    if kind == "Claim":
        return " ".join(str(props.get(key, "")) for key in ["text", "claim_type", "polarity", "source_span"])
    if kind == "Method":
        aliases = props.get("aliases", [])
        aliases_text = " ".join(aliases if isinstance(aliases, list) else [])
        return " ".join(str(props.get(key, "")) for key in ["name", "description", "category"]) + " " + aliases_text
    if kind == "Repo":
        topics = props.get("topics", [])
        topics_text = " ".join(topics if isinstance(topics, list) else [])
        return " ".join(str(props.get(key, "")) for key in ["full_name", "description", "license"]) + " " + topics_text
    if kind == "Community":
        return " ".join(str(props.get(key, "")) for key in ["name", "summary", "report"])
    return " ".join(str(value) for value in props.values() if value)


def embedding_records_for_nodes(nodes: list[Any]) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for node in nodes:
        labels = set(getattr(node, "labels", []))
        props = getattr(node, "properties", {})
        for kind in ["DocumentChunk", "Claim", "Method", "Repo", "Community"]:
            if kind not in labels:
                continue
            property_name = "readme_embedding" if kind == "Repo" else "embedding"
            records.append(EmbeddingRecord(node.id, kind, embedding_text_for_kind(kind, props), property_name))
            break
    return records


def apply_embeddings_to_nodes(nodes: list[Any], provider: EmbeddingProvider | None = None) -> None:
    selected_provider = provider or provider_from_env()
    records = embedding_records_for_nodes(nodes)
    if not records:
        return
    vectors = selected_provider.embed_texts([record.text for record in records])
    by_id = {getattr(node, "id"): node for node in nodes}
    for record, vector in zip(records, vectors):
        node = by_id[record.id]
        node.properties[record.property_name] = vector
        node.properties["embedding_provider"] = selected_provider.name
        node.properties["embedding_model"] = selected_provider.model
        node.properties["embedding_dimensions"] = selected_provider.dimensions
