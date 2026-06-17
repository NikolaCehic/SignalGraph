from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .models import EntityResolutionDecision, NormalizedCorpus
from .utils import short_hash, utc_now


EXACT_MATCH = "exact"
PROBABLE_MATCH = "probable"
POSSIBLE_DUPLICATE = "possible_duplicate"


@dataclass(frozen=True)
class ResolutionEntity:
    id: str
    label: str
    name: str
    source_url: str
    stable_ids: dict[str, str]
    weak_signals: dict[str, str]


def resolve_entities(corpus: NormalizedCorpus) -> list[EntityResolutionDecision]:
    entities = _entities(corpus)
    decisions: list[EntityResolutionDecision] = []
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if left.label != right.label:
                continue
            decision = _resolve_pair(left, right)
            if decision:
                decisions.append(decision)
    return decisions


def possible_duplicate_edges(decisions: list[EntityResolutionDecision]) -> list[EntityResolutionDecision]:
    return [decision for decision in decisions if decision.state == POSSIBLE_DUPLICATE]


def _resolve_pair(left: ResolutionEntity, right: ResolutionEntity) -> EntityResolutionDecision | None:
    exact_signals = _shared_stable_signals(left, right)
    if exact_signals:
        return _decision(left, right, EXACT_MATCH, 1.0, exact_signals, review_required=False)
    if _canonical_url(left.source_url) and _canonical_url(left.source_url) == _canonical_url(right.source_url):
        return _decision(left, right, EXACT_MATCH, 0.98, ["canonical_url"], review_required=False)

    left_name = _normalize_name(left.name)
    right_name = _normalize_name(right.name)
    if not left_name or not right_name:
        return None
    if left_name == right_name:
        return _decision(left, right, PROBABLE_MATCH, 0.92, ["normalized_name"], review_required=False)

    similarity = SequenceMatcher(None, left_name, right_name).ratio()
    weak_overlap = _weak_signal_overlap(left, right)
    if similarity >= 0.9 and weak_overlap:
        return _decision(left, right, PROBABLE_MATCH, round(similarity, 3), ["fuzzy_name", *weak_overlap], review_required=False)
    if similarity >= 0.76 or _token_overlap(left_name, right_name) >= 0.72:
        score = round(max(similarity, _token_overlap(left_name, right_name)), 3)
        return _decision(left, right, POSSIBLE_DUPLICATE, score, ["fuzzy_name_below_merge_threshold", *weak_overlap], review_required=True)
    return None


def _decision(
    left: ResolutionEntity,
    right: ResolutionEntity,
    state: str,
    score: float,
    signals: list[str],
    *,
    review_required: bool,
) -> EntityResolutionDecision:
    canonical_id = min(left.id, right.id)
    canonical_url = _canonical_url(left.source_url) or _canonical_url(right.source_url)
    return EntityResolutionDecision(
        id=f"entity_resolution:{short_hash([left.id, right.id, state, signals])}",
        state=state,
        left_id=left.id,
        right_id=right.id,
        entity_label=left.label,
        score=score,
        signals=signals,
        canonical_id=canonical_id,
        canonical_url=canonical_url,
        review_required=review_required,
        created_at=utc_now(),
    )


def _entities(corpus: NormalizedCorpus) -> list[ResolutionEntity]:
    entities: list[ResolutionEntity] = []
    for paper in corpus.papers:
        entities.append(
            ResolutionEntity(
                id=paper.id,
                label="Paper",
                name=paper.title,
                source_url=paper.source_url,
                stable_ids={
                    "doi": paper.doi,
                    "arxiv_id": paper.arxiv_id,
                    "openalex_id": paper.openalex_id,
                    "semantic_scholar_id": paper.semantic_scholar_id,
                },
                weak_signals={"published_year": _year(paper.published_at), "venue": paper.venue},
            )
        )
    for repo in corpus.repos:
        entities.append(
            ResolutionEntity(
                id=repo.id,
                label="Repo",
                name=repo.full_name or repo.name,
                source_url=repo.url,
                stable_ids={"full_name": repo.full_name.lower()},
                weak_signals={"owner": repo.owner.lower(), "default_branch": repo.default_branch},
            )
        )
    for method in corpus.methods:
        entities.append(
            ResolutionEntity(
                id=method.id,
                label="Method",
                name=method.name,
                source_url=method.source_url,
                stable_ids={},
                weak_signals={"aliases": " ".join(method.aliases), "category": method.category},
            )
        )
    for benchmark in corpus.benchmarks:
        entities.append(
            ResolutionEntity(
                id=benchmark.id,
                label="Benchmark",
                name=benchmark.name,
                source_url=benchmark.source_url,
                stable_ids={},
                weak_signals={"task": benchmark.task, "metric": benchmark.metric},
            )
        )
    for dataset in corpus.datasets:
        entities.append(
            ResolutionEntity(
                id=dataset.id,
                label="Dataset",
                name=dataset.name,
                source_url=dataset.source_url,
                stable_ids={"huggingface_id": dataset.huggingface_id},
                weak_signals={"domain": dataset.domain, "license": dataset.license},
            )
        )
    return entities


def _shared_stable_signals(left: ResolutionEntity, right: ResolutionEntity) -> list[str]:
    signals: list[str] = []
    for key, left_value in left.stable_ids.items():
        right_value = right.stable_ids.get(key, "")
        if left_value and right_value and _normalize_identifier(left_value) == _normalize_identifier(right_value):
            signals.append(key)
    return signals


def _weak_signal_overlap(left: ResolutionEntity, right: ResolutionEntity) -> list[str]:
    signals: list[str] = []
    for key, left_value in left.weak_signals.items():
        right_value = right.weak_signals.get(key, "")
        if left_value and right_value and _normalize_name(left_value) == _normalize_name(right_value):
            signals.append(key)
    return signals


def _canonical_url(url: str) -> str:
    value = (url or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.rstrip("/")
    value = value.replace("www.", "")
    return value


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9./:-]+", "", str(value).lower())


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if token}
    right_tokens = {token for token in right.split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _year(value: str) -> str:
    match = re.search(r"(19|20)\d{2}", value or "")
    return match.group(0) if match else ""
