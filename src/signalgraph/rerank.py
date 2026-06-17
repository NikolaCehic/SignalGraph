from __future__ import annotations

from dataclasses import dataclass, asdict

from .quality import score_source_quality
from typing import Protocol

from .utils import parse_yearish, token_set


SOURCE_QUALITY = {
    "arxiv": 0.88,
    "openalex": 0.9,
    "github": 0.82,
    "sample": 0.8,
    "paper_full_text": 0.9,
    "official_repo_readme": 0.82,
    "repo_issue": 0.55,
    "blog_post": 0.45,
    "llm_inferred": 0.2,
}


@dataclass
class RerankFeatures:
    semantic_relevance: float
    lexical_relevance: float
    graph_path_quality: float
    source_quality: float
    freshness: float
    confidence: float
    evidence_strength: float
    graph_path_score: float = 0.0
    rrf_score: float = 0.0
    mmr_score: float = 0.0
    diversity_score: float = 0.0

    def combined_score(self) -> float:
        weights = {
            "semantic_relevance": 0.2,
            "lexical_relevance": 0.15,
            "graph_path_quality": 0.15,
            "source_quality": 0.1,
            "freshness": 0.07,
            "confidence": 0.08,
            "evidence_strength": 0.09,
            "graph_path_score": 0.08,
            "rrf_score": 0.05,
            "diversity_score": 0.03,
        }
        total = 0.0
        for name, weight in weights.items():
            total += max(0.0, min(1.0, getattr(self, name))) * weight
        return round(total, 4)

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["combined_score"] = self.combined_score()
        return values


def source_quality(
    source_name: str,
    extraction_method: str = "",
    record_source_type: str = "",
    section: str = "",
    source_url: str = "",
    has_source_span: bool = True,
    claim_type: str = "",
) -> float:
    return score_source_quality(
        source_name=source_name,
        extraction_method=extraction_method,
        record_source_type=record_source_type,
        section=section,
        source_url=source_url,
        has_source_span=has_source_span,
        claim_type=claim_type,
    ).score


def freshness_score(date_value: str | None, current_year: int = 2026) -> float:
    year = parse_yearish(date_value)
    if year is None:
        return 0.45
    age = max(0, current_year - year)
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.82
    if age <= 6:
        return 0.6
    return 0.35


def graph_path_quality(path_length: int, typed_relationships: int, has_claim: bool, has_source_span: bool) -> float:
    if path_length <= 0:
        return 0.0
    length_factor = 1.0 if path_length <= 4 else max(0.35, 1.0 - ((path_length - 4) * 0.1))
    type_factor = min(1.0, typed_relationships / max(1, path_length - 1))
    evidence_bonus = 0.15 if has_claim else 0.0
    span_bonus = 0.15 if has_source_span else 0.0
    return round(min(1.0, (0.7 * length_factor * type_factor) + evidence_bonus + span_bonus), 4)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, candidate_id in enumerate(ranked, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + (1.0 / (k + rank))
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {candidate_id: round(score / max_score, 6) for candidate_id, score in scores.items()}


class DiversityCandidate(Protocol):
    id: str
    text: str
    score: float


def maximal_marginal_relevance(candidates: list[DiversityCandidate], limit: int, lambda_mult: float = 0.72) -> list[DiversityCandidate]:
    remaining = list(candidates)
    selected: list[DiversityCandidate] = []
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda candidate: (
                _mmr_value(candidate, selected, lambda_mult),
                candidate.score,
                candidate.id,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def diversity_against_selected(text: str, selected_texts: list[str]) -> float:
    if not selected_texts:
        return 1.0
    return round(1.0 - max(_token_jaccard(text, selected) for selected in selected_texts), 4)


def _mmr_value(candidate: DiversityCandidate, selected: list[DiversityCandidate], lambda_mult: float) -> float:
    diversity = diversity_against_selected(candidate.text, [item.text for item in selected])
    return round((lambda_mult * candidate.score) + ((1.0 - lambda_mult) * diversity), 6)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
