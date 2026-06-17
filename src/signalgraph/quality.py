from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SourceQualityScore:
    score: float
    source_type: str
    evidence_tier: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_source_quality(
    *,
    source_name: str = "",
    extraction_method: str = "",
    record_source_type: str = "",
    section: str = "",
    source_url: str = "",
    has_source_span: bool = True,
    claim_type: str = "",
) -> SourceQualityScore:
    source_type = classify_source_type(
        source_name=source_name,
        extraction_method=extraction_method,
        record_source_type=record_source_type,
        section=section,
        source_url=source_url,
        has_source_span=has_source_span,
    )
    base_scores = {
        "api_metadata": 0.9,
        "paper_full_text": 0.88,
        "official_repo_readme": 0.82,
        "official_repo_docs": 0.78,
        "repo_release": 0.7,
        "repo_issue": 0.55,
        "model_or_dataset_card": 0.74,
        "blog_post": 0.45,
        "llm_inferred": 0.2,
        "sample_fixture": 0.8,
        "unknown": 0.5,
    }
    score = base_scores.get(source_type, 0.5)
    reasons = [f"source_type:{source_type}"]
    if extraction_method == "llm" and source_type != "llm_inferred":
        score -= 0.08
        reasons.append("llm_extracted_but_span_backed")
    if not has_source_span and extraction_method != "api":
        score = min(score, 0.25)
        reasons.append("missing_source_span")
    if claim_type in {"repo_risk", "limitation"} and source_type == "repo_issue":
        score += 0.08
        reasons.append("issue_is_relevant_for_risk_claims")
    if source_name in {"openalex", "arxiv", "semantic_scholar"} and source_type == "api_metadata":
        reasons.append("deterministic_public_metadata")
    score = round(max(0.0, min(1.0, score)), 3)
    return SourceQualityScore(score=score, source_type=source_type, evidence_tier=_tier(score), reasons=reasons)


def score_from_properties(properties: dict[str, Any], labels: list[str] | None = None) -> SourceQualityScore:
    labels = labels or []
    return score_source_quality(
        source_name=str(properties.get("source_name", "")),
        extraction_method=str(properties.get("extraction_method", "")),
        record_source_type=str(properties.get("source_type", "") or _label_source_type(labels)),
        section=str(properties.get("section", "")),
        source_url=str(properties.get("source_url") or properties.get("url") or ""),
        has_source_span=bool(properties.get("source_span") or properties.get("text") or properties.get("abstract")),
        claim_type=str(properties.get("claim_type", "")),
    )


def classify_source_type(
    *,
    source_name: str = "",
    extraction_method: str = "",
    record_source_type: str = "",
    section: str = "",
    source_url: str = "",
    has_source_span: bool = True,
) -> str:
    lowered_section = section.lower()
    lowered_url = source_url.lower()
    if extraction_method == "llm" and not has_source_span:
        return "llm_inferred"
    if source_name == "sample":
        return "sample_fixture"
    if any(host in lowered_url for host in ["medium.com", "substack.com", "dev.to"]) or "blog" in lowered_url:
        return "blog_post"
    if record_source_type == "Issue" or lowered_section == "issue":
        return "repo_issue"
    if record_source_type == "Release" or "release" in lowered_section:
        return "repo_release"
    if record_source_type == "RepoDocument" and lowered_section == "readme":
        return "official_repo_readme"
    if record_source_type == "RepoDocument" or lowered_section in {"docs", "changelog"}:
        return "official_repo_docs"
    if record_source_type in {"Model", "Dataset"} or "huggingface" in source_name:
        return "model_or_dataset_card"
    if record_source_type == "Paper" and lowered_section in {"abstract", "paper_full_text", ""}:
        return "paper_full_text"
    if extraction_method == "api":
        return "api_metadata"
    return "unknown"


def _label_source_type(labels: list[str]) -> str:
    for label in ["Claim", "DocumentChunk", "Paper", "Repo", "RepoDocument", "Issue", "Release", "Model", "Dataset"]:
        if label in labels:
            return label
    return labels[0] if labels else ""


def _tier(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.62:
        return "medium"
    if score >= 0.4:
        return "low"
    return "weak"
