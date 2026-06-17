from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ProjectPaths, SOURCE_TERMS
from .models import NormalizedCorpus, SourceRecord
from .sources import source_policy_metadata
from .utils import append_jsonl, read_json, read_jsonl, stable_hash, utc_now, write_json


class RawStorage:
    """Append-only raw payload archive with source metadata manifest."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.paths.ensure()

    def store(
        self,
        *,
        source_name: str,
        source_url: str,
        source_id: str,
        request_params: dict[str, Any],
        raw_payload: Any,
        fetched_at: str | None = None,
        license_or_terms_note: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        fetched = fetched_at or utc_now()
        response_hash = stable_hash(raw_payload)
        day = fetched[:10]
        payload_path = self.paths.raw_dir / source_name / day / f"{response_hash}.json"
        cache_status = "hit" if payload_path.exists() else "miss"
        write_json(payload_path, raw_payload)
        metadata = source_policy_metadata(source_name)
        metadata.update(source_metadata or {})
        terms_note = license_or_terms_note or metadata.get("license_or_terms_note") or SOURCE_TERMS.get(source_name, "")
        quality_status, quality_reasons = evaluate_source_quality(
            source_name=source_name,
            source_url=source_url,
            source_id=source_id,
            raw_payload=raw_payload,
            license_or_terms_note=terms_note,
        )
        record = SourceRecord(
            id=f"source:{source_name}:{response_hash[:16]}",
            source_name=source_name,
            source_url=source_url,
            source_id=source_id,
            fetched_at=fetched,
            request_params=request_params,
            response_hash=response_hash,
            raw_payload_path=str(payload_path.relative_to(self.paths.root)),
            license_or_terms_note=terms_note,
            freshness_policy=metadata.get("freshness_policy", ""),
            stale_after_days=int(metadata.get("stale_after_days") or 0),
            rate_limit_note=metadata.get("rate_limit_note", ""),
            cache_policy=metadata.get("cache_policy", ""),
            cache_key=stable_hash([source_name, source_id, request_params]),
            cache_status=cache_status,
            quality_gate_status=quality_status,
            quality_gate_reasons=quality_reasons,
        )
        append_jsonl(self.paths.raw_manifest_path, asdict(record))
        return record

    def records(self) -> list[SourceRecord]:
        return [SourceRecord(**row) for row in read_jsonl(self.paths.raw_manifest_path)]

    def read_payload(self, record: SourceRecord) -> Any:
        path = Path(record.raw_payload_path)
        if not path.is_absolute():
            path = self.paths.root / path
        return read_json(path)


class NormalizedStore:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.paths.ensure()

    def load(self) -> NormalizedCorpus:
        if not self.paths.normalized_corpus_path.exists():
            return NormalizedCorpus()
        return NormalizedCorpus.from_dict(read_json(self.paths.normalized_corpus_path))

    def save(self, corpus: NormalizedCorpus) -> None:
        write_json(self.paths.normalized_corpus_path, corpus.to_dict())

    def merge_and_save(self, corpus: NormalizedCorpus) -> NormalizedCorpus:
        merged = self.load().merge(corpus)
        self.save(merged)
        return merged


def evaluate_source_quality(
    *,
    source_name: str,
    source_url: str,
    source_id: str,
    raw_payload: Any,
    license_or_terms_note: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not source_url:
        reasons.append("missing_source_url")
    if not source_id:
        reasons.append("missing_stable_source_id")
    if not license_or_terms_note:
        reasons.append("missing_terms_note")
    if raw_payload in ({}, [], None):
        reasons.append("empty_payload")
    if source_name == "github" and isinstance(raw_payload, dict) and not raw_payload.get("full_name"):
        reasons.append("missing_github_full_name")
    if source_name == "huggingface" and isinstance(raw_payload, dict) and not (raw_payload.get("modelId") or raw_payload.get("id") or raw_payload.get("_id")):
        reasons.append("missing_huggingface_asset_id")
    if source_name == "semantic_scholar" and isinstance(raw_payload, dict) and not raw_payload.get("paperId"):
        reasons.append("missing_semantic_scholar_paper_id")
    return ("quarantine" if reasons else "pass", reasons)
