from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import (
    BenchmarkRecord,
    ClaimRecord,
    DatasetRecord,
    ExtractionQuarantineRecord,
    MethodRecord,
    SourceRecord,
)
from .providers import (
    DeterministicStructuredExtractionProvider,
    ExtractionContext,
    StructuredExtractionProvider,
    default_structured_extraction_provider,
)
from .utils import short_hash, utc_now


@dataclass
class ExtractionBundle:
    methods: list[MethodRecord]
    benchmarks: list[BenchmarkRecord]
    datasets: list[DatasetRecord]
    claims: list[ClaimRecord]
    quarantine: list[ExtractionQuarantineRecord]

    @classmethod
    def empty(cls) -> "ExtractionBundle":
        return cls([], [], [], [], [])

    def extend(self, other: "ExtractionBundle") -> None:
        self.methods.extend(other.methods)
        self.benchmarks.extend(other.benchmarks)
        self.datasets.extend(other.datasets)
        self.claims.extend(other.claims)
        self.quarantine.extend(other.quarantine)


def extract_traceable_records(
    *,
    source_type: str,
    source_id: str,
    title: str,
    text: str,
    source_url: str,
    source_record: SourceRecord,
    provider: StructuredExtractionProvider | None = None,
) -> ExtractionBundle:
    context = ExtractionContext(
        source_type=source_type,
        source_id=source_id,
        title=title,
        text=_clean(text),
        source_url=source_url,
        source_record_id=source_record.id,
    )
    selected = provider or default_structured_extraction_provider()
    bundle = _extract_with_provider(context, source_record, selected)
    if _needs_deterministic_fallback(bundle, selected):
        fallback = _extract_with_provider(context, source_record, DeterministicStructuredExtractionProvider())
        bundle.extend(fallback)
    return bundle


def _extract_with_provider(
    context: ExtractionContext,
    source_record: SourceRecord,
    provider: StructuredExtractionProvider,
) -> ExtractionBundle:
    try:
        payload = provider.extract(context)
    except Exception as exc:  # provider adapters should never break local normalization
        return ExtractionBundle(
            [],
            [],
            [],
            [],
            [
                _quarantine(
                    context,
                    source_record,
                    provider,
                    attempted_record_type="provider_payload",
                    reason=f"provider_error:{type(exc).__name__}",
                    payload={"error": str(exc)},
                )
            ],
        )
    return validate_extraction_payload(payload, context, source_record, provider)


def validate_extraction_payload(
    payload: Any,
    context: ExtractionContext,
    source_record: SourceRecord,
    provider: StructuredExtractionProvider,
) -> ExtractionBundle:
    bundle = ExtractionBundle.empty()
    if not isinstance(payload, dict):
        bundle.quarantine.append(
            _quarantine(context, source_record, provider, "provider_payload", "schema:top_level_not_object", {"payload": repr(payload)})
        )
        return bundle
    for key in ["methods", "benchmarks", "datasets", "claims"]:
        rows = payload.get(key, [])
        if rows in (None, ""):
            continue
        if not isinstance(rows, list):
            bundle.quarantine.append(_quarantine(context, source_record, provider, key, "schema:list_required", rows))
            continue
        for row in rows:
            _append_validated_row(bundle, row, key[:-1], context, source_record, provider)
    return bundle


def _append_validated_row(
    bundle: ExtractionBundle,
    row: Any,
    record_type: str,
    context: ExtractionContext,
    source_record: SourceRecord,
    provider: StructuredExtractionProvider,
) -> None:
    if not isinstance(row, dict):
        bundle.quarantine.append(_quarantine(context, source_record, provider, record_type, "schema:item_not_object", {"payload": repr(row)}))
        return
    reason = _invalid_reason(row, record_type, context.text)
    if reason:
        bundle.quarantine.append(_quarantine(context, source_record, provider, record_type, reason, row))
        return
    created_at = utc_now()
    confidence = _confidence(row)
    source_span = _clean(str(row.get("source_span", "")))
    if record_type == "method":
        name = _clean(str(row["name"]))
        bundle.methods.append(
            MethodRecord(
                id=f"method:{_safe_id(name)}",
                name=name,
                aliases=[str(alias) for alias in row.get("aliases", []) if str(alias).strip()],
                description=_clean(str(row.get("description", ""))),
                category=_clean(str(row.get("category", ""))),
                source_url=context.source_url,
                source_record_id=source_record.id,
                source_span=source_span,
                extracted_at=created_at,
                extractor_version=provider.version,
                extraction_method=provider.extraction_method,
                confidence=confidence,
            )
        )
    elif record_type == "benchmark":
        name = _clean(str(row["name"]))
        bundle.benchmarks.append(
            BenchmarkRecord(
                id=f"benchmark:{_safe_id(name)}",
                name=name,
                task=_clean(str(row.get("task", ""))),
                metric=_clean(str(row.get("metric", ""))),
                source_url=context.source_url,
                source_record_id=source_record.id,
                source_span=source_span,
                extracted_at=created_at,
                extractor_version=provider.version,
                extraction_method=provider.extraction_method,
                confidence=confidence,
            )
        )
    elif record_type == "dataset":
        name = _clean(str(row["name"]))
        bundle.datasets.append(
            DatasetRecord(
                id=f"dataset:{_safe_id(name)}",
                name=name,
                domain=_clean(str(row.get("domain", ""))),
                license=_clean(str(row.get("license", ""))),
                source_url=context.source_url,
                source_record_id=source_record.id,
                source_span=source_span,
                extracted_at=created_at,
                extractor_version=provider.version,
                extraction_method=provider.extraction_method,
                confidence=confidence,
            )
        )
    elif record_type == "claim":
        text = _clean(str(row["text"]))
        bundle.claims.append(
            ClaimRecord(
                id=f"claim:{short_hash([context.source_id, text, source_span])}",
                text=text,
                claim_type=_clean(str(row["claim_type"])),
                confidence=confidence,
                polarity=_clean(str(row.get("polarity", "neutral"))),
                source_span=source_span,
                source_url=context.source_url,
                extracted_at=created_at,
                extractor_version=provider.version,
                source_type=context.source_type,
                source_id=context.source_id,
                source_record_id=source_record.id,
                extraction_method=provider.extraction_method,
            )
        )


def _invalid_reason(row: dict[str, Any], record_type: str, source_text: str) -> str:
    required = {
        "method": ["name", "source_span"],
        "benchmark": ["name", "source_span"],
        "dataset": ["name", "source_span"],
        "claim": ["text", "claim_type", "confidence", "polarity", "source_span"],
    }[record_type]
    missing = [field for field in required if row.get(field) in (None, "", [])]
    if missing:
        return "schema:missing_" + ",".join(missing)
    span = _clean(str(row.get("source_span", "")))
    if not span or span not in source_text:
        return "untraceable_source_span"
    confidence = row.get("confidence", 0.0)
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "schema:confidence_not_number"
    if value < 0.0 or value > 1.0:
        return "schema:confidence_out_of_range"
    if record_type == "claim" and _clean(str(row.get("text", ""))) not in source_text:
        return "untraceable_claim_text"
    return ""


def _needs_deterministic_fallback(bundle: ExtractionBundle, provider: StructuredExtractionProvider) -> bool:
    if provider.extraction_method == "deterministic":
        return False
    valid_count = len(bundle.methods) + len(bundle.benchmarks) + len(bundle.datasets) + len(bundle.claims)
    return valid_count == 0


def _quarantine(
    context: ExtractionContext,
    source_record: SourceRecord,
    provider: StructuredExtractionProvider,
    attempted_record_type: str,
    reason: str,
    payload: Any,
) -> ExtractionQuarantineRecord:
    return ExtractionQuarantineRecord(
        id=f"extraction_quarantine:{short_hash([source_record.id, attempted_record_type, reason, payload])}",
        source_type=context.source_type,
        source_id=context.source_id,
        source_record_id=source_record.id,
        source_url=context.source_url,
        attempted_record_type=attempted_record_type,
        reason=reason,
        payload=payload if isinstance(payload, dict) else {"payload": repr(payload)},
        source_span=str(payload.get("source_span", "")) if isinstance(payload, dict) else "",
        extractor_version=provider.version,
        extraction_method=provider.extraction_method,
        created_at=utc_now(),
    )


def _confidence(row: dict[str, Any]) -> float:
    return round(max(0.0, min(1.0, float(row.get("confidence", 0.65)))), 3)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).lower()).strip("-")
    return cleaned or short_hash(value)
