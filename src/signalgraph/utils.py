from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-+.]*")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    return stable_hash(value)[:length]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def slugify(text: str, fallback: str = "artifact") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or fallback


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def token_set(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1}


def lexical_overlap(query: str, text: str) -> float:
    q = token_set(query)
    t = token_set(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def cosine_bow(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    q_counts: dict[str, int] = {}
    t_counts: dict[str, int] = {}
    for token in q_tokens:
        q_counts[token] = q_counts.get(token, 0) + 1
    for token in t_tokens:
        t_counts[token] = t_counts.get(token, 0) + 1
    dot = sum(q_counts[token] * t_counts.get(token, 0) for token in q_counts)
    q_norm = math.sqrt(sum(count * count for count in q_counts.values()))
    t_norm = math.sqrt(sum(count * count for count in t_counts.values()))
    if q_norm == 0 or t_norm == 0:
        return 0.0
    return dot / (q_norm * t_norm)


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def first_sentence(text: str, max_chars: int = 280) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return (text or "")[:max_chars].strip()
    return sentences[0][:max_chars].strip()


def compact_text(text: str, max_chars: int = 800) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_yearish(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None

