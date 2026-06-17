from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .utils import compact_text


@dataclass(frozen=True)
class SourcePolicy:
    license_or_terms_note: str
    freshness_policy: str
    stale_after_days: int
    rate_limit_note: str
    cache_policy: str


SOURCE_POLICIES: dict[str, SourcePolicy] = {
    "arxiv": SourcePolicy(
        "arXiv public API metadata; acknowledge arXiv when presenting public results.",
        "paper metadata refresh daily for active topics and weekly for longer-tail topics",
        7,
        "Use a descriptive User-Agent and keep requests modest for the public arXiv API.",
        "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
    ),
    "openalex": SourcePolicy(
        "OpenAlex public API metadata under OpenAlex terms.",
        "paper and institution metadata refresh weekly; citation expansion refresh weekly to monthly",
        14,
        "Optional OPENALEX_MAILTO is sent when configured; bounded per-page requests avoid large pulls.",
        "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
    ),
    "semantic_scholar": SourcePolicy(
        "Semantic Scholar Graph API metadata; optional API key improves rate limits.",
        "paper, author, reference, and citation metadata refresh weekly for active topics",
        14,
        "Optional SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY is sent as x-api-key when configured; requests are bounded.",
        "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
    ),
    "github": SourcePolicy(
        "GitHub REST API metadata and official repo content; optional token improves rate limits.",
        "repo metadata refresh daily for top repos; README/docs/releases/issues refresh daily for selected repos",
        3,
        "Optional GITHUB_TOKEN is sent as a bearer token; README/docs/releases/issues collection is bounded per repo.",
        "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
    ),
    "huggingface": SourcePolicy(
        "Hugging Face Hub public model and dataset metadata/cards; optional token improves rate limits.",
        "model and dataset cards refresh weekly unless linked assets change",
        7,
        "Optional HUGGINGFACE_TOKEN or HF_TOKEN is sent as a bearer token; model/dataset searches are bounded.",
        "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
    ),
    "sample": SourcePolicy(
        "Bundled deterministic sample corpus for local tests and evaluation runs.",
        "static local fixture data",
        3650,
        "No external requests.",
        "Stored as append-only local fixture payloads.",
    ),
}


@dataclass
class RawDocument:
    source_name: str
    source_url: str
    source_id: str
    request_params: dict[str, Any]
    raw_payload: dict[str, Any]
    source_metadata: dict[str, Any] = field(default_factory=dict)


class UrlFetcher(Protocol):
    def get_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        ...

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        ...


class StandardUrlFetcher:
    def get_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        request = urllib.request.Request(url, headers=headers or {"User-Agent": "SignalGraph/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        return json.loads(self.get_text(url, headers=headers))


class ArxivClient:
    source_name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    def __init__(self, fetcher: UrlFetcher | None = None):
        self.fetcher = fetcher or StandardUrlFetcher()

    def search(self, topic: str, limit: int = 25) -> list[RawDocument]:
        params = {
            "search_query": f'all:"{topic}"',
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        xml_text = self.fetcher.get_text(url, headers={"User-Agent": "SignalGraph/0.1"})
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        documents: list[RawDocument] = []
        for entry in root.findall("atom:entry", ns):
            source_url = _xml_text(entry, "atom:id", ns)
            links = []
            for link in entry.findall("atom:link", ns):
                links.append(link.attrib)
            payload = {
                "id": source_url,
                "title": compact_text(_xml_text(entry, "atom:title", ns), 500),
                "summary": compact_text(_xml_text(entry, "atom:summary", ns), 5000),
                "published": _xml_text(entry, "atom:published", ns),
                "updated": _xml_text(entry, "atom:updated", ns),
                "authors": [_xml_text(author, "atom:name", ns) for author in entry.findall("atom:author", ns)],
                "categories": [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)],
                "doi": _xml_text(entry, "arxiv:doi", ns),
                "links": links,
            }
            source_id = source_url.rstrip("/").rsplit("/", 1)[-1]
            documents.append(raw_document(self.source_name, source_url, source_id, params, payload))
        return documents


class OpenAlexClient:
    source_name = "openalex"
    base_url = "https://api.openalex.org/works"

    def __init__(self, fetcher: UrlFetcher | None = None, mailto: str | None = None):
        self.fetcher = fetcher or StandardUrlFetcher()
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO", "")

    def search(self, topic: str, limit: int = 25) -> list[RawDocument]:
        params = {"search": topic, "per-page": min(limit, 200), "sort": "relevance_score:desc"}
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        response = self.fetcher.get_json(url, headers={"User-Agent": "SignalGraph/0.1"})
        documents = []
        for item in response.get("results", [])[:limit]:
            source_url = item.get("id") or item.get("doi") or item.get("primary_location", {}).get("landing_page_url", "")
            source_id = (item.get("id") or source_url).rstrip("/").rsplit("/", 1)[-1]
            documents.append(raw_document(self.source_name, source_url, source_id, params, item))
        return documents


class SemanticScholarClient:
    source_name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, fetcher: UrlFetcher | None = None, api_key: str | None = None):
        self.fetcher = fetcher or StandardUrlFetcher()
        self.api_key = api_key if api_key is not None else (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY", ""))

    def search(self, topic: str, limit: int = 25) -> list[RawDocument]:
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "year",
                "venue",
                "publicationDate",
                "externalIds",
                "url",
                "citationCount",
                "referenceCount",
                "influentialCitationCount",
                "authors",
                "openAccessPdf",
                "references.paperId",
                "references.title",
                "references.url",
                "citations.paperId",
                "citations.title",
                "citations.url",
            ]
        )
        params = {"query": topic, "limit": min(limit, 100), "fields": fields}
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "SignalGraph/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = self.fetcher.get_json(url, headers=headers)
        documents: list[RawDocument] = []
        for item in _items(response, "data")[:limit]:
            source_id = str(item.get("paperId") or item.get("corpusId") or item.get("url") or "")
            source_url = item.get("url") or (f"https://www.semanticscholar.org/paper/{source_id}" if source_id else "")
            documents.append(raw_document(self.source_name, source_url, source_id, params, item))
        return documents


class GitHubClient:
    source_name = "github"
    base_url = "https://api.github.com/search/repositories"
    repo_api_url = "https://api.github.com/repos"

    def __init__(self, fetcher: UrlFetcher | None = None, token: str | None = None, enrich: bool = True):
        self.fetcher = fetcher or StandardUrlFetcher()
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        self.enrich = enrich

    def search(self, topic: str, limit: int = 25) -> list[RawDocument]:
        params = {"q": f"{topic} in:name,description,readme", "sort": "stars", "order": "desc", "per_page": min(limit, 100)}
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "SignalGraph/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self.fetcher.get_json(url, headers=headers)
        documents: list[RawDocument] = []
        for item in response.get("items", [])[:limit]:
            payload = self._enrich_repository(item, headers) if self.enrich else dict(item)
            source_url = item.get("html_url", "")
            source_id = item.get("full_name", source_url)
            documents.append(raw_document(self.source_name, source_url, source_id, params, payload))
        return documents

    def _enrich_repository(self, item: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        payload = dict(item)
        full_name = payload.get("full_name", "")
        if not full_name:
            return payload
        repo_url = f"{self.repo_api_url}/{full_name}"
        readme = self._fetch_content_file(f"{repo_url}/readme", headers, "README.md", "readme")
        docs = [
            doc
            for doc in [
                self._fetch_content_file(f"{repo_url}/contents/docs/README.md", headers, "docs/README.md", "docs"),
                self._fetch_content_file(f"{repo_url}/contents/docs/index.md", headers, "docs/index.md", "docs"),
            ]
            if doc.get("text")
        ]
        changelog = {}
        for path in ["CHANGELOG.md", "CHANGELOG", "changelog.md", "docs/changelog.md"]:
            changelog = self._fetch_content_file(f"{repo_url}/contents/{urllib.parse.quote(path)}", headers, path, "changelog")
            if changelog.get("text"):
                break
        releases = self._selected_releases(repo_url, headers)
        issues = self._selected_issues(repo_url, headers)
        payload.update(
            {
                "readme": readme,
                "docs": docs,
                "changelog": changelog if changelog.get("text") else {},
                "releases": releases,
                "issues": issues,
                "risk_signals": _github_risk_signals(payload, issues, releases, bool(changelog.get("text"))),
            }
        )
        return payload

    def _selected_releases(self, repo_url: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        url = f"{repo_url}/releases?per_page=5"
        releases = []
        for release in _items(self._safe_get_json(url, headers), "items")[:5]:
            releases.append(
                {
                    "tag_name": release.get("tag_name", ""),
                    "name": release.get("name", ""),
                    "published_at": release.get("published_at", ""),
                    "html_url": release.get("html_url", ""),
                    "body": compact_text(release.get("body", ""), 2000),
                    "prerelease": bool(release.get("prerelease", False)),
                }
            )
        return releases

    def _selected_issues(self, repo_url: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        params = {"state": "open", "per_page": 10, "sort": "updated", "direction": "desc"}
        url = f"{repo_url}/issues?{urllib.parse.urlencode(params)}"
        issues = []
        for issue in _items(self._safe_get_json(url, headers), "items")[:10]:
            if issue.get("pull_request"):
                continue
            labels = [label.get("name", "") if isinstance(label, dict) else str(label) for label in issue.get("labels", [])]
            issues.append(
                {
                    "number": int(issue.get("number") or 0),
                    "title": issue.get("title", ""),
                    "state": issue.get("state", ""),
                    "labels": [label for label in labels if label],
                    "html_url": issue.get("html_url", ""),
                    "created_at": issue.get("created_at", ""),
                    "updated_at": issue.get("updated_at", ""),
                    "body": compact_text(issue.get("body", ""), 2000),
                }
            )
        return issues

    def _fetch_content_file(self, url: str, headers: dict[str, str], path: str, doc_type: str) -> dict[str, Any]:
        payload = self._safe_get_json(url, headers)
        if not isinstance(payload, dict):
            return {}
        text = _decode_github_content(payload)
        return {
            "doc_type": doc_type,
            "path": payload.get("path") or path,
            "name": payload.get("name") or path.rsplit("/", 1)[-1],
            "html_url": payload.get("html_url") or "",
            "download_url": payload.get("download_url") or "",
            "text": compact_text(text, 12000),
        }

    def _safe_get_json(self, url: str, headers: dict[str, str]) -> Any:
        try:
            return self.fetcher.get_json(url, headers=headers)
        except RuntimeError:
            return None


class HuggingFaceClient:
    source_name = "huggingface"
    base_url = "https://huggingface.co/api"

    def __init__(self, fetcher: UrlFetcher | None = None, token: str | None = None):
        self.fetcher = fetcher or StandardUrlFetcher()
        self.token = token if token is not None else (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN", ""))

    def search(self, topic: str, limit: int = 25) -> list[RawDocument]:
        documents: list[RawDocument] = []
        model_limit = max(1, (limit + 1) // 2)
        dataset_limit = max(1, limit - model_limit)
        documents.extend(self._search_kind("model", topic, model_limit))
        if len(documents) < limit:
            documents.extend(self._search_kind("dataset", topic, dataset_limit))
        return documents[:limit]

    def _search_kind(self, asset_type: str, topic: str, limit: int) -> list[RawDocument]:
        endpoint = "models" if asset_type == "model" else "datasets"
        params = {"search": topic, "limit": min(limit, 100), "full": "true"}
        url = f"{self.base_url}/{endpoint}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "SignalGraph/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self.fetcher.get_json(url, headers=headers)
        documents: list[RawDocument] = []
        for item in _items(response, endpoint)[:limit]:
            payload = dict(item)
            payload["asset_type"] = asset_type
            asset_id = str(payload.get("modelId") or payload.get("id") or payload.get("_id") or "")
            source_url = f"https://huggingface.co/{'datasets/' if asset_type == 'dataset' else ''}{asset_id}" if asset_id else ""
            documents.append(raw_document(self.source_name, source_url, asset_id, params | {"asset_type": asset_type}, payload))
        return documents


class SourceSearchService:
    def __init__(self, clients: dict[str, Any] | None = None):
        self.clients = clients or {
            "arxiv": ArxivClient(),
            "openalex": OpenAlexClient(),
            "semantic_scholar": SemanticScholarClient(),
            "github": GitHubClient(),
            "huggingface": HuggingFaceClient(),
        }

    def search(self, topic: str, limit: int = 25, source_names: list[str] | None = None, per_source_limit: int | None = None) -> list[RawDocument]:
        selected = source_names or list(self.clients)
        per_source = per_source_limit or max(1, limit // max(1, len(selected)))
        documents: list[RawDocument] = []
        for name in selected:
            if name not in self.clients:
                raise ValueError(f"Unknown source '{name}'. Expected one of: {', '.join(sorted(self.clients))}")
            documents.extend(self.clients[name].search(topic, per_source))
        return documents[:limit]


def raw_document(source_name: str, source_url: str, source_id: str, request_params: dict[str, Any], raw_payload: dict[str, Any]) -> RawDocument:
    return RawDocument(
        source_name=source_name,
        source_url=source_url,
        source_id=source_id,
        request_params=request_params,
        raw_payload=raw_payload,
        source_metadata=source_policy_metadata(source_name),
    )


def source_policy_metadata(source_name: str) -> dict[str, Any]:
    policy = SOURCE_POLICIES.get(source_name)
    if not policy:
        return {
            "license_or_terms_note": "",
            "freshness_policy": "source-specific freshness policy not configured",
            "stale_after_days": 0,
            "rate_limit_note": "Use bounded requests and source-specific published limits.",
            "cache_policy": "Append-only raw payload cache keyed by source, stable id, request params, and response hash.",
        }
    return asdict(policy)


def _xml_text(element: ET.Element, path: str, ns: dict[str, str]) -> str:
    found = element.find(path, ns)
    return "".join(found.itertext()).strip() if found is not None else ""


def _items(response: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        values = response.get(key) or response.get("data") or response.get("items") or response.get("results")
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
    return []


def _decode_github_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, str):
        return ""
    if payload.get("encoding") == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return content


def _github_risk_signals(payload: dict[str, Any], issues: list[dict[str, Any]], releases: list[dict[str, Any]], has_changelog: bool) -> list[str]:
    signals: list[str] = []
    if not payload.get("license"):
        signals.append("missing_license")
    if not releases:
        signals.append("no_recent_release_metadata")
    if not has_changelog:
        signals.append("missing_changelog")
    issue_labels = " ".join(" ".join(issue.get("labels", [])) for issue in issues).lower()
    issue_text = " ".join(f"{issue.get('title', '')} {issue.get('body', '')}" for issue in issues).lower()
    if any(word in issue_labels or word in issue_text for word in ["bug", "regression", "broken", "install", "security"]):
        signals.append("recent_issue_risk")
    if int(payload.get("open_issues_count") or 0) > 100:
        signals.append("high_open_issue_count")
    return signals
