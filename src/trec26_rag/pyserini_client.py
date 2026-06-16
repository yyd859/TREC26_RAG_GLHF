from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


class PyseriniClientError(RuntimeError):
    """Raised when the Pyserini REST API request fails."""


@dataclass(frozen=True)
class SearchHit:
    docid: str
    rank: int
    score: float
    doc: Any | None = None

    @property
    def text(self) -> str:
        return extract_document_text(self.doc)

    def has_sufficient_text(self, min_text_chars: int) -> bool:
        return len(self.text) >= min_text_chars


def load_env_file(path: str | Path = ".env.local") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_pyserini_token() -> str:
    load_env_file()
    token = os.environ.get("PYSERINI_API_TOKEN")
    if not token:
        raise PyseriniClientError(
            "PYSERINI_API_TOKEN is missing. Add it to .env.local or the environment."
        )
    return token


class PyseriniClient:
    def __init__(
        self,
        base_url: str,
        index: str,
        token: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.index = index
        self.token = token or get_pyserini_token()
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, hits: int) -> list[SearchHit]:
        url = f"{self.base_url}/v1/{self.index}/search"
        response = requests.get(
            url,
            params={"query": query, "hits": hits},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 401:
            raise PyseriniClientError(
                "Pyserini authorization failed. The local token may be missing, expired, or invalid."
            )
        if response.status_code >= 400:
            raise PyseriniClientError(
                f"Pyserini search failed with HTTP {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise PyseriniClientError("Pyserini search response did not include a candidates list.")
        return [self._parse_hit(candidate, index) for index, candidate in enumerate(candidates, 1)]

    def fetch_doc(self, docid: str) -> Any:
        escaped_docid = quote(docid, safe="")
        url = f"{self.base_url}/v1/{self.index}/doc/{escaped_docid}"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 401:
            raise PyseriniClientError(
                "Pyserini authorization failed. The local token may be missing, expired, or invalid."
            )
        if response.status_code >= 400:
            raise PyseriniClientError(
                f"Pyserini document fetch failed with HTTP {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("doc") or payload.get("document") or payload
        return payload

    def hydrate_hits(
        self,
        hits: list[SearchHit],
        min_text_chars: int = 200,
        max_docs: int | None = None,
    ) -> list[SearchHit]:
        hydrated: list[SearchHit] = []
        fetch_budget = len(hits) if max_docs is None else max_docs
        for hit in hits:
            if fetch_budget > 0 and not hit.has_sufficient_text(min_text_chars):
                hydrated.append(replace(hit, doc=self.fetch_doc(hit.docid)))
                fetch_budget -= 1
            else:
                hydrated.append(hit)
        return hydrated

    @staticmethod
    def _parse_hit(candidate: dict[str, Any], fallback_rank: int) -> SearchHit:
        docid = candidate.get("docid")
        if not docid:
            raise PyseriniClientError(f"Search candidate is missing docid: {candidate}")
        rank = int(candidate.get("rank") or fallback_rank)
        score = float(candidate.get("score") or 0.0)
        return SearchHit(docid=str(docid), rank=rank, score=score, doc=candidate.get("doc"))


def extract_document_text(payload: Any) -> str:
    text = _extract_document_text(payload)
    return " ".join(text.split())


def _extract_document_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (int, float, bool)):
        return ""
    if isinstance(payload, list):
        return " ".join(filter(None, (extract_document_text(item) for item in payload)))
    if isinstance(payload, dict):
        preferred_keys = (
            "text",
            "contents",
            "content",
            "body",
            "raw",
            "passage",
            "document",
            "doc",
        )
        for key in preferred_keys:
            if key in payload:
                text = extract_document_text(payload[key])
                if text:
                    return text
        return " ".join(
            filter(
                None,
                (
                    extract_document_text(value)
                    for key, value in payload.items()
                    if key not in {"id", "docid", "url", "rank", "score"}
                ),
            )
        )
    return ""
