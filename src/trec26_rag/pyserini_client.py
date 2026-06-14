from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class PyseriniClientError(RuntimeError):
    """Raised when the Pyserini REST API request fails."""


@dataclass(frozen=True)
class SearchHit:
    docid: str
    rank: int
    score: float
    doc: Any | None = None


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

    @staticmethod
    def _parse_hit(candidate: dict[str, Any], fallback_rank: int) -> SearchHit:
        docid = candidate.get("docid")
        if not docid:
            raise PyseriniClientError(f"Search candidate is missing docid: {candidate}")
        rank = int(candidate.get("rank") or fallback_rank)
        score = float(candidate.get("score") or 0.0)
        return SearchHit(docid=str(docid), rank=rank, score=score, doc=candidate.get("doc"))
