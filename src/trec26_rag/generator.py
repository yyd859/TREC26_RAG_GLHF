from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .pyserini_client import load_env_file
from .runfile import Topic


ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CLAUDE_HAIKU_45 = "claude-haiku-4-5-20251001"


class AnswerGeneratorError(RuntimeError):
    """Raised when an answer generation request cannot be prepared or submitted."""


@dataclass(frozen=True)
class EvidenceDocument:
    docid: str
    text: str


@dataclass(frozen=True)
class AnswerGenerationRequest:
    topic: Topic
    evidence: list[EvidenceDocument]


@dataclass(frozen=True)
class BatchJob:
    id: str
    processing_status: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class BatchRequestAssignment:
    custom_id: str
    request: AnswerGenerationRequest


def get_anthropic_api_key() -> str:
    load_env_file()
    token = os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        raise AnswerGeneratorError(
            "ANTHROPIC_API_KEY is missing. Add it to .env.local or GitHub Actions secrets."
        )
    return token


def safe_custom_id(topic_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", topic_id)
    if not cleaned:
        cleaned = "topic"
    return cleaned[:64]


def assign_custom_ids(
    requests_to_generate: Iterable[AnswerGenerationRequest],
) -> list[BatchRequestAssignment]:
    assignments: list[BatchRequestAssignment] = []
    seen_ids: set[str] = set()
    for request in requests_to_generate:
        custom_id = safe_custom_id(request.topic.id)
        if custom_id in seen_ids:
            custom_id = safe_custom_id(f"{request.topic.id}_{len(seen_ids)}")
        seen_ids.add(custom_id)
        assignments.append(BatchRequestAssignment(custom_id=custom_id, request=request))
    return assignments


def render_rag_prompt(
    request: AnswerGenerationRequest,
    prompt_template: str,
    max_evidence_chars: int = 4000,
) -> str:
    evidence_blocks = []
    for index, doc in enumerate(request.evidence):
        text = " ".join(doc.text.split())
        if len(text) > max_evidence_chars:
            text = text[:max_evidence_chars].rsplit(" ", 1)[0]
        evidence_blocks.append(f"[{index}] {doc.docid}\n{text}")
    return "\n\n".join(
        [
            prompt_template.strip(),
            f"Topic ID: {request.topic.id}",
            f"Title: {request.topic.title}",
            f"Narrative: {request.topic.narrative}",
            "Evidence:",
            "\n\n".join(evidence_blocks),
            (
                "Return JSON only with this shape: "
                '{"references":["docid"],"answer":[{"text":"sentence","citations":[0]}]}. '
                "Citations must be zero-indexed positions into references."
            ),
        ]
    )


class AnthropicBatchAnswerGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str = ANTHROPIC_API_BASE_URL,
        model: str = DEFAULT_CLAUDE_HAIKU_45,
        max_output_tokens: int = 800,
        prompt_template: str = "",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or get_anthropic_api_key()
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.prompt_template = prompt_template
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnthropicBatchAnswerGenerator":
        rag_config = config.get("rag", {})
        provider = rag_config.get("generator_provider")
        if provider != "anthropic_batch":
            raise AnswerGeneratorError(f"Unsupported generator_provider for Claude batch: {provider}")
        return cls(
            model=rag_config.get("model") or DEFAULT_CLAUDE_HAIKU_45,
            max_output_tokens=int(rag_config.get("max_output_tokens", 800)),
            prompt_template=rag_config.get("prompt_template") or "",
        )

    def build_batch_requests(
        self,
        requests_to_generate: Iterable[AnswerGenerationRequest],
    ) -> list[dict[str, Any]]:
        batch_requests: list[dict[str, Any]] = []
        for assignment in assign_custom_ids(requests_to_generate):
            request = assignment.request
            batch_requests.append(
                {
                    "custom_id": assignment.custom_id,
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_output_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": render_rag_prompt(request, self.prompt_template),
                            }
                        ],
                    },
                }
            )
        if not batch_requests:
            raise AnswerGeneratorError("Cannot create an empty Anthropic message batch.")
        return batch_requests

    def create_batch(self, requests_to_generate: Iterable[AnswerGenerationRequest]) -> BatchJob:
        payload = {"requests": self.build_batch_requests(requests_to_generate)}
        response = requests.post(
            f"{self.api_base_url}/v1/messages/batches",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "create Anthropic message batch")
        body = response.json()
        return BatchJob(
            id=str(body["id"]),
            processing_status=str(body.get("processing_status", "unknown")),
            raw=body,
        )

    def retrieve_batch(self, batch_id: str) -> BatchJob:
        response = requests.get(
            f"{self.api_base_url}/v1/messages/batches/{batch_id}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "retrieve Anthropic message batch")
        body = response.json()
        return BatchJob(
            id=str(body["id"]),
            processing_status=str(body.get("processing_status", "unknown")),
            raw=body,
        )

    def download_results_jsonl(self, batch_id: str, output_path: str | Path) -> Path:
        response = requests.get(
            f"{self.api_base_url}/v1/messages/batches/{batch_id}/results",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "download Anthropic message batch results")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return path

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    @staticmethod
    def _raise_for_error(response: requests.Response, action: str) -> None:
        if response.status_code >= 400:
            raise AnswerGeneratorError(
                f"Failed to {action} with HTTP {response.status_code}: {response.text[:300]}"
            )


def parse_batch_results_jsonl(path: str | Path) -> dict[str, str]:
    results: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            custom_id = payload.get("custom_id")
            result = payload.get("result", {})
            if not custom_id:
                raise AnswerGeneratorError(f"Batch result line {line_number} is missing custom_id")
            if result.get("type") != "succeeded":
                continue
            message = result.get("message", {})
            text_parts = [
                block.get("text", "")
                for block in message.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            results[str(custom_id)] = "\n".join(part for part in text_parts if part)
    return results
