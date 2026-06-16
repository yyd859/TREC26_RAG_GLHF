from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .runfile import Topic


@dataclass(frozen=True)
class AnswerSentence:
    text: str
    citations: list[int]


@dataclass(frozen=True)
class RagResponse:
    topic: Topic
    team_id: str
    run_id: str
    references: list[str]
    answer: list[AnswerSentence]
    run_type: str = "automatic"
    prompt: str | None = None


def rag_response_to_json(response: RagResponse) -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": response.team_id,
            "run_id": response.run_id,
            "type": response.run_type,
            "narrative_id": response.topic.id,
            "title": response.topic.title,
            "narrative": response.topic.narrative,
            "prompt": response.prompt,
        },
        "references": response.references,
        "answer": [
            {
                "text": sentence.text,
                "citations": sentence.citations,
            }
            for sentence in response.answer
        ],
    }


def write_rag_jsonl(responses: Iterable[RagResponse], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps(rag_response_to_json(response), ensure_ascii=False))
            handle.write("\n")


def keep_cited_references_only(response: RagResponse) -> RagResponse:
    cited_indices = sorted(
        {
            citation
            for sentence in response.answer
            for citation in sentence.citations
            if 0 <= citation < len(response.references)
        }
    )
    if not cited_indices or len(cited_indices) == len(response.references):
        return response

    index_mapping = {old_index: new_index for new_index, old_index in enumerate(cited_indices)}
    return RagResponse(
        topic=response.topic,
        team_id=response.team_id,
        run_id=response.run_id,
        references=[response.references[index] for index in cited_indices],
        answer=[
            AnswerSentence(
                text=sentence.text,
                citations=[
                    index_mapping[citation]
                    for citation in sentence.citations
                    if citation in index_mapping
                ],
            )
            for sentence in response.answer
        ],
        run_type=response.run_type,
        prompt=response.prompt,
    )


def extract_answer_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end >= object_start:
        return text[object_start : object_end + 1]
    return text


def parse_answer_json(
    raw_text: str,
    topic: Topic,
    team_id: str,
    run_id: str,
    fallback_references: list[str],
    prompt: str | None = None,
) -> RagResponse:
    payload = json.loads(extract_answer_json_text(raw_text))
    references = payload.get("references") or fallback_references
    answer_payload = payload.get("answer") or []
    if not isinstance(references, list):
        raise ValueError("RAG answer payload references must be a list.")
    if not isinstance(answer_payload, list):
        raise ValueError("RAG answer payload answer must be a list.")
    answer = [
        AnswerSentence(
            text=str(sentence.get("text", "")),
            citations=[int(citation) for citation in sentence.get("citations", [])],
        )
        for sentence in answer_payload
        if isinstance(sentence, dict)
    ]
    return RagResponse(
        topic=topic,
        team_id=team_id,
        run_id=run_id,
        references=[str(reference) for reference in references],
        answer=answer,
        prompt=prompt,
    )
