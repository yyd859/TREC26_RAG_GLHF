from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_FIELDS = (
    "team_id",
    "run_id",
    "type",
    "narrative_id",
    "title",
    "narrative",
)


def validate_rag_jsonl(path: str | Path, topic_ids: set[str] | None = None) -> dict[str, Any]:
    output_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    seen_topic_ids: set[str] = set()
    per_topic: dict[str, dict[str, Any]] = {}
    invalid_citation_count = 0
    uncited_reference_count = 0
    empty_answer_count = 0
    total_references = 0
    total_sentences = 0
    total_citations = 0
    object_count = 0

    if not output_path.exists():
        return {
            "valid": False,
            "errors": [f"RAG output does not exist: {output_path}"],
            "warnings": [],
            "metrics": {},
            "diagnostics": {},
        }

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            object_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSONL object: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"Line {line_number}: each JSONL object must be a JSON object")
                continue

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                errors.append(f"Line {line_number}: metadata must be an object")
                metadata = {}
            missing_fields = [
                field for field in REQUIRED_METADATA_FIELDS if not str(metadata.get(field) or "").strip()
            ]
            if missing_fields:
                errors.append(
                    f"Line {line_number}: metadata missing required field(s): {', '.join(missing_fields)}"
                )
            topic_id = str(metadata.get("narrative_id") or f"line-{line_number}")
            if topic_id in seen_topic_ids:
                errors.append(f"Line {line_number}: duplicate RAG object for topic {topic_id}")
            seen_topic_ids.add(topic_id)

            references = payload.get("references")
            if not isinstance(references, list):
                errors.append(f"Line {line_number}: references must be a list")
                references = []
            references = [str(reference) for reference in references]
            if not references:
                errors.append(f"Line {line_number}: references must be non-empty")

            answer = payload.get("answer")
            if not isinstance(answer, list):
                errors.append(f"Line {line_number}: answer must be a sentence-level list")
                answer = []
            if not answer:
                empty_answer_count += 1
                errors.append(f"Line {line_number}: answer must contain at least one sentence")

            cited_indices: set[int] = set()
            topic_invalid_citations = 0
            topic_empty_sentences = 0
            for sentence_index, sentence in enumerate(answer):
                if not isinstance(sentence, dict):
                    errors.append(f"Line {line_number}: answer[{sentence_index}] must be an object")
                    continue
                text = sentence.get("text")
                if not isinstance(text, str) or not text.strip():
                    topic_empty_sentences += 1
                    errors.append(f"Line {line_number}: answer[{sentence_index}].text must be non-empty")
                citations = sentence.get("citations")
                if not isinstance(citations, list):
                    errors.append(f"Line {line_number}: answer[{sentence_index}].citations must be a list")
                    citations = []
                for citation in citations:
                    if not isinstance(citation, int) or citation < 0 or citation >= len(references):
                        topic_invalid_citations += 1
                        errors.append(
                            f"Line {line_number}: answer[{sentence_index}] has invalid citation index {citation}"
                        )
                        continue
                    cited_indices.add(citation)
                total_citations += len(citations)
            topic_uncited_references = len(set(range(len(references))) - cited_indices)
            if topic_uncited_references:
                errors.append(
                    f"Line {line_number}: {topic_uncited_references} reference(s) are not cited"
                )

            total_references += len(references)
            total_sentences += len(answer)
            invalid_citation_count += topic_invalid_citations
            uncited_reference_count += topic_uncited_references
            per_topic[topic_id] = {
                "reference_count": len(references),
                "answer_sentence_count": len(answer),
                "citation_count": sum(
                    len(sentence.get("citations", [])) for sentence in answer if isinstance(sentence, dict)
                ),
                "invalid_citation_count": topic_invalid_citations,
                "uncited_reference_count": topic_uncited_references,
                "empty_sentence_count": topic_empty_sentences,
            }

    missing_topics: list[str] = []
    extra_topics: list[str] = []
    if topic_ids is not None:
        missing_topics = sorted(topic_ids - seen_topic_ids)
        extra_topics = sorted(seen_topic_ids - topic_ids)
        if missing_topics:
            errors.append(
                f"Missing RAG output for {len(missing_topics)} topic(s): {', '.join(missing_topics[:10])}"
            )
        if extra_topics:
            warnings.append(
                f"RAG output contains {len(extra_topics)} topic(s) not in topic file: {', '.join(extra_topics[:10])}"
            )

    topic_count = len(seen_topic_ids)
    metrics = {
        "rag_object_count": object_count,
        "rag_topic_count": topic_count,
        "rag_expected_topic_count": len(topic_ids) if topic_ids is not None else topic_count,
        "rag_missing_topic_count": len(missing_topics),
        "rag_extra_topic_count": len(extra_topics),
        "rag_reference_count_total": total_references,
        "rag_reference_count_mean": total_references / topic_count if topic_count else 0.0,
        "rag_answer_sentence_count_total": total_sentences,
        "rag_answer_sentence_count_mean": total_sentences / topic_count if topic_count else 0.0,
        "rag_citation_count_total": total_citations,
        "rag_citation_count_mean": total_citations / topic_count if topic_count else 0.0,
        "rag_invalid_citation_count": invalid_citation_count,
        "rag_uncited_reference_count": uncited_reference_count,
        "rag_empty_answer_count": empty_answer_count,
        "rag_validation_error_count": len(errors),
        "rag_validation_warning_count": len(warnings),
    }
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "diagnostics": {
            "per_topic": per_topic,
            "missing_topics": missing_topics,
            "extra_topics": extra_topics,
        },
    }
