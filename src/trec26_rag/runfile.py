from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Iterable


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    narrative: str


@dataclass(frozen=True)
class RunRow:
    topic_id: str
    docid: str
    rank: int
    score: float
    run_id: str

    def to_trec_line(self) -> str:
        return f"{self.topic_id} Q0 {self.docid} {self.rank} {self.score:.6f} {self.run_id}"


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def read_topics(path: str | Path) -> list[Topic]:
    topics_path = Path(path)
    topics: list[Topic] = []
    with topics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {topics_path}: {exc}") from exc
            for field in ("id", "title", "narrative"):
                if field not in payload:
                    raise ValueError(f"Topic line {line_number} is missing required field: {field}")
            topics.append(
                Topic(
                    id=str(payload["id"]),
                    title=str(payload["title"]),
                    narrative=str(payload["narrative"]),
                )
            )
    if not topics:
        raise ValueError(f"No topics found in {topics_path}")
    return topics


def render_query(template: str, topic: Topic) -> str:
    formatter = Formatter()
    query = formatter.vformat(template, (), SafeFormatDict(topic.__dict__))
    return " ".join(query.split())


def write_runfile(rows: Iterable[RunRow], path: str | Path) -> None:
    runfile_path = Path(path)
    runfile_path.parent.mkdir(parents=True, exist_ok=True)
    with runfile_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_trec_line())
            handle.write("\n")


def validate_runfile(path: str | Path, topic_ids: set[str] | None = None) -> dict[str, Any]:
    runfile_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    topic_counts: dict[str, int] = {}
    last_rank: dict[str, int] = {}
    last_score: dict[str, float] = {}
    total_rows = 0

    if not runfile_path.exists():
        return {
            "valid": False,
            "errors": [f"Runfile does not exist: {runfile_path}"],
            "warnings": [],
            "metrics": {},
        }

    with runfile_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            parts = line.split()
            if len(parts) != 6:
                errors.append(f"Line {line_number}: expected 6 columns, found {len(parts)}")
                continue
            topic_id, q0, docid, rank_text, score_text, run_id = parts
            if q0 != "Q0":
                errors.append(f"Line {line_number}: second column must be Q0")
            if not docid:
                errors.append(f"Line {line_number}: docid is empty")
            if not run_id:
                errors.append(f"Line {line_number}: run_id is empty")
            try:
                rank = int(rank_text)
            except ValueError:
                errors.append(f"Line {line_number}: rank is not an integer")
                continue
            try:
                score = float(score_text)
            except ValueError:
                errors.append(f"Line {line_number}: score is not numeric")
                continue

            expected_rank = last_rank.get(topic_id, 0) + 1
            if rank != expected_rank:
                errors.append(
                    f"Line {line_number}: rank for topic {topic_id} should be {expected_rank}, found {rank}"
                )
            previous_score = last_score.get(topic_id)
            if previous_score is not None and score > previous_score:
                errors.append(f"Line {line_number}: score increased within topic {topic_id}")
            last_rank[topic_id] = rank
            last_score[topic_id] = score
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1

    if topic_ids is not None:
        observed = set(topic_counts)
        missing = sorted(topic_ids - observed)
        extra = sorted(observed - topic_ids)
        if missing:
            errors.append(f"Missing output for {len(missing)} topic(s): {', '.join(missing[:10])}")
        if extra:
            warnings.append(f"Runfile contains {len(extra)} topic(s) not in topic file: {', '.join(extra[:10])}")

    n_topics = len(topic_counts)
    candidate_counts = list(topic_counts.values())
    candidate_count_mean = (
        sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "n_topics": n_topics,
            "total_rows": total_rows,
            "candidate_count_mean": candidate_count_mean,
            "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
            "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
            "validation_error_count": len(errors),
            "validation_warning_count": len(warnings),
        },
    }
