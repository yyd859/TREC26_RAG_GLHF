from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_TOPIC_URLS = [
    "https://raw.githubusercontent.com/TREC-RAG/trec-rag-data/main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    "https://raw.githubusercontent.com/TREC-RAG/trec-rag-data/main/trec-rag-2026/development-data/topics/research-rubrics-topics-dev.tsv",
]


def read_text_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        request = Request(source, headers={"User-Agent": "trec26-rag-glhf/0.1"})
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def title_from_prompt(prompt: str, max_chars: int = 180) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0]


def parse_topic_tsv(text: str) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    reader = csv.reader(text.splitlines(), delimiter="\t")
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < 2:
            raise ValueError(f"Expected at least two TSV columns, found {len(row)}: {row}")
        topic_id = row[0].strip()
        prompt = row[1].strip()
        if topic_id.lower() in {"id", "topic_id", "qid"}:
            continue
        topics.append(
            {
                "id": topic_id,
                "title": title_from_prompt(prompt),
                "narrative": prompt,
            }
        )
    return topics


def prepare_topics(sources: list[str], output_path: str | Path) -> int:
    seen_ids: set[str] = set()
    topics: list[dict[str, str]] = []
    for source in sources:
        for topic in parse_topic_tsv(read_text_source(source)):
            if topic["id"] in seen_ids:
                continue
            seen_ids.add(topic["id"])
            topics.append(topic)
    if not topics:
        raise ValueError("No topics were prepared.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for topic in topics:
            handle.write(json.dumps(topic, ensure_ascii=True))
            handle.write("\n")
    return len(topics)
