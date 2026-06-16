#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from trec26_rag.rag_validation import validate_rag_jsonl
from trec26_rag.runfile import read_topics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TREC RAG 2026 RAG JSONL output.")
    parser.add_argument("--rag-output", required=True)
    parser.add_argument("--topics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topic_ids = None
    if args.topics:
        topic_ids = {topic.id for topic in read_topics(args.topics)}
    report = validate_rag_jsonl(args.rag_output, topic_ids=topic_ids)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
