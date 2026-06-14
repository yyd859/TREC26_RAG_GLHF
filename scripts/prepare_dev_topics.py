#!/usr/bin/env python
from __future__ import annotations

import argparse

from trec26_rag.topics import DEFAULT_TOPIC_URLS, prepare_topics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TREC RAG 2026 dev topics as JSONL.")
    parser.add_argument("--source", action="append", help="TSV URL or local TSV path. Can be repeated.")
    parser.add_argument("--output", default="data/trec_rag_2026_queries.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = args.source or DEFAULT_TOPIC_URLS
    count = prepare_topics(sources, args.output)
    print(f"Prepared {count} topics at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
