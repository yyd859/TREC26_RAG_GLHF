#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from trec26_rag.evaluation import evaluate_retrieval_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval runfile against qrels.")
    parser.add_argument("--runfile", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--relevance-threshold", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_retrieval_run(
        runfile_path=args.runfile,
        qrels_path=args.qrels,
        relevance_threshold=args.relevance_threshold,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
