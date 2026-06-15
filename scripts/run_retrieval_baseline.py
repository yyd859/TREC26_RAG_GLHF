#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trec26_rag.config import load_config
from trec26_rag.pyserini_client import PyseriniClient
from trec26_rag.runfile import RunRow, read_topics, render_query, validate_runfile, write_runfile
from trec26_rag.wandb_logging import log_retrieval_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TREC RAG 2026 retrieval baseline.")
    parser.add_argument("--config", default="configs/baseline_retrieval.yaml")
    parser.add_argument("--topics", help="Override topics path from config.")
    parser.add_argument("--output", help="Override runfile output path.")
    parser.add_argument("--limit", type=int, help="Limit topics for smoke tests.")
    parser.add_argument("--log-wandb", action="store_true", help="Log metrics and artifacts to W&B.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    topics_path = Path(args.topics or config["data"]["topics_path"])
    topics = read_topics(topics_path)
    if args.limit:
        topics = topics[: args.limit]

    retrieval_config = config["retrieval"]
    output_config = config["output"]
    output_dir = Path(output_config["output_dir"])
    runfile_path = Path(args.output) if args.output else output_dir / output_config["runfile_name"]
    report_path = output_dir / output_config["validation_report_name"]

    client = PyseriniClient(
        base_url=retrieval_config["api_base_url"],
        index=retrieval_config["index"],
        timeout_seconds=int(retrieval_config.get("timeout_seconds", 30)),
    )

    started_at = time.monotonic()
    rows: list[RunRow] = []
    per_topic_latency: dict[str, float] = {}
    for topic in topics:
        query = render_query(retrieval_config["query_template"], topic)
        topic_started_at = time.monotonic()
        hits = client.search(query=query, hits=int(retrieval_config["hits"]))
        per_topic_latency[topic.id] = round(time.monotonic() - topic_started_at, 3)
        for rank, hit in enumerate(hits, 1):
            rows.append(
                RunRow(
                    topic_id=topic.id,
                    docid=hit.docid,
                    rank=rank,
                    score=hit.score,
                    run_id=config["experiment"]["run_id"],
                )
            )

    write_runfile(rows, runfile_path)
    report = validate_runfile(runfile_path, topic_ids={topic.id for topic in topics})
    metrics = dict(report["metrics"])
    metrics["runtime_seconds"] = round(time.monotonic() - started_at, 3)
    metrics["requested_topics"] = len(topics)
    metrics["requested_hits"] = int(retrieval_config["hits"])
    latency_values = list(per_topic_latency.values())
    metrics["latency_seconds_min"] = min(latency_values) if latency_values else 0.0
    metrics["latency_seconds_max"] = max(latency_values) if latency_values else 0.0
    metrics["latency_seconds_mean"] = (
        sum(latency_values) / len(latency_values) if latency_values else 0.0
    )
    metrics["latency_seconds_total"] = sum(latency_values)
    report["metrics"] = metrics
    diagnostics = report.setdefault("diagnostics", {})
    per_topic = diagnostics.setdefault("per_topic", {})
    for topic_id, latency in per_topic_latency.items():
        per_topic.setdefault(topic_id, {})["latency_seconds"] = latency
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.log_wandb:
        run_url = log_retrieval_run(config, metrics, [runfile_path, report_path, args.config])
        if run_url:
            print(f"W&B run: {run_url}")

    print(json.dumps({"runfile": str(runfile_path), "report": str(report_path), **metrics}, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
