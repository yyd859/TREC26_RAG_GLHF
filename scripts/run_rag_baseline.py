#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from trec26_rag.config import load_config
from trec26_rag.generator import (
    AnthropicBatchAnswerGenerator,
    AnswerGenerationRequest,
    AnswerGeneratorError,
    EvidenceDocument,
    assign_custom_ids,
    parse_batch_results_jsonl,
    render_rag_prompt,
)
from trec26_rag.pyserini_client import PyseriniClient, SearchHit
from trec26_rag.rag_output import RagResponse, parse_answer_json, write_rag_jsonl
from trec26_rag.rag_validation import validate_rag_jsonl
from trec26_rag.runfile import Topic, read_topics, render_query
from trec26_rag.wandb_logging import log_rag_run


TERMINAL_BATCH_STATUSES = {"ended", "canceled", "expired"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TREC RAG 2026 RAG baseline.")
    parser.add_argument("--config", default="configs/baseline_rag.yaml")
    parser.add_argument("--topics", help="Override topics path from config.")
    parser.add_argument("--output", help="Override RAG JSONL output path.")
    parser.add_argument("--limit", type=int, help="Limit topics for smoke tests.")
    parser.add_argument("--batch-id", help="Reuse an existing Anthropic message batch.")
    parser.add_argument("--raw-results-output", help="Override raw Anthropic batch results path.")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    parser.add_argument("--log-wandb", action="store_true", help="Log metrics and artifacts to W&B.")
    return parser.parse_args()


def build_generation_requests(
    topics: list[Topic],
    client: PyseriniClient,
    config: dict[str, Any],
) -> list[AnswerGenerationRequest]:
    retrieval_config = config["retrieval"]
    rag_config = config["rag"]
    evidence_top_k = int(rag_config["evidence_top_k"])
    requested_hits = max(int(retrieval_config.get("hits", evidence_top_k)), evidence_top_k)

    requests: list[AnswerGenerationRequest] = []
    for topic in topics:
        query = render_query(retrieval_config["query_template"], topic)
        hits = client.search(query=query, hits=requested_hits)
        evidence_hits = client.hydrate_hits(
            hits[:evidence_top_k],
            min_text_chars=100,
            max_docs=evidence_top_k,
        )
        requests.append(
            AnswerGenerationRequest(
                topic=topic,
                evidence=[hit_to_evidence(hit) for hit in evidence_hits],
            )
        )
    return requests


def hit_to_evidence(hit: SearchHit) -> EvidenceDocument:
    text = hit.text
    if not text and hit.doc is not None:
        text = json.dumps(hit.doc, ensure_ascii=False, sort_keys=True)
    if not text:
        text = f"No text was available for document {hit.docid}."
    return EvidenceDocument(docid=hit.docid, text=text)


def wait_for_batch(
    generator: AnthropicBatchAnswerGenerator,
    batch_id: str,
    poll_interval_seconds: int,
    max_wait_seconds: int,
) -> str:
    started_at = time.monotonic()
    while True:
        job = generator.retrieve_batch(batch_id)
        status = job.processing_status
        print(json.dumps({"batch_id": batch_id, "processing_status": status}))
        if status in TERMINAL_BATCH_STATUSES:
            return status
        if time.monotonic() - started_at > max_wait_seconds:
            raise AnswerGeneratorError(
                f"Anthropic batch {batch_id} did not finish within {max_wait_seconds} seconds."
            )
        time.sleep(poll_interval_seconds)


def build_rag_responses(
    answer_requests: list[AnswerGenerationRequest],
    batch_results: dict[str, str],
    config: dict[str, Any],
) -> list[RagResponse]:
    experiment_config = config["experiment"]
    prompt_template = config["rag"].get("prompt_template") or ""
    responses: list[RagResponse] = []
    missing_custom_ids: list[str] = []
    for assignment in assign_custom_ids(answer_requests):
        request = assignment.request
        raw_answer = batch_results.get(assignment.custom_id)
        if raw_answer is None:
            missing_custom_ids.append(assignment.custom_id)
            continue
        fallback_references = [doc.docid for doc in request.evidence]
        responses.append(
            parse_answer_json(
                raw_text=raw_answer,
                topic=request.topic,
                team_id=experiment_config["team_id"],
                run_id=experiment_config["run_id"],
                fallback_references=fallback_references,
                prompt=render_rag_prompt(request, prompt_template),
            )
        )
    if missing_custom_ids:
        raise AnswerGeneratorError(
            "Anthropic batch results were missing topic result(s): "
            + ", ".join(missing_custom_ids[:10])
        )
    return responses


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not config.get("rag", {}).get("enabled"):
        raise AnswerGeneratorError("RAG is disabled in config. Set rag.enabled: true.")

    topics_path = Path(args.topics or config["data"]["topics_path"])
    topics = read_topics(topics_path)
    if args.limit:
        topics = topics[: args.limit]

    output_config = config["output"]
    output_dir = Path(output_config["output_dir"])
    rag_output_path = Path(args.output) if args.output else output_dir / output_config["rag_output_name"]
    report_path = output_dir / output_config.get(
        "rag_validation_report_name", "rag_validation_report.json"
    )
    raw_results_path = (
        Path(args.raw_results_output)
        if args.raw_results_output
        else output_dir / "anthropic_batch_results.jsonl"
    )

    client = PyseriniClient(
        base_url=config["retrieval"]["api_base_url"],
        index=config["retrieval"]["index"],
        timeout_seconds=int(config["retrieval"].get("timeout_seconds", 30)),
    )
    generator = AnthropicBatchAnswerGenerator.from_config(config)

    started_at = time.monotonic()
    answer_requests = build_generation_requests(topics, client, config)
    if args.batch_id:
        batch_id = args.batch_id
    else:
        batch = generator.create_batch(answer_requests)
        batch_id = batch.id
        print(json.dumps({"batch_id": batch_id, "processing_status": batch.processing_status}))

    final_status = wait_for_batch(
        generator=generator,
        batch_id=batch_id,
        poll_interval_seconds=args.poll_interval_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    if final_status != "ended":
        raise AnswerGeneratorError(f"Anthropic batch {batch_id} ended with status: {final_status}")

    generator.download_results_jsonl(batch_id, raw_results_path)
    batch_results = parse_batch_results_jsonl(raw_results_path)
    responses = build_rag_responses(answer_requests, batch_results, config)
    write_rag_jsonl(responses, rag_output_path)

    report = validate_rag_jsonl(rag_output_path, topic_ids={topic.id for topic in topics})
    metrics = dict(report["metrics"])
    metrics["runtime_seconds"] = round(time.monotonic() - started_at, 3)
    metrics["requested_topics"] = len(topics)
    metrics["requested_evidence_top_k"] = int(config["rag"]["evidence_top_k"])
    metrics["anthropic_batch_completed"] = 1
    report["batch"] = {
        "id": batch_id,
        "processing_status": final_status,
        "raw_results_path": str(raw_results_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.log_wandb:
        run_url = log_rag_run(config, metrics, [rag_output_path, report_path, raw_results_path, args.config])
        if run_url:
            print(f"W&B run: {run_url}")

    print(
        json.dumps(
            {
                "rag_output": str(rag_output_path),
                "report": str(report_path),
                "batch_id": batch_id,
                **metrics,
            },
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
