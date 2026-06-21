from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "baseline_title_top100",
        "hypothesis": "A title-only BM25 query is the smallest useful retrieval baseline.",
        "task": "retrieval",
        "track_year": 2026,
        "team_id": "glhf",
        "run_id": "glhf-title-top100",
    },
    "data": {
        "topics_path": "data/trec_rag_2026_queries.jsonl",
    },
    "retrieval": {
        "api_base_url": "http://99.251.12.72:8081",
        "index": "climbmix-400b",
        "query_template": "{title}",
        "hits": 100,
        "timeout_seconds": None,
        "max_retries": 8,
        "retry_backoff_seconds": 5.0,
        "min_request_interval_seconds": 3.0,
    },
    "output": {
        "output_dir": "outputs",
        "runfile_name": "r_output_trec_rag_2026.tsv",
        "rag_output_name": "rag_output_trec_rag_2026.jsonl",
        "validation_report_name": "retrieval_validation_report.json",
        "rag_validation_report_name": "rag_validation_report.json",
        "rag_proxy_metrics_name": "rag_proxy_metrics.json",
        "rag_citation_diagnostics_name": "rag_citation_diagnostics.json",
        "rag_viewer_name": "rag_viewer.html",
        "rag_table_csv_name": "rag_outputs_table.csv",
        "rag_table_jsonl_name": "rag_outputs_table.jsonl",
    },
    "evaluation": {
        "qrels_path": None,
        "relevance_threshold": 1,
        "metrics": ["ndcg@10", "recall@100", "map", "mrr"],
    },
    "rag": {
        "enabled": False,
        "evidence_top_k": 5,
        "generator_provider": "anthropic_batch",
        "model": "claude-haiku-4-5-20251001",
        "prompt_template": (
            "Answer the topic using only the provided ClimbMix evidence. Break the answer "
            "into concise sentences and cite each sentence with the supporting reference "
            "indices."
        ),
        "max_output_tokens": 800,
    },
    "wandb": {
        "project": "trec26-rag-glhf",
        "entity": None,
        "mode": "online",
        "tags": ["baseline", "retrieval", "climbmix-400b", "dev"],
    },
    "optimization": {
        "objective_metric": "candidate_count_mean",
        "objective_direction": "maximize",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


def write_config(config: dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)


def env_or_config(value: Any, env_value: str | None) -> Any:
    return env_value if env_value not in (None, "") else value
