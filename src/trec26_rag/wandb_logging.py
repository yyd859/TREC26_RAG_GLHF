from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .pyserini_client import load_env_file


def log_retrieval_run(
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path],
) -> str | None:
    return _log_run(
        config=config,
        metrics=metrics,
        artifacts=artifacts,
        artifact_type="retrieval-run",
        task="retrieval",
    )


def log_rag_run(
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path],
    tables: dict[str, dict[str, Any]] | None = None,
    htmls: dict[str, str | Path] | None = None,
) -> str | None:
    artifact_metadata = {
        "valid_output": bool(metrics.get("rag_proxy_valid_output", 0)),
        "validation_error_count": metrics.get("rag_validation_error_count", 0),
        "validation_warning_count": metrics.get("rag_validation_warning_count", 0),
        "proxy": {
            "response_rate": metrics.get("rag_proxy_response_rate", 0.0),
            "evidence_docs_mean": metrics.get("rag_proxy_evidence_docs_mean", 0.0),
            "answer_words_mean": metrics.get("rag_proxy_answer_words_mean", 0.0),
        },
        "citation": {
            "coverage_mean": metrics.get("rag_proxy_citation_coverage_mean", 0.0),
            "density_mean": metrics.get("rag_proxy_citation_density_mean", 0.0),
            "uncited_reference_rate": metrics.get("rag_proxy_uncited_reference_rate", 0.0),
            "invalid_citation_rate": metrics.get("rag_proxy_invalid_citation_rate", 0.0),
        },
    }
    return _log_run(
        config=config,
        metrics=metrics,
        artifacts=artifacts,
        artifact_type="rag-run",
        task="rag",
        artifact_metadata=artifact_metadata,
        tables=tables,
        htmls=htmls,
    )


def _log_run(
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path],
    artifact_type: str,
    task: str,
    artifact_metadata: dict[str, Any] | None = None,
    tables: dict[str, dict[str, Any]] | None = None,
    htmls: dict[str, str | Path] | None = None,
) -> str | None:
    load_env_file()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is not installed. Run `python -m pip install -e .`.") from exc

    wandb_config = config.get("wandb", {})
    project = os.environ.get("WANDB_PROJECT") or wandb_config.get("project")
    entity = os.environ.get("WANDB_ENTITY") or wandb_config.get("entity")
    mode = os.environ.get("WANDB_MODE") or wandb_config.get("mode") or "online"
    if not project:
        raise RuntimeError("W&B project is missing. Set WANDB_PROJECT or config.wandb.project.")

    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=config.get("experiment", {}).get("name"),
        config=config,
        tags=wandb_config.get("tags") or [],
    )
    try:
        wandb.log(metrics)
        for table_name, table_payload in (tables or {}).items():
            wandb.log(
                {
                    table_name: wandb.Table(
                        columns=table_payload["columns"],
                        data=table_payload["data"],
                    )
                }
            )
        for html_name, html_path in (htmls or {}).items():
            path = Path(html_path)
            if path.exists():
                wandb.log({html_name: wandb.Html(str(path), inject=False)})
        artifact = wandb.Artifact(
            name=f"{config.get('experiment', {}).get('run_id', artifact_type)}-outputs",
            type=artifact_type,
            metadata={
                "task": task,
                "track_year": 2026,
                **(artifact_metadata or {}),
            },
        )
        for artifact_path in artifacts:
            path = Path(artifact_path)
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        return run.url
    finally:
        wandb.finish()
