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
        artifact = wandb.Artifact(
            name=f"{config.get('experiment', {}).get('run_id', 'retrieval-run')}-outputs",
            type="retrieval-run",
            metadata={"task": "retrieval", "track_year": 2026},
        )
        for artifact_path in artifacts:
            path = Path(artifact_path)
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        return run.url
    finally:
        wandb.finish()
