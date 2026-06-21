from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import deep_merge, load_config, write_config
from .pyserini_client import load_env_file


@dataclass(frozen=True)
class RunRecord:
    id: str
    name: str
    url: str | None
    config: dict[str, Any]
    summary: dict[str, Any]
    tags: list[str]


EXPERIMENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "suffix": "title_narrative_top100",
        "hypothesis": "Adding the narrative to the title query may improve recall over title-only retrieval.",
        "retrieval": {"query_template": "{title} {narrative}", "hits": 100},
    },
    {
        "suffix": "title_boosted_narrative_top100",
        "hypothesis": "Repeating the title while adding narrative may preserve precision while improving recall.",
        "retrieval": {"query_template": "{title} {title} {narrative}", "hits": 100},
    },
    {
        "suffix": "title_top200",
        "hypothesis": "Increasing retrieval depth may improve downstream evidence coverage.",
        "retrieval": {"query_template": "{title}", "hits": 200},
    },
    {
        "suffix": "title_boosted_narrative_top200",
        "hypothesis": "Combining title emphasis, narrative context, and deeper retrieval may improve evidence coverage.",
        "retrieval": {"query_template": "{title} {title} {narrative}", "hits": 200},
    },
]


def fetch_wandb_runs(project: str, entity: str | None = None, max_runs: int = 25) -> list[RunRecord]:
    load_env_file()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is not installed. Run `python -m pip install -e .`.") from exc

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    records: list[RunRecord] = []
    for run in api.runs(path, per_page=max_runs):
        summary = dict(run.summary)
        records.append(
            RunRecord(
                id=run.id,
                name=run.name,
                url=getattr(run, "url", None),
                config=dict(run.config),
                summary=summary,
                tags=list(run.tags or []),
            )
        )
    return records


def select_best_run(
    runs: list[RunRecord],
    metric: str,
    direction: str,
) -> RunRecord | None:
    valid_runs = [
        run
        for run in runs
        if run.summary.get("validation_error_count", 0) == 0 and run.summary.get(metric) is not None
    ]
    if not valid_runs:
        return None
    reverse = direction == "maximize"
    return sorted(valid_runs, key=lambda run: float(run.summary[metric]), reverse=reverse)[0]


def config_signature(config: dict[str, Any]) -> tuple[str, int]:
    retrieval = config.get("retrieval", {})
    return (str(retrieval.get("query_template")), int(retrieval.get("hits") or 0))


def propose_next_config(
    base_config: dict[str, Any],
    runs: list[RunRecord],
    now: datetime | None = None,
) -> dict[str, Any]:
    optimization = base_config.get("optimization", {})
    metric = optimization.get("objective_metric", "candidate_count_mean")
    direction = optimization.get("objective_direction", "maximize")
    best_run = select_best_run(runs, metric, direction)
    parent_config = deep_merge(base_config, best_run.config) if best_run else base_config
    existing_signatures = {config_signature(run.config) for run in runs}
    existing_signatures.add(config_signature(base_config))

    selected_template = None
    for template in EXPERIMENT_TEMPLATES:
        candidate = deep_merge(parent_config, {"retrieval": template["retrieval"]})
        if config_signature(candidate) not in existing_signatures:
            selected_template = template
            break
    if selected_template is None:
        selected_template = {
            "suffix": "title_narrative_top300",
            "hypothesis": "The first template set is exhausted; try a deeper title plus narrative retrieval pool.",
            "retrieval": {"query_template": "{title} {narrative}", "hits": 300},
        }

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    suffix = selected_template["suffix"]
    run_id = f"glhf-{suffix.replace('_', '-')}"
    proposal = deep_merge(parent_config, {"retrieval": selected_template["retrieval"]})
    proposal = deep_merge(
        proposal,
        {
            "experiment": {
                "name": f"{timestamp}_{suffix}",
                "hypothesis": selected_template["hypothesis"],
                "run_id": run_id,
                "parent_wandb_run_id": best_run.id if best_run else None,
                "parent_wandb_run_url": best_run.url if best_run else None,
            },
            "wandb": {
                "tags": sorted(set((parent_config.get("wandb", {}).get("tags") or []) + ["proposed"])),
            },
        },
    )
    return proposal


def write_next_experiment_config(
    base_config_path: str | Path,
    output_dir: str | Path,
    runs: list[RunRecord] | None = None,
    max_runs: int = 25,
) -> Path:
    base_config = load_config(base_config_path)
    if runs is None:
        wandb_config = base_config.get("wandb", {})
        project = os.environ.get("WANDB_PROJECT") or wandb_config.get("project")
        entity = os.environ.get("WANDB_ENTITY") or wandb_config.get("entity")
        if not project:
            raise RuntimeError("W&B project is missing. Set WANDB_PROJECT or config.wandb.project.")
        runs = fetch_wandb_runs(project=project, entity=entity, max_runs=max_runs)
    proposal = propose_next_config(base_config, runs)
    output_path = Path(output_dir) / f"{proposal['experiment']['name']}.yaml"
    write_config(proposal, output_path)
    return output_path
