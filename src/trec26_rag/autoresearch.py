from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

from .config import load_config
from .experiment_optimizer import RunRecord, fetch_wandb_runs, propose_next_config, select_best_run
from .pyserini_client import load_env_file


SECRET_KEY_PATTERN = re.compile(
    r"(^token$|[_-]token$|api[_-]?key|secret|password|credential)",
    re.IGNORECASE,
)
SAFE_PLACEHOLDER_VALUES = {"", "null", "none", "changeme", "placeholder", "..."}


@dataclass(frozen=True)
class AutoresearchPolicy:
    raw: dict[str, Any]
    path: Path

    @property
    def allowed_paths(self) -> list[str]:
        return [str(path) for path in self.raw.get("allowed_paths", [])]

    @property
    def approved_config_keys(self) -> set[str]:
        return {str(key) for key in self.raw.get("approved_config_keys", [])}

    @property
    def routes(self) -> dict[str, dict[str, Any]]:
        routes = self.raw.get("routes", {})
        if not isinstance(routes, dict):
            raise ValueError("autoresearch routes must be a mapping")
        return routes

    @property
    def objective(self) -> dict[str, Any]:
        objective = self.raw.get("objective", {})
        return objective if isinstance(objective, dict) else {}

    @property
    def review_requires_pr(self) -> bool:
        review = self.raw.get("review", {})
        return bool(isinstance(review, dict) and review.get("require_pr", True))

    @property
    def github_repository(self) -> str:
        github = self.raw.get("github", {})
        if not isinstance(github, dict) or not github.get("repository"):
            raise ValueError("autoresearch.github.repository is required")
        return str(github["repository"])

    @property
    def github_base_branch(self) -> str:
        github = self.raw.get("github", {})
        return str(github.get("base_branch") or "main") if isinstance(github, dict) else "main"


@dataclass(frozen=True)
class RouteDecision:
    name: str
    task: str
    workflow: str
    base_config: str
    default_limit: str
    mode: str


@dataclass(frozen=True)
class GitHubWorkflowRun:
    id: int
    name: str
    status: str
    conclusion: str | None
    html_url: str | None
    head_branch: str | None
    head_sha: str | None


def load_autoresearch_policy(path: str | Path = "configs/autoresearch.yaml") -> AutoresearchPolicy:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Autoresearch policy must be a mapping: {policy_path}")
    policy = AutoresearchPolicy(raw=raw, path=policy_path)
    validate_policy(policy)
    return policy


def validate_policy(policy: AutoresearchPolicy) -> None:
    if not policy.allowed_paths:
        raise ValueError("autoresearch.allowed_paths must include at least one path")
    required_routes = {"retrieval", "rag", "evaluation-only", "proposer-only"}
    missing_routes = sorted(required_routes - set(policy.routes))
    if missing_routes:
        raise ValueError(f"autoresearch routes missing: {', '.join(missing_routes)}")
    for name, route in policy.routes.items():
        if not isinstance(route, dict):
            raise ValueError(f"autoresearch route {name} must be a mapping")
        if not route.get("workflow"):
            raise ValueError(f"autoresearch route {name} missing workflow")
        if not route.get("base_config"):
            raise ValueError(f"autoresearch route {name} missing base_config")


def route_for(policy: AutoresearchPolicy, route_name: str) -> RouteDecision:
    route = policy.routes.get(route_name)
    if route is None:
        raise ValueError(f"Unknown autoresearch route: {route_name}")
    return RouteDecision(
        name=route_name,
        task=str(route.get("task") or route_name),
        workflow=str(route["workflow"]),
        base_config=str(route["base_config"]),
        default_limit=str(route.get("default_limit") or ""),
        mode=str(route.get("mode") or "experiment"),
    )


def is_path_allowed(path: str | Path, allowed_paths: Iterable[str]) -> bool:
    normalized = Path(path).as_posix().lstrip("./")
    for allowed in allowed_paths:
        allowed_text = str(allowed)
        allowed_is_dir = allowed_text.endswith("/")
        allowed_normalized = Path(allowed_text).as_posix().lstrip("./")
        if allowed_is_dir:
            allowed_dir = allowed_normalized.rstrip("/")
            if normalized == allowed_dir or normalized.startswith(f"{allowed_dir}/"):
                return True
        elif normalized == allowed_normalized:
            return True
    return False


def validate_changed_paths(paths: Iterable[str | Path], policy: AutoresearchPolicy) -> list[str]:
    violations = [
        str(path)
        for path in paths
        if not is_path_allowed(path, policy.allowed_paths)
    ]
    return violations


def flattened_config_keys(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix} if prefix else set()
    keys: set[str] = set()
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            keys.update(flattened_config_keys(child, child_prefix))
        else:
            keys.add(child_prefix)
    return keys


def find_unapproved_config_keys(config: dict[str, Any], policy: AutoresearchPolicy) -> list[str]:
    approved = policy.approved_config_keys
    if not approved:
        return []
    return sorted(key for key in flattened_config_keys(config) if key not in approved)


def changed_config_keys(base: Any, proposal: Any, prefix: str = "") -> set[str]:
    if isinstance(base, dict) and isinstance(proposal, dict):
        keys: set[str] = set()
        for key in sorted(set(base) | set(proposal)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            keys.update(changed_config_keys(base.get(key), proposal.get(key), child_prefix))
        return keys
    if base != proposal:
        return {prefix} if prefix else set()
    return set()


def validate_config_delta(
    base_config_path: str | Path,
    proposal_path: str | Path,
    policy: AutoresearchPolicy,
) -> list[str]:
    base_config = load_config(base_config_path)
    proposal_config = load_config(proposal_path)
    approved = policy.approved_config_keys
    changed_keys = changed_config_keys(base_config, proposal_config)
    unapproved = sorted(key for key in changed_keys if key not in approved)
    if not unapproved:
        return []
    return ["proposal changes unapproved config key(s): " + ", ".join(unapproved)]


def find_secret_like_values(value: Any, prefix: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if SECRET_KEY_PATTERN.search(str(key)) and child is not None:
                rendered = str(child).strip()
                if rendered.lower() not in SAFE_PLACEHOLDER_VALUES:
                    violations.append(child_prefix)
            violations.extend(find_secret_like_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(find_secret_like_values(child, f"{prefix}[{index}]"))
    return violations


def validate_proposal_file(path: str | Path, policy: AutoresearchPolicy) -> list[str]:
    proposal_path = Path(path)
    errors: list[str] = []
    if not is_path_allowed(proposal_path, policy.allowed_paths):
        errors.append(f"proposal path is not allowed: {proposal_path}")
    if proposal_path.suffix not in {".yaml", ".yml"}:
        errors.append(f"proposal must be YAML: {proposal_path}")
    if not proposal_path.exists():
        errors.append(f"proposal does not exist: {proposal_path}")
        return errors
    config = load_config(proposal_path)
    secret_paths = find_secret_like_values(config)
    if secret_paths:
        errors.append("proposal contains secret-like fields: " + ", ".join(secret_paths))
    return errors


def valid_runs_for_objective(runs: list[RunRecord], metric: str) -> list[RunRecord]:
    valid: list[RunRecord] = []
    for run in runs:
        has_metric = run.summary.get(metric) is not None
        retrieval_valid = run.summary.get("validation_error_count", 0) == 0
        rag_valid = run.summary.get("rag_validation_error_count", 0) == 0
        if has_metric and retrieval_valid and rag_valid:
            valid.append(run)
    return valid


def select_current_best_run(policy: AutoresearchPolicy, runs: list[RunRecord]) -> RunRecord | None:
    objective = policy.objective
    metric = str(objective.get("metric") or "candidate_count_mean")
    direction = str(objective.get("direction") or "maximize")
    best = select_best_run(valid_runs_for_objective(runs, metric), metric, direction)
    if best is not None:
        return best
    fallback_metric = objective.get("fallback_metric")
    if not fallback_metric:
        return None
    fallback_direction = str(objective.get("fallback_direction") or direction)
    return select_best_run(
        valid_runs_for_objective(runs, str(fallback_metric)),
        str(fallback_metric),
        fallback_direction,
    )


def summarize_best_run(policy: AutoresearchPolicy, runs: list[RunRecord]) -> dict[str, Any]:
    objective = policy.objective
    metric = str(objective.get("metric") or "candidate_count_mean")
    direction = str(objective.get("direction") or "maximize")
    best = select_best_run(valid_runs_for_objective(runs, metric), metric, direction)
    used_metric = metric
    used_direction = direction
    used_fallback = False
    if best is None and objective.get("fallback_metric"):
        used_metric = str(objective["fallback_metric"])
        used_direction = str(objective.get("fallback_direction") or direction)
        best = select_best_run(valid_runs_for_objective(runs, used_metric), used_metric, used_direction)
        used_fallback = True
    if best is None:
        return {
            "metric": used_metric,
            "direction": used_direction,
            "used_fallback": used_fallback,
            "best_run": None,
        }
    return {
        "metric": used_metric,
        "direction": used_direction,
        "used_fallback": used_fallback,
        "best_run": {
            "id": best.id,
            "name": best.name,
            "url": best.url,
            "value": best.summary.get(used_metric),
            "tags": best.tags,
        },
    }


def fetch_policy_wandb_runs(policy: AutoresearchPolicy) -> list[RunRecord]:
    load_env_file()
    wandb_config = policy.raw.get("wandb", {})
    if not isinstance(wandb_config, dict):
        wandb_config = {}
    project = os.environ.get("WANDB_PROJECT") or wandb_config.get("project")
    entity = os.environ.get("WANDB_ENTITY") or wandb_config.get("entity")
    max_runs = int(wandb_config.get("max_runs") or 25)
    if not project:
        raise RuntimeError("W&B project is missing. Set WANDB_PROJECT or autoresearch.wandb.project.")
    return fetch_wandb_runs(project=project, entity=entity, max_runs=max_runs)


def propose_config_for_route(
    policy: AutoresearchPolicy,
    route_name: str,
    runs: list[RunRecord] | None = None,
    output_dir: str | Path = "configs/experiments",
    base_config_path: str | Path | None = None,
) -> Path:
    decision = route_for(policy, route_name)
    if decision.mode == "proposer_only":
        decision = route_for(policy, "retrieval")
    proposal_runs = fetch_policy_wandb_runs(policy) if runs is None else runs
    base_config = load_config(base_config_path or decision.base_config)
    proposal = propose_next_config(base_config, proposal_runs)
    proposal["experiment"]["task"] = decision.task
    output_path = Path(output_dir) / f"{proposal['experiment']['name']}.yaml"
    from .config import write_config

    write_config(proposal, output_path)
    errors = validate_proposal_file(output_path, policy)
    errors.extend(validate_config_delta(base_config_path or decision.base_config, output_path, policy))
    if errors:
        raise ValueError("Unsafe autoresearch proposal: " + "; ".join(errors))
    return output_path


class GitHubActionsClient:
    def __init__(self, repository: str, token: str | None = None) -> None:
        self.repository = repository
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required for GitHub Actions API calls.")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def dispatch_workflow(
        self,
        workflow: str,
        ref: str,
        inputs: dict[str, str] | None = None,
    ) -> None:
        url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{workflow}/dispatches"
        response = requests.post(
            url,
            headers=self.headers,
            json={"ref": ref, "inputs": inputs or {}},
            timeout=30,
        )
        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(
                f"GitHub workflow dispatch failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

    def list_workflow_runs(
        self,
        workflow: str,
        branch: str | None = None,
        per_page: int = 10,
    ) -> list[GitHubWorkflowRun]:
        url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{workflow}/runs"
        params: dict[str, Any] = {"per_page": per_page}
        if branch:
            params["branch"] = branch
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub workflow runs lookup failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        return [
            GitHubWorkflowRun(
                id=int(run["id"]),
                name=str(run.get("name") or workflow),
                status=str(run.get("status") or "unknown"),
                conclusion=run.get("conclusion"),
                html_url=run.get("html_url"),
                head_branch=run.get("head_branch"),
                head_sha=run.get("head_sha"),
            )
            for run in payload.get("workflow_runs", [])
        ]


def dispatch_route(
    policy: AutoresearchPolicy,
    route_name: str,
    config_path: str,
    ref: str | None = None,
    limit: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    decision = route_for(policy, route_name)
    errors = validate_proposal_file(config_path, policy)
    if route_name in {"retrieval", "rag", "evaluation-only"}:
        errors.extend(validate_config_delta(decision.base_config, config_path, policy))
    if route_name in {"retrieval", "rag", "evaluation-only"} and errors:
        raise ValueError("Unsafe config path for dispatch: " + "; ".join(errors))
    client = GitHubActionsClient(policy.github_repository, token=token)
    workflow_inputs = {"config": config_path}
    selected_limit = decision.default_limit if limit is None else str(limit)
    if selected_limit:
        workflow_inputs["limit"] = selected_limit
    client.dispatch_workflow(
        workflow=decision.workflow,
        ref=ref or policy.github_base_branch,
        inputs=workflow_inputs,
    )
    return {
        "route": route_name,
        "workflow": decision.workflow,
        "ref": ref or policy.github_base_branch,
        "inputs": workflow_inputs,
    }


def summarize_runs(runs: list[GitHubWorkflowRun]) -> dict[str, Any]:
    if not runs:
        return {"status": "missing", "summary": "No workflow runs found."}
    latest = runs[0]
    return {
        "status": latest.status,
        "conclusion": latest.conclusion,
        "run_id": latest.id,
        "url": latest.html_url,
        "summary": (
            f"{latest.name} on {latest.head_branch or 'unknown branch'} is "
            f"{latest.status}"
            + (f" / {latest.conclusion}" if latest.conclusion else "")
            + "."
        ),
    }


def build_autoresearch_summary(
    policy: AutoresearchPolicy,
    route_name: str,
    workflow_summary: dict[str, Any],
    wandb_runs: list[RunRecord] | None = None,
) -> dict[str, Any]:
    summary = {
        "route": route_name,
        "workflow": workflow_summary,
        "objective": {
            "metric": policy.objective.get("metric"),
            "direction": policy.objective.get("direction"),
            "fallback_metric": policy.objective.get("fallback_metric"),
            "fallback_direction": policy.objective.get("fallback_direction"),
        },
    }
    if wandb_runs is not None:
        summary["current_best"] = summarize_best_run(policy, wandb_runs)
    return summary


def log_autoresearch_summary(policy: AutoresearchPolicy, summary: dict[str, Any]) -> str | None:
    load_env_file()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is not installed. Run `python -m pip install -e .`.") from exc

    wandb_config = policy.raw.get("wandb", {})
    if not isinstance(wandb_config, dict):
        wandb_config = {}
    project = os.environ.get("WANDB_PROJECT") or wandb_config.get("project")
    entity = os.environ.get("WANDB_ENTITY") or wandb_config.get("entity")
    mode = os.environ.get("WANDB_MODE") or wandb_config.get("mode") or "online"
    if not project:
        raise RuntimeError("W&B project is missing. Set WANDB_PROJECT or autoresearch.wandb.project.")

    workflow = summary.get("workflow", {})
    current_best = summary.get("current_best", {})
    best_run = current_best.get("best_run") if isinstance(current_best, dict) else None
    metrics: dict[str, Any] = {
        "autoresearch_workflow_completed": 1 if workflow.get("status") == "completed" else 0,
        "autoresearch_workflow_success": 1 if workflow.get("conclusion") == "success" else 0,
    }
    if isinstance(best_run, dict) and best_run.get("value") is not None:
        try:
            metrics["autoresearch_best_objective_value"] = float(best_run["value"])
        except (TypeError, ValueError):
            pass

    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=f"autoresearch_{summary.get('route', 'unknown')}_{workflow.get('run_id', 'latest')}",
        config={"autoresearch": summary},
        tags=["autoresearch", str(summary.get("route", "unknown"))],
    )
    try:
        wandb.log(metrics)
        return run.url
    finally:
        wandb.finish()


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
