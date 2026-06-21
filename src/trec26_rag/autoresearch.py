from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

from .config import load_config, write_config
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
    def branch_prefix(self) -> str:
        branching = self.raw.get("branching", {})
        if isinstance(branching, dict) and branching.get("experiment_branch_prefix"):
            return str(branching["experiment_branch_prefix"])
        review = self.raw.get("review", {})
        if isinstance(review, dict) and review.get("pr_branch_prefix"):
            return str(review["pr_branch_prefix"])
        return "codex/autoresearch-"

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

    @property
    def branch_push_triggers_enabled(self) -> bool:
        triggers = self.raw.get("branch_triggers", {})
        return bool(isinstance(triggers, dict) and triggers.get("enabled", False))

    @property
    def research_memory_path(self) -> Path:
        memory = self.raw.get("research_memory", {})
        if isinstance(memory, dict) and memory.get("path"):
            return Path(str(memory["path"]))
        return Path("outputs/autoresearch_memory.json")


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


@dataclass(frozen=True)
class GitHubPullRequest:
    number: int
    title: str
    html_url: str
    state: str
    draft: bool


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


def compact_run_record(run: RunRecord) -> dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    return {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "tags": run.tags,
        "experiment": config.get("experiment", {}),
        "retrieval": config.get("retrieval", {}),
        "rag": config.get("rag", {}),
        "evaluation": config.get("evaluation", {}),
        "wandb": {"tags": (config.get("wandb", {}) or {}).get("tags", [])},
        "summary": {
            key: value
            for key, value in run.summary.items()
            if key
            in {
                "ndcg@10",
                "recall@100",
                "map",
                "mrr",
                "candidate_count_mean",
                "candidate_count_min",
                "candidate_count_max",
                "duplicate_doc_rate",
                "empty_topic_count",
                "latency_mean_seconds",
                "validation_error_count",
                "rag_validation_error_count",
                "rag_reference_coverage",
                "rag_answer_count",
                "rag_citation_error_count",
            }
        },
    }


def collect_experiment_configs(config_dir: str | Path = "configs/experiments") -> list[dict[str, Any]]:
    root = Path(config_dir)
    if not root.exists():
        return []
    configs: list[dict[str, Any]] = []
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            config = load_config(path)
        except Exception as exc:  # pragma: no cover - defensive context building
            configs.append({"path": path.as_posix(), "error": str(exc)})
            continue
        configs.append(
            {
                "path": path.as_posix(),
                "experiment": config.get("experiment", {}),
                "retrieval": config.get("retrieval", {}),
                "rag": config.get("rag", {}),
                "evaluation": config.get("evaluation", {}),
                "wandb": {"tags": (config.get("wandb", {}) or {}).get("tags", [])},
            }
        )
    return configs


def load_research_memory(path: str | Path) -> dict[str, Any]:
    memory_path = Path(path)
    if not memory_path.exists():
        return {"version": 1, "decisions": [], "updated_at": None}
    with memory_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Autoresearch memory must be a JSON object: {memory_path}")
    payload.setdefault("version", 1)
    payload.setdefault("decisions", [])
    return payload


def save_research_memory(path: str | Path, memory: dict[str, Any]) -> Path:
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = datetime.now(timezone.utc).isoformat()
    with memory_path.open("w", encoding="utf-8") as handle:
        json.dump(memory, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return memory_path


def append_research_decision(
    memory: dict[str, Any],
    *,
    route_name: str,
    proposal_path: str | Path | None = None,
    branch: str | None = None,
    workflow_summary: dict[str, Any] | None = None,
    wandb_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions = memory.setdefault("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
        memory["decisions"] = decisions
    decisions.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "route": route_name,
            "proposal": None if proposal_path is None else Path(proposal_path).as_posix(),
            "branch": branch,
            "workflow": workflow_summary or {},
            "wandb": wandb_summary or {},
        }
    )
    return memory


def config_signature_text(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}
    retrieval = config.get("retrieval", {}) if isinstance(config.get("retrieval"), dict) else {}
    rag = config.get("rag", {}) if isinstance(config.get("rag"), dict) else {}
    return "|".join(
        [
            str(experiment.get("task") or ""),
            str(retrieval.get("query_template") or ""),
            str(retrieval.get("hits") or ""),
            str(rag.get("evidence_top_k") or ""),
            str(rag.get("max_output_tokens") or ""),
        ]
    )


def build_research_context(
    policy: AutoresearchPolicy,
    runs: list[RunRecord] | None = None,
    config_dir: str | Path = "configs/experiments",
    memory_path: str | Path | None = None,
) -> dict[str, Any]:
    wandb_runs = fetch_policy_wandb_runs(policy) if runs is None else runs
    historical_configs = collect_experiment_configs(config_dir)
    memory = load_research_memory(memory_path or policy.research_memory_path)
    compact_runs = [compact_run_record(run) for run in wandb_runs]
    tried_signatures = sorted(
        {
            config_signature_text(run.config)
            for run in wandb_runs
            if isinstance(run.config, dict)
        }
        | {
            config_signature_text(config)
            for config in historical_configs
            if isinstance(config, dict) and "error" not in config
        }
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "metric": policy.objective.get("metric"),
            "direction": policy.objective.get("direction"),
            "fallback_metric": policy.objective.get("fallback_metric"),
            "fallback_direction": policy.objective.get("fallback_direction"),
        },
        "routes": policy.routes,
        "current_best": summarize_best_run(policy, wandb_runs),
        "historical_configs": historical_configs,
        "wandb_runs": compact_runs,
        "tried_config_signatures": tried_signatures,
        "research_memory": memory,
        "agent_instruction": (
            "Use this whole context, not only current_best, to choose the next "
            "safe config-only experiment under configs/experiments/."
        ),
    }


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
    write_config(proposal, output_path)
    errors = validate_proposal_file(output_path, policy)
    errors.extend(validate_config_delta(base_config_path or decision.base_config, output_path, policy))
    if errors:
        raise ValueError("Unsafe autoresearch proposal: " + "; ".join(errors))
    return output_path


def apply_runtime_limit(proposal_path: str | Path, limit: str | None) -> None:
    if limit in (None, ""):
        return
    config = load_config(proposal_path)
    runtime = config.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
        config["runtime"] = runtime
    runtime["limit"] = str(limit)
    write_config(config, proposal_path)


def slugify_branch_component(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug or "experiment")[:max_length].rstrip("-._")


def experiment_branch_name(policy: AutoresearchPolicy, proposal_path: str | Path) -> str:
    stem = Path(proposal_path).stem
    return f"{policy.branch_prefix}{slugify_branch_component(stem)}"


def run_git(args: list[str], cwd: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def commit_and_push_experiment_branch(
    policy: AutoresearchPolicy,
    proposal_path: str | Path,
    branch_name: str | None = None,
    base_ref: str | None = None,
) -> dict[str, Any]:
    proposal = Path(proposal_path)
    branch = branch_name or experiment_branch_name(policy, proposal)
    if base_ref:
        run_git(["fetch", "origin", base_ref])
        run_git(["checkout", "-B", branch, f"origin/{base_ref}"])
    else:
        run_git(["checkout", "-B", branch])
    run_git(["add", proposal.as_posix()])
    run_git(["commit", "-m", f"Add autoresearch config {proposal.stem}"])
    run_git(["push", "-u", "origin", branch])
    return {"branch": branch, "proposal": proposal.as_posix()}


def run_branch_iteration(
    policy: AutoresearchPolicy,
    route_name: str,
    ref: str | None = None,
    limit: str | None = None,
    output_dir: str | Path = "configs/experiments",
    token: str | None = None,
    dispatch: bool | None = None,
) -> dict[str, Any]:
    decision = route_for(policy, route_name)
    proposal_path = propose_config_for_route(
        policy=policy,
        route_name=route_name,
        output_dir=output_dir,
    )
    apply_runtime_limit(proposal_path, limit)
    errors = validate_proposal_file(proposal_path, policy)
    if route_name in {"retrieval", "rag", "evaluation-only"}:
        errors.extend(validate_config_delta(decision.base_config, proposal_path, policy))
    if errors:
        raise ValueError("Unsafe autoresearch branch iteration: " + "; ".join(errors))
    branch_result = commit_and_push_experiment_branch(
        policy=policy,
        proposal_path=proposal_path,
        base_ref=ref or policy.github_base_branch,
    )
    should_dispatch = (not policy.branch_push_triggers_enabled) if dispatch is None else dispatch
    dispatch_payload = build_dispatch_payload(
        policy=policy,
        route_name=route_name,
        config_path=proposal_path.as_posix(),
        ref=branch_result["branch"],
        limit=limit,
    )
    if should_dispatch:
        dispatch_payload = dispatch_route(
            policy=policy,
            route_name=route_name,
            config_path=proposal_path.as_posix(),
            ref=branch_result["branch"],
            limit=limit,
            token=token,
        )
        trigger = "workflow_dispatch"
    else:
        trigger = "branch_push"
    return {
        "route": route_name,
        "proposal": proposal_path.as_posix(),
        "branch": branch_result["branch"],
        "workflow": dispatch_payload["workflow"],
        "trigger": trigger,
        "dispatch": dispatch_payload,
    }


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

    def create_pull_request(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = True,
    ) -> GitHubPullRequest:
        url = f"https://api.github.com/repos/{self.repository}/pulls"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "head": head,
                "base": base,
                "title": title,
                "body": body,
                "draft": draft,
                "maintainer_can_modify": True,
            },
            timeout=30,
        )
        if response.status_code not in {200, 201}:
            raise RuntimeError(
                f"GitHub PR creation failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        return GitHubPullRequest(
            number=int(payload["number"]),
            title=str(payload.get("title") or title),
            html_url=str(payload["html_url"]),
            state=str(payload.get("state") or "open"),
            draft=bool(payload.get("draft", draft)),
        )


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
    payload = build_dispatch_payload(policy, route_name, config_path, ref=ref, limit=limit)
    client.dispatch_workflow(
        workflow=payload["workflow"],
        ref=payload["ref"],
        inputs=payload["inputs"],
    )
    return payload


def build_dispatch_payload(
    policy: AutoresearchPolicy,
    route_name: str,
    config_path: str,
    ref: str | None = None,
    limit: str | None = None,
) -> dict[str, Any]:
    decision = route_for(policy, route_name)
    workflow_inputs = {"config": config_path}
    selected_limit = decision.default_limit if limit is None else str(limit)
    if selected_limit:
        workflow_inputs["limit"] = selected_limit
    return {
        "route": route_name,
        "workflow": decision.workflow,
        "ref": ref or policy.github_base_branch,
        "inputs": workflow_inputs,
    }


def route_name_for_config(policy: AutoresearchPolicy, config_path: str | Path) -> str:
    config = load_config(config_path)
    experiment = config.get("experiment", {})
    task = str(experiment.get("task") or "").strip().lower() if isinstance(experiment, dict) else ""
    rag = config.get("rag", {})
    rag_enabled = bool(rag.get("enabled")) if isinstance(rag, dict) else False
    if task == "rag" or rag_enabled:
        return "rag"
    if task in {"retrieval", "evaluation"}:
        return "retrieval"
    for route_name, route in policy.routes.items():
        if str(route.get("task") or "").strip().lower() == task:
            return route_name
    return "retrieval"


def workflow_limit_for_config(policy: AutoresearchPolicy, config_path: str | Path) -> str:
    config = load_config(config_path)
    runtime = config.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("limit") not in (None, ""):
        return str(runtime["limit"])
    return route_for(policy, route_name_for_config(policy, config_path)).default_limit


def latest_changed_experiment_config(
    policy: AutoresearchPolicy,
    route_name: str | None = None,
    ref: str = "HEAD",
) -> Path:
    candidates: list[Path] = []
    try:
        output = run_git(["show", "--name-only", "--format=", ref])
        candidates = [
            Path(line.strip())
            for line in output.splitlines()
            if line.strip().startswith("configs/experiments/")
            and Path(line.strip()).suffix in {".yaml", ".yml"}
            and Path(line.strip()).exists()
        ]
    except subprocess.CalledProcessError:
        candidates = []
    if not candidates:
        root = Path("configs/experiments")
        candidates = (
            sorted(
                [*root.rglob("*.yaml"), *root.rglob("*.yml")],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if root.exists()
            else []
        )
    if route_name:
        candidates = [
            path
            for path in candidates
            if route_name_for_config(policy, path) == route_name
        ]
    if not candidates:
        route_text = f" for route {route_name}" if route_name else ""
        raise FileNotFoundError(f"No experiment config found{route_text}.")
    return candidates[0]


def build_autoresearch_pr_body() -> str:
    return (
        "## Summary\n\n"
        "Adds the first review-gated autoresearch orchestration layer inspired by "
        "karpathy/autoresearch, adapted to this repo's Level 2 config-only workflow.\n\n"
        "- Adds `configs/autoresearch.yaml` with runner tradeoffs, route policy, "
        "allowed paths, objective config, and PR review guardrails.\n"
        "- Adds `scripts/autoresearch.py` plus `trec26_rag.autoresearch` for routes, "
        "W&B best-run lookup, safe config proposal generation, workflow dispatch, "
        "monitor summaries, and optional W&B autoresearch logging.\n"
        "- Adds the `Autoresearch Orchestrator` GitHub workflow for "
        "propose/dispatch/monitor operations.\n"
        "- Documents operating instructions in README and AGENTS.\n"
        "- Adds tests for routing, safety constraints, proposal generation, "
        "objective selection, and summary construction.\n\n"
        "## Verification\n\n"
        "- `PYTHONPATH=src python -m unittest discover -s tests`\n"
        "- `PYTHONPATH=src python scripts/autoresearch.py check --route retrieval "
        "configs/experiments/`\n\n"
        "## Bootstrap note\n\n"
        "After this lands on `main`, the new `Autoresearch Orchestrator` workflow "
        "can be manually dispatched from the GitHub Actions UI."
    )


def open_autoresearch_bootstrap_pr(
    policy: AutoresearchPolicy,
    head: str = "codex/autoresearch-v1",
    base: str | None = None,
    title: str = "Add autoresearch orchestrator v1",
    body: str | None = None,
    draft: bool = True,
    token: str | None = None,
    client: GitHubActionsClient | None = None,
) -> GitHubPullRequest:
    github = client or GitHubActionsClient(policy.github_repository, token=token)
    return github.create_pull_request(
        head=head,
        base=base or policy.github_base_branch,
        title=title,
        body=body or build_autoresearch_pr_body(),
        draft=draft,
    )


def compare_url(policy: AutoresearchPolicy, head: str, base: str | None = None) -> str:
    return f"https://github.com/{policy.github_repository}/compare/{base or policy.github_base_branch}...{head}"


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


def log_research_memory(policy: AutoresearchPolicy, memory_path: str | Path) -> str | None:
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

    memory_file = Path(memory_path)
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=f"autoresearch_memory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        tags=["autoresearch", "memory"],
    )
    try:
        artifact = wandb.Artifact("autoresearch-memory", type="autoresearch-memory")
        artifact.add_file(memory_file.as_posix())
        run.log_artifact(artifact)
        return run.url
    finally:
        wandb.finish()


def run_autoresearch_loop(
    policy: AutoresearchPolicy,
    route_name: str,
    max_rounds: int,
    poll_seconds: int,
    limit: str | None = None,
    ref: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _round in range(max_rounds):
        iteration = run_branch_iteration(
            policy=policy,
            route_name=route_name,
            ref=ref,
            limit=limit,
            token=token,
        )
        branch = str(iteration["branch"])
        decision = route_for(policy, route_name)
        client = GitHubActionsClient(policy.github_repository, token=token)
        workflow_summary: dict[str, Any] = {"status": "missing", "summary": "No workflow run observed yet."}
        while True:
            workflow_summary = summarize_runs(
                client.list_workflow_runs(decision.workflow, branch=branch, per_page=1)
            )
            if workflow_summary.get("status") == "completed":
                break
            time.sleep(poll_seconds)
        wandb_runs = fetch_policy_wandb_runs(policy)
        summary = build_autoresearch_summary(policy, route_name, workflow_summary, wandb_runs)
        run_url = log_autoresearch_summary(policy, summary)
        if run_url:
            summary["autoresearch_wandb_run_url"] = run_url
        memory = load_research_memory(policy.research_memory_path)
        append_research_decision(
            memory,
            route_name=route_name,
            proposal_path=str(iteration["proposal"]),
            branch=branch,
            workflow_summary=workflow_summary,
            wandb_summary=summary.get("current_best", {}),
        )
        memory_path = save_research_memory(policy.research_memory_path, memory)
        log_research_memory(policy, memory_path)
        iteration["summary"] = summary
        iteration["memory_path"] = memory_path.as_posix()
        results.append(iteration)
    return results


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
