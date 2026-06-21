#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from trec26_rag.autoresearch import (
    GitHubActionsClient,
    build_autoresearch_summary,
    dispatch_route,
    dumps_json,
    fetch_policy_wandb_runs,
    load_autoresearch_policy,
    log_autoresearch_summary,
    propose_config_for_route,
    route_for,
    select_current_best_run,
    summarize_runs,
    validate_changed_paths,
    validate_config_delta,
    validate_proposal_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch orchestration utilities.")
    parser.add_argument("--policy", default="configs/autoresearch.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    routes = subparsers.add_parser("routes", help="Print configured experiment routes.")

    best = subparsers.add_parser("best-run", help="Read W&B and print the current best run.")
    best.add_argument("--max-runs", type=int)

    propose = subparsers.add_parser("propose", help="Generate a safe config-only proposal.")
    propose.add_argument("--route", default="retrieval")
    propose.add_argument("--base-config")
    propose.add_argument("--output-dir", default="configs/experiments")

    check = subparsers.add_parser("check", help="Check proposed paths/configs against policy.")
    check.add_argument("--route", default="retrieval")
    check.add_argument("--base-config")
    check.add_argument("paths", nargs="+")

    dispatch = subparsers.add_parser("dispatch", help="Dispatch the workflow for a reviewed config.")
    dispatch.add_argument("--route", required=True)
    dispatch.add_argument("--config", required=True)
    dispatch.add_argument("--ref")
    dispatch.add_argument("--limit")

    monitor = subparsers.add_parser("monitor", help="Summarize recent GitHub Actions status.")
    monitor.add_argument("--route", required=True)
    monitor.add_argument("--branch")
    monitor.add_argument("--per-page", type=int, default=5)
    monitor.add_argument("--include-wandb", action="store_true")
    monitor.add_argument("--log-wandb", action="store_true")

    # Touch subparser objects so linters do not mistake them as unused in simple scripts.
    _ = (routes,)
    return parser.parse_args()


def write_github_step_summary(summary: dict[str, object]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    workflow = summary.get("workflow", {})
    current_best = summary.get("current_best", {})
    best_run = current_best.get("best_run") if isinstance(current_best, dict) else None
    lines = [
        "## Autoresearch Monitor",
        "",
        f"- Route: `{summary.get('route', 'unknown')}`",
        f"- Status: `{workflow.get('status', 'unknown') if isinstance(workflow, dict) else 'unknown'}`",
        f"- Conclusion: `{workflow.get('conclusion', '') if isinstance(workflow, dict) else ''}`",
        f"- Run ID: `{workflow.get('run_id', '') if isinstance(workflow, dict) else ''}`",
        f"- URL: {workflow.get('url', '') if isinstance(workflow, dict) else ''}",
        "",
        str(workflow.get("summary", "") if isinstance(workflow, dict) else ""),
        "",
    ]
    if isinstance(best_run, dict):
        metric = current_best.get("metric") if isinstance(current_best, dict) else ""
        lines.extend(
            [
                "### Current Best W&B Run",
                "",
                f"- Metric: `{metric}`",
                f"- Value: `{best_run.get('value', '')}`",
                f"- Run: {best_run.get('url') or best_run.get('id')}",
                "",
            ]
        )
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def proposal_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.yaml")))
            files.extend(sorted(path.rglob("*.yml")))
        elif path.is_file() and path.suffix in {".yaml", ".yml"}:
            files.append(path)
    return files


def main() -> int:
    args = parse_args()
    policy = load_autoresearch_policy(args.policy)

    if args.command == "routes":
        print(dumps_json({"routes": policy.routes, "runner": policy.raw.get("runner", {})}))
        return 0

    if args.command == "best-run":
        if args.max_runs is not None:
            policy.raw.setdefault("wandb", {})["max_runs"] = args.max_runs
        runs = fetch_policy_wandb_runs(policy)
        best = select_current_best_run(policy, runs)
        print(
            dumps_json(
                {
                    "best_run": None
                    if best is None
                    else {
                        "id": best.id,
                        "name": best.name,
                        "url": best.url,
                        "summary": best.summary,
                        "tags": best.tags,
                    }
                }
            )
        )
        return 0

    if args.command == "propose":
        output_path = propose_config_for_route(
            policy=policy,
            route_name=args.route,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
        )
        print(dumps_json({"proposal": str(output_path), "route": args.route}))
        return 0

    if args.command == "check":
        decision = route_for(policy, args.route)
        base_config = args.base_config or decision.base_config
        path_errors = validate_changed_paths(args.paths, policy)
        config_errors = [
            error
            for path in proposal_files(args.paths)
            for error in validate_proposal_file(path, policy)
        ]
        delta_errors = [
            error
            for path in proposal_files(args.paths)
            for error in validate_config_delta(base_config, path, policy)
        ]
        errors = path_errors + config_errors + delta_errors
        print(dumps_json({"valid": not errors, "errors": errors}))
        return 0 if not errors else 1

    if args.command == "dispatch":
        result = dispatch_route(
            policy=policy,
            route_name=args.route,
            config_path=args.config,
            ref=args.ref,
            limit=args.limit,
        )
        print(dumps_json(result))
        return 0

    if args.command == "monitor":
        decision = route_for(policy, args.route)
        client = GitHubActionsClient(policy.github_repository)
        runs = client.list_workflow_runs(
            workflow=decision.workflow,
            branch=args.branch,
            per_page=args.per_page,
        )
        workflow_summary = summarize_runs(runs)
        wandb_runs = fetch_policy_wandb_runs(policy) if args.include_wandb or args.log_wandb else None
        summary = build_autoresearch_summary(
            policy=policy,
            route_name=args.route,
            workflow_summary=workflow_summary,
            wandb_runs=wandb_runs,
        )
        if args.log_wandb:
            run_url = log_autoresearch_summary(policy, summary)
            if run_url:
                summary["autoresearch_wandb_run_url"] = run_url
        write_github_step_summary(summary)
        print(dumps_json(summary))
        return 0

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
