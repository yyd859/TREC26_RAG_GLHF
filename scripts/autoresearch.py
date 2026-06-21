#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from trec26_rag.autoresearch import (
    GitHubWorkflowRun,
    append_research_decision,
    build_dispatch_payload,
    GitHubActionsClient,
    build_autoresearch_summary,
    build_research_context,
    compare_url,
    dispatch_route,
    dumps_json,
    fetch_policy_wandb_runs,
    latest_changed_experiment_config,
    load_autoresearch_policy,
    load_research_memory,
    log_autoresearch_summary,
    log_research_memory,
    open_autoresearch_bootstrap_pr,
    propose_config_for_route,
    route_for,
    route_name_for_config,
    run_autoresearch_loop,
    run_branch_iteration,
    save_research_memory,
    select_current_best_run,
    summarize_runs,
    validate_changed_paths,
    validate_config_delta,
    validate_proposal_file,
    workflow_limit_for_config,
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

    iterate = subparsers.add_parser(
        "iterate",
        help="Create a new experiment branch, commit a config proposal, and dispatch its workflow.",
    )
    iterate.add_argument("--route", default="retrieval")
    iterate.add_argument("--ref")
    iterate.add_argument("--limit")
    iterate.add_argument("--output-dir", default="configs/experiments")
    iterate.add_argument(
        "--dispatch",
        action="store_true",
        help="Dispatch explicitly instead of relying on branch push triggers.",
    )

    latest_config = subparsers.add_parser(
        "latest-config",
        help="Print the latest changed experiment config, optionally filtered by route.",
    )
    latest_config.add_argument("--route")
    latest_config.add_argument("--ref", default="HEAD")

    route_config = subparsers.add_parser("route-config", help="Infer the autoresearch route for a config.")
    route_config.add_argument("--config", required=True)
    route_config.add_argument(
        "--field",
        choices=["route", "workflow", "task", "default_limit"],
        default="route",
    )

    research_context = subparsers.add_parser(
        "research-context",
        help="Print the holistic context the local agent should read before proposing.",
    )
    research_context.add_argument("--max-runs", type=int)
    research_context.add_argument("--config-dir", default="configs/experiments")
    research_context.add_argument("--memory-path")

    loop = subparsers.add_parser(
        "loop",
        help="Run a bounded local autoresearch loop: propose, branch, push, monitor, log, remember.",
    )
    loop.add_argument("--route", default="retrieval")
    loop.add_argument("--ref")
    loop.add_argument("--max-rounds", type=int)
    loop.add_argument("--poll-seconds", type=int)
    loop.add_argument("--limit")

    open_pr = subparsers.add_parser("open-pr", help="Open the autoresearch bootstrap PR.")
    open_pr.add_argument("--head", default="codex/autoresearch-v1")
    open_pr.add_argument("--base")
    open_pr.add_argument("--title", default="Add autoresearch orchestrator v1")
    open_pr.add_argument("--ready", action="store_true", help="Create a non-draft PR.")

    dry_run = subparsers.add_parser("dry-run", help="Run a local no-network autoresearch simulation.")
    dry_run.add_argument("--route", default="retrieval")
    dry_run.add_argument("--ref", default="main")
    dry_run.add_argument("--limit")

    monitor = subparsers.add_parser("monitor", help="Summarize recent GitHub Actions status.")
    monitor.add_argument("--route", required=True)
    monitor.add_argument("--branch")
    monitor.add_argument("--per-page", type=int, default=5)
    monitor.add_argument("--include-wandb", action="store_true")
    monitor.add_argument("--log-wandb", action="store_true")
    monitor.add_argument("--update-memory", action="store_true")

    # Touch subparser objects so linters do not mistake them as unused in simple scripts.
    _ = (routes,)
    return parser.parse_args()


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

    if args.command == "iterate":
        result = run_branch_iteration(
            policy=policy,
            route_name=args.route,
            ref=args.ref,
            limit=args.limit,
            output_dir=args.output_dir,
            dispatch=args.dispatch,
        )
        print(dumps_json(result))
        return 0

    if args.command == "latest-config":
        path = latest_changed_experiment_config(policy, route_name=args.route, ref=args.ref)
        print(path.as_posix())
        return 0

    if args.command == "route-config":
        route_name = route_name_for_config(policy, args.config)
        decision = route_for(policy, route_name)
        values = {
            "route": route_name,
            "workflow": decision.workflow,
            "task": decision.task,
            "default_limit": workflow_limit_for_config(policy, args.config),
        }
        print(values[args.field])
        return 0

    if args.command == "research-context":
        if args.max_runs is not None:
            policy.raw.setdefault("wandb", {})["max_runs"] = args.max_runs
        context = build_research_context(
            policy=policy,
            config_dir=args.config_dir,
            memory_path=args.memory_path,
        )
        print(dumps_json(context))
        return 0

    if args.command == "loop":
        loop_config = policy.raw.get("loop", {})
        if not isinstance(loop_config, dict):
            loop_config = {}
        max_rounds = args.max_rounds or int(loop_config.get("max_rounds") or 1)
        poll_seconds = args.poll_seconds or int(loop_config.get("poll_seconds") or 120)
        result = run_autoresearch_loop(
            policy=policy,
            route_name=args.route,
            max_rounds=max_rounds,
            poll_seconds=poll_seconds,
            limit=args.limit,
            ref=args.ref,
        )
        print(dumps_json({"rounds": result}))
        return 0

    if args.command == "open-pr":
        try:
            pr = open_autoresearch_bootstrap_pr(
                policy=policy,
                head=args.head,
                base=args.base,
                title=args.title,
                draft=not args.ready,
            )
            print(
                dumps_json(
                    {
                        "number": pr.number,
                        "title": pr.title,
                        "url": pr.html_url,
                        "state": pr.state,
                        "draft": pr.draft,
                    }
                )
            )
            return 0
        except RuntimeError as exc:
            fallback = compare_url(policy, args.head, args.base)
            print(dumps_json({"error": str(exc), "manual_compare_url": fallback}))
            return 1

    if args.command == "dry-run":
        with tempfile.TemporaryDirectory(prefix="autoresearch-", dir="configs/experiments") as output_dir:
            proposal_path = propose_config_for_route(
                policy=policy,
                route_name=args.route,
                runs=[],
                output_dir=output_dir,
            )
            check_errors = validate_proposal_file(proposal_path, policy)
            check_errors.extend(
                validate_config_delta(route_for(policy, args.route).base_config, proposal_path, policy)
            )
            dispatch_payload = build_dispatch_payload(
                policy=policy,
                route_name=args.route,
                config_path=proposal_path.as_posix(),
                ref=args.ref,
                limit=args.limit,
            )
            workflow_summary = summarize_runs(
                [
                    GitHubWorkflowRun(
                        id=0,
                        name=f"Dry-run {args.route}",
                        status="completed",
                        conclusion="success" if not check_errors else "failure",
                        html_url=None,
                        head_branch=args.ref,
                        head_sha=None,
                    )
                ]
            )
            summary = build_autoresearch_summary(
                policy=policy,
                route_name=args.route,
                workflow_summary=workflow_summary,
                wandb_runs=[],
            )
            print(
                dumps_json(
                    {
                        "valid": not check_errors,
                        "proposal": proposal_path.as_posix(),
                        "errors": check_errors,
                        "dispatch": dispatch_payload,
                        "summary": summary,
                    }
                )
            )
            return 0 if not check_errors else 1

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
        if args.update_memory:
            memory = load_research_memory(policy.research_memory_path)
            append_research_decision(
                memory,
                route_name=args.route,
                branch=args.branch,
                workflow_summary=workflow_summary,
                wandb_summary=summary.get("current_best", {}),
            )
            memory_path = save_research_memory(policy.research_memory_path, memory)
            summary["research_memory_path"] = memory_path.as_posix()
            if args.log_wandb:
                memory_run_url = log_research_memory(policy, memory_path)
                if memory_run_url:
                    summary["research_memory_wandb_run_url"] = memory_run_url
        print(dumps_json(summary))
        return 0

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
