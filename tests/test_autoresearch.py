from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from trec26_rag.autoresearch import (
    append_research_decision,
    apply_runtime_limit,
    GitHubPullRequest,
    GitHubWorkflowRun,
    build_dispatch_payload,
    build_autoresearch_summary,
    build_research_context,
    changed_config_keys,
    compare_url,
    experiment_branch_name,
    find_secret_like_values,
    is_path_allowed,
    latest_changed_experiment_config,
    load_autoresearch_policy,
    load_research_memory,
    open_autoresearch_bootstrap_pr,
    propose_config_for_route,
    route_for,
    route_name_for_config,
    restore_unapproved_config_changes,
    save_research_memory,
    select_current_best_run,
    slugify_branch_component,
    summarize_best_run,
    summarize_runs,
    validate_changed_paths,
    validate_config_delta,
    validate_proposal_file,
    workflow_limit_for_config,
)
from trec26_rag.config import load_config, write_config
from trec26_rag.experiment_optimizer import RunRecord


class AutoresearchTest(unittest.TestCase):
    def test_policy_loads_required_routes(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        self.assertEqual(route_for(policy, "retrieval").workflow, "run-retrieval-baseline.yml")
        self.assertEqual(route_for(policy, "rag").workflow, "run-rag-baseline.yml")
        self.assertEqual(route_for(policy, "evaluation-only").mode, "evaluation_only")
        self.assertEqual(route_for(policy, "proposer-only").mode, "proposer_only")
        self.assertFalse(policy.review_requires_pr)
        self.assertEqual(policy.branch_prefix, "codex/autoresearch-")

    def test_path_allowlist_accepts_only_experiment_configs(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        self.assertTrue(is_path_allowed("configs/experiments/next.yaml", policy.allowed_paths))
        self.assertFalse(is_path_allowed("src/trec26_rag/config.py", policy.allowed_paths))
        self.assertEqual(
            validate_changed_paths(["configs/experiments/next.yaml", "README.md"], policy),
            ["README.md"],
        )

    def test_secret_like_values_are_detected(self) -> None:
        self.assertEqual(
            find_secret_like_values({"wandb": {"api_key": "real-value"}}),
            ["wandb.api_key"],
        )
        self.assertEqual(find_secret_like_values({"rag": {"max_output_tokens": 800}}), [])
        self.assertEqual(find_secret_like_values({"wandb": {"api_key": "..."}}), [])

    def test_validate_proposal_rejects_secret_in_allowed_path(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            proposal_path = Path(tmp) / "unsafe.yaml"
            write_config(
                {
                    "experiment": {
                        "name": "unsafe",
                        "hypothesis": "should fail",
                        "run_id": "unsafe",
                    },
                    "wandb": {"api_key": "do-not-commit"},
                },
                proposal_path,
            )
            policy_path = Path(tmp) / "policy.yaml"
            raw_policy = dict(policy.raw)
            raw_policy["allowed_paths"] = [f"{tmp}/"]
            policy_path.write_text(yaml.safe_dump(raw_policy), encoding="utf-8")
            tmp_policy = load_autoresearch_policy(policy_path)

            errors = validate_proposal_file(proposal_path, tmp_policy)

        self.assertTrue(any("secret-like" in error for error in errors))

    def test_config_delta_rejects_unapproved_key_changes(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir="configs/experiments") as output_dir:
            proposal_path = Path(output_dir) / "bad_delta.yaml"
            write_config(
                {
                    "experiment": {
                        "name": "bad_delta",
                        "hypothesis": "changes output path",
                        "run_id": "bad-delta",
                    },
                    "output": {"runfile_name": "surprise.tsv"},
                },
                proposal_path,
            )

            errors = validate_config_delta("configs/baseline_retrieval.yaml", proposal_path, policy)

        self.assertTrue(any("output.runfile_name" in error for error in errors))

    def test_changed_config_keys_reports_leaf_changes(self) -> None:
        self.assertEqual(
            changed_config_keys(
                {"retrieval": {"hits": 100, "query_template": "{title}"}},
                {"retrieval": {"hits": 200, "query_template": "{title}"}},
            ),
            {"retrieval.hits"},
        )

    def test_restore_unapproved_config_changes_keeps_only_policy_surface(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        base = {
            "retrieval": {
                "query_template": "{title}",
                "hits": 100,
                "max_retries": 5,
            }
        }
        proposal = {
            "retrieval": {
                "query_template": "{title} {narrative}",
                "hits": 200,
                "max_retries": 8,
            }
        }

        restored = restore_unapproved_config_changes(base, proposal, policy)

        self.assertEqual(restored["retrieval"]["query_template"], "{title} {narrative}")
        self.assertEqual(restored["retrieval"]["hits"], 200)
        self.assertEqual(restored["retrieval"]["max_retries"], 5)

    def test_propose_config_for_route_writes_safe_experiment_config(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir="configs/experiments") as output_dir:
            output_path = propose_config_for_route(
                policy=policy,
                route_name="retrieval",
                runs=[],
                output_dir=output_dir,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(validate_proposal_file(output_path, policy), [])
            self.assertIn("configs/experiments", output_path.as_posix())

    def test_select_current_best_run_uses_fallback_when_primary_missing(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        runs = [
            RunRecord(
                id="small",
                name="small",
                url=None,
                config={},
                summary={"candidate_count_mean": 10, "validation_error_count": 0},
                tags=[],
            ),
            RunRecord(
                id="large",
                name="large",
                url=None,
                config={},
                summary={"candidate_count_mean": 20, "validation_error_count": 0},
                tags=[],
            ),
        ]

        best = select_current_best_run(policy, runs)

        self.assertIsNotNone(best)
        self.assertEqual(best.id, "large")

    def test_summarize_best_run_reports_metric_and_value(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        runs = [
            RunRecord(
                id="run-1",
                name="run-1",
                url="https://wandb.test/run-1",
                config={},
                summary={"candidate_count_mean": 20, "validation_error_count": 0},
                tags=["retrieval"],
            )
        ]

        summary = summarize_best_run(policy, runs)

        self.assertEqual(summary["metric"], "candidate_count_mean")
        self.assertTrue(summary["used_fallback"])
        self.assertEqual(summary["best_run"]["value"], 20)

    def test_summarize_runs_reports_latest_action_state(self) -> None:
        summary = summarize_runs(
            [
                GitHubWorkflowRun(
                    id=123,
                    name="Run RAG Baseline",
                    status="completed",
                    conclusion="success",
                    html_url="https://example.test/run",
                    head_branch="dev",
                    head_sha="abc",
                )
            ]
        )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["conclusion"], "success")
        self.assertIn("Run RAG Baseline", summary["summary"])

    def test_build_autoresearch_summary_combines_workflow_and_wandb_best(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        workflow_summary = {"status": "completed", "conclusion": "success", "run_id": 123}
        runs = [
            RunRecord(
                id="best",
                name="best",
                url=None,
                config={},
                summary={"candidate_count_mean": 42, "validation_error_count": 0},
                tags=[],
            )
        ]

        summary = build_autoresearch_summary(policy, "retrieval", workflow_summary, runs)

        self.assertEqual(summary["route"], "retrieval")
        self.assertEqual(summary["workflow"]["conclusion"], "success")
        self.assertEqual(summary["current_best"]["best_run"]["id"], "best")

    def test_open_bootstrap_pr_uses_policy_repo_and_base(self) -> None:
        class FakeGitHubClient:
            def __init__(self) -> None:
                self.payload = None

            def create_pull_request(self, **kwargs):
                self.payload = kwargs
                return GitHubPullRequest(
                    number=7,
                    title=kwargs["title"],
                    html_url="https://github.test/pr/7",
                    state="open",
                    draft=kwargs["draft"],
                )

        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        client = FakeGitHubClient()

        pr = open_autoresearch_bootstrap_pr(policy, client=client)

        self.assertEqual(pr.number, 7)
        self.assertEqual(client.payload["head"], "codex/autoresearch-v1")
        self.assertEqual(client.payload["base"], "main")
        self.assertTrue(client.payload["draft"])

    def test_compare_url_points_to_bootstrap_branch(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        self.assertEqual(
            compare_url(policy, "codex/autoresearch-v1"),
            "https://github.com/yyd859/TREC26_RAG_GLHF/compare/main...codex/autoresearch-v1",
        )

    def test_experiment_branch_name_uses_safe_slug(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        self.assertEqual(
            slugify_branch_component("2026 Next/Thing!"),
            "2026-next-thing",
        )
        self.assertEqual(
            experiment_branch_name(policy, "configs/experiments/20260621_Title Boosted.yaml"),
            "codex/autoresearch-20260621_title-boosted",
        )

    def test_build_dispatch_payload_uses_route_defaults(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        payload = build_dispatch_payload(
            policy=policy,
            route_name="rag",
            config_path="configs/experiments/rag.yaml",
        )

        self.assertEqual(payload["workflow"], "run-rag-baseline.yml")
        self.assertEqual(payload["ref"], "main")
        self.assertEqual(payload["inputs"]["config"], "configs/experiments/rag.yaml")
        self.assertEqual(payload["inputs"]["limit"], "2")

    def test_route_name_for_config_uses_experiment_task(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir="configs/experiments") as output_dir:
            rag_path = Path(output_dir) / "rag_route.yaml"
            retrieval_path = Path(output_dir) / "retrieval_route.yaml"
            write_config({"experiment": {"task": "rag"}, "rag": {"enabled": True}}, rag_path)
            write_config({"experiment": {"task": "retrieval"}, "rag": {"enabled": False}}, retrieval_path)

            self.assertEqual(route_name_for_config(policy, rag_path), "rag")
            self.assertEqual(route_name_for_config(policy, retrieval_path), "retrieval")

    def test_workflow_limit_prefers_runtime_limit_in_config(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir="configs/experiments") as output_dir:
            config_path = Path(output_dir) / "limited.yaml"
            write_config({"experiment": {"task": "retrieval"}}, config_path)
            apply_runtime_limit(config_path, "2")

            self.assertEqual(workflow_limit_for_config(policy, config_path), "2")
            self.assertEqual(load_config(config_path)["retrieval"]["timeout_seconds"], 30)
            self.assertEqual(route_name_for_config(policy, config_path), "retrieval")
            self.assertEqual(validate_config_delta("configs/baseline_retrieval.yaml", config_path, policy), [])

    def test_latest_changed_experiment_config_falls_back_to_newest_file(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        with tempfile.TemporaryDirectory(dir="configs/experiments") as output_dir:
            old_path = Path(output_dir) / "old.yaml"
            new_path = Path(output_dir) / "new.yaml"
            write_config({"experiment": {"task": "retrieval"}}, old_path)
            write_config({"experiment": {"task": "rag"}, "rag": {"enabled": True}}, new_path)
            old_path.touch()
            new_path.touch()

            latest = latest_changed_experiment_config(policy, ref="missing-ref")

        self.assertEqual(latest.name, "new.yaml")

    def test_research_memory_round_trips_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            memory_path = Path(output_dir) / "memory.json"
            memory = load_research_memory(memory_path)
            append_research_decision(
                memory,
                route_name="rag",
                proposal_path="configs/experiments/rag.yaml",
                branch="codex/autoresearch-rag",
                workflow_summary={"conclusion": "success"},
                wandb_summary={"best_run": {"id": "run-1"}},
            )
            save_research_memory(memory_path, memory)
            reloaded = load_research_memory(memory_path)

        self.assertEqual(reloaded["decisions"][0]["route"], "rag")
        self.assertEqual(reloaded["decisions"][0]["branch"], "codex/autoresearch-rag")

    def test_build_research_context_combines_configs_runs_and_memory(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")
        runs = [
            RunRecord(
                id="run-1",
                name="run-1",
                url="https://wandb.test/run-1",
                config={
                    "experiment": {"task": "retrieval", "name": "run-1"},
                    "retrieval": {"query_template": "{title}", "hits": 100},
                    "rag": {"evidence_top_k": 5, "max_output_tokens": 800},
                },
                summary={"candidate_count_mean": 10, "validation_error_count": 0},
                tags=["retrieval"],
            )
        ]
        with tempfile.TemporaryDirectory(dir="configs/experiments") as config_dir:
            config_path = Path(config_dir) / "context.yaml"
            memory_path = Path(config_dir) / "memory.json"
            write_config(
                {
                    "experiment": {"task": "rag", "name": "context"},
                    "retrieval": {"query_template": "{title} {narrative}", "hits": 20},
                    "rag": {"enabled": True, "evidence_top_k": 5},
                },
                config_path,
            )
            save_research_memory(memory_path, {"version": 1, "decisions": [{"route": "rag"}]})

            context = build_research_context(
                policy,
                runs=runs,
                config_dir=config_dir,
                memory_path=memory_path,
            )

        self.assertEqual(context["wandb_runs"][0]["id"], "run-1")
        self.assertEqual(context["historical_configs"][0]["path"], config_path.as_posix())
        self.assertEqual(context["research_memory"]["decisions"][0]["route"], "rag")
        self.assertTrue(context["tried_config_signatures"])


if __name__ == "__main__":
    unittest.main()
