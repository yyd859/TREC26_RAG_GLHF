from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from trec26_rag.autoresearch import (
    GitHubWorkflowRun,
    build_autoresearch_summary,
    changed_config_keys,
    find_secret_like_values,
    is_path_allowed,
    load_autoresearch_policy,
    propose_config_for_route,
    route_for,
    select_current_best_run,
    summarize_best_run,
    summarize_runs,
    validate_changed_paths,
    validate_config_delta,
    validate_proposal_file,
)
from trec26_rag.config import write_config
from trec26_rag.experiment_optimizer import RunRecord


class AutoresearchTest(unittest.TestCase):
    def test_policy_loads_required_routes(self) -> None:
        policy = load_autoresearch_policy("configs/autoresearch.yaml")

        self.assertEqual(route_for(policy, "retrieval").workflow, "run-retrieval-baseline.yml")
        self.assertEqual(route_for(policy, "rag").workflow, "run-rag-baseline.yml")
        self.assertEqual(route_for(policy, "evaluation-only").mode, "evaluation_only")
        self.assertEqual(route_for(policy, "proposer-only").mode, "proposer_only")

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


if __name__ == "__main__":
    unittest.main()
