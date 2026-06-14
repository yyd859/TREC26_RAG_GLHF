from __future__ import annotations

import unittest
from datetime import datetime, timezone

from trec26_rag.config import DEFAULT_CONFIG
from trec26_rag.experiment_optimizer import RunRecord, propose_next_config, select_best_run


class ExperimentOptimizerTest(unittest.TestCase):
    def test_select_best_run_maximizes_metric_and_skips_invalid(self) -> None:
        runs = [
            RunRecord("bad", "bad", None, {}, {"candidate_count_mean": 200, "validation_error_count": 1}, []),
            RunRecord("ok", "ok", None, {}, {"candidate_count_mean": 100, "validation_error_count": 0}, []),
            RunRecord("best", "best", None, {}, {"candidate_count_mean": 150, "validation_error_count": 0}, []),
        ]
        best = select_best_run(runs, "candidate_count_mean", "maximize")
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "best")

    def test_propose_next_config_changes_one_retrieval_strategy(self) -> None:
        now = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
        config = propose_next_config(DEFAULT_CONFIG, [], now=now)
        self.assertEqual(config["experiment"]["name"], "20260614_120000_title_narrative_top100")
        self.assertEqual(config["retrieval"]["query_template"], "{title} {narrative}")
        self.assertEqual(config["retrieval"]["hits"], 100)
        self.assertIn("proposed", config["wandb"]["tags"])


if __name__ == "__main__":
    unittest.main()
