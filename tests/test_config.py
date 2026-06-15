from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trec26_rag.config import load_config


class ConfigTest(unittest.TestCase):
    def test_load_config_includes_evaluation_defaults(self) -> None:
        config = load_config("configs/baseline_retrieval.yaml")
        self.assertIn("evaluation", config)
        self.assertIsNone(config["evaluation"]["qrels_path"])
        self.assertEqual(config["evaluation"]["relevance_threshold"], 1)
        self.assertIn("ndcg@10", config["evaluation"]["metrics"])

    def test_load_config_deep_merges_evaluation_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment.yaml"
            path.write_text(
                "\n".join(
                    [
                        "evaluation:",
                        "  qrels_path: data/qrels/example.qrels",
                        "retrieval:",
                        "  hits: 200",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["evaluation"]["qrels_path"], "data/qrels/example.qrels")
            self.assertEqual(config["evaluation"]["relevance_threshold"], 1)
            self.assertEqual(config["retrieval"]["hits"], 200)
            self.assertEqual(config["retrieval"]["index"], "climbmix-400b")


if __name__ == "__main__":
    unittest.main()
