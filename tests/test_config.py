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
        self.assertIn("rag", config)
        self.assertFalse(config["rag"]["enabled"])
        self.assertEqual(config["rag"]["evidence_top_k"], 5)
        self.assertEqual(config["rag"]["generator_provider"], "anthropic_batch")
        self.assertEqual(config["rag"]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(config["rag"]["max_output_tokens"], 800)

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

    def test_load_config_deep_merges_rag_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag_experiment.yaml"
            path.write_text(
                "\n".join(
                    [
                        "rag:",
                        "  enabled: true",
                        "  generator_provider: openai",
                        "  model: gpt-4.1-mini",
                        "  evidence_top_k: 8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertTrue(config["rag"]["enabled"])
            self.assertEqual(config["rag"]["generator_provider"], "openai")
            self.assertEqual(config["rag"]["model"], "gpt-4.1-mini")
            self.assertEqual(config["rag"]["evidence_top_k"], 8)
            self.assertEqual(config["rag"]["max_output_tokens"], 800)
            self.assertEqual(config["output"]["rag_output_name"], "rag_output_trec_rag_2026.jsonl")


if __name__ == "__main__":
    unittest.main()
