from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from trec26_rag.evaluation import evaluate_retrieval_run, read_qrels, read_runfile_rankings


class EvaluationTest(unittest.TestCase):
    def test_read_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.qrels"
            path.write_text("1 0 doc-a 2\n1 0 doc-b 0\n", encoding="utf-8")
            qrels = read_qrels(path)
            self.assertEqual(qrels, {"1": {"doc-a": 2, "doc-b": 0}})

    def test_read_runfile_rankings_dedupes_docids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.tsv"
            path.write_text(
                "1 Q0 doc-a 1 10 run\n1 Q0 doc-a 2 9 run\n1 Q0 doc-b 3 8 run\n",
                encoding="utf-8",
            )
            rankings = read_runfile_rankings(path)
            self.assertEqual(rankings, {"1": ["doc-a", "doc-b"]})

    def test_evaluate_retrieval_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            qrels_path = Path(tmpdir) / "test.qrels"
            runfile_path = Path(tmpdir) / "run.tsv"
            qrels_path.write_text(
                "\n".join(
                    [
                        "1 0 doc-a 2",
                        "1 0 doc-b 1",
                        "1 0 doc-c 0",
                        "2 0 doc-x 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runfile_path.write_text(
                "\n".join(
                    [
                        "1 Q0 doc-c 1 10 run",
                        "1 Q0 doc-a 2 9 run",
                        "1 Q0 doc-b 3 8 run",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_retrieval_run(runfile_path, qrels_path)
            metrics = report["metrics"]
            self.assertEqual(metrics["qrels_topic_count"], 2)
            self.assertEqual(metrics["run_topics_with_qrels_count"], 1)
            self.assertEqual(metrics["qrels_topics_missing_run_count"], 1)
            self.assertAlmostEqual(metrics["recall@100"], 0.5)
            self.assertAlmostEqual(metrics["map"], ((1 / 2 + 2 / 3) / 2) / 2)
            self.assertAlmostEqual(metrics["mrr"], 0.25)
            self.assertGreater(metrics["ndcg@10"], 0)
            self.assertLessEqual(metrics["ndcg@10"], 1)

    def test_evaluate_perfect_ranking_has_perfect_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            qrels_path = Path(tmpdir) / "test.qrels"
            runfile_path = Path(tmpdir) / "run.tsv"
            qrels_path.write_text("1 0 doc-a 2\n1 0 doc-b 1\n", encoding="utf-8")
            runfile_path.write_text("1 Q0 doc-a 1 10 run\n1 Q0 doc-b 2 9 run\n", encoding="utf-8")
            metrics = evaluate_retrieval_run(runfile_path, qrels_path)["metrics"]
            self.assertTrue(math.isclose(metrics["ndcg@10"], 1.0))
            self.assertTrue(math.isclose(metrics["recall@100"], 1.0))
            self.assertTrue(math.isclose(metrics["map"], 1.0))
            self.assertTrue(math.isclose(metrics["mrr"], 1.0))


if __name__ == "__main__":
    unittest.main()
