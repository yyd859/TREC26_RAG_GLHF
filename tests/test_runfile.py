from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec26_rag.runfile import RunRow, Topic, read_topics, render_query, validate_runfile, write_runfile


class RunfileTest(unittest.TestCase):
    def test_render_query_uses_topic_fields(self) -> None:
        topic = Topic(id="1", title="Industrial Revolution", narrative="causes and effects")
        self.assertEqual(
            render_query("{title} {title} {narrative}", topic),
            "Industrial Revolution Industrial Revolution causes and effects",
        )

    def test_validate_good_runfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.tsv"
            write_runfile(
                [
                    RunRow("1", "doc-a", 1, 10.0, "run"),
                    RunRow("1", "doc-b", 2, 9.0, "run"),
                    RunRow("2", "doc-c", 1, 8.0, "run"),
                ],
                path,
            )
            report = validate_runfile(path, topic_ids={"1", "2"})
            self.assertTrue(report["valid"])
            self.assertEqual(report["metrics"]["total_rows"], 3)
            self.assertEqual(report["metrics"]["empty_topic_count"], 0)
            self.assertEqual(report["metrics"]["duplicate_doc_count"], 0)
            self.assertEqual(report["metrics"]["score_min"], 8.0)
            self.assertEqual(report["metrics"]["score_max"], 10.0)
            self.assertEqual(report["diagnostics"]["per_topic"]["1"]["candidate_count"], 2)

    def test_validate_missing_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.tsv"
            write_runfile([RunRow("1", "doc-a", 1, 10.0, "run")], path)
            report = validate_runfile(path, topic_ids={"1", "2"})
            self.assertFalse(report["valid"])
            self.assertIn("Missing output", report["errors"][0])
            self.assertEqual(report["metrics"]["empty_topic_count"], 1)
            self.assertEqual(report["diagnostics"]["per_topic"]["2"]["candidate_count"], 0)

    def test_validate_duplicate_doc_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.tsv"
            write_runfile(
                [
                    RunRow("1", "doc-a", 1, 10.0, "run"),
                    RunRow("1", "doc-a", 2, 9.0, "run"),
                ],
                path,
            )
            report = validate_runfile(path, topic_ids={"1"})
            self.assertTrue(report["valid"])
            self.assertEqual(report["metrics"]["duplicate_doc_count"], 1)
            self.assertEqual(report["metrics"]["duplicate_topic_count"], 1)
            self.assertEqual(report["diagnostics"]["duplicate_by_topic"], {"1": 1})

    def test_read_topics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "topics.jsonl"
            path.write_text(
                json.dumps({"id": "1", "title": "Title", "narrative": "Narrative"}) + "\n",
                encoding="utf-8",
            )
            topics = read_topics(path)
            self.assertEqual(topics[0].id, "1")


if __name__ == "__main__":
    unittest.main()
