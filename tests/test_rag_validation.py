from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec26_rag.rag_output import AnswerSentence, RagResponse, write_rag_jsonl
from trec26_rag.rag_validation import validate_rag_jsonl
from trec26_rag.runfile import Topic


def valid_payload(topic_id: str = "14") -> dict:
    return {
        "metadata": {
            "team_id": "glhf",
            "run_id": "rag-run",
            "type": "automatic",
            "narrative_id": topic_id,
            "title": "Title",
            "narrative": "Narrative",
            "prompt": "Prompt",
        },
        "references": ["doc-a", "doc-b"],
        "answer": [
            {"text": "First sentence.", "citations": [0]},
            {"text": "Second sentence.", "citations": [1]},
        ],
    }


class RagValidationTest(unittest.TestCase):
    def test_validate_good_rag_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(valid_payload()) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path, topic_ids={"14"})
        self.assertTrue(report["valid"])
        self.assertEqual(report["metrics"]["rag_object_count"], 1)
        self.assertEqual(report["metrics"]["rag_reference_count_total"], 2)
        self.assertEqual(report["metrics"]["rag_answer_sentence_count_total"], 2)
        self.assertEqual(report["metrics"]["rag_invalid_citation_count"], 0)
        self.assertEqual(report["metrics"]["rag_uncited_reference_count"], 0)

    def test_valid_jsonl_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertIn("invalid JSONL object", report["errors"][0])

    def test_one_object_per_topic_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(
                json.dumps(valid_payload("14")) + "\n" + json.dumps(valid_payload("14")) + "\n",
                encoding="utf-8",
            )
            report = validate_rag_jsonl(path, topic_ids={"14"})
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate RAG object" in error for error in report["errors"]))

    def test_required_metadata_fields_check(self) -> None:
        payload = valid_payload()
        payload["metadata"].pop("team_id")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("metadata missing required field" in error for error in report["errors"]))

    def test_references_non_empty_check(self) -> None:
        payload = valid_payload()
        payload["references"] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("references must be non-empty" in error for error in report["errors"]))

    def test_answer_sentence_level_array_check(self) -> None:
        payload = valid_payload()
        payload["answer"] = {"text": "Not a list"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("answer must be a sentence-level list" in error for error in report["errors"]))

    def test_valid_citation_indices_check(self) -> None:
        payload = valid_payload()
        payload["answer"][0]["citations"] = [99]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertEqual(report["metrics"]["rag_invalid_citation_count"], 1)

    def test_every_reference_cited_check(self) -> None:
        payload = valid_payload()
        payload["answer"] = [{"text": "Only cites one reference.", "citations": [0]}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertEqual(report["metrics"]["rag_uncited_reference_count"], 1)

    def test_no_empty_answer_sentences_check(self) -> None:
        payload = valid_payload()
        payload["answer"][0]["text"] = " "
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("text must be non-empty" in error for error in report["errors"]))

    def test_missing_topic_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            path.write_text(json.dumps(valid_payload("14")) + "\n", encoding="utf-8")
            report = validate_rag_jsonl(path, topic_ids={"14", "31"})
        self.assertFalse(report["valid"])
        self.assertEqual(report["metrics"]["rag_missing_topic_count"], 1)

    def test_writer_output_validates(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a"],
            answer=[AnswerSentence(text="Answer sentence.", citations=[0])],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rag.jsonl"
            write_rag_jsonl([response], path)
            report = validate_rag_jsonl(path, topic_ids={"14"})
        self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
