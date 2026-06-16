from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec26_rag.rag_output import (
    AnswerSentence,
    RagResponse,
    extract_answer_json_text,
    keep_cited_references_only,
    parse_answer_json,
    rag_response_to_json,
    write_rag_jsonl,
)
from trec26_rag.runfile import Topic


class RagOutputTest(unittest.TestCase):
    def test_rag_response_to_json_matches_expected_shape(self) -> None:
        topic = Topic("14", "Industrial Revolution", "Explain causes and effects.")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a", "doc-b"],
            answer=[
                AnswerSentence(
                    text="Industrialization changed labor and production.",
                    citations=[0, 1],
                )
            ],
            prompt="prompt text",
        )
        payload = rag_response_to_json(response)
        self.assertEqual(payload["metadata"]["team_id"], "glhf")
        self.assertEqual(payload["metadata"]["narrative_id"], "14")
        self.assertEqual(payload["references"], ["doc-a", "doc-b"])
        self.assertEqual(payload["answer"][0]["citations"], [0, 1])

    def test_write_rag_jsonl(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a"],
            answer=[AnswerSentence(text="Answer sentence.", citations=[0])],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rag.jsonl"
            write_rag_jsonl([response], output)
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["metadata"]["run_id"], "rag-run")
        self.assertEqual(payload["answer"][0]["text"], "Answer sentence.")

    def test_parse_answer_json_uses_fallback_references(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = parse_answer_json(
            raw_text='{"answer":[{"text":"Answer sentence.","citations":[0]}]}',
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            fallback_references=["doc-a"],
        )
        self.assertEqual(response.references, ["doc-a"])
        self.assertEqual(response.answer[0].citations, [0])

    def test_parse_answer_json_accepts_fenced_json(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = parse_answer_json(
            raw_text='```json\n{"answer":[{"text":"Answer sentence.","citations":[0]}]}\n```',
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            fallback_references=["doc-a"],
        )
        self.assertEqual(response.answer[0].text, "Answer sentence.")

    def test_extract_answer_json_text_ignores_surrounding_text(self) -> None:
        raw_text = 'Here is the JSON:\n{"answer":[]}\nDone.'
        self.assertEqual(extract_answer_json_text(raw_text), '{"answer":[]}')

    def test_keep_cited_references_only_removes_uncited_references_and_remaps(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a", "doc-b", "doc-c"],
            answer=[
                AnswerSentence(text="Uses b and c.", citations=[1, 2]),
                AnswerSentence(text="Uses c.", citations=[2]),
            ],
        )
        normalized = keep_cited_references_only(response)
        self.assertEqual(normalized.references, ["doc-b", "doc-c"])
        self.assertEqual(normalized.answer[0].citations, [0, 1])
        self.assertEqual(normalized.answer[1].citations, [1])

    def test_keep_cited_references_only_leaves_clean_response_unchanged(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a"],
            answer=[AnswerSentence(text="Uses a.", citations=[0])],
        )
        self.assertIs(keep_cited_references_only(response), response)


if __name__ == "__main__":
    unittest.main()
