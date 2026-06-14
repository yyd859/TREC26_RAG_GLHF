from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trec26_rag.topics import parse_topic_tsv, prepare_topics, title_from_prompt


class TopicsTest(unittest.TestCase):
    def test_parse_topic_tsv(self) -> None:
        topics = parse_topic_tsv("id\tprompt\nabc\tWrite a detailed report about retrieval systems.\n")
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["id"], "abc")
        self.assertEqual(topics[0]["narrative"], "Write a detailed report about retrieval systems.")

    def test_title_from_prompt_truncates_on_word_boundary(self) -> None:
        title = title_from_prompt("one two three four", max_chars=11)
        self.assertEqual(title, "one two")

    def test_prepare_topics_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "topics.tsv"
            output = Path(tmpdir) / "topics.jsonl"
            source.write_text("a\tPrompt A\nb\tPrompt B\na\tPrompt A duplicate\n", encoding="utf-8")
            count = prepare_topics([str(source)], output)
            self.assertEqual(count, 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
