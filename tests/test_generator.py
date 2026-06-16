from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from trec26_rag.generator import (
    AnthropicBatchAnswerGenerator,
    AnswerGenerationRequest,
    AnswerGeneratorError,
    EvidenceDocument,
    parse_batch_results_jsonl,
    render_rag_prompt,
    safe_custom_id,
)
from trec26_rag.runfile import Topic


class GeneratorTest(unittest.TestCase):
    def test_safe_custom_id(self) -> None:
        self.assertEqual(safe_custom_id("topic:14/abc"), "topic_14_abc")
        self.assertLessEqual(len(safe_custom_id("x" * 100)), 64)

    def test_render_rag_prompt_includes_topic_and_evidence(self) -> None:
        request = AnswerGenerationRequest(
            topic=Topic("14", "Title", "Narrative"),
            evidence=[EvidenceDocument("doc-a", "Evidence text")],
        )
        prompt = render_rag_prompt(request, "Use evidence only.")
        self.assertIn("Title: Title", prompt)
        self.assertIn("[0] doc-a", prompt)
        self.assertIn("Return JSON only", prompt)

    def test_build_batch_requests_defaults_to_haiku_45(self) -> None:
        generator = AnthropicBatchAnswerGenerator(
            api_key="key",
            model="claude-haiku-4-5-20251001",
            max_output_tokens=800,
            prompt_template="Use evidence only.",
        )
        request = AnswerGenerationRequest(
            topic=Topic("14", "Title", "Narrative"),
            evidence=[EvidenceDocument("doc-a", "Evidence text")],
        )
        batch_requests = generator.build_batch_requests([request])
        self.assertEqual(batch_requests[0]["custom_id"], "14")
        self.assertEqual(batch_requests[0]["params"]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(batch_requests[0]["params"]["max_tokens"], 800)

    def test_build_batch_requests_rejects_empty_batch(self) -> None:
        generator = AnthropicBatchAnswerGenerator(api_key="key")
        with self.assertRaises(AnswerGeneratorError):
            generator.build_batch_requests([])

    def test_create_batch_calls_anthropic_batch_endpoint(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"id": "msgbatch_123", "processing_status": "in_progress"}
        generator = AnthropicBatchAnswerGenerator(api_key="key", api_base_url="https://api.test")
        request = AnswerGenerationRequest(
            topic=Topic("14", "Title", "Narrative"),
            evidence=[EvidenceDocument("doc-a", "Evidence text")],
        )
        with patch("trec26_rag.generator.requests.post", return_value=response) as post:
            job = generator.create_batch([request])
        self.assertEqual(job.id, "msgbatch_123")
        post.assert_called_once()
        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        self.assertEqual(url, "https://api.test/v1/messages/batches")
        self.assertEqual(payload["requests"][0]["params"]["model"], "claude-haiku-4-5-20251001")

    def test_parse_batch_results_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "custom_id": "14",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "{\"answer\": []}"},
                                ]
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_batch_results_jsonl(path), {"14": "{\"answer\": []}"})


if __name__ == "__main__":
    unittest.main()
