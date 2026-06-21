from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import scripts.run_rag_baseline as run_rag_baseline
from scripts.run_rag_baseline import (
    build_citation_diagnostics,
    build_generation_requests,
    build_proxy_metrics,
    build_rag_responses,
    write_json,
)
from trec26_rag.generator import AnswerGenerationRequest, EvidenceDocument
from trec26_rag.rag_output import AnswerSentence, RagResponse
from trec26_rag.pyserini_client import SearchHit
from trec26_rag.runfile import Topic


class FakePyseriniClient:
    def search(self, query: str, hits: int) -> list[SearchHit]:
        self.query = query
        self.hits = hits
        return [
            SearchHit("doc-a", 1, 9.0, doc={"text": "Short"}),
            SearchHit("doc-b", 2, 8.0, doc={"text": "Long enough hydrated evidence text."}),
            SearchHit("doc-c", 3, 7.0, doc={"text": "Ignored because top k is two."}),
        ]

    def hydrate_hits(
        self,
        hits: list[SearchHit],
        min_text_chars: int = 200,
        max_docs: int | None = None,
    ) -> list[SearchHit]:
        return [
            SearchHit(hit.docid, hit.rank, hit.score, doc={"text": f"Hydrated text for {hit.docid}."})
            for hit in hits
        ]


def rag_config() -> dict:
    return {
        "experiment": {"team_id": "glhf", "run_id": "rag-run"},
        "retrieval": {"query_template": "{title}", "hits": 20},
        "rag": {"evidence_top_k": 2, "prompt_template": "Use evidence only."},
    }


class RunRagBaselineTest(unittest.TestCase):
    def test_build_generation_requests_retrieves_and_hydrates_evidence(self) -> None:
        topic = Topic("14", "Industrial Revolution", "Explain causes and effects.")
        client = FakePyseriniClient()
        requests = build_generation_requests([topic], client, rag_config())
        self.assertEqual(client.query, "Industrial Revolution")
        self.assertEqual(client.hits, 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual([doc.docid for doc in requests[0].evidence], ["doc-a", "doc-b"])
        self.assertEqual(requests[0].evidence[0].text, "Hydrated text for doc-a.")

    def test_build_rag_responses_maps_batch_result_to_topic(self) -> None:
        topic = Topic("topic:14", "Title", "Narrative")
        request = AnswerGenerationRequest(
            topic=topic,
            evidence=[EvidenceDocument("doc-a", "Evidence text.")],
        )
        responses = build_rag_responses(
            answer_requests=[request],
            batch_results={
                "topic_14": '{"answer":[{"text":"Answer sentence.","citations":[0]}]}',
            },
            config=rag_config(),
        )
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].topic.id, "topic:14")
        self.assertEqual(responses[0].references, ["doc-a"])
        self.assertEqual(responses[0].answer[0].citations, [0])

    def test_build_citation_diagnostics_summarizes_per_topic_citations(self) -> None:
        response = RagResponse(
            topic=Topic("14", "Title", "Narrative"),
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a", "doc-b"],
            answer=[AnswerSentence("Sentence one.", [0]), AnswerSentence("Sentence two.", [0])],
        )
        diagnostics = build_citation_diagnostics(
            [response],
            {
                "diagnostics": {
                    "per_topic": {
                        "14": {
                            "invalid_citation_count": 0,
                            "uncited_reference_count": 1,
                        }
                    }
                }
            },
        )
        self.assertEqual(diagnostics["summary"]["citation_coverage_mean"], 0.5)
        self.assertEqual(diagnostics["summary"]["citation_density_mean"], 1.0)
        self.assertEqual(diagnostics["per_topic"]["14"]["uncited_reference_indices"], [1])
        self.assertEqual(diagnostics["per_topic"]["14"]["validator"]["uncited_reference_count"], 1)

    def test_build_proxy_metrics_includes_evidence_answer_and_citation_rates(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        request = AnswerGenerationRequest(
            topic=topic,
            evidence=[
                EvidenceDocument("doc-a", "Evidence one."),
                EvidenceDocument("doc-b", "Evidence two."),
            ],
        )
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a", "doc-b"],
            answer=[AnswerSentence("Answer sentence.", [0])],
        )
        validation_report = {
            "valid": False,
            "metrics": {
                "rag_reference_count_total": 2,
                "rag_uncited_reference_count": 1,
                "rag_citation_count_total": 1,
                "rag_invalid_citation_count": 0,
            },
            "diagnostics": {"per_topic": {}},
        }
        citation_diagnostics = build_citation_diagnostics([response], validation_report)
        metrics = build_proxy_metrics([request], [response], validation_report, citation_diagnostics)
        self.assertEqual(metrics["rag_proxy_response_rate"], 1.0)
        self.assertEqual(metrics["rag_proxy_evidence_docs_mean"], 2.0)
        self.assertEqual(metrics["rag_proxy_valid_output"], 0)
        self.assertEqual(metrics["rag_proxy_uncited_reference_rate"], 0.5)
        self.assertEqual(metrics["rag_proxy_invalid_citation_rate"], 0.0)

    def test_write_json_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "metrics.json"
            write_json(path, {"metric": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "metric": 1\n}')

    def test_main_returns_success_when_rag_validation_fails(self) -> None:
        topic = Topic("14", "Title", "Narrative")
        response = RagResponse(
            topic=topic,
            team_id="glhf",
            run_id="rag-run",
            references=["doc-a"],
            answer=[AnswerSentence("Answer sentence.", [0])],
        )
        config = {
            "experiment": {"team_id": "glhf", "run_id": "rag-run"},
            "data": {"topics_path": "topics.jsonl"},
            "retrieval": {"api_base_url": "http://example.test", "index": "idx"},
            "rag": {"enabled": True, "evidence_top_k": 1},
            "output": {
                "output_dir": "outputs",
                "rag_output_name": "rag.jsonl",
                "rag_validation_report_name": "report.json",
                "rag_proxy_metrics_name": "proxy.json",
                "rag_citation_diagnostics_name": "citations.json",
                "rag_viewer_name": "viewer.html",
                "rag_table_csv_name": "table.csv",
                "rag_table_jsonl_name": "table.jsonl",
            },
        }
        invalid_report = {
            "valid": False,
            "errors": ["invalid citation"],
            "warnings": [],
            "metrics": {
                "rag_object_count": 1,
                "rag_topic_count": 1,
                "rag_expected_topic_count": 1,
                "rag_missing_topic_count": 0,
                "rag_extra_topic_count": 0,
                "rag_reference_count_total": 1,
                "rag_reference_count_mean": 1.0,
                "rag_answer_sentence_count_total": 1,
                "rag_answer_sentence_count_mean": 1.0,
                "rag_citation_count_total": 1,
                "rag_citation_count_mean": 1.0,
                "rag_invalid_citation_count": 1,
                "rag_uncited_reference_count": 0,
                "rag_empty_answer_count": 0,
                "rag_validation_error_count": 1,
                "rag_validation_warning_count": 0,
            },
            "diagnostics": {"per_topic": {"14": {}}, "missing_topics": [], "extra_topics": []},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["output_dir"] = tmpdir
            argv = ["run_rag_baseline.py", "--config", "config.yaml", "--raw-results-output", f"{tmpdir}/raw.jsonl"]
            with (
                patch("sys.argv", argv),
                patch.object(run_rag_baseline, "load_config", return_value=config),
                patch.object(run_rag_baseline, "read_topics", return_value=[topic]),
                patch.object(run_rag_baseline, "PyseriniClient"),
                patch.object(run_rag_baseline.AnthropicBatchAnswerGenerator, "from_config"),
                patch.object(run_rag_baseline, "build_generation_requests", return_value=[]),
                patch.object(run_rag_baseline, "wait_for_batch", return_value="ended"),
                patch.object(run_rag_baseline, "parse_batch_results_jsonl", return_value={}),
                patch.object(run_rag_baseline, "build_rag_responses", return_value=[response]),
                patch.object(run_rag_baseline, "validate_rag_jsonl", return_value=invalid_report),
            ):
                generator = run_rag_baseline.AnthropicBatchAnswerGenerator.from_config.return_value
                generator.create_batch.return_value.id = "batch-1"
                generator.create_batch.return_value.processing_status = "in_progress"

                with redirect_stdout(StringIO()):
                    exit_code = run_rag_baseline.main()

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
