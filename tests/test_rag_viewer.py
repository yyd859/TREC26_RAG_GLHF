from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from trec26_rag.generator import AnswerGenerationRequest, EvidenceDocument
from trec26_rag.rag_output import AnswerSentence, RagResponse
from trec26_rag.rag_viewer import (
    build_rag_table_rows,
    build_rag_viewer_data,
    render_rag_viewer_html,
    write_rag_table_csv,
    write_rag_table_jsonl,
    write_rag_viewer_html,
)
from trec26_rag.runfile import Topic


def sample_inputs() -> tuple[list[AnswerGenerationRequest], list[RagResponse], dict, dict, dict]:
    topic = Topic("14", "Industrial Revolution", "Explain causes and effects.")
    request = AnswerGenerationRequest(
        topic=topic,
        evidence=[
            EvidenceDocument("doc-a", "Evidence text A."),
            EvidenceDocument("doc-b", "Evidence text B."),
        ],
    )
    response = RagResponse(
        topic=topic,
        team_id="glhf",
        run_id="rag-run",
        references=["doc-a"],
        answer=[AnswerSentence("Factories changed work.", [0])],
    )
    validation_report = {
        "valid": True,
        "metrics": {"rag_validation_error_count": 0, "requested_topics": 1},
        "errors": [],
        "warnings": [],
    }
    citation_diagnostics = {
        "summary": {"citation_coverage_mean": 1.0},
        "per_topic": {
            "14": {
                "citation_coverage": 1.0,
                "citation_density_per_sentence": 1.0,
                "uncited_reference_count": 0,
                "answer_word_count": 3,
            }
        },
    }
    proxy_metrics = {"rag_proxy_response_rate": 1.0}
    return [request], [response], validation_report, citation_diagnostics, proxy_metrics


class RagViewerTest(unittest.TestCase):
    def test_build_rag_viewer_data_includes_topic_answer_and_evidence(self) -> None:
        answer_requests, responses, validation_report, citation_diagnostics, proxy_metrics = (
            sample_inputs()
        )
        data = build_rag_viewer_data(
            answer_requests=answer_requests,
            responses=responses,
            validation_report=validation_report,
            citation_diagnostics=citation_diagnostics,
            proxy_metrics=proxy_metrics,
        )
        self.assertTrue(data["summary"]["valid"])
        self.assertEqual(data["topics"][0]["topic_id"], "14")
        self.assertEqual(data["topics"][0]["answer"][0]["text"], "Factories changed work.")
        self.assertEqual(data["topics"][0]["evidence"][0]["text"], "Evidence text A.")
        self.assertEqual(data["topics"][0]["diagnostics"]["citation_coverage"], 1.0)

    def test_render_rag_viewer_html_embeds_parseable_json(self) -> None:
        html = render_rag_viewer_html({"summary": {"valid": True}, "topics": []})
        self.assertIn("RAG Run Viewer", html)
        self.assertIn('id="rag-viewer-data"', html)
        self.assertNotIn("{{", html)
        self.assertNotIn("}}", html)
        start = html.index('<script id="rag-viewer-data" type="application/json">')
        start = html.index(">", start) + 1
        end = html.index("</script>", start)
        payload = json.loads(html[start:end])
        self.assertTrue(payload["summary"]["valid"])

    def test_render_rag_viewer_html_javascript_syntax(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        html = render_rag_viewer_html({"summary": {"valid": True}, "topics": []})
        script_start = html.rindex("<script>")
        script_start = html.index(">", script_start) + 1
        script_end = html.index("</script>", script_start)
        script = html[script_start:script_end]
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "viewer.js"
            script_path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [node, "--check", str(script_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_and_write_rag_table_files(self) -> None:
        answer_requests, responses, validation_report, citation_diagnostics, proxy_metrics = (
            sample_inputs()
        )
        viewer_data = build_rag_viewer_data(
            answer_requests=answer_requests,
            responses=responses,
            validation_report=validation_report,
            citation_diagnostics=citation_diagnostics,
            proxy_metrics=proxy_metrics,
        )
        rows = build_rag_table_rows(viewer_data)
        self.assertEqual(rows[0]["topic_id"], "14")
        self.assertEqual(rows[0]["answer_text"], "Factories changed work.")
        self.assertIn("doc-a", rows[0]["references_json"])
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = write_rag_table_csv(rows, Path(tmpdir) / "rag_outputs_table.csv")
            jsonl_path = write_rag_table_jsonl(rows, Path(tmpdir) / "rag_outputs_table.jsonl")
            self.assertIn("topic_id,title,narrative", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(jsonl_path.read_text(encoding="utf-8"))["topic_id"],
                "14",
            )

    def test_write_rag_viewer_html_creates_file(self) -> None:
        answer_requests, responses, validation_report, citation_diagnostics, proxy_metrics = (
            sample_inputs()
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "viewer" / "rag_viewer.html"
            write_rag_viewer_html(
                path=path,
                answer_requests=answer_requests,
                responses=responses,
                validation_report=validation_report,
                citation_diagnostics=citation_diagnostics,
                proxy_metrics=proxy_metrics,
            )
            html = path.read_text(encoding="utf-8")
        self.assertIn("Industrial Revolution", html)
        self.assertIn("Factories changed work.", html)


if __name__ == "__main__":
    unittest.main()
