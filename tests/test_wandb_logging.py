from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from trec26_rag.wandb_logging import log_rag_run


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeRun:
    url = "https://wandb.test/run"

    def __init__(self) -> None:
        self.artifact: FakeArtifact | None = None

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifact = artifact


class FakeTable:
    def __init__(self, columns: list[str], data: list[list[object]]) -> None:
        self.columns = columns
        self.data = data


class FakeHtml:
    def __init__(self, path: str, inject: bool = True) -> None:
        self.path = path
        self.inject = inject


class WandbLoggingTest(unittest.TestCase):
    def test_log_rag_run_adds_artifacts_and_metadata(self) -> None:
        fake_run = FakeRun()
        logged_metrics: list[dict] = []
        fake_wandb = types.SimpleNamespace(
            init=lambda **kwargs: fake_run,
            log=lambda metrics: logged_metrics.append(metrics),
            Artifact=FakeArtifact,
            Table=FakeTable,
            Html=FakeHtml,
            finish=lambda: None,
        )
        original_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        self.addCleanup(self._restore_wandb, original_wandb)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "rag_output_trec_rag_2026.jsonl"
            artifact_path.write_text("{}", encoding="utf-8")
            viewer_path = Path(tmpdir) / "rag_viewer.html"
            viewer_path.write_text("<html></html>", encoding="utf-8")
            run_url = log_rag_run(
                config={
                    "experiment": {"name": "rag-test", "run_id": "rag-run"},
                    "wandb": {"project": "project", "mode": "offline"},
                },
                metrics={
                    "rag_proxy_valid_output": 1,
                    "rag_validation_error_count": 0,
                    "rag_proxy_response_rate": 1.0,
                    "rag_proxy_citation_coverage_mean": 0.75,
                },
                artifacts=[artifact_path, Path(tmpdir) / "missing.json"],
                tables={
                    "rag_outputs": {
                        "columns": ["topic_id", "answer_text"],
                        "data": [["14", "Answer sentence."]],
                    }
                },
                htmls={"rag_viewer": viewer_path},
            )

        self.assertEqual(run_url, "https://wandb.test/run")
        self.assertEqual(logged_metrics[0]["rag_proxy_response_rate"], 1.0)
        self.assertEqual(logged_metrics[1]["rag_outputs"].columns, ["topic_id", "answer_text"])
        self.assertEqual(logged_metrics[1]["rag_outputs"].data[0][0], "14")
        self.assertEqual(logged_metrics[2]["rag_viewer"].path, str(viewer_path))
        self.assertFalse(logged_metrics[2]["rag_viewer"].inject)
        self.assertIsNotNone(fake_run.artifact)
        assert fake_run.artifact is not None
        self.assertEqual(fake_run.artifact.type, "rag-run")
        self.assertEqual(fake_run.artifact.metadata["task"], "rag")
        self.assertTrue(fake_run.artifact.metadata["valid_output"])
        self.assertEqual(fake_run.artifact.metadata["citation"]["coverage_mean"], 0.75)
        self.assertEqual(len(fake_run.artifact.files), 1)

    @staticmethod
    def _restore_wandb(original_wandb: object | None) -> None:
        if original_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original_wandb


if __name__ == "__main__":
    unittest.main()
