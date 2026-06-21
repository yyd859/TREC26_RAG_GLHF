from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from trec26_rag.pyserini_client import PyseriniClient, SearchHit, extract_document_text


class PyseriniClientTest(unittest.TestCase):
    def test_extract_document_text_handles_common_shapes(self) -> None:
        self.assertEqual(extract_document_text({"text": "hello   world"}), "hello world")
        self.assertEqual(extract_document_text({"doc": {"contents": "document body"}}), "document body")
        self.assertEqual(
            extract_document_text({"segments": [{"body": "first"}, {"body": "second"}]}),
            "first second",
        )

    def test_fetch_doc_uses_doc_endpoint(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"docid": "shard_1", "doc": {"contents": "full text"}}
        with patch("trec26_rag.pyserini_client.requests.get", return_value=response) as get:
            client = PyseriniClient("http://example.test", "climbmix-400b", token="token")
            doc = client.fetch_doc("shard_1")
        self.assertEqual(doc, {"contents": "full text"})
        get.assert_called_once_with(
            "http://example.test/v1/climbmix-400b/doc/shard_1",
            headers={"Authorization": "Bearer token"},
            timeout=None,
        )

    def test_search_uses_explicit_timeout_when_configured(self) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {"candidates": [{"docid": "doc-a", "score": 1.0}]}
        with patch("trec26_rag.pyserini_client.requests.get", return_value=response) as get:
            client = PyseriniClient(
                "http://example.test",
                "climbmix-400b",
                token="token",
                timeout_seconds=120,
            )
            hits = client.search("query", hits=10)
        self.assertEqual(hits[0].docid, "doc-a")
        get.assert_called_once_with(
            "http://example.test/v1/climbmix-400b/search",
            params={"query": "query", "hits": 10},
            headers={"Authorization": "Bearer token"},
            timeout=120,
        )

    def test_search_retries_retryable_status_with_backoff(self) -> None:
        overloaded = Mock()
        overloaded.status_code = 429
        overloaded.headers = {"Retry-After": "1"}
        overloaded.text = "slow down"
        ok = Mock()
        ok.status_code = 200
        ok.headers = {}
        ok.json.return_value = {"candidates": [{"docid": "doc-a", "score": 1.0}]}
        with (
            patch("trec26_rag.pyserini_client.requests.get", side_effect=[overloaded, ok]) as get,
            patch("trec26_rag.pyserini_client.time.sleep") as sleep,
        ):
            client = PyseriniClient(
                "http://example.test",
                "climbmix-400b",
                token="token",
                max_retries=1,
                retry_backoff_seconds=1.0,
                min_request_interval_seconds=0,
            )
            hits = client.search("query", hits=10)
        self.assertEqual(hits[0].docid, "doc-a")
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_hydrate_hits_fetches_when_text_is_missing_or_short(self) -> None:
        hits = [
            SearchHit("doc-a", 1, 10.0, {"contents": "short"}),
            SearchHit("doc-b", 2, 9.0, {"contents": "already enough text"}),
            SearchHit("doc-c", 3, 8.0, None),
        ]
        client = PyseriniClient("http://example.test", "climbmix-400b", token="token")
        with patch.object(client, "fetch_doc", side_effect=[{"contents": "full text a"}]) as fetch_doc:
            hydrated = client.hydrate_hits(hits, min_text_chars=10, max_docs=1)
        self.assertEqual(hydrated[0].text, "full text a")
        self.assertEqual(hydrated[1].text, "already enough text")
        self.assertIsNone(hydrated[2].doc)
        fetch_doc.assert_called_once_with("doc-a")


if __name__ == "__main__":
    unittest.main()
