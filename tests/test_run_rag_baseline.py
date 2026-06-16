from __future__ import annotations

import unittest

from scripts.run_rag_baseline import build_generation_requests, build_rag_responses
from trec26_rag.generator import AnswerGenerationRequest, EvidenceDocument
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


if __name__ == "__main__":
    unittest.main()
