from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalConfig:
    relevance_threshold: int = 1
    ndcg_cutoff: int = 10
    recall_cutoff: int = 100


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels_path = Path(path)
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Qrels line {line_number}: expected 4 columns, found {len(parts)}")
            topic_id, _unused, docid, relevance_text = parts
            try:
                relevance = int(relevance_text)
            except ValueError as exc:
                raise ValueError(f"Qrels line {line_number}: relevance must be an integer") from exc
            qrels.setdefault(topic_id, {})[docid] = relevance
    if not qrels:
        raise ValueError(f"No qrels found in {qrels_path}")
    return qrels


def read_runfile_rankings(path: str | Path) -> dict[str, list[str]]:
    runfile_path = Path(path)
    rankings: dict[str, list[tuple[int, str]]] = {}
    with runfile_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Runfile line {line_number}: expected 6 columns, found {len(parts)}")
            topic_id, _q0, docid, rank_text, _score, _run_id = parts
            try:
                rank = int(rank_text)
            except ValueError as exc:
                raise ValueError(f"Runfile line {line_number}: rank must be an integer") from exc
            rankings.setdefault(topic_id, []).append((rank, docid))

    deduped_rankings: dict[str, list[str]] = {}
    for topic_id, ranked_docids in rankings.items():
        seen: set[str] = set()
        ordered_docids: list[str] = []
        for _rank, docid in sorted(ranked_docids):
            if docid in seen:
                continue
            seen.add(docid)
            ordered_docids.append(docid)
        deduped_rankings[topic_id] = ordered_docids
    return deduped_rankings


def dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(docid, 0) for docid in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(
    ranking: list[str],
    qrels: dict[str, int],
    k: int,
    relevance_threshold: int,
) -> float:
    relevant_docids = {docid for docid, rel in qrels.items() if rel >= relevance_threshold}
    if not relevant_docids:
        return 0.0
    retrieved_relevant = relevant_docids.intersection(ranking[:k])
    return len(retrieved_relevant) / len(relevant_docids)


def average_precision(ranking: list[str], qrels: dict[str, int], relevance_threshold: int) -> float:
    relevant_docids = {docid for docid, rel in qrels.items() if rel >= relevance_threshold}
    if not relevant_docids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, docid in enumerate(ranking, 1):
        if docid not in relevant_docids:
            continue
        hits += 1
        precision_sum += hits / index
    return precision_sum / len(relevant_docids)


def reciprocal_rank(ranking: list[str], qrels: dict[str, int], relevance_threshold: int) -> float:
    for index, docid in enumerate(ranking, 1):
        if qrels.get(docid, 0) >= relevance_threshold:
            return 1 / index
    return 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    eval_config = config or EvalConfig()
    per_topic: dict[str, dict[str, float | int]] = {}
    for topic_id in sorted(qrels):
        topic_qrels = qrels[topic_id]
        ranking = rankings.get(topic_id, [])
        relevant_count = sum(
            1 for relevance in topic_qrels.values() if relevance >= eval_config.relevance_threshold
        )
        metrics = {
            "retrieved_count": len(ranking),
            "relevant_doc_count": relevant_count,
            "ndcg@10": ndcg_at_k(ranking, topic_qrels, eval_config.ndcg_cutoff),
            "recall@100": recall_at_k(
                ranking,
                topic_qrels,
                eval_config.recall_cutoff,
                eval_config.relevance_threshold,
            ),
            "ap": average_precision(ranking, topic_qrels, eval_config.relevance_threshold),
            "rr": reciprocal_rank(ranking, topic_qrels, eval_config.relevance_threshold),
        }
        per_topic[topic_id] = metrics

    evaluated_topic_count = len(per_topic)
    qrel_topic_ids = set(qrels)
    run_topic_ids = set(rankings)
    summary = {
        "level2_evaluation_enabled": 1,
        "qrels_topic_count": len(qrel_topic_ids),
        "qrels_doc_count": sum(len(topic_qrels) for topic_qrels in qrels.values()),
        "evaluated_topic_count": evaluated_topic_count,
        "run_topics_with_qrels_count": len(qrel_topic_ids & run_topic_ids),
        "run_topics_without_qrels_count": len(run_topic_ids - qrel_topic_ids),
        "qrels_topics_missing_run_count": len(qrel_topic_ids - run_topic_ids),
        "relevance_threshold": eval_config.relevance_threshold,
        "ndcg@10": _mean_topic_metric(per_topic, "ndcg@10"),
        "recall@100": _mean_topic_metric(per_topic, "recall@100"),
        "map": _mean_topic_metric(per_topic, "ap"),
        "mrr": _mean_topic_metric(per_topic, "rr"),
    }
    return {
        "enabled": True,
        "metrics": summary,
        "per_topic": per_topic,
        "qrels_topics_missing_run": sorted(qrel_topic_ids - run_topic_ids),
        "run_topics_without_qrels": sorted(run_topic_ids - qrel_topic_ids),
    }


def evaluate_retrieval_run(
    runfile_path: str | Path,
    qrels_path: str | Path,
    relevance_threshold: int = 1,
) -> dict[str, Any]:
    qrels = read_qrels(qrels_path)
    rankings = read_runfile_rankings(runfile_path)
    return evaluate_rankings(
        rankings,
        qrels,
        EvalConfig(relevance_threshold=relevance_threshold),
    )


def _mean_topic_metric(per_topic: dict[str, dict[str, float | int]], metric: str) -> float:
    if not per_topic:
        return 0.0
    return sum(float(topic_metrics[metric]) for topic_metrics in per_topic.values()) / len(per_topic)
