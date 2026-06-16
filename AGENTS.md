# Agent Workflow

This repository uses a Level 2 experiment loop for TREC RAG 2026.

## Operating Rules

- Keep optimization changes reviewable through GitHub PRs.
- Prefer config-only changes under `configs/experiments/` for early iterations.
- Do not commit `.env.local`, `.curlrc.pyserini-rest`, W&B local state, or output runfiles.
- Log real experiment outputs to W&B artifacts.
- Run local tests before proposing changes:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Baseline Loop

1. Prepare development topics with `scripts/prepare_dev_topics.py`.
2. Run `scripts/run_retrieval_baseline.py`.
3. Sync metrics and artifacts to W&B with `--log-wandb`.
4. Run `scripts/propose_next_experiment.py` to create the next config.
5. Open or review a PR for the generated config before running it.

## Evaluation Layers

- `Level 0`: validator checks runfile format, complete topic coverage, rank order, and score order.
- `Level 1`: diagnostics summarize empty topics, duplicate docs, score stats, candidate counts, and latency.
- `Level 2`: optional qrels-based metrics add `nDCG@10`, `Recall@100`, `MAP`, and `MRR`.

Level 2 should be treated as optional. If `evaluation.qrels_path` is null or
missing, do not block the baseline run.

## RAG Config

The YAML config includes a `rag` section with `enabled`, `evidence_top_k`,
`generator_provider`, `model`, `prompt_template`, and `max_output_tokens`.
Retrieval workflows preserve this section but do not consume it. RAG workflows
should use `configs/baseline_rag.yaml` and `scripts/run_rag_baseline.py`.

Use `PyseriniClient.hydrate_hits(...)` for RAG evidence preparation when search
hits do not include enough document text. Do not duplicate ClimbMix document
fetch logic in runner scripts.

Use `AnthropicBatchAnswerGenerator` for the first RAG generator path. It uses
Anthropic Message Batches by default with Claude Haiku 4.5 and requires
`ANTHROPIC_API_KEY` only when generation is actually submitted.

Use `write_rag_jsonl(...)` for RAG submissions. The default output file is
`rag_output_trec_rag_2026.jsonl` under the configured output directory.
The RAG runner validates output with `validate_rag_jsonl(...)` before returning
success, and the GitHub workflow uploads all files under `outputs/`.

RAG W&B logging should include scalar proxy metrics, citation diagnostics, and
a `rag-run` artifact with the RAG JSONL, validation report, proxy metrics JSON,
citation diagnostics JSON, a self-contained `rag_viewer.html`, raw Anthropic
batch results, and run config.

## Allowed Early Optimization Surface

Early agent-generated experiments should change only:

- `retrieval.query_template`
- `retrieval.hits`
- `experiment.name`
- `experiment.hypothesis`
- `experiment.run_id`
- `wandb.tags`
- `evaluation.qrels_path`
- `evaluation.relevance_threshold`

Changing retrieval client behavior, output formats, or validation rules should be a separate human-reviewed PR.
