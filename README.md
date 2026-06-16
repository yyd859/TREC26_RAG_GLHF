# TREC26_RAG_GLHF

Workspace for the TREC 2026 RAG track by team GLHF.

This repo starts with a Level 2 experiment workflow:

1. Humans or Codex open a branch/PR for code and config changes.
2. The retrieval baseline runs against the official Pyserini REST API.
3. Weights & Biases records configs, metrics, validation reports, and runfiles.
4. The experiment proposer reads W&B history and creates the next config-only PR.
5. A human reviews the PR before the next real experiment runs.

The important guardrail is that optimization starts by changing configs under
`configs/experiments/`, not arbitrary pipeline code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create `.env.local` locally if you have a Pyserini token:

```bash
PYSERINI_API_TOKEN=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
WANDB_PROJECT=trec26-rag-glhf
WANDB_ENTITY=...
```

Do not commit `.env.local` or `.curlrc.pyserini-rest`.

## Run The Retrieval Baseline

Put development topics at `data/trec_rag_2026_queries.jsonl`, then run:

```bash
python scripts/prepare_dev_topics.py
python scripts/run_retrieval_baseline.py \
  --config configs/baseline_retrieval.yaml \
  --log-wandb
```

For a small smoke test:

```bash
python scripts/run_retrieval_baseline.py \
  --config configs/baseline_retrieval.yaml \
  --limit 2
```

The baseline writes:

- `outputs/r_output_trec_rag_2026.tsv`
- `outputs/retrieval_validation_report.json`

## RAG Config

The shared YAML config includes a placeholder RAG section for the upcoming RAG
baseline:

```yaml
rag:
  enabled: false
  evidence_top_k: 5
  generator_provider: anthropic_batch
  model: claude-haiku-4-5-20251001
  prompt_template: |
    Answer the topic using only the provided ClimbMix evidence.
  max_output_tokens: 800
```

The current retrieval baseline does not generate RAG answers yet. This section
exists so future RAG experiments can be config-driven without changing the
retrieval workflow contract.

The Pyserini client already supports fetching full ClimbMix documents by
`docid` and hydrating search hits when the search response lacks enough text.
Future RAG runners should use this instead of duplicating document fetch logic.

The minimal generator wrapper supports Anthropic Message Batches using
`ANTHROPIC_API_KEY`; by default it targets Claude Haiku 4.5.

RAG output writing is available through `write_rag_jsonl(...)` and defaults to
`outputs/rag_output_trec_rag_2026.jsonl`.

RAG JSONL validation is available with:

```bash
python scripts/validate_rag_output.py \
  --rag-output outputs/rag_output_trec_rag_2026.jsonl \
  --topics data/trec_rag_2026_queries.jsonl
```

## Evaluation Layers

The validation report separates evaluation into three layers:

- `Level 0`: runfile validity and completeness. This checks TREC column format,
  rank order, score order, missing topics, and validation errors.
- `Level 1`: retrieval diagnostics. This records empty topic count, duplicate
  doc rate, score statistics, candidate count distribution, and latency.
- `Level 2`: relevance metrics. When qrels are configured, this adds
  `nDCG@10`, `Recall@100`, `MAP`, and `MRR`.

## Validate A Runfile

```bash
python scripts/validate_retrieval_run.py \
  --runfile outputs/r_output_trec_rag_2026.tsv \
  --topics data/trec_rag_2026_queries.jsonl
```

## Level 2 Retrieval Evaluation

If qrels are available, add them to `data/qrels/` and set:

```yaml
evaluation:
  qrels_path: data/qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning.qrels
  relevance_threshold: 1
```

The baseline runner will add `nDCG@10`, `Recall@100`, `MAP`, and `MRR` to the
validation report and W&B metrics. You can also evaluate an existing runfile:

```bash
python scripts/evaluate_retrieval_run.py \
  --runfile outputs/r_output_trec_rag_2026.tsv \
  --qrels data/qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning.qrels
```

If `evaluation.qrels_path` is `null`, Level 2 is skipped and
`level2_evaluation_enabled` is logged as `0`.

## Propose The Next Experiment

After at least one W&B run exists:

```bash
python scripts/propose_next_experiment.py \
  --base-config configs/baseline_retrieval.yaml \
  --output-dir configs/experiments
```

The script reads recent W&B runs, chooses the best valid run, and writes a new
config file with a single main change. GitHub Actions can then open a PR for
that config.

## Local Checks

Without installing the package:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

After `python -m pip install -e ".[dev]"`:

```bash
python -m unittest discover -s tests
```

## GitHub Secrets

Recommended repository secrets:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `PYSERINI_API_TOKEN`
- `ANTHROPIC_API_KEY`

`PYSERINI_API_TOKEN` is only needed for real retrieval runs.
`ANTHROPIC_API_KEY` is only needed for RAG generation runs.
