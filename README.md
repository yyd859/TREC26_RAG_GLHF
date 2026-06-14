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

## Validate A Runfile

```bash
python scripts/validate_retrieval_run.py \
  --runfile outputs/r_output_trec_rag_2026.tsv \
  --topics data/trec_rag_2026_queries.jsonl
```

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

`PYSERINI_API_TOKEN` is only needed for real retrieval runs.
