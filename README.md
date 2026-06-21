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

## RAG Baseline

RAG runs use `configs/baseline_rag.yaml` by default and write
`outputs/rag_output_trec_rag_2026.jsonl` plus
`outputs/rag_validation_report.json`.

```bash
python scripts/run_rag_baseline.py \
  --config configs/baseline_rag.yaml \
  --limit 2
```

For a full local smoke test with W&B logging:

```bash
python scripts/prepare_dev_topics.py
python scripts/run_rag_baseline.py \
  --config configs/baseline_rag.yaml \
  --limit 2 \
  --log-wandb
```

For a GitHub Actions smoke test, open **Actions -> Run RAG Baseline -> Run
workflow**, choose the branch to test, and use:

- `config`: `configs/baseline_rag.yaml`
- `limit`: `2`

The RAG workflow can be manually dispatched on any branch. Repository secrets
must include `PYSERINI_API_TOKEN`, `ANTHROPIC_API_KEY`, `WANDB_API_KEY`,
`WANDB_PROJECT`, and usually `WANDB_ENTITY`.

The Pyserini client supports fetching full ClimbMix documents by
`docid` and hydrating search hits when the search response lacks enough text.
The RAG runner uses that path before submitting evidence to Anthropic Message
Batches. It requires `PYSERINI_API_TOKEN` and `ANTHROPIC_API_KEY`; W&B logging
also requires the W&B environment variables listed above.

When `--log-wandb` is enabled, the RAG run logs scalar validator and proxy
metrics plus a `rag-run` artifact containing:

- `rag_output_trec_rag_2026.jsonl`
- `rag_validation_report.json`
- `rag_proxy_metrics.json`
- `rag_citation_diagnostics.json`
- `rag_viewer.html`
- `rag_outputs_table.csv`
- `rag_outputs_table.jsonl`
- `anthropic_batch_results.jsonl`
- the config used for the run

The same per-topic rows are also logged to W&B as a table named
`rag_outputs`, so a run can be inspected without downloading JSONL artifacts.
The self-contained viewer is also logged as W&B HTML media named
`rag_viewer`, so it can be opened from the run page as well as from the
artifact snapshot.

After a successful smoke run, check:

- The Actions artifact `rag-baseline-outputs`.
- The W&B scalar metrics and `rag_outputs` table.
- The W&B HTML media panel `rag_viewer`.
- `rag_validation_error_count == 0` for a passing smoke run.

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

## Autoresearch V1

Autoresearch is the review-gated experiment loop inspired by
`karpathy/autoresearch`, adapted to this repo's Level 2 constraint: the agent
does not edit core code during routine optimization. It reads W&B/GitHub state,
chooses the next experiment, writes a config under `configs/experiments/`, and
opens a PR for review.

V1 selects **GitHub Actions scheduled/manual workflows** as the runner
environment. This is the lowest-friction option because the repo already uses
Actions secrets, workflow dispatch, PR review, and W&B logging. Self-hosted GPU
runners, local long-running agents, Modal/RunPod/Vast, and Codex automations are
documented as candidate backends in `configs/autoresearch.yaml`, but they are
not the default until the workflow needs long-running orchestration or GPU
compute.

The policy file is:

```bash
configs/autoresearch.yaml
```

It defines:

- allowed paths: only `configs/experiments/`
- routes: `retrieval`, `rag`, `evaluation-only`, and `proposer-only`
- objective metric and direction
- GitHub workflow mapping for each experiment route
- review mode: PR required

Useful local commands:

```bash
python scripts/autoresearch.py routes
python scripts/autoresearch.py best-run
python scripts/autoresearch.py propose --route retrieval
python scripts/autoresearch.py propose --route rag
python scripts/autoresearch.py check configs/experiments/
python scripts/autoresearch.py dry-run --route retrieval --ref main
python scripts/autoresearch.py dry-run --route rag --ref main
python scripts/autoresearch.py open-pr --head codex/autoresearch-v1
```

`dry-run` is a no-network local simulation. It creates a temporary proposal
under `configs/experiments/`, validates policy constraints, builds the workflow
dispatch payload, and emits the same monitor summary shape used by real runs.
Use it before relying on GitHub/W&B credentials.

After a generated config PR is reviewed and merged, trigger the matching
workflow:

```bash
python scripts/autoresearch.py dispatch \
  --route retrieval \
  --config configs/experiments/<proposal>.yaml \
  --ref main
```

Then monitor the route:

```bash
python scripts/autoresearch.py monitor --route retrieval --branch main
```

`dispatch` and `monitor` require `GITHUB_TOKEN`. W&B inspection requires
`WANDB_API_KEY`, `WANDB_PROJECT`, and optionally `WANDB_ENTITY`.

GitHub Actions also has **Autoresearch Orchestrator**, which can:

- propose a config-only PR manually or on the weekly schedule
- show the current W&B best run
- dispatch a reviewed config to the retrieval/RAG workflow
- monitor the latest workflow status and log autoresearch summaries to W&B

Keep GitHub lightweight: Actions should provide trigger/status/log plumbing,
while experiment comparisons, best-run state, and autoresearch summaries should
live in W&B whenever credentials are available. The monitor command prints JSON
to the Actions log for debugging, but it does not write an extra GitHub step
summary.

Bootstrap note: a newly added workflow such as `autoresearch.yml` must be
merged to the repository default branch before it reliably appears in the
GitHub Actions UI for manual dispatch. Use `scripts/autoresearch.py open-pr`
when `GITHUB_TOKEN` is available. If the GitHub connector or local CLI cannot
create the first PR because of permissions, the command prints a manual compare
URL for the pushed feature branch; open that PR manually, then use the workflow
after it lands on `main`.

For workflow dispatch or PR creation from inside another GitHub workflow, add
`AUTORESEARCH_GITHUB_TOKEN` if the default `GITHUB_TOKEN` cannot trigger the
target workflow or open pull requests in this repository. The orchestrator falls
back to the default token when this secret is not present.

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
- `AUTORESEARCH_GITHUB_TOKEN`

`PYSERINI_API_TOKEN` is only needed for real retrieval runs.
`ANTHROPIC_API_KEY` is only needed for RAG generation runs.
`AUTORESEARCH_GITHUB_TOKEN` is only needed when repository settings block the
default GitHub Actions token from opening PRs or dispatching downstream
workflows.
