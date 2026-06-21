# TREC26_RAG_GLHF

Workspace for the TREC 2026 RAG track by team GLHF.

This repo starts with a Level 2 experiment workflow:

1. Humans or Codex open a branch/PR for code changes.
2. The retrieval baseline runs against the official Pyserini REST API.
3. Weights & Biases records configs, metrics, validation reports, and runfiles.
4. The local autoresearch agent reads historical configs plus W&B evals, creates a config-only experiment branch, and lets the branch push trigger the matching workflow.
5. W&B is the experiment ledger: workflow outputs, summaries, research memory, and per-run artifacts should live there. PRs are reserved for code changes or promoting a shared baseline.

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

## External GPU Runner

Use a Vast AI GPU instance as a GitHub self-hosted runner when an experiment
needs local GPU compute. Keep normal retrieval/RAG baselines on
`ubuntu-latest`; route GPU-only jobs to a runner labeled `gpu` and `vast-ai`.

The first smoke-test workflow is **GPU Runner Smoke**. It targets:

```yaml
runs-on: [self-hosted, linux, x64, gpu, vast-ai]
```

Recommended setup:

1. Create a Vast AI instance from an NVIDIA CUDA/PyTorch image with enough disk
   for the model/cache you want to test.
2. In GitHub, open repository **Settings -> Actions -> Runners -> New
   self-hosted runner** and copy the Linux x64 setup commands.
3. On the Vast instance, run the generated download/config commands, but add
   labels and make the runner ephemeral:

```bash
./config.sh \
  --url https://github.com/yyd859/TREC26_RAG_GLHF \
  --token <GITHUB_GENERATED_RUNNER_TOKEN> \
  --labels gpu,vast-ai,cuda \
  --ephemeral
./run.sh
```

4. Trigger **Actions -> GPU Runner Smoke**. A healthy runner should show
   `nvidia-smi`, install the package, and pass unit tests.
5. Stop or destroy the Vast instance after the job. Treat rented self-hosted
   runners as disposable because they can access repository checkout contents
   and workflow secrets used by jobs routed to them.

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
config file with a single main change. For autonomous iterations, prefer
`scripts/autoresearch.py iterate`, which commits the config on a new experiment
branch and lets the branch push trigger the matching workflow.

## Autoresearch V1

Autoresearch is the branch-native experiment loop inspired by
`karpathy/autoresearch`, adapted to this repo's Level 2 constraint: the local
agent does not edit core code during routine optimization. It reads all
historical experiment configs, recent W&B evals, current best-run state, and
research memory before choosing the next config-only experiment.

V1 treats the local Codex/Claude agent as the research brain, GitHub Actions as
the CPU/GPU runner layer, and W&B as the durable experiment ledger. The agent
pushes a unique `codex/autoresearch-*` branch; that push triggers the matching
retrieval or RAG workflow, and the workflow logs outputs back to W&B. PRs are
not part of the routine loop; use them only for core code changes or for
promoting a shared baseline.

The policy file is:

```bash
configs/autoresearch.yaml
```

It defines:

- allowed paths: only `configs/experiments/`
- routes: `retrieval`, `rag`, `evaluation-only`, and `proposer-only`
- objective metric and direction
- GitHub workflow mapping for each experiment route
- branching mode: direct experiment branches for routine iterations
- branch push triggers for `codex/autoresearch-*`
- research memory at `outputs/autoresearch_memory.json`, mirrored to W&B

Useful local commands:

```bash
python scripts/autoresearch.py routes
python scripts/autoresearch.py best-run
python scripts/autoresearch.py research-context --max-runs 50
python scripts/autoresearch.py iterate --route retrieval --ref main --limit 2
python scripts/autoresearch.py monitor --route retrieval --branch <codex/autoresearch-branch> --include-wandb --log-wandb --update-memory
python scripts/autoresearch.py loop --route retrieval --max-rounds 3 --poll-seconds 120
python scripts/autoresearch.py propose --route retrieval
python scripts/autoresearch.py propose --route rag
python scripts/autoresearch.py check configs/experiments/
python scripts/autoresearch.py dry-run --route retrieval --ref main
python scripts/autoresearch.py dry-run --route rag --ref main
python scripts/autoresearch.py latest-config
python scripts/autoresearch.py route-config --config configs/experiments/<proposal>.yaml
```

`dry-run` is a no-network local simulation. It creates a temporary proposal
under `configs/experiments/`, validates policy constraints, builds the workflow
payload, and emits the same monitor summary shape used by real runs.
Use it before relying on GitHub/W&B credentials.

For a full branch-native iteration, run:

```bash
python scripts/autoresearch.py iterate --route retrieval --ref main --limit 2
```

This creates a new `codex/autoresearch-*` branch, commits the generated config,
and pushes the branch. The retrieval and RAG workflows both listen to these
branch pushes, resolve the latest experiment config from the commit, infer the
route from `experiment.task`, and skip themselves if the config belongs to the
other route. When `iterate --limit` is used, the generated config includes
`runtime.limit` so push-triggered workflows can honor the same smoke/full-run
setting without manual workflow inputs.

Then monitor the route or a specific branch:

```bash
python scripts/autoresearch.py monitor --route retrieval --branch <codex/autoresearch-branch> --include-wandb --log-wandb --update-memory
```

`research-context` is the main handoff point for the local agent: it prints a
single JSON object containing historical configs, W&B summaries, best-run
selection, tried config signatures, and research memory. The bounded `loop`
command wires together propose, branch, push, monitor, W&B summary logging, and
memory update for unattended overnight runs.

Local `iterate` uses your normal Git remote/CLI auth. `monitor`, `dispatch`,
and the GitHub API-backed pieces require `GITHUB_TOKEN`. W&B inspection and
logging require `WANDB_API_KEY`, `WANDB_PROJECT`, and optionally `WANDB_ENTITY`.

GitHub Actions also has **Autoresearch Orchestrator**, which can:

- run a branch-native experiment iteration manually or on the weekly schedule
- generate a config-only proposal locally for inspection
- show the current W&B best run
- dispatch a config to the retrieval/RAG workflow on the selected branch
- monitor the latest workflow status and log autoresearch summaries to W&B

Keep GitHub lightweight: Actions should provide trigger/status/log plumbing and
compute. Experiment comparisons, best-run state, autoresearch summaries, and
research memory should live in W&B whenever credentials are available. The
monitor command prints JSON to logs for debugging, but it does not write an
extra GitHub step summary.

Bootstrap note: a newly added workflow must be merged to the repository default
branch before it reliably appears in the GitHub Actions UI or can trigger from
push events. That rule is for workflow/code changes, not routine experiment
branches.

For workflow dispatch or branch creation from inside another GitHub workflow, add
`AUTORESEARCH_GITHUB_TOKEN` if the default `GITHUB_TOKEN` cannot trigger the
target workflow or push branches in this repository. The orchestrator falls back
to the default token when this secret is not present.

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
default GitHub Actions token from pushing experiment branches or dispatching
downstream workflows.
