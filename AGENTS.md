# Agent Workflow

This repository uses a Level 2 experiment loop for TREC RAG 2026.

## Operating Rules

- Keep core code changes reviewable through GitHub PRs.
- Run routine experiment optimization on dedicated `codex/autoresearch-*`
  branches without requiring PR review before execution.
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
5. For autoresearch iterations, commit the generated config on a dedicated
   experiment branch and run the matching workflow on that branch.

## Autoresearch Loop

Autoresearch v1 keeps the agent local and uses GitHub Actions as the compute
runner. The agent stays at Level 2 during routine optimization: it may propose
config-only changes under `configs/experiments/`, but it must not directly
change core code. The policy lives in `configs/autoresearch.yaml`.

The intended loop is:

1. Agent reads `research-context`: historical experiment configs, W&B evals,
   current best run, tried signatures, and research memory.
2. Agent proposes the next config-only experiment.
3. Agent creates a unique `codex/autoresearch-*` branch and pushes the config.
4. The branch push triggers the matching retrieval/RAG workflow.
5. The workflow logs metrics and artifacts to W&B.
6. Agent monitors the branch workflow, summarizes to W&B, updates
   `outputs/autoresearch_memory.json`, and starts the next round.

Use the router command for local orchestration:

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

Use `dry-run` before touching GitHub/W&B. It performs a local no-network
simulation that creates a temporary proposal under `configs/experiments/`,
validates safety constraints, builds the workflow payload, and emits a
monitor-style summary.

Prefer W&B as the durable home for experiment results and autoresearch
summaries. GitHub should stay focused on PR review, workflow status, and
minimal logs; do not add extra GitHub step summaries unless a human explicitly
asks for them.

For routine local autoresearch, create a new experiment branch and let the push
trigger the matching workflow:

```bash
python scripts/autoresearch.py iterate --route retrieval --ref main --limit 2
```

This reads W&B history, proposes a config under `configs/experiments/`, commits
it to a new `codex/autoresearch-*` branch, and pushes that branch. The
retrieval and RAG workflows both listen to autoresearch branch pushes, resolve
the latest config, infer its route, and skip themselves if the config belongs to
the other route. When `iterate --limit` is used, the limit is written to
`runtime.limit` inside the generated config so push-triggered workflows can
honor the same smoke/full-run setting.

Then summarize the latest GitHub Actions status:

```bash
python scripts/autoresearch.py monitor --route retrieval --branch <codex/autoresearch-branch> --include-wandb --log-wandb --update-memory
```

The supported routes are:

- `retrieval`: runs `run-retrieval-baseline.yml`
- `rag`: runs `run-rag-baseline.yml`
- `evaluation-only`: reruns the retrieval workflow for evaluation-focused configs
- `proposer-only`: creates a config proposal without triggering an experiment

The default runner selection is documented in `configs/autoresearch.yaml`:
the local agent is the orchestrator brain, GitHub Actions is the default CPU
runner, and self-hosted GPU runners, Modal/RunPod/Vast, and Codex automations
remain candidate compute/orchestration backends.

Autoresearch safety rules:

- Generated files must stay under `configs/experiments/`.
- Do not commit API keys, tokens, passwords, or credentials in YAML.
- `runtime.limit` is allowed only as workflow runtime metadata for branch-push
  experiments.
- Routine experiment iterations do not require PRs. Use PRs to promote useful
  configs to shared branches or to change core code.
- Local agent pushes should use a normal GitHub-authenticated remote/CLI so
  branch pushes trigger workflows. If running the GitHub Orchestrator workflow
  itself, use `AUTORESEARCH_GITHUB_TOKEN` when the default `GITHUB_TOKEN` cannot
  push branches or dispatch another workflow.

Bootstrap note: new workflow files usually need to land on the default branch
before they appear in the GitHub Actions UI or can trigger reliably. This
bootstrap rule is for workflow/code changes, not for routine experiment
branches.

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
batch results, `rag_outputs_table.csv`, `rag_outputs_table.jsonl`, and run
config. It should also log a W&B table named `rag_outputs` for per-topic
inspection and W&B HTML media named `rag_viewer` for opening the viewer from
the run page without downloading the artifact.

## RAG Smoke Tests

Use a two-topic smoke test before treating RAG workflow changes as ready:

```bash
python scripts/prepare_dev_topics.py
python scripts/run_rag_baseline.py \
  --config configs/baseline_rag.yaml \
  --limit 2 \
  --log-wandb
```

When using GitHub Actions, trigger **Run RAG Baseline** manually on the branch
under test with `config=configs/baseline_rag.yaml` and `limit=2`. The workflow
is allowed on all branches so feature branches can be validated before merging
to `dev`.

A passing RAG smoke test should produce `outputs/rag_output_trec_rag_2026.jsonl`,
`outputs/rag_validation_report.json`, `outputs/rag_viewer.html`,
`outputs/rag_outputs_table.csv`, and `outputs/rag_outputs_table.jsonl`. In W&B,
confirm scalar metrics, the `rag_outputs` table, and the `rag_viewer` HTML media
exist. Treat nonzero `rag_validation_error_count` as a failed smoke test.

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
