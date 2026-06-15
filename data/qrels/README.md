# Qrels

Optional retrieval qrels can be placed here for Level 2 evaluation.

For the current TREC RAG 2026 development data, the public qrels observed so far
cover the RAG25 development subset, not the ResearchRubrics development topics.

Example config:

```yaml
evaluation:
  qrels_path: data/qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning.qrels
  relevance_threshold: 1
```
