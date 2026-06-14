# Data Directory

Place TREC RAG 2026 development topics here as:

```text
data/trec_rag_2026_queries.jsonl
```

The expected JSONL fields are:

- `id`
- `title`
- `narrative`

Large downloaded files are ignored by git. Keep durable experiment artifacts in
Weights & Biases instead of committing them here.
