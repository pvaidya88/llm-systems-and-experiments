# Benchmark Dataset Format

This benchmark expects two JSONL files:

## `corpus.jsonl`
Each line is a JSON object with at least:

```json
{"doc_id": "doc-1", "text": "...", "metadata": {"source": "..."}}
```

Optional fields:
- `title`
- `metadata`
- `updated_at`
- `source`

## `queries.jsonl`
Each line is a JSON object with:

```json
{
  "qid": "q1",
  "question": "...",
  "answer": "...",
  "bucket": "low_overlap",
  "gold_doc_ids": ["doc-1"],
  "gold_spans": [{"doc_id": "doc-1", "start": 120, "end": 200}],
  "gold_snippets": ["exact snippet"],
  "updated_at": "2025-01-01"
}
```

Fields:
- `answer` is optional but required for EM/F1.
- `gold_doc_ids` required for retrieval recall.
- `gold_spans` or `gold_snippets` improve citation checks.
- `bucket` is optional; the harness can infer buckets.

## Artifacts
The harness writes:
- `artifacts/{run_id}/per_query.jsonl`
- `artifacts/{run_id}/summary.json`
- `artifacts/{run_id}/summary.csv`
