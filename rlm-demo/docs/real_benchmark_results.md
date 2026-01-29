# Real benchmark results (SQuAD-mini)

Date: Jan 29, 2026

This run replaces the synthetic corpus + hash-vector stub with a small real benchmark and real embeddings.

## Dataset
- 12-question ASCII-only subset from SQuAD v1.1 (Wikipedia)
- Built with `bench/tools/build_squad_mini.py`
- Files: `bench/data/squad_mini_corpus.jsonl`, `bench/data/squad_mini_queries.jsonl`
- License: Wikipedia content is CC BY-SA 4.0; SQuAD v1.1 derived from Wikipedia (Rajpurkar et al.)

## Run config
- Config: `bench/configs/squad_mini.json`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Pipeline: `vector_oracle` (dense retrieval + oracle extractive answer)

Reproduce:
```powershell
cd rlm-demo
python -m bench.run --config bench/configs/squad_mini.json
```
Requires: `pip install datasets sentence-transformers`

## Results (N=12)

| Metric | Value |
| --- | --- |
| Accuracy (exact match) | 1.00 |
| Recall@10 | 1.00 |
| P50 latency | 43 ms |
| P95 latency | 78 ms |

Notes:
- `vector_oracle` uses the gold answer string to extract from retrieved chunks. This is an oracle-style
  extractor meant to isolate retrieval quality; it is not a generative LLM baseline.
- No citations are emitted in this pipeline, so `hallucination_rate` remains 1.0.
