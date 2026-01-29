# Public benchmark results (SQuAD-120)

Date: Jan 29, 2026

This run uses a public dataset (SQuAD v1.1 subset) with real embeddings and reports both a
retrieval ceiling (oracle extractive) and a non-LLM heuristic extractor.

## Dataset
- 120-question ASCII-only subset from SQuAD v1.1 (Wikipedia)
- Built with `bench/tools/build_squad_mini.py --max-questions 120`
- Files: `bench/data/squad_public_corpus.jsonl`, `bench/data/squad_public_queries.jsonl`
- License: Wikipedia content is CC BY-SA 4.0; SQuAD v1.1 derived from Wikipedia (Rajpurkar et al.)

## Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Backend: sentence-transformers

## Runs

### 1) Retrieval ceiling (oracle extractive)
Config: `bench/configs/squad_public_oracle.json`

| Metric | Value |
| --- | --- |
| EM / Accuracy | 0.992 |
| F1 | 0.992 |
| Recall@10 | 0.992 |
| nDCG@10 | 0.978 |
| P50 latency | 9.28 ms |
| P95 latency | 12.19 ms |

Notes:
- `vector_oracle` returns the gold answer if it appears in retrieved text. This isolates retrieval quality.
- Not a generative end-to-end baseline.

### 2) Heuristic extractive (non-oracle, non-LLM)
Config: `bench/configs/squad_public.json`

| Metric | Value |
| --- | --- |
| EM / Accuracy | 0.000 |
| F1 | 0.052 |
| Recall@10 | 0.992 |
| nDCG@10 | 0.978 |
| P50 latency | 9.08 ms |
| P95 latency | 11.29 ms |

Notes:
- `dense_extractive` selects the sentence with highest token overlap; it is intentionally simple and weak.
- This provides a lower bound without requiring API calls.

## Reproduce
```powershell
cd rlm-demo
python -m bench.run --config bench/configs/squad_public_oracle.json
python -m bench.run --config bench/configs/squad_public.json
```
Requires: `pip install datasets sentence-transformers`
