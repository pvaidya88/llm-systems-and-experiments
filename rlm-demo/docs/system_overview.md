# System overview (public analogue)

## Goal
A minimal evaluation harness that mirrors production RAG/retrieval release gates.

## High-level flow

[Corpus] -> [Chunking] -> [Index (BM25/Dense)] -> [Retrieve] -> [Optional Rerank] -> [Answer/Extract] -> [Metrics + Gates]

## Components
- `bench/` : end-to-end benchmark harness
- `bench/index/` : BM25 (sqlite FTS5) and dense index (in-memory)
- `bench/pipelines.py` : reference pipelines (bm25_only, dense_extractive, vector_oracle, hybrid_rag)
- `bench/gates.py` : non-inferiority gate logic
- `bench/compare_runs.py` : baseline vs candidate gate evaluation

## Metrics
- EM/F1 for answer quality (when answers are comparable)
- Recall@k and nDCG@10 for retrieval quality
- Latency p50/p95

## Failure modes to watch
- Retrieval misses on low-overlap queries
- Latency spikes due to slow embedding/rerank paths
- Silent regressions hidden by small smoke sets

## Operational analogues
- CI smoke gates: fast signal for regressions
- Public benchmark: larger dataset for credibility
- Release decision doc: maps gates to ship/rollback criteria
