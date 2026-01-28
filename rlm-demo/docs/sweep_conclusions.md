# Sweep Conclusions (Jan 28, 2026)

This page summarizes the full sweep captured in `sweep_live_v2.csv`, a nested-vs-delegation check in
`nested_vs_delegation.csv`, and a semantic probe in `semantic_embed_probe.csv`. It is meant to be a
one-page, decision-oriented summary.

## Coverage and caveats
- Corpus: `examples/data/book_large.txt` (synthetic book + paraphrase notes).
- Sweep coverage: **384 rows total (full grid)** across selectors/policies/depths/delegations.
- Tasks: 8 synthetic QA tasks; no repeats.
- Limits: `SWEEP_TOP_K=1`, `SWEEP_MAX_HITS=1` and strict JSON protocol.

## High-level takeaways (observed, partial)
- **Depth=2 improves accuracy** on this corpus, but with higher cost.
- **Selector choice matters at depth=1**; BM25 outperforms grep/stuff in the observed runs.
- **Depth=2 flattens some selector differences** but not all (grep underperforms BM25 in the recursive runs).
- **Subcall evidence exists, but tiny** (2 tasks, 1 policy).

## Depth=1 vs Depth=2 (overall sweep)

| Depth | Accuracy | P50 latency | P50 model_input_chars | P50 surfaced_chars |
| --- | --- | --- | --- | --- |
| 1 | **43.2%** | **4.53s** | **2,880** | **163** |
| 2 | **83.3%** | **8.03s** | **8,850** | **1,770** |

Depth=2 / Depth=1 ratios (median): **1.77× latency**, **3.07× model input**, **10.83× surfaced chars**.

## Recommended defaults (based on observed data)

**Accuracy-first (delegation-heavy):**
- `depth=2`, `selector=bm25`, `policy=expand`, `delegation=llm`
- Observed: **100%** accuracy, **p50 4.73s**, **3,297** model_input_chars, **164** surfaced chars

**Latency-first (cheap):**
- `depth=1`, `selector=bm25`, `policy=expand`, `delegation=none`
- Observed: **62.5%** accuracy, **p50 2.42s**, **2,471** model_input_chars, **282** surfaced chars

**Recursive (no subcall) fallback:**
- `depth=2`, `selector=bm25`, `policy=expand`, `delegation=rlm`
- Observed: **87.5%** accuracy, **p50 8.46s**, **8,963** model_input_chars, **2,113** surfaced chars

These are the best two points to ship as defaults until we finish the embedding runs.

## Nested recursion vs plain delegation (hard tasks)

File: `nested_vs_delegation.csv` (t4, t5, t9; grep + single; LL M verify on)

- `depth2_rlm`: 3/3 correct, **0 subcalls**, avg latency **12.71s**, avg model_input **10,192**
- `depth1_llm`: 3/3 correct, **2 subcalls/run**, avg latency **10.91s**, avg model_input **3,222**

This is still small‑N, but it finally exercises subcalls with a strict verify step.

## Semantic probe (bm25 vs embed, single policy)

File: `semantic_embed_probe.csv` (5 semantic tasks, `book_semantic.txt`)

- Depth=1: **bm25 100%**, **embed 100%**
- Depth=2: **bm25 80%**, **embed 100%**

These tasks are still short and **the embedding search is lexical** (hash‑vector), so this does *not*
prove true semantic superiority. It does show that embeddings don’t hurt under the current stub.

## Plots and Pareto sets

Generated plot files (SVG):
- `plots/accuracy_vs_latency.svg`
- `plots/accuracy_vs_model_input.svg`
- `plots/accuracy_vs_surfaced.svg`

Aggregates and Pareto configs:
- `plots/sweep_agg.csv`
- `plots/sweep_pareto.csv`

## Boundaries / what this does NOT show yet
- Larger corpora or real benchmarks.
- True semantic embeddings (current embed_search is hash‑vector / lexical).
- Multi‑hop tasks with k>1 and broader retrieval.
- Robust subcall vs recursion tradeoffs beyond the small 3‑task head‑to‑head.
