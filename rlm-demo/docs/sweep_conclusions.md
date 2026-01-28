# Sweep Conclusions (Jan 28, 2026)

This page summarizes the partial sweep captured in `sweep_live_v2.csv` and a tiny subcall check in
`llm_subcall_mini.csv`. It is meant to be a one-page, decision-oriented summary.

## Coverage and caveats
- Corpus: `examples/data/book_large.txt` (synthetic book + paraphrase notes).
- Sweep coverage: 294 rows total; **embedding selector runs are incomplete** (all `embed/*` depth=2 and
  `embed/expand|iterative` depth=1 are missing).
- Tasks: 8 synthetic QA tasks; no repeats.
- Limits: `SWEEP_TOP_K=1`, `SWEEP_MAX_HITS=1` and strict JSON protocol.

## High-level takeaways (observed, partial)
- **Depth=2 improves accuracy** on this corpus, but with higher cost.
- **Selector choice matters at depth=1**; BM25 outperforms grep/stuff in the observed runs.
- **Depth=2 flattens some selector differences** but not all (grep underperforms BM25 in the recursive runs).
- **Subcall evidence exists, but tiny** (2 tasks, 1 policy).

## Depth=1 vs Depth=2 (overall, partial sweep)

| Depth | Accuracy | P50 latency | P50 model_input_chars | P50 surfaced_chars |
| --- | --- | --- | --- | --- |
| 1 | **44.7%** | **4.25s** | **2,852** | **155** |
| 2 | **84.7%** | **7.69s** | **6,322** | **998** |

Depth=2 / Depth=1 ratios (median): **1.81× latency**, **2.22× model input**, **6.44× surfaced chars**.

## Recommended defaults (based on observed data)

**Accuracy-first (recursive):**
- `depth=2`, `selector=bm25`, `policy=expand`, `delegation=rlm`
- Observed: **87.5%** accuracy, **p50 8.46s**, **8,963** model_input_chars, **2,113** surfaced chars

**Latency-first (cheap):**
- `depth=1`, `selector=bm25`, `policy=expand`, `delegation=none`
- Observed: **62.5%** accuracy, **p50 2.42s**, **2,471** model_input_chars, **282** surfaced chars

These are the best two points to ship as defaults until we finish the embedding runs.

## Subcall check (single-policy, tiny)

File: `llm_subcall_mini.csv` (grep + single, t4/t5 only)

- `depth2_rlm`: 2/2 correct, **0 subcalls**, avg latency **14.29s**
- `depth1_llm`: 2/2 correct, **1 subcall per run**, avg latency **9.97s**

This is only **2 tasks**, so it’s **evidence of subcall usage**, not a general performance claim.

## Plots and Pareto sets

Generated plot files (SVG):
- `plots/accuracy_vs_latency.svg`
- `plots/accuracy_vs_model_input.svg`
- `plots/accuracy_vs_surfaced.svg`

Aggregates and Pareto configs:
- `plots/sweep_agg.csv`
- `plots/sweep_pareto.csv`

## Boundaries / what this does NOT show yet
- Embedding selector performance (missing runs).
- Larger corpora or real benchmarks.
- True multi-hop tasks with k>1 and broader retrieval.
- Robust subcall vs recursion tradeoffs beyond the tiny check above.
