# Sweep Conclusions (Jan 28, 2026)

This page summarizes the full sweep in `sweep_live_v2.csv`, the nested-vs-delegation check in
`nested_vs_delegation.csv`, and the semantic probe in `semantic_embed_probe.csv`. It is a concise,
decision-oriented summary for the current codebase.

See also: `docs/real_benchmark_results.md` for a small real benchmark run with sentence-transformer embeddings.
See also: `docs/public_benchmark_results.md` for a 120-question public benchmark.

## Coverage and caveats
- Corpus: `examples/data/book_large.txt` (synthetic book + paraphrase notes).
- Sweep coverage: **384 rows total (full grid)** across selectors/policies/depths/delegations.
- Tasks: 8 synthetic QA tasks; no repeats.
- Limits: `SWEEP_TOP_K=1`, `SWEEP_MAX_HITS=1`, strict JSON protocol.
- Embedding selector uses the current **hash-vector stub** (lexical), not real embeddings.

## Key takeaways (full sweep + follow-ups)
- **Depth helps a lot.** Overall: depth=1 **43.2%** vs depth=2 **83.3%** accuracy.
- **Best practical config** on this corpus is **bm25 + expand + depth=2 + delegation=llm** (8/8).
- **Nested recursion is dominated by plain delegation** on both cost and accuracy in this sweep.
- **Selector choice matters at depth=1**, especially for bm25 vs grep/stuff.
- **Embeddings show only a small edge** in the semantic probe, and the current embed_search is lexical.

## Depth=1 vs Depth=2 (overall sweep)

| Depth | Accuracy | P50 latency | P50 model_input_chars | P50 surfaced_chars |
| --- | --- | --- | --- | --- |
| 1 | **43.2%** | **4.53s** | **2,880** | **163** |
| 2 | **83.3%** | **8.03s** | **8,850** | **1,770** |

Depth=2 / Depth=1 ratios (median): **1.77x latency**, **3.07x model input**, **10.83x surfaced chars**.

### Depth medians by delegation
- Depth=1 delegation=none: **45.8%** accuracy, p50 **3.09s**
- Depth=1 delegation=llm: **40.6%** accuracy, p50 **5.21s**
- Depth=2 delegation=llm: **86.5%** accuracy, p50 **5.79s**
- Depth=2 delegation=rlm: **80.2%** accuracy, p50 **9.96s**

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

## Evidence highlights

**RLM-style iteration helps.** Example: bm25+expand goes from **62.5% at depth=1 (none)** to
**100% at depth=2 (llm)** on this corpus.

**Nested recursion is not the winner here.** Example: bm25+expand depth=2 **rlm = 87.5%** vs
bm25+expand depth=2 **llm = 100%**, with rlm showing much higher model_input and surfaced chars.

## Nested recursion vs plain delegation (hard tasks)

File: `nested_vs_delegation.csv` (t4, t5, t9; grep + single; LLM verify on)

- `depth2_rlm`: 3/3 correct, **0 subcalls**, avg latency **12.71s**, avg model_input **10,192**
- `depth1_llm`: 3/3 correct, **2 subcalls/run**, avg latency **10.91s**, avg model_input **3,222**

Small-N, but it finally exercises subcalls with a strict verify step.

## Semantic probe (bm25 vs embed, single policy)

File: `semantic_embed_probe.csv` (5 semantic tasks, `book_semantic.txt`)

- Depth=1: **bm25 100%**, **embed 100%**
- Depth=2: **bm25 80%**, **embed 100%**

These tasks are short and **the embedding search is lexical** (hash-vector), so this does *not*
prove true semantic superiority. It does show that embeddings do not hurt under the current stub.

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
- True semantic embeddings (current embed_search is hash-vector / lexical).
- Multi-hop tasks with k>1 and broader retrieval.
- Robust subcall vs recursion tradeoffs beyond the small 3-task head-to-head.
