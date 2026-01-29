# LLM Systems and Experiments

Evaluation-focused demos and applied ML systems. The strongest proof in this repo is the eval harness + sweep data below.

## Results at a glance (Jan 28, 2026)
Sweep: 384-row grid across selectors, policies, depths, and delegations.

| Depth | Accuracy | P50 latency | P50 model_input_chars | P50 surfaced_chars |
| --- | --- | --- | --- | --- |
| 1 | 43.2% | 4.53s | 2,880 | 163 |
| 2 | 83.3% | 8.03s | 8,850 | 1,770 |

Recommended defaults (from the sweep):
- Accuracy-first: bm25 + expand + depth=2 + delegation=llm (observed 100% accuracy, p50 4.73s)
- Latency-first: bm25 + expand + depth=1 + delegation=none (observed 62.5% accuracy, p50 2.42s)

Reproduce the sweep (scripted/offline):
```powershell
cd rlm-demo
$env:USE_SCRIPTED_SWEEP = "1"
$env:SWEEP_OUTPUT = "sweep_live.csv"
python -m examples.rlm_selector_sweep
```

Notes: The sweep corpus is synthetic and the embed selector uses a hash-vector stub. See `rlm-demo/docs/sweep_conclusions.md` for full detail and caveats.

## Real benchmark (SQuAD-mini + real embeddings)
- Config: `bench/configs/squad_mini.json` (sentence-transformers/all-MiniLM-L6-v2)
- Results (oracle extractive baseline): Accuracy 1.00, Recall@10 1.00, P50 latency ~43 ms
- Details: `rlm-demo/docs/real_benchmark_results.md`

## Benchmark harness + regression gates
- Run a benchmark: `python -m bench.run --config bench/configs/default.yaml`
- Run non-inferiority gates: `python -m bench.gates --run artifacts/{run_id}/summary.json`
- CI smoke gate: `.github/workflows/bench-gates.yml` runs `bench/configs/smoke.json` against a stored baseline.

## Start here (recruiter friendly)
- `rlm-demo` - Recursive Language Model demo + evaluation harness and sweep
- `docqa_vectorless_gpt52` - Vectorless document QA with table extraction, sparse retrieval, and citations
- `learning-to-discover-at-test-time_ttt-discover_ramsey-r55` - TTT-Discover style search for a Ramsey R(5,5) graph

## What this repo demonstrates
- LLM systems with controlled prompting, routing, and evaluation
- Retrieval and reasoning without a vector database when constraints require it
- Search and verification loops for hard combinatorial problems

## Repo layout
- Each top-level folder is a standalone project with its own README
- Some projects are paper-inspired, some are applied systems
