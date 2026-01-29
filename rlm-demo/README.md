# RLM Demo

Minimal, dependency-light implementation of Recursive Language Models (RLMs) inspired by the [paper](https://arxiv.org/abs/2512.24601).
The RLM treats long context as an external resource accessed through a Python REPL instead of stuffing
it into the prompt.

## Why it matters
- Demonstrates a clean separation between reasoning and tool use on long context
- Shows how root and sub model strength affects outcomes under different load bearing regimes

## What to look at
- `rlm_demo/repl.py` for the REPL interface and execution loop
- `examples/root_sub_strength_demo.py` for root vs sub model experiments

## Quick start (dummy model)

This uses scripted LLM responses to exercise the REPL loop:

```bash
python -m examples.dummy_run
```

## OpenAI-compatible endpoint

Point the client at any OpenAI-compatible `/v1/chat/completions` server (local or hosted).

```powershell
$env:LLM_BASE_URL = "http://localhost:8000"
$env:LLM_MODEL = "your-model-name"
$env:LLM_API_KEY = "optional"
python -m examples.openai_compatible_run
```

## OpenAI Responses API (official SDK)

Use the OpenAI Python SDK with the Responses API, with a higher-effort root model and a lower-effort sub-model.

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-5.2"
$env:OPENAI_REASONING_EFFORT = "xhigh"
$env:OPENAI_SUB_REASONING_EFFORT = "low"
$env:OPENAI_TEXT_VERBOSITY = "medium"
python -m examples.openai_responses_run
```

Note: Keep your API key out of source control; prefer environment variables.

## Expected outputs
- Pass rate summaries per root and sub model configuration
- Counts of invalid outputs and max step errors to diagnose brittleness

## Sample results

Run settings:
```powershell
$env:WEAK_ROOT_MODEL = "gpt-4.1-nano"
$env:STRONG_ROOT_MODEL = "gpt-5.2"
$env:WEAK_SUB_MODEL = "gpt-4.1-nano"
$env:STRONG_SUB_MODEL = "gpt-5.2"
$env:NUM_TRIALS = "10"
$env:RANDOM_SEED = "42"
$env:FIXED_TRIALS = "1"
$env:STRICT_REPL_TEMPLATE = "1"
$env:ROOT_STRICT_REPL_TEMPLATE = "1"
$env:HIDE_NOTE_FROM_ROOT = "1"
$env:FULL_FACTORIAL = "1"
$env:ORACLE_ABLATIONS = "1"
python -m examples.root_sub_strength_demo
```

Sub load bearing full factorial:
- root gpt-4.1-nano / sub gpt-4.1-nano pass rate 5/10 (50%)
- root gpt-4.1-nano / sub gpt-5.2 pass rate 9/10 (90%)
- root gpt-5.2 / sub gpt-4.1-nano pass rate 5/10 (50%)
- root gpt-5.2 / sub gpt-5.2 pass rate 9/10 (90%)

Oracle ablations in sub load bearing:
- oracle_sub root gpt-4.1-nano pass rate 10/10 (100%)
- oracle_sub root gpt-5.2 pass rate 10/10 (100%)
- oracle_root sub gpt-4.1-nano pass rate 0/10 (0%)
- oracle_root sub gpt-5.2 pass rate 10/10 (100%)

Root load bearing full factorial:
- root gpt-4.1-nano / sub gpt-4.1-nano pass rate 10/10 (100%)
- root gpt-4.1-nano / sub gpt-5.2 pass rate 10/10 (100%)
- root gpt-5.2 / sub gpt-4.1-nano pass rate 10/10 (100%)
- root gpt-5.2 / sub gpt-5.2 pass rate 10/10 (100%)

Oracle ablations in root load bearing:
- oracle_sub root gpt-4.1-nano pass rate 10/10 (100%)
- oracle_sub root gpt-5.2 pass rate 10/10 (100%)
- oracle_root sub gpt-4.1-nano pass rate 2/10 (20%)
- oracle_root sub gpt-5.2 pass rate 10/10 (100%)

## Demos

### RLM vs RAG toy demo

This example shows a case where naive retrieval misses relevant rows and produces the wrong aggregate,
while the RLM uses the REPL to compute the correct answer from the full context.

```powershell
python -m examples.rlm_vs_rag_demo
```

### Root vs sub-LM strength demo

Runs controlled experiments with a root model and a sub model. The demo can enforce REPL usage,
hide the note from the root, and simulate sub-load-bearing vs root-load-bearing regimes.

```powershell
python -m examples.root_sub_strength_demo
```

### Book QA without vector DB

Uses `ctx.bm25_search` + `ctx.expand` in the REPL to answer a book question without embeddings.

```powershell
python -m examples.book_qa_vectorless_demo
```

### Selector sweep harness

Sweeps selectors (grep/BM25/embeddings), recursion policy, and depth with trace + budget metrics.
Writes CSV if `SWEEP_OUTPUT` is set.

```powershell
$env:OPENAI_API_KEY = "your-api-key" # optional for live runs
$env:USE_SCRIPTED_SWEEP = "1" # optional for offline scripted run
$env:SWEEP_OUTPUT = "sweep.csv" # optional
python -m examples.rlm_selector_sweep
```

## Benchmark harness (bench/)

This repo now includes a full evaluation harness for comparing:
- Vector RAG (dense retrieval + LLM rerank + generate)
- Hybrid RAG (BM25 + dense + rerank + generate)
- RLM-vectorless (tool-orchestrated retrieval with **no embeddings**)

### Dataset format
See `docs/benchmark_dataset_format.md` for `corpus.jsonl` + `queries.jsonl`.

### Run a benchmark
```powershell
python -m bench.run --config bench/configs/default.yaml
```

Outputs:
- `artifacts/{run_id}/per_query.jsonl`
- `artifacts/{run_id}/summary.json`
- `artifacts/{run_id}/summary.csv`

### Scaling curves
```powershell
python -m bench.scale_sweep --config bench/configs/default.yaml
```

### Non-inferiority gates
```powershell
python -m bench.gates --run artifacts/{run_id}/summary.json
```

### CI smoke gate
GitHub Actions runs a tiny offline benchmark and enforces non-inferiority gates:
- config: `bench/configs/smoke.json`
- thresholds: `bench/configs/smoke_thresholds.json`
- baseline: `bench/baselines/smoke_summary.json`

### Real benchmark (SQuAD-mini)
Runs a small real dataset with sentence-transformer embeddings and an oracle extractive baseline:
```powershell
python -m bench.run --config bench/configs/squad_mini.json
```
Results: `docs/real_benchmark_results.md`
Requires: `pip install datasets sentence-transformers`

### Public benchmark (SQuAD-120)
Larger public dataset with real embeddings:
```powershell
python -m bench.run --config bench/configs/squad_public_oracle.json
python -m bench.run --config bench/configs/squad_public.json
```
Results: `docs/public_benchmark_results.md`

### Release decision + system overview
- `docs/release_decision.md`
- `docs/system_overview.md`

### Build a dataset from a directory
```powershell
python -m bench.tools.build_dataset_from_dir --input ./docs --output corpus.jsonl
```

### Sweep conclusions (Jan 28, 2026)

Summary, defaults, and caveats for the most recent partial sweep:
- `docs/sweep_conclusions.md`
- Pareto plots and aggregates in `plots/` (generated via `python -m examples.plot_sweep`)

## Root vs sub-LM demo configuration

Model selection:
- `WEAK_ROOT_MODEL` (default: `gpt-4.1-nano`)
- `WEAK_SUB_MODEL` (default: `gpt-4.1-nano`)
- `STRONG_ROOT_MODEL` (default: `gpt-5.2`)
- `STRONG_SUB_MODEL` (default: `gpt-5.2`)

Reasoning and verbosity:
- `ROOT_MODEL_EFFORT` (default: empty)
- `SUB_MODEL_EFFORT` (default: empty)
- `OPENAI_TEXT_VERBOSITY` (default: `low`)

Attempts and steps:
- `MAX_ATTEMPTS` (default: `2`)
- `MAX_ATTEMPTS_WEAK` (default: `MAX_ATTEMPTS`)
- `MAX_ATTEMPTS_STRONG` (default: `MAX_ATTEMPTS`)
- `MAX_STEPS` (default: `10`)

Load-bearing controls:
- `SUBLM_LOAD_BEARING` (default: `1`)
- `ROOTLM_LOAD_BEARING` (default: `0`)
- `ROOT_STRICT_REPL_TEMPLATE` (default: `1`)
- `STRICT_REPL_TEMPLATE` (default: `1`)
- `HIDE_NOTE_FROM_ROOT` (default: `1`)
- `HARD_NOTES` (default: `0`)

Sub-call mitigation:
- `SUB_MITIGATE` (default: `0`)
- `SUB_VOTE_K` (default: `3` when `SUB_MITIGATE=1`, otherwise `1`)
- `LLM_YESNO_MAX_RETRIES` (default: `4`)
- `MAX_SUB_CALLS` (default: `32`)

Experiment controls:
- `NUM_TRIALS` (default: `10`)
- `RANDOM_SEED` (default: `42`)
- `FIXED_TRIALS` (default: `1`)
- `FULL_FACTORIAL` (default: `0`) - runs 2x2 root/sub grid for both regimes
- `ORACLE_ABLATIONS` (default: `0`)
- `VERBOSE_TRIALS` (default: `0`)
- `LOG_REPL_OUTPUTS` (default: `0`)

Scripted demo:
- `USE_SCRIPTED_DEMO` (default: `0`)

## Notes and tips

- `gpt-4.1-nano` only supports `OPENAI_TEXT_VERBOSITY=medium`; the demo forces `medium` for that model.
- Set `ROOTLM_LOAD_BEARING=1` to forbid sub calls and make the root parse the note itself.
- Set `HIDE_NOTE_FROM_ROOT=1` to redact the note from context and force `note_yesno` in sub-load-bearing mode.
- Set `HARD_NOTES=1` to break ceilings in root-load-bearing runs.
- Set `SUB_MITIGATE=1` to majority-vote sub classification calls (increase `MAX_SUB_CALLS` if needed).
- Hidden note isolation runs a separate subprocess and only supports `OpenAICompatibleClient` and
  `OpenAIResponsesClient` sub-models. For scripted/custom sub-LLMs, set `RLM_ALLOW_INSECURE_NOTE=1`
  to fall back to in-process `note_yesno` (results may be compromised).
- The REPL executes in a subprocess with timeouts and memory/CPU limits; tune via `RLMOptions`
  (`repl_timeout_s`, `repl_memory_mb`, `repl_cpu_seconds`) if needed.
- Set `RLMOptions.protocol="json"` to use a strict JSON output protocol for REPL/FINAL responses.
  This removes regex parsing and makes outputs machine-parseable.

To run the scripted (deterministic) version instead of live models:

```powershell
$env:USE_SCRIPTED_DEMO = "1"
python -m examples.root_sub_strength_demo
```

## Safety

The REPL executes arbitrary Python. Use trusted models and run in a sandbox if needed.
