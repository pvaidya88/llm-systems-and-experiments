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

To run the scripted (deterministic) version instead of live models:

```powershell
$env:USE_SCRIPTED_DEMO = "1"
python -m examples.root_sub_strength_demo
```

## Safety

The REPL executes arbitrary Python. Use trusted models and run in a sandbox if needed.
