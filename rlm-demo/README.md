# RLM Demo (arXiv:2512.24601)

This folder contains a minimal, dependency-light implementation of Recursive Language Models (RLMs) inspired by the paper. The RLM treats the long context as an external resource accessed through a Python REPL rather than stuffing it into the LLM prompt.

## Quick start (dummy model)

This uses a scripted LLM response to exercise the REPL loop:

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

## RLM vs RAG toy demo

This example shows a case where naive retrieval misses relevant rows and produces the wrong aggregate,
while the RLM uses the REPL to compute the correct answer from the full context.

```powershell
python -m examples.rlm_vs_rag_demo
```

## Root vs sub-LM strength demo

This example runs two experiments: a weak root+sub pair and a strong pair using actual OpenAI models.
It uses a harder note with negations and synonyms, enforces REPL usage, and a strict output format,
and will retry invalid or incorrect outputs. You can give the strong model more attempts.
Defaults: weak = gpt-4.1-nano, strong = gpt-5.2. Override with env vars below.

```powershell
python -m examples.root_sub_strength_demo
```

Environment overrides:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:WEAK_ROOT_MODEL = "gpt-4.1-nano"
$env:WEAK_SUB_MODEL = "gpt-4.1-nano"
$env:STRONG_ROOT_MODEL = "gpt-5.2"
$env:STRONG_SUB_MODEL = "gpt-5.2"
$env:ROOT_MODEL_EFFORT = "xhigh"
$env:SUB_MODEL_EFFORT = "medium"
$env:OPENAI_TEXT_VERBOSITY = "low"
$env:MAX_ATTEMPTS = "2"
$env:MAX_ATTEMPTS_WEAK = "1"
$env:MAX_ATTEMPTS_STRONG = "3"
$env:MAX_STEPS = "10"
$env:SUBLM_LOAD_BEARING = "1"
$env:ROOTLM_LOAD_BEARING = "0"
$env:MIN_SUB_CALLS = "5"
$env:LLM_YESNO_MAX_RETRIES = "4"
$env:FIXED_TRIALS = "1"
$env:STRICT_REPL_TEMPLATE = "1"
$env:LOG_REPL_OUTPUTS = "1"
$env:NUM_TRIALS = "10"
$env:RANDOM_SEED = "42"
$env:VERBOSE_TRIALS = "1"
```

Note: `reasoning.effort` is only applied to reasoning-capable models (gpt-5 / o-series).
Some weaker models may violate the REPL protocol (e.g., missing variables); the demo will report the error
and continue so you can still see the strong-model result.
Note: `gpt-4.1-nano` only supports `OPENAI_TEXT_VERBOSITY=medium`; the demo will force `medium` for that model.
Note: set `ROOTLM_LOAD_BEARING=1` to force the root model to parse the note without any sub-LM calls.

To run the scripted (deterministic) version instead of live models:

```powershell
$env:USE_SCRIPTED_DEMO = "1"
python -m examples.root_sub_strength_demo
```

## Notes

- The system prompt in `rlm_demo/prompts.py` is a compact version of the Appendix D prompt described in the paper.
- The REPL executes arbitrary Python. Use trusted models and run in a sandbox if needed.
