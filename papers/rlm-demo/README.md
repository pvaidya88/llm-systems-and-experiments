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

This example runs two experiments: a weak root+sub pair that fails and a strong pair that succeeds.

```powershell
python -m examples.root_sub_strength_demo
```

## Notes

- The system prompt in `rlm_demo/prompts.py` is a compact version of the Appendix D prompt described in the paper.
- The REPL executes arbitrary Python. Use trusted models and run in a sandbox if needed.
