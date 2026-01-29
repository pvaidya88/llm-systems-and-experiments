# Release decision (public analogue)

Date: Jan 29, 2026

This document shows how a team would decide to ship a retrieval update using the benchmark harness and gates.
It mirrors a production process without exposing proprietary data.

## Objective
Ship candidate retrieval changes only if quality does not regress beyond agreed thresholds.

## Inputs
- Benchmark summaries: `artifacts/{run_id}/summary.json`
- Gates: `bench.compare_runs` comparing baseline vs candidate
- Smoke dataset: `bench/data/smoke_*` (fast regression signal)
- Public benchmark: `docs/public_benchmark_results.md` (120-question SQuAD subset)

## Gate thresholds (smoke)
Source: `bench/configs/smoke_thresholds.json`
- Accuracy diff >= -0.02
- Recall@10 diff >= -0.03
- Latency ratio <= 50x (smoke is intentionally small; wide threshold)

## Release criteria
A candidate is ship-ready if:
- Smoke gates PASS on CI
- Public benchmark retrieval metrics (Recall@10, nDCG@10) do not regress beyond -0.02
- No new failure mode is introduced in manual spot checks (10 queries)

## Decision example
- Baseline: `bench/baselines/smoke_summary.json`
- Candidate: `artifacts/{run_id}/summary.json`
- Result: PASS => proceed to roll out

## Rollout plan (public analogue)
1. Canary 5% of queries (internal only)
2. Monitor latency and answer quality dashboards
3. Expand to 25% after 24h with no regressions
4. Full rollout if SLOs hold for 72h

## Rollback triggers
- Accuracy drop > 2% absolute vs baseline
- Unsupported-claim rate increases > 2% absolute
- P95 latency exceeds budget by > 25%
