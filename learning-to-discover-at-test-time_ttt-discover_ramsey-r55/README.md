# Learning to Discover at Test Time (TTT-Discover) - Ramsey R(5,5)

This project implements a TTT-Discover-style search to find a (5,5)-Ramsey graph on n=43 vertices:
- No 5-clique in G
- No 5-clique in complement(G)

The implementation is CPU-first and uses a lightweight edge-flip policy (learned logits over edges), a PUCT-style state buffer with max-reward Q(s), and an adaptive entropic objective where KL(q_beta || uniform) = ln 2.

## Quickstart

Install deps:

```
pip install -r requirements.txt
```

Run a search (defaults are CPU friendly):

```
python scripts/run_search.py --seed 0 --steps 200 --rollouts 32 --flips 32
```

Verify a graph (edge-list format) or a random graph:

```
python scripts/verify_graph.py --input outputs/latest/best_graph.txt
python scripts/verify_graph.py --seed 123
```

## Outputs

Each run creates an `outputs/run_...` folder with:
- `best_graph.txt` (best-so-far graph, even if invalid)
- `checkpoint.json` (search state summary)
- `run.log` (step-by-step metrics)
- `verification_log.json` (only when a valid graph is confirmed twice)

## Notes

- Deterministic given a seed (all randomness from a single RNG).
- A candidate is only accepted if the exact verifier passes twice independently.
- The verifier provides K5 witnesses in G or complement(G).
