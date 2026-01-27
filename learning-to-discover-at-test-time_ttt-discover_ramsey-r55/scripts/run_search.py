import argparse
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from runner import run_search


def main() -> None:
    parser = argparse.ArgumentParser(description="TTT-Discover Ramsey R(5,5) search")
    parser.add_argument("--n", type=int, default=43)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--flips", type=int, default=32)
    parser.add_argument("--witness-cap", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=math.log(2.0))
    parser.add_argument("--shaped-scale", type=float, default=0.05)
    parser.add_argument("--buffer-capacity", type=int, default=128)
    parser.add_argument("--buffer-add-k", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--policy-lr", type=float, default=0.2)
    parser.add_argument("--policy-l2", type=float, default=1e-3)
    parser.add_argument("--start-p", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_search(
        n=args.n,
        seed=args.seed,
        steps=args.steps,
        rollouts=args.rollouts,
        flips=args.flips,
        witness_cap=args.witness_cap,
        gamma=args.gamma,
        shaped_scale=args.shaped_scale,
        buffer_capacity=args.buffer_capacity,
        buffer_add_k=args.buffer_add_k,
        c_puct=args.c_puct,
        policy_lr=args.policy_lr,
        policy_l2=args.policy_l2,
        start_p=args.start_p,
        output_dir=args.output_dir,
    )

    status = result.get("status")
    out_dir = result.get("output_dir")
    print(f"status={status} output_dir={out_dir}")


if __name__ == "__main__":
    main()
