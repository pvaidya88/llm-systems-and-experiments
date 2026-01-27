import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from export import read_graph
from graph import Graph
from verifier import verify_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Ramsey R(5,5) graph")
    parser.add_argument("--input", type=str, default=None, help="Path to edge-list file")
    parser.add_argument("--n", type=int, default=43)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    if args.input:
        n, edges = read_graph(args.input)
        g = Graph.from_edge_list(n, edges)
    else:
        rng = np.random.default_rng(args.seed)
        g = Graph.random(args.n, rng, p=args.p)
        n = args.n

    v1 = verify_graph(g.adj, n)
    v2 = verify_graph(g.adj, n)
    valid = v1["valid"] and v2["valid"]

    if valid:
        print("VALID")
    else:
        print("INVALID")
        if v1.get("witness_g"):
            print(f"witness_g={v1['witness_g']}")
        if v1.get("witness_c"):
            print(f"witness_c={v1['witness_c']}")


if __name__ == "__main__":
    main()
