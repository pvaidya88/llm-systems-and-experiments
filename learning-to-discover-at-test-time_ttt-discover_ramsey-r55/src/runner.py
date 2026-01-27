"""Search runner for TTT-Discover style Ramsey R(5,5) search."""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from buffer_puct import BufferPUCT
from entropic import entropic_weights
from export import append_log, ensure_dir, write_graph, write_json
from graph import Graph
from policy import EdgeFlipPolicy
from reward import compute_reward
from verifier import verify_graph


def run_search(
    n: int = 43,
    seed: int = 0,
    steps: int = 200,
    rollouts: int = 32,
    flips: int = 32,
    witness_cap: int = 30,
    gamma: float = math.log(2.0),
    shaped_scale: float = 0.05,
    buffer_capacity: int = 128,
    buffer_add_k: int = 4,
    c_puct: float = 1.5,
    policy_lr: float = 0.2,
    policy_l2: float = 1e-3,
    start_p: float = 0.5,
    output_dir: str | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    policy = EdgeFlipPolicy(n, rng)
    buffer = BufferPUCT(capacity=buffer_capacity, c_puct=c_puct)

    start_graph = Graph.random(n, rng, p=start_p)
    buffer.add_state(start_graph.to_key(), tuple(start_graph.adj), prior=1.0)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_root = os.path.join(project_root, "outputs")
    ensure_dir(outputs_root)

    if output_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(outputs_root, f"run_{stamp}_seed{seed}")
    ensure_dir(output_dir)
    latest_dir = os.path.join(outputs_root, "latest")
    ensure_dir(latest_dir)

    log_path = os.path.join(output_dir, "run.log")
    append_log(log_path, "step,max_reward,mean_reward,beta,best_reward,buffer_size")

    best_reward = -1e9
    best_graph = start_graph
    write_graph(os.path.join(output_dir, "best_graph.txt"), best_graph.adj, n)
    write_graph(os.path.join(latest_dir, "best_graph.txt"), best_graph.adj, n)

    for step in range(steps):
        rec = buffer.select()
        base_graph = Graph(n, rec["adj"])

        actions: List[List[int]] = []
        rewards: List[float] = []
        next_graphs: List[Graph] = []

        for _ in range(rollouts):
            g = base_graph.clone()
            action = policy.sample_action(flips)
            g.apply_flips(action, policy.edges)

            reward, info = compute_reward(
                g.adj,
                n,
                witness_cap=witness_cap,
                shaped_scale=shaped_scale,
            )
            actions.append(action)
            rewards.append(float(reward))
            next_graphs.append(g)

            if reward > best_reward:
                best_reward = float(reward)
                best_graph = g
                write_graph(os.path.join(output_dir, "best_graph.txt"), best_graph.adj, n)
                write_graph(os.path.join(latest_dir, "best_graph.txt"), best_graph.adj, n)

            if info.get("valid"):
                v1 = verify_graph(g.adj, n)
                v2 = verify_graph(g.adj, n)
                if v1["valid"] and v2["valid"]:
                    write_graph(os.path.join(output_dir, "valid_graph.txt"), g.adj, n)
                    write_json(
                        os.path.join(output_dir, "verification_log.json"),
                        {"first": v1, "second": v2},
                    )
                    append_log(log_path, f"{step},1.0,1.0,0.0,1.0,{len(buffer)}")
                    return {
                        "status": "valid",
                        "output_dir": output_dir,
                        "step": step,
                        "best_reward": best_reward,
                    }

        beta, weights = entropic_weights(rewards, gamma)
        policy.update(actions, weights, lr=policy_lr, l2=policy_l2)

        buffer.update(rec["key"], max(rewards))

        top_k = min(buffer_add_k, len(next_graphs))
        if top_k > 0:
            top_idx = np.argsort(weights)[-top_k:]
            for idx in top_idx:
                g = next_graphs[int(idx)]
                buffer.add_state(g.to_key(), tuple(g.adj), prior=float(weights[int(idx)]))

        mean_reward = float(np.mean(rewards))
        max_reward = float(np.max(rewards))
        append_log(
            log_path,
            f"{step},{max_reward:.6f},{mean_reward:.6f},{beta:.6f},{best_reward:.6f},{len(buffer)}",
        )

        checkpoint = {
            "step": step,
            "best_reward": best_reward,
            "buffer_size": len(buffer),
            "beta": beta,
            "config": {
                "n": n,
                "seed": seed,
                "steps": steps,
                "rollouts": rollouts,
                "flips": flips,
                "witness_cap": witness_cap,
                "gamma": gamma,
                "shaped_scale": shaped_scale,
                "buffer_capacity": buffer_capacity,
                "buffer_add_k": buffer_add_k,
                "c_puct": c_puct,
                "policy_lr": policy_lr,
                "policy_l2": policy_l2,
                "start_p": start_p,
            },
        }
        write_json(os.path.join(output_dir, "checkpoint.json"), checkpoint)

    return {
        "status": "not_found",
        "output_dir": output_dir,
        "step": steps,
        "best_reward": best_reward,
    }
