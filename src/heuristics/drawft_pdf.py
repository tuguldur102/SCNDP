"""
stochastic_cndp.py
------------------
Heuristic solver for the Stochastic Critical-Node Detection Problem (CNDP)
with *edge* uncertainty, as defined in Stochastic_CNDP.pdf.

Implements:
  • Algorithm 1  – Monte-Carlo estimator of σ(S)
  • Algorithm 2  – greedy attacker using that estimator
Supports:
  • Erdős–Rényi G(n,p) graphs
  • Watts–Strogatz Watts–Strogatz(n, k, β) graphs
"""

from __future__ import annotations
import random, itertools, math, time
import networkx as nx
import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------
# 1.  GRAPH BUILDING UTILITIES
# ------------------------------------------------------------
def er_prob_graph(n: int, p_edge: float,
                  p_low: float=0.1, p_high: float=0.9,
                  seed: int|None=None) -> nx.Graph:
    """G(n,p_edge) with iid edge-existence probs U(p_low,p_high)."""
    rng = random.Random(seed)
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    for u, v in G.edges():
        G[u][v]["prob"] = rng.uniform(p_low, p_high)
    return G


def ws_prob_graph(n: int, k: int, beta: float,
                  p_low: float=0.1, p_high: float=0.9,
                  seed: int|None=None) -> nx.Graph:
    """Watts–Strogatz small-world graph with iid edge probs."""
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, k, beta, seed=seed)
    for u, v in G.edges():
        G[u][v]["prob"] = rng.uniform(p_low, p_high)
    return G


# ------------------------------------------------------------
# 2.  PAIRWISE CONNECTIVITY ON ONE *SCENARIO*
# ------------------------------------------------------------
def pairwise_cost_det(G_det: nx.Graph) -> int:
    """Σ_{components} |C|·(|C|-1)/2 – deterministic definition
    used inside the Monte-Carlo loop."""
    cost = 0
    for comp in nx.connected_components(G_det):
        s = len(comp)
        cost += s * (s - 1) // 2
    return cost


# ------------------------------------------------------------
# 3.  ALGORITHM 1 –  MONTE-CARLO ESTIMATE  σ(S)
# ------------------------------------------------------------
def expected_pairwise_connectivity(
    G: nx.Graph,
    S: set[int] | set[str],
    num_samples: int = 10_000,
    rng: random.Random | None = None,
    show_bar: bool = True,
) -> float:
    """
    Monte-Carlo estimator of σ(S) from Algorithm 1​ :contentReference[oaicite:0]{index=0}
    """
    if rng is None:
        rng = random.Random()

    remaining_nodes = set(G.nodes()) - S
    if not remaining_nodes:
        return 0.0

    total_cost = 0
    iterator = range(num_samples)
    if show_bar:
        iterator = tqdm(iterator, desc="MC-samples", leave=False)

    for _ in iterator:
        # sample a *live-edge* scenario
        H = nx.Graph()
        H.add_nodes_from(remaining_nodes)
        
        for u, v, data in G.edges(data=True):
            if u in S or v in S:
                continue
            if rng.random() < data["prob"]:
                H.add_edge(u, v)

        total_cost += pairwise_cost_det(H)

    return total_cost / num_samples


# ------------------------------------------------------------
# 4.  ALGORITHM 2 –  GREEDY ATTACKER
# ------------------------------------------------------------
def greedy_cndp(
    G: nx.Graph,
    K: int,
    num_samples: int = 2_000,
    seed: int | None = None,
) -> tuple[set[int], list[float]]:
    """Return (selected_set, σ values after each pick)."""
    rng = random.Random(seed)
    S: set[int] = set()
    sigmas: list[float] = []

    current_sigma = expected_pairwise_connectivity(G, S, num_samples, rng)
    sigmas.append(current_sigma)

    for _ in range(K):
        best_node, best_sigma = None, float("inf")

        for v in (set(G.nodes()) - S):
            sigma_v = expected_pairwise_connectivity(G, S | {v},
                                                     num_samples, rng, False)
            if sigma_v < best_sigma:
                best_sigma, best_node = sigma_v, v

        S.add(best_node)                # exploit 1-step look-ahead
        current_sigma = best_sigma
        sigmas.append(current_sigma)
        print(f"● Picked {best_node:>3};   σ = {current_sigma:.1f}")

    return S, sigmas


# ------------------------------------------------------------
# 5.  QUICK DRIVER FOR EXPERIMENTS
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, pathlib

    parser = argparse.ArgumentParser(
        description="Greedy heuristic for stochastic CNDP")
    parser.add_argument("--model", choices=["er", "ws"], default="er")
    parser.add_argument("-n", type=int, default=100,
                        help="number of nodes")
    parser.add_argument("--p", type=float, default=0.05,
                        help="edge probability for ER (or rewiring β for WS)")
    parser.add_argument("--k", type=int, default=4,
                        help="nearest-neighbour degree in WS")
    parser.add_argument("--budget", "-K", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model == "er":
        G = er_prob_graph(args.n, args.p, seed=args.seed)
    else:
        G = ws_prob_graph(args.n, args.k, args.p, seed=args.seed)

    start = time.perf_counter()
    S_star, sigmas = greedy_cndp(G, args.budget,
                                 num_samples=args.samples,
                                 seed=args.seed)
    elapsed = time.perf_counter() - start
    print("\n=====  RESULT  =====")
    print(f"Removed nodes: {sorted(S_star)}")
    print("σ after each pick:", [round(x, 1) for x in sigmas])
    print(f"Elapsed: {elapsed:.1f} s")
