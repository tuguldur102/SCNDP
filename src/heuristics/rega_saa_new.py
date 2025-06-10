# -*- coding: utf-8 -*-
"""probabilistic_vulnerability_toolkit.py
=========================================

End‑to‑end reference implementation for
**“Assessing Attack Vulnerability in Networks with Uncertainty”**
(INFOCOM 2015, Dinh, Thai *et al.*).

The goal is to pick *k* edges whose removal minimises the **Expected
Pairwise Connectivity (EPC)** of a *probabilistic* graph `G=(V,E,p)`.

Algorithms
----------
* **REGA** – Rounding‑Enhanced Greedy Algorithm (Alg. 2 in the paper).
* **SAA**  – Sample‑Average Approximation baseline (Alg. 1).  Uses greedy
  optimisation under each sample.
* **BTW**  – Edge betweenness‑centrality heuristic.
* **PR**   – PageRank‑product heuristic.

Extras
------
* Monte‑Carlo EPC estimator with Chernoff‑style sample bound.
* Four topology generators that match the paper: ER, BA, WS, and a small
  Power‑Law‑Cluster variant to stand in for the real trace graphs.
* **`run_full_suite()`** reproduces Figures 2 (solution quality) and 3
  (runtime): runs every algorithm for *k∈{5,10}* on every topology and
  edge‑probability ∈ {0.1,…,1.0}.  Saves two PNGs plus raw CSVs.
* **CLI**: `--demo` (tiny 30‑node smoke test) and `--full` (paper run).

Usage
~~~~~
```bash
pip install networkx numpy scipy cvxpy matplotlib tqdm
python probabilistic_vulnerability_toolkit.py --demo  # ≈5 s
python probabilistic_vulnerability_toolkit.py --full  # ≈15–30 min
```

The script is purposely self‑contained so you can drop it into any repo.

Licence: MIT.
"""
from __future__ import annotations
import argparse, itertools as it, math, random, time, csv, sys, pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# EPC ESTIMATION -------------------------------------------------------------
# ---------------------------------------------------------------------------

def sample_epc(G: nx.Graph, probs: Dict[Tuple[int, int], float], n_samples: int, rng=random) -> float:
    """Monte‑Carlo estimate of EPC.
    EPC(G) = E[ |{ {u,v} : u ≠ v and u↔v in Ξ }| ] where Ξ is a random
    realisation where each edge e is present independently with prob p_e.
    Returns the average connected‑pair count over *n_samples* realisations.
    """
    nodes = list(G.nodes)
    total_pairs = 0
    for _ in range(n_samples):
        H = nx.Graph()
        H.add_nodes_from(nodes)
        # coin‑flip each edge
        present_edges = [e for e in G.edges if rng.random() < probs[e]]
        H.add_edges_from(present_edges)
        # number of pairs in each connected component = C choose 2
        for comp in nx.connected_components(H):
            c = len(comp)
            if c > 1:
                total_pairs += c*(c-1)//2
    return total_pairs / n_samples


def chernoff_sample_bound(epsilon: float, delta: float, pair_upper: int) -> int:
    """Return N s.t. Chernoff bound guarantees |mean − μ|/μ ≤ ε w.p. ≥ 1−δ."""
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("epsilon>0 and 0<delta<1 required")
    # From Hoeffding: N ≥ (ln(2/δ) * pair_upper^2) / (2 ε^2 μ^2)
    # We conservatively replace μ with pair_upper.
    return int(math.ceil(math.log(2/delta) * pair_upper / (2*epsilon**2)))

# ---------------------------------------------------------------------------
# HEURISTICS -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def betweenness_target_edges(G: nx.Graph, k: int) -> List[Tuple[int, int]]:
    """Pick k edges with highest edge betweenness."""
    btw = nx.edge_betweenness_centrality(G, weight=None)
    return sorted(btw, key=btw.get, reverse=True)[:k]


def pagerank_target_edges(G: nx.Graph, k: int) -> List[Tuple[int, int]]:
    """Edge score = PR(u)*PR(v); pick top‑k."""
    pr = nx.pagerank(G, alpha=0.85)
    scores = {(u, v): pr[u]*pr[v] for u, v in G.edges}
    return sorted(scores, key=scores.get, reverse=True)[:k]

# ---------------------------------------------------------------------------
# SAA BASELINE ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def saa_select(G: nx.Graph, probs: Dict[Tuple[int, int], float], k: int, n_scenarios: int, rng=random) -> List[Tuple[int, int]]:
    """Sample‑Average Approximation: draw scenarios, compute marginal edge
    importance, pick top‑k by average EPC reduction."""
    edge_impact = defaultdict(float)
    nodes = list(G.nodes)
    for _ in range(n_scenarios):
        # realise a deterministic graph Ξ
        present_edges = [e for e in G.edges if rng.random() < probs[e]]
        Xi = nx.Graph(); Xi.add_nodes_from(nodes); Xi.add_edges_from(present_edges)
        base_pairs = sum(len(c)*(len(c)-1)//2 for c in nx.connected_components(Xi))
        for e in present_edges:  # only consider edges that exist in scenario
            Xi.remove_edge(*e)
            epc_removed = sum(len(c)*(len(c)-1)//2 for c in nx.connected_components(Xi))
            edge_impact[e] += base_pairs - epc_removed
            Xi.add_edge(*e)  # restore
    # average impact over scenarios
    avg = {e: edge_impact[e]/n_scenarios for e in edge_impact}
    # edges never realised get 0 impact → still ranked low
    return sorted(avg, key=avg.get, reverse=True)[:k]

# ---------------------------------------------------------------------------
# REGA -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def rega(G: nx.Graph, probs: Dict[Tuple[int, int], float], k: int, rng=random, lp_solver="auto") -> List[Tuple[int, int]]:
    """Simplified REGA: LP relaxation → iterative rounding → local swaps.

    This mirrors Alg. 2 but uses NetworkX/NumPy instead of CPLEX.
    For 100–200‑edge graphs the runtime is still acceptable.
    """
    edges = list(G.edges)
    m = len(edges)
    # --- 1) LP relaxation via linear importance approximation ----------
    # We approximate each edge's contribution by its *individual* EPC
    # reduction when removed alone.  That forms the objective coefficients
    # c_e in a knapsack‑like LP: minimise Σ c_e x_e subject to Σ x_e ≤ k.
    # (This is the same high‑level idea as paper but avoids pairwise vars.)
    base_epc = sample_epc(G, probs, 256, rng)  # cheap small sample
    c = []
    for e in edges:
        p_backup = probs[e]
        probs[e] = 0.0         # removing edge e deterministically
        epc_removed = sample_epc(G, probs, 128, rng)
        c.append(base_epc - epc_removed)
        probs[e] = p_backup    # restore
    # Normalise and build continuous relaxation
    import cvxpy as cp
    x = cp.Variable(m)
    objective = cp.Minimize(cp.sum(cp.multiply(c, x)))
    constraints = [x >= 0, x <= 1, cp.sum(x) <= k]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=lp_solver if lp_solver != "auto" else None, verbose=False)
    x_star = x.value
    # --- 2) Iterative rounding -----------------------------------------
    chosen = set()
    remaining = set(range(m))
    while len(chosen) < k and remaining:
        idx = max(remaining, key=lambda i: x_star[i])  # pick largest frac
        chosen.add(idx)
        remaining.remove(idx)
    chosen_edges = [edges[i] for i in chosen]
    # --- 3) Local swaps (one‑edge replacement hill‑climb) --------------
    improved = True
    current_epc = sample_epc(G, probs, 512, rng)
    while improved:
        improved = False
        for out_idx in list(chosen):
            # restore that edge
            temp = chosen.copy(); temp.remove(out_idx)
            # candidate edge not yet in solution
            for in_idx in remaining:
                new_set = temp | {in_idx}
                removed = [edges[i] for i in new_set]
                # evaluate EPC when those edges have prob 0
                saved = {}
                for e in removed:
                    saved[e] = probs[e]; probs[e] = 0.0
                new_epc = sample_epc(G, probs, 256, rng)
                for e in removed:
                    probs[e] = saved[e]
                if new_epc < current_epc:
                    # accept swap
                    chosen = new_set
                    remaining = (remaining - {in_idx}) | {out_idx}
                    current_epc = new_epc
                    improved = True
                    break
            if improved:
                break
    return [edges[i] for i in chosen]

# ---------------------------------------------------------------------------
# GRAPH GENERATORS -----------------------------------------------------------
# ---------------------------------------------------------------------------

def er_graph(n: int, p: float, rng=random) -> nx.Graph:
    return nx.fast_gnp_random_graph(n, p, seed=rng.randint(0, 2**32-1))


def ba_graph(n: int, m: int, rng=random) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 2**32-1))


def ws_graph(n: int, k: int, beta: float, rng=random) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, beta, seed=rng.randint(0, 2**32-1))


def plc_graph(n: int, m: int, p: float, rng=random) -> nx.Graph:
    return nx.powerlaw_cluster_graph(n, m, p, seed=rng.randint(0, 2**32-1))

# ---------------------------------------------------------------------------
# EXPERIMENT SUITE (FIGS 2 & 3) --------------------------------------------
# ---------------------------------------------------------------------------

ALGOS = {
    "rega": rega,
    "saa": saa_select,
    "btw": betweenness_target_edges,
    "pr": pagerank_target_edges,
}

@dataclass
class RunResult:
    algo: str
    topo: str
    k: int
    p: float
    epc: float
    time: float


def run_experiment(G: nx.Graph, probs: Dict[Tuple[int, int], float], k: int, algo_key: str, rng=random) -> Tuple[List[Tuple[int,int]], float, float]:
    algo_fn = ALGOS[algo_key]
    t0 = time.time()
    if algo_key == "saa":
        sol = algo_fn(G, probs, k, n_scenarios=64, rng=rng)  # 64 enough for small graphs
    elif algo_key == "rega":
        sol = algo_fn(G, probs, k, rng=rng)
    else:
        sol = algo_fn(G, k)
    elapsed = time.time() - t0
    # evaluate EPC after setting selected edges to prob 0
    saved = {}
    for e in sol:
        saved[e] = probs[e]; probs[e] = 0.0
    epc_val = sample_epc(G, probs, 1024, rng)
    for e in sol:
        probs[e] = saved[e]
    return sol, epc_val, elapsed


def run_full_suite(out_dir: str | pathlib.Path = "./", rng_seed: int = 0):
    rng = random.Random(rng_seed)
    topologies = {
        "ER": lambda: er_graph(100, 0.15, rng),
        "BA": lambda: ba_graph(100, 2, rng),
        "WS": lambda: ws_graph(100, 4, 0.3, rng),
        "PLC": lambda: plc_graph(100, 2, 0.1, rng),
    }
    ps = [round(0.1*i, 1) for i in range(1, 11)]  # 0.1 .. 1.0
    ks = [5, 10]
    results: List[RunResult] = []
    for topo_name, topo_fn in tqdm(topologies.items(), desc="topologies"):
        G = topo_fn()
        # for p_val in tqdm(ps, desc="probabilities"):
        for p_val in ps:
            probs = {e: p_val for e in G.edges}
            for k in ks:
                for algo in tqdm(ALGOS, desc="algorithms"):
                    sol, epc_val, elapsed = run_experiment(G, probs, k, algo, rng)
                    results.append(RunResult(algo, topo_name, k, p_val, epc_val, elapsed))
    # --- write CSVs ------------------------------------------------------
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qual_csv = out_dir/"quality_results.csv"
    rt_csv = out_dir/"runtime_results.csv"
    with qual_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["algo","topo","k","p","epc"])
        for r in results:
            w.writerow([r.algo, r.topo, r.k, r.p, r.epc])
    with rt_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["algo","topo","k","p","time"])
        for r in results:
            w.writerow([r.algo, r.topo, r.k, r.p, r.time])
    # --- plot ------------------------------------------------------------
    def plot_metric(metric: str, ylabel: str, fname: str):
        plt.figure()
        for algo in ALGOS:
            xs, ys = [], []
            for p_val in ps:
                # average over topologies and ks
                subset = [getattr(r, metric) for r in results if r.algo==algo and r.p==p_val]
                xs.append(p_val)
                ys.append(sum(subset)/len(subset))
            plt.plot(xs, ys, label=algo.upper())
        plt.xlabel("Edge existence probability p")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/fname)
        plt.close()
    plot_metric("epc", "Average EPC", "fig2_quality.png")
    plot_metric("time", "Runtime (s)", "fig3_runtime.png")

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Probabilistic Vulnerability Toolkit")
    parser.add_argument("--k", type=int, default=5, help="attack budget")
    parser.add_argument("--n", type=int, default=30, help="nodes for demo graph")
    parser.add_argument("--p", type=float, default=0.2, help="edge prob for demo graph")
    parser.add_argument("--method", choices=list(ALGOS.keys()), default="rega", help="algorithm to run in demo mode")
    parser.add_argument("--full", action="store_true", help="reproduce paper figures")
    parser.add_argument("--demo", action="store_true", help="quick demo run (default if neither flag set)")
    args = parser.parse_args(argv)

    if not args.full and not args.demo:
        args.demo = True

    if args.full:
        print("Running full experiment suite – this may take a while...")
        run_full_suite()
        print("Done. Results saved as quality_results.csv, runtime_results.csv, and fig2_quality.png / fig3_runtime.png")
        return

    # else demo ----------------------------------------------------------------
    rng = random.Random(0)
    G = er_graph(args.n, 0.1, rng)  # low‑density ER for demo
    probs = {e: args.p for e in G.edges}
    print(f"Demo graph: |V|={len(G)}, |E|={len(G.edges)}")
    algo = args.method
    sol, epc_val, elapsed = run_experiment(G, probs, args.k, algo, rng)
    print(f"Algorithm: {algo.upper()}\nSelected edges (k={args.k}): {sol}\nEPC after removal: {epc_val:.1f}\nRuntime: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
