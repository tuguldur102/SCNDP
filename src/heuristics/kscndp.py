# kpcnd.py – v2 (2025‑06‑10)
"""Reproducibility toolkit for the probabilistic critical‑node detection
benchmarks from Thai et al., *INFOCOM 2015*.

Changelog (v2)
==============
* **Probability distributions**
  * *fixed* – every edge has the same existence probability *p* (default).
  * *uniform* – each edge draws *pₑ* independently, uniformly at random
    from the discrete set {0.1, 0.2, …, 1.0} (override via `--prob-set`).
* **Progress bars** – all expensive Monte‑Carlo/FPRAS loops wrapped in tqdm.
* **Matplotlib output** – `--full` generates Figures 2 & 3 equivalents
  (EPC ↓ and runtime ↑ versus edge probability) and saves them as
  `figure2_epc.png`, `figure3_time.png`.
* **CLI shortcuts** – `--demo` (fast sanity check) and `--full` (replicate
  paper curves) added; see `python kpcnd.py --help`.

The script remains dependency‑light: `networkx numpy pulp tqdm matplotlib`.
Install with::

    pip install networkx pulp numpy tqdm matplotlib

GUROBI support unchanged (export `KPCND_GUROBI=1`).
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt  # for figures 2 & 3
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def ordered(u: int, v: int) -> Tuple[int, int]:
    return edge_key(u, v)


def combinations_index(nodes: Iterable[int]):
    nodes = list(nodes)
    for i, j in itertools.combinations(nodes, 2):
        yield (i, j)


@dataclass
class ProbabilisticGraph:
    g: nx.Graph
    p: Dict[Tuple[int, int], float]  # edge‑existence probabilities

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def sample_realisation(self, rng: np.random.Generator | None = None) -> nx.Graph:
        if rng is None:
            rng = np.random.default_rng()
        h = nx.Graph()
        h.add_nodes_from(self.g.nodes)
        for (u, v) in self.g.edges:
            if rng.random() < self.p[edge_key(u, v)]:
                h.add_edge(u, v)
        return h

    # ---------------------------------------------------------------------
    # Monte‑Carlo EPC (slow, debugging only)
    # ---------------------------------------------------------------------
    def monte_carlo_epc(self, samples: int = 1_000, rng: np.random.Generator | None = None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        n = self.g.number_of_nodes()
        pairs = n * (n - 1) // 2
        acc = 0
        for _ in tqdm(range(samples), desc="MC", leave=False):
            acc += connected_pair_count(self.sample_realisation(rng))
        return acc / samples / pairs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def connected_pair_count(g: nx.Graph) -> int:
    total = 0
    for cc in nx.connected_components(g):
        m = len(cc)
        total += m * (m - 1) // 2
    return total


# ---------------------------------------------------------------------------
# FPRAS – Component Sampling Procedure (Algorithm 2)
# ---------------------------------------------------------------------------

def csp_epc_estimate(pg: ProbabilisticGraph, eps: float = 0.05, delta: float = 1e-2,
                     rng: np.random.Generator | None = None, show_tqdm: bool = False) -> float:
    if rng is None:
        rng = np.random.default_rng()
    g = pg.g
    n = g.number_of_nodes()
    p_e = sum(pg.p.values())
    if p_e < eps ** 2 / (n ** 2):
        return p_e  # cheap bound
    # Upper bound µ̂ for stopping‑time calculation
    m = g.number_of_edges()
    mu_hat = min((1 + p_e / m) ** m, n * (n - 1) / 2)
    N = int(np.ceil(4 * np.e ** 2 * np.log(2 / delta) / (eps ** 2 * mu_hat)))

    acc = 0
    it = tqdm(range(N), desc="CSP", leave=False) if show_tqdm else range(N)
    for _ in it:
        u = rng.choice(tuple(g.nodes))
        queue, visited = [u], {u}
        while queue:
            v = queue.pop()
            for w in g.adj[v]:
                if w in visited:
                    continue
                if rng.random() < pg.p[edge_key(v, w)]:
                    visited.add(w)
                    queue.append(w)
        acc += len(visited) - 1
    return n * acc / (2 * N)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def betweenness_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    btw = nx.betweenness_centrality(pg.g)
    return sorted(btw, key=btw.get, reverse=True)[:k]


def pagerank_edge_removal(pg: ProbabilisticGraph, k: int) -> List[Tuple[int, int]]:
    pr = nx.pagerank(pg.g, alpha=0.85)
    edges = sorted(pg.g.edges, key=lambda e: pr[e[0]] + pr[e[1]], reverse=True)
    return edges[:k]


# ---------------------------------------------------------------------------
# REGA (Algorithm 1)
# ---------------------------------------------------------------------------

class REGA:
    def __init__(self, pg: ProbabilisticGraph, k: int):
        self.pg, self.k = pg, k
        self.n = pg.g.number_of_nodes()
        self.lp_solver = "GUROBI" if os.getenv("KPCND_GUROBI") else "PULP_CBC_CMD"
        self._rng = np.random.default_rng()

    # ---------------------- LP relaxation ------------------------------
    def _solve_lp_relaxation(self) -> np.ndarray:
        g = self.pg.g
        prob = pulp.LpProblem("MIPR_LP", pulp.LpMinimize)
        s = pulp.LpVariable.dicts("s", g.nodes, 0, 1)
        x = pulp.LpVariable.dicts("x", combinations_index(g.nodes), 0, 1)
        prob += pulp.lpSum(1 - x[i, j] for (i, j) in x)  # objective
        prob += pulp.lpSum(s[i] for i in g.nodes) <= self.k  # cardinality
        pij = self.pg.p
        for (i, j) in g.edges:
            prob += x[ordered(i, j)] <= s[i] + s[j] + 1 - pij[edge_key(i, j)]
        # Constraint (18): triangle inequality on x_ij variables.
        # Skip k that coincides with i or j because (i,i) is not a valid pair
        # in the combinations_index dictionary (would raise KeyError).
        for (i, j) in g.edges:
            for k in g.nodes:
              if k in (i, j):
                continue  # avoid invalid (i,i) or (j,j) look‑ups
              prob += x[ordered(i, j)] + x[ordered(j, k)] >= x[ordered(i, k)]
        solver = pulp.getSolver(self.lp_solver, timeLimit=300, msg=False)
        prob.solve(solver)
        return np.array([s[i].value() for i in range(self.n)])

    # ---------------------- main entry -------------------------------
    def run(self, eps: float = 0.05, delta: float = 1e-2) -> List[int]:
        g = self.pg.g
        selected, fixed = [], set()
        for _ in range(self.k):
            frac = self._solve_lp_relaxation()
            cand = max((i for i in range(self.n) if i not in fixed), key=lambda i: frac[i])
            selected.append(cand)
            fixed.add(cand)
        improved = True
        while improved:
            improved = False
            epc_sel = self._epc_after_node_removal(selected, eps, delta)
            for u in selected:
                for v in (set(g.nodes) - set(selected)):
                    trial = [x for x in selected if x != u] + [v]
                    epc_trial = self._epc_after_node_removal(trial, eps, delta)
                    if epc_trial < epc_sel:
                        selected, epc_sel = trial, epc_trial
                        improved = True
                        break
                if improved:
                    break
        return selected

    # ---------------------- helper ----------------------------------
    def _epc_after_node_removal(self, rem: Sequence[int], eps, delta) -> float:
        g2 = self.pg.g.copy()
        g2.remove_nodes_from(rem)
        p2 = {edge_key(u, v): self.pg.p[edge_key(u, v)] for (u, v) in g2.edges}
        return csp_epc_estimate(ProbabilisticGraph(g2, p2), eps, delta, self._rng, False)


# ---------------------------------------------------------------------------
# SAA baseline
# ---------------------------------------------------------------------------

class SAA:
    def __init__(self, pg: ProbabilisticGraph, k: int, T: int = 30):
        self.pg, self.k, self.T = pg, k, T

    def run(self) -> List[int]:
        rng = np.random.default_rng()
        samples = [self.pg.sample_realisation(rng) for _ in range(self.T)]
        freq = defaultdict(int)
        for h in samples:
            for e in h.edges:
                freq[edge_key(*e)] += 1
        avg_g = nx.Graph()
        avg_g.add_nodes_from(self.pg.g.nodes)
        for e, cnt in freq.items():
            if cnt > 0:
                avg_g.add_edge(*e)
        avg_pg = ProbabilisticGraph(avg_g, {e: cnt / self.T for e, cnt in freq.items()})
        return betweenness_node_removal(avg_pg, self.k)


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def make_random_graph(model: str, n: int, m: int, p: float, edge_distr: str,
                      prob_set: Sequence[float]) -> ProbabilisticGraph:
    rng = random.Random(42)
    if model == "er":
        g = nx.gnm_random_graph(n, m, seed=42)
    elif model == "ba":
        g = nx.barabasi_albert_graph(n, max(1, m // n), seed=42)
    elif model == "ws":
        k_neigh = max(2, int(round(2 * m / n)))
        g = nx.watts_strogatz_graph(n, k_neigh, 0.3, seed=42)
    else:
        raise ValueError(model)

    if edge_distr == "fixed":
        prob = {edge_key(u, v): p for (u, v) in g.edges}
    elif edge_distr == "uniform":
        prob = {edge_key(u, v): rng.choice(prob_set) for (u, v) in g.edges}
    else:
        raise ValueError(edge_distr)
    return ProbabilisticGraph(g, prob)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_all(pg: ProbabilisticGraph, k: int, show_tqdm: bool = False):
    out = {}
    t0 = time.perf_counter()
    res_rega = REGA(pg, k).run()
    out["REGA"] = (res_rega, time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_saa = SAA(pg, k).run()
    out["SAA"] = (res_saa, time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_btw = betweenness_node_removal(pg, k)
    out["Betweenness"] = (res_btw, time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_pr = pagerank_edge_removal(pg, k)
    out["PageRank"] = (res_pr, time.perf_counter() - t0)
    return out


def full_benchmark(model: str, n: int, m: int, k: int, edge_distr: str,
                   prob_set: Sequence[float], eps: float):
    ps = np.arange(0.1, 1.01, 0.1)
    methods = ["REGA", "SAA", "Betweenness", "PageRank"]
    epc_curves = {m: [] for m in methods}
    time_curves = {m: [] for m in methods}

    rng = np.random.default_rng()
    for p in tqdm(ps, desc="p sweep"):
        pg = make_random_graph(model, n, m, p, edge_distr, prob_set)
        out = run_all(pg, k)
        # Estimate EPC for each solution (FPRAS, same eps for fairness)
        for meth in tqdm(methods, desc="methods"):
            sol, t = out[meth]
            if meth == "PageRank":
                # remove edges
                g2 = pg.g.copy()
                g2.remove_edges_from(sol)
                p2 = {edge_key(u, v): pg.p[edge_key(u, v)] for (u, v) in g2.edges}
            else:
                g2 = pg.g.copy()
                g2.remove_nodes_from(sol)
                p2 = {edge_key(u, v): pg.p[edge_key(u, v)] for (u, v) in g2.edges}
            epc = csp_epc_estimate(ProbabilisticGraph(g2, p2), eps, 0.01, rng, False)
            epc_curves[meth].append(epc)
            time_curves[meth].append(t)

    # Figure 2 – EPC vs p
    plt.figure()
    for meth in methods:
        plt.plot(ps, epc_curves[meth], label=meth, marker="o")
    plt.xlabel("Edge existence probability p")
    plt.ylabel("EPC (lower is better)")
    plt.title("Figure 2 – EPC vs p")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure2_epc.png")

    # Figure 3 – Time vs p
    plt.figure()
    for meth in methods:
        plt.plot(ps, time_curves[meth], label=meth, marker="o")
    plt.xlabel("Edge existence probability p")
    plt.ylabel("Running time [s]")
    plt.yscale("log")
    plt.title("Figure 3 – Time vs p")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figure3_time.png")
    print("Figures saved to figure2_epc.png and figure3_time.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="k‑pCND reproducibility script")
    ap.add_argument("--model", choices=["er", "ba", "ws"], default="er")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.5, help="edge probability for fixed distr")
    ap.add_argument("--edge-distr", choices=["fixed", "uniform"], default="fixed")
    ap.add_argument("--prob-set", type=float, nargs="*", default=[0.1 * i for i in range(1, 11)],
                    help="list of probabilities for 'uniform' distribution")
    ap.add_argument("--eps", type=float, default=0.05, help="FPRAS ε")
    ap.add_argument("--demo", action="store_true", help="quick sanity run (default)")
    ap.add_argument("--full", action="store_true", help="produce Figures 2 & 3")
    args = ap.parse_args()

    if not args.demo and not args.full:
        args.demo = True  # default action

    if args.demo:
        print("[DEMO] single run → table of nodes/edges removed + runtime")
        pg = make_random_graph(args.model, args.n, args.m, args.p,
                                args.edge_distr, args.prob_set)
        res = run_all(pg, args.k, show_tqdm=True)
        print("Method\tRemoved\tTime[s]")
        for name, (removed, dur) in res.items():
            print(f"{name}\t{removed}\t{dur:.3f}")

    if args.full:
        print("[FULL] sweeping p and generating figures …")
        full_benchmark(args.model, args.n, args.m, args.k, args.edge_distr,
                       args.prob_set, args.eps)


if __name__ == "__main__":
    main()
