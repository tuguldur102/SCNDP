# kpcnd.py – v5 (2025‑06‑10)
"""Stochastic Critical‑Node Detection benchmark suite
======================================================
Implements all baselines from **Thai et al. (2015)** plus the new
heuristics sketched in the draft *Stochastic CNDP with uncertain edges*fileciteturn5file0:

* **Greedy‑MC** – Algorithm 2 in the draft (greedy node selection using the
  Monte‑Carlo estimator of expected pairwise connectivity, σ).  Provides the
  (1 – 1/e) approximation guarantee for monotone sub‑modular objectives
  (§2, Theorem 1).  MC sample size is configurable (default = 2000).
* **Prob‑Degree** – degree‑centrality heuristic weighted by edge probabilities
  (end of §2 in the draft).
* Existing baselines: **REGA**, **SAA**, **Betweenness**, **PageRank**.

New CLI flags
-------------
* `--samples N` – number of Monte‑Carlo samples for Greedy‑MC (≤50 000).
* `--methods …` – comma‑separated subset to run (defaults to all seven).

Example
-------
```bash
# Quick demo with the new heuristics
python kpcnd.py --demo --methods GreedyMC,ProbDegree

# Full sweep with 5 000 MC samples
python kpcnd.py --full --samples 5000
```

Dependencies: `networkx numpy pulp tqdm matplotlib` (+ `gurobipy` optional).
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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Basic graph helpers
# ---------------------------------------------------------------------------

def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def ordered(u: int, v: int) -> Tuple[int, int]:
    return edge_key(u, v)


def combinations_index(nodes: Iterable[int]):
    nodes = list(nodes)
    for i, j in itertools.combinations(nodes, 2):
        yield (i, j)


# ---------------------------------------------------------------------------
# ProbabilisticGraph wrapper
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticGraph:
    g: nx.Graph
    p: Dict[Tuple[int, int], float]  # edge probabilities

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
    # Monte‑Carlo EPC (slow but unbiased)
    # ---------------------------------------------------------------------
    def monte_carlo_epc(self, samples: int = 2000, rng: np.random.Generator | None = None,
                        prog: bool = False) -> float:
        """Return *normalised* EPC in [0,1] via simple MC sampling."""
        if rng is None:
            rng = np.random.default_rng()
        n = self.g.number_of_nodes()
        pairs = n * (n - 1) // 2
        acc = 0
        it = tqdm(range(samples), desc="MC", leave=False) if prog else range(samples)
        for _ in it:
            acc += connected_pair_count(self.sample_realisation(rng))
        return acc / (samples * pairs)


# ---------------------------------------------------------------------------
# Helper: connected pair count in deterministic graph
# ---------------------------------------------------------------------------

def connected_pair_count(g: nx.Graph) -> int:
    total = 0
    for cc in nx.connected_components(g):
        m = len(cc)
        total += m * (m - 1) // 2
    return total


# ---------------------------------------------------------------------------
# FPRAS (Thai et al., Algorithm 2)
# ---------------------------------------------------------------------------

def csp_epc_estimate(pg: ProbabilisticGraph, eps: float = 0.05, delta: float = 1e-2,
                     rng: np.random.Generator | None = None, show_bar: bool = False) -> float:
    if rng is None:
        rng = np.random.default_rng()
    g, n = pg.g, pg.g.number_of_nodes()
    p_e = sum(pg.p.values())
    if p_e < eps ** 2 / (n ** 2):
        return p_e  # cheap bound
    m = g.number_of_edges()
    mu_hat = min((1 + p_e / m) ** m, n * (n - 1) / 2)
    N = int(np.ceil(4 * np.e ** 2 * np.log(2 / delta) / (eps ** 2 * mu_hat)))

    acc = 0
    it = tqdm(range(N), desc="CSP", leave=False) if show_bar else range(N)
    nodes = tuple(g.nodes)
    for _ in it:
        u = rng.choice(nodes)  # FIX: only choose existing IDs
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
# Baseline heuristics (existing)
# ---------------------------------------------------------------------------

def betweenness_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    btw = nx.betweenness_centrality(pg.g)
    return sorted(btw, key=btw.get, reverse=True)[:k]


def pagerank_edge_removal(pg: ProbabilisticGraph, k: int) -> List[Tuple[int, int]]:
    pr = nx.pagerank(pg.g, alpha=0.85)
    edges = sorted(pg.g.edges, key=lambda e: pr[e[0]] + pr[e[1]], reverse=True)
    return edges[:k]


# ---------------------------------------------------------------------------
# REGA (Thai et al., Algorithm 1) – unchanged from v4
# ---------------------------------------------------------------------------

class REGA:
    def __init__(self, pg: ProbabilisticGraph, k: int):
        self.pg, self.k = pg, k
        self.n = pg.g.number_of_nodes()
        self.lp_solver = "GUROBI" if os.getenv("KPCND_GUROBI") else "PULP_CBC_CMD"
        self._rng = np.random.default_rng()

    def _solve_lp_relaxation(self) -> np.ndarray:
        g = self.pg.g
        prob = pulp.LpProblem("MIPR_LP", pulp.LpMinimize)
        s = pulp.LpVariable.dicts("s", g.nodes, 0, 1)
        x = pulp.LpVariable.dicts("x", combinations_index(g.nodes), 0, 1)
        prob += pulp.lpSum(1 - x[i, j] for (i, j) in x)
        prob += pulp.lpSum(s[i] for i in g.nodes) <= self.k
        pij = self.pg.p
        for (i, j) in g.edges:
            prob += x[ordered(i, j)] <= s[i] + s[j] + 1 - pij[edge_key(i, j)]
        for (i, j) in g.edges:
            for k in g.nodes:
                if k in (i, j):
                    continue
                prob += x[ordered(i, j)] + x[ordered(j, k)] >= x[ordered(i, k)]
        solver = pulp.getSolver(self.lp_solver, timeLimit=300, msg=False)
        prob.solve(solver)
        return np.array([s[i].value() for i in range(self.n)])

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

    def _epc_after_node_removal(self, rem: Sequence[int], eps, delta) -> float:
        g2 = self.pg.g.copy(); g2.remove_nodes_from(rem)
        p2 = {edge_key(u, v): self.pg.p[edge_key(u, v)] for (u, v) in g2.edges}
        return csp_epc_estimate(ProbabilisticGraph(g2, p2), eps, delta, self._rng, False)


# ---------------------------------------------------------------------------
# SAA baseline (unchanged)
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
        avg_g = nx.Graph(); avg_g.add_nodes_from(self.pg.g.nodes)
        for e, cnt in freq.items():
            if cnt > 0:
                avg_g.add_edge(*e)
        avg_pg = ProbabilisticGraph(avg_g, {e: cnt / self.T for e, cnt in freq.items()})
        return betweenness_node_removal(avg_pg, self.k)


# ---------------------------------------------------------------------------
# NEW – Greedy Monte‑Carlo heuristic (Algorithm 2 in draft)
# ---------------------------------------------------------------------------

class GreedyMC:
    """Greedy node selection using MC estimator of σ(S)."""

    def __init__(self, pg: ProbabilisticGraph, k: int, samples: int = 2000):
        self.pg, self.k, self.samples = pg, k, samples
        self._rng = np.random.default_rng()

    def _sigma(self, rem: Sequence[int]) -> float:
        g2 = self.pg.g.copy(); g2.remove_nodes_from(rem)
        p2 = {edge_key(u, v): self.pg.p[edge_key(u, v)] for (u, v) in g2.edges}
        pg2 = ProbabilisticGraph(g2, p2)
        return pg2.monte_carlo_epc(self.samples, self._rng, prog=False)

    def run(self) -> List[int]:
        S: List[int] = []
        baseline = self._sigma(S)
        for _ in tqdm(range(self.k), desc="GreedyMC", leave=False):
            best_gain, best_v = -1, None
            for v in (set(self.pg.g.nodes) - set(S)):
                sig = self._sigma(S + [v])
                gain = baseline - sig
                if gain > best_gain:
                    best_gain, best_v = gain, v
            S.append(best_v)
            baseline -= best_gain  # new σ(S)
        return S


# ---------------------------------------------------------------------------
# NEW – Probabilistic degree‑centrality heuristic
# ---------------------------------------------------------------------------

def prob_degree_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    score = {u: 0.0 for u in pg.g.nodes}
    for (u, v), p in pg.p.items():
        score[u] += p; score[v] += p
    return sorted(score, key=score.get, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Graph generators (v4 + unchanged)
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
# Experiment orchestrators
# ---------------------------------------------------------------------------

def run_all(pg: ProbabilisticGraph, k: int, methods: Sequence[str], samples: int,
            show_bar: bool = False):
    out: Dict[str, Tuple[Sequence[int] | Sequence[Tuple[int, int]], float]] = {}
    rng = np.random.default_rng()

    def record(name, func):
        t0 = time.perf_counter()
        res = func()
        out[name] = (res, time.perf_counter() - t0)

    for m in methods:
        if m == "REGA":
            record("REGA", lambda: REGA(pg, k).run())
        elif m == "SAA":
            record("SAA", lambda: SAA(pg, k).run())
        elif m == "Betweenness":
            record("Betweenness", lambda: betweenness_node_removal(pg, k))
        elif m == "PageRank":
            record("PageRank", lambda: pagerank_edge_removal(pg, k))
        elif m == "GreedyMC":
            record("GreedyMC", lambda: GreedyMC(pg, k, samples).run())
        elif m == "ProbDegree":
            record("ProbDegree", lambda: prob_degree_node_removal(pg, k))
        else:
            raise ValueError(f"Unknown method {m}")
    return out


# Full benchmark (extends v4 with new methods)
# ---------------------------------------------------------------------------

def full_benchmark(model: str, n: int, m: int, k: int, edge_distr: str,
                   prob_set: Sequence[float], eps: float, samples: int,
                   methods: Sequence[str]):
    ps = np.arange(0.1, 1.01, 0.1)
    epc_curves = {meth: [] for meth in methods}
    time_curves = {meth: [] for meth in methods}
    rng = np.random.default_rng()

    for p in tqdm(ps, desc="p sweep"):
        pg = make_random_graph(model, n, m, p, edge_distr, prob_set)
        out = run_all(pg, k, methods, samples)
        for meth in methods:
            sol, t = out[meth]
            if meth == "PageRank":
                g2 = pg.g.copy(); g2.remove_edges_from(sol)
            else:
                g2 = pg.g.copy(); g2.remove_nodes_from(sol)
            p2 = {edge_key(u, v): pg.p[edge_key(u, v)] for (u, v) in g2.edges}
            epc = csp_epc_estimate(ProbabilisticGraph(g2, p2), eps, 0.01, rng, False)
            epc_curves[meth].append(epc)
            time_curves[meth].append(t)

    # Plot EPC
    plt.figure()
    for meth in methods:
        plt.plot(ps, epc_curves[meth], label=meth, marker="o")
    plt.xlabel("Edge existence probability p")
    plt.ylabel("EPC (lower is better)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("figure2_epc.png")

    # Plot runtime (log‑scale)
    plt.figure()
    for meth in methods:
        plt.plot(ps, time_curves[meth], label=meth, marker="o")
    plt.xlabel("Edge existence probability p"); plt.ylabel("Runtime [s]")
    plt.yscale("log"); plt.legend(); plt.grid(True, which="both"); plt.tight_layout()
    plt.savefig("figure3_time.png")
    print("Figures saved to figure2_epc.png and figure3_time.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="k-pCND benchmark suite with new heuristics")
    parser.add_argument("--model", choices=["er", "ba", "ws"], default="er")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--edge-distr", choices=["fixed", "uniform"], default="fixed")
    parser.add_argument("--prob-set", type=float, nargs="*", default=[0.1 * i for i in range(1, 11)])
    parser.add_argument("--eps", type=float, default=0.05, help="FPRAS ε for evaluation")
    parser.add_argument("--samples", type=int, default=2000, help="MC samples for GreedyMC")
    parser.add_argument("--methods", type=str, default="REGA,SAA,Betweenness,PageRank,GreedyMC,ProbDegree")
    parser.add_argument("--demo", action="store_true", help="quick run on a single p value")
    parser.add_argument("--full", action="store_true", help="sweep p and generate plots")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not args.demo and not args.full:
        args.demo = True

    if args.demo:
        print("[DEMO] single run → nodes/edges removed + runtime\n")
        pg = make_random_graph(args.model, args.n, args.m, args.p, args.edge_distr, args.prob_set)
        res = run_all(pg, args.k, methods, args.samples, show_bar=True)
        print("Method\tRemoved\tTime[s]")
        for name, (removed, dur) in res.items():
            print(f"{name}\t{removed}\t{dur:.3f}")

    if args.full:
        print("[FULL] sweeping p and generating figures …")
        full_benchmark(args.model, args.n, args.m, args.k, args.edge_distr, args.prob_set,
                       args.eps, args.samples, methods)


if __name__ == "__main__":
    main()
