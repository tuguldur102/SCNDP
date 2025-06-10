# kpcnd.py – **v6** (2025‑06‑11)
"""Stochastic k‑Critical‑Node Detection benchmark suite
========================================================
*Reproduces Thai et al. (INFOCOM 2015) Figures 2–3 and extends them with
Greedy‑MC and Prob‑Degree heuristics from the Stochastic‑CNDP draft.*

Changes in **v6**
-----------------
1. **PageRank now removes *nodes*** (bug‑fix; paper uses k‑node variants).
2. **Multi‑seed averaging** (`--seeds N`, default 30 for `--full`) matches
   the paper’s “30 independent graphs per data point”.
3. Paper‑matching defaults for full runs: *n = 200, m ≈ 400, k = 10,
   ε = 0.01*.
4. Progress bars on outer seed‑loop & p‑sweep; runtime averages reported.

Quick start
-----------
```bash
# Very fast sanity check on one graph (n=100, k=5)
python kpcnd.py --demo

# Full reproduction of Fig 2(a)/Fig 3(a) (Erdős‑Rényi)
python kpcnd.py --full --model er

# BA & WS variants
python kpcnd.py --full --model ba
python kpcnd.py --full --model ws
```

Dependencies: `networkx numpy pulp tqdm matplotlib` (+ `gurobipy` optional).
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

SAMPLES = 10000
T = 30
# ---------------------------------------------------------------------------
# Utility helpers
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
# Probabilistic graph wrapper
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticGraph:
    g: nx.Graph
    p: Dict[Tuple[int, int], float]  # edge‑existence probabilities

    # --------------------------- Sampling -----------------------------
    def sample_realisation(self, rng: np.random.Generator | None = None) -> nx.Graph:
        if rng is None:
            rng = np.random.default_rng()
        h = nx.Graph(); h.add_nodes_from(self.g.nodes)
        for (u, v) in self.g.edges:
            if rng.random() < self.p[edge_key(u, v)]:
                h.add_edge(u, v)
        return h

    # -------------------- Monte‑Carlo EPC (debug) ---------------------
    def monte_carlo_epc(self, samples: int = SAMPLES, rng: np.random.Generator | None = None,
                        show_bar: bool = False) -> float:
        if rng is None:
            rng = np.random.default_rng()
        n = self.g.number_of_nodes(); pairs = n * (n - 1) // 2
        acc = 0
        it = tqdm(range(samples), desc="MC", leave=False) if show_bar else range(samples)
        for _ in it:
            acc += connected_pair_count(self.sample_realisation(rng))
        return acc / (samples * pairs)


# ---------------------------------------------------------------------------
# Deterministic helper
# ---------------------------------------------------------------------------

def connected_pair_count(g: nx.Graph) -> int:
    total = 0
    for cc in nx.connected_components(g):
        m = len(cc); total += m * (m - 1) // 2
    return total


# ---------------------------------------------------------------------------
# FPRAS – Component‑Sampling Procedure (Thai et al. Alg 2)
# ---------------------------------------------------------------------------

def csp_epc_estimate(pg: ProbabilisticGraph, eps: float = 0.05, delta: float = 1e-2,
                     rng: np.random.Generator | None = None, show_bar: bool = False) -> float:
    if rng is None:
        rng = np.random.default_rng()
    g = pg.g; n = g.number_of_nodes()
    p_e = sum(pg.p.values())
    if p_e < eps ** 2 / (n ** 2):
        return p_e
    m = g.number_of_edges()
    mu_hat = min((1 + p_e / m) ** m, n * (n - 1) / 2)
    N = int(np.ceil(4 * np.e ** 2 * np.log(2 / delta) / (eps ** 2 * mu_hat)))

    acc = 0; nodes = tuple(g.nodes)
    it = tqdm(range(N), desc="CSP", leave=False) if show_bar else range(N)
    for _ in it:
        u = rng.choice(nodes)
        queue, visited = [u], {u}
        while queue:
            v = queue.pop()
            for w in g.adj[v]:
                if w in visited:
                    continue
                if rng.random() < pg.p[edge_key(v, w)]:
                    visited.add(w); queue.append(w)
        acc += len(visited) - 1
    return n * acc / (2 * N)


# ---------------------------------------------------------------------------
# Baseline heuristics – node variants
# ---------------------------------------------------------------------------

def betweenness_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    btw = nx.betweenness_centrality(pg.g)
    return sorted(btw, key=btw.get, reverse=True)[:k]


def pagerank_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    pr = nx.pagerank(pg.g, alpha=0.85)
    return sorted(pr, key=pr.get, reverse=True)[:k]

# ---------------------------------------------------------------------------
# Probabilistic degree heuristic
# ---------------------------------------------------------------------------

def prob_degree_node_removal(pg: ProbabilisticGraph, k: int) -> List[int]:
    score = {u: 0.0 for u in pg.g.nodes}
    for (u, v), p in pg.p.items():
        score[u] += p; score[v] += p
    return sorted(score, key=score.get, reverse=True)[:k]


# ---------------------------------------------------------------------------
# REGA (Thai et al. Alg 1)
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

    def _epc_after_node_removal(self, rem: Sequence[int], eps, delta) -> float:
        g2 = self.pg.g.copy(); g2.remove_nodes_from(rem)
        p2 = {edge_key(u, v): self.pg.p[edge_key(u, v)] for (u, v) in g2.edges}
        return csp_epc_estimate(ProbabilisticGraph(g2, p2), eps, delta, self._rng, False)

    def run(self, eps: float, delta: float = 1e-2) -> List[int]:
        g = self.pg.g
        selected, fixed = [], set()
        for _ in range(self.k):
            frac = self._solve_lp_relaxation()
            cand = max((i for i in range(self.n) if i not in fixed), key=lambda i: frac[i])
            selected.append(cand); fixed.add(cand)
        improved = True
        while improved:
            improved = False; epc_sel = self._epc_after_node_removal(selected, eps, delta)
            for u in selected:
                for v in (set(g.nodes) - set(selected)):
                    trial = [x for x in selected if x != u] + [v]
                    epc_trial = self._epc_after_node_removal(trial, eps, delta)
                    if epc_trial < epc_sel:
                        selected, epc_sel = trial, epc_trial; improved = True; break
                if improved: break
        return selected


# ---------------------------------------------------------------------------
# SAA baseline
# ---------------------------------------------------------------------------

class SAA:
    def __init__(self, pg: ProbabilisticGraph, k: int, T: int = 30):
        self.pg, self.k, self.T = pg, k, T

    def run(self) -> List[int]:
        rng = np.random.default_rng(); samples = [self.pg.sample_realisation(rng) for _ in range(self.T)]
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
# Greedy Monte‑Carlo heuristic
# ---------------------------------------------------------------------------

class GreedyMC:
    def __init__(self, pg: ProbabilisticGraph, k: int, samples: int = SAMPLES):
        self.pg, self.k, self.samples = pg, k, samples
        self._rng = np.random.default_rng()

    def _sigma(self, rem: Sequence[int]) -> float:
        g2 = self.pg.g.copy(); g2.remove_nodes_from(rem)
        p2 = {edge_key(u, v): self.pg.p[edge_key(u, v)] for (u, v) in g2.edges}
        return ProbabilisticGraph(g2, p2).monte_carlo_epc(self.samples, self._rng, False)

    def run(self) -> List[int]:
        S: List[int] = []; baseline = self._sigma(S)
        for _ in tqdm(range(self.k), desc="GreedyMC", leave=False):
            best_gain, best_v = -1, None
            for v in (set(self.pg.g.nodes) - set(S)):
                sig = self._sigma(S + [v]); gain = baseline - sig
                if gain > best_gain:
                    best_gain, best_v = gain, v
            S.append(best_v); baseline -= best_gain
        return S


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def make_random_graph(model: str, n: int, m: int, p: float, edge_distr: str,
                      prob_set: Sequence[float], seed: int) -> ProbabilisticGraph:
    rng = random.Random(seed)
    if model == "er":
        g = nx.gnm_random_graph(n, m, seed=seed)
    elif model == "ba":
        g = nx.barabasi_albert_graph(n, max(1, m // n), seed=seed)
    elif model == "ws":
        k_neigh = max(2, int(round(2 * m / n)))
        g = nx.watts_strogatz_graph(n, k_neigh, 0.3, seed=seed)
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
# Run a set of methods on one graph
# ---------------------------------------------------------------------------

def run_all(pg: ProbabilisticGraph, k: int, methods: Sequence[str], samples: int,
            eps: float) -> Dict[str, Tuple[Sequence[int], float]]:
    out: Dict[str, Tuple[Sequence[int], float]] = {}

    def record(name: str, fn):
        t0 = time.perf_counter(); sol = fn(); dur = time.perf_counter() - t0
        out[name] = (sol, dur)

    for m in tqdm(methods, desc="methods"):
        if m == "REGA":
            record(m, lambda: REGA(pg, k).run(eps))
        elif m == "SAA":
            record(m, lambda: SAA(pg, k).run())
        elif m == "Betweenness":
            record(m, lambda: betweenness_node_removal(pg, k))
        elif m == "PageRank":
            record(m, lambda: pagerank_node_removal(pg, k))
        elif m == "GreedyMC":
            record(m, lambda: GreedyMC(pg, k, samples).run())
        elif m == "ProbDegree":
            record(m, lambda: prob_degree_node_removal(pg, k))
        else:
            raise ValueError(m)
    return out

# ---------------------------------------------------------------------------
# Full benchmark – averages over multiple seeds
# ---------------------------------------------------------------------------

def full_benchmark(args):
    methods = args.methods
    ps = np.arange(0.1, 1.01, 0.1)
    epc_sum = {m: np.zeros_like(ps, dtype=float) for m in methods}
    time_sum = {m: np.zeros_like(ps, dtype=float) for m in methods}

    for seed in tqdm(range(args.seeds), desc="seeds"):
        rng = np.random.default_rng(seed)
        for idx, p in enumerate(ps):
            pg = make_random_graph(args.model, args.n, args.m, p, args.edge_distr,
                                   args.prob_set, seed * 100 + idx)  # unique seed per graph
            res = run_all(pg, args.k, methods, args.samples, args.eps)
            for m in methods:
                sol, t = res[m]
                g2 = pg.g.copy(); g2.remove_nodes_from(sol)
                p2 = {edge_key(u, v): pg.p[edge_key(u, v)] for (u, v) in g2.edges}
                epc = csp_epc_estimate(ProbabilisticGraph(g2, p2), args.eps, 0.01, rng, False)
                epc_sum[m][idx] += epc; time_sum[m][idx] += t

    epc_avg = {m: epc_sum[m] / args.seeds for m in methods}
    time_avg = {m: time_sum[m] / args.seeds for m in methods}

    # --------- plotting ---------
    plt.figure()
    for m in methods:
        plt.plot(ps, epc_avg[m], label=m, marker="o")
    plt.xlabel("Edge existence probability p"); plt.ylabel("EPC (lower is better)")
    plt.title("Figure 2 – EPC vs p")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig("figure2_epc.png")

    plt.figure()
    for m in methods:
        plt.plot(ps, time_avg[m], label=m, marker="o")
    plt.xlabel("Edge existence probability p"); plt.ylabel("Running time [s]")
    plt.yscale("log"); plt.title("Figure 3 – Time vs p")
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout(); plt.savefig("figure3_time.png")
    print("Saved figure2_epc.png and figure3_time.png")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="k‑pCND benchmark – v6 (paper‑faithful)")
    ap.add_argument("--model", choices=["er", "ba", "ws"], default="er")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--edge-distr", choices=["fixed", "uniform"], default="fixed")
    ap.add_argument("--prob-set", type=float, nargs="*", default=[0.1 * i for i in range(1, 11)])
    ap.add_argument("--eps", type=float, default=0.01, help="FPRAS ε (REGA & eval)")
    ap.add_argument("--samples", type=int, default=SAMPLES, help="MC samples for GreedyMC")
    ap.add_argument("--methods", type=str,
                    # default="REGA,SAA,Betweenness,PageRank,GreedyMC,ProbDegree")
                    default="REGA,SAA,Betweenness,PageRank")
    ap.add_argument("--seeds", type=int, default=1, help="# graphs per p (30 in paper)")
    ap.add_argument("--demo", action="store_true", help="single‑graph quick run")
    ap.add_argument("--full", action="store_true", help="sweep p & average over seeds")
    return ap.parse_args()


def main():
    args = parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # If user picked --full but left default seeds=1, raise to 30 as in paper
    if args.full and args.seeds == 1:
        args.seeds = T

    if args.demo or not args.full:
        pg = make_random_graph(args.model, args.n, args.m, args.p, args.edge_distr,
                               args.prob_set, seed=42)
        res = run_all(pg, args.k, args.methods, args.samples, args.eps)
        print("Method	Removed	Time[s]")
        for m, (rem, t) in res.items():
            print(f"{m}	{rem}	{t:.3f}")

    if args.full:
        full_benchmark(args)


if __name__ == "__main__":
    main()
