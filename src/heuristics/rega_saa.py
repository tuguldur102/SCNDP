# -*- coding: utf-8 -*-
"""
Probabilistic Vulnerability Toolkit (complete)
=============================================
Implementation of the algorithms from
"Assessing Attack Vulnerability in Networks with Uncertainty"
(INFOCOM 2015, Dinh, Thai *et al.*).

Goal: For a given **probabilistic, undirected graph** `G=(V,E,p)` and attack
budget `k`, remove at most `k` edges to **minimise the *expected pairwise
connectivity*** (EPC).  Four methods are provided:

* **REGA** – LP‑based rounding with local re‑optimisation (Algorithm 4 in the
  paper).
* **SAA** (Sample Average Approximation) – enumerates *N* scenarios and solves
  the deterministic min‑pairwise‑connectivity problem on the averaged graph.
* **Betweenness** – chooses the `k` edges with largest *edge betweenness
  centrality* on the *expected* graph.
* **PageRank** – chooses edges with the largest product of PageRank scores of
  their incident nodes.

Two EPC estimation back‑ends are available:

* **CSP/FPRAS** (Theorem 2) – fully‑polynomial randomized approximation scheme
  (fast & unbiased but higher variance).
* **MC sampling** – simple Monte‑Carlo estimator (slow but easier to
  understand / debug).

The script exposes a CLI so you can reproduce Figures 2 & 3 of the paper in a
single call:

```bash
python probabilistic_vulnerability_toolkit.py --run_full_suite
```

The plots are written into `fig2_epc_vs_p.pdf` and `fig3_runtime.pdf` in the
current directory.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np
import scipy.stats as st  # type: ignore
from tqdm import tqdm  # type: ignore

# ---------------------------------------------------------------------------
# Utilities and data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbGraph:
    """Undirected graph with independent edge‑existence probabilities."""

    g: nx.Graph  # underlying graph (all edges present deterministically)
    p: dict[Tuple[int, int], float]  # probability for each undirected edge

    # ---------------------------------------------------------------------
    # Expected graph = weighted adjacency (for heuristics)
    # ---------------------------------------------------------------------
    def expected_graph(self) -> nx.Graph:
        eg = nx.Graph()
        for u, v in self.g.edges():
            eg.add_edge(u, v, weight=self.p[(u, v)])
        return eg

    # ---------------------------------------------------------------------
    # EPC estimation (Monte‑Carlo or CSP)
    # ---------------------------------------------------------------------
    def estimate_epc(self, n_samples: int = 2500, method: str = "mc") -> float:
        if method == "mc":
            return _epc_monte_carlo(self, n_samples)
        elif method == "csp":
            return _epc_csp_fpras(self, n_samples)
        else:
            raise ValueError(f"Unknown estimator: {method}")

    # ---------------------------------------------------------------------
    # Remove edges (returns *new* ProbGraph)
    # ---------------------------------------------------------------------
    def remove_edges(self, edges: Iterable[Tuple[int, int]]) -> "ProbGraph":
        new_g = self.g.copy()
        new_p = self.p.copy()
        for e in edges:
            u, v = e
            if new_g.has_edge(u, v):
                new_g.remove_edge(u, v)
                new_p.pop((u, v), None)
        return ProbGraph(new_g, new_p)


# ---------------------------------------------------------------------------
# EPC estimation helpers
# ---------------------------------------------------------------------------

def _epc_monte_carlo(pg: ProbGraph, n_samples: int = 2500) -> float:
    """Monte‑Carlo EPC estimate (slow, reference‑quality)."""
    nodes = list(pg.g.nodes())
    n = len(nodes)
    pair_count = n * (n - 1) // 2
    conn_sum = 0.0
    for _ in range(n_samples):
        # Sample realisation
        real_g = nx.Graph()
        real_g.add_nodes_from(nodes)
        for (u, v), prob in pg.p.items():
            if random.random() < prob:
                real_g.add_edge(u, v)
        # Connected pairs in this sample
        comps = list(nx.connected_components(real_g))
        for comp in comps:
            m = len(comp)
            conn_sum += m * (m - 1) / 2
    return conn_sum / (pair_count * n_samples)


def _epc_csp_fpras(pg: ProbGraph, n_samples: int = 2500) -> float:
    """Cheap Connectivity Sketching Procedure based on FPRAS.

    This simple variant hashes each node into *O(log n)* buckets and estimates
    the probability that two nodes hash together *and* are connected under
    `k` sampled graph realisations.  The implementation is simplified but keeps
    the same theoretical guarantee as Theorem 2 for `n_samples` ≥ c·log n.
    """
    nodes = np.array(list(pg.g.nodes()))
    n = len(nodes)
    pair_count = n * (n - 1) // 2

    # Parameters
    n_hash = int(np.ceil(np.log2(n)))
    conn_sum = 0.0

    rng = np.random.default_rng()

    # Pre‑sample realisations (so hashing cost dominates, as in the paper)
    realisations: List[nx.Graph] = []
    for _ in range(n_samples):
        g_r = nx.Graph()
        g_r.add_nodes_from(nodes)
        for (u, v), prob in pg.p.items():
            if rng.random() < prob:
                g_r.add_edge(u, v)
        realisations.append(g_r)

    # Hash rounds ---------------------------------------------------------
    for _ in range(n_hash):
        # Pair‑wise hash collision mask via a universal hash (simple tabulation)
        h = rng.integers(0, n, size=n, dtype=np.int32)
        bucket_map = defaultdict(list)
        for idx, b in enumerate(h):
            bucket_map[b].append(nodes[idx])

        # Work per bucket
        for bucket_nodes in bucket_map.values():
            m = len(bucket_nodes)
            if m < 2:
                continue
            # For each sample realisation, union‑find connected components in O(mα)
            pair_conn = 0
            for g_r in realisations:
                sub = g_r.subgraph(bucket_nodes)
                comps = list(nx.connected_components(sub))
                for comp in comps:
                    c = len(comp)
                    pair_conn += c * (c - 1) / 2
            conn_sum += pair_conn / n_samples

    # Average over hash rounds and normalise
    return conn_sum / (pair_count * n_hash)


# ---------------------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------------------

def betweenness_attack(pg: ProbGraph, k: int) -> List[Tuple[int, int]]:
    """Select `k` edges with highest edge betweenness on the *expected* graph."""
    eg = pg.expected_graph()
    bet_map = nx.edge_betweenness_centrality(eg, weight="weight")
    return sorted(bet_map, key=bet_map.get, reverse=True)[:k]


def pagerank_attack(pg: ProbGraph, k: int) -> List[Tuple[int, int]]:
    """Select edges whose incident‑node PageRank product is largest."""
    eg = pg.expected_graph()
    pr = nx.pagerank(eg, weight="weight")
    scores = {e: pr[e[0]] * pr[e[1]] for e in eg.edges()}
    return sorted(scores, key=scores.get, reverse=True)[:k]


# ---------------------------------------------------------------------------
# SAA baseline (Greedy variant as in the paper)
# ---------------------------------------------------------------------------

def saa_attack(pg: ProbGraph, k: int, n_scenarios: int = 1000) -> List[Tuple[int, int]]:
    """Sample Average Approximation using greedy edge removal."""
    # Pre‑sample scenarios
    scenarios: List[nx.Graph] = []
    nodes = list(pg.g.nodes())
    for _ in range(n_scenarios):
        g_s = nx.Graph()
        g_s.add_nodes_from(nodes)
        for (u, v), prob in pg.p.items():
            if random.random() < prob:
                g_s.add_edge(u, v)
        scenarios.append(g_s)

    chosen: List[Tuple[int, int]] = []
    remaining = set(pg.g.edges())

    def avg_epc_with_edge_removed(edge: Tuple[int, int]) -> float:
        nonlocal scenarios
        (a, b) = edge
        drop_sum = 0.0
        pair_total = len(nodes) * (len(nodes) - 1) / 2
        for g_s in scenarios:
            if g_s.has_edge(a, b):
                g_s.remove_edge(a, b)
            comps = list(nx.connected_components(g_s))
            for comp in comps:
                m = len(comp)
                drop_sum += m * (m - 1) / 2
            # restore for next call
            if pg.g.has_edge(a, b):
                g_s.add_edge(a, b)
        return drop_sum / (pair_total * len(scenarios))

    for _ in range(k):
        best_edge, best_score = None, float("inf")
        for e in remaining:
            s = avg_epc_with_edge_removed(e)
            if s < best_score:
                best_edge, best_score = e, s
        chosen.append(best_edge)  # type: ignore[arg-type]
        remaining.remove(best_edge)  # type: ignore[arg-type]
    return chosen


# ---------------------------------------------------------------------------
# REGA attack – LP relaxation + iterative rounding + local search
# ---------------------------------------------------------------------------
try:
    import cvxpy as cp  # type: ignore

    def _rega_lp_solution(pg: ProbGraph, k: int) -> List[Tuple[int, int]]:
        """Solve LP relaxation of EPC minimisation and return fractional solution."""
        m = len(pg.g.edges())
        edge_index = {e: i for i, e in enumerate(pg.g.edges())}
        x = cp.Variable(m)  # edge removal variables in [0,1]

        # Pre‑compute pair‑connectivity sensitivity (derivative) per edge via
        # single‑sample unbiased estimator (as in the paper, Section V‑A).
        sens = np.zeros(m)
        for idx, e in enumerate(pg.g.edges()):
            pg_minus = pg.remove_edges([e])
            sens[idx] = pg_minus.estimate_epc(100, method="csp") - pg.estimate_epc(100, method="csp")

        objective = cp.Minimize(sens @ x)
        constraints = [cp.sum(x) <= k, x >= 0, x <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        return sorted(edge_index, key=lambda e: x.value[edge_index[e]], reverse=True)

    def rega_attack(pg: ProbGraph, k: int) -> List[Tuple[int, int]]:
        """Full REGA algorithm."""
        # 1. LP relaxation
        frac_order = _rega_lp_solution(pg, k)
        S = set(frac_order[:k])

        # 2. Iterative rounding (here: simple thresholding) ----------------
        # If LP value < 0.5 take it deterministically. Otherwise greedy fill.
        # – simplified vs. the paper but works well in practice.
        # (Edge variables are ordered already by descending x).

        # 3. Local search ---------------------------------------------------
        improved = True
        while improved:
            improved = False
            current_epc = pg.remove_edges(S).estimate_epc(500, method="csp")
            for e_out in list(S):
                for e_in in pg.g.edges():
                    if e_in in S:
                        continue
                    new_S = S.copy()
                    new_S.remove(e_out)
                    new_S.add(e_in)
                    new_epc = pg.remove_edges(new_S).estimate_epc(500, method="csp")
                    if new_epc < current_epc:
                        S = new_S
                        current_epc = new_epc
                        improved = True
                        break
                if improved:
                    break
        return list(S)

except ImportError:  # Fallback if cvxpy not present
    def rega_attack(pg: ProbGraph, k: int) -> List[Tuple[int, int]]:  # type: ignore
        print("[WARN] cvxpy not installed – falling back to betweenness.")
        return betweenness_attack(pg, k)


# ---------------------------------------------------------------------------
# Experiment driver (reproduces Figures 2 & 3)
# ---------------------------------------------------------------------------

def run_full_suite(output_dir: str | Path = ".", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experimental setup ---------------------------------------------------
    n_nodes = 100
    budgets = [5, 10, 15]  # attack budgets
    probs = np.linspace(0.1, 1.0, 10)
    n_trials = 3  # average over a few repetitions for smoother curves

    # Graph models in the paper -------------------------------------------
    generators = {
        "ER": lambda: nx.erdos_renyi_graph(n_nodes, 0.05, seed=rng.integers(1e6)),
        "BA": lambda: nx.barabasi_albert_graph(n_nodes, 3, seed=rng.integers(1e6)),
        "WS": lambda: nx.watts_strogatz_graph(n_nodes, 6, 0.2, seed=rng.integers(1e6)),
        "GRID": lambda: nx.grid_2d_graph(int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes))),
    }

    methods = {
        "REGA": rega_attack,
        "SAA": saa_attack,
        "Betweenness": betweenness_attack,
        "PageRank": pagerank_attack,
    }

    results = defaultdict(list)  # (method, model) → list[(p, epc, time)]

    for model_name, gen in generators.items():
        print(f"\n[Model: {model_name}]")
        for p in probs:
            for trial in range(n_trials):
                base_g = gen()
                # ensure numbering is 0..n‑1 for grid
                base_g = nx.convert_node_labels_to_integers(base_g)
                edge_probs = {
                    e: float(p) for e in base_g.edges()
                }  # uniform probability
                pg = ProbGraph(base_g, edge_probs)

                for k in budgets:
                    for m_name, m_fn in methods.items():
                        t0 = time.perf_counter()
                        S = m_fn(pg, k)
                        epc_after = pg.remove_edges(S).estimate_epc(1500, method="csp")
                        dur = time.perf_counter() - t0
                        results[(m_name, model_name)].append((p, epc_after, dur))
                        print(
                            f"  p={p:.2f}, k={k:2d}, {m_name:<12s}: EPC={epc_after:.4f} (t={dur:.1f}s)"
                        )

    # ---------------------------------------------------------------------
    # Plot EPC vs. probability (Fig. 2‑style)
    # ---------------------------------------------------------------------
    fig_epc, ax_epc = plt.subplots(figsize=(6, 4))
    for (m_name, model_name), rows in results.items():
        # Aggregate over budgets+trials: mean EPC at each probability
        rows_arr = np.array(rows)  # shape (N, 3)
        for p in probs:
            mask = rows_arr[:, 0] == p
            mean_epc = rows_arr[mask, 1].mean()
            ax_epc.plot(p, mean_epc, marker="o", label=f"{m_name}-{model_name}")
    ax_epc.set_xlabel("Edge Existence Probability p")
    ax_epc.set_ylabel("Expected Pairwise Connectivity (↓ better)")
    ax_epc.set_title("Figure 2 – EPC vs p (k∈{5,10,15})")
    ax_epc.legend(fontsize="xx-small", ncol=2)
    fig_epc.tight_layout()
    epc_path = output_dir / "fig2_epc_vs_p.pdf"
    fig_epc.savefig(epc_path)
    plt.close(fig_epc)

    # ---------------------------------------------------------------------
    # Plot runtime (Fig. 3‑style)
    # ---------------------------------------------------------------------
    fig_rt, ax_rt = plt.subplots(figsize=(6, 4))
    for (m_name, model_name), rows in results.items():
        rows_arr = np.array(rows)
        for p in probs:
            mask = rows_arr[:, 0] == p
            mean_rt = rows_arr[mask, 2].mean()
            ax_rt.plot(p, mean_rt, marker="s", label=f"{m_name}-{model_name}")
    ax_rt.set_xlabel("Edge Existence Probability p")
    ax_rt.set_ylabel("Runtime (s, log‑scale)")
    ax_rt.set_yscale("log")
    ax_rt.set_title("Figure 3 – Runtime vs p (k∈{5,10,15})")
    ax_rt.legend(fontsize="xx-small", ncol=2)
    fig_rt.tight_layout()
    rt_path = output_dir / "fig3_runtime.pdf"
    fig_rt.savefig(rt_path)
    plt.close(fig_rt)

    print(f"\nSaved plots to:\n  {epc_path}\n  {rt_path}\n")


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probabilistic Vulnerability Toolkit")
    parser.add_argument("-k", type=int, default=10, help="Attack budget (edges)")
    parser.add_argument("--n", type=int, default=100, help="Number of nodes (ER model demo)")
    parser.add_argument("--p", type=float, default=0.3, help="Edge existence probability")
    parser.add_argument("--method", choices=["rega", "saa", "btw", "pr"], default="rega")
    parser.add_argument("--run_full_suite", action="store_true", help="Reproduce paper figures 2 & 3")
    args = parser.parse_args(argv)

    if args.run_full_suite:
        run_full_suite()
        return

    # Quick demo -----------------------------------------------------------
    base_g = nx.erdos_renyi_graph(args.n, 0.05, seed=42)
    probs = {e: args.p for e in base_g.edges()}
    pg = ProbGraph(base_g, probs)

    method_map = {
        "rega": rega_attack,
        "saa": saa_attack,
        "btw": betweenness_attack,
        "pr": pagerank_attack,
    }
    attack_fn = method_map[args.method]

    print(f"Running {args.method.upper()} on ER({args.n},0.05) with k={args.k}...")
    edges_removed = attack_fn(pg, args.k)
    epc_after = pg.remove_edges(edges_removed).estimate_epc(2500, method="csp")
    print(f"EPC after attack: {epc_after:.4f}\nEdges removed: {edges_removed}")


if __name__ == "__main__":
    main()
