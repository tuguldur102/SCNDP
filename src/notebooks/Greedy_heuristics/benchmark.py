import networkx as nx
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict, deque
import math, random
import heapq
import scipy.sparse as sp
from collections import deque
from joblib import Parallel, delayed
from numba import njit
from typing import Tuple, Dict, List, Set, Sequence, Union
import itertools

def remove_k_betweenness(G: nx.Graph, k: int) -> nx.Graph:
    bc = nx.betweenness_centrality(G)
    topk = sorted(bc, key=bc.get, reverse=True)[:k]
    H = G.copy()
    H.remove_nodes_from(topk)
    return H

def remove_k_pagerank_edges(G: nx.Graph, k: int) -> nx.Graph:
    L = nx.line_graph(G)
    pr = nx.pagerank(L)
    topk = sorted(pr, key=pr.get, reverse=True)[:k]
    H = G.copy()
    H.remove_edges_from(topk)
    return H

from pulp import (
    LpProblem, LpVariable, lpSum,
    LpBinary, LpMinimize, PULP_CBC_CMD, LpStatus
)
from numba import njit, prange

def connected_pairs(G_det):
    """ Σ |C|·(|C|−1)/2 over connected components of a deterministic graph. """
    total = 0
    for comp in nx.connected_components(G_det):
        s = len(comp)
        total += s * (s - 1) // 2
    return total


def nx_to_arrays(G):
    """
    Convert an undirected NetworkX graph with edge attribute 'p'
    into flat NumPy arrays (u, v, p) for numba kernels.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    u = np.empty(m, dtype=np.int32)
    v = np.empty(m, dtype=np.int32)
    p = np.empty(m, dtype=np.float32)
    for idx, (a, b, d) in enumerate(G.edges(data=True)):
        u[idx] = a
        v[idx] = b
        p[idx] = d['p']
    return n, u, v, p

@njit
def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@njit
def _uf_union(parent, size, a, b):
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra == rb:
        return
    # union by size (small → big)
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]


@njit(parallel=True)
def _epc_mc_numba(n, u, v, p, removed_mask, n_samples):
    """
    Parallel Monte-Carlo EPC estimate.
    removed_mask : boolean array length n (True ⇒ node deleted)
    """
    m = u.shape[0]
    epc_sum = 0.0

    for s in prange(n_samples):
        parent = np.arange(n, dtype=np.int32)
        size   = np.ones(n,  dtype=np.int32)

        for e in range(m):
            if np.random.rand() < p[e]:
                a, b = u[e], v[e]
                if removed_mask[a] or removed_mask[b]:
                    continue
                _uf_union(parent, size, a, b)

        seen = np.zeros(n, np.uint8)
        for i in range(n):
            if removed_mask[i]:
                continue
            r = _uf_find(parent, i)
            if seen[r]:
                continue
            seen[r] = 1
            k = size[r]
            epc_sum += k * (k - 1) / 2.0

    return epc_sum / n_samples


def component_sampling_epc_mc(G_prob, removed_set, n_samples=1000):
    """
    Fast drop-in EPC estimator (Algorithm 2 in the paper) using numba.
    """
    n, u, v, p = nx_to_arrays(G_prob)
    removed_mask = np.zeros(n, dtype=np.bool_)
    for node in removed_set:
        removed_mask[node] = True
    return _epc_mc_numba(n, u, v, p, removed_mask, n_samples)

def sample_scenarios(G_prob, T=30, rng=None):
    rng = rng or random.Random()
    scenarios = []
    for _ in range(T):
        H = nx.Graph()
        H.add_nodes_from(G_prob.nodes)
        for u, v, d in G_prob.edges(data=True):
            if rng.random() < d['p']:
                H.add_edge(u, v)
        scenarios.append(H)
    return scenarios

def solve_saa_mip_cbc(scenarios, k, msg=False):
    """
    Solve the SAA master problem with CBC and return initial set S₀ (|S₀|=k).
    """
    n = len(scenarios[0])
    T = len(scenarios)
    prob = LpProblem("k-pCND-SAA-CBC", LpMinimize)

    s = [LpVariable(f"s_{i}", cat=LpBinary) for i in range(n)]

    x = {}
    for l, H in enumerate(scenarios):
        for i, j in itertools.combinations(range(n), 2):
            x[(l, i, j)] = LpVariable(f"x_{l}_{i}_{j}", cat=LpBinary)
        for i in range(n):
            x[(l, i, i)] = 1

    prob += lpSum(s) == k

    for l, H in enumerate(scenarios):
        for i, j in itertools.combinations(range(n), 2):
            prob += x[(l, i, j)] <= 1 - s[i]
            prob += x[(l, i, j)] <= 1 - s[j]

        for i, j in itertools.combinations(range(n), 2):
            if not nx.has_path(H, i, j):
                prob += x[(l, i, j)] == 0

        for i, h, j in itertools.permutations(range(n), 3):
            ij = tuple(sorted((i, j)))
            ih = tuple(sorted((i, h)))
            hj = tuple(sorted((h, j)))
            prob += x[(l, ij[0], ij[1])] <= x[(l, ih[0], ih[1])]
            prob += x[(l, ij[0], ij[1])] <= x[(l, hj[0], hj[1])]

    prob += (1 / T) * lpSum(x.values())

    status = prob.solve(PULP_CBC_CMD(msg=msg))
    if LpStatus[status] != "Optimal":
        print("⚠ CBC ended with status:", LpStatus[status])

    S0 = {i for i in range(n) if s[i].value() > 0.5}
    return S0

def local_search(G_prob, S0, num_samples=1000, max_iter=1000, seed=None):
    rng = random.Random(seed)
    S = set(S0)
    best_val = component_sampling_epc_mc(G_prob, S, num_samples)
    for _ in range(max_iter):
        improved = False
        for u in list(S):
            # random permutation of outside nodes
            outside = rng.sample(list(set(G_prob.nodes) - S), len(G_prob) - len(S))
            for v in outside:
                candidate = (S - {u}) | {v}      # keeps size = k
                val = component_sampling_epc_mc(G_prob, candidate, num_samples)
                if val < best_val:
                    S, best_val = candidate, val
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S, best_val

def final_evaluation(G_prob, S, num_samples=100_000):
    return component_sampling_epc_mc(G_prob, S, num_samples)

def saa_algorithm(G_prob, k, T=30, ls_samples=1000, seed=None):
    """
    Full 4-phase SAA algorithm (CBC core + numba EPC).
    Returns
    -------
    S_best      : set[int]       # final deletion set (size = k)
    epc_est     : float          # EPC estimate after local search (ls_samples draws)
    epc_final   : float          # high-accuracy EPC (100 000 draws)
    """
    scenarios = sample_scenarios(G_prob, T, rng=random.Random(seed))

    S0 = solve_saa_mip_cbc(scenarios, k, msg=False)
    S_best, epc_est = local_search(G_prob, S0, ls_samples, seed=seed)

    epc_final = final_evaluation(G_prob, S_best)

    return S_best, epc_est, epc_final

def solve_lp_relaxation(G: nx.Graph, D: set, k: int) -> dict:
    nodes = list(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    m = len(edges)

    idx_s = {nodes[i]: i for i in range(n)}
    idx_z = {edges[j]: n + j for j in range(m)}

    bounds = [(0,1)] * (n + m)
    for u in D:
        bounds[idx_s[u]] = (1,1)

    A_eq = np.zeros((1, n + m))
    for u in nodes:
        A_eq[0, idx_s[u]] = 1
    b_eq = [k]

    A_ub, b_ub = [], []
    for (u, v) in edges:
        iu, iv = idx_s[u], idx_s[v]
        iz = idx_z[(u, v)]
        row = np.zeros(n + m); row[iu] = 1; row[iz] = 1
        A_ub.append(row); b_ub.append(1)
        row = np.zeros(n + m); row[iv] = 1; row[iz] = 1
        A_ub.append(row); b_ub.append(1)
        row = np.zeros(n + m); row[iz] = -1; row[iu] = -1; row[iv] = -1
        A_ub.append(row); b_ub.append(-1)
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)

    c = np.zeros(n + m)
    for j, (u, v) in enumerate(edges):
        c[n + j] = G.edges[u, v]['p']

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError("LP infeasible: " + res.message)
    return {nodes[i]: float(res.x[idx_s[nodes[i]]]) for i in range(n)}

def REGA_with_LP(G: nx.Graph, k: int, R: int, alpha: float) -> set:
    best_S, best_obj = None, float('inf')
    nodes = set(G.nodes())

    for _ in range(R):
        D = set()
        for _ in range(k):
            s_vals = solve_lp_relaxation(G, D, k)
            rem = list(nodes - D)
            sorted_nodes = sorted(rem, key=lambda u: s_vals[u], reverse=True)
            m = max(1, int(alpha * len(sorted_nodes)))
            D.add(random.choice(sorted_nodes[:m]))
        current_obj = component_sampling_epc_mc(G, D, 1000)
        improved = True
        while improved:
            improved = False
            for u in list(D):
                for v in nodes - D:
                    newS = (D - {u}) | {v}
                    val = component_sampling_epc_mc(G, newS, 1000)
                    if val < current_obj:
                        D, current_obj, improved = newS, val, True
                        break
                if improved:
                    break
        if current_obj < best_obj:
            best_S, best_obj = D.copy(), current_obj
    return best_S

import math
import heapq
from typing import Tuple, Dict, List, Set

import networkx as nx
import numpy as np
from numba import njit, prange

def nx_to_csr(G: nx.Graph) -> Tuple[List[int], Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
     """Convert an undirected NetworkX graph (edge attr `'p'`) to CSR arrays."""
     nodes: List[int] = list(G.nodes())
     idx_of: Dict[int, int] = {u: i for i, u in enumerate(nodes)}

     indptr: List[int] = [0]
     indices: List[int] = []
     probs: List[float] = []

     for u in nodes:
         for v in G.neighbors(u):
             indices.append(idx_of[v])
             probs.append(G.edges[u, v]['p'])
         indptr.append(len(indices))

     return (
         nodes,
         idx_of,
         np.asarray(indptr, dtype=np.int32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(probs, dtype=np.float32),
     )

@njit(inline="always")
def _bfs_component_size(start: int,
                    indptr: np.ndarray,
                    indices: np.ndarray,
                    probs: np.ndarray,
                    deleted: np.ndarray) -> int:
    """Return |C_u|−1 for **one** random realisation (stack BFS)."""
    n = deleted.size
    stack = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.uint8)

    size = 1
    top = 0
    stack[top] = start
    top += 1
    visited[start] = 1

    while top:
        top -= 1
        v = stack[top]
        for eid in range(indptr[v], indptr[v + 1]):
            w = indices[eid]
            if deleted[w]:
                continue
            if np.random.random() >= probs[eid]:  # edge absent
                continue
            if visited[w]:
                continue
            visited[w] = 1
            stack[top] = w
            top += 1
            size += 1
    return size - 1

@njit(parallel=True)
def epc_mc(indptr: np.ndarray,
            indices: np.ndarray,
            probs: np.ndarray,
            deleted: np.ndarray,
            num_samples: int) -> float:
    """Monte‑Carlo estimator of **expected pairwise connectivity** (EPC)."""
    surv = np.where(~deleted)[0]
    m = surv.size
    if m < 2:
        return 0.0

    acc = 0.0
    for _ in prange(num_samples):
        u = surv[np.random.randint(m)]
        acc += _bfs_component_size(u, indptr, indices, probs, deleted)

    return (m * acc) / (2.0 * num_samples)

from typing import Tuple, Dict, List, Set, Sequence, Union

def greedy_cndp_epc_celf(
    G: nx.Graph,
    K: int,
    *,
    num_samples: int = 20_000,
    reuse_csr: Tuple = None,
    return_trace: bool = False,
) -> Union[Set[int], Tuple[Set[int], List[float]]]:
    """Select **K** nodes that minimise EPC using CELF & Numba.

    Parameters
    ----------
    return_trace : bool, default *False*
        If *True*, also return a list `[σ(S₁), σ(S₂), …]` where `S_i` is the
        prefix after deleting *i* nodes.  Useful for plots.
    """

    # CSR cache --------------------------------------------------------
    if reuse_csr is None:
        nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
    else:
        nodes, idx_of, indptr, indices, probs = reuse_csr
    n = len(nodes)

    deleted = np.zeros(n, dtype=np.bool_)
    current_sigma = epc_mc(indptr, indices, probs, deleted, num_samples)

    pq: List[Tuple[float, int, int]] = []  # (-gain, v, last_round)
    gains = np.empty(n, dtype=np.float32)

    for v in range(n):
        deleted[v] = True
        gains[v] = current_sigma - epc_mc(indptr, indices, probs, deleted, num_samples)
        deleted[v] = False
        heapq.heappush(pq, (-gains[v], v, 0))

    S: Set[int] = set()
    trace: List[float] = []
    round_ = 0

    trace.append(current_sigma)

    while len(S) < K and pq:
        neg_gain, v, last = heapq.heappop(pq)
        if last == round_:
            # gain up‑to‑date → accept
            S.add(nodes[v])
            deleted[v] = True
            current_sigma += neg_gain  # add neg (= subtract gain)
            round_ += 1
            if return_trace:
                trace.append(current_sigma)
        else:
            # recompute gain lazily
            deleted[v] = True
            new_gain = current_sigma - epc_mc(indptr, indices, probs, deleted, num_samples)
            deleted[v] = False
            heapq.heappush(pq, (-new_gain, v, round_))

    return (S, trace) if return_trace else S

def optimise_epc(
     G: nx.Graph,
     K: int,
     *,
     num_samples: int = 20_000,
     return_trace: bool = False,
 ) -> Union[Set[int], Tuple[Set[int], List[float]]]:
     csr = nx_to_csr(G)
     return greedy_cndp_epc_celf(G, K, num_samples=num_samples, reuse_csr=csr, return_trace=return_trace)

def run_experiments(models, ps, k,
                    T_saa=30,
                    R_rega=20,
                    alpha_rega=0.1,
                    N_eval=100_000,
                    seed=0):

    rng_global = random.Random(seed)
    records = []

    for name, G0 in tqdm(models.items(), desc="Running experiments", total=len(models)):
        for p in tqdm(ps, desc=f"model={name}", total=len(ps)):
            def fresh_graph():
                H = G0.copy()
                for u, v in H.edges():
                    H[u][v]['p'] = p
                return H

            # Betweenness
            t0 = time.perf_counter()
            G_bc  = remove_k_betweenness(fresh_graph(), k)
            t_bc  = time.perf_counter() - t0
            epc_bc = component_sampling_epc_mc(G_bc, set(), N_eval)

            # PageRank
            t0 = time.perf_counter()
            G_pr  = remove_k_pagerank_edges(fresh_graph(), k)
            t_pr  = time.perf_counter() - t0
            epc_pr = component_sampling_epc_mc(G_pr, set(), N_eval)

            print("\nSAA\n")

            # SAA
            t0 = time.perf_counter()
            S_saa, _, _ = saa_algorithm(fresh_graph(), k, T_saa)
            t_saa = time.perf_counter() - t0
            epc_saa = component_sampling_epc_mc(fresh_graph(), S_saa, N_eval)

            print("\nREGA\n")

            # REGA
            t0 = time.perf_counter()
            S_rega = REGA_with_LP(fresh_graph(), k,
                                  R=R_rega,
                                  alpha=alpha_rega)
            t_rega = time.perf_counter() - t0
            epc_rega = component_sampling_epc_mc(fresh_graph(), S_rega, N_eval)

            print("\ngreedy_optimise\n")

            # Greedy (CELF + parallelism) + EPC
            t0 = time.perf_counter()
            S_grd, _ = optimise_epc(fresh_graph(), K=k, num_samples=N_eval, return_trace=True)
            t_grd = time.perf_counter() - t0
            epc_grd = component_sampling_epc_mc(fresh_graph(), S_grd, N_eval)

            print("\finished!!!\n")

            for algo, t, e in [
                ('Betweenness', t_bc, epc_bc),
                ('PageRank',    t_pr, epc_pr),
                ('SAA',         t_saa, epc_saa),
                ('REGA-LP',     t_rega, epc_rega),
                ('Greedy-EPC',  t_grd, epc_grd),
            ]:
                records.append({
                    'model': name,
                    'p':     p,
                    'algo':  algo,
                    'time':  t,
                    'epc':   e,
                })

    return pd.DataFrame(records)

models = {
    'ER': nx.erdos_renyi_graph(100, 0.045, seed=42),
    'BA': nx.barabasi_albert_graph(100, 2, seed=42),
    'WS': nx.watts_strogatz_graph(100, 4, 0.3, seed=42),
}

print(f"Erdos-Renyi graph #nodes: {models['ER'].number_of_nodes()} && #edges: {models['ER'].number_of_edges()}")
print(f"Barabasi-Albert graph #nodes: {models['BA'].number_of_nodes()} && #edges: {models['BA'].number_of_edges()}")
print(f"Watts-Strogatz graph #nodes: {models['WS'].number_of_nodes()} && #edges: {models['WS'].number_of_edges()}")

ps = np.arange(0.0, 1.2, 0.2)
k = 15
T_saa = 30
R_rega = 5
alpha_rega = 0.3
N_eval = 1_000

# Execute experiments
df = run_experiments(models, ps, k, T_saa, R_rega, alpha_rega, N_eval)

# Plot EPC vs p and Time vs p
for name in models:
    plt.figure()
    for algo in df.algo.unique():
        sub = df[(df.model == name) & (df.algo == algo)]
        plt.plot(sub.p, sub.epc, label=algo)
    plt.title(f"{name} — EPC vs p")
    plt.xlabel("p"); plt.ylabel("EPC"); plt.grid(True); plt.legend()
    plt.savefig(f"{name}_epc_vs_p.png")

    plt.figure()
    for algo in df.algo.unique():
        sub = df[(df.model == name) & (df.algo == algo)]
        plt.plot(sub.p, sub.time, label=algo)
    plt.title(f"{name} — Time vs p")
    plt.xlabel("p"); plt.ylabel("Time (s)"); plt.grid(True); plt.legend()
    plt.savefig(f"{name}_time_vs_p.png")
plt.show()