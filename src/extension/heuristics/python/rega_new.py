# === Standard Library ===
import math
import random
import time
import heapq
import itertools
from collections import defaultdict, deque
from itertools import combinations
from typing import Any, Tuple, Dict, List, Set, Sequence, Union
from itertools import permutations, combinations
from scipy import sparse

# === Third-Party Libraries ===

# --- Scientific Computing ---
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

# --- Plotting ---
import matplotlib.pyplot as plt

# --- Parallel Processing ---
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Graph Processing ---
import networkx as nx

# --- JIT Compilation ---
from numba import njit, prange

def sigma_exact(
    G: nx.Graph,
    S: set,
    use_tqdm: bool = False
) -> int:
    from itertools import product
    edges = list(G.edges())
    total_conn = 0.0

    for state in product([0,1], repeat=len(edges)):
        p_state = 1
        Gp = nx.Graph()
        Gp.add_nodes_from(set(G.nodes())-S)

        for (e, keep) in zip(edges, state):
            p_edge = G.edges[e]['p']
            p_state *= (p_edge if keep else (1-p_edge))

            if keep and e[0] not in S and e[1] not in S:
                Gp.add_edge(*e)

        # count connected i<j pairs in Gp−S
        for i,j in combinations(set(G.nodes())-S, 2):
            if nx.has_path(Gp, i, j):
                total_conn += p_state

    return total_conn

def component_sampling_epc_mc(G, S, num_samples=10_000,
                              epsilon=None, delta=None, use_tqdm=False):
  """
  Theoretic bounds: compute N = N(epsilon, delta) by the theoretical bound.
  Experimentation:  Otherwise, use the N as input for sample count.
  """

  # Surviving vertex set and its size
  V_remaining = set(G.nodes()) - S
  n_rem = len(V_remaining)

  # base case
  if n_rem < 2:
    return 0.0

  if num_samples is None:
    assert epsilon is not None and delta is not None
    P_E = sum(G.edges[u, v]['p'] for u, v in G.edges())
    coeff = 4 * (math.e - 2) * math.log(2 / delta)
    num_samples = math.ceil(coeff * n_rem * (n_rem - 1) /
                            (epsilon ** 2 * P_E))

  C2 = 0
  it = tqdm(range(num_samples), desc='Component sampling',
            total=num_samples) if use_tqdm else range(num_samples)

  for _ in it:
    u = random.choice(tuple(V_remaining))

    # BFS based on edge probabilities

    visited = {u}
    queue = [u]

    while queue:

      v = queue.pop()
      for w in G.neighbors(v):

        # flip a coin biased by the edge probability
        # w not in deleted nodes
        if w in V_remaining and random.random() < G.edges[v, w]['p']:

          # if w is not visited
          if w not in visited:
              visited.add(w)
              queue.append(w)

    # component counting
    C2 += (len(visited) - 1)

  return (n_rem * C2) / (2 * num_samples)

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
            if np.random.random() >= probs[eid]:
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

def epc_mc_deleted(
  G: nx.Graph,
  S: set,
  num_samples: int = 100_000,
) -> float:
  # build csr once
  nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
  n = len(nodes)

  # turn python set S into a mask (node-IDs to delete)
  deleted = np.zeros(n, dtype=np.bool_)
  for u in S:
    deleted[idx_of[u]] = True

  epc = epc_mc(indptr, indices, probs, deleted, num_samples)

  return epc

def solve_lp_(G, pre_fixed, k):
    
    V = list(G.nodes())

    # x for all pairs
    AllPairs = [tuple(sorted((u, v))) for u,v in combinations(V, 2)]
    n, m = len(V), len(AllPairs)

    s_idx = {v: i for i,v in enumerate(V)}
    x_idx = {e: n + j for j,e in enumerate(AllPairs)}
    N = n + m

    # bounds
    bounds = [(0,1)]*N
    for v in pre_fixed:
      bounds[s_idx[v]] = (1,1)

    # objective: min sum(1 - x)
    c = np.zeros(N)
    for e in AllPairs:
      c[x_idx[e]] = -1.0

    # constraints
    A_ub, b_ub = [], []

    # budget
    row = np.zeros(N); row[:n]=1
    A_ub.append(row); b_ub.append(k)

    # edge upper bounds only for real edges
    for u,v in G.edges():
        
      u,v = min(u,v), max(u,v)
      p_uv = G.edges[u,v]['p']

      row = np.zeros(N)
      row[x_idx[(u,v)]] =  1
      row[s_idx[u]]     = -1
      row[s_idx[v]]     = -1

      A_ub.append(row); b_ub.append(1 - p_uv)

    # triangles only for each real edge (i,j) and every k
    for i,j in G.edges():
      i,j = min(i,j), max(i,j)
      for k in V:
        if k==i or k==j: 
          continue
        
        a = x_idx[(min(i,k),max(i,k))]
        b = x_idx[(min(i,j),max(i,j))]
        c_ = x_idx[(min(j,k),max(j,k))]
        
        row = np.zeros(N)
        row[a]  = +1
        row[b]  = -1
        row[c_] = -1
        A_ub.append(row); b_ub.append(0)

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")
    
    assert res.success, res.message

    s_vals = {v: res.x[s_idx[v]] for v in V}
    x_sum  = res.x[n:].sum()

    objective = len(AllPairs) - x_sum

    return s_vals, objective

def rega(G: nx.Graph,
        k: int,
        epc_func,
        num_samples: int = 100_000,
        # epsilon: float = None,
        # delta: float = None,
        use_tqdm: bool = False):
    """
    Full REGA pipeline: LP‐rounding + CSP‐refined local swaps.
    """

    # iterative rounding
    D = set()
    for _ in range(k):
      s_vals, _ = solve_lp_(G, pre_fixed=D, k=k)
      # pick the fractional s_i largest among V\D
      u = max((v for v in G.nodes() if v not in D),
              key=lambda v: s_vals[v])
      D.add(u)

    # local‐swap refinement
    current_epc = epc_func(G, D,
                           num_samples=num_samples,
                        #    epsilon=epsilon,
                        #    delta=delta,
                        #    use_tqdm=use_tqdm
                           )

    improved = True
    
    while improved:

        improved = False
        best_epc = current_epc
        best_swap = None

        for u in list(D):
            for v in G.nodes():

                if v in D: 
                    continue

                D_new = (D - {u}) | {v}

                epc_val = epc_func(G, D_new,
                                   num_samples=num_samples,
                                #    epsilon=epsilon,
                                #    delta=delta,
                                #    use_tqdm=use_tqdm
                                   )
                
                if epc_val < best_epc:
                    best_epc = epc_val
                    best_swap = (u, v)

        if best_swap is not None:

            u, v = best_swap
            D.remove(u)
            D.add(v)
            current_epc = best_epc
            improved = True

    return D

def solve_lp_relaxation_edges_only(G: nx.Graph, pre_fixed: set, k: int):
    V = list(G.nodes())
    # ─── build x only for real edges ─────────────────────────
    E = [tuple(sorted(e)) for e in G.edges()]
    n, m = len(V), len(E)

    s_idx = {v: i         for i, v in enumerate(V)}
    x_idx = {e: n + j     for j, e in enumerate(E)}
    N     = n + m

    # ─── variable bounds ────────────────────────────────────
    bounds = [(0,1)] * N
    for v in pre_fixed:
        bounds[s_idx[v]] = (1,1)

    # ─── objective: min Σ(1−x) ⇔ min −Σ x ──────────────────
    c = np.zeros(N)
    for e in E:
        c[x_idx[e]] = -1.0

    # ─── build A_ub, b_ub ───────────────────────────────────
    A_ub, b_ub = [], []

    # (a) budget: Σ s_i ≤ k
    row = np.zeros(N); row[:n] = 1
    A_ub.append(row); b_ub.append(k)

    # (b) edge upper bounds:  x_uv − s_u − s_v ≤ 1 − p_uv
    for (u,v) in E:
        p_uv = G.edges[u,v]["p"]
        row = np.zeros(N)
        row[x_idx[(u,v)]] =  1
        row[s_idx[u]]      = -1
        row[s_idx[v]]      = -1
        A_ub.append(row)
        b_ub.append(1 - p_uv)

    # (c) triangle cuts only when all three edges exist
    def has_x(a,b): return tuple(sorted((a,b))) in x_idx
    for i,j,k in permutations(V,3):
        if i<k:
            e_ij, e_jk, e_ik = tuple(sorted((i,j))), tuple(sorted((j,k))), tuple(sorted((i,k)))
            # only if every pair is a real edge
            if e_ij in x_idx and e_jk in x_idx and e_ik in x_idx:
                row = np.zeros(N)
                row[x_idx[e_ik]] =  1
                row[x_idx[e_ij]] = -1
                row[x_idx[e_jk]] = -1
                A_ub.append(row)
                b_ub.append(0)

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    # ─── solve ───────────────────────────────────────────────
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    # ─── unpack ──────────────────────────────────────────────
    s_vals = {v: res.x[s_idx[v]] for v in V}
    x_sum  = res.x[n:].sum()
    obj    = len(E) - x_sum  # Σ(1−x)

    return s_vals, obj

def rega_edges(G: nx.Graph,
        k: int,
        epc_func,
        num_samples: int = 100_000,
        # epsilon: float = None,
        # delta: float = None,
        use_tqdm: bool = False):
    """
    Full REGA pipeline: LP‐rounding + CSP‐refined local swaps.
    """

    # iterative rounding
    D = set()
    for _ in range(k):
      s_vals, _ = solve_lp_relaxation_edges_only(G, pre_fixed=D, k=k)
      # pick the fractional s_i largest among V\D
      u = max((v for v in G.nodes() if v not in D),
              key=lambda v: s_vals[v])
      D.add(u)

    # local‐swap refinement
    current_epc = epc_func(G, D,
                           num_samples=num_samples,
                        #    epsilon=epsilon,
                        #    delta=delta,
                        #    use_tqdm=use_tqdm
                           )

    improved = True
    
    while improved:

        improved = False
        best_epc = current_epc
        best_swap = None

        for u in list(D):
            for v in G.nodes():

                if v in D: 
                    continue

                D_new = (D - {u}) | {v}

                epc_val = epc_func(G, D_new,
                                   num_samples=num_samples,
                                #    epsilon=epsilon,
                                #    delta=delta,
                                #    use_tqdm=use_tqdm
                                   )
                
                if epc_val < best_epc:
                    best_epc = epc_val
                    best_swap = (u, v)

        if best_swap is not None:

            u, v = best_swap
            D.remove(u)
            D.add(v)
            current_epc = best_epc
            improved = True

    return D

def solve_lp_reaga_sparse(G: nx.Graph, pre_fixed: set, k: int):
    V = list(G.nodes())
    n = len(V)

    # 1) x_ij for every pair i<j
    Pairs = [tuple(sorted(e)) for e in combinations(V, 2)]
    m2 = len(Pairs)
    Nvar = n + m2

    s_idx = {v: i       for i, v in enumerate(V)}
    x_idx = {e: n + j   for j, e in enumerate(Pairs)}

    # 2) count rows: 1 budget + |E| edge‐bounds + |E|*(n-2) triangles
    E = [tuple(sorted(e)) for e in G.edges()]
    n_rows = 1 + len(E) + len(E) * (n - 2)

    # 3) allocate a sparse LIL matrix for easy row‐by‐row insertion
    A = sparse.lil_matrix((n_rows, Nvar), dtype=float)
    b = np.zeros(n_rows, dtype=float)
    row = 0

    # (a) budget: Σ s_i ≤ k
    A[row, :n] = 1
    b[row]     = k
    row += 1

    # (b) edge‐upper‐bounds: x_uv - s_u - s_v ≤ 1 - p_uv
    for (u, v) in E:
        puv = G.edges[u, v]['p']
        A[row, x_idx[(u, v)]] =  1
        A[row, s_idx[u]]      = -1
        A[row, s_idx[v]]      = -1
        b[row]                = 1 - puv
        row += 1

    # (c) triangles only for each real edge (i,j) and all k ≠ i,j
    for (i, j) in E:
        for k in V:
            if k == i or k == j:
                continue
            # x_ik - x_ij - x_jk ≤ 0
            A[row, x_idx[tuple(sorted((i, k)))]] =  1
            A[row, x_idx[(i, j)]]               = -1
            A[row, x_idx[tuple(sorted((j, k)))]] = -1
            b[row]                              = 0
            row += 1

    # 4) convert to CSR for linprog
    A_ub = A.tocsr()
    b_ub = b

    # 5) variable bounds
    bounds = [(0,1)] * Nvar
    for v in pre_fixed:
        bounds[s_idx[v]] = (1,1)

    # 6) objective: min Σ(1 - x) <=> min -Σ x
    c = np.zeros(Nvar)
    for e in Pairs:
        c[x_idx[e]] = -1

    # 7) call HiGHS
    res = linprog(c,
                  A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds,
                  method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    # 8) extract
    s_vals = {v: res.x[s_idx[v]] for v in V}
    x_sum  = res.x[n:].sum()
    obj    = len(Pairs) - x_sum

    return s_vals, obj

def rega_sparse(G: nx.Graph,
        k: int,
        epc_func,
        num_samples: int = 100_000,
        # epsilon: float = None,
        # delta: float = None,
        use_tqdm: bool = False):
    """
    Full REGA pipeline: LP‐rounding + CSP‐refined local swaps.
    """

    # iterative rounding
    D = set()
    for _ in range(k):
      s_vals, _ = solve_lp_reaga_sparse(G, pre_fixed=D, k=k)
      # pick the fractional s_i largest among V\D
      u = max((v for v in G.nodes() if v not in D),
              key=lambda v: s_vals[v])
      D.add(u)

    # local‐swap refinement
    current_epc = epc_func(G, D,
                           num_samples=num_samples,
                        #    epsilon=epsilon,
                        #    delta=delta,
                        #    use_tqdm=use_tqdm
                           )

    improved = True
    
    while improved:

        improved = False
        best_epc = current_epc
        best_swap = None

        for u in list(D):
            for v in G.nodes():

                if v in D: 
                    continue

                D_new = (D - {u}) | {v}

                epc_val = epc_func(G, D_new,
                                   num_samples=num_samples,
                                #    epsilon=epsilon,
                                #    delta=delta,
                                #    use_tqdm=use_tqdm
                                   )
                
                if epc_val < best_epc:
                    best_epc = epc_val
                    best_swap = (u, v)

        if best_swap is not None:

            u, v = best_swap
            D.remove(u)
            D.add(v)
            current_epc = best_epc
            improved = True

    return D

SEED = 42
N_SAMPLE = 100_000

K = 10
NODES = 100

# nodes 100, edges 200 (Sparse Graphs)
graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

# nodes 50, p = 0.5 (Dense Graphs) 2500 edges
# graph_models_dense = {
#   'ER': nx.erdos_renyi_graph(NODES, 0.5025, seed=SEED),
#   'BA': nx.barabasi_albert_graph(NODES,25,seed=SEED),
#   'SW': nx.watts_strogatz_graph(NODES, 25, 0.3, seed=SEED)
# }

# dist_funcs = {
#   'uniform': lambda: np.random.uniform(0.0, 1.0),
#   'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
#   'beta': lambda: np.random.beta(2, 5),
# }

records = []

for name_model, G in tqdm(
  graph_models.items(), 
  desc="Processing models", 
  total=len(graph_models)):
  
  for p in tqdm(np.arange(0.0, 1.1, 0.1), desc="Processing", total=int(1.1/0.1)):

    def fresh_graph():
      H = G.copy()
      for u, v in H.edges():
        H[u][v]['p'] = p
      return H
    
    # REGA
    t0 = time.perf_counter()

    rega_D = rega(
      fresh_graph(),
      k=K,
      epc_func=epc_mc_deleted,
      num_samples=N_SAMPLE,
      use_tqdm=False)
    
    rega_epc = epc_mc_deleted(fresh_graph(), rega_D, N_SAMPLE)
    t_rega = time.perf_counter() - t0

    # REGA edges
    # t0 = time.perf_counter()

    # rega_D_edges = rega_edges(
    #   fresh_graph(),
    #   k=K,
    #   epc_func=epc_mc_deleted,
    #   num_samples=N_SAMPLE,
    #   use_tqdm=False)
    
    # rega_epc_edges = epc_mc_deleted(fresh_graph(), rega_D_edges, N_SAMPLE)
    # t_rega_edges = time.perf_counter() - t0

    # REGA sparse
    # t0 = time.perf_counter()

    # rega_D_sparse = rega_sparse(
    #   fresh_graph(),
    #   k=K,
    #   epc_func=epc_mc_deleted,
    #   num_samples=N_SAMPLE,
    #   use_tqdm=False)
    
    # rega_epc_sparse = epc_mc_deleted(fresh_graph(), rega_D_sparse, N_SAMPLE)
    # t_rega_sparse = time.perf_counter() - t0

    # print(f"\nGreedy ES init: {initial_epc_greedy_es} vs {final_epc_greedy_es}\n")
    # print(f"\nGreedy MIS init: {mis_epc_initial} vs {mis_epc_final}\n")
    
    print(f"\n ~~~ model: {name_model} p: {p} --- name: REGA:\
           {rega_epc}")
    
    # print(f"\t edges: {rega_epc_edges} vs sparse: {rega_epc_sparse}~~~")

    for algo, t, epc, std in [
      # ('Greedy_ES_initial', t_greedy_es_initial, initial_epc_greedy_es, 0.0),
      # ('Greedy_ES_final', t_greedy_es_final, final_epc_greedy_es, 0.0),

      # ('Greedy_MIS_initial', t_greedy_mis_initial, mis_epc_initial, mis_epc_init_std),
      # ('Greedy_MIS_final', t_greedy_mis_final, mis_epc_final, mis_epc_final_std),

      ('REGA', t_rega, rega_epc, 0)
      # ('REGA_edges', t_rega, rega_epc, 0)
      # ('REGA_sparse', t_rega, rega_epc, 0)
    ]:
      
      records.append({
        'model': name_model,
        'p': p,
        'algo': algo,
        'time': t,
        'epc': epc,
        'epc_std': std,
      })

SAVE_ROOT_PATH = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/heuristics/results"

df = pd.DataFrame(records)
df.to_csv(f"{SAVE_ROOT_PATH}/csv/Result_REGA_{NODES}_{K}.csv", index=False)