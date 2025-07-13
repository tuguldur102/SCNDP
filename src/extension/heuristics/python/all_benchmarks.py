# === Standard Library ===
import math
import random
import time
import heapq
import itertools
from collections import defaultdict, deque
from itertools import combinations
from typing import Any, Tuple, Dict, List, Set, Sequence, Union

# === Third-Party Libraries ===

# --- Scientific Computing ---
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse    import coo_matrix
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

# def local_search_swap(
#   # G: nx.Graph,
#   S: Set[int],
#   *,
#   csr: Tuple[List[int], Dict[int,int], np.ndarray, np.ndarray, np.ndarray],
#   num_samples: int = 100_000,
#   max_iter: int = 5
# ) -> Set[int]:
#   """
#   Given initial delete-set S, try 1-for-1 swaps to reduce EPC.
#   csr = (nodes, idx_of, indptr, indices, probs).
#   """
#   nodes, idx_of, indptr, indices, probs = csr
#   n = len(nodes)

#   # build boolean mask from S
#   deleted = np.zeros(n, dtype=np.bool_)
#   for u in S:
#     deleted[idx_of[u]] = True

#   # current EPC
#   curr = epc_mc(indptr, indices, probs, deleted, num_samples)

#   for it in range(max_iter):
#     best_delta = 0.0
#     best_swap = None

#     # try swapping each i in S with each j not in S
#     for i in list(S):
#       ii = idx_of[i]
#       # undelete i
#       deleted[ii] = False

#       for j in nodes:
#         jj = idx_of[j]
#         if deleted[jj]:
#           continue
#         # delete j
#         deleted[jj] = True

#         sigma = epc_mc(indptr, indices, probs, deleted, num_samples)
#         delta = curr - sigma
#         if delta > best_delta:
#           best_delta = delta
#           best_swap = (ii, jj, sigma)

#         # revert j
#         deleted[jj] = False

#       # revert i

#       deleted[ii] = True

#     if best_swap is None:
#       break   

#     # commit the best swap
#     ii, jj, new_sigma = best_swap
#     deleted[ii] = False
#     deleted[jj] = True
#     curr = new_sigma

#     # update S
#     S.remove(nodes[ii])
#     S.add(nodes[jj])

#   return S

def local_search_(
  G: nx.Graph,
  S_init: set,
  num_samples: int = 10_000
):
  """1-swap local search"""

  S = S_init.copy()
  nodes_not_in_set = set(G.nodes()) - S

  current_epc = epc_mc_deleted(G, S, num_samples)

  improved = True
  while improved:
    improved = False
    best_swap = None

    for u in list(S):
      for v in nodes_not_in_set:        
        
        D_new = (S - {u}) | {v}

        temp_epc = epc_mc_deleted(G, D_new, num_samples)

        if temp_epc < current_epc:
            current_epc = temp_epc
            best_swap = (u, v)
            improved = True

    if improved and best_swap:
      u, v = best_swap

      S.remove(u)
      S.add(v)
      nodes_not_in_set.remove(v)
      nodes_not_in_set.add(u)
  
  return S

def greedy_es_local_opt(
    G, K,
    num_samples=10_000, local_iter=1
):
  t0 = time.perf_counter()

  greedy_es_S = optimise_epc(
    G=G.copy(), K=K, num_samples=num_samples)
  
  t_greedy_es = time.perf_counter() - t0

  # initial greedy es with no ls
  epc_greedy_es = epc_mc_deleted(G.copy(), greedy_es_S, 100_000)
  # epc_greedy_es = sigma_delta[-1]

  # csr = nx_to_csr(G)

  S_opt = local_search_(G, greedy_es_S, num_samples)

  # S_opt = local_search_swap(
  #   greedy_es_S, csr=csr, 
  #   num_samples=num_samples, max_iter=local_iter)

  # epc_final = epc_mc_deleted(G, S_opt, num_samples)

  return t_greedy_es, epc_greedy_es, S_opt,

def greedy_cndp_epc(
    G: nx.Graph,
    K: int,
    num_samples: int = 10000,
    exact: bool = False,
    use_tqdm: bool = False
) -> set:
  """
  Algorithm 2 from the paper: Greedy selection of S |S| <= K
  to minimize sigma(S) via sigma_monte_carlo().

  Returns the list S (in pick order).
  """

  # S <= {Empty set} init
  S = set()

  Sigma_delta = []
  # Current sigma(S) for the empty set
  sigma_S = 0
  if exact:
    sigma_S = sigma_exact(G, S)
  else:
    sigma_S = component_sampling_epc_mc(G, S, num_samples=num_samples)

  Sigma_delta.append(sigma_S)
  # print(f"Initial sigma(S): {sigma_S}")

  if use_tqdm:
    it = tqdm(range(K), desc='Greedy selection', total=K)
  else:
    it = range(K)

  # Greedily select K nodes
  for _ in it:
    # inits
    best_j = None
    best_gain = -float('inf')
    best_sigma = None

    # find v maximizing gain sigma(S) - sigma(S ∪ j)
    for j in G:
      # Skip if j is already in S to avoid redundant calculations
      # j ∈ S
      if j in S:
        continue

      # S ∪ j = S + {j}
      if exact:
        sigma_Sj = sigma_exact(G, S | {j})
      else:
        sigma_Sj = component_sampling_epc_mc(G, S | {j}, num_samples=num_samples)

      gain = sigma_S - sigma_Sj

      # j <= argmax_{j ∈ V\S} (sigma(S) - sigma(S ∪ j))

      if gain > best_gain:
        best_gain = gain
        best_j = j
        best_sigma = sigma_Sj


    # add the best node
    if best_j is None:
      break

    S.add(best_j)
    sigma_S = best_sigma

    Sigma_delta.append(best_sigma)
    # print(f"Selected node {best_j}, gain: {best_gain}, new sigma(S): {sigma_S}")

  return S, Sigma_delta

def remove_k_betweenness(G: nx.Graph, k: int) -> nx.Graph:
  bc = nx.betweenness_centrality(G)
  topk = sorted(bc, key=bc.get, reverse=True)[:k]
  # H = G.copy()
  # H.remove_nodes_from(topk)
  return topk

def remove_k_pagerank_nodes(
    G: nx.Graph,
    k: int,
    *,
    pagerank_kwargs: Dict[str, Any] | None = None,
) -> nx.Graph:
    """
    Return a copy of *G* after deleting the k nodes with the
    highest PageRank scores.
    """
    pagerank_kwargs = {} if pagerank_kwargs is None else dict(pagerank_kwargs)

    # Compute PR on the node set
    pr = nx.pagerank(G, **pagerank_kwargs)

    # Pick the k nodes with largest score
    topk = sorted(pr, key=pr.get, reverse=True)[:k]

    # Remove and return a fresh graph
    # H = G.copy()
    # H.remove_nodes_from(topk)
    return topk

def remove_k_degree_centrality(G: nx.Graph, k: int) -> nx.Graph:
    """
    Remove the k nodes with highest *degree centrality*, 
    """
    # {node: centrality}
    dc = nx.degree_centrality(G)             
    topk = sorted(dc, key=dc.get, reverse=True)[:k]
    # H = G.copy()
    # H.remove_nodes_from(topk)
    return topk

def greedy_epc_mis(G, k, num_samples):

  # Maximal independent set
  MIS = nx.maximal_independent_set(G)
  R = set(MIS)
  target = len(G) - k
  V = G.number_of_nodes()

  sigma_delta = []

  # print(f"#MIS: {len(R)}")

  # Greedy grow R set until |R| = |V| - k
  while len(R) < target:
    best_j, best_sigma = None, float('inf')
    for j in G.nodes():
      if j in R:
        continue

      # delete node
      S_j = set(G.nodes()) - (R | {j})
      sigma = component_sampling_epc_mc(G, S=S_j, num_samples=num_samples)

      if sigma < best_sigma:
        best_sigma, best_j = sigma, j

        sigma_delta.append(best_sigma)

    R.add(best_j)
  
  D = set(G.nodes()) - R
  return D, sigma_delta

@njit
def greedy_epc_mis_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    probs: np.ndarray,
    deleted: np.ndarray,
    n: int,
    k: int,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    target_survivors = n - k
    survivors = n - np.sum(deleted)

    flips_needed = target_survivors - survivors
    if flips_needed < 0:
        flips_needed = 0

    max_steps = 1 + flips_needed
    trace_np = np.empty(max_steps, dtype=np.float64)

    step = 0
    curr_epc = epc_mc(indptr, indices, probs, deleted, num_samples)
    trace_np[step] = curr_epc
    step += 1

    while survivors < target_survivors:
        best_sigma = 1e18
        best_j = -1

        for j in range(n):
            if deleted[j]:
                deleted[j] = False
                sigma = epc_mc(indptr, indices, probs, deleted, num_samples)
                deleted[j] = True
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_j = j

        deleted[best_j] = False
        survivors += 1
        curr_epc = best_sigma

        trace_np[step] = curr_epc
        step += 1
    return deleted, trace_np, step

def greedy_mis_optimized(
  G: nx.Graph,
  k: int,
  *,
  num_samples: int = 100_000,
  return_trace: bool = False,
) -> Union[Set[int], Tuple[Set[int], list]]:
    
    # CSR conversion
    nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
    n = len(nodes)

    # Build initial MIS mask
    # MIS = nx.maximal_independent_set(G)
    # deleted = np.ones(n, dtype=np.bool_)
    # for u in MIS:
    #     deleted[idx_of[u]] = False

    # best_deleted: np.ndarray = None
    # best_sigma = float("inf")

    # for i in range(mis_rounds):
    #     MIS = nx.maximal_independent_set(G)

    #     # deleted[i]==True means node i is removed
    #     deleted = np.ones(n, dtype=np.bool_)
        
    #     for u in MIS:
    #         deleted[idx_of[u]] = False
        
    #     curr_sigma = epc_mc(indptr, indices, probs, deleted, num_samples)

    #     # print(f"{i}-th rount sigma: {curr_sigma}")

    #     if curr_sigma < best_sigma:
    #         best_sigma = curr_sigma
    #         best_deleted = deleted.copy()

    MIS = nx.algorithms.approximation.maximum_independent_set(G)

    deleted = np.ones(n, dtype=np.bool_)
    for u in MIS:
        deleted[idx_of[u]] = False

    best_deleted = deleted.copy()

    # Call the fast Numba core
    final_deleted, trace_np, cnt = greedy_epc_mis_numba(
        indptr, indices, probs, 
        best_deleted, 
        n, k, num_samples
    )

    # Slice out only the filled portion of the trace
    trace = trace_np[:cnt].tolist()

    # Map mask back to node-IDs
    D = {nodes[i] for i in range(n) if final_deleted[i]}

    return (D, trace) if return_trace else D

def robust_greedy_mis_optimized(
  G, k, 
  num_samples=10_000, trials=5, 
  max_iter=1):
  # best_D, best_epc = None, float('inf')
  csr = nx_to_csr(G)
  epcs_initial = []
  epcs = []
  time_initial = []

  for _ in range(trials):
    
    t0 = time.perf_counter()

    S = greedy_mis_optimized(
      G, k,
      num_samples=num_samples,
      return_trace=False)
    
    epc_initial = epc_mc_deleted(G, S, 100_000)

    epcs_initial.append(epc_initial)

    t_greedy_mis_initial = time.perf_counter() - t0

    time_initial.append(t_greedy_mis_initial)

    S_opt = local_search_(G, S, num_samples)

    # S_opt = local_search_swap(
    #   S, csr=csr, num_samples=num_samples, max_iter=max_iter)
    
    epc_final = epc_mc_deleted(G, S_opt, 100_000)

    epcs.append(epc_final)

    # epc_final = component_sampling_epc_mc(G, S_opt, num_samples)
    # epcs.append(epc_final)

  mean_epc_initial = sum(epcs_initial) / trials
  std_epc_initial = (sum((e - mean_epc_initial)**2 for e in epcs_initial) / trials)**0.5

  mean_epc = sum(epcs) / trials
  std_epc = (sum((e - mean_epc)**2 for e in epcs) / trials)**0.5

  time_initial_final = sum(time_initial) / trials

  return time_initial_final, mean_epc_initial, std_epc_initial, mean_epc, std_epc
 
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

def solve_lp_reaga_sparse(G: nx.Graph, pre_fixed: set, k: int):
    V = list(G.nodes())
    n = len(V)

    # variables: s_i  (i = 0…n-1)      x_ij (j = 0…m2-1)
    Pairs = [tuple(sorted(e)) for e in combinations(V, 2)]
    m2    = len(Pairs)
    Nvar  = n + m2
    s_idx = {v: i         for i, v in enumerate(V)}
    x_idx = {e: n + j     for j, e in enumerate(Pairs)}

    
    rows, cols, data = [], [], []
    rhs              = []

    def add_coef(r, c, val):
        rows.append(r); cols.append(c); data.append(val)

    r = 0 

    # budget 
    for i in range(n):
        add_coef(r, i, 1.0)
    rhs.append(k); r += 1

    # edge upper bounds  x_uv − s_u − s_v ≤ 1 − p_uv
    for (u, v) in G.edges():
        u, v   = sorted((u, v))
        puv    = G.edges[u, v]['p']
        add_coef(r, x_idx[(u, v)],  1.0)
        add_coef(r, s_idx[u],      -1.0)
        add_coef(r, s_idx[v],      -1.0)
        rhs.append(1 - puv); r += 1

    # triangle cuts for each real edge (i,j) and every
    for (i, j) in G.edges():
        i, j = sorted((i, j))
        for k_ in V:
            if k_ == i or k_ == j:
                continue
            add_coef(r, x_idx[tuple(sorted((i, k_)))],  1.0)  
            add_coef(r, x_idx[(i, j)]               , -1.0)  
            add_coef(r, x_idx[tuple(sorted((j, k_)))], -1.0)   
            rhs.append(0.0); r += 1

    n_rows = r
    A_ub   = coo_matrix((data, (rows, cols)), shape=(n_rows, Nvar)).tocsr()
    b_ub   = np.asarray(rhs)

    # bounds 
    bounds = [(0.0, 1.0)] * Nvar
    for v in pre_fixed:
        bounds[s_idx[v]] = (1.0, 1.0)

    #  objective 
    c = np.zeros(Nvar)
    for e in Pairs:
        c[x_idx[e]] = -1.0

    # 
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP infeasible: " + res.message)

    #
    s_vals = {v: res.x[s_idx[v]] for v in V}
    x_sum  = res.x[n:].sum()
    obj    = len(Pairs) - x_sum
    return s_vals, obj

def rega(G: nx.Graph,
        k: int,
        epc_func,
        num_samples: int = 100_000,
        max_iter: int = 1,
        # epsilon: float = None,
        # delta: float = None,
        use_tqdm: bool = False):
    """
    Full REGA pipeline: LP‐rounding + CSP‐refined local swaps.
    """

    csr = nx_to_csr(G)

    # iterative rounding
    D = set()
    for _ in range(k):
      # s_vals, _ = solve_lp_(G, pre_fixed=D, k=k)
      s_vals, _ = solve_lp_reaga_sparse(G, pre_fixed=D, k=k)

      # pick the fractional s_i largest among V\D
      u = max((v for v in G.nodes() if v not in D),
              key=lambda v: s_vals[v])
      D.add(u)

    # local‐swap refinement

    S_opt = local_search_(G, D, num_samples)
    
    # S_opt = local_search_(G, greedy_es_S, num_samples)

    # S_opt = local_search_swap(
    #   D, csr=csr, num_samples=num_samples, max_iter=max_iter)
    
    # improved = True
    
    # while improved:

    #     improved = False
    #     best_epc = current_epc
    #     best_swap = None

    #     for u in list(D):
    #         for v in G.nodes():

    #             if v in D: 
    #                 continue

    #             D_new = (D - {u}) | {v}

    #             epc_val = epc_func(G, D_new,
    #                                num_samples=num_samples,
    #                             #    epsilon=epsilon,
    #                             #    delta=delta,
    #                             #    use_tqdm=use_tqdm
    #                                )
                
    #             if epc_val < best_epc:
    #                 best_epc = epc_val
    #                 best_swap = (u, v)

    #     if best_swap is not None:

    #         u, v = best_swap
    #         D.remove(u)
    #         D.add(v)
    #         current_epc = best_epc
    #         improved = True

    return S_opt

def build_csr(G: nx.Graph):

    nodes = sorted(G.nodes())
    idx_of = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    degs = [len(list(G.neighbors(u))) for u in nodes]
    indptr = np.zeros(n + 1, dtype=int)
    indptr[1:] = np.cumsum(degs)

    indices = np.empty(indptr[-1], dtype=int)
    probs = np.empty(indptr[-1], dtype=float)
    ptr = 0

    for u in nodes:
        for v in G.neighbors(u):
            indices[ptr] = idx_of[v]
            probs[ptr] = G.edges[u, v]['p']
            ptr += 1
            
    return nodes, idx_of, indptr, indices, probs

def grasp_cndp(G: nx.Graph,
               K: int,
               alpha: float = 0.1,
               num_samples: int = 1000,
               restarts: int = 1,
               use_tqdm: bool = False):
    """
    GRASP for Stochastic CNDP:
    """
    best_S, best_score = None, float('inf')

    if use_tqdm:
        it = tqdm(range(restarts), desc="Processing GRASP", total=restarts)
    else:
        it = range(restarts)

    for _ in it:
        S = set()
        # precompute sigma(empty)
        sigma_S = epc_mc_deleted(G, S, num_samples)

        for k in range(K):
            # compute improvement d_j = sigma(S) – sigma(S ∪ {j})
            improvements = {}
            for j in G.nodes():
                if j in S: 
                    continue
                sigma_Sj = epc_mc_deleted(G, S | {j}, num_samples)
                improvements[j] = sigma_S - sigma_Sj

            # find best and worst d
            max_imp = max(improvements.values())
            min_imp = min(improvements.values())

            # build RCL = { j : d_j >= max_imp – alpha*(max_imp – min_imp) }
            threshold = max_imp - alpha * (max_imp - min_imp)
            RCL = [j for j, d in improvements.items() if d >= threshold]

            # pick one at random from RCL
            v = random.choice(RCL)
            S.add(v)

            # update sigma(S)
            sigma_S = epc_mc_deleted(G, S, num_samples)

        if sigma_S < best_score:
            best_score = sigma_S
            best_S = S.copy()

    return best_S, best_score

def grasp_with_local_search_outside(
    G: nx.Graph,
    K: int,
    alpha: float = 0.2,
    mc_samples_grasp: int = 10000,
    mc_samples_ls: int = 10000,
    restarts: int = 30,
    max_ls_iter: int = 1,
    use_tqdm: bool = False
) -> Tuple[Set[int], float]:
    """
    Combined GRASP + local_search_swap procedure.
    """

    csr = build_csr(G)
    # best_inner_S, best_inner_score = set(), float('inf')
    best_S, best_score = set(), float('inf')

    S_grasp, epc_grasp = grasp_cndp(
        G, K, num_samples=mc_samples_grasp, 
        alpha=alpha, restarts=restarts, use_tqdm=False)
    
    # print(f"\nGrashp EPC: {epc_grasp}\n")
    # score_inner = epc_mc_deleted(
    #     G, S_grasp, 
    #     num_samples=mc_samples_grasp)

    S_opt = local_search_(G, S_grasp, mc_samples_ls)

    # S_last = local_search_swap(
    #     S_grasp, csr=csr, num_samples=mc_samples_ls, 
    #     max_iter=max_ls_iter)
    
    score_last = epc_mc_deleted(
            G, S_opt, 
            num_samples=mc_samples_grasp)

    return S_opt, score_last

class ReactiveAlpha:
    def __init__(self, alpha_vals: List[float]):
        self.alpha_vals = alpha_vals
        self.weights = [1.0] * len(alpha_vals)

    def sample(self) -> Tuple[int, float]:
        total = sum(self.weights)
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return i, self.alpha_vals[i]
        return len(self.weights)-1, self.alpha_vals[-1]

    def reward(self, idx: int, amount: float = 1.0):
        self.weights[idx] += amount

    def penalize(self, idx: int, factor: float = 0.99):
        self.weights[idx] *= factor

def grasp_construct(G: nx.Graph,
                    K: int,
                    alpha: float,
                    mc_samples: int) -> Tuple[Set[int], float]:
    """One GRASP construction (no restarts)."""
    S: Set[int] = set()
    cache: Dict[frozenset, float] = {}

    def sigma(SetS: Set[int]) -> float:
        key = frozenset(SetS)
        if key not in cache:
            cache[key] = epc_mc_deleted(G, SetS, num_samples=mc_samples)
        return cache[key]

    sigma_S = sigma(S)
    for _ in range(K):
        gains = {}
        for v in G.nodes():
            if v in S:
                continue
            gains[v] = sigma_S - sigma(S | {v})
        d_max, d_min = max(gains.values()), min(gains.values())
        thresh = d_max - alpha * (d_max - d_min)
        RCL = [v for v, d in gains.items() if d >= thresh]
        choice = random.choice(RCL)
        S.add(choice)
        sigma_S = sigma(S)

    return S, sigma_S

def path_relink(S: Set[int], E: Set[int],
                G: nx.Graph,
                mc_samples: int) -> Tuple[Set[int], float]:
    """Greedy walk from S toward E, returning best intermediate."""

    T = S.copy()
    best_T, best_score = T.copy(), epc_mc_deleted(G, T, mc_samples)
    D_add = list(E - T)
    D_rm  = list(T - E)

    while D_add and D_rm:
        best_move = None
        best_delta = 0.0
        for i in D_rm:
            for j in D_add:
                T_candidate = T.copy()
                T_candidate.remove(i)
                T_candidate.add(j)
                score = epc_mc_deleted(G, T_candidate, mc_samples)
                delta = best_score - score
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, score)
        if best_move is None:
            break
        i, j, new_score = best_move
        T.remove(i); T.add(j)
        D_rm.remove(i); D_add.remove(j)
        if new_score < best_score:
            best_score = new_score
            best_T = T.copy()

    return best_T, best_score

def insert_into_elite(elite: List[Tuple[Set[int], float]],
                      candidate: Tuple[Set[int], float],
                      max_size: int = 10):
    elite.append(candidate)
    elite.sort(key=lambda x: x[1])
    if len(elite) > max_size:
        elite.pop()

def grasp_meta(
    G: nx.Graph,
    K: int,
    restarts: int = 30,
    mc_samples_grasp: int = 10_000,
    mc_samples_final: int = 10_000,
    mc_samples_ls: int = 10_000,
    max_ls_iter: int = 1,
    elite_size: int = 5,
) -> Tuple[Set[int], float]:
    reactive = ReactiveAlpha([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
    elite: List[Tuple[Set[int], float]] = []

    # construction only loop
    for _ in range(restarts):
        idx_alpha, alpha = reactive.sample()

        # build a solution with greedy random adaptive construction
        S_raw, _ = grasp_construct(G, K, alpha, mc_samples_grasp)

        # evaluate the raw solution with a medium sample budget
        score_raw = epc_mc_deleted(G, S_raw, num_samples=mc_samples_final)

        # maintain the elite list
        insert_into_elite(elite, (S_raw, score_raw), max_size=elite_size)

        # reward or penalise that alpha using the raw score
        best_elite_score = elite[0][1]
        
        if score_raw <= best_elite_score:
            reactive.reward(idx_alpha)
        else:
            reactive.penalize(idx_alpha)

    # take the best raw set and polish once
    best_raw_S, _ = elite[0]

    # csr = build_csr(G)

    S_final = local_search_(G, best_raw_S, mc_samples_ls)

    # S_final = local_search_swap(
    #     best_raw_S,
    #     csr=csr,
    #     num_samples=mc_samples_ls,
    #     max_iter=max_ls_iter
    # )

    # final high quality evaluation
    final_score = epc_mc_deleted(G, S_final, num_samples=100_000)

    return S_final, final_score

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000
LOCAL_SEARCH_ITER = 1
GRASP_RESTARTS = 3

# K = 10
# NODES = 100

# nodes 100, edges 200 (Sparse Graphs)
# graph_models = {
#   'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
#   'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
#   'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
# }

DENSE_NODES = 50 
K = 5
# nodes 50, p = 0.5 (Dense Graphs) 2500 edges

graph_models_dense = {
  'ER': nx.erdos_renyi_graph(DENSE_NODES, 0.5025, seed=SEED),
  'BA': nx.barabasi_albert_graph(DENSE_NODES, 25,seed=SEED),
  'SW': nx.watts_strogatz_graph(DENSE_NODES, 25, 0.3, seed=SEED)
}

dist_funcs = {
  'uniform': lambda: np.random.uniform(0.0, 1.0),
  'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
  'beta': lambda: np.random.beta(2, 5),
}


for name_model, G in tqdm(
  graph_models_dense.items(), 
  desc="Processing models", 
  total=len(graph_models_dense)):

  records = []

  for p in tqdm(np.arange(0.0, 1.1, 0.1), desc="Processing", total=int(1.1/0.1)):

    def fresh_graph():
      H = G.copy()
      for u, v in H.edges():
        H[u][v]['p'] = p
      return H
  
  # dist functions
  # for name_dist, dist_func in tqdm(
  #   dist_funcs.items(),
  #   desc="Processing",
  #   total=len(dist_funcs)):

  #   def fresh_graph():
  #     """Fresh graph with new edge weights."""
  #     H = G.copy()
  #     for u, v in H.edges():
  #       H[u][v]['p'] = dist_func()
  #     return H


    # Heuristics 1: Degree-Based Centrality
    t0 = time.perf_counter()
    degree_S  = remove_k_degree_centrality(fresh_graph(), K)

    # print(set(degree_S))
    epc_degree = epc_mc_deleted(fresh_graph(), set(degree_S), N_SAMPLE_EVAL)
    t_degree  = time.perf_counter() - t0


    # Heuristics 2: Betweenness
    t0 = time.perf_counter()
    between_S  = remove_k_betweenness(fresh_graph(), K)

    epc_between = epc_mc_deleted(fresh_graph(), set(between_S), N_SAMPLE_EVAL)

    t_between  = time.perf_counter() - t0

    # Heuristics 3: PageRank node
    t0 = time.perf_counter()
    pagerank_S  = remove_k_pagerank_nodes(fresh_graph(), K)

    epc_pagerank = epc_mc_deleted(fresh_graph(), set(pagerank_S), N_SAMPLE_EVAL)

    t_pagerank  = time.perf_counter() - t0

    # heuristics 4: Greedy ES optimized
    t0 = time.perf_counter()

    t_greedy_es, epc_greedy_es, greedy_es_S_ls = greedy_es_local_opt(
      fresh_graph(), K, num_samples=N_SAMPLE_LS,
      local_iter=LOCAL_SEARCH_ITER)
    
    epc_greedy_es_ls = epc_mc_deleted(fresh_graph(), greedy_es_S_ls, N_SAMPLE_EVAL)
    
    t_greedy_es_final = time.perf_counter() - t0

    # heuristics 5: Greedy MIS optimized
    t0 = time.perf_counter()

    t_greedy_mis_initial, mis_epc_initial, mis_epc_init_std, mis_epc_final, mis_epc_final_std = robust_greedy_mis_optimized(
      fresh_graph(), K, num_samples=N_SAMPLE_LS,
      trials=10, max_iter=LOCAL_SEARCH_ITER)
    
    t_greedy_mis_final = time.perf_counter() - t0

    # heuristics 6: REGA
    t0 = time.perf_counter()

    rega_D = rega(
      fresh_graph(),
      k=K,
      epc_func=epc_mc_deleted,
      num_samples=N_SAMPLE_LS,
      max_iter=LOCAL_SEARCH_ITER,
      use_tqdm=False)
    
    rega_epc = epc_mc_deleted(fresh_graph(), rega_D, N_SAMPLE_EVAL)
    t_rega = time.perf_counter() - t0

    # print(f"\n ---- REGA!!! ---- \n")
    # heuristics 7: Grasp + ls
    t0 = time.perf_counter()

    S_star_outside, epc_outside = grasp_with_local_search_outside(
          fresh_graph(), K=K,
          alpha=0.05,
          mc_samples_grasp=N_SAMPLE_LS,
          mc_samples_ls=N_SAMPLE_LS,
          restarts=GRASP_RESTARTS,
          max_ls_iter=LOCAL_SEARCH_ITER
          )
    
    epc_grasp_ls = epc_mc_deleted(fresh_graph(), S_star_outside, N_SAMPLE_EVAL)

    t_grasp_ls = time.perf_counter() - t0

    # print(f"\n ---- GRASP!!! ---- \n")

    # heuristics 8: Grasp + ls + path relinking + reactive alpha
    t0 = time.perf_counter()

    grasp_opt_S, grasp_opt_epc = grasp_meta(
      fresh_graph(), K, 
      restarts=GRASP_RESTARTS, mc_samples_grasp = N_SAMPLE_LS, 
      mc_samples_final=N_SAMPLE_LS, mc_samples_ls=N_SAMPLE_LS,
      max_ls_iter=LOCAL_SEARCH_ITER, elite_size=5)
    
    t_grasp_opt = time.perf_counter() - t0

    # print(f"\n ---- grasp ls!!! ---- \n")

    # print(f"\nGreedy ES init: {epc_greedy_es} vs {epc_greedy_es_ls}\n")
    # print(f"\nGreedy MIS init: {mis_epc_initial} vs {mis_epc_final}\n")

    for algo, t, epc, std in [
      ('Degree-based', t_degree, epc_degree, 0.0),
      ('Betweenness', t_between, epc_between, 0.0),
      ('PageRank', t_pagerank, epc_pagerank, 0.0),

      ('Greedy_ES_initial', t_greedy_es, epc_greedy_es, 0.0),
      ('Greedy_ES_final', t_greedy_es_final, epc_greedy_es_ls, 0.0),
      # ('Greedy_MIS_initial', t_greedy_mis_initial, mis_epc_initial, mis_epc_init_std),
      # ('Greedy_MIS_final', t_greedy_mis_final, mis_epc_final, mis_epc_final_std),

      ('REGA', t_rega, rega_epc, 0.0),
      ('grasp', t_grasp_ls, epc_grasp_ls, 0.0),
      ('grasp_path_relink', t_grasp_opt, grasp_opt_epc, 0.0),
    ]:
      
      records.append({
        'model': name_model,
        'p': p,
        'algo': algo,
        'time': t,
        'epc': epc,
        'epc_std': std,
      })

    # SAVE_PATH_ROOT = r"C:\Users\btugu\Documents\develop\research\SCNDP\src\extension\heuristics\results"

    # df = pd.DataFrame(records)
    # df.to_csv(f"{SAVE_PATH_ROOT}/csv/sparse/Result_heuristics_{name_model}_{NODES}_{K}_all_ls_.csv", index=False)

    SAVE_PATH_ROOT = r"C:\Users\btugu\Documents\develop\research\SCNDP\src\extension\heuristics\results"

    df = pd.DataFrame(records)
    df.to_csv(f"{SAVE_PATH_ROOT}/csv/dense/Result_heuristics_{name_model}_{DENSE_NODES}_{K}_all_DENSE.csv", index=False)

    # SAVE_PATH_ROOT = r"C:\Users\btugu\Documents\develop\research\SCNDP\src\extension\heuristics\results"

    # df = pd.DataFrame(records)
    # df.to_csv(f"{SAVE_PATH_ROOT}/csv/dist/Result_heuristics_{name_model}_{NODES}_{K}_all_DIST_FUNC.csv", index=False)