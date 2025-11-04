import numpy as np
from typing import Tuple, Set, Dict
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
from itertools import combinations
import networkx as nx

def solve_lp(
  G: nx.Graph, 
  pre_fixed: set, 
  k: int
  ) -> Tuple[Dict[int, float], float]:
  """
    Solve the LP relaxation of the REGA problem using a sparse representation.
  """
  V = list(G.nodes())
  n = len(V)

  Pairs = [tuple(sorted(e)) for e in combinations(V, 2)]
  m2    = len(Pairs)
  Nvar  = n + m2
  s_idx = {v: i for i, v in enumerate(V)}
  x_idx = {e: n + j for j, e in enumerate(Pairs)}

  
  rows, cols, data, rhs = [], [], [], []

  def _add_coeff(r, c, val):
    rows.append(r); cols.append(c); data.append(val)

  r = 0 

  # budget 
  for i in range(n):
      _add_coeff(r, i, 1.0)
  rhs.append(k); r += 1

  # edge upper bounds
  for (u, v) in G.edges():
      u, v   = sorted((u, v))
      puv    = G.edges[u, v]['p']
      _add_coeff(r, x_idx[(u, v)], 1.0)
      _add_coeff(r, s_idx[u], -1.0)
      _add_coeff(r, s_idx[v], -1.0)
      rhs.append(1 - puv); r += 1

  # triangle cuts for each real edge (i, j) and every
  for (i, j) in G.edges():
      i, j = sorted((i, j))
      for k_ in V:
          if k_ == i or k_ == j:
              continue
          _add_coeff(r, x_idx[tuple(sorted((i, k_)))], 1.0)  
          _add_coeff(r, x_idx[(i, j)], -1.0)  
          _add_coeff(r, x_idx[tuple(sorted((j, k_)))], -1.0)   
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

  res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                bounds=bounds, method="highs")
  if not res.success:
      raise RuntimeError("LP infeasible: " + res.message)

  s_vals = {v: res.x[s_idx[v]] for v in V}
  x_sum  = res.x[n:].sum()
  obj    = len(Pairs) - x_sum
  return s_vals, obj

def rega(
  G: nx.Graph,
  k: int,
  num_samples: int = 100_000
  ) -> Set[int]:
  """
    Full REGA pipeline: LP‐rounding + CSP‐refined local swaps.
  """

  # iterative rounding
  D = set()
  for _ in range(k):

    s_vals, _ = solve_lp(G, pre_fixed=D, k=k)

    # pick the fractional s_i largest among V \ D
    u = max((v for v in G.nodes() if v not in D),
            key=lambda v: s_vals[v])
    D.add(u)

  # local‐swap refinement
  # S_opt = local_search(G, D, num_samples)

  return D