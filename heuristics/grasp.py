import random
from typing import Tuple, Set
import networkx as nx
from tqdm import tqdm

from .utils import epc_mc_deleted, local_search

def grasp_cndp(
    G: nx.Graph,
    K: int,
    alpha: float = 0.1,
    num_samples: int = 10_000,
    restarts: int = 3,
    use_tqdm: bool = False
  ) -> Tuple[Set[int], float]:
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
    restarts: int = 30
  ) -> Tuple[Set[int], float]:
  """
    Combined GRASP with local search procedure.
  """

  S_grasp, _ = grasp_cndp(
      G.copy(), K, num_samples=mc_samples_grasp, 
      alpha=alpha, restarts=restarts, use_tqdm=False)

  S_opt = local_search(G.copy(), S_grasp, mc_samples_ls)

  return S_opt