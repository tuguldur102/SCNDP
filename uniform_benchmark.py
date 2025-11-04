import argparse
import time
from itertools import combinations
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import random

from heuristics.greedy_es_variants import greedy_empty_set_celf, greedy_empty_set_celf_local_search
from heuristics.greedy_mis_variants import greedy_with_mis, greedy_with_mis_local_search, robust_greedy_mis_optimized
from heuristics.graph_centrality import remove_k_betweenness, remove_k_degree_centrality, remove_k_pagerank_nodes
from heuristics.grasp import grasp_cndp, grasp_meta
from heuristics.rega import rega  
from heuristics.utils import local_search, epc_mc_deleted

from learning.model import SAGEEdgeProbModel
from learning.gnn_1_shot import predict
from learning.greedy_gnn import greedy_gnn

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000

LOCAL_SEARCH_ITER = 1
GRASP_RESTARTS = 3
GRASP_ALPHA = 0.05

CHKPNT_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/learning/notebooks/gnn/checkpoints"
CKPT_PATH = f"{CHKPNT_ROOT}/best_model_cla_30_diff.pt"

TRIAL = 1
NODES = 100
K = int(NODES * 0.1)

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# nodes 100, edges 200 (Sparse Graphs)
graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

# DENSE_NODES = 50 
# K = 5
# # nodes 50, p = 0.5 (Dense Graphs) 2500 edges

# graph_models_dense = {
#   'ER': nx.erdos_renyi_graph(DENSE_NODES, 0.5025, seed=SEED),
#   'BA': nx.barabasi_albert_graph(DENSE_NODES, 25,seed=SEED),
#   'SW': nx.watts_strogatz_graph(DENSE_NODES, 25, 0.3, seed=SEED)
# }

dist_funcs = {
  'uniform': lambda: np.random.uniform(0.0, 1.0),
  'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
  'beta': lambda: np.random.beta(2, 5),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SAGEEdgeProbModel(in_dim=11, hidden_dim=256, heads=8,
                        dropout=0.4, aggr='mean').to(device)

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()


for name_model, G in tqdm(
  graph_models.items(), 
  desc="Processing models", 
  total=len(graph_models)):
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

    # heuristics 4: Greedy ES CELF
    t0 = time.perf_counter()

    greedy_S, _ = greedy_empty_set_celf(
      fresh_graph(), K, num_samples=N_SAMPLE_LS)
    epc_greedy = epc_mc_deleted(fresh_graph(), greedy_S, N_SAMPLE_EVAL)
    t_greedy_es = time.perf_counter() - t0

    t0 = time.perf_counter()
    greedy_S, _ = greedy_empty_set_celf(
      fresh_graph(), K, num_samples=N_SAMPLE_LS)
    greedy_es_S_ls = local_search(fresh_graph(), greedy_S, num_samples=N_SAMPLE_LS)
    epc_greedy_es_ls = epc_mc_deleted(fresh_graph(), greedy_es_S_ls, N_SAMPLE_EVAL)
    t_greedy_es_ls = time.perf_counter() - t0

    # heuristics 5: Greedy MIS CELF
    t0 = time.perf_counter()

    greedy_mis_S, _ = greedy_with_mis(
      fresh_graph(), K, num_trails=30, num_samples=N_SAMPLE_LS)
    epc_greedy_mis = epc_mc_deleted(fresh_graph(), greedy_mis_S, N_SAMPLE_EVAL)
    t_greedy_mis = time.perf_counter() - t0

    t0 = time.perf_counter()

    greedy_mis_S, _ = greedy_with_mis(
      fresh_graph(), K, num_trails=30, num_samples=N_SAMPLE_LS)
    greedy_mis_S_ls = local_search(fresh_graph(), greedy_mis_S, num_samples=N_SAMPLE_LS)
    epc_greedy_mis_ls = epc_mc_deleted(fresh_graph(), greedy_mis_S_ls, N_SAMPLE_EVAL)
    t_greedy_mis_ls = time.perf_counter() - t0

    # heuristics 6: REGA
    t0 = time.perf_counter()

    rega_S = rega(
      fresh_graph(),
      k=K,
      num_samples=N_SAMPLE_LS)
    epc_rega = epc_mc_deleted(fresh_graph(), rega_S, N_SAMPLE_EVAL)
    t_rega = time.perf_counter() - t0

    t0 = time.perf_counter()
    rega_S = rega(
      fresh_graph(),
      k=K,
      num_samples=N_SAMPLE_LS)
    rega_S_ls = local_search(fresh_graph(), rega_S, N_SAMPLE_LS)
    epc_rega_ls = epc_mc_deleted(fresh_graph(), rega_S_ls, N_SAMPLE_EVAL)
    t_rega_ls = time.perf_counter() - t0

    # heuristics 7: Grasp + ls
    t0 = time.perf_counter()

    grasp_S, _ = grasp_cndp(
        fresh_graph(), K, num_samples=N_SAMPLE_LS, 
        alpha=GRASP_ALPHA, restarts=GRASP_RESTARTS, use_tqdm=False)
    epc_grasp = epc_mc_deleted(fresh_graph(), grasp_S, N_SAMPLE_EVAL)
    t_grasp = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    grasp_S, _ = grasp_cndp(
        fresh_graph(), K, num_samples=N_SAMPLE_LS, 
        alpha=GRASP_ALPHA, restarts=GRASP_RESTARTS, use_tqdm=False)
    grasp_S_ls = local_search(fresh_graph(), grasp_S, N_SAMPLE_LS)
    epc_grasp_ls = epc_mc_deleted(fresh_graph(), grasp_S_ls, N_SAMPLE_EVAL)
    t_grasp_ls = time.perf_counter() - t0

    # heuristics 8: Grasp + ls + path relinking + reactive alpha
    t0 = time.perf_counter()

    grasp_path_S = grasp_meta(
      fresh_graph(), K, 
      restarts=GRASP_RESTARTS, 
      mc_samples=N_SAMPLE_LS, elite_size=5)
    epc_grasp_path = epc_mc_deleted(fresh_graph(), grasp_path_S, N_SAMPLE_EVAL)
    t_grasp_path = time.perf_counter() - t0

    t0 = time.perf_counter()
    grasp_path_S = grasp_meta(
      fresh_graph(), K, 
      restarts=GRASP_RESTARTS, 
      mc_samples=N_SAMPLE_LS, elite_size=5)
    grasp_path_S_ls = local_search(fresh_graph(), grasp_path_S, N_SAMPLE_LS)
    epc_grasp_path_ls = epc_mc_deleted(fresh_graph(), grasp_path_S_ls, N_SAMPLE_EVAL)
    t_grasp_path_ls = time.perf_counter() - t0

    # metaheuristics 1: GNN (1 shot)
    t0 = time.perf_counter()
    gnn_S = predict(model, fresh_graph(), K, device)
    epc_gnn = epc_mc_deleted(fresh_graph(), gnn_S, N_SAMPLE_EVAL)
    t_gnn = time.perf_counter() - t0

    t0 = time.perf_counter()
    gnn_S = predict(model, fresh_graph(), K, device)
    gnn_S_ls = local_search(fresh_graph(), gnn_S, N_SAMPLE_LS)
    epc_gnn_ls = epc_mc_deleted(fresh_graph(), gnn_S_ls, N_SAMPLE_EVAL)
    t_gnn_ls = time.perf_counter() - t0

    # metaheuristics 2: Greedy-GNN
    t0 = time.perf_counter()
    greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
    epc_greedy_gnn = epc_mc_deleted(fresh_graph(), greedy_gnn_S, N_SAMPLE_EVAL)
    t_greedy_gnn = time.perf_counter() - t0

    t0 = time.perf_counter()
    greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
    greedy_gnn_S_ls = local_search(fresh_graph(), greedy_gnn_S, N_SAMPLE_LS)
    epc_greedy_gnn_ls = epc_mc_deleted(fresh_graph(), greedy_gnn_S_ls, N_SAMPLE_EVAL)
    t_greedy_gnn_ls = time.perf_counter() - t0
    
    for algo, t, epc in [
      ('Degree-based', t_degree, epc_degree),
      ('Betweenness', t_between, epc_between),
      ('PageRank', t_pagerank, epc_pagerank),

      ('Greedy', t_greedy_es, epc_greedy),
      ('Greedy + Local Search', t_greedy_es_ls, epc_greedy_es_ls),

      ('Greedy with MIS', t_greedy_mis, epc_greedy_mis),
      ('Greedy with MIS + Local Search', t_greedy_mis_ls, epc_greedy_mis_ls),
      
      ('REGA', t_rega, epc_rega),
      ('REGA + Local Search', t_rega_ls, epc_rega_ls),

      ('GRASP', t_grasp, epc_grasp),
      ('GRASP + Local Search', t_grasp_ls, epc_grasp_ls),

      ('GRASP + Path Relinking', t_grasp_path, epc_grasp_path),
      ('GRASP + Path Relinking + Local Search', t_grasp_path_ls, epc_grasp_path_ls),

      ('GNN (1 shot)', t_gnn, epc_gnn),
      ('GNN (1 shot) + Local Search', t_gnn_ls, epc_gnn_ls),

      ('Greedy GNN', t_greedy_gnn, epc_greedy_gnn),
      ('Greedy GNN + Local Search', t_greedy_gnn_ls, epc_greedy_gnn_ls),
    ]:
      
      records.append({
        'model': name_model,
        'p': np.round(p, 1),
        'algo': algo,
        'time': t,
        'epc': epc
      })

    SAVE_PATH_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/final/csv"

    df = pd.DataFrame(records)
    df.to_csv(f"{SAVE_PATH_ROOT}/Result_heuristics_{name_model}_{NODES}_{K}_all_{TRIAL}.csv", index=False)

  # SAVE_PATH_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/heuristics/results"

  # df = pd.DataFrame(records)
  # df.to_csv(f"{SAVE_PATH_ROOT}/csv/dist_job/Result_heuristics_{NODES}_{K}_dist.csv", index=False)
