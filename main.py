import argparse
import os
import time
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import random

from heuristics.greedy_es_variants import greedy_empty_set_celf
from heuristics.greedy_mis_variants import greedy_with_mis
from heuristics.graph_centrality import (
  remove_k_betweenness,
  remove_k_degree_centrality,
  remove_k_pagerank_nodes,
)
from heuristics.grasp import grasp_cndp
from heuristics.rega import rega
from heuristics.utils import local_search, epc_mc_deleted

from learning.model import SAGEEdgeProbModel
from learning.gnn_1_shot import predict
from learning.greedy_gnn import greedy_gnn

# Helper functions

def set_seeds(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)

def load_model(
  ckpt_path: str,
  device: torch.device,
  in_dim: int = 11,
  hidden_dim: int = 256,
  heads: int = 8,
  dropout: float = 0.4,
  aggr: str = 'mean'
) -> torch.nn.Module:
  model = SAGEEdgeProbModel(
    in_dim=in_dim, hidden_dim=hidden_dim, heads=heads,
    dropout=dropout, aggr=aggr
  ).to(device)
  state = torch.load(ckpt_path, map_location=device)
  model.load_state_dict(state)
  model.eval()
  return model

def make_graph_models(
  n: int,
  seed: int,
  include_er: bool = True,
  include_ba: bool = True,
  include_sw: bool = True
) -> Dict[str, nx.Graph]:
  models: Dict[str, nx.Graph] = {}
  if include_er:
    models['ER'] = nx.erdos_renyi_graph(n, 0.0443, seed=seed)
  if include_ba:
    models['BA'] = nx.barabasi_albert_graph(n, 2, seed=seed)
  if include_sw:
    models['SW'] = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
  return models

DIST_FUNCS = {
  'uniform': lambda: float(np.random.uniform(0.0, 1.0)),
  'normal':  lambda: float(np.clip(np.random.normal(0.5, 0.2), 0, 1)),
  'beta':    lambda: float(np.random.beta(2, 5)),
}

def ensure_outdir(path: str) -> None:
  os.makedirs(path, exist_ok=True)

def _timecall(fn: Callable[[], Iterable[int]]) -> Tuple[Iterable[int], float]:
  t0 = time.perf_counter()
  S = fn()
  return S, (time.perf_counter() - t0)

def _eval_epc(fresh_graph: Callable[[], nx.Graph], S: Iterable[int], n_samples_eval: int) -> float:
  return epc_mc_deleted(fresh_graph(), set(S), n_samples_eval)

# Suite runner

def run_suite(
  fresh_graph: Callable[[], nx.Graph],
  K: int,
  model: torch.nn.Module,
  device: torch.device,
  n_samples_ls: int,
  n_samples_eval: int,
  grasp_alpha: float,
  grasp_restarts: int,
  mis_trials: int,
  with_ls: bool,
  include_rega: bool,
  include_grasp: bool,
) -> List[Dict]:
  """Run selected algorithms; return list of record dicts."""
  records = []

  # 1) Centralities
  S, dt = _timecall(lambda: remove_k_degree_centrality(fresh_graph(), K))
  records.append(("Degree-based", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  S, dt = _timecall(lambda: remove_k_betweenness(fresh_graph(), K))
  records.append(("Betweenness", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  S, dt = _timecall(lambda: remove_k_pagerank_nodes(fresh_graph(), K))
  records.append(("PageRank", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  # 2) Greedy ES (CELF)
  def _greedy():
    S_, _ = greedy_empty_set_celf(fresh_graph(), K, num_samples=n_samples_ls)
    return S_
  S, dt = _timecall(_greedy)
  records.append(("Greedy", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  if with_ls:
    S, _ = _timecall(_greedy)
    S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, num_samples=n_samples_ls))
    records.append(("Greedy + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  # 3) Greedy MIS
  def _greedy_mis():
    S_, _ = greedy_with_mis(fresh_graph(), K, num_trails=mis_trials, num_samples=n_samples_ls)
    return S_
  S, dt = _timecall(_greedy_mis)
  records.append(("Greedy with MIS", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  if with_ls:
    S, _ = _timecall(_greedy_mis)
    S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, num_samples=n_samples_ls))
    records.append(("Greedy with MIS + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  # 4) REGA
  if include_rega:
    def _rega():
      return rega(fresh_graph(), k=K, num_samples=n_samples_ls)
    S, dt = _timecall(_rega)
    records.append(("REGA", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

    if with_ls:
      S, _ = _timecall(_rega)
      S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, n_samples_ls))
      records.append(("REGA + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  # 5) GRASP
  if include_grasp:
    def _grasp():
      S_, _ = grasp_cndp(
        fresh_graph(), K, num_samples=n_samples_ls,
        alpha=grasp_alpha, restarts=grasp_restarts, use_tqdm=False
      )
      return S_
    S, dt = _timecall(_grasp)
    records.append(("GRASP", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

    if with_ls:
      S, _ = _timecall(_grasp)
      S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, n_samples_ls))
      records.append(("GRASP + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  # 6) GNN (1-shot)
  def _gnn():
    return predict(model, fresh_graph(), K, device)
  S, dt = _timecall(_gnn)
  records.append(("GNN (1 shot)", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  if with_ls:
    S, _ = _timecall(_gnn)
    S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, n_samples_ls))
    records.append(("GNN (1 shot) + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  # 7) Greedy-GNN
  def _greedy_gnn_call():
    return greedy_gnn(model, fresh_graph(), K, device)
  S, dt = _timecall(_greedy_gnn_call)
  records.append(("Greedy GNN", dt, _eval_epc(fresh_graph, S, n_samples_eval)))

  if with_ls:
    S, _ = _timecall(_greedy_gnn_call)
    S_ls, dt = _timecall(lambda: local_search(fresh_graph(), S, n_samples_ls))
    records.append(("Greedy GNN + Local Search", dt, _eval_epc(fresh_graph, S_ls, n_samples_eval)))

  return [{"algo": a, "time": t, "epc": epc} for (a, t, epc) in records]

# Tasks

def task_uniform(args):
  """Fixed p in [start, stop] step."""
  set_seeds(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_model(args.ckpt_path, device)
  ensure_outdir(args.outdir)

  graph_models = make_graph_models(
    args.nodes, args.seed,
    include_er=('ER' in args.models),
    include_ba=('BA' in args.models),
    include_sw=('SW' in args.models),
  )
  K = args.k if args.k is not None else int(args.nodes * args.k_frac)
  p_values = list(np.round(np.arange(args.p_start, args.p_stop + 1e-9, args.p_step), 3))

  all_rows: List[Dict] = []
  for name_model, G in tqdm(graph_models.items(), desc="Graph models", total=len(graph_models)):
    for p in tqdm(p_values, desc=f"p grid ({name_model})", total=len(p_values), leave=False):
      def fresh_graph():
        H = G.copy()
        for u, v in H.edges():
          H[u][v]['p'] = float(p)
        return H

      suite = run_suite(
        fresh_graph, K, model, device,
        n_samples_ls=args.ls_samples, n_samples_eval=args.eval_samples,
        grasp_alpha=args.grasp_alpha, grasp_restarts=args.grasp_restarts,
        mis_trials=args.mis_trials,
        with_ls=True,
        include_rega=True,
        include_grasp=True,
      )
      for r in suite:
        all_rows.append({ "model": name_model, "p": float(p), **r })

  df = pd.DataFrame(all_rows)
  fn = os.path.join(args.outdir, f"Result_heuristics_uniform_{args.nodes}_{K}_trial_{args.trial}.csv")
  df.to_csv(fn, index=False)
  print(f"Saved: {fn}")

def task_heterogeneous(args):
  """Edge probabilities drawn from distributions."""
  set_seeds(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_model(args.ckpt_path, device)
  ensure_outdir(args.outdir)

  graph_models = make_graph_models(
    args.nodes, args.seed,
    include_er=('ER' in args.models),
    include_ba=('BA' in args.models),
    include_sw=('SW' in args.models),
  )
  K = args.k if args.k is not None else int(args.nodes * args.k_frac)

  chosen_dists = args.dists
  for d in chosen_dists:
    if d not in DIST_FUNCS:
      raise ValueError(f"Unknown dist: {d}. Choose from {list(DIST_FUNCS)}")

  all_rows: List[Dict] = []
  for name_model, G in tqdm(graph_models.items(), desc="Graph models", total=len(graph_models)):
    for dist_name in tqdm(chosen_dists, desc=f"Distributions ({name_model})", total=len(chosen_dists), leave=False):
      dist_func = DIST_FUNCS[dist_name]

      def fresh_graph():
        H = G.copy()
        for u, v in H.edges():
          H[u][v]['p'] = dist_func()
        return H

      suite = run_suite(
        fresh_graph, K, model, device,
        n_samples_ls=args.ls_samples, n_samples_eval=args.eval_samples,
        grasp_alpha=args.grasp_alpha, grasp_restarts=args.grasp_restarts,
        mis_trials=args.mis_trials,
        with_ls=True,
        include_rega=True,
        include_grasp=True,
      )
      for r in suite:
        all_rows.append({ "model": name_model, "name_dist": dist_name, **r })

  df = pd.DataFrame(all_rows)
  fn = os.path.join(args.outdir, f"Results_heterogeneous_{args.nodes}_{K}_all_DIST.csv")
  df.to_csv(fn, index=False)
  print(f"Saved: {fn}")

def _run_large_common(
  args,
  with_ls: bool,
  csv_tag: str,
  include_er: bool,
  include_ba: bool,
  include_sw: bool,
  also_run_dist: bool = True,
):
  set_seeds(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_model(args.ckpt_path, device)
  ensure_outdir(args.outdir)

  nodes_list = args.nodes_list
  if not nodes_list:
    raise ValueError("--nodes-list is required for large-* tasks, e.g. --nodes-list 200,300")

  for N in tqdm(nodes_list, desc="Processing node sizes"):
    K = args.k if args.k is not None else int(N * args.k_frac)

    # Fixed-p pass
    graph_models = make_graph_models(N, args.seed, include_er, include_ba, include_sw)
    rows: List[Dict] = []
    for name_model, G in tqdm(graph_models.items(), desc=f"Models (N={N})", total=len(graph_models), leave=False):
      for p in tqdm(args.p_list, desc=f"p-list ({name_model})", total=len(args.p_list), leave=False):
        def fresh_graph():
          H = G.copy()
          for u, v in H.edges():
            H[u][v]['p'] = float(p)
          return H

        suite = run_suite(
          fresh_graph, K, model, device,
          n_samples_ls=args.ls_samples, n_samples_eval=args.eval_samples,
          grasp_alpha=args.grasp_alpha, grasp_restarts=args.grasp_restarts,
          mis_trials=args.mis_trials if args.mis_trials is not None else (50 if not with_ls or N >= 200 else 30),
          with_ls=with_ls,
          include_rega=False,
          include_grasp=False,
        )
        for r in suite:
          rows.append({ "model": name_model, "p": float(p), **r })

    df = pd.DataFrame(rows)
    fn = os.path.join(args.outdir, f"Result_heuristics_{N}_{K}_all_{csv_tag}.csv")
    df.to_csv(fn, index=False)
    print(f"Saved: {fn}")

    # Heterogeneous-probability (DIST)
    if also_run_dist:
      rows = []
      for name_model, G in tqdm(graph_models.items(), desc=f"Models DIST (N={N})", total=len(graph_models), leave=False):
        for dist_name in tqdm(args.dists, desc=f"Dists ({name_model})", total=len(args.dists), leave=False):
          dist_func = DIST_FUNCS[dist_name]

          def fresh_graph():
            H = G.copy()
            for u, v in H.edges():
              H[u][v]['p'] = dist_func()
            return H

          suite = run_suite(
            fresh_graph, K, model, device,
            n_samples_ls=args.ls_samples, n_samples_eval=args.eval_samples,
            grasp_alpha=args.grasp_alpha, grasp_restarts=args.grasp_restarts,
            mis_trials=args.mis_trials if args.mis_trials is not None else (50 if not with_ls or N >= 200 else 30),
            with_ls=with_ls,
            include_rega=False,
            include_grasp=False,
          )
          for r in suite:
            rows.append({ "model": name_model, "name_dist": dist_name, **r })

      df = pd.DataFrame(rows)
      fn = os.path.join(args.outdir, f"Result_heuristics_{N}_{K}_all_{csv_tag}_DIST.csv")
      df.to_csv(fn, index=False)
      print(f"Saved: {fn}")

def task_large_with_ls(args):
  _run_large_common(
    args,
    with_ls=True,
    csv_tag="large",
    include_er=('ER' in args.models),
    include_ba=('BA' in args.models),
    include_sw=('SW' in args.models),
    also_run_dist=(not args.skip_dist),
  )

def task_large_without_ls(args):
  _run_large_common(
    args,
    with_ls=False,
    csv_tag="large_no_ls",
    include_er=('ER' in args.models),
    include_ba=('BA' in args.models),
    include_sw=('SW' in args.models),
    also_run_dist=(not args.skip_dist),
  )

# CLI

def parse_list_of_ints(csv: str) -> List[int]:
  return [int(x.strip()) for x in csv.split(",") if x.strip()]

def parse_list_of_floats(csv: str) -> List[float]:
  return [float(x.strip()) for x in csv.split(",") if x.strip()]

def parse_models(csv: str) -> List[str]:
  items = [x.strip().upper() for x in csv.split(",") if x.strip()]
  for it in items:
    if it not in {"ER", "BA", "SW"}:
      raise argparse.ArgumentTypeError(f"Unknown model '{it}'. Choose ER, BA, SW.")
  return items

def parse_dists(csv: str) -> List[str]:
  items = [x.strip().lower() for x in csv.split(",") if x.strip()]
  for it in items:
    if it not in DIST_FUNCS:
      raise argparse.ArgumentTypeError(f"Unknown dist '{it}'. Choose from {list(DIST_FUNCS)}.")
  return items

parser = argparse.ArgumentParser(description="SCNDP experiments – unified runner")
sub = parser.add_subparsers(dest="task", required=True)

# Common options for small (uniform/heterogeneous)
def add_common_small(p):
  p.add_argument("--nodes", type=int, default=100, help="Number of nodes (default: 100)")
  p.add_argument("--k", type=int, default=None, help="Exact K (overrides --k-frac)")
  p.add_argument("--k-frac", type=float, default=0.1, help="K as fraction of N (default: 0.1)")
  p.add_argument("--models", type=parse_models, default=parse_models("ER,BA,SW"),
                  help="Comma-separated graph models from {ER,BA,SW}")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--eval-samples", type=int, default=100_000)
  p.add_argument("--ls-samples", type=int, default=10_000)
  p.add_argument("--grasp-restarts", type=int, default=3)
  p.add_argument("--grasp-alpha", type=float, default=0.05)
  p.add_argument("--mis-trials", type=int, default=30)
  p.add_argument("--ckpt-path", type=str, default="/learning/checkpoints/best_model_cla_30_diff.pt")
  p.add_argument("--outdir", type=str, default="/results/csv")

# Common options for large
def add_common_large(p):
  p.add_argument("--nodes-list", type=parse_list_of_ints, required=True,
                  help="Comma-separated list, e.g. 200,300,500")
  p.add_argument("--k", type=int, default=None, help="Exact K (overrides --k-frac)")
  p.add_argument("--k-frac", type=float, default=0.1)
  p.add_argument("--models", type=parse_models, default=parse_models("ER,BA,SW"),
                  help="Comma-separated graph models from {ER,BA,SW}")
  p.add_argument("--p-list", type=parse_list_of_floats, default=parse_list_of_floats("0.1,0.2,0.3,0.4,0.5,0.7,1.0"))
  p.add_argument("--dists", type=parse_dists, default=parse_dists("uniform,normal,beta"))
  p.add_argument("--skip-dist", action="store_true", help="Skip the distribution-based runs")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--eval-samples", type=int, default=100_000)
  p.add_argument("--ls-samples", type=int, default=10_000)
  p.add_argument("--grasp-restarts", type=int, default=3)
  p.add_argument("--grasp-alpha", type=float, default=0.05)
  p.add_argument("--mis-trials", type=int, default=None, help="Override MIS trials (default 50 for large, else 30)")
  p.add_argument("--ckpt-path", type=str, default="/learning/checkpoints/best_model_cla_30_diff.pt")
  p.add_argument("--outdir", type=str, default="/results/csv")

# uniform
p_uni = sub.add_parser("uniform", help="Fixed p sweep on small graphs (with LS, REGA, GRASP).")
add_common_small(p_uni)
p_uni.add_argument("--p-start", type=float, default=0.0)
p_uni.add_argument("--p-stop",  type=float, default=1.0)
p_uni.add_argument("--p-step",  type=float, default=0.1)
p_uni.add_argument("--trial",   type=int,   default=1)
p_uni.set_defaults(func=task_uniform)

# heterogeneous
p_het = sub.add_parser("heterogeneous", help="Edge probabilities drawn from distributions (with LS, REGA, GRASP).")
add_common_small(p_het)
p_het.add_argument("--dists", type=parse_dists, default=parse_dists("uniform,normal,beta"))
p_het.set_defaults(func=task_heterogeneous)

# large with LS
p_large_ls = sub.add_parser("large_with_ls", help="Large graphs, fixed p (+ optional DIST), with local search (no REGA/GRASP).")
add_common_large(p_large_ls)
p_large_ls.set_defaults(func=task_large_with_ls)

# large without LS
p_large_no = sub.add_parser("large_without_ls", help="Large graphs, fixed p (+ optional DIST), NO local search (no REGA/GRASP).")
add_common_large(p_large_no)
p_large_no.set_defaults(func=task_large_without_ls)

args = parser.parse_args()
args.func(args)
