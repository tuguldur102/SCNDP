Stochastic Critical Node Detection Problem (SCNDP)

This repository contains implementations and experiments for solving the Stochastic Critical Node Detection Problem (SCNDP), including heuristic, metaheuristic, and learning-based approaches.

1. Setup

Create a virtual environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

Install required packages

pip install --upgrade pip
pip install -r requirements.txt

Project structure

├── conference/
                # Implementation of My Thai et al. paper
                # MIS-based greedy heuristic code
                # ES-based greedy heuristic code
                # Parallel and optimized versions
|
├── extension/
│   ├── heuristics/                     # REGA algorithm details and scratch 
│   ├── learning/                     # learning-based algorithm details and 
|
├── requirements.txt              # Python dependencies
├── README.md                     # This file

2. Detailed Folder Descriptions

2.1 conference/

my_thai_paper/: Reproduces the algorithm from My Thai et al. (2015) paper, including theoretical background and experimental validation.

greedy_mis/: Implements a greedy strategy based on Maximum Independent Set (MIS) formulations.

greedy_es/: Implements a greedy heuristic using Expected Surviving connectivity (ES) criteria.

parallelized/: Optimized and parallelized versions of the above heuristics, utilizing multiprocessing and efficient data structures (e.g., CSR format).

Each subfolder includes:

Source code (*.py files)

Example usage scripts (run_*.sh or Jupyter notebooks)

2.2 extension/

rega/: From-scratch implementation of the REGA algorithm, with parameter tuning and detailed comments.

grasp/: Greedy Randomized Adaptive Search Procedure (GRASP) implementation, including reactive alpha and path-relinking extensions.

learning/: Learning-based pipelines combining Graph Neural Networks (GNNs) for encoding uncertainty and Reinforcement Learning (RL) decoders for node selection. Contains training scripts, model checkpoints, and evaluation routines.
