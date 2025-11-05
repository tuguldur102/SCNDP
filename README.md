# Heuristic Algorithms for the Stochastic Critical Node Detection Problem

This repository contains the source code accompanying the paper  
**“Heuristic Algorithms for the Stochastic Critical Node Detection Problem.”**

It includes implementation of the **Rounding the Expected Graph Algorithm (REGA)** from **"Dinh T. N, and Thai. M. T (2015) Assessing attack vulnerability in networks with uncertainty. INFOCOM"** and our proposed heuristic methods:
- Greedy algorithm
- Greedy with Maximal Independent Set (MIS)
- Greedy randomized adaptive search procedures (GRASP) from **"Feo, Thomas A. and Resende, Mauricio G. C (1995) Greedy randomized adaptive search procedures. Journal of Global Optimization"**

And learning-based methods:
- Greedy Graph Neural Network (GNN)
- GNN (1-shot)

All experiments were conducted on **Ubuntu** using **Python 3.12**.

## Project Structure

```text
├── /results
│   └── # Contains training, validation, and test set graph instances for learning-based algorithms
├── /heuristics
│   └── # Implementations of heuristic algorithms
├── /learning
│   ├── /checkpoints
│   │   └── # Stores the best model checkpoints; new checkpoints will also be saved here
│   └── # Implementations of learning-based algorithms
├── /results
│   └── # Contains plots and CSV files of experimental results
├── heterogeneous_benchmark.py             # Python script for benchmark comparison under a heterogeneous probability setting
├── uniform_benchmark.py                   # Python script for benchmark comparison under a uniform probability setting
├── requirements.txt                       # Python dependencies
├── README.md                           
```

## Setup

To use the code, first clone the repository:
```bash
git clone https://github.com/tuguldur102/StochasticCNDP.git
```

Create a virtual environment (recommended):
```bash
python -m venv env
```

Load to the created virtual environemnt:
```bash
cd env
```

To activate the virtual environment

On Windows:
```bash
venv\\Scripts\\activate
```

or 

on Ubuntu:
```bash
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Running the Experiments

The repository provides two benchmark scripts corresponding to the edge probability settings used in the paper:

```bash
python main.py <task> [options]
```

- `uniform` – small graphs, fixed edge probability p

- `heterogeneous` – small graphs, edge probs from distributions

- `large_with_ls` – large graphs, with local search

- `large_without_ls` – large graphs, without local search

### Examples

Uniform (fixed p sweep):

```bash
python main.py uniform \
  --nodes 100 \
  --models ER,BA,SW \
  --p-start 0.0 --p-stop 1.0 --p-step 0.1 \
  --outdir ./results
```

Runs all algorithms on small graphs for multiple p values.

-----

Heterogeneous (random edge probabilities):

```bash
python main.py heterogeneous \
  --nodes 100 \
  --models ER,BA \
  --dists uniform,normal,beta \
  --outdir ./results
```

Draws each edge probability from the given distributions.

-----

The algorithms including Greedy, Greedy with MIS, Greedy GNN and GNN (1-shot) are used for benchmark evaluation for larger graph instances.

Large (with local search):

```bash
python main.py large_with_ls \
  --nodes-list 200,300,500 \
  --p-list 0.1,0.3,0.5 \
  --outdir ./results
```

Larger graphs with local search procedure.

-----

Large (without local search)

```bash
python main.py large_without_ls \
  --nodes-list 200,300,500 \
  --p-list 0.1,0.3,0.5 \
  --outdir ./results
```

Larger graphs without local search procedure.

-----

After execution, results are automatically saved as CSV files in the project root directory.

### Common Options

| Flag | Description | Default |
|------|--------------|----------|
| `--k` | exact K nodes to remove | derived from `--k-frac` |
| `--k-frac` | K as fraction of N | 0.1 |
| `--seed` | random seed | 42 |
| `--eval-samples` | Monte Carlo evaluation samples | 100000 |
| `--ls-samples` | samples for local search | 10000 |
| `--ckpt-path` | path to GNN model | see code |
| `--outdir` | output folder | see code |

## Question/Need Support?
If you have any questions or encounter issues, please open an issue. We’ll do our best to help.

## Citation
If you use this repository in your research, please cite the following paper:

```bibtex
@inproceedings{Bayarsaikhan2025,
  author    = {Tuguldur Bayarsaikhan and Altannar Chinchuluun and Ashwin Arulselvan},
  title     = {Heuristic Algorithms for the Stochastic Critical Node Detection Problem},
  journal   = {Submitted},
  address   = {Ulaanbaatar, Mongolia},
  pages     = {Sumbitted},
  year      = {2025},
}
```

## License

MIT License
