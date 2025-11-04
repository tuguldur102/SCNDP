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
├── heuristics/
│ └── # Implementations of heuristic algorithms
├── /results
│ └── # Contains plots and CSV files of experimental results
├── heterogeneous_benchmark.py             # Python code for benchmark comparison on heterogeneous probability setting
├── uniform_benchmark.py                   # Python code for benchmark comparison on uniform probability setting
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

- Uniform probability setting:
```bash
python uniform_benchmark.py
```

- Heterogeneous probability setting:
```bash
python heterogeneous_benchmark.py
```

Each script performs benchmark comparisons on 100-node graphs with a deletion budget K = 10.
After execution, results are automatically saved as CSV files in the project root directory.

## Question/Need Support?
If you have any questions or encounter issues, please open an issue. We’ll do our best to help.

## Citation
If you use this repository in your research, please cite the following paper:

```bibtex
@inproceedings{Bayarsaikhan2025,
  author    = {Tuguldur Bayarsaikhan and Altannar Chinchuluun and Ashwin Arulselvan},
  title     = {A Maximal Independent Set Heuristic for the Stochastic Critical Node Detection Problem},
  booktitle = {Proceedings of the 19th International Conference on Algorithmic Aspects in Information and Management (AAIM 2025)},
  address   = {Ulaanbaatar, Mongolia},
  pages     = {accepted},
  year      = {2025},
  publisher = {Lecture Notes in Computer Science, Springer},
}
```

## License

MIT License
