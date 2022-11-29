# Quantum Local Search

<img src="https://user-images.githubusercontent.com/20692050/125842635-8fd047a0-2674-449a-a68a-3bf557f73091.png" width="400" class="center">

## Overview
Constrained combinatorial optimization is a ubiquitous problem with applications across finance, medicine, machine learning, etc. Our ability to solve these problems with quantum algorithms is constrained by the limited number of qubits that are available in current NISQ processors. The quantum local search algorithm addresses this issue by finding solutions to smaller subproblems that can be recombined into a global solution. The code in this repository gives an implementation of quantum local search algorithm for finding approximate solutions to the Maximum Independent Set problem on large graphs. A paper describing our work is available online: https://arxiv.org/abs/2107.04109.

## Getting started
The code in this repo requires the `qcopt` package which can be found [here](https://github.com/teaguetomesh/quantum-constrained-optimization.git). Follow the instructions below to install the `qcopt` package.

```bash
git clone https://github.com/teaguetomesh/quantum-constrained-optimization.git
python3 -m venv your_new_venv       # create a new Python virtual environment
source your_new_venv/bin/activate   # activate that virtual environment
cd quantum-constrained-optimization
pip install -r requirements.txt     # install the package dependecies
pip install -e .                    # install the qcopt package
```

The code which implements the QLS algorithm is contained in `solve_mis.py`. This file makes calls to the functions defined in `qls_ansatz.py` to construct the quantum circuits. The two notebooks `plotting.ipynb` and `plotting-revision.ipynb` use the data found in `benchmark_results.tar` to generate the plots found in the published paper. The tar file should be decompressed before running these notebooks:

```bash
tar -xzvf benchmark_results.tar
```

The file `QLS_benchmark.py` can be used to perform additional simulations of the QLS algorithm for the benchmark graphs found in `benchmark_graphs` or any other graphs of interest.


## Citation
If you use this code, please cite our work:


    Teague Tomesh, Zain H. Saleem, and Martin Suchara. "Quantum Local Search with the Quantum Alternating Operator Ansatz." *Quantum* 6 (2022): 781.
