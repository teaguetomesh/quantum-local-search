# Quantum Local Search

<img src="https://user-images.githubusercontent.com/20692050/125842635-8fd047a0-2674-449a-a68a-3bf557f73091.png" width="400" class="center">

## Overview
Constrained combinatorial optimization is a ubiquitous problem with applications across finance, medicine, machine learning, etc. Our ability to solve these problems with quantum algorithms is constrained by the limited number of qubits that are available in current NISQ processors. The quantum local search algorithm addresses this issue by finding solutions to smaller subproblems that can be recombined into a global solution. The code in this repository gives an implementation of quantum local search for finding approximate solutions to the Maximum Independent Set problem on large graphs. A paper describing our work is available online: https://arxiv.org/abs/2107.04109.


## Citation
If you use this code, please cite our work:


    Teague Tomesh, Zain H. Saleem, and Martin Suchara, Quantum Local Search with Quantum Alternating Operator Ansatz,
    arXiv preprint arXiv:2107.04109 (2021). 
