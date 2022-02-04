#!/usr/bin/env python
import glob
from pathlib import Path
import networkx as nx
import numpy as np

import qcopt

all_graph_types = glob.glob('benchmark_graphs/N*graphs')

for graph_type in all_graph_types:
    all_graphs = glob.glob(f'{graph_type}/G*.txt')
    savepath = f'benchmark_results/BoppanaHalldorsson/{graph_type.split("/")[-1]}'
    Path(savepath).mkdir(parents=True, exist_ok=True)
    for graph in all_graphs:
        G = qcopt.graph_funcs.graph_from_file(graph)
        bh_mis = []
        for _ in range(5):
            bh_mis.append(len(nx.algorithms.approximation.maximum_independent_set(G)))
        print(f'{"/".join(graph.split("/")[-2:])} average BH mis size: {np.mean(bh_mis)}')
        with open(f'{savepath}/{graph.split("/")[-1].strip(".txt")}_bh_results.txt', 'w') as fn:
            fn.write(f'Average BH mis size over 5 repetitions: {np.mean(bh_mis)}')
