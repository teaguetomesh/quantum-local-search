#!/usr/bin/env python
"""
Generate random Erdos-Renyi graphs.

Based on the description in Farhi, Gamarnik, Gutman 2020 (https://arxiv.org/abs/2004.09002)
"""
import os
import glob
import networkx as nx
import numpy as np


average_degree = 3
n_vals = [20, 60, 100]

for n in n_vals:
    m = int(average_degree * n / 2)
    print(f'Generating (fixed edge count) Erdos Renyi graphs with {n} nodes and {m} edges')

    folder = f'N{n}_er{average_degree}_graphs/'
    if not os.path.isdir(folder):
        os.mkdir(folder)

    count = 1
    while count <= 40:
        all_possible_edges = []
        for i in range(n-1):
            for j in range(i+1, n):
                all_possible_edges.append((i,j))

        np.random.shuffle(all_possible_edges)

        G = nx.Graph()
        G.add_edges_from(all_possible_edges[:m])

        if nx.is_connected(G) and len(G.nodes) == n:
            edges = list(G.edges())

            with open(folder+'/G{}.txt'.format(count), 'w') as fn:
                edgestr = ''.join([f'{e}, ' for e in edges])
                edgestr = edgestr.strip(', ')
                fn.write(edgestr)

            count += 1
print('DONE')
