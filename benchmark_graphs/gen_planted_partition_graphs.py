#!/usr/bin/env python
import os
import glob
import networkx as nx

graph_params = [
                (20, 1, 0.10, 0.02),
                (20, 3, 0.10, 0.02),
                (20, 5, 0.10, 0.02),
               ]

for commSize, numComm, p_in, p_out in graph_params:
    print(f'{commSize * numComm} nodes, {numComm} communities, p_in = {p_in}, p_out = {p_out}')

    folder = f'N{commSize * numComm}_com{numComm}_pin{int(p_in*100)}_pout{int(p_out*100)}_graphs/'
    if not os.path.isdir(folder):
        os.mkdir(folder)

    count = 1
    while count <= 40:
        G = nx.generators.planted_partition_graph(l=numComm, k=commSize,
                                                  p_in=p_in, p_out=p_out)
        if nx.is_connected(G):
            print('|', end='')
            edges = list(G.edges())

            with open(folder+'G{}.txt'.format(count), 'w') as fn:
                edgestr = ''.join(['{}, '.format(e) for e in edges])
                edgestr = edgestr.strip(', ')
                fn.write(edgestr)

            count += 1
    print('\nDONE')

