import glob
import networkx as nx

import sys
sys.path.append('../')

from utils.graph_funcs import graph_from_file


def is_unique(folder, G):
    all_graphs = glob.glob(folder+'/*')

    for graph in all_graphs:
        cur_G = graph_from_file(graph)
        if nx.is_isomorphic(G, cur_G):
            return False

    return True

dirs = glob.glob('N*_d3_graphs')

for folder in dirs:
    print(folder)
    n = int(folder.split('_')[0][1:])
    d = int(folder.split('_')[1][1:])
    print('Nodes: {}, degree: {}'.format(n, d))

    count = 0
    while count < 20:
        G = nx.random_regular_graph(d, n)
        if nx.is_connected(G): #and is_unique(folder, G):
            count += 1
            edges = list(G.edges())

            with open(folder+'/G{}.txt'.format(count), 'w') as fn:
                edgestr = ''.join(['{}, '.format(e) for e in edges])
                edgestr = edgestr.strip(', ')
                fn.write(edgestr)

