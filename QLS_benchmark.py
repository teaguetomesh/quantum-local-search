"""
Solve the MIS problem on different benchmark graphs using QLS
"""
import os, sys, argparse, glob
import numpy as np
import solve_mis
import pickle, random
from utils.graph_funcs import graph_from_file, is_indset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', type=str, default=None,
                        help='path to dqva project')
    parser.add_argument('--graph', type=str, default=None,
                        help='glob path to the benchmark graph(s)')
    parser.add_argument('--reps', type=int, default=4,
                        help='Number of repetitions to run')
    parser.add_argument('-v', type=int, default=1,
                        help='verbose')
    parser.add_argument('--npm', type=int, default=None,
                        help='Number of partial mixers to use')
    parser.add_argument('--mnd', type=int, default=None,
                        help='Maximum node distance to use when growing neighborhood')
    parser.add_argument('--extend', type=int, default=0,
                        help='Flag for whether to overwrite or extend repetitions')
    parser.add_argument('--startrep', type=int, default=0,
                        help='Integer repetition label')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of parallel threads passed to Aer')
    parser.add_argument('--hnp', type=int, default=3,
                        help='Number of hot node permutations to try within each local neighborhood')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    QLSROOT = args.path
    if QLSROOT[-1] != '/':
        QLSROOT += '/'
    sys.path.append(QLSROOT)

    all_graphs = glob.glob(QLSROOT + args.graph)
    graph_type = all_graphs[0].split('/')[-2]

    savepath = QLSROOT+'benchmark_results/QLS_{}/'.format(graph_type)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    savepath += 'NPM_{}/'.format(args.npm)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    for graphfn in all_graphs:
        graphname = graphfn.split('/')[-1].strip('.txt')
        cur_savepath = savepath + '{}/'.format(graphname)
        if not os.path.isdir(cur_savepath):
            os.mkdir(cur_savepath)

        G = graph_from_file(graphfn)
        print('\nNew graph:', graphname)
        print(G.edges())
        nq = len(G.nodes)
        init_state = '0'*nq

        if args.extend:
            print('savepath:', cur_savepath)
            all_reps = glob.glob(cur_savepath + '*rep*')
            print('{} reps completed'.format(len(all_reps)))
            if args.startrep > 0:
                rep_range = range(args.startrep, args.startrep+args.reps)
            elif len(all_reps) < args.reps:
                rep_range = range(len(all_reps)+1, args.reps+1)
            else:
                print('Skipping graph {}'.format(graphname))
                continue
            print('rep_range =', list(rep_range))
        else:
            rep_range = range(1, args.reps+1)

        for rep in rep_range:
            print('\n','%'*10, 'Start of Rep', rep+1, '%'*10,'\n')
            out = solve_mis.quantum_local_search(init_state, G, args.npm,
                                                 args.mnd, verbose=args.v,
                                                 threads=args.threads,
                                                 hot_node_permutations=args.hnp)

            # Save the results
            savename = '{}_QLS_rep{}.pickle'.format(graphname, rep)

            with open(cur_savepath+savename, 'ab') as pf:
                pickle.dump({'graph':graphfn, 'out':out}, pf)

if __name__ == '__main__':
    main()

