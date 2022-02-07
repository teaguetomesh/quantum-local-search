#!/usr/bin/env python
"""
Solve the MIS problem on different benchmark graphs using QLS
"""
import os, sys, argparse, glob
import numpy as np
import solve_mis
import pickle, random
from pathlib import Path

import qcopt

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
    parser.add_argument('--quantum', type=int, default=1,
                        help='Use a quantum solver (1) or a classical solver (0)')
    parser.add_argument('--tag', type=str, default=None,
                        help='A tag to be added to the final savename for help with debugging.')
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

    if args.quantum:
        savepath = f'{QLSROOT}benchmark_results/QLS_{graph_type}/NS{args.mnd}_NPM{args.npm}/'
    else:
        savepath = f'{QLSROOT}benchmark_results/CLS_{graph_type}/NS{args.mnd}/'
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for graphfn in all_graphs:
        graphname = graphfn.split('/')[-1].strip('.txt')
        cur_savepath = savepath + '{}/'.format(graphname)
        if not os.path.isdir(cur_savepath):
            os.mkdir(cur_savepath)

        G = qcopt.graph_funcs.graph_from_file(graphfn)
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
            print('\n','%'*10, 'Start of Rep', rep, '%'*10,'\n')
            if args.quantum:
                out = solve_mis.quantum_local_search(init_state, G, args.npm,
                        args.mnd, verbose=args.v, threads=args.threads,
                        hot_node_permutations=args.hnp)
            else:
                out = solve_mis.classical_local_search(init_state, G, args.mnd,
                        verbose=args.v)
                if not qcopt.graph_funcs.is_indset(out[0], G, little_endian=False):
                    raise Exception('CLS produced an invalid MIS!')

            # Save the results
            if args.quantum:
                savename = f'{graphname}_QLS_rep{rep}.pickle'
            else:
                savename = f'{graphname}_CLS_rep{rep}.pickle'

            if args.tag:
                savename = f'{args.tag}_{savename}'

            with open(cur_savepath+savename, 'ab') as pf:
                pickle.dump({'graph':graphfn, 'out':out}, pf)

if __name__ == '__main__':
    main()

