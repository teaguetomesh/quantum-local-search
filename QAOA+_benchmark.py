#!/usr/bin/env python
"""
Use the benchmark graphs to test the performance of QAOA+
"""
import os, sys, argparse, glob
import pickle, random
from pathlib import Path

import qcopt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None, help="path to dqva project")
    parser.add_argument(
        "--graph", type=str, default=None, help="glob path to the benchmark graph(s)"
    )
    parser.add_argument("-P", type=int, default=1, help="P-value for algorithm")
    parser.add_argument("--reps", type=int, default=4, help="Number of repetitions to run")
    parser.add_argument("-v", type=int, default=1, help="verbose")
    parser.add_argument(
        "--threads", type=int, default=0, help="Number of threads to pass to Aer simulator"
    )
    parser.add_argument("--lamda", type=float, default=1, help="Value of the Lagrange multiplier")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ROOT = args.path
    if ROOT[-1] != "/":
        ROOT += "/"
    sys.path.append(ROOT)

    all_graphs = glob.glob(ROOT + args.graph)
    graph_type = all_graphs[0].split("/")[-2]

    savepath = ROOT + f"benchmark_results/QAOA+_P{args.P}_qasm/{graph_type}/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for graphfn in all_graphs:
        graphname = graphfn.split("/")[-1].strip(".txt")
        cur_savepath = savepath + f"{graphname}/"
        Path(cur_savepath).mkdir(parents=True, exist_ok=True)

        G = qcopt.utils.graph_funcs.graph_from_file(graphfn)
        print(graphname, G.edges())

        for rep in range(1, args.reps + 1):
            out = qcopt.qaoa_plus_mis.solve_mis(args.P, G, args.lamda, threads=args.threads)

            ranked_probs = qcopt.qaoa_plus_mis.get_ranked_probs(args.P, G, out["x"], threads=args.threads)

            data_dict = {
                    "lambda": args.lamda,
                    "graph": graphfn,
                    "P": args.P,
                    "function_evals": out["nfev"],
                    "opt_params": out["x"],
                    "top_output": ranked_probs[:5],
            }

            # Save the results
            savename = f"{graphname}_QAOA+_P{args.P}_rep{rep}.pickle"
            with open(cur_savepath + savename, "ab") as pf:
                pickle.dump(data_dict, pf)


if __name__ == "__main__":
    main()
