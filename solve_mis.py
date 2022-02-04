"""
04/29/2021 - Teague Tomesh

Solve MIS problem on a given graph using quantum local search
"""
import copy
import numpy as np
import networkx as nx

from networkx.generators.ego import ego_graph
from scipy.optimize import minimize

import qiskit
from qiskit import Aer
from qiskit.quantum_info import Statevector

from utils.graph_funcs import *
from utils.helper_funcs import *

from qls_ansatz import gen_qlsa


def quantum_local_search(init_state, G, num_partial_mixers, max_node_dist,
                         verbose=0, threads=0, hot_node_permutations=3):
    """
    Find the MIS of G using Quantum Local Search (QLS), this
    ansatz is composed of two types of unitaries: the cost unitary U_C and the
    mixer unitary U_M. The mixer U_M is made up of individual partial mixers
    which are independently parametrized.

    QLS's key feature is its dynamic reuse of quantum resources
    (i.e. the partial mixers for qubits which are in the MIS are turned off and
    applied to other qubits not currently in the set)
    """

    # Initialization
    simulator = Aer.get_backend(name='aer_simulator_statevector',
                                max_parallel_threads=threads)

    # This function will be what scipy.minimize optimizes
    def f(params):
        # Generate the ansatz
        circ = gen_qlsa(induced_G, cur_mis_state, hot_nodes, params,
                        qubits_to_nodes, nodes_to_qubits, barriers=0,
                        decompose_level=1, verbose=0)

        circ.save_statevector()

        # Compute the cost function
        result = simulator.run(circ).result()
        sv = Statevector(result.get_statevector(circ))
        probs = strip_ancillas(sv.probabilities_dict(decimals=5), circ)

        avg_cost = 0
        for sample, val in probs.items():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += val * sum(x)

        # Return the negative of the cost for minimization
        return -avg_cost

    # Stopping condition is reached when all nodes are either flipped on or
    # have a neighbor which is flipped on
    num_nodes = len(G.nodes)
    stopping_condition = [0 for _ in range(num_nodes)]

    # Continue until stopping condition is reached
    cur_mis_state = init_state
    history = []
    iteration = 1
    while sum(stopping_condition) < len(G.nodes):
        # Select a new initial node
        init_node = np.random.choice([node for node, val in enumerate(stopping_condition) if val == 0])
        if verbose:
            print('-'*10, 'Iteration:', iteration, '-'*10)
            print('Stopping condition:', stopping_condition)
            print('Selected node:', init_node)
            print('Current mis state:', cur_mis_state)

        # Select the nodes to hit with partial mixers
        hot_nodes = []
        k = 1
        candidates = []
        while len(hot_nodes) < num_partial_mixers and k <= max_node_dist:
            # Get the graph induced by the init_node and all neighbors up to
            # distance k edges away
            induced_G = ego_graph(G, init_node, radius=k, center=True)

            # Update the candidates list
            for node in induced_G.nodes:
                # All neighbors should be within the induced graph
                if node not in hot_nodes and node not in candidates \
                        and induced_G.degree(node) == G.degree(node):
                    # None of the neighbors should be turned on
                    if not any([cur_mis_state[n] == '1' for n in list(induced_G[node]) + [node]]):
                        candidates.append(node)

            # Check the nodes in the candidates list, if all of a node's
            # neighbors are contained in the induced graph, then add it to
            # the list of hot nodes
            for node in candidates:
                if induced_G.degree(node) == G.degree(node):
                    hot_nodes.append(node)

                if len(hot_nodes) >= num_partial_mixers:
                    break

            # Clean up the candidates list
            candidates = [node for node in candidates if node not in hot_nodes]

            # Update the candidates list with all nodes at the current k-level
            # that weren't added in this round.
            # This helps keep the hot nodes close to the initial node in
            # terms of k-distance.
            for node in induced_G.nodes:
                if node not in hot_nodes and node not in candidates:
                    if not any([cur_mis_state[n] == '1' for n in list(G[node]) + [node]]):
                        candidates.append(node)

            # increase the radius of the induced graph
            k += 1

        # Collect the nodes which will be involved in the optimization
        participants = [n for n in hot_nodes]
        for node in hot_nodes:
            participants.extend(list(induced_G[node]))
        participants = list(set(participants))

        # Variationally optimize the ansatz
        num_params = len(hot_nodes) + 1
        num_qubits = len(participants)
        qubits_to_nodes = {q : participants[q] for q in range(num_qubits)}
        nodes_to_qubits = {qubits_to_nodes[q] : q for q in qubits_to_nodes.keys()}
        if verbose:
            print('\tCurrent Hot Nodes:', hot_nodes)
            print('\tNum qubits = {}, Num params = {}'.format(num_qubits, num_params))
            print('\tqubits_to_nodes:', qubits_to_nodes)
            print('\tnodes_to_qubits:', nodes_to_qubits)

        # Simulate n = hot_node_permutations independent trials using randomly
        # permuted orderings of the hot nodes
        opt_cost = 1000
        for i in range(hot_node_permutations):
            # Important to start from random initial points
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)

            out = minimize(f, x0=init_params, method='COBYLA')

            temp_params = out['x']
            temp_cost = out['fun']

            if temp_cost < opt_cost:
                opt_cost = temp_cost
                opt_params = temp_params
                best_hot_nodes = copy.copy(hot_nodes)

            hot_nodes = list(np.random.permutation(hot_nodes))

        if verbose:
            print('\tOptimal cost:', opt_cost)
            print('\tOptimal hot nodes:', best_hot_nodes)

        # Get the results of the optimized circuit
        opt_circ = gen_qlsa(induced_G, cur_mis_state, best_hot_nodes,
                            opt_params, qubits_to_nodes, nodes_to_qubits,
                            barriers=0, decompose_level=1, verbose=0)

        opt_circ.save_statevector()

        result = simulator.run(opt_circ).result()
        sv = Statevector(result.get_statevector(opt_circ))
        probs = strip_ancillas(sv.probabilities_dict(decimals=5), opt_circ)

        sorted_probs = sorted([(key, val) for key, val in probs.items()],
                              key=lambda tup: tup[1], reverse=True)
        most_likely_str = sorted_probs[0][0]
        if verbose:
            print('\tMost probable bitstring:', most_likely_str)

        # Update the current initial state with the result of the local search
        prev_mis_state = cur_mis_state
        if verbose:
            print('\tPrevious mis:', prev_mis_state)
        temp_state = list(prev_mis_state)
        for qubit, bit in enumerate(reversed(most_likely_str)):
            temp_state[qubits_to_nodes[qubit]] = bit
        cur_mis_state = ''.join(temp_state)
        if verbose:
            print('\tUpdated mis: ', cur_mis_state)
            print()

        # Save results to the history
        history.append((init_node, prev_mis_state, induced_G, opt_cost,
                        opt_params, best_hot_nodes, qubits_to_nodes,
                        nodes_to_qubits, cur_mis_state))

        # Update the stopping condition by turning all bits to 1 if they OR a
        # neighbor were switched on
        for node, bit in enumerate(cur_mis_state):
            if bit == '1':
                for n in list(G[node]) + [node]:
                    stopping_condition[n] = 1

        iteration += 1

    return cur_mis_state, history


def classical_local_search(init_state, G, max_node_dist, verbose=0, threads=0):
    """
    Find the MIS of G using Classical Local Search (CLS).
    At every round of the algorithm, a subset of G's nodes are selected and the
    Boppana-Halldorsson algorithm is used to find an independent set on the subset.
    """
    # Stopping condition is reached when all nodes are either flipped on or
    # have a neighbor which is flipped on
    num_nodes = len(G.nodes)
    stopping_condition = [0 for _ in range(num_nodes)]

    # Continue until stopping condition is reached
    cur_mis_state = init_state
    history = []
    iteration = 1
    while sum(stopping_condition) < len(G.nodes):
        # Select a new initial node
        init_node = np.random.choice([node for node, val in enumerate(stopping_condition) if val == 0])
        if verbose:
            print('-'*10, 'Iteration:', iteration, '-'*10)
            print('Stopping condition:', stopping_condition)
            print('Selected node:', init_node)
            print('Current mis state:', cur_mis_state)

        # Get the neighborhood induced by the init_node and all neighbors up to
        # distance max_node_dist edges away
        induced_G = ego_graph(G, init_node, radius=max_node_dist, center=True)

        # Evaluate the Boppana-Halldorsson algorithm on the induced subgraph
        induced_G_mis = nx.algorithms.approximation.maximum_independent_set(induced_G)

        # Check to ensure a valid independent set is maintained on the full graph
        valid_indset = []
        for node in induced_G_mis:
            valid = True
            for neighbor in G[node]:
                if cur_mis_state[neighbor] == '1':
                    # This node was flipped when its neighbor was already
                    # in the independent set. Turn this node off.
                    valid = False
                    break
            if valid:
                valid_indset.append(node)

        # Update the current initial state with the result of the local search
        prev_mis_state = cur_mis_state
        if verbose:
            print('\tPrevious mis:', prev_mis_state)
        temp_mis_state = list(cur_mis_state)
        for node in valid_indset:
            temp_mis_state[node] = '1'
        cur_mis_state = ''.join(temp_mis_state)

        if verbose:
            print('\tUpdated mis: ', cur_mis_state)
            print()

        # Save results to the history
        history.append((init_node, prev_mis_state, induced_G, cur_mis_state))

        # Update the stopping condition by turning all bits to 1 if they were included
        # in the BH evaluation
        for node in induced_G.nodes:
            stopping_condition[node] = 1

        iteration += 1

    return cur_mis_state, history

