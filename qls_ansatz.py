"""
01/30/2021 - Teague Tomesh

The functions in the file are used to generate the Quantum
Local Search Ansatz (QLSA)
"""
import numpy as np
import qiskit as qk
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def gen_qlsa(circuit_G, init_state, hot_nodes, params, qubits_to_nodes, nodes_to_qubits,
             barriers=1, decompose_level=1, verbose=0):

    nq = len(qubits_to_nodes.keys())

    # Circuit construction
    circ = qk.QuantumCircuit(nq, name='q')

    # Add an ancilla qubit for implementing the mixer unitaries
    anc_reg = qk.AncillaRegister(1, 'anc')
    circ.add_register(anc_reg)

    # Initialize any qubits that have already been flipped to |1>
    # init_state lists the graph nodes in big endian order:
    #    n_0, n_1, ..., n_N
    for qubit in range(nq):
        if init_state[qubits_to_nodes[qubit]] == '1':
            circ.x(qubit)

    if barriers > 0:
        circ.barrier()

    # check the number of variational parameters
    assert (len(params) == len(hot_nodes) + 1),"Incorrect number of parameters!"

    # parse the given parameter list into alphas (for the mixers) and
    # gamma (for the drivers)
    alphas = params[:-1]
    gamma = params[-1]

    # Apply partial mixers to the hot nodes
    for alpha, hot_node in zip(alphas, hot_nodes):

        qubit = nodes_to_qubits[hot_node]
        neighbors = list(circuit_G.neighbors(hot_node))
        qubit_neighbors = [nodes_to_qubits[n] for n in neighbors]

        if verbose > 0:
            print('node:', hot_node, 'neighbors:', neighbors, 'qubit:', qubit, 'qubit neighbors:', qubit_neighbors)

        # The partial mixer unitary consists of a multi-controlled Toffoli gate, with open-controls on q's neighbors,
        # a controlled X rotation on the hot qubit, and another multi-controlled Toffoli for uncompute
        ctrl_qubits = [circ.qubits[i] for i in qubit_neighbors]
        if decompose_level > 0:
            # Qiskit has bugs when attempting to simulate custom controlled gates.
            # Instead, wrap a regular multi-controlled toffoli with X-gates targetting the ancilla qubit.
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
            circ.mcx(ctrl_qubits, circ.ancillas[0])
            for ctrl in ctrl_qubits:
                circ.x(ctrl)

            # The new AerSimulator is unable to simulate crx gates, so use a CU gate instead
            circ.cu3(2*alpha, -np.pi/2, np.pi/2, circ.ancillas[0], circ.qubits[qubit])

            # Uncompute the parity
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
            circ.mcx(ctrl_qubits, circ.ancillas[0])
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
        else:
            # Define a custom multi-controlled Toffoli
            mc_toffoli = ControlledGate('mc_toffoli', len(neighbors)+1, [], num_ctrl_qubits=len(neighbors),
                                        ctrl_state='0'*len(neighbors), base_gate=XGate())

            # Compute parity
            circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[0]])
            # Controlled Rotation
            circ.crx(2*alpha, circ.ancillas[0], circ.qubits[qubit])
            # Uncompute parity
            circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[0]])


        if barriers > 1:
            circ.barrier()

    if barriers == 1:
        circ.barrier()

    # Apply the phase separator unitary
    circ.rz(2*gamma, circ.qubits)

    if barriers >= 1:
        circ.barrier()

    if decompose_level > 1:
        basis_gates = ['rz', 'x', 'h', 'sx', 'cu3', 'mcx', 'cx', 't', 'tdg']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        circ = pm.run(circ)

    return circ
