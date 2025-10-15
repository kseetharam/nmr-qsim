from openfermion import QubitOperator, hermitian_conjugated
import openfermion as of
import cirq
import numpy as np
import sys
sys.path.append('./')
from basis_utils import Sx,Sy,Sz,S_plus,S_minus
import re
import numpy as np


###to parse data from Gaussian...
def parse_gaussian_jmatrix(path):
    """
    Parse 'Total nuclear spin-spin coupling J (Hz):' matrix from a Gaussian log file.
    Handles multi-block structure correctly (continuous rows, block column headers).
    Returns a symmetric NumPy array in Hz.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the start of the section
    start = None
    for i, line in enumerate(lines):
        if "Total nuclear spin-spin coupling J (Hz):" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("No J-coupling section found.")

    data = {}
    col_headers = []
    N = 0
    i = start

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        tokens = line.split()

        # Detect column header lines (only integers)
        if all(re.fullmatch(r"\d+", t) for t in tokens):
            col_headers = [int(t) for t in tokens]
            N = max(N, max(col_headers))
            i += 1
            continue

        # Data line (starts with row index, followed by numbers)
        if re.match(r"\s*\d+", line):
            parts = line.split()
            row_idx = int(parts[0])
            values = [float(v.replace("D", "E")) for v in parts[1:]]
            for c, v in zip(col_headers, values):
                data[(row_idx, c)] = v
                data[(c, row_idx)] = v  # symmetry
            N = max(N, row_idx)
            i += 1
            continue

        # Stop if the line doesn't fit any known pattern
        else:
            break

    # Construct full symmetric matrix
    jmat = np.zeros((N, N))
    for (r, c), v in data.items():
        jmat[r - 1, c - 1] = v
    jmat = (jmat + jmat.T) / 2.0
    return jmat


def connect_map_from_jmat(jmat,tar_nuc_idxs):

    N=  len(tar_nuc_idxs)

    graph = []
    for i in range(N):
        for j in range(i+1,N):
            graph.append((i,j,jmat[tar_nuc_idxs[i],tar_nuc_idxs[j]]))

    return graph







def multiz_gadget(qub_reg,targets,gamma: float):

    circuit = cirq.Circuit()

    for i in range(len(targets) - 1):
        circuit.append(cirq.CNOT(qub_reg[targets[i]], qub_reg[targets[i + 1]]))

        #circuit.append(cirq.Rz(targets[-1], gamma))
    circuit.append(cirq.rz(gamma)(qub_reg[targets[-1]]))

    for j in range(len(targets) - 1):
        #circuit.append(cirq.CNOT(qub_reg[targets[-j - 1]], qub_reg[targets[-j - 2]]))
        circuit.append(cirq.CNOT(qub_reg[targets[-j - 2]],qub_reg[targets[-j - 1]]))

    return circuit

def multiz_gadget_parallel(qub_reg, targets, gamma: float):
    """
    Applies a parallelized parity-check gadget: computes the parity of `targets`
    onto the last qubit, applies an Rz rotation, then uncomputes the parity.

    Args:
        qub_reg (List[cirq.Qid]): Register of qubits.
        targets (List[int]): Indices of qubits to include in the parity.
        gamma (float): Angle for the Rz rotation.

    Returns:
        cirq.Circuit: Circuit implementing the gadget.
    """
    circuit = cirq.Circuit()
    active = [qub_reg[i] for i in targets]

    forward_layers = []

    # Forward pass: build CNOT tree
    while len(active) > 1:
        layer = []
        new_active = []

        for i in range(0, len(active) - 1, 2):
            control = active[i]
            target = active[i + 1]
            layer.append(cirq.CNOT(control, target))
            new_active.append(target)

        # If odd number of qubits, carry the last one forward
        if len(active) % 2 == 1:
            new_active.append(active[-1])

        forward_layers.append(layer)
        active = new_active

    # Apply Rz to the final target (holds the parity)
    final_target = active[0]
    circuit.append(forward_layers)
    circuit.append(cirq.rz(gamma)(final_target))

    # Reverse pass: uncompute the parity
    for layer in reversed(forward_layers):
        circuit.append(layer[::-1])  # reverse gate order in each layer

    return circuit

def pauli_basis_change(qub_reg,targets, start: str, end: str):
    # assert len(targets) == len(start)
    # assert len(targets) == len(end)


    circuit = cirq.Circuit()
    # for qubit, start_pauli, end_pauli in zip(targets, start, end):
    for i in range(len(targets)):
        qubit = qub_reg[targets[i]]
        start_pauli = start[i]
        end_pauli = end[i]

        target = start_pauli + end_pauli
        if target == "ZX":
            circuit.append(cirq.ry(np.pi/2)(qubit))
        
        elif target == "ZY":
            circuit.append(cirq.rx(-np.pi/2)(qubit))
            
        elif target == "XY":
            circuit.append(cirq.rz(np.pi/2)(qubit))
            
        elif target == "XZ":
            circuit.append(cirq.ry(-np.pi/2)(qubit))
            
        elif target == "YX":
            circuit.append(cirq.rz(-np.pi/2)(qubit))
          
        elif target == "YZ":
            circuit.append(cirq.rx(np.pi/2)(qubit))
            
    return circuit

def pauli_exponential(qub_reg, targets, pauli: str, gamma: float,parallel=True):
    # assert len(targets) == len(pauli)
    circuit = cirq.Circuit()

    circuit.append(pauli_basis_change(qub_reg,targets=targets, start= "Z" * len(targets), end=pauli))

    #pauli_basis_change(targets=targets, start="Z" * len(targets), end=pauli)
    if parallel:
        circuit.append(multiz_gadget_parallel(qub_reg,targets,gamma))
    else:
        circuit.append(multiz_gadget(qub_reg,targets,gamma))
    
    circuit.append(pauli_basis_change(qub_reg,targets=targets, start=pauli, end="Z" * len(targets)))

    return circuit

def compile_group(qub_reg,qub_op: QubitOperator):
    
    circuit = cirq.Circuit()
    
    for term, coeff in qub_op.terms.items():
    
        sorted_term = tuple(sorted(term, key=lambda x: x[1],reverse=True))  # sort by qubit index
        #print(sorted_term)
        # Convert to readable Pauli string
        #pauli_str = ' '.join(f'{p}{q}' for p, q in sorted_term)
        pauli_str = ''.join(f'{q}' for p,q in sorted_term)
        qargs = np.array([p for p,q in sorted_term])

        sort_indices = np.argsort(qargs)  

        sorted_pauli_str = ''.join(pauli_str[i] for i in sort_indices)

        circuit.append(pauli_exponential(qub_reg, qargs[sort_indices],sorted_pauli_str, 2*coeff)) #The factor of 2 is added to 
                                                                                                    #counteract the division by 2 in the underlying compilation to R_{a} axes
        #print(f"{pauli_str if pauli_str else 'I'} : {coeff}")
        #print("qargs are:", qargs)
    
    #in the CZ+PhXZ gate set
    return cirq.optimize_for_target_gateset(cirq.Circuit(circuit),gateset=cirq.CZTargetGateset())

    #return circuit



def compile_var_op(qub_reg,qub_op: QubitOperator, sym,parallel=True):
    circuit = cirq.Circuit()

    for term, coeff in qub_op.terms.items():
    
        sorted_term = tuple(sorted(term, key=lambda x: x[1],reverse=True))  # sort by qubit index
        #print(sorted_term)
        # Convert to readable Pauli string
        #pauli_str = ' '.join(f'{p}{q}' for p, q in sorted_term)
        pauli_str = ''.join(f'{q}' for p,q in sorted_term)
        qargs = np.array([p for p,q in sorted_term])

        sort_indices = np.argsort(qargs)  

        sorted_pauli_str = ''.join(pauli_str[i] for i in sort_indices)
        #print(f"Sorted Pauli string: {sorted_pauli_str}")

        circuit.append(pauli_exponential(qub_reg, qargs[sort_indices],sorted_pauli_str, sym*coeff,parallel=parallel))

        #print(f"{pauli_str if pauli_str else 'I'} : {coeff}")
        #print("qargs are:", qargs)
    
    return circuit

def get_dil_Ham(hamiltonian,deltaT,ListJumpOps):
    """
    Construct the dilated Hamiltonian
    Args:
    hamiltonian: target coherent Hamiltonian in openfermion's format
    deltaT: time step considered for a first-order Trotter formula
    ListJumpOps: list of scaled jump operators \sqrt{\kappa}*L_{i}, that define the different dissipation channels
    """
    theta = np.sqrt(deltaT)
    nqubs = of.count_qubits(hamiltonian)

    nancs = len(ListJumpOps) ###one ancilla qubit per jump operator...

    anc_idxs = [i+nqubs for i in range(nancs)]
    dil_Ham = of.QubitOperator()
    dil_Ham+=theta*hamiltonian

    counter=0
    for anc_idx in anc_idxs:

        proj1 = S_plus(anc_idx)
        proj2 = S_minus(anc_idx)

        dil_Ham+=proj1*ListJumpOps[counter]
        dil_Ham+=proj2*hermitian_conjugated(ListJumpOps[counter])
        counter+=1

    return dil_Ham


def generate_heisenberg_hamiltonian(h_list, coupling_graph):
    """
    Generate a Heisenberg Hamiltonian:
        H = sum_i h_i Sz_i + sum_{i,j} J_{ij} (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)

    Parameters:
        h_list: list of local Z field coefficients, h_i
        coupling_graph: list of tuples (i, j, J_ij), the spin coupling graph

    Returns:
        QubitOperator representing the Hamiltonian
    """
    ham = QubitOperator()

    # Local Z field terms
    for i, h_i in enumerate(h_list):
        if h_i != 0:
            ham += h_i*Sz(i)

    # Interaction terms
    for i, j, Jij in coupling_graph:
        #for op in ['X', 'Y', 'Z']:
            #term = QubitOperator(f'{op}{i} {op}{j}', Jij )
        term = Jij * (Sx(i)*Sx(j)+Sy(i)*Sy(j)+Sz(i)*Sz(j))
        ham += term

    return ham

###Function to compile Trotterized Heisenberg Hamiltonian with dissipation for Nsteps
def FirstOrderTrotterEvol(sys_qubit_reg,anc_qub_reg, dil_Ham,deltaT, n_steps=1, parallel=True):
    """
    Compiles a first-order Trotter evolution circuit for a given dilated Hamiltonian, for n_steps 
    sys_qubit_reg: cirq LineQubit register for target system spins
    anc_qub_reg: cirq LineQubit register for ancilla qubits
    dil_Ham: openfermion Qubit operator that encodes the dilated Hamiltonian
    deltaT: the target time step to simulate
    n_steps: number of Trotter steps
    parallel: compile the Pauli exponentiation gadgets using parallel CNOT gates
    """
    circuit = cirq.Circuit()
    #n_ancs = len(anc_qub_reg)

    Trot_step = compile_var_op(sys_qubit_reg+anc_qub_reg, dil_Ham, 2.0*np.sqrt(deltaT), parallel=parallel) #The factor of 2 is to "counteract" the division by 2 in cirq.Rz gates during compilation
    #Trot_step.append(cirq.measure(*anc_qub_reg, key=key))
    Trot_step.append([cirq.ResetChannel()(q) for q in anc_qub_reg])

    for i in range(n_steps):
        # Compile the dilated Hamiltonian for each step
        circuit.append(Trot_step)
        
        # Add a small time evolution step
        
    return circuit






