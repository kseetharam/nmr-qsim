import numpy as np
import cirq
from cirq import ops
from copy import copy, deepcopy
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import qiskit

# ---- HELPER FUNCTIONS ----

def basisStates(Nspin):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in both integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) and as vectors
    """
#     integerList = range(0,2**Nspin)
    integerList = [i for i in range(2**Nspin-1,-1,-1)]
    vectorList = []
    for i in integerList:
        bVec = [0]*(2**Nspin); bVec[int(2**Nspin)-i-1] = 1
        vectorList.append(bVec)
    return integerList, vectorList


def SzTot_weighted_obs(Nspin, weights):
    spinBasis = spin_basis_1d(Nspin, pauli=False)
    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(Nspin)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False) 
    return np.diag(ham_Rz.todense())


def SzTot_expectation(state_prob, Sz_Tot):
    """
    Args: state_prob = list of occupation probabilities (average population) in the computational basis for a state |\psi>. Specifically, is the vector |<psi|\psi>|^2 expressed in the computational basis. This vector can either be directly computed from the state vector of a simulation output or estimated from the histogram of counts resulting from a set of experimental shots. It is assumed that the basis ordering is consistent with the functions basisStates(Nspin) and SzTot_obs(Nspin).
    Note: state_prob can also be a list of lists, each which correspond to the vector |<psi|\psi>|^2 for a different state |\psi>.
    Returns: expectation value of total z-magnetization (SzTot) for each state |\psi>.
    """
    state_prob_array = np.array(state_prob)
    if state_prob_array.ndim > 1:
        Nspin = int(np.log2(len(state_prob[0])))
        return np.sum(np.multiply(Sz_Tot[:, None], np.transpose(state_prob_array)), axis=0)
    else:
        Nspin = int(np.log2(len(state_prob)))
        return np.dot(Sz_Tot, state_prob_array)


def responseFunc_sample(state_prob_list, SzTot, SzTot_comp, Nspin):
    """
    Args:
    state_prob_list = list of state_prob lists, each such list corresponds to a different positive magnetization basis state that the system was initialized in, with each of these states then evolved for the same sample time t
    SzTot_comp = list of SzTot (positive non-zero) magnetization values corresponding to the initial state of each list in state_prob_list
    Returns: sample of response function S(t) evaluated at the sample time t that all the lists in state_prob_list were evaluated at
    """
    return 2 * np.sum(SzTot_expectation(state_prob_list, SzTot) * SzTot_comp / (2**Nspin))


def basisStates_positiveSz_ZULF(Nspin,SzTot):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) which have postive non-zero magnetization, and a list of their corresponding magnetization
    """
    basisList_integer, basisList_vector = basisStates(Nspin)  # gets list of basis states both in integer encoding and vector format
    magMask = np.logical_not(np.isclose(SzTot, 0.0, atol=1e-3)) * (SzTot > 0.0)
    return np.array(basisList_integer)[magMask], SzTot[magMask]
        
               
def state_preparation(qubits_to_flip, encoding: int = 0) -> cirq.Circuit:
    circuit = qiskit.QuantumCircuit(qubits_to_flip)
    n_qubits = len(qubits_to_flip)
    benc = format(2**n_qubits - 1 - encoding, '0{:d}b'.format(n_qubits))
    for i in range(n_qubits):
        if benc[i] == '1':
            circuit.x(qubits_to_flip[i]) 
    return circuit

def ZULF_circuitList(t_ind, path, SzTot):
    """
    Args: t_ind = evolution time of unitary, path = path to folder containing pickled Cirq circuits , qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    qubit_reg = qiskit.QuantumRegister(4,'q')
    
    if t_ind != 0:
        filename = path + 'U_t_{:d}.qasm'.format(int(t_ind))
        evolution_circuit = qiskit.QuantumCircuit.from_qasm_file(filename)
    
    basisList_comp, SzTot_comp = basisStates_positiveSz_ZULF(len(qubit_reg), SzTot)
#     print(len(basisList_comp), len(SzTot_comp))
    circuitList = []
    for indb, bint in enumerate(basisList_comp):
        circuit = state_preparation(qubit_reg, bint)
        if t_ind != 0:
            circuit = circuit.compose(evolution_circuit)
        circuitList.append(circuit)
    return circuitList, SzTot_comp.tolist()


# ---- SIMULATIONS ----

def ZULF_noiselessSim(t_ind_List, path, weights):
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            # Simulation (noiseless)
            state = qiskit.quantum_info.Statevector.from_int(0, 2**4)
            circuit = circuit.reverse_bits()
            vec = state.evolve(circuit).data
            avg_pop = np.abs(vec)**2
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, 4))
    return responseFunc


def ZULF_snSim(t_ind_List, path, weights, repetitions, backend):
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, SzTot)
        state_prob_list = []
        for indc, circuit in enumerate(circuitList):
            measurement = qiskit.QuantumCircuit(4,4)
            measurement.barrier(range(4))
            measurement.measure(range(4),range(4))
            circuit = measurement.compose(circuit, range(4), front=True)
            circuit = circuit.reverse_bits()
            circuit_compiled = qiskit.transpile(circuit, backend)
            job_sim = backend.run(circuits=circuit_compiled, job_name='basis_{:d}_U_t_{:d}'.format(int(indc), int(t_ind)), shots=repetitions)
            results = job_sim.result()      
            counts = results.get_counts()
            avg_pop = counts_to_prob(counts, repetitions, 4)
            state_prob_list.append(avg_pop)
        print(t_ind)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, 4))
    return responseFunc


def counts_to_prob(counts, repetitions, num_qubits):
        prob = np.zeros(2**num_qubits)
        possibleKeys = [format(i, '0{:d}b'.format(num_qubits)) for i in range(2**num_qubits)]
        keys = counts.keys()
        for ind, key in enumerate(possibleKeys):
            if key in keys:
                prob[ind] = counts[key]/repetitions
            else:
                prob[ind] = 0
        return prob
    

# ---- PROCESS EXPERIMENTAL DATA ----

def ZULF_expData(t_ind_List, path, weights):
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    SzTot_comp = np.array([1.6257476076555024, 1.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976])
    for indt, t_ind in enumerate(t_ind_List):
        state_prob_list = []
        for indb, b in enumerate(SzTot_comp):
            with open(path + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,t_ind)) as file:
                result_file = yaml.full_load(file)
                result = result_file['result']
#                 repetitions = result_file['shots']
                avg_pop = result
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, 4))
    return responseFunc

def ZULF_ibmExp(t_ind, path, weights, repetitions, backend):
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, SzTot)
    job_list = []
    for indc, circuit in enumerate(circuitList):
        measurement = qiskit.QuantumCircuit(4,4)
        measurement.barrier(range(4))
        measurement.measure(range(4),range(4))
        circuit = measurement.compose(circuit, range(4), front=True)
        circuit = circuit.reverse_bits()
        circuit_compiled = qiskit.transpile(circuit, backend)
        job = backend.run(circuits=circuit_compiled, job_name='U_t_{:d}_basis_{:d}'.format(int(t_ind), int(indc)), shots=repetitions)
        job_list.append(job)
    return job_list

def ZULF_ibmDataProc(t_ind_List, weights):
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    SzTot_comp = np.array([1.6257476076555024, 1.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976])
    for indt, t_ind in enumerate(t_ind_List):
        state_prob_list = []
        for indb, b in enumerate(SzTot_comp):
            avg_pop = np.loadtxt('IBM_data/U_t_{:d}_basis_{:d}.txt'.format(int(t_ind),int(indb)))
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, 4))
    return responseFunc

def ZULF_ibmExp_circuitStats(t_ind, path, weights, repetitions, backend):
    SzTot = np.array(SzTot_weighted_obs(4, weights))
    circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, SzTot)
    stats_list = []
    for indc, circuit in enumerate(circuitList):
        measurement = qiskit.QuantumCircuit(4,4)
        measurement.barrier(range(4))
        measurement.measure(range(4),range(4))
        circuit = measurement.compose(circuit, range(4), front=True)
        circuit = circuit.reverse_bits()
        circuit_compiled = qiskit.transpile(circuit, backend)
        stats_list.append(circuit_compiled.count_ops())
    return stats_list