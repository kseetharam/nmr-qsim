import numpy as np
import cirq
from cirq import ops
from copy import copy, deepcopy
import pickle
import yaml
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

# ---- HELPER FUNCTIONS ----

def spinParams_to_pauliParams(Jij):
    """
    Args: Jij = matrix of Hamiltonian parameters defined for spin operators
    Returns: Jij_pauli = matrix of Hamiltonian parameters adjusted for Pauli operators (Jij -> Jij/4 for i!=j)
    """
    Jij_pauli = deepcopy(Jij)
    matShape = np.array(Jij).shape
    for i in range(matShape[0]):
        for j in range(matShape[1]):
            if i != j:
                Jij_pauli[i][j] = Jij[i][j]/4
            else:
                Jij_pauli[i][j] = Jij[i][j]
    return Jij_pauli

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

def SzTot_obs(Nspin):
    """
    Args: Ns = number of spins in the system
    Returns: list containing values of the total z-magnetization for each basis vector in the spin basis. Note that SzTot is diagonal in the basis.
    Notes: total z-magnetization is (N_up - N_down)/2 = (N_up - (L - N_up))/2.
    Encoding is such that the first element of the returned list corresponds to the maximal magnetization state (all +Z) while the last element corresponds to the all -Z state
    Magnetization in returned list corresponds to basis state order in basisStates(Nspin)
    """
    return [bin(bint).count("1") - (Nspin / 2) for bint in range(2**Nspin-1,-1,-1)]

def SzTot_weighted_obs(Nspin, weights):
    spinBasis = spin_basis_1d(Nspin, pauli=False)
    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(Nspin)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False) 
    return np.diag(ham_Rz.todense())

def basisStates_positiveSz_ZULF(Nspin,SzTot):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) which have postive non-zero magnetization, and a list of their corresponding magnetization
    """
    basisList_integer, basisList_vector = basisStates(Nspin)  # gets list of basis states both in integer encoding and vector format
    magMask = np.logical_not(np.isclose(SzTot, 0.0, atol=1e-3)) * (SzTot > 0.0)
    return np.array(basisList_integer)[magMask], SzTot[magMask]


def basisStates_positiveSz(Nspin):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) which have postive non-zero magnetization, and a list of their corresponding magnetization
    """
    basisList_integer, basisList_vector = basisStates(Nspin)  # gets list of basis states both in integer encoding and vector format
    SzTot = np.array(SzTot_obs(Nspin))  # gets total Sz magnetization for each basis state
    magMask = np.logical_not(np.isclose(SzTot, 0.0, atol=1e-3)) * (SzTot > 0.0)
    return np.array(basisList_integer)[magMask], SzTot[magMask]

# def hist_To_probVec(histogram, repetitions, Nspin):
#     ave_prob = np.array([histogram[i] for i in range(int(2**Nspin-1))])/repetitions
#     return ave_prob


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
        
               
def state_preparation(qubits_to_flip, encoding: int = 0) -> cirq.Circuit:
    """Assume qubits are all in the ground state, will prepare them in the Z basis according to provided encoding
    Args:
        circuit: circuit to add Z preparation to
        qubits_to_flip: list of cirq.LineQubits to perform the encoding on
        encoding: an integer whos binary coding encodes the binary string for the Z basis of the qubits.
            encoding = 15 -> +Z+Z+Z+Z
            encoding = 0 -> -Z-Z-Z-Z
    Returns: cirq circuit with added operation
    """
    n_qubits = len(qubits_to_flip)
    benc = format(2**n_qubits - 1 - encoding, '0{:d}b'.format(n_qubits))
    for i in range(n_qubits):
        if benc[i] == '1':
            yield cirq.rx(np.pi).on(qubits_to_flip[i]) 
        else:
            yield cirq.rx(0).on(qubits_to_flip[i]) 
            
def ZULF_circuitList(t_ind, path, qubit_reg, SzTot,json_format=True):
    """
    Args: t_ind = evolution time of unitary, path = path to folder containing pickled Cirq circuits , qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    if t_ind == 0:
        mapped_evolution_circuit = []
    else:
        if json_format:
            if t_ind.is_integer():
                filename = path + 'U_t_{:d}.json'.format(int(t_ind))
            else:
                filename = path + 'U_t_{:.1f}.json'.format(t_ind)
        else:
            if t_ind.is_integer():

                filename = path + 'U_t_{:d}.pickle'.format(int(t_ind))
            else:
                filename = path + 'U_t_{:.1f}.pickle'.format(t_ind)
        evolution_circuit = load_circuit(filename,json_format=json_format)
#         evo_qubits = evolution_circuit.all_qubits()
        evo_qubits = [cirq.NamedQubit('q_{:d}'.format(i)) for i in range(len(qubit_reg))]
#         if len(evo_qubits) != len(qubit_reg):
#             print('NOT ALL QUBITS USED', t_ind, len(evo_qubits))
        #     map the generic qubits that evolution_circuit operators on to the provided qubit register
        sorted_evo_qubits = sorted(evo_qubits, key=lambda x: x.name, reverse=False)
        qmap = dict(zip(sorted_evo_qubits, qubit_reg[0:len(evo_qubits)])) 
        mapped_evolution_circuit = evolution_circuit.transform_qubits(lambda q: qmap[q])
    
    basisList_comp, SzTot_comp = basisStates_positiveSz_ZULF(len(qubit_reg), SzTot)
#     print(len(basisList_comp), len(SzTot_comp))
    circuitList = []
    for indb, bint in enumerate(basisList_comp):
#         print(indb, bint, SzTot_comp[indb]); print(cirq.Circuit(state_preparation(qubit_reg, bint)))
        circuit = cirq.Circuit()
#         circuit.__name__ = "NMR_tind_{:d}_bind_{:d}".format(t_ind, bint)
        circuit.__name__ = "NMR_tind_{:.1f}_bind_{:d}".format(t_ind, bint)
        circuit.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(mapped_evolution_circuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList.append(circuit)
    return circuitList, SzTot_comp.tolist()

def fill_circuit(qubit_reg, delta_2q) -> cirq.Circuit:
    i = 0; j = 1
    for n in np.arange(delta_2q//2):
#         print(i,j)
        qubit_i = qubit_reg[i]; qubit_j = qubit_reg[j] 
        yield cirq.ms(np.pi/4).on(qubit_i,qubit_j)
        yield cirq.rz(np.pi/2).on(qubit_i)
        yield cirq.rz(np.pi/2).on(qubit_j)
        yield cirq.rx(np.pi/2).on(qubit_i)
        yield cirq.rx(np.pi/2).on(qubit_j)
        yield cirq.rx(-np.pi/2).on(qubit_i)
        yield cirq.rx(-np.pi/2).on(qubit_j)
        yield cirq.rz(-np.pi/2).on(qubit_i)
        yield cirq.rz(-np.pi/2).on(qubit_j)
        yield cirq.ms(-np.pi/4).on(qubit_i,qubit_j)
        i+=1; j+=1;
        i=i%4; j=j%4

def ZULF_circuitList_fill(t_ind, path, qubit_reg, SzTot):
    """
    Args: t_ind = evolution time of unitary, path = path to folder containing pickled Cirq circuits, qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    if t_ind == 0:
        mapped_evolution_circuit = []
    else:
        if t_ind.is_integer():
            filename = path + 'U_t_{:d}.pickle'.format(int(t_ind))
        else:
            filename = path + 'U_t_{:.1f}.pickle'.format(t_ind)
        evolution_circuit = load_circuit(filename)
#         evo_qubits = evolution_circuit.all_qubits()
        evo_qubits = [cirq.NamedQubit('q_{:d}'.format(i)) for i in range(len(qubit_reg))]
#         if len(evo_qubits) != len(qubit_reg):
#             print('NOT ALL QUBITS USED', t_ind, len(evo_qubits))
        #     map the generic qubits that evolution_circuit operators on to the provided qubit register
        sorted_evo_qubits = sorted(evo_qubits, key=lambda x: x.name, reverse=False)
        qmap = dict(zip(sorted_evo_qubits, qubit_reg[0:len(evo_qubits)])) 
        mapped_evolution_circuit = evolution_circuit.transform_qubits(lambda q: qmap[q])
    
    ms_c = total_ms_count(mapped_evolution_circuit)
    rp_c, rz_c = total_1q_count(mapped_evolution_circuit)
    delta_2q = 45 - ms_c
    delta_1q = 196 - (rp_c+rz_c)
#     print(delta_2q, delta_1q, 4*delta_2q)
#     print(delta_2q, delta_2q//2)
    fcircuit = cirq.Circuit(fill_circuit(qubit_reg, delta_2q))
    ms_f = total_ms_count(fcircuit)
    rp_f, rz_f = total_1q_count(fcircuit)
#     print(ms_c+ms_f,rp_c+rz_c+rp_f+rz_f)
    
    basisList_comp, SzTot_comp = basisStates_positiveSz_ZULF(len(qubit_reg), SzTot)
    circuitList = []
    for indb, bint in enumerate(basisList_comp):
        circuit = cirq.Circuit()
        circuit.__name__ = "NMR_tind_{:.1f}_bind_{:d}".format(t_ind, bint)
        circuit.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(mapped_evolution_circuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(fcircuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList.append(circuit)

    return circuitList, SzTot_comp.tolist()


def COSY_circuitList(t1_ind, t2_ind, path, qubit_reg):
    """
    Args: t_ind = evolution time of unitary, path = path to folder containing pickled Cirq circuits , qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    evo_qubits = [cirq.NamedQubit('q_{:d}'.format(i)) for i in range(len(qubit_reg))]
    sorted_evo_qubits = sorted(evo_qubits, key=lambda x: x.name, reverse=False)
    qmap = dict(zip(sorted_evo_qubits, qubit_reg[0:len(evo_qubits)])) 
    
    if t1_ind == 0:
        mapped_evolution_circuit_1 = []
    else:
        filename1 = path + 'U_t_{:d}.pickle'.format(t1_ind)
        evolution_circuit_1 = load_circuit(filename1)
        mapped_evolution_circuit_1 = evolution_circuit_1.transform_qubits(lambda q: qmap[q])
    if t2_ind == 0:
        mapped_evolution_circuit_2 = []
    else:
        filename2 = path + 'U_t_{:d}.pickle'.format(t2_ind)
        evolution_circuit_2 = load_circuit(filename2)
        mapped_evolution_circuit_2 = evolution_circuit_2.transform_qubits(lambda q: qmap[q])
    
    basisList_comp, SzTot_comp = basisStates_positiveSz(len(qubit_reg))
    circuitList_re = []
    circuitList_im = []
    for indb, bint in enumerate(basisList_comp):
        circuit_re = cirq.Circuit()
        circuit_re.__name__ = "COSY_re_t1_{:d}_t2_{:d}_bind_{:d}".format(t1_ind, t2_ind, bint)
        circuit_re.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit_re.append(mapped_evolution_circuit_1, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit_re.append([cirq.rx(-1*np.pi/2).on(qubit) for qubit in qubit_reg])
        circuit_re.append([cirq.rz(-1*np.pi/2).on(qubit) for qubit in qubit_reg])
        circuit_re.append(mapped_evolution_circuit_2, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList_re.append(circuit_re)
        
        circuit_im = cirq.Circuit()
        circuit_im.__name__ = "COSY_im_t1_{:d}_t2_{:d}_bind_{:d}".format(t1_ind, t2_ind, bint)
        circuit_im.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit_im.append(mapped_evolution_circuit_1, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit_im.append([cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg])
        circuit_im.append(mapped_evolution_circuit_2, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList_im.append(circuit_im)
        
    return circuitList_re, circuitList_im, SzTot_comp.tolist()

def save_circuit(circuit: cirq.Circuit, filename: str):
    """Save the given circuit as a pickle file.
    """
    with open(str(filename), "wb") as file:
        pickle.dump(circuit, file)
    return


def load_circuit(filename: str,json_format=True):
    """Load a Cirq circuit from a pickle file."""
    if json_format:
        circuit = cirq.read_json(filename)
    else:

        with open(str(filename), "rb") as file:
            circuit = pickle.load(file)
    return circuit

# ---- MISC CIRCUIT FUNCTIONS ----

def ZULF_circuitStats_ion(t_ind_List, path, nmr_reg, weights, gateTime_1q, gateTime_2q):
    tot_count = []
    ms_count = []
    rp_count = []
    expTime = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        ms = []; rp = []; rz = []
        for circuit in circuitList:
            ms_c = total_ms_count(circuit); ms.append(ms_c)
            rp_c, rz_c = total_1q_count(circuit); rp.append(rp_c); rz.append(rz_c)
        ms_ave = np.average(ms); rp_ave = np.average(rp); rz_ave = np.average(rz)
        ms_count.append(ms_ave)
        rp_count.append(rp_ave)
        tot_count.append(ms_ave + rp_ave + rz_ave)
        expTime.append(ms_ave*gateTime_2q + rp_ave*gateTime_1q)
    return np.array(tot_count).astype(int), expTime, np.array(rp_count).astype(int), np.array(ms_count).astype(int)


def ZULF_circuitStats_sc(t_ind_List, path, nmr_reg, weights, gateTime_1q, gateTime_2q):
    tot_count = []
    cz_count = []
    rp_count = []
    expTime = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        cz = []; rp = []; rz = []
        for circuit in circuitList:
            cz_c = total_cz_count(circuit); cz.append(cz_c)
            rp_c, rz_c = total_1q_count(circuit); rp.append(rp_c); rz.append(rz_c)
        cz_ave = np.average(cz); rp_ave = np.average(rp); rz_ave = np.average(rz)
        cz_count.append(cz_ave)
        rp_count.append(rp_ave)
        tot_count.append(cz_ave + rp_ave + rz_ave)
        expTime.append(cz_ave*gateTime_2q + rp_ave*gateTime_1q)
    return tot_count, expTime, rp_count, cz_count

def total_ms_count(circuit: cirq.Circuit):
    total_ms = 0
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            assert len(op.qubits) <= 2
            if len(op.qubits) == 2:
                assert isinstance(op, cirq.GateOperation)
                assert isinstance(op.gate, cirq.XXPowGate)
                total_ms += 1
    return total_ms

def total_cz_count(circuit: cirq.Circuit):
    total_cz = 0
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            assert len(op.qubits) <= 2
            if len(op.qubits) == 2:
                assert isinstance(op, cirq.GateOperation)
                assert isinstance(op.gate, cirq.CZPowGate)
                total_cz += 1
    return total_cz


def total_1q_count(circuit: cirq.Circuit):
    total_1q_p = 0
    total_1q_z = 0
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            if len(op.qubits) == 1:
                if isinstance(op.gate, ops.ZPowGate):
                    total_1q_z += 1
                else:
                    total_1q_p += 1
    return total_1q_p, total_1q_z


class depolarizingNoiseModel(cirq.NoiseModel):
    def __init__(self, p1q: float, p2q: float) -> None:
        self._p1q = p1q
        self._p2q = p2q
    def noisy_operation(self, op):
        q = op.qubits
        num_qubits = len(q)
        if num_qubits == 1:            
            yield op
            yield cirq.depolarize(p=self._p1q,n_qubits=1).on(*q)
        elif num_qubits == 2:
            yield op
            yield cirq.depolarize(p=self._p2q,n_qubits=2).on(*q)

            
def addDepolarizingNoise(circuit, p1q, p2q):
#     adds depolarizing channels to single qubit gates with probability p1q and two qubit gates with probability p2q
    new_circuit = cirq.Circuit()
    new_circuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    noise_model = depolarizingNoiseModel(p1q,p2q)
    for moment in circuit[1::]:
        for op in moment.operations:
            noisy_op = noise_model.noisy_operation(op)
            new_circuit.append(noisy_op)
#             noisy_op = noise_model(op)
#             new_circuit.append(op)
#             if len(op.qubits) == 1:
#                 new_circuit.append(cirq.depolarize(p=p1q,n_qubits=1).on(*op.qubits))
#             elif len(op.qubits) == 2:
#                 new_circuit.append(cirq.depolarize(p=p2q,n_qubits=2).on(*op.qubits))
    return new_circuit

def addAmplitudeDampingNoise(circuit, p):
#     adds amplitude or phase damping channels to each gate with probability p
    new_circuit = cirq.Circuit()
    new_circuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    for moment in circuit[1::]:
        for op in moment.operations:
            new_circuit.append(op)
            for qubit in op.qubits:
                new_circuit.append(cirq.amplitude_damp(p).on(qubit))
    return new_circuit

def addPhaseDampingNoise(circuit, p):
#     adds amplitude or phase damping channels to each gate with probability p
    new_circuit = cirq.Circuit()
    new_circuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    for moment in circuit[1::]:
        for op in moment.operations:
            new_circuit.append(op)
            for qubit in op.qubits:
                new_circuit.append(cirq.phase_damp(p).on(qubit))
    return new_circuit

def addAmpPhaseNoise(circuit, p_amp, p_phase):
#     adds amplitude and phase damping channels to each gate with probabilities p_amp and p_phase respectively
    new_circuit = cirq.Circuit()
    new_circuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    for moment in circuit[1::]:
        for op in moment.operations:
            new_circuit.append(op)
            if len(op.qubits) == 2: 
                for qubit in op.qubits:
                    new_circuit.append(cirq.amplitude_damp(p_amp).on(qubit))
                    new_circuit.append(cirq.phase_damp(p_phase).on(qubit))
    return new_circuit

# ---- SIMULATIONS ----

def ZULF_noiselessSim(t_ind_List, path, nmr_reg, weights,json_format=True):
    simulator = cirq.Simulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot,json_format=json_format)
        state_prob_list = []
        for circuit in circuitList:
            # Simulation (noiseless)
            vec = simulator.simulate(circuit).final_state_vector
            avg_pop = np.abs(vec)**2
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_noisySim(t_ind_List, path, nmr_reg, weights, repetitions, p1q, p2q):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addDepolarizingNoise(circuit, p1q, p2q)
            noisy_circuit.append([cirq.measure(*nmr_reg, key='measure_all')])
            run_result = simulator.run(program=noisy_circuit, repetitions=repetitions)
            histogram = run_result.histogram(key='measure_all')
            avg_pop = np.array([histogram[i] for i in range(int(2**len(nmr_reg)))])/repetitions
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_snSim(t_ind_List, path, nmr_reg, weights, repetitions):
    simulator = cirq.Simulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            circuit.append([cirq.measure(*nmr_reg, key='measure_all')])
            run_result = simulator.run(program=circuit, repetitions=repetitions)
            histogram = run_result.histogram(key='measure_all')
            avg_pop = np.array([histogram[i] for i in range(int(2**len(nmr_reg)))])/repetitions
#             print(histogram)
#             print(avg_pop*repetitions)
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_noisySim_MC(t_ind_List, path, nmr_reg, weights, repetitions, p1q, p2q):
    simulator = cirq.Simulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addDepolarizingNoise(circuit, p1q, p2q)
            noisy_circuit.append([cirq.measure(*nmr_reg, key='measure_all')])
            run_result = simulator.run(program=noisy_circuit, repetitions=repetitions)
            histogram = run_result.histogram(key='measure_all')
            avg_pop = np.array([histogram[i] for i in range(int(2**len(nmr_reg)))])/repetitions
#             print(histogram)
#             print(avg_pop*repetitions)
            state_prob_list.append(avg_pop)
        print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_decohSim(t_ind_List, path, nmr_reg, weights, p1q, p2q):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addDepolarizingNoise(circuit, p1q, p2q)
            rho = simulator.simulate(noisy_circuit).final_density_matrix
            avg_pop = np.abs(np.diag(rho))
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_decohSim_fill(t_ind_List, path, nmr_reg, weights, p1q, p2q):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList_fill(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addDepolarizingNoise(circuit, p1q, p2q)
            rho = simulator.simulate(noisy_circuit).final_density_matrix
            avg_pop = np.abs(np.diag(rho))
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc


def ZULF_dampSim(t_ind_List, path, nmr_reg, weights, p):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addAmplitudeDampingNoise(circuit, p)
#             noisy_circuit = addPhaseDampingNoise(circuit, p)
            rho = simulator.simulate(noisy_circuit).final_density_matrix
            avg_pop = np.abs(np.diag(rho))
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_AmpPhaseSim(t_ind_List, path, nmr_reg, weights, p_amp, p_phase):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addAmpPhaseNoise(circuit, p_amp, p_phase)
            rho = simulator.simulate(noisy_circuit).final_density_matrix
            avg_pop = np.abs(np.diag(rho))
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc

def ZULF_AmpPhaseSim_fill(t_ind_List, path, nmr_reg, weights, p_amp, p_phase):
    simulator = cirq.DensityMatrixSimulator()
    responseFunc = []
    SzTot = np.array(SzTot_weighted_obs(len(nmr_reg), weights))
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = ZULF_circuitList_fill(t_ind, path, nmr_reg, SzTot)
        state_prob_list = []
        for circuit in circuitList:
            noisy_circuit = addAmpPhaseNoise(circuit, p_amp, p_phase)
            rho = simulator.simulate(noisy_circuit).final_density_matrix
            avg_pop = np.abs(np.diag(rho))
            state_prob_list.append(avg_pop)
#         print(indt)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot, SzTot_comp, len(nmr_reg)))
    return responseFunc


def COSY_noiselessSim(t_ind_List, path, nmr_reg):
    simulator = cirq.Simulator()
    responseFunc_re = []
    responseFunc_im = []
    SzTot = np.array(SzTot_obs(len(nmr_reg)))
    for indt2, t2_ind in enumerate(t_ind_List):
        for indt1, t1_ind in enumerate(t_ind_List):
            circuitList_re, circuitList_im, SzTot_comp = COSY_circuitList(t1_ind, t2_ind, path, nmr_reg)
            state_prob_list_re = []
            for circuit in circuitList_re:
                # Simulation (noiseless)
                vec = simulator.simulate(circuit).final_state_vector
                avg_pop = np.abs(vec)**2
                state_prob_list_re.append(avg_pop)
            responseFunc_re.append(responseFunc_sample(state_prob_list_re, SzTot, SzTot_comp, len(nmr_reg)))
            
            state_prob_list_im = []
            for circuit in circuitList_im:
                # Simulation (noiseless)
                vec = simulator.simulate(circuit).final_state_vector
                avg_pop = np.abs(vec)**2
                state_prob_list_im.append(avg_pop)
            responseFunc_im.append(responseFunc_sample(state_prob_list_im, SzTot, SzTot_comp, len(nmr_reg)))
        print(path[-3:-1], t2_ind)

    responsefunc_complex = np.array(responseFunc_re) + 1j * np.array(responseFunc_im)
    fid = responsefunc_complex.reshape((len(t_ind_List),len(t_ind_List)))
    return fid

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

def ZULF_dataDist(t_ind_List, path1, path2, circuitPath):
    hellDist_List = []; ms_List = []; rp_List = []; rz_List = []; ind_List = []
    SzTot_comp = np.array([1.6257476076555024, 1.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976])
    for indt, t_ind in enumerate(t_ind_List):
        evolution_circuit = load_circuit(circuitPath + 'U_t_{:d}.pickle'.format(int(t_ind)))
        ms_c = total_ms_count(evolution_circuit)
        rp_c, rz_c = total_1q_count(evolution_circuit)
        for indb, b in enumerate(SzTot_comp):
            with open(path1 + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,t_ind)) as file:
                result_file = yaml.full_load(file)
                result1 = result_file['result']
            with open(path2 + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,t_ind)) as file:
                result_file = yaml.full_load(file)
                result2 = result_file['result']
            hellDist = np.sqrt(np.sum((np.sqrt(result1) - np.sqrt(result2)) ** 2)) / np.sqrt(2)
            hellDist_List.append(hellDist); ms_List.append(ms_c); rp_List.append(rp_c); rz_List.append(rz_c); ind_List.append((indb, t_ind))

    return np.array(hellDist_List), np.array(ms_List).astype(int), np.array(rp_List).astype(int), np.array(rz_List).astype(int), np.array(ind_List)

def ZULF_dataDist_ave(t_ind_List, path1, path2, circuitPath):
    hellDist_List = []; overlap_List = []; ms_List = []; rp_List = []; rz_List = []; ind_List = []
    SzTot_comp = np.array([1.6257476076555024, 1.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976])
    for indt, t_ind in enumerate(t_ind_List):
        evolution_circuit = load_circuit(circuitPath + 'U_t_{:d}.pickle'.format(int(t_ind)))
        ms_c = total_ms_count(evolution_circuit)
        rp_c, rz_c = total_1q_count(evolution_circuit)
        ms_List.append(ms_c); rp_List.append(rp_c); rz_List.append(rz_c)
        hellDist = np.zeros(len(SzTot_comp))
        for indb, b in enumerate(SzTot_comp):
            with open(path1 + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,t_ind)) as file:
                result_file = yaml.full_load(file)
                result1 = result_file['result']
            with open(path2 + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,t_ind)) as file:
                result_file = yaml.full_load(file)
                result2 = result_file['result']
            hellDist[indb] = np.sqrt(np.sum((np.sqrt(result1) - np.sqrt(result2)) ** 2)) / np.sqrt(2)
        hellDist_List.append(np.mean(hellDist))

    return np.array(hellDist_List), np.array(ms_List).astype(int), np.array(rp_List).astype(int), np.array(rz_List).astype(int)