import sys
sys.path.append('./')
from ham_comp_utils import compile_group
from nmrfuncs import basisStates
import sys
sys.path.append('../../acetonitrile/')
from basis_utils import InnProd
from scipy.sparse.linalg import expm
from scipy import sparse
import openfermion as of
import numpy as np
from copy import deepcopy
from ham_comp_utils import generate_heisenberg_hamiltonian
import cirq
from bloqade.cirq_utils import noise


def decode_integer_to_statevector(index: int, num_qubits: int) -> np.ndarray:
    """Return the computational basis state vector |index⟩ for a system of num_qubits.
    
    Args:
        index: Integer corresponding to the computational basis state.
        num_qubits: Total number of qubits in the system.

    Returns:
        A numpy array representing the state vector with a 1 at position `index`.
    """
    dimension = 2 ** num_qubits
    if index < 0 or index >= dimension:
        raise ValueError(f"Index {index} is out of range for {num_qubits} qubits.")

    statevector = np.zeros(dimension, dtype=complex)
    statevector[index] = 1.0
    return statevector

def apodize_exp1d(St, k):
    St_fixed = deepcopy(St); St_fixed[0] = 0.5 * St_fixed[0]  # fixes double counting in the FFT so that the spectrum correctly starts at zero instead of having a constant shift
    x = np.linspace(0, 1, St_fixed.size)
    return St_fixed * np.exp(-1 * k * x)


def basisStates_positiveSz_ZULF(Nspin,SzTot):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) which have postive non-zero magnetization, and a list of their corresponding magnetization
    """
    #basisList_integer, basisList_vector = basisStates(Nspin)  # gets list of basis states both in integer encoding and vector format
    basisList_integer = np.arange(2**Nspin)
    magMask = np.logical_not(np.isclose(SzTot, 0.0, atol=1e-3)) * (SzTot > 0.0)
    return np.array(basisList_integer)[magMask], SzTot[magMask]


def gen_sing_trot_unitary(hamiltonian,time):
    
    nqubs = of.count_qubits(hamiltonian)
    trot_unit = np.eye(2**nqubs)
    
    for term, coeff in hamiltonian.terms.items():
        sp_op = of.get_sparse_operator(coeff*of.QubitOperator(term))
        
        trot_unit = sparse.linalg.expm(-1j*sp_op*time)@trot_unit

    return trot_unit


def MatRepSimpleCom(basis,op,n_qubits):

    Nbasis=len(basis)
    Matrix = np.zeros([Nbasis,Nbasis],dtype=complex)

    for i in range(Nbasis):
        for j in range(Nbasis):
            Matrix[i,j] = InnProd(basis[i],of.commutator(op,basis[j]),n_qubits=n_qubits)

    return Matrix


def MatRepOp(basis,op,n_qubits):

    Nbasis=len(basis)
    Matrix = np.zeros([Nbasis,Nbasis],dtype=complex)

    for i in range(Nbasis):
        for j in range(Nbasis):
            Matrix[i,j] = InnProd(basis[i],op*basis[j],n_qubits=n_qubits)

    return Matrix



from openfermion import QubitOperator

# Define Pauli labels
paulis = ['I', 'X', 'Y', 'Z']

def pauli_basis(n_qubits):
    """
    Generate a list of QubitOperators corresponding to the full Pauli basis
    for n_qubits.
    """
    basis = []
    for indices in np.ndindex(*(4,) * n_qubits):  # tuple of length n_qubits
        term = []
        for qubit, label_idx in enumerate(indices):
            if paulis[label_idx] != 'I':
                term.append((qubit, paulis[label_idx]))
        op = QubitOperator(tuple(term))  # Pauli string
        basis.append(op)
    return basis

def swap_pauli_string(op, i, j):
    """
    Given a QubitOperator with a single term, swap labels at positions i and j.
    Returns a new QubitOperator.
    """
    if len(op.terms) != 1:
        raise ValueError("Expected single-term QubitOperator.")
    (pauli_string), coeff = list(op.terms.items())[0]

    # Convert to dict for easy lookup
    d = dict(pauli_string)

    # Fill with I if absent
    labels = []
    max_index = max(i, j, *(d.keys() if d else [0]))
    for k in range(max_index+1):
        labels.append(d.get(k, 'I'))

    # Swap i and j
    labels[i], labels[j] = labels[j], labels[i]

    # Rebuild term
    new_term = tuple((q, p) for q, p in enumerate(labels) if p != 'I')
    return QubitOperator(new_term, coeff)

def permutation_matrix(basis, i, j):
    """
    Construct the permutation matrix P_{i,j} in the given Pauli basis.
    """
    dim = len(basis)
    P = np.zeros((dim, dim), dtype=int)

    for l, op in enumerate(basis):
        swapped = swap_pauli_string(op, i, j)
        # Find matching basis element
        m = basis.index(swapped)
        P[l, m] = 1
    return P


def get_FID_sing_Trot(h_list, J_coup_graph,tgrid,bin_encs,m_vals,Sz_mat,lamb=1.0,nqubs=4,return_exact=False,noise_model=None):
    """
    Get the FID for the hamiltonian H=J\mathbf{S}_{3}\cdot\left(\mathbf {S}_{0}+\mathbf{S}_{1}+\mathbf{S}_{2}\right) built from h_list and J_coup_grap, using a Trotter approximation
    and considering a single Trotter step only,
    in the time grid tgrid, and summing over states encoded in bin_encs
    """

    mod_J_coups = deepcopy(J_coup_graph)

    row = list(mod_J_coups[2])      # convert tuple -> list
    row[2] = lamb*np.float64(mod_J_coups[2][2]) # modify the 3rd element
    mod_J_coups[2] = tuple(row) 

    tar_ham = generate_heisenberg_hamiltonian(h_list, mod_J_coups)
    print("Target Hamiltonian:",tar_ham)

    nqubs = of.count_qubits(tar_ham)
    if return_exact:
        sp_ham = of.get_sparse_operator(tar_ham)
        FID_exact = np.zeros(len(tgrid))
        

    FID= np.zeros(len(tgrid))


    if noise_model is not None:
        qub_reg = cirq.LineQubit.range(nqubs)
        dm_sim = cirq.DensityMatrixSimulator()


    for j in range(len(tgrid)):
        #fid_t = 0.0
        #fid_b_t = 0.0
        fid_u_t = 0.0
        fid_u_exact = 0.0
        for i in range(len(bin_encs)):

            init_state = decode_integer_to_statevector(bin_encs[i], nqubs) 

            
            #tar_ham = generate_heisenberg_hamiltonian(h_list, mod_J_coups)
            if noise_model is not None:
                noisy_circ = noise.transform_circuit(compile_group(qub_reg,tgrid[j]*tar_ham),model=noise_model)
                res = dm_sim.simulate(noisy_circ,initial_state=init_state)

                dm = res.final_density_matrix
                fid_u_t += m_vals[i]*np.trace(dm@Sz_mat)
            else:
                sing_trot_u = gen_sing_trot_unitary(tar_ham,tgrid[j])
            #U_t = expm(-1j*tgrid[j]*mat_ham)

            #res = sim.simulate(compile_group(qub_reg,2*tar_ham),initial_state=init_state)

            #prop_wf = res.final_state_vector

            #prop_wf_b = U_t.toarray()@init_state
                prop_wf_u = sing_trot_u@init_state

            if return_exact:
                prop_wf_exact = expm(-1j*tgrid[j]*sp_ham)@init_state
                fid_u_exact += m_vals[i]*np.vdot(prop_wf_exact,Sz_mat@prop_wf_exact)

            #fid_t+=m_vals[i]*np.vdot(prop_wf,Sz_mat@prop_wf)
            #fid_b_t+= m_vals[i]*np.vdot(prop_wf_b,Sz_mat@prop_wf_b)
            if noise_model is None:
                fid_u_t += m_vals[i]*np.vdot(prop_wf_u,Sz_mat@prop_wf_u)
            #fid_t+=np.vdot(prop_wf,Sz_mat@prop_wf)

        FID[j] = fid_u_t
        if return_exact:
            FID_exact[j] = fid_u_exact
    
    if return_exact:
        return FID, FID_exact
    else:
        return FID


def get_spectrum_from_ham(h_list, J_coup_graph,tgrid,bin_encs,m_vals,Sz_mat,lamb=1.0,nqubs=4,return_exact=False,noise_model=None):

    if return_exact:
        FID,FID_exact = get_FID_sing_Trot(h_list, J_coup_graph,tgrid,bin_encs,m_vals,Sz_mat,lamb=lamb,nqubs=nqubs,return_exact=True,noise_model=noise_model)
        fid_apo = apodize_exp1d(FID-np.mean(FID), 12)
        spec_apo = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))

        fid_exact_apo = apodize_exp1d(FID_exact-np.mean(FID_exact), 12)
        spec_exact_apo = np.real(np.fft.fftshift(np.fft.fft(fid_exact_apo)))

        return spec_apo, spec_exact_apo
    else:

        FID = get_FID_sing_Trot(h_list, J_coup_graph,tgrid,bin_encs,m_vals,Sz_mat,lamb=lamb,nqubs=nqubs)

        fid_apo = apodize_exp1d(FID-np.mean(FID), 12)
        spec_apo = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))

        return spec_apo
