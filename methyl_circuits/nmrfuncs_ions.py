import numpy as np
import cirq
from cirq import ops
from copy import copy, deepcopy
from scipy.stats import uniform, randint, multivariate_normal
import pickle
import yaml


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
    integerList = range(0,2**Nspin)
    vectorList = []
    for i in integerList:
        bVec = [0]*(2**Nspin); bVec[i] = 1
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


def basisStates_positiveSz(Nspin):
    """
    Args: Ns = number of spins in the system
    Returns: list of Sz basis states in integer encoding (0 = all +Z state and (2**Nspin)-1 = all -Z state) which have postive non-zero magnetization, and a list of their corresponding magnetization
    """
    basisList_integer, basisList_vector = basisStates(Nspin)  # gets list of basis states both in integer encoding and vector format
    SzTot = np.array(SzTot_obs(Nspin))  # gets total Sz magnetization for each basis state
    magMask = np.logical_not(np.isclose(SzTot, 0.0, atol=1e-3)) * (SzTot > 0.0)
    return np.array(basisList_integer)[magMask], SzTot[magMask]


def SzTot_expectation(state_prob):
    """
    Args: state_prob = list of occupation probabilities (average population) in the computational basis for a state |\psi>. Specifically, is the vector |<psi|\psi>|^2 expressed in the computational basis. This vector can either be directly computed from the state vector of a simulation output or estimated from the histogram of counts resulting from a set of experimental shots. It is assumed that the basis ordering is consistent with the functions basisStates(Nspin) and SzTot_obs(Nspin).
    Note: state_prob can also be a list of lists, each which correspond to the vector |<psi|\psi>|^2 for a different state |\psi>.
    Returns: expectation value of total z-magnetization (SzTot) for each state |\psi>.
    """
    state_prob_array = np.array(state_prob)
    if state_prob_array.ndim > 1:
        Nspin = int(np.log2(len(state_prob[0])))
        Sz_Tot = np.array(SzTot_obs(Nspin))
        return np.sum(np.multiply(Sz_Tot[:, None], np.transpose(state_prob_array)), axis=0)
    else:
        Nspin = int(np.log2(len(state_prob)))
        Sz_Tot = np.array(SzTot_obs(Nspin))
        return np.dot(Sz_Tot, state_prob_array)
    
def responseFunc_sample(state_prob_list, SzTot_comp, Nspin):
    """
    Args:
    state_prob_list = list of state_prob lists, each such list corresponds to a different positive magnetization basis state that the system was initialized in, with each of these states then evolved for the same sample time t
    SzTot_comp = list of SzTot (positive non-zero) magnetization values corresponding to the initial state of each list in state_prob_list
    Returns: sample of response function S(t) evaluated at the sample time t that all the lists in state_prob_list were evaluated at
    """
    return 2 * np.sum(SzTot_expectation(state_prob_list) * SzTot_comp / (2**Nspin))

def spectFunc(responseFunc, tVals, decayRate=1.0):
    """
    Args: responseFunc S(t), tVals = time values (e.g. in s) that the response function is evaluated at, decayRate sets the amount of broadening introduced by hand into the spectrum
    Returns: spectrum A(\omega), fVals = frequency values (e.g. in Hz) that the spectrum is evaluated at
    """    
    St_fixed = deepcopy(responseFunc); St_fixed[0] = 0.5 * St_fixed[0]  # fixes double counting in the FFT so that the spectrum correctly starts at zero instead of having a constant shift
    decay_window = np.exp(-1 * decayRate * tVals)
    nsamples = St_fixed.size; dt = tVals[1] - tVals[0]
    fVals = np.fft.fftshift(np.fft.fftfreq(nsamples, dt))
    FTVals = np.fft.fftshift(np.fft.fft(St_fixed * decay_window) / nsamples)
    spectrum = np.real(FTVals)
    return spectrum, fVals

# ---- MISC NOISE FUNCTIONS ----

def gateFidelity_Paulis(theta, theta_perfect):
    return np.cos(theta - theta_perfect)**2

def floorNoiseConstant(epsilon, sign='negative'):
    """
    Args: epsilon = error in fidelity (e.g. 2% error corresponds to epsilon = 0.02), sign = whether the noisy angles are smaller or larger than the true angles
    Returns: constant c such that theta_noisy = c * theta_perfect
    """  
    if sign == 'negative':
        return 1 - np.arccos(np.sqrt(1-epsilon))*4/np.pi
    elif sign == 'positive':
        return 1 + np.arccos(np.sqrt(1-epsilon))*4/np.pi
    else:
        print('SIGN ERROR')
        return

def xphiGate_generator(phi):
    """
    Outputs the decomposition of the Hermitian generator \hat{\sigma}^{x}_{i} * (cos(\phi)*\hat{\sigma}^{x}_{j} + sin(\phi)*\hat{\sigma}^{y}_{j})
    """
    H_xphi = np.array([[0,0,0,np.exp(-1j*phi)],[0,0,np.exp(1j*phi),0],[0,np.exp(-1j*phi),0,0],[np.exp(1j*phi),0,0,0]])
    diagH, P = np.linalg.eigh(H_xphi)
    return diagH, P

def xphiGate(theta, diagH, P):
    """
    Outputs the unitary gate enacting \hat{U}(\theta, \phi)_{i,j} =  exp( \theta * \hat{\sigma}^{x}_{i} * (cos(\phi)*\hat{\sigma}^{x}_{j} + sin(\phi)*\hat{\sigma}^{y}_{j}) )
    """
    U_xphi = P @ np.diag(np.exp(-1j * diagH * theta)) @ np.conj(P).T
    return cirq.MatrixGate(U_xphi)


# def xphiGate(theta, phi):
#     """
#     Outputs the unitary gate enacting \hat{U}(\theta, \phi)_{i,j} =  exp( \theta * \hat{\sigma}^{x}_{i} * (cos(\phi)*\hat{\sigma}^{x}_{j} + sin(\phi)*\hat{\sigma}^{y}_{j}) )
#     """
#     H_xphi = np.array([[0,0,0,np.exp(-1j*phi)],[0,0,np.exp(1j*phi),0],[0,np.exp(-1j*phi),0,0],[np.exp(1j*phi),0,0,0]])
#     diagH, P = np.linalg.eigh(H_xphi)
#     U_xphi = P @ np.diag(np.exp(-1j * diagH * theta)) @ np.conj(P).T
#     return cirq.MatrixGate(U_xphi)
        
        
# ---- NMR CIRCUIT FUNCTIONS ----

def NMR_2seq(dt, Jij, qubit_reg, qubit_i_index, qubit_j_index, hi_frac=1.0, hj_frac=1.0):
#     assuming that parameter matrix Jij is defined with the Hamiltonian written in terms of spin operators, not Pauli operators
    i, j = qubit_i_index, qubit_j_index
    theta = 2*Jij[i][j]*dt
    hi = Jij[i][i]/hi_frac; hj = Jij[j][j]/hj_frac
    phi_m = (hi - hj)*dt/2
    phi_p = (hi + hj)*dt/2
    norm = np.sign(theta)*np.sqrt((theta/2)**2 + phi_m**2)

    if norm > 0:
        win = int(norm//(np.pi/2))%4
    elif norm < 0:
        win = int(-1*(1+norm//(np.pi/2)))%4
    else:
        print('Norm error')

#   Either fix beta or alpha but not both
    
    btemp = np.arcsin(np.sin(norm)*(theta/2)/norm)
    if win == 0 or win == 3:
        beta = 2 * btemp
    elif win == 1 or win == 2:
        if norm > 0:
            beta = 2 * (np.pi - btemp)
        elif norm < 0:
            beta = 2 * (-1 * btemp - np.pi)          
    else:
        print('Window error')
    alpha = np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
    
#     beta = 2 * np.arcsin(np.sin(norm)*(theta/2)/norm)
#     if win == 0 or win == 3:
#         alpha = np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
#     elif win == 1 or win == 2:
#         if norm > 0:
#             alpha = np.pi - np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
#         elif norm < 0:
#             alpha = -1 * np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2)) - np.pi
#     else:
#         print('Window error')    

    theta = theta/4
    beta = beta/4

    theta_xx = theta % (np.pi)
    nXXs=int(theta_xx/(np.pi/4))
        
    qubit_i = qubit_reg[i]; qubit_j = qubit_reg[j]
#     qubit_i = qubit_reg[0]; qubit_j = qubit_reg[1]
    
    # apply X's + XX                     
    if nXXs==0:
        yield cirq.rx(phi_p + alpha/2).on(qubit_i)
        yield cirq.rx(phi_p - alpha/2).on(qubit_j)
        yield cirq.ms(theta_xx).on(qubit_i,qubit_j)
    if nXXs==3:
        yield cirq.rx(phi_p + alpha/2).on(qubit_i)
        yield cirq.rx(phi_p - alpha/2).on(qubit_j)
        yield cirq.ms(theta_xx-np.pi).on(qubit_i,qubit_j)
    if nXXs==1 or nXXs==2:
        yield cirq.rx(np.pi + phi_p + alpha/2).on(qubit_i)
        yield cirq.rx(np.pi + phi_p - alpha/2).on(qubit_j)
        yield cirq.ms(theta_xx-np.pi/2).on(qubit_i,qubit_j)

    # apply YY + ZZ + X's
    beta_yyzz = beta%(np.pi)
    nYYZZs=int(beta_yyzz/(np.pi/4))
    if nYYZZs==0:
        yield cirq.rz(np.pi/2).on(qubit_i)
        yield cirq.rz(np.pi/2).on(qubit_j)
        yield cirq.ms(beta_yyzz).on(qubit_i,qubit_j)
        yield cirq.rz(-np.pi/2).on(qubit_i)
        yield cirq.rz(-np.pi/2).on(qubit_j)
        
        yield cirq.ry(np.pi/2).on(qubit_i)
        yield cirq.ry(np.pi/2).on(qubit_j)
        yield cirq.ms(beta_yyzz).on(qubit_i,qubit_j)

        # note that Rx(alpha)Ry(-pi/2) = Ry(-pi/2)Rz(-alpha) and  Rx(alpha)Ry(-pi/2)Rx(pi) = Rx(alpha)Ry(pi/2)Rz(pi) = Ry(pi/2)Rz(alpha)Rz(pi) = Ry(pi/2)Rz(pi + alpha)
        yield cirq.rz(-1*alpha/2).on(qubit_i)
        yield cirq.rz(alpha/2).on(qubit_j)
        yield cirq.ry(-np.pi/2).on(qubit_i)
        yield cirq.ry(-np.pi/2).on(qubit_j)
        
    if nYYZZs==3:
        yield cirq.rz(np.pi/2).on(qubit_i)
        yield cirq.rz(np.pi/2).on(qubit_j)
        yield cirq.ms(beta_yyzz-np.pi).on(qubit_i,qubit_j)
        yield cirq.rz(-np.pi/2).on(qubit_i)
        yield cirq.rz(-np.pi/2).on(qubit_j)
        
        yield cirq.ry(np.pi/2).on(qubit_i)
        yield cirq.ry(np.pi/2).on(qubit_j)
        yield cirq.ms(beta_yyzz-np.pi).on(qubit_i,qubit_j)

        yield cirq.rz(-1*alpha/2).on(qubit_i)
        yield cirq.rz(alpha/2).on(qubit_j)
        yield cirq.ry(-np.pi/2).on(qubit_i)
        yield cirq.ry(-np.pi/2).on(qubit_j)
        
    if nYYZZs==1 or nYYZZs==2:
        yield cirq.rz(np.pi/2).on(qubit_i)
        yield cirq.rz(np.pi/2).on(qubit_j)
        yield cirq.rx(np.pi).on(qubit_i)
        yield cirq.rx(np.pi).on(qubit_j)
        yield cirq.ms(beta_yyzz-np.pi/2).on(qubit_i,qubit_j)
        yield cirq.rz(-np.pi/2).on(qubit_i)
        yield cirq.rz(-np.pi/2).on(qubit_j)
        
        yield cirq.ry(np.pi/2).on(qubit_i)
        yield cirq.ry(np.pi/2).on(qubit_j)
        yield cirq.ms(beta_yyzz-np.pi/2).on(qubit_i,qubit_j)
        yield cirq.rz(np.pi + alpha/2).on(qubit_i)
        yield cirq.rz(np.pi - alpha/2).on(qubit_j)
        yield cirq.ry(np.pi/2).on(qubit_i)
        yield cirq.ry(np.pi/2).on(qubit_j)        
        

def NMR_2seq_unitTest(dt, Jij, qubit_reg, qubit_i_index, qubit_j_index, hi_frac=1.0, hj_frac=1.0):
#     assuming that parameter matrix Jij is defined with the Hamiltonian written in terms of spin operators, not Pauli operators
    i, j = qubit_i_index, qubit_j_index
    theta = 2*Jij[i][j]*dt
    hi = Jij[i][i]/hi_frac; hj = Jij[j][j]/hj_frac
    phi_m = (hi - hj)*dt/2
    phi_p = (hi + hj)*dt/2
    norm = np.sign(theta)*np.sqrt((theta/2)**2 + phi_m**2)

    if norm > 0:
        win = int(norm//(np.pi/2))%4
    elif norm < 0:
        win = int(-1*(1+norm//(np.pi/2)))%4
    else:
        print('Norm error')
        
#   Either fix beta or alpha but not both

    btemp = np.arcsin(np.sin(norm)*(theta/2)/norm)
    if win == 0 or win == 3:
        beta = 2 * btemp
    elif win == 1 or win == 2:
        if norm > 0:
            beta = 2 * (np.pi - btemp)
        elif norm < 0:
            beta = 2 * (-1 * btemp - np.pi)          
    else:
        print('Window error')
    alpha = np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
    
#     beta = 2 * np.arcsin(np.sin(norm)*(theta/2)/norm)
#     if win == 0 or win == 3:
#         alpha = np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
#     elif win == 1 or win == 2:
#         if norm > 0:
#             alpha = np.pi - np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2))
#         elif norm < 0:
#             alpha = -1 * np.arcsin(np.sin(norm)*(phi_m/norm)/np.cos(beta/2)) - np.pi
#     else:
#         print('Window error')    
    
    beta_check = np.isclose(np.cos(beta/2)**2 - (np.cos(norm)**2 + (np.sin(norm)**2) * (phi_m/norm)**2), 0)    
    alpha_check = np.isclose(np.cos(alpha)-np.cos(norm)/np.cos(beta/2), 0)
    
    q = cirq.NamedQubit('q0')
    testCircuit = cirq.Circuit([cirq.rz(alpha).on(q), cirq.rx(beta).on(q), cirq.rz(alpha).on(q)])
    circuitUnitary = cirq.unitary(testCircuit)
    unitaryMat = np.cos(norm)*np.array([[1,0],[0,1]]) - 1j*np.sin(norm)*((theta/2/norm)*np.array([[0,1],[1,0]]) + (phi_m/norm)*np.array([[1,0],[0,-1]]))
    eqTest = np.allclose(np.around(circuitUnitary,12),np.around(unitaryMat,12))

#     from quspin.operators import hamiltonian
#     from quspin.basis import spin_basis_1d
#     spinBasis = spin_basis_1d(1, pauli=1)
#     static = [["x", [[theta/2, 0]]], ["z", [[phi_m, 0]]]]
#     ham = hamiltonian(static, [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH, P = ham.eigh()
#     unitaryMat = P @ np.diag(np.exp(-1j * diagH)) @ np.conj(P).T
    
#     from scipy.optimize import minimize
#     def objFunc(angles, unitary):
#         a = angles[0]; b = angles[1]; g = angles[2]
#         q = cirq.NamedQubit('q1')
#         circuitUnitary = cirq.unitary(cirq.Circuit([cirq.rz(g).on(q), cirq.rx(b).on(q), cirq.rz(a).on(q)]))
#         diff = np.sum(np.abs(unitary - circuitUnitary)**2)
#         return diff
#     params = np.array([alpha, beta, alpha])
#     res = minimize(objFunc, x0=params, args=(unitaryMat), method='Nelder-Mead', options={'maxfev': 1000, 'disp': False})
#     optTest = np.allclose(params, res.x) and np.isclose(np.around(res.fun,10),0)

#     if not eqTest:
#         alpha_prime = np.pi - alpha
#         if np.abs(alpha_prime) > np.pi:
#             alpha_prime -= 2*np.pi
#         print(alpha_prime)

#     from quspin.operators import hamiltonian
#     from quspin.basis import spin_basis_1d

#     spinBasis = spin_basis_1d(2, pauli=0)

#     ham_tot = hamiltonian([["xx", [[2*Jij[i][j], 0, 1]]], ["yy", [[2*Jij[i][j], 0, 1]]], ["zz", [[2*Jij[i][j], 0, 1]]], ["x", [[Jij[i][i], 0], [Jij[j][j], 1]]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH_tot, P_tot = ham_tot.eigh()
#     unitaryMat_tot = P_tot @ np.diag(np.exp(-1j * diagH_tot * dt)) @ np.conj(P_tot).T
    
#     ham_xx = hamiltonian([["xx", [[2*Jij[i][j]*dt, 0, 1]]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH_xx, P_xx = ham_xx.eigh()
#     unitaryMat_xx = P_xx @ np.diag(np.exp(-1j * diagH_xx * 1)) @ np.conj(P_xx).T
    
#     ham_yz = hamiltonian([["yy", [[1, 0, 1]]], ["zz", [[1, 0, 1]]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH_yz, P_yz = ham_yz.eigh()
#     unitaryMat_yz = P_yz @ np.diag(np.exp(-1j * diagH_yz * beta)) @ np.conj(P_yz).T

#     ham_x = hamiltonian([["x", [[1, 0], [-1, 1]]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH_x, P_x = ham_x.eigh()
#     unitaryMat_x = P_x @ np.diag(np.exp(-1j * diagH_x * alpha/2)) @ np.conj(P_x).T
    
#     ham_x2 = hamiltonian([["x", [[(hi + hj)*dt/2 + alpha/2, 0], [(hi + hj)*dt/2 - alpha/2, 1]]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
#     diagH_x2, P_x2 = ham_x2.eigh()
#     unitaryMat_x2 = P_x2 @ np.diag(np.exp(-1j * diagH_x2 * 1)) @ np.conj(P_x2).T
    
#     unitaryMat_decomp = unitaryMat_x @ unitaryMat_yz @ unitaryMat_xx @ unitaryMat_x2 
#     decompTest = np.allclose(unitaryMat_tot, unitaryMat_decomp) 

    testOutput = eqTest and beta_check and alpha_check
#     testOutput = eqTest and beta_check and alpha_check and optTest
#     testOutput = eqTest and beta_check and alpha_check and decompTest

    return testOutput
        
        
def apply_XXs(dt, Jij, qubit_reg):
    Nqubits = len(qubit_reg)
    extra_flips = [0] * Nqubits
    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==1 or nXXs==2:
                    extra_flips[i]+=1
                    extra_flips[j]+=1
                    
    for i in range(Nqubits):
        if extra_flips[i]%2==1:
            yield cirq.rx(np.pi).on(qubit_reg[i])
        
    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==0:
                    yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
                if nXXs==3:
                    yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
                if nXXs==1 or nXXs==2:    
                    yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])

def apply_ZZs(dt, Jij, qubit_reg):
    Nqubits = len(qubit_reg)
    extra_flips = [0] * Nqubits
    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==1 or nXXs==2:
                    extra_flips[i]+=1
                    extra_flips[j]+=1
                    
    for i in range(Nqubits):
        if extra_flips[i]%2==0:
            yield cirq.ry(np.pi/2).on(qubit_reg[i])
        else:
            yield cirq.ry(-np.pi/2).on(qubit_reg[i])
            yield cirq.rz(np.pi).on(qubit_reg[i])

    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==0:
                    yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
                if nXXs==3:
                    yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
                if nXXs==1 or nXXs==2:
                    yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])
                    
def apply_ZZs_sym(dt, Jij, qubit_reg):
    Nqubits = len(qubit_reg)
    extra_flips = [0] * Nqubits
    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==1 or nXXs==2:
                    extra_flips[i]+=1
                    extra_flips[j]+=1
                    
    for i in range(Nqubits):
        for j in range(i):
            if not Jij[i][j]==0:
                theta = 2*Jij[i][j]*dt
                theta = theta%(np.pi)
                nXXs=int(theta/(np.pi/4))
                
                if nXXs==0:
                    yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
                if nXXs==3:
                    yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
                if nXXs==1 or nXXs==2:
                    yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])

    for i in range(Nqubits):
        if extra_flips[i]%2==0:
            yield cirq.ry(-np.pi/2).on(qubit_reg[i])
        else:
            yield cirq.rz(np.pi).on(qubit_reg[i])
            yield cirq.ry(np.pi/2).on(qubit_reg[i])                    
                    
def trotter_block_1_p(dt, Jij, qubit_reg):
    yield apply_XXs(dt, Jij, qubit_reg)
    yield [cirq.rz(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield apply_XXs(dt, Jij, qubit_reg)
    yield [cirq.rz(-np.pi/2).on(qubit) for qubit in qubit_reg]
#     yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
#     yield apply_XXs(dt, Jij, qubit_reg)
    yield apply_ZZs(dt, Jij, qubit_reg)
    yield [cirq.rz(-1*Jij[i][i]*dt).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]

def trotter_block_2_p(dt, Jij, qubit_reg):
    yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield [cirq.rz(-1*Jij[i][i]*dt/2).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield apply_ZZs_sym(dt/2, Jij, qubit_reg)
#     yield apply_XXs(dt/2, Jij, qubit_reg)
#     yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]
    
    yield [cirq.rz(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield apply_XXs(dt/2, Jij, qubit_reg)
    yield [cirq.rz(-np.pi/2).on(qubit) for qubit in qubit_reg]
    
    yield apply_XXs(dt, Jij, qubit_reg)
    
    yield [cirq.rz(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield apply_XXs(dt/2, Jij, qubit_reg)
    yield [cirq.rz(-np.pi/2).on(qubit) for qubit in qubit_reg]

#     yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
#     yield apply_XXs(dt/2, Jij, qubit_reg)
    yield apply_ZZs(dt/2, Jij, qubit_reg)
    yield [cirq.rz(-1*Jij[i][i]*dt/2).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]

def apply_XYZs(dt, Jij, qubit_reg, pairList):
    for pair in pairList:
        i, j = pair
        if not Jij[i][j]==0:
            theta = 2*Jij[i][j]*dt
            theta = theta%(np.pi)
            nXXs=int(theta/(np.pi/4))

            # apply XX                     
            if nXXs==0:
                yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
            if nXXs==3:
                yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
            if nXXs==1 or nXXs==2:
                yield cirq.rx(np.pi).on(qubit_reg[i])
                yield cirq.rx(np.pi).on(qubit_reg[j])
                yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])

            # apply YY
            yield cirq.rz(np.pi/2).on(qubit_reg[i])
            yield cirq.rz(np.pi/2).on(qubit_reg[j])
            if nXXs==0:
                yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
            if nXXs==3:
                yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
            if nXXs==1 or nXXs==2:
                yield cirq.rx(np.pi).on(qubit_reg[i])
                yield cirq.rx(np.pi).on(qubit_reg[j])
                yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])
            yield cirq.rz(-np.pi/2).on(qubit_reg[i])
            yield cirq.rz(-np.pi/2).on(qubit_reg[j])

            # apply ZZ
            yield cirq.ry(np.pi/2).on(qubit_reg[i])
            yield cirq.ry(np.pi/2).on(qubit_reg[j])
            if nXXs==0:
                yield cirq.ms(theta).on(qubit_reg[i],qubit_reg[j])
                yield cirq.ry(-np.pi/2).on(qubit_reg[i])
                yield cirq.ry(-np.pi/2).on(qubit_reg[j])
            if nXXs==3:
                yield cirq.ms(theta-np.pi).on(qubit_reg[i],qubit_reg[j])
                yield cirq.ry(-np.pi/2).on(qubit_reg[i])
                yield cirq.ry(-np.pi/2).on(qubit_reg[j])
            if nXXs==1 or nXXs==2:
                yield cirq.ms(theta-np.pi/2).on(qubit_reg[i],qubit_reg[j])
#                     yield cirq.rx(np.pi).on(qubit_reg[i])
#                     yield cirq.ry(-np.pi/2).on(qubit_reg[i])
#                     yield cirq.rx(np.pi).on(qubit_reg[j])
#                     yield cirq.ry(-np.pi/2).on(qubit_reg[j])
                yield cirq.rz(np.pi).on(qubit_reg[i])
                yield cirq.rz(np.pi).on(qubit_reg[j])
                yield cirq.ry(np.pi/2).on(qubit_reg[i])
                yield cirq.ry(np.pi/2).on(qubit_reg[j])

                
def trotter_block_1_e(dt, Jij, qubit_reg):
    Nqubits = len(qubit_reg)
    JijOrder = []
    disjointPairs = []
    for i in range(Nqubits//2):
        disjointPairs.append((2*i+1,2*i)) # important that we do (2i+1, 2i) as we only pick pairs where the first index is bigger than the second
    remainingPairs = []
    for i in range(Nqubits):
        for j in range(i):
            if (i,j) not in disjointPairs:
                remainingPairs.append((i,j))
    
    yield apply_XYZs(dt, Jij, qubit_reg, disjointPairs)
    yield apply_XYZs(dt, Jij, qubit_reg, remainingPairs)    
    yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield [cirq.rz(-1*Jij[i][i]*dt).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]

def trotter_block_2_e(dt, Jij, qubit_reg):
    Nqubits = len(qubit_reg)
    JijOrder = []
    disjointPairs = []
    for i in range(Nqubits//2):
        disjointPairs.append((2*i+1,2*i)) # important that we do (2i+1, 2i) as we only pick pairs where the first index is bigger than the second
    remainingPairs = []
    for i in range(Nqubits):
        for j in range(i):
            if (i,j) not in disjointPairs:
                remainingPairs.append((i,j))
    remainingPairs_reverse = deepcopy(remainingPairs)[::-1]
    
    yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield [cirq.rz(-1*Jij[i][i]*dt/2).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]
    yield apply_XYZs(dt/2, Jij, qubit_reg, remainingPairs_reverse)
    yield apply_XYZs(dt, Jij, qubit_reg, disjointPairs)
    yield apply_XYZs(dt/2, Jij, qubit_reg, remainingPairs)    
    yield [cirq.ry(np.pi/2).on(qubit) for qubit in qubit_reg]
    yield [cirq.rz(-1*Jij[i][i]*dt/2).on(qubit) for i, qubit in enumerate(qubit_reg)]
    yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]
    
def NMR_evolution(t, r, Jij, qubit_reg, trotterOrder, trotterType):
    """
    Args: t = evolution time of unitary, r = number of Trotter steps, Jij = parameters of Hamiltonian, qubit_reg = cirq qubit register,
          trotterOrder = order of product formula (1 = 1st order, 2 = symmetrized 2nd order),
          trotterType = arrangement of product terms ('p' = lower physical time/gate count, 'e' = lower Trotter error)
    Returns: circuit elements for Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    """
    if trotterOrder == 1 and trotterType == 'p':
        trotter_block = trotter_block_1_p
    elif trotterOrder == 2 and trotterType == 'p':
        trotter_block = trotter_block_2_p
    elif trotterOrder == 1 and trotterType == 'e':
        trotter_block = trotter_block_1_e
    elif trotterOrder == 2 and trotterType == 'e':
        trotter_block = trotter_block_2_e

    for i in range(r):
        yield trotter_block(t/r, Jij, qubit_reg)
        
def state_preparation(qubits_to_flip, encoding: int = 0) -> cirq.Circuit:
    """Assume qubits are all in the ground state, will prepare them in the Z basis according to provided encoding
    Taken from euriqafrontend.circuit_builder.prepare_in_Z function

    Args:
        circuit: circuit to add Z preparation to
        qubits_to_flip: list of cirq.LineQubits to perform the encoding on
        encoding: an integer whos binary coding encodes the binary string for the Z basis of the qubits.
                    The last qubit in the list is the most significant bit.
            encoding = 0 -> +Z+Z+Z+Z
            encoding = 1 -> -Z+Z+Z+Z
            encoding = 8 -> +Z+Z+Z-Z
    Returns: cirq circuit with added operation

    """
    n_qubits = len(qubits_to_flip)
    assert 2**n_qubits > encoding, "Encoded bit string longer than addressed ions"
    b = bin(encoding)
    b = b[2:]
    state = np.zeros(n_qubits)
    for j in range(len(b)):
            state[j] = int(b[-1-j])
    state = np.array(state)
    for i in range(n_qubits):
#         yield cirq.ry(np.pi*(state[i])).on(qubits_to_flip[i])
        yield cirq.rx(np.pi*(state[i])).on(qubits_to_flip[i])

#         if state[i] !=0:
#             yield cirq.ry(np.pi*(state[i])).on(qubits_to_flip[i])


def NMR_circuitList(t, r, Jij, qubit_reg, trotterOrder, trotterType):
    """
    Args: t = evolution time of unitary, r = number of Trotter steps, Jij = parameters of Hamiltonian, qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    basisList_comp, SzTot_comp = basisStates_positiveSz(len(qubit_reg))
    circuitList = []
#     evolution_circuit = cirq.Circuit(NMR_evolution(t, r, Jij, qubit_reg, trotterOrder, trotterType))
#     with open(str('mT_circuits/U_t_{:.2f}.pickle'.format(t)), "wb") as file:
#         pickle.dump(evolution_circuit, file)
    for indb, bint in enumerate(basisList_comp):
        circuit = cirq.Circuit()
        circuit.__name__ = "NMR_r_{:d}_t_{:.2f}_b_{:d}".format(r, t, bint)
        circuit.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(NMR_evolution(t, r, Jij, qubit_reg, trotterOrder, trotterType), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList.append(circuit)
        
    return circuitList, SzTot_comp.tolist()

def NMR1D_circuitList_fromFile(t_ind, path, qubit_reg):
    """
    Args: t_ind = evolution time of unitary, path = path to folder containing pickled Cirq circuits , qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    if len(qubit_reg) == 5:
        filename = path + 'U5a_t_{:d}.pickle'.format(t_ind)
#         filename = path + 'U5b_t_{:d}.pickle'.format(t_ind)
    elif len(qubit_reg) == 3:
        filename = path + 'U3a_t_{:d}.pickle'.format(t_ind)
    evolution_circuit = load_circuit(filename)
    evo_qubits = evolution_circuit.all_qubits()
    if len(evo_qubits) != len(qubit_reg):
        print('INVALID NUMBER OF QUBITS')
        return
#     map the generic qubits that evolution_circuit operators on to the provided qubit register
    sorted_evo_qubits = sorted(evo_qubits, key=lambda x: x.name, reverse=False)
    qmap = dict(zip(sorted_evo_qubits, qubit_reg)) 
    mapped_evolution_circuit = evolution_circuit.transform_qubits(lambda q: qmap[q])
    
    basisList_comp, SzTot_comp = basisStates_positiveSz(len(qubit_reg))
    circuitList = []
    for indb, bint in enumerate(basisList_comp):
        circuit = cirq.Circuit()
        circuit.__name__ = "NMR_tind_{:d}_bind_{:d}".format(t_ind, bint)
        circuit.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(mapped_evolution_circuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList.append(circuit)
    return circuitList, SzTot_comp.tolist()

def save_circuit(circuit: cirq.Circuit, filename: str):
    """Save the given circuit as a pickle file.
    """
    with open(str(filename), "wb") as file:
        pickle.dump(circuit, file)
    return


def load_circuit(filename: str):
    """Load a Cirq circuit from a pickle file."""
    with open(str(filename), "rb") as file:
        circuit = pickle.load(file)
    return circuit

# ---- MISC CIRCUIT FUNCTIONS ----

def NMR1D_circuitStats_fromFile(t_ind_List, path, nmr_reg, gateTime_1q, gateTime_2q):
    tot_count = []
    ms_count = []
    rx_count = []
    expTime = []
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = NMR1D_circuitList_fromFile(t_ind, path, nmr_reg)
        ms = []; rx = []; rz = []
        for circuit in circuitList:
            ms_c = total_ms_count(circuit); ms.append(ms_c)
            rx_c, rz_c = total_1q_count(circuit); rx.append(rx_c); rz.append(rz_c)
        ms_ave = np.average(ms); rx_ave = np.average(rx); rz_ave = np.average(rz)
        ms_count.append(ms_ave)
        rx_count.append(rx_ave)
        tot_count.append(ms_ave + rx_ave + rz_ave)
        expTime.append(ms_ave*gateTime_2q + rx_ave*gateTime_1q)
    return tot_count, expTime, rx_count, ms_count

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


# def NMR_circuitStats(tgrid, r, Jij, nmr_reg, gateTime_1q, gateTime_2q, trotterOrder = 1, trotterType = 'p'):
#     gateCount = []
#     expTime = []
#     gateCount1q = []
#     gateCount2q = []
#     for indt, t in enumerate(tgrid):
#         circuitList, SzTot_comp = NMR_circuitList(t, r, Jij, nmr_reg, trotterOrder, trotterType)
#         circuit = circuitList[0]
#         currentGateCount = len(circuit.all_qubits())
#         currentExpTime = len(circuit.all_qubits())*gateTime_1q # time taken to do initial state preparation
#         current1qGateCount = len(circuit.all_qubits())
#         current2qGateCount = 0
#         for mom in circuit:
#             for op in mom.operations:
#                 currentGateCount += 1
#                 gateName = op.gate.__class__.__name__
#                 if gateName == 'MSGate':
#                     currentExpTime += gateTime_2q
#                     current2qGateCount += 1
#                 elif gateName == 'YPowGate' or gateName == 'XPowGate':
#                     currentExpTime += gateTime_1q
#                     current1qGateCount += 1
#                 else:
#                     currentExpTime = currentExpTime
# #                     current1qGateCount += 1

# #                 if op.gate.num_qubits() > 1:
# #                     currentExpTime += gateTime_2q

#         gateCount.append(currentGateCount)
#         expTime.append(currentExpTime)
#         gateCount1q.append(current1qGateCount)
#         gateCount2q.append(current2qGateCount)
#     return gateCount, expTime, gateCount1q, gateCount2q

def NMR_circuitStats(tgrid, r, Jij, nmr_reg, gateTime_1q, gateTime_2q, trotterOrder = 1, trotterType = 'p'):
    tot_count = []
    ms_count = []
    rp_count = []
    expTime = []
    for indt, t in enumerate(tgrid):
        circuitList, SzTot_comp = NMR_circuitList(t, r, Jij, nmr_reg, trotterOrder, trotterType)
        ms = []; rp = []; rz = []
        for circuit in circuitList:
            ms_c = total_ms_count(circuit); ms.append(ms_c)
            rp_c, rz_c = total_1q_count(circuit); rp.append(rp_c); rz.append(rz_c)
        ms_ave = np.average(ms); rp_ave = np.average(rp); rz_ave = np.average(rz)
        ms_count.append(ms_ave)
        rp_count.append(rp_ave)
        tot_count.append(ms_ave + rp_ave + rz_ave)
        expTime.append(ms_ave*gateTime_2q + rp_ave*gateTime_1q)
    return tot_count, expTime, rp_count, ms_count

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

# ---- HEATING SIMULATION HELPER FUNCTIONS ----

def generateNoisyGateAngle(theta_input, c2, gateTime, csAmplitude):
    if c2 == 0:
        return theta_input, csAmplitude
    csMean = np.array([np.real(csAmplitude), np.imag(csAmplitude)])
    csCov = (c2 * gateTime / 2) * np.eye(2)
    posteriorRV = multivariate_normal(mean=csMean, cov=csCov)
    new_csSample = posteriorRV.rvs(size=1)
    new_csAmplitude = new_csSample[0] + 1j * new_csSample[1]
    theta = theta_input * np.exp(-1 * np.abs(new_csAmplitude)**2)
    return theta, new_csAmplitude

def optimumInputAngle(c2, currentExpTime, theta_perfect):
    optrange = [1e-5, 2 * np.pi]
    # optrange = [theta_perfect, 1.5 * theta_perfect]
    def fopt(theta_input): return averageGateFidelity(theta_input, c2, currentExpTime, theta_perfect) - gateFidelity(theta_input, theta_perfect)
    tpsign = np.sign(theta_perfect)
    try:
        sol = root_scalar(fopt, bracket=[tpsign * optrange[0], tpsign * optrange[1]], method='brentq')
        theta_input_opt = sol.root
        finite_fidelity = averageGateFidelity(theta_input_opt, c2, currentExpTime, theta_perfect)
        zero_fidelity = averageGateFidelity(0, c2, currentExpTime, theta_perfect)
        inf_fidelity = averageGateFidelity(1e10, c2, currentExpTime, theta_perfect)
        fidOptions = np.array([finite_fidelity, zero_fidelity, inf_fidelity])
        angleOptions = np.array([theta_input_opt, 0, 1e10])
        maxInd = np.argmax(fidOptions)
        outputAngle = angleOptions[maxInd]
    except ValueError:
        print('Root Scalar Value Error')
        outputAngle = optimumInputAngle_grid(c2, currentExpTime, theta_perfect)
        print(theta_perfect / np.pi, outputAngle / np.pi)
    except Exception as e:
        print('Root Scalar Strange Error')
        outputAngle = optimumInputAngle_grid(c2, currentExpTime, theta_perfect)
        print(theta_perfect / np.pi, outputAngle / np.pi)
    return outputAngle


def optimumInputAngle_grid(c2, currentExpTime, theta_perfect):
    theta_input_Vals = np.linspace(theta_perfect, 1.5 * theta_perfect, 1000)
    fopt_Vals = averageGateFidelity_vectorized(theta_input_Vals, c2, currentExpTime, theta_perfect) - gateFidelity(theta_input_Vals, theta_perfect)
    solind = np.argmin(np.abs(fopt_Vals))
    theta_input_opt = theta_input_Vals[solind]
    finite_fidelity = averageGateFidelity(theta_input_opt, c2, currentExpTime, theta_perfect)
    zero_fidelity = averageGateFidelity(0, c2, currentExpTime, theta_perfect)
    inf_fidelity = averageGateFidelity(1e10, c2, currentExpTime, theta_perfect)
    fidOptions = np.array([finite_fidelity, zero_fidelity, inf_fidelity])
    angleOptions = np.array([theta_input_opt, 0, 1e10])
    maxInd = np.argmax(fidOptions)
    return angleOptions[maxInd]


def averageGateFidelity(theta_input, c2, currentExpTime, theta_perfect):
    eta = c2 * currentExpTime
    hyp1f2 = mpm.hyp1f2
    return (1 / 2) + (1 / 2) * np.cos(theta_input / 2) * np.cos(theta_perfect / 2) + (eta * (theta_input**2) / (8 + 16 * eta)) * np.cos(theta_perfect / 2) * hyp1f2(1 + 1 / (2 * eta), 3 / 2, 2 + 1 / (2 * eta), -(theta_input**2) / 16) + (theta_input / (4 + 4 * eta)) * np.sin(theta_perfect / 2) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 16)


def averageGateFidelity_vectorized(theta_input, c2, currentExpTime, theta_perfect):
    eta = c2 * currentExpTime
    hyp1f2 = np.vectorize(mpm.hyp1f2)
    return (1 / 2) + (1 / 2) * np.cos(theta_input / 2) * np.cos(theta_perfect / 2) + (eta * (theta_input**2) / (8 + 16 * eta)) * np.cos(theta_perfect / 2) * hyp1f2(1 + 1 / (2 * eta), 3 / 2, 2 + 1 / (2 * eta), -(theta_input**2) / 16) + (theta_input / (4 + 4 * eta)) * np.sin(theta_perfect / 2) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 16)


def gateFidelity(theta_input, theta_perfect):
    return np.cos((theta_input - theta_perfect) / 4)**2

# ---- HEATING NOISE CIRCUIT MODIFIERS ----

def circuitMod_FFCorrection(circuit, c2, gateTime_1q, gateTime_2q, EV_only=False):
    """
    Args: circuit = ion circuit consisting of single qubit rotations and MS gates (doesn't include initial moment of state preparation rotations or final oment of measurements), c2 = ion heating rate, gateTime_1q = time to enact a single qubit rotation (in ms), gateTime_2q = time to enact a MS gate (in ms), EV_only = only apply a correction so the average noisy gate angle is correct rather than the full correction of the average two qubit gate fidelity (default of False applies the full correction)
    Returns: newcircuit = copy of old circuit with two qubit (MS) gate angles adjusted with the feedforward correction
    """
    newcircuit = cirq.Circuit()
#     newcircuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    currentExpTime = len(circuit.all_qubits())*gateTime_1q # time taken to do initial state preparation
    for mom in circuit:
        newmom = []
        for op in mom.operations:
            gateName = op.gate.__class__.__name__
            
            if gateName == 'MSGate':
                theta_ms = op.gate.exponent*np.pi/2
                if EV_only:
                    theta_ms_new = theta_ms * (1 + c2 * currentExpTime)  # This corrects the average applied angle
                else:
                    theta_ms_new = optimumInputAngle(c2, currentExpTime, theta_ms)  # This optimizes the average fidelity
                newms_op = cirq.ms(theta_ms_new).on(*op.qubits)
                newmom.append(newms_op)
                currentExpTime += gateTime_2q
            else:
                newmom.append(op)
                if gateName == 'YPowGate' or gateName == 'XPowGate':
                    currentExpTime += gateTime_1q
                
#             if op.gate.num_qubits() > 1:
#                 theta_ms = op.gate.exponent*np.pi/2
#                 if EV_only:
#                     theta_ms_new = theta_ms * (1 + c2 * currentExpTime)  # This corrects the average applied angle
#                 else:
#                     theta_ms_new = optimumInputAngle(c2, currentExpTime, theta_ms)  # This optimizes the average fidelity
#                 newms_op = cirq.ms(theta_ms_new).on(*op.qubits)
#                 newmom.append(newms_op)
#                 currentExpTime += gateTime_2q
#             else:
#                 newmom.append(op)
#                 currentExpTime += gateTime_1q
        newmom = cirq.Moment(newmom)
        newcircuit.append(newmom)
#     newcircuit.append(circuit2[-1]) # add measurement
    return newcircuit

def circuitMod_heatingNoise(circuit, c2, gateTime_1q, gateTime_2q):
    """
    Args: circuit = ion circuit including an initial moment of single qubit rotations (state preparation) and a circuit consisting of single qubit rotations and MS gates (doesn't include final moment of measurements), c2 = ion heating rate, gateTime_1q = time to enact a single qubit rotation (in ms), gateTime_2q = time to enact a MS gate (in ms)
    Returns: newcircuit = copy of old circuit with two qubit (MS) gate angles adjusted with simulated heating noise
    """
    newcircuit = cirq.Circuit()
    newcircuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    currentExpTime = len(circuit.all_qubits())*gateTime_1q # time taken to do initial state preparation
    csAmplitude = 0 + 0j  # Should technically pull this from a thermal distribution with a temperature set by how much they cool the ions between each shot
    last2qTime = 0
    for mom in circuit[1::]:
        newmom = []
        for op in mom.operations:
            gateName = op.gate.__class__.__name__
            if gateName == 'MSGate':
                theta_ms = op.gate.exponent*np.pi/2
                
                elapsedTime = currentExpTime - last2qTime
                theta_ms_new, csAmplitude = generateNoisyGateAngle(theta_ms, c2, elapsedTime, csAmplitude)                
                
                newms_op = cirq.ms(theta_ms_new).on(*op.qubits)
                newmom.append(newms_op)
                
                last2qTime = copy(currentExpTime)
                currentExpTime += gateTime_2q

            else:
                newmom.append(op)
                if gateName == 'YPowGate' or gateName == 'XPowGate':
                    currentExpTime += gateTime_1q
            
#             if op.gate.num_qubits() > 1:
#                 theta_ms = op.gate.exponent*np.pi/2
                
#                 elapsedTime = currentExpTime - last2qTime
#                 theta_ms_new, csAmplitude = generateNoisyGateAngle(theta_ms, c2, elapsedTime, csAmplitude)                
                
#                 newms_op = cirq.ms(theta_ms_new).on(*op.qubits)
#                 newmom.append(newms_op)
                
#                 last2qTime = copy(currentExpTime)
#                 currentExpTime += gateTime_2q
#             else:
#                 newmom.append(op)
#                 currentExpTime += gateTime_1q
        newmom = cirq.Moment(newmom)
        newcircuit.append(newmom)
    return newcircuit


def NMR_circuitList_FFCorrection(t, r, Jij, qubit_reg, c2, gateTime_1q, gateTime_2q, EV_only, trotterOrder, trotterType):
    """
    Args: t = evolution time of unitary, r = number of Trotter steps, Jij = parameters of Hamiltonian, qubit_reg = cirq qubit register
    Returns: circuitList = list of Cirq circuits which prepare a positive magnetization basis state and then enact Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
    SzTot_comp = list of magnetization values of corresponding to the initial state of each circuit in circuitList
    Note: measurements are not added to these circuits and therefore should be added afterwards if desired
    """
    basisList_comp, SzTot_comp = basisStates_positiveSz(len(qubit_reg))
    circuitList = []
    for indb, bint in enumerate(basisList_comp):
        circuit = cirq.Circuit()
        circuit.__name__ = "NMR_r_{:d}_t_{:.2f}_b_{:d}".format(r, t, bint)
        circuit.append(state_preparation(qubit_reg, bint), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        timeEvoCircuit = cirq.Circuit(NMR_evolution(t, r, Jij, qubit_reg, trotterOrder, trotterType), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(circuitMod_FFCorrection(timeEvoCircuit, c2, gateTime_1q, gateTime_2q, EV_only), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuitList.append(circuit)
    return circuitList, SzTot_comp.tolist()

# ---- OTHER NOISE CIRCUIT MODIFIERS ----

def circuitMod_floorNoise(circuit, epsilon, sign='negative'):
    """
    Args: circuit = ion circuit including an initial moment of single qubit rotations (state preparation) and a circuit consisting of single qubit rotations and MS gates (doesn't include final moment of measurements), epsilon = error in fidelity (e.g. 2% error corresponds to epsilon = 0.02), sign = whether the noisy angles are smaller or larger than the true angles
    Returns: newcircuit = copy of old circuit with two qubit (MS) gate angles adjusted with simulated floor noise
    """
    newcircuit = cirq.Circuit()
    newcircuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    c = floorNoiseConstant(epsilon, 'negative')
    for mom in circuit[1::]:
        newmom = []
        for op in mom.operations:
            gateName = op.gate.__class__.__name__
            if gateName == 'MSGate':
                theta_ms = op.gate.exponent*np.pi/2                
                theta_ms_new = c * theta_ms                
                newms_op = cirq.ms(theta_ms_new).on(*op.qubits)
                newmom.append(newms_op)

            else:
                newmom.append(op)

        newmom = cirq.Moment(newmom)
        newcircuit.append(newmom)
    return newcircuit

def circuitMod_crossTalkNoise(circuit, qubit_reg):
    """
    Args: circuit = ion circuit including an initial moment of single qubit rotations (state preparation) and a circuit consisting of single qubit rotations and MS gates (doesn't include final moment of measurements), assume the angles of the spurious two-qubit gates have a magnitude that is between 2-3% of the intended angle
    Returns: newcircuit = copy of old circuit with two qubit (MS) gate angles adjusted with simulated cross talk noise
    """
    newcircuit = cirq.Circuit()
    newcircuit.append(circuit[0]) # add initial single qubit rotations that do state preparation
    ep = np.random.uniform(0.01,0.03) # sets the magnitude of the spurious gates (relative to the intended angle)
    phi = np.random.uniform(0,2*np.pi) # sets the phase of the spurious gates
    diagH, P = xphiGate_generator(phi)
    for mom in circuit[1::]:
        newmom = []
        for op in mom.operations:
            gateName = op.gate.__class__.__name__
            if gateName == 'MSGate':
                theta_ms = op.gate.exponent*np.pi/2                
                qi = op.qubits[0]; qj = op.qubits[1]                
                newmom.append(op) # make sure to add the actual xx gate
                
                theta_ms_new = ep * theta_ms
                crossTalkGate = xphiGate(theta_ms_new, diagH, P)
                if (qj+1) in qubit_reg and (qj+1) != qi:
                    newmom.append(crossTalkGate.on(qi,qj+1))                
                if (qj-1) in qubit_reg and (qj-1) != qi:
                    newmom.append(crossTalkGate.on(qi,qj-1))
                if (qi+1) in qubit_reg and (qi+1) != qj:
                    newmom.append(crossTalkGate.on(qj,qi+1))    
                if (qi-1) in qubit_reg and (qi-1) != qj:
                    newmom.append(crossTalkGate.on(qj,qi-1))

            else:
                newmom.append(op)

#         newmom = cirq.Moment(newmom)  # all the above operations won't necessarily be a single non-overlapping moment
        newcircuit.append(newmom)
    return newcircuit

# ---- SIMULATIONS ----

def NMR1D_noiselessSim_fromFile(t_ind_List, path, nmr_reg):
    simulator = cirq.Simulator()
    responseFunc = []
    for indt, t_ind in enumerate(t_ind_List):
        circuitList, SzTot_comp = NMR1D_circuitList_fromFile(t_ind, path, nmr_reg)
        state_prob_list = []
        for circuit in circuitList:
            # Simulation (noiseless)
            vec = simulator.simulate(circuit).final_state_vector
            avg_pop = np.abs(vec)**2
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot_comp, len(nmr_reg)))
    return responseFunc


def NMR_noiselessSim(tgrid, r, Jij, nmr_reg, trotterOrder = 1, trotterType = 'p'):
    simulator = cirq.Simulator()
    responseFunc = []
    for indt, t in enumerate(tgrid):
        circuitList, SzTot_comp = NMR_circuitList(t, r, Jij, nmr_reg, trotterOrder, trotterType)
        state_prob_list = []
        for circuit in circuitList:
            # Simulation (noiseless)
            vec = simulator.simulate(circuit).final_state_vector
            avg_pop = np.abs(vec)**2
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot_comp, len(nmr_reg)))
    return responseFunc


def NMR_heatingSim(tgrid, r, Jij, nmr_reg, runAvg, c2, gateTime_1q, gateTime_2q, FFCorrection=True, EV_only=False, trotterOrder = 1, trotterType = 'p'):
    simulator = cirq.Simulator()
    responseFunc = []
    for indt, t in enumerate(tgrid):
        if FFCorrection:
            circuitList, SzTot_comp = NMR_circuitList_FFCorrection(t, r, Jij, nmr_reg, c2, gateTime_1q, gateTime_2q, EV_only, trotterOrder, trotterType)
        else:
            circuitList, SzTot_comp = NMR_circuitList(t, r, Jij, nmr_reg, trotterOrder, trotterType)
        state_prob_list = []
        for circuit in circuitList:
            avg_pop_List = []
            for run in np.arange(runAvg):
                noisycircuit = circuitMod_heatingNoise(circuit, c2, gateTime_1q, gateTime_2q)
                vec = simulator.simulate(noisycircuit).final_state_vector
                avg_pop = np.abs(vec)**2
                avg_pop_List.append(avg_pop)
            avg_pop_mean = np.mean(np.array(avg_pop_List), axis=0)
            
            # Add vector of average population of each basis state measured via either simulation or experiment       
            state_prob_list.append(avg_pop_mean)

        responseFunc.append(responseFunc_sample(state_prob_list, SzTot_comp, len(nmr_reg)))
    return responseFunc


def NMR_noisySim(tgrid, r, Jij, nmr_reg, runAvg, noiseParams, trotterOrder = 1, trotterType = 'p'):
    heatingNoise = noiseParams['heatingNoise']
    floorNoise = noiseParams['floorNoise']
    crossTalkNoise = noiseParams['crossTalkNoise']
    FFCorrection = False
    if heatingNoise:
        c2 = noiseParams['c2']; gateTime_1q = noiseParams['gateTime_1q']; gateTime_2q = noiseParams['gateTime_2q']
        FFCorrection = noiseParams['FFCorrection']; EV_only = noiseParams['EV_only']
    if floorNoise:
        epsilon = noiseParams['epsilon']; sign = noiseParams['sign']
        
    simulator = cirq.Simulator()
    responseFunc = []
    for indt, t in enumerate(tgrid):
        if FFCorrection:
            circuitList, SzTot_comp = NMR_circuitList_FFCorrection(t, r, Jij, nmr_reg, c2, gateTime_1q, gateTime_2q, EV_only, trotterOrder, trotterType)
        else:
            circuitList, SzTot_comp = NMR_circuitList(t, r, Jij, nmr_reg, trotterOrder, trotterType)
        state_prob_list = []
        for circuit in circuitList:
            avg_pop_List = []
            for run in np.arange(runAvg):
                noisycircuit = circuit
                if heatingNoise:
                    noisycircuit = circuitMod_heatingNoise(noisycircuit, c2, gateTime_1q, gateTime_2q)
                if floorNoise:
                    noisycircuit = circuitMod_floorNoise(noisycircuit, epsilon, sign)
                if crossTalkNoise:
                    noisycircuit = circuitMod_crossTalkNoise(noisycircuit, nmr_reg)
                vec = simulator.simulate(noisycircuit).final_state_vector
                avg_pop = np.abs(vec)**2
                avg_pop_List.append(avg_pop)
            avg_pop_mean = np.mean(np.array(avg_pop_List), axis=0)
            
            # Add vector of average population of each basis state measured via either simulation or experiment       
            state_prob_list.append(avg_pop_mean)

        responseFunc.append(responseFunc_sample(state_prob_list, SzTot_comp, len(nmr_reg)))
    return responseFunc

# # ---- FIRST SIMULATION ----

# def trotter_block_initialSim(dt, Jij, qubit_reg):
#     yield apply_XXs(dt, Jij, qubit_reg)
#     yield [cirq.rz(np.pi/2).on(qubit) for qubit in qubit_reg]
#     yield apply_XXs(dt, Jij, qubit_reg)
#     yield [cirq.rz(-np.pi/2).on(qubit) for qubit in qubit_reg]
#     yield apply_ZZs(dt, Jij, qubit_reg)
#     yield [cirq.rz(2*Jij[i][i]*dt).on(qubit) for i, qubit in enumerate(qubit_reg)]
#     yield [cirq.ry(-np.pi/2).on(qubit) for qubit in qubit_reg]
        
# def NMR_evolution_initialSim(t, r, Jij, qubit_reg):
#     """
#     Args: t = evolution time of unitary, r = number of Trotter steps, Jij = parameters of Hamiltonian, qubit_reg = cirq qubit register
#     Returns: circuit elements for Trotterized time-evolution U(t) = exp(-i*H*t) with t split into r Trotter steps
#     """
#     for i in range(r):
#         yield trotter_block_initialSim(t/r, Jij, qubit_reg)

# def NMR_initialSim(tgrid, r, Jij, nmr_reg, runAvg, c2, gateTime_1q, gateTime_2q, FFCorrection=True, EV_only=True):
#     simulator = cirq.Simulator()
#     magnetization = []
#     basisList_comp, SzTot_comp = basisStates_positiveSz(len(nmr_reg))
#     for indt, t in enumerate(tgrid):
#         circuit = cirq.Circuit()
#         if FFCorrection:
#             timeEvoCircuit = cirq.Circuit(NMR_evolution_initialSim(t, r, Jij, nmr_reg), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
#             circuit.append(circuitMod_FFCorrection(timeEvoCircuit, c2, gateTime_1q, gateTime_2q, EV_only), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
#         else:
#             circuit.append(NMR_evolution_initialSim(t, r, Jij, nmr_reg), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)  
#         # Simulation (noiseless)
#         avg_pop_List = []
#         for run in np.arange(runAvg):
#             noisycircuit = circuitMod_heatingNoise(circuit, c2, gateTime_1q, gateTime_2q)
#             vec = simulator.simulate(noisycircuit).final_state_vector
#             avg_pop = np.abs(vec)**2
#             avg_pop_List.append(avg_pop)
#         avg_pop_mean = np.mean(np.array(avg_pop_List), axis=0)
#         magnetization.append(-1*SzTot_expectation(avg_pop_mean)) # counting magnetization according to Marko's convention

#     return magnetization

def mT_expData(path):
    responseFunc = []
    SzTot_comp = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    for indt in np.arange(6):
        state_prob_list = []
        for indb in np.arange(5):
            with open(path + 'basis_{:d}_U_t_{:d}.yaml'.format(indb,indt)) as file:
                result_file = yaml.full_load(file)
                result = result_file['result']
                avg_pop = result
            state_prob_list.append(avg_pop)
        responseFunc.append(responseFunc_sample(state_prob_list, SzTot_comp, 4))
    return responseFunc