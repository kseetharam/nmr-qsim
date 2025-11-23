import numpy as np
import scipy.io as spio
import openfermion as of
from scipy import sparse
import sys
sys.path.append('./')
from basis_utils import Sx,Sy,Sz


def trace(mat):
    return np.trace(mat)

def comm(A, B):
    return A @ B - B @ A


def Sp(i):
    """Raising operator S_+ = Sx + i Sy for spin i."""
    return Sx(i) + 1j * Sy(i)

def Sm(i):
    """Lowering operator S_- = Sx - i Sy for spin i."""
    return Sx(i) - 1j * Sy(i)

def T2_plus2(i, Bx, By):
    Bp = Bx + 1j*By
    return 0.5 * Sp(i) * Bp

def T2_minus2(i, Bx, By):
    Bm = Bx - 1j*By
    return 0.5 * Sm(i) * Bm

def T2_plus1(i, Bx, By, Bz):
    Bp = Bx + 1j*By
    return -0.5 * ( Sz(i) * Bp + Sp(i) * Bz )

def T2_minus1(i, Bx, By, Bz):
    Bm = Bx - 1j*By
    return 0.5 * ( Sz(i) * Bm + Sm(i) * Bz )

def T2_0(i, Bx, By, Bz):
    Bp = Bx + 1j*By
    Bm = Bx - 1j*By
    return np.sqrt(2/3) * (
        Sz(i) * Bz - 0.25 * ( Sp(i)*Bm + Sm(i)*Bp )
    )

def T2_all(i, Bx, By, Bz):
    """Return dict of all T_{2,m} operators for spin i."""
    return {
        2:  T2_plus2(i, Bx, By),
        1:  T2_plus1(i, Bx, By, Bz),
        0:  T2_0(i, Bx, By, Bz),
        -1: T2_minus1(i, Bx, By, Bz),
        -2: T2_minus2(i, Bx, By)
    }


def T2_plus2_ops(i, j):
    
    return 0.5 * Sp(i) * Sp(j)

def T2_minus2_ops(i,j):
    return 0.5 * Sm(i) * Sm(j)

def T2_plus1_ops(i, j):
    
    return -0.5 * ( Sz(i) * Sp(j) + Sp(i) * Sz(j) )

def T2_minus1_ops(i, j):
    #Bm = Bx - 1j*By
    return 0.5 * ( Sz(i) * Sm(j) + Sm(i) * Sz(j) )

def T2_0_ops(i, j):
    
    return np.sqrt(2/3) * (
        Sz(i) * Sz(j) - 0.25 * ( Sp(i)*Sm(j) + Sm(i)*Sp(j) )
    )


def T2_all_ops(i, j):
    """Return dict of all T_{2,m} operators for spin i."""
    return {
        2:  T2_plus2_ops(i, j),
        1:  T2_plus1_ops(i, j),
        0:  T2_0_ops(i, j),
        -1: T2_minus1_ops(i, j),
        -2: T2_minus2_ops(i, j)
    }


def a2m_from_tensor(A):
    """
    Compute the spherical rank-2 coefficients a_{2,m}
    from a 3x3 tensor A (numpy array or list of lists).

    A must be ordered as [[axx, axy, axz],
                          [ayx, ayy, ayz],
                          [azx, azy, azz]]
    """
    A = np.array(A, dtype=complex)  # allow complex just in case

    axx, axy, axz = A[0,0], A[0,1], A[0,2]
    ayx, ayy, ayz = A[1,0], A[1,1], A[1,2]
    azx, azy, azz = A[2,0], A[2,1], A[2,2]

    coeffs = {}

    # m = ±2
    coeffs[ 2] = 0.5 * ((axx - ayy) - 1j * (axy + ayx))
    coeffs[-2] = 0.5 * ((axx - ayy) + 1j * (axy + ayx))

    # m = ±1
    coeffs[ 1] = -0.5 * ((axz + azx) - 1j * (ayz + azy))
    coeffs[-1] =  0.5 * ((axz + azx) + 1j * (ayz + azy))

    # m = 0
    coeffs[ 0] = (1/np.sqrt(6)) * (2*azz - (axx + ayy))

    return coeffs

def Dip_tensor(coord1,coord2):
    diff = coord2-coord1
    r = np.sqrt(np.dot(diff,diff))

    A = np.zeros([3,3])

    for i in range(3):
        for j in range(3):
            #A[i,j]=3.0*diff[i]*diff[j]/r**2
            #if i==j:
            #    A[i,j]+=-1.0
            if i==j:
                A[i,j]=3.0*diff[i]*diff[j]/r**2-1.0
            else:
                A[i,j]=3.0*diff[i]*diff[j]/r**2
                

    return A


def get_zeeman_mats(filename,tar_idxs):

    loadMat = spio.loadmat(filename,squeeze_me=True)
    zeeman_mats = []

    for i in range(len(tar_idxs)):
        zeeman_mats.append(loadMat['inter_dft']['zeeman'].item()['matrix'].item()[tar_idxs[i]])

    return zeeman_mats

def get_coords(filename,tar_idxs):

    loadMat = spio.loadmat(filename,squeeze_me=True)

    coords = loadMat['inter_dft']['coordinates'].item()
    rep_coords = []

    for i in range(len(tar_idxs)):
        rep_coords.append(coords[tar_idxs[i]])

    return np.array(rep_coords)


def symmetrize_zeeman_coups(zeeman_mats,magfield_vect,gyro_ratios,Gref=np.eye(3),verbose=False):
    """
    Given a list of zeeman coupling tensors, return the isotropic and anisotropic parts in the IST basis. We get the sum of the isotropic contributions per spin
    together with a dictionary that contains the Q_{p,q} operators corresponding to the anisotropic part of the Zeeman interaction. All operators in sparse matrix form
    Args:
    """
    
    nspins = len(zeeman_mats)

    iso_op=of.QubitOperator()
    dict_Qpq = {}

    sum_zeeman_terms = sparse.csc_matrix((2**nspins,2**nspins))

    for i in range(-2,3):
        for j in range(-2,3):
            dict_Qpq[(i,j)] = sparse.csc_matrix((2**nspins,2**nspins))

    Bx, By, Bz = magfield_vect

    zeeman_anis = []

    for i in range(len(zeeman_mats)):

        shift_zeeman_mat = 1e-6*zeeman_mats[i]+Gref

        a_00 = (np.trace(shift_zeeman_mat)/3.0)
        iso_ = a_00*np.eye(3)
        anis_ = shift_zeeman_mat-iso_

        ####computing sum of zeeman terms as reference and for debugging purposes...
        effB = shift_zeeman_mat@magfield_vect
        sum_zeeman_terms+= -1.0*gyro_ratios[i]*(effB[0]*of.get_sparse_operator(Sx(i),n_qubits=nspins)+
                                                effB[1]*of.get_sparse_operator(Sy(i),n_qubits=nspins)+
                                                effB[2]*of.get_sparse_operator(Sz(i),n_qubits=nspins))
       
        
        iso_op += -1.0*gyro_ratios[i]*a_00*(Sz(i)*magfield_vect[2]+Sy(i)*magfield_vect[1]+Sx(i)*magfield_vect[0])
        if verbose:
            print("Isotropic part of the Zeeman tensor for spin ", i)
            #print(f'{a_00*gyro_ratios[i]}*(S_z*{Bz}+S_y*{By}+S_x*{Bx})')
            print(f'{-1.0*a_00*gyro_ratios[i]*Bz}*S_z')


        ###Processing of the anisotropic parts...
        anis_coeffs = a2m_from_tensor(anis_)
        T2_ops = T2_all(i, Bx, By, Bz)

        if verbose:
            print("Anisotropic part of the Zeeman tensor for spin ", i)
            print(f'{-anis_coeffs[1]*gyro_ratios[i]*Bz}*-0.5*S_plus')
            test = 0.5*anis_coeffs[1]*gyro_ratios[i]*Bz*of.get_sparse_operator(Sp(i),n_qubits=nspins)
            ref = -gyro_ratios[i]*anis_coeffs[1]*of.get_sparse_operator(T2_ops[1],n_qubits=nspins)
        
            print("sanity check diff:",np.linalg.norm(test.toarray()-ref.toarray()) )
            print(f'{-anis_coeffs[0]*gyro_ratios[i]*Bz}*sqrt(2/3)*Sz')
            print(f'{-anis_coeffs[-1]*gyro_ratios[i]*Bz}*0.5*S_minus')
            
            #print(f'{-anis_coeffs[2]*gyro_ratios[i]*Bz}*0.5*S_plus*S_plus')
            #print(f'{-anis_coeffs[-2]*gyro_ratios[i]*Bz}*0.5*S_minus*S_minus')
        #We sort the ISTs according to m, but subject to change according to future requirements...

        for p in range(-2,3):
            for q in range(-2,3):
                dict_Qpq[(p,q)]+= -gyro_ratios[i]*anis_coeffs[p]*of.get_sparse_operator(T2_ops[q],n_qubits=nspins)

        
        dict_zeem_ani = {}
        T2_ops_for_i = T2_all(i, Bx, By, Bz)
        for q in range(-2,3):
            dict_zeem_ani[(2,q)] = -gyro_ratios[i]*anis_coeffs[q]*of.get_sparse_operator(T2_ops_for_i[q],n_qubits=nspins)

        zeeman_anis.append(dict_zeem_ani)


    if verbose:
        ##For verification purposes...
        comp_zeeman = of.get_sparse_operator(iso_op,n_qubits=nspins)
        for p in range(-2,3):
            #for q in range(-2,3):
            comp_zeeman+=dict_Qpq[(p,p)]
        
        print("Difference between sum of zeeman terms and the sum of symmetrized terms:",
               np.linalg.norm(comp_zeeman.toarray()-sum_zeeman_terms.toarray()))



    return of.get_sparse_operator(iso_op,n_qubits=nspins), dict_Qpq, zeeman_anis

def symmetrize_dipolar_ham(coords,gyro_ratios,verbose=False):
    """
    Returns: a dictionary that contains the Q_{p,q} operators corresponding to the anisotropic part of the dipolar interaction. All operators in sparse matrix form
    """

    hbar = 1.054571628*1e-34
    nspins = len(coords)
    dict_Qpq = {}

    nspins = len(coords)

    for i in range(-2,3):
        for j in range(-2,3):
            dict_Qpq[(i,j)] = sparse.csc_matrix((2**nspins,2**nspins))


    ####For the sake of verifying that the sum of dipolar terms is the same in the symmetrized and in the original basis...
    sum_dipolar_terms = of.QubitOperator()


    for i in range(nspins):
        coord1 = coords[i]
        for j in range(i+1,nspins):
            coord2 = coords[j]

            A = Dip_tensor(coord1,coord2)
            #print("dipolar tensor is:", A)

            diff1 = coord2-coord1
            #diff2 = coord4 - coord3
            r = np.sqrt(np.dot(diff1,diff1))
            #r2 = np.sqrt(np.dot(diff2,diff2))

            prefact = -np.pi*hbar * 1e-7*gyro_ratios[i]*gyro_ratios[j]/(r**3)


            ####For the sake of verifying that the sum of dipolar terms is the same in the symmetrized and in the original basis...
            sum_dipolar_terms+=prefact*np.dot([Sx(i),Sy(i),Sz(i)],A@[Sx(j),Sy(j),Sz(j)])

            if verbose:
                print("Dipolar couplings between spins",i,j)
                print(prefact*A)

            anis_coeffs = a2m_from_tensor(A)
            ISTS_2spins = T2_all_ops(i, j)

            ###For sanity check..
            if verbose:
                print("complete anisotropic coupling for spins",i,j)
                print(f'{prefact*anis_coeffs[2]}*T2_plus2')
                print(f'{prefact*anis_coeffs[1]}*T2_plus1')
                print(f'{prefact*anis_coeffs[0]}*T2_0')
                print(f'{prefact*anis_coeffs[-1]}*T2_minus1')
                print(f'{prefact*anis_coeffs[-2]}*T2_minus2')

            #Calculation of Q_pq operators...

            for p in range(-2,3):
                for q in range(-2,3):
                    dict_Qpq[(p,q)]+= prefact*anis_coeffs[p]*of.get_sparse_operator(ISTS_2spins[q],n_qubits=nspins)

    if verbose:
        sp_sum_dipolar= of.get_sparse_operator(sum_dipolar_terms,n_qubits=nspins)
        test_sym = sparse.csc_matrix((2**nspins,2**nspins))
        for p in range(-2,3):
            #for q in range(-2,3):
            test_sym+=dict_Qpq[(p,p)]
        print("Difference between sum of dipolar terms and the sum of symmetrized terms:",
               np.linalg.norm(sp_sum_dipolar.toarray()-test_sym.toarray()))

        

    return dict_Qpq

def integrate_double_commutator(O1, O2, H0, Qs, tc, qm_keys=None, tol=1e-8):
    """
    Compute -(1/5) * sum_{k,m} Tr{ O1^† * Integral_0^∞ e^{-tau/tc} [Q, [ e^{-iH0 tau} Q^† e^{iH0 tau}, O2]] dτ }.

    Inputs:
      O1, O2, H0 : (N,N) numpy arrays (complex)  -- H0 should be Hermitian (but not strictly required)
      Qs         : either
                    * list/array of 25 (N,N) numpy arrays (complex), or
                    * dict mapping (k,m) -> (N,N) numpy arrays, where k,m in [-2,-1,0,1,2]
      tc         : positive float (correlation time)
      qm_keys    : optional list of keys specifying the iteration order over Qs when Qs is a list.
                   If Qs is a dict you can ignore qm_keys.
      tol        : tolerance for tiny denominators (not usually needed)

    Returns:
      complex scalar: the full value of the expression.
    """
    # --- basic checks ---
    O1 = np.asarray(O1, dtype=complex)
    O2 = np.asarray(O2, dtype=complex)
    H0 = np.asarray(H0, dtype=complex)
    N = H0.shape[0]
    assert O1.shape == (N, N) and O2.shape == (N, N)

    # prepare list of Q operators in canonical order if dict provided
    # canonical key order: (k,m) with k,m in [-2,-1,0,1,2] (sorted)
    if isinstance(Qs, dict):
        ordered_keys = [(k, m) for k in [-2, -1, 0, 1, 2] for m in [-2, -1, 0, 1, 2]]
        Q_list = [np.asarray(Qs[(k, m)], dtype=complex) for (k, m) in ordered_keys]
    else:
        Q_list = list(Qs)
        if len(Q_list) != 25:
            raise ValueError("If Qs is a list/array it must contain 25 operators (for k,m = -2..2).")
        # if user provided qm_keys, we don't change order; otherwise assume the list order is desired.

    # --- diagonalize H0 (assume Hermitian) ---
    # Use eigh for Hermitian matrices (stable, returns real eigenvalues)
    E, V = np.linalg.eigh(2*np.pi*H0)         # H0 = V diag(E) V^†
    # V: columns are eigenvectors. For non-Hermitian H0, one would need general eig.

    # precompute frequency differences omega_ab = E[a] - E[b]
    # we will need denominator 1/tc + i * omega_ab for each element
    E = E.reshape((-1,))             # shape (N,)
    omega = E[:, None] - E[None, :]  # shape (N,N)  (omega_ab)

    # compute kernel factor F_ab = ∫_0^∞ e^{-τ/tc} e^{-i ω_ab τ} dτ = 1/(1/tc + i ω_ab)
    denom = 1.0 / tc + 1j * omega
    # avoid division by extremely small numbers numerically:
    denom[np.abs(denom) < tol] = tol
    F = 1.0 / denom  
    F= np.real(F)                 # shape (N,N), complex

    # compute contribution for each Q
    total = 0.0 + 0.0j

    # Precompute O1^† for trace
    O1_dag = O1.conj().T

    # for each Q: compute Qd_eig = V^† Q^† V (matrix in eigenbasis), multiply elementwise by F,
    # transform back to lab basis -> Qd_integrated, then form double commutator [Q, [Qd_int, O2]]
    # and accumulate Tr{ O1^† * that }.
    V_dag = V.conj().T

    for Q in Q_list:
        Q = np.asarray(Q, dtype=complex)
        # Q_dag in lab basis
        Qdag = Q.conj().T

        # transform Qdag to eigenbasis of H0
        Qdag_eig = V_dag @ Qdag @ V   # shape (N,N)

        # elementwise multiply by F to perform the time integral
        Qdag_int_eig = Qdag_eig * F   # broadcasting elementwise multiplication

        # transform back to lab basis
        Qdag_int = V @ Qdag_int_eig @ V_dag

        # compute double commutator
        inner_comm = comm(Qdag_int, O2)    # [Qd_int, O2]
        double_comm = comm(Q, inner_comm)  # [Q, [Qd_int, O2]]

        # trace term
        trval = trace(O1_dag @ double_comm)
        total += trval

    prefactor = -1.0 / 5.0

    return prefactor * total

def integrate_Qs(H0, Qs, tc, qm_keys=None, tol=1e-8):
    """
    Returns Integral_0^∞ e^{-tau/tc} e^{-iH0 tau} Q^† e^{iH0 tau} for all Q basis tensors Qs
    """

    H0 = np.asarray(H0, dtype=complex)
    N = H0.shape[0]
    
    # prepare list of Q operators in canonical order if dict provided
    # canonical key order: (k,m) with k,m in [-2,-1,0,1,2] (sorted)
    if isinstance(Qs, dict):
        ordered_keys = [(k, m) for k in [-2, -1, 0, 1, 2] for m in [-2, -1, 0, 1, 2]]
        Q_list = [np.asarray(Qs[(k, m)], dtype=complex) for (k, m) in ordered_keys]
    else:
        Q_list = list(Qs)
        if len(Q_list) != 25:
            raise ValueError("If Qs is a list/array it must contain 25 operators (for k,m = -2..2).")
        # if user provided qm_keys, we don't change order; otherwise assume the list order is desired.

    # --- diagonalize H0 (assume Hermitian) ---
    # Use eigh for Hermitian matrices (stable, returns real eigenvalues)
    E, V = np.linalg.eigh(2*np.pi*H0)         # H0 = V diag(E) V^†
    # V: columns are eigenvectors. For non-Hermitian H0, one would need general eig.

    # precompute frequency differences omega_ab = E[a] - E[b]
    # we will need denominator 1/tc + i * omega_ab for each element
    E = E.reshape((-1,))             # shape (N,)
    omega = E[:, None] - E[None, :]  # shape (N,N)  (omega_ab)

    # compute kernel factor F_ab = ∫_0^∞ e^{-τ/tc} e^{-i ω_ab τ} dτ = 1/(1/tc + i ω_ab)
    denom = 1.0 / tc + 1j * omega
    # avoid division by extremely small numbers numerically:
    denom[np.abs(denom) < tol] = tol
    F = 1.0 / denom  
    F= np.real(F)                 # shape (N,N), complex

    # compute contribution for each Q
    total = 0.0 + 0.0j

    # for each Q: compute Qd_eig = V^† Q^† V (matrix in eigenbasis), multiply elementwise by F,
    # transform back to lab basis -> Qd_integrated, then form double commutator [Q, [Qd_int, O2]]
    # and accumulate Tr{ O1^† * that }.
    V_dag = V.conj().T

    int_Qs = {}

    counter = 0
    for Q in Q_list:
        Q = np.asarray(Q, dtype=complex)
        # Q_dag in lab basis
        Qdag = Q.conj().T

        # transform Qdag to eigenbasis of H0
        Qdag_eig = V_dag @ Qdag @ V   # shape (N,N)

        # elementwise multiply by F to perform the time integral
        Qdag_int_eig = Qdag_eig * F   # broadcasting elementwise multiplication

        # transform back to lab basis
        Qdag_int = V @ Qdag_int_eig @ V_dag
        int_Qs[ordered_keys[counter]] = Qdag_int
        counter+=1

        # compute double commutator
        #inner_comm = comm(Qdag_int, O2)    # [Qd_int, O2]
        #double_comm = comm(Q, inner_comm)  # [Q, [Qd_int, O2]]

        # trace term
        #trval = trace(O1_dag @ double_comm)
        #total += trval

    #prefactor = -1.0 / 5.0

    return int_Qs