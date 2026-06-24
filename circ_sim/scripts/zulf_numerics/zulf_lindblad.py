"""
ZULF Lindblad simulation of the 19F-13C-1H three-spin system in
4-fluorophenylalanine.

Geometry from the DFT calculation in
  circ_sim/data/operator_dist/4_fluoro_phe.log
using the final Standard-orientation block (last occurrence).

"""

import os
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# =============================================================================
# SECTION 1: PHYSICAL CONSTANTS
# =============================================================================

hbar = 1.054571817e-34   # J·s
mu0  = 1.25663706212e-6  # T·m/A

# =============================================================================
# SECTION 2: NUCLEAR SPECIES AND GYROMAGNETIC RATIOS (rad / s / T)
#
# Values from CODATA 2018 / IUPAC recommendations.
# Convention matches Spinach: omega_k = -gamma_k * B0
# =============================================================================

GAMMA = {
    '19F':  251.81520e6,   # rad/(s·T)
    '13C':   67.28284e6,   # rad/(s·T)
    '1H':   267.52218e6,   # rad/(s·T)
}

# Spin ordering throughout this script: spin 1 = 19F, 2 = 13C, 3 = 1H
# (same as fch_triple_redfield.m)
NUCLEI = ['19F', '13C', '1H']
gamma  = np.array([GAMMA[n] for n in NUCLEI])   # shape (3,)

# =============================================================================
# SECTION 3: NUCLEAR COORDINATES  (Angstroms, DFT optimized geometry)
#
# Source: final Standard-orientation block in 4_fluoro_phe.log
#   Center  8   C (13C)   Gaussian atomic number 6
#   Center 22   F (19F)   Gaussian atomic number 9
#   Center 23   H (1H)    Gaussian atomic number 1  (ortho aromatic H)
#
# Converted to metres below for dipolar coupling calculation.
# =============================================================================

ANGSTROM = 1e-10   # m

coords_ang = {
    '19F': np.array([-3.9805 , -0.5583 ,  -0.6136]),   
    '13C': np.array([-2.7608 ,  -0.2372 ,  -0.1625]),   
    '1H':  np.array([-0.8183  ,  2.4463  ,  0.3685]),   
}
coords_m = {k: v * ANGSTROM for k, v in coords_ang.items()}

# Inter-nuclear displacement vectors r_ij = r_j - r_i  (metres)
r_FC = coords_m['13C'] - coords_m['19F']
r_FH = coords_m['1H']  - coords_m['19F']
r_CH = coords_m['1H']  - coords_m['13C']

def dipolar_coupling(gamma_i, gamma_j, r_vec):
    """Return the dipolar coupling constant b_ij (rad/s).

    b_ij = -(mu0 / 4pi) * gamma_i * gamma_j * hbar / r^3
    The secular (traceless) dipolar Hamiltonian is
        H_D = b_ij * (3 cos^2 theta - 1) * (Iz_i Iz_j - I_i.I_j / 4)
    after powder / isotropic averaging over molecular tumbling.
    """
    r = np.linalg.norm(r_vec)
    return -(mu0 / (4 * np.pi)) * gamma_i * gamma_j * hbar / r**3

b_FC = dipolar_coupling(GAMMA['19F'], GAMMA['13C'], r_FC)
b_FH = dipolar_coupling(GAMMA['19F'], GAMMA['1H'],  r_FH)
b_CH = dipolar_coupling(GAMMA['13C'], GAMMA['1H'],  r_CH)

print("Inter-nuclear distances (Angstrom):")
print(f"  r(19F-13C) = {np.linalg.norm(r_FC)/ANGSTROM:.4f}")
print(f"  r(19F-1H)  = {np.linalg.norm(r_FH)/ANGSTROM:.4f}")
print(f"  r(13C-1H)  = {np.linalg.norm(r_CH)/ANGSTROM:.4f}")
print()
print("Dipolar coupling constants b_ij / (2pi)  (Hz):")
print(f"  b(19F-13C) / 2pi = {b_FC / (2*np.pi):.2f}")
print(f"  b(19F-1H)  / 2pi = {b_FH / (2*np.pi):.2f}")
print(f"  b(13C-1H)  / 2pi = {b_CH / (2*np.pi):.2f}")

# =============================================================================
# SECTION 4: J COUPLINGS (Hz)
#
# From fch_triple_redfield.m comments and inter.coupling.scalar.
# No 19F-1H scalar coupling is included (set to zero as in the MATLAB script).
# =============================================================================

J_FC = 243.5    # Hz  1J(19F-13C)
J_CH =  10.71   # Hz  J(13C-1H)   — likely a 2J or long-range coupling
J_FH =   0.0    # Hz  (not included)

# Convert to rad/s for use in the Hamiltonian: 2pi * J
J_FC_rad = 2 * np.pi * J_FC
J_CH_rad = 2 * np.pi * J_CH
J_FH_rad = 2 * np.pi * J_FH

# =============================================================================
# SECTION 5: SIMULATION PARAMETERS
# =============================================================================

tau_c = 1e-5    # s   rotational correlation time (default from MATLAB script)
B0    = 0.0     # T   static field (ZULF)

# Larmor frequencies omega_k = -gamma_k * B0  (rad/s)
omega_F = -GAMMA['19F'] * B0
omega_C = -GAMMA['13C'] * B0
omega_H = -GAMMA['1H']  * B0

# =============================================================================
# SECTION 6: SUMMARY
# =============================================================================

# =============================================================================
# SECTION 7: QUTIP SPIN OPERATORS AND H0 HAMILTONIAN
# =============================================================================

def spin_op(op_char, idx, n=3):
    """Single spin-1/2 operator embedded in the full n-spin Hilbert space."""
    ops = [qt.qeye(2)] * n
    ops[idx] = qt.jmat(0.5, op_char)
    return qt.tensor(ops)

# Cartesian and ladder operators in the 8-dimensional three-spin space.
# Spin order: 0 = 19F,  1 = 13C,  2 = 1H
Ix = [spin_op('x', k) for k in range(3)]
Iy = [spin_op('y', k) for k in range(3)]
Iz = [spin_op('z', k) for k in range(3)]
Ip = [Ix[k] + 1j * Iy[k] for k in range(3)]   # I_+  raising operator
Im = [Ix[k] - 1j * Iy[k] for k in range(3)]   # I_-  lowering operator

def heisenberg_dot(i, j):
    """Isotropic Heisenberg term I_i · I_j."""
    return Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j]

# H0 = 2pi*J_FC (I_F·I_C) + 2pi*J_CH (I_C·I_H) + 2pi*J_FH (I_F·I_H)
# J_*_rad are already in rad/s.
H0 = (J_FC_rad * heisenberg_dot(0, 1)
    + J_CH_rad * heisenberg_dot(1, 2)
    + J_FH_rad * heisenberg_dot(0, 2))

# Diagonalise H0 to get eigenstates (needed in the next stage for jump operators)
evals, ekets = H0.eigenstates()   # evals in rad/s, ekets as Qobj column vectors

# Assign S, M_tot quantum numbers to each eigenstate.
# [H0, M_tot] = [H0, S_tot^2] = 0 (each I_i·I_j term is a rotational scalar),
# so S and M are good quantum numbers.  Eigenstates in different M sectors are
# automatically orthogonal; within the degenerate S=3/2 quartet all four M
# values are distinct, so QuTiP returns them as M_tot eigenstates.
_M_tot_op = Iz[0] + Iz[1] + Iz[2]
_S2_tot_op = ((Ix[0]+Ix[1]+Ix[2])**2
             + (Iy[0]+Iy[1]+Iy[2])**2
             + (Iz[0]+Iz[1]+Iz[2])**2)

_M_qn  = np.round(2 * np.real([qt.expect(_M_tot_op,  ek) for ek in ekets])) / 2
_S2_ev = np.real([qt.expect(_S2_tot_op, ek) for ek in ekets])
_S_qn  = np.round(2 * (np.sqrt(np.clip(4*_S2_ev + 1, 0, None)) - 1) / 2) / 2

# Two S=1/2 doublets separated by energy; label lower-energy one α, higher β
_E_Hz_arr   = evals / (2 * np.pi)
_s12_Es     = np.unique(np.round(_E_Hz_arr[np.abs(_S_qn - 0.5) < 0.1], 1))  # 2 values

def _state_label(n):
    S, M = _S_qn[n], _M_qn[n]
    if abs(M - 1.5) < 0.1:    Mstr = '+3/2'
    elif abs(M + 1.5) < 0.1:  Mstr = '-3/2'
    elif abs(M - 0.5) < 0.1:  Mstr = '+1/2'
    else:                       Mstr = '-1/2'
    if abs(S - 1.5) < 0.1:
        return rf'$|3/2,{Mstr}\rangle$'
    sub = r'\alpha' if round(_E_Hz_arr[n], 1) == _s12_Es[0] else r'\beta'
    return rf'$|1/2_{{{sub}}},{Mstr}\rangle$'

state_labels = [_state_label(n) for n in range(len(evals))]


# =============================================================================
# SECTION 8: RANK-2 IST OPERATORS T_{2,k}^{ij} AND COUPLING OPERATORS Q_{k,m}
# =============================================================================

def T2_op(i, j, k):
    """
    Rank-2 irreducible spherical tensor T_{2,k}^{ij} for spin pair (i, j).
    Spin operators are dimensionless (eigenvalues +/-1/2).

    k =  0 :  (1/sqrt6)(2 Iz_i Iz_j - Ix_i Ix_j - Iy_i Iy_j)
    k = +1 :  -(1/2)(I+_i Iz_j + Iz_i I+_j)
    k = -1 :  +(1/2)(I-_i Iz_j + Iz_i I-_j)
    k = +2 :  (1/2) I+_i I+_j
    k = -2 :  (1/2) I-_i I-_j
    """
    if k == 0:
        return (2*Iz[i]*Iz[j] - Ix[i]*Ix[j] - Iy[i]*Iy[j]) / np.sqrt(6)
    elif k == 1:
        return -(Ip[i]*Iz[j] + Iz[i]*Ip[j]) / 2
    elif k == -1:
        return  (Im[i]*Iz[j] + Iz[i]*Im[j]) / 2
    elif k == 2:
        return  Ip[i]*Ip[j] / 2
    elif k == -2:
        return  Im[i]*Im[j] / 2
    raise ValueError(f"k must be in {{-2,-1,0,1,2}}, got {k}")


def dipolar_tensor(r_vec):
    """Traceless symmetric dipolar tensor A_{ab} = 3*rhat_a*rhat_b - delta_{ab}."""
    rhat = r_vec / np.linalg.norm(r_vec)
    return 3.0 * np.outer(rhat, rhat) - np.eye(3)


def a2m_factors(r_vec):
    """
    Orientational factors a_{2,m} (m = -2,-1,0,+1,+2) for a spin pair with
    displacement r_vec, following eqs. (alm_start)-(alm_end) in the notes.

    For the symmetric dipolar tensor A (A_ab = A_ba):
      a_{2, 0} = (1/sqrt6)(2 A_zz - A_xx - A_yy)
      a_{2,+1} = -(A_xz - i A_yz)
      a_{2,-1} = +(A_xz + i A_yz)
      a_{2,+2} = (A_xx - A_yy - 2i A_xy) / 2
      a_{2,-2} = (A_xx - A_yy + 2i A_xy) / 2

    Sanity check: for a pair along z, a_{2,0} = sqrt(6), all others vanish,
    and Q_{0,0} = b_ij*(3 Iz_i Iz_j - I_i.I_j), the secular dipolar Hamiltonian.
    """
    A = dipolar_tensor(r_vec)
    return {
         0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
    }


def J_spectral(omega):
    """
    Lorentzian spectral density for isotropic tumbling.
    J(omega) = (1/5) * tau_c / (1 + omega^2 * tau_c^2)
    The 1/5 prefactor is the orientational average <|D_{km}^{(2)}|^2> = 1/5.
    Units: seconds  (multiply by b_ij^2 in rad^2/s^2 to get rates in rad/s).
    """
    return 0.2 * tau_c / (1.0 + (omega * tau_c)**2)


# Spin-pair table: (spin index i, spin index j, dipolar coupling b_ij, r_ij)
PAIRS = [
    (0, 1, b_FC, r_FC),   # 19F – 13C
    (0, 2, b_FH, r_FH),   # 19F – 1H
    (1, 2, b_CH, r_CH),   # 13C – 1H
]

# Orientational factors a_{2,m}^{ij} for each pair, computed from DFT geometry
alm = {(i, j): a2m_factors(r_vec) for i, j, _, r_vec in PAIRS}

# Q_{k,m} = sum_{pairs ij} b_ij * a_{2,m}^{ij} * T_{2,k}^{ij}
# These are the 25 system operators that couple to the bath (one per (k,m) mode).
# For a rigidly tumbling isotropic molecule, distinct (k,m) components of the
# Wigner D-matrix are uncorrelated bath modes sharing the same spectral density
# J(omega), so all 25 enter the Redfield tensor independently.
Q_ops = {
    (k, m): sum(b * alm[(i, j)][m] * T2_op(i, j, k)
                for i, j, b, _ in PAIRS)
    for k in range(-2, 3)
    for m in range(-2, 3)
}


# =============================================================================
# SECTION 9: JUMP OPERATORS, STRUCTURE MATRIX C, AND Γ± MATRICES
#
# Jump operators: sigma_{nm} = |n><m|, indexed as i = n*n_states + m
# Transition frequency: omega_{nm} = E_m - E_n  (rad/s)
#
# Matrix elements of Q operators in eigenbasis:
#   M[alpha, i] = <E_n|Q_alpha|E_m>  =  Tr{ sigma_{nm}^dag  Q_alpha }
#
# Structure matrix:
#   C[i,j] = sum_alpha  M[alpha,i]^*  M[alpha,j]
#
# Rate matrices (eq:Red_Gamma coefficients):
#   Gamma_minus[i,j] = (1/4) C[i,j] (J(omega_j) - J(omega_i))   [commutator]
#   Gamma_plus[i,j]  = (1/4) C[i,j] (J(omega_j) + J(omega_i))   [Lindblad-type]
# =============================================================================

n_states = len(evals)   # 8 for a 3-spin-1/2 system
n_trans  = n_states ** 2   # 64 ordered transitions

# Transition frequencies omega_{nm} = E_m - E_n  (rad/s)
omega_trans = np.array([evals[m_] - evals[n_]
                        for n_ in range(n_states)
                        for m_ in range(n_states)])   # shape (64,)

# Eigenvector array: ekets_arr[n, k] = k-th component of |E_n> in standard basis
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)  # (8, 8)

# Stack all 25 Q_{k,m} matrices into a single array for vectorised contraction
KM_LIST  = [(k, m) for k in range(-2, 3) for m in range(-2, 3)]   # 25 modes
n_modes  = len(KM_LIST)
Q_stack  = np.array([Q_ops[km].full() for km in KM_LIST], dtype=complex)  # (25, 8, 8)

# M_3d[alpha, n, m] = <E_n|Q_alpha|E_m>  (all indices simultaneously via einsum)
M_3d    = np.einsum('nk,akl,ml->anm', ekets_arr.conj(), Q_stack, ekets_arr)  # (25, 8, 8)
M_matrix = M_3d.reshape(n_modes, n_trans)   # (25, 64)

# Structure matrix C_{ij} = sum_alpha M[alpha,i]^* M[alpha,j]
# C is Hermitian PSD; its magnitude encodes geometric coupling between transitions
C_matrix = M_matrix.conj().T @ M_matrix   # (64, 64)

# Spectral density evaluated at each transition frequency
J_omegas = np.array([J_spectral(w) for w in omega_trans])   # (64,)

# Build Γ± via broadcasting:  shape (64, 64)
dJ = J_omegas[np.newaxis, :] - J_omegas[:, np.newaxis]   # J(omega_j) - J(omega_i)
sJ = J_omegas[np.newaxis, :] + J_omegas[:, np.newaxis]   # J(omega_j) + J(omega_i)

Gamma_minus = 0.5 * C_matrix * dJ   # coefficient of [sigma_i^dag sigma_j, rho]
Gamma_plus  = 0.5 * C_matrix * sJ   # coefficient of -{...} + 2 sigma_j rho sigma_i^dag


# =============================================================================
# SECTION 9b: FERMI'S GOLDEN RULE POPULATION TRANSFER RATES
#
# From the FGR subsection of the notes:
#   k_{i->f} = J(omega_fi) * sum_{k,m} |<f|Q_{k,m}|i>|^2
#
# M_3d[alpha, n, m] = <E_n|Q_alpha|E_m> is already computed above.
# So <f|Q_alpha|i> = M_3d[alpha, f, i], giving:
#   _Q2_fi[f, i] = sum_alpha |<E_f|Q_alpha|E_i>|^2
# =============================================================================

# sum_alpha |<E_f|Q_alpha|E_i>|^2  — shape (n_states, n_states): rows=f, cols=i
_Q2_fi = np.sum(np.abs(M_3d)**2, axis=0)

# omega_fi = E_f - E_i  (rad/s), shape (n_states, n_states)
_omega_fi_mat = evals[:, np.newaxis] - evals[np.newaxis, :]

# Spectral density at each (f, i) transition frequency
_J_fi_mat = np.vectorize(J_spectral)(_omega_fi_mat)

# FGR rate matrix (rad/s): fgr_rates[f, i] = transfer rate from state i to state f
fgr_rates = _J_fi_mat * _Q2_fi
np.fill_diagonal(fgr_rates, 0.0)   # no self-transitions


# =============================================================================
# SECTION 10: LIOUVILLIAN CONSTRUCTION, INITIAL STATE, AND PROPAGATOR
# =============================================================================

# Initial state  |↑↑↑⟩  =  |↑⟩_F ⊗ |↑⟩_C ⊗ |↑⟩_H
_up  = qt.basis(2, 0)
rho0 = qt.ket2dm(qt.tensor(_up, _up, _up))

# Vectorise rho0 using QuTiP's own convention (consistent with L_total)
_rho0_vec_qobj = qt.operator_to_vector(rho0)
_vec_dims       = _rho0_vec_qobj.dims
_vec_shape      = _rho0_vec_qobj.shape


def build_liouvillian(gamma_scale=1.0):
    """Build the Liouvillian superoperator from the full Redfield equation.

    Implements the first-line double-commutator form (sec. 9.2 of the notes):

        L[ρ] = −i[H₀, ρ] + R[ρ]
        R[ρ]  = −∑_{i,j} Γᵢⱼ(ωⱼ) [σᵢ, [σⱼ†, ρ]]
        Γᵢⱼ(ωⱼ) = C[i,j] · J(ωⱼ)

    The sum runs over ALL 64×64 transition-operator pairs (i,j); no secular
    or block-diagonal approximation is applied.

    The dissipator is built in two steps:

    Step 1 — construct L_diss in the EIGENSTATE Liouville basis, where the
    transition operators σᵢ = |E_{nᵢ}⟩⟨E_{mᵢ}| are simple rank-1 matrices
    with a single nonzero element.  In this basis the 4-index formula is exact:

        L4[a,b,c,d] = −δ(b,d)·T1[a,c] + GR[a,c,b,d] + GR[d,b,c,a] − δ(a,c)·T4[b,d]

    Step 2 — transform to the COMPUTATIONAL basis used by QuTiP, via the
    unitary  V = kron(U*, U)  where  U = ekets_arr.T  is the eigenvector matrix:

        L_diss_comp = V · L_diss_eig · V†

    Parameters
    ----------
    gamma_scale : float
        Uniform scale applied to all relaxation rates.

    Returns
    -------
    L_mat        : (64,64) complex ndarray
    L_evals      : (64,) complex ndarray
    L_evecs      : (64,64) complex ndarray — column eigenvectors
    L_evecs_inv  : (64,64) complex ndarray
    c0           : (64,) complex ndarray  — initial coefficients in eigenbasis
    """
    # Hamiltonian part (computational basis)
    L_ham_mat = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

    # Γᵢⱼ(ωⱼ) = C[i,j] · J(ωⱼ) · gamma_scale
    Gamma_raw = C_matrix * J_omegas[np.newaxis, :] * gamma_scale  # (64, 64)

    # --- Step 1: L_diss in eigenstate Liouville basis ---
    # GR[nᵢ, mᵢ, nⱼ, mⱼ] = Gamma_raw[nᵢ·nₛ+mᵢ, nⱼ·nₛ+mⱼ]
    GR = Gamma_raw.reshape(n_states, n_states, n_states, n_states)

    T1 = np.einsum('amcm->ac', GR)   # ∑ₘ GR[a,m,c,m]
    T4 = np.einsum('nbnd->bd', GR)   # ∑ₙ GR[n,b,n,d]

    L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
            + np.einsum('acbd->abcd', GR)
            + np.einsum('dbca->abcd', GR)
            - np.einsum('ac,bd->abcd', np.eye(n_states), T4))

    # Column-major reshape: L_diss_eig[a + b·nₛ, c + d·nₛ] = L_4d[a,b,c,d]
    L_diss_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_states**2, n_states**2)

    # --- Step 2: rotate to computational basis ---
    # U[a, n] = n-th eigenstate's a-th component in computational basis
    # For column-major vec: vec(UρU†) = kron(U*, U) @ vec(ρ)
    U = ekets_arr.T                    # shape (n_states, n_states)
    V = np.kron(U.conj(), U)           # shape (n_states², n_states²), unitary
    L_diss_mat = V @ L_diss_eig @ V.conj().T

    _mat      = L_ham_mat + L_diss_mat
    _ev, _vec = np.linalg.eig(_mat)
    _vec_inv  = np.linalg.inv(_vec)
    _c0       = _vec_inv @ _rho0_vec_qobj.full().flatten()
    return _mat, _ev, _vec, _vec_inv, _c0


def build_liouvillian_IST(gamma_scale=1.0):
    """Build the Liouvillian with jump operators in the two-spin rank-2 IST basis.

    Projects the eigenstate transition operators sigma_j = |E_n><E_m| onto the
    15 normalized two-spin rank-2 IST operators

        p_delta = T^{(2,k)}_{ij} / ||T^{(2,k)}_{ij}||_HS

    for all spin pairs (i,j) in {(0,1),(0,2),(1,2)} and k in {-2,...,+2}.

    The projected master equation (eq:chg_bas) is:
        rho_dot = -i[H, rho]
                  - sum_{mu,delta} Delta_minus[mu,delta] [p_mu^dag p_delta, rho]
                  + sum_{mu,delta} Delta_plus[mu,delta]  (-{p_mu^dag p_delta, rho}
                                                          + 2 p_delta rho p_mu^dag)
    where:
        c[j, delta]  = Tr(p_delta^dag  sigma_j)    (projection coefficients)
        Delta_minus  = c^H  Gamma_minus  c           (15x15)
        Delta_plus   = c^H  Gamma_plus   c           (15x15)

    The Liouville matrix is built in the same column-major QuTiP convention as
    build_liouvillian(), so the returned tuple is drop-in compatible with rho_at().
    """
    L_ham_mat = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

    # --- Step 1: Normalized two-spin rank-2 IST operators ---
    IST_pairs = [(0, 1), (0, 2), (1, 2)]
    IST_ks    = [-2, -1, 0, 1, 2]

    P_raw    = [T2_op(si, sj, k).full()
                for (si, sj) in IST_pairs for k in IST_ks]
    P_arr    = np.array(P_raw, dtype=complex)   # (15, 8, 8)
    hs_norms = np.sqrt(np.array(
        [np.real(np.trace(p.conj().T @ p)) for p in P_arr]
    ))
    P_norm   = P_arr / hs_norms[:, np.newaxis, np.newaxis]   # unit H-S norm
    n_IST    = len(P_norm)   # 15

    # --- Step 2: Transition operators sigma_j = |E_n><E_m| (computational basis) ---
    sigma_arr = np.array([
        np.outer(ekets_arr[n_], ekets_arr[m_].conj())
        for n_ in range(n_states)
        for m_ in range(n_states)
    ], dtype=complex)   # (64, 8, 8)

    # --- Step 3: Projection coefficients c[j, delta] = Tr(p_delta^dag sigma_j) ---
    # Frobenius inner product: sum_{a,b} p_delta[a,b]^* sigma_j[a,b]
    c_mat = np.einsum('dab,jab->jd', P_norm.conj(), sigma_arr)   # (64, 15)

    # --- Step 4: Delta matrices  Delta_pm = c^H Gamma_pm c  (15x15) ---
    Gm = Gamma_minus * gamma_scale
    Gp = Gamma_plus  * gamma_scale

    Delta_minus = c_mat.conj().T @ Gm @ c_mat   # (15, 15)
    Delta_plus  = c_mat.conj().T @ Gp @ c_mat   # (15, 15)

    # --- Step 5: IST dissipator in Liouville space ---
    # Column-major (QuTiP) Liouville representations:
    #   spre(A)        -> kron(I, A)          rho -> A rho
    #   spost(A)       -> kron(A.T, I)        rho -> rho A   (.T = plain transpose)
    #   sprepost(A, B) -> kron(B.T, A)        rho -> A rho B
    #
    # For pair (mu, delta) with P_md = p_mu^dag @ p_delta:
    #   commutator:  -dm * [P_md, rho]
    #     = -dm * (kron(I, P_md) - kron(P_md.T, I))
    #   Lindblad:    dp * (-{P_md, rho} + 2 p_delta rho p_mu^dag)
    #     = dp * (-kron(I, P_md) - kron(P_md.T, I) + 2 kron(p_mu.conj(), p_delta))
    #   where sprepost(p_delta, p_mu^dag) -> kron((p_mu^dag).T, p_delta)
    #                                      = kron(p_mu.conj(), p_delta)

    I8 = np.eye(n_states)
    L_IST_diss = np.zeros((n_states**2, n_states**2), dtype=complex)

    for mu in range(n_IST):
        p_mu     = P_norm[mu]
        p_mu_dag = p_mu.conj().T
        for delta in range(n_IST):
            p_delta = P_norm[delta]
            P_md    = p_mu_dag @ p_delta   # (8, 8)
            dm = Delta_minus[mu, delta]
            dp = Delta_plus[mu, delta]

            L_IST_diss += -dm * (np.kron(I8, P_md) - np.kron(P_md.T, I8))
            L_IST_diss += dp  * (-np.kron(I8, P_md) - np.kron(P_md.T, I8)
                                 + 2.0 * np.kron(p_mu.conj(), p_delta))

    _mat      = L_ham_mat + L_IST_diss
    _ev, _vec = np.linalg.eig(_mat)
    _vec_inv  = np.linalg.inv(_vec)
    _c0       = _vec_inv @ _rho0_vec_qobj.full().flatten()
    return _mat, _ev, _vec, _vec_inv, _c0


def build_canonical_IST(gamma_scale=1.0):
    """Canonical Lindblad form obtained by diagonalising Delta_plus.

    Delta_plus (15x15) is Hermitian PSD; its eigendecomposition

        Delta_plus = U diag(d) U^dagger    d_k >= 0

    defines canonical Lindblad operators

        L_k = sum_delta  U*[delta,k]  p_delta

    so that the Lindblad-type part of the dissipator reads

        2 sum_k d_k ( L_k rho L_k^dag  -  1/2 {L_k^dag L_k, rho} )

    which is the standard diagonal GKS-Lindblad form.

    The commutator (Delta_minus) term is left in the IST basis, unchanged.

    Returns
    -------
    d_vals      : (15,) real ndarray, eigenvalues of Delta_plus (ascending)
    L_ops       : (15, 8, 8) complex ndarray, canonical Lindblad operators
    Delta_plus  : (15, 15) Hermitian PSD matrix
    Delta_minus : (15, 15) anti-Hermitian matrix
    L_mat       : (64, 64) Liouvillian (canonical form, numerically equal to IST)
    evals, evecs, evecs_inv, c0 : eigensystem for propagation
    """
    L_ham_mat = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

    # Rebuild IST components (same setup as build_liouvillian_IST)
    IST_pairs = [(0, 1), (0, 2), (1, 2)]
    IST_ks    = [-2, -1, 0, 1, 2]
    P_arr     = np.array([T2_op(si, sj, k).full()
                          for (si, sj) in IST_pairs for k in IST_ks],
                         dtype=complex)   # (15, 8, 8)
    hs_norms  = np.sqrt(np.array(
        [np.real(np.trace(p.conj().T @ p)) for p in P_arr]
    ))
    P_norm    = P_arr / hs_norms[:, np.newaxis, np.newaxis]
    n_IST     = len(P_norm)   # 15

    sigma_arr = np.array([
        np.outer(ekets_arr[n_], ekets_arr[m_].conj())
        for n_ in range(n_states) for m_ in range(n_states)
    ], dtype=complex)
    c_mat = np.einsum('dab,jab->jd', P_norm.conj(), sigma_arr)   # (64, 15)

    Gm = Gamma_minus * gamma_scale
    Gp = Gamma_plus  * gamma_scale
    Delta_minus = c_mat.conj().T @ Gm @ c_mat   # (15, 15)
    Delta_plus  = c_mat.conj().T @ Gp @ c_mat   # (15, 15)

    # Diagonalise Delta_plus via eigh (guaranteed real eigenvalues, unitary U)
    d_vals, U = np.linalg.eigh(Delta_plus)   # d_vals ascending; U[:,k] = eigenvector k

    # Canonical Lindblad operators: L_k = sum_delta U*[delta,k] p_delta
    # Proof: sum_{mu,delta} Delta+[mu,delta] p_delta rho p_mu^dag
    #      = sum_k d_k (sum_delta U*[delta,k] p_delta) rho (sum_mu U[mu,k] p_mu)^dag
    #      = sum_k d_k L_k rho L_k^dag
    L_ops = np.einsum('dk,dab->kab', U.conj(), P_norm)   # (15, 8, 8)

    # Build canonical Liouvillian:
    #   commutator term  — identical to IST form (Delta_minus, P_norm unchanged)
    #   Lindblad term    — sum_k 2*d_k * D_k[rho]  with standard D_k formula
    I8 = np.eye(n_states)
    L_can_diss = np.zeros((n_states**2, n_states**2), dtype=complex)

    # Commutator: -sum_{mu,delta} Delta_minus[mu,delta] [p_mu^dag p_delta, rho]
    for mu in range(n_IST):
        p_mu_dag = P_norm[mu].conj().T
        for delta in range(n_IST):
            P_md = p_mu_dag @ P_norm[delta]
            dm   = Delta_minus[mu, delta]
            L_can_diss += -dm * (np.kron(I8, P_md) - np.kron(P_md.T, I8))

    # Canonical Lindblad: 2*d_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
    for k in range(n_IST):
        Lk    = L_ops[k]          # (8, 8)
        LdL   = Lk.conj().T @ Lk  # L_k^dag L_k
        dk    = d_vals[k]
        # sprepost(Lk, Lk^dag) -> kron((Lk^dag).T, Lk) = kron(Lk.conj(), Lk)
        L_can_diss += 2.0 * dk * (
            np.kron(Lk.conj(), Lk)
            - 0.5 * np.kron(I8, LdL)
            - 0.5 * np.kron(LdL.T, I8)
        )

    _mat      = L_ham_mat + L_can_diss
    _ev, _vec = np.linalg.eig(_mat)
    _vec_inv  = np.linalg.inv(_vec)
    _c0       = _vec_inv @ _rho0_vec_qobj.full().flatten()
    return d_vals, U, L_ops, Delta_plus, Delta_minus, _mat, _ev, _vec, _vec_inv, _c0


def build_approx_liouvillian(d_vals, L_ops, indices):
    """Approximate Liouvillian: Hamiltonian + selected canonical Lindblad terms only.

    Delta_minus (commutator) is set to zero; only the canonical operators whose
    indices appear in `indices` are included.

    Parameters
    ----------
    d_vals  : (n_IST,) eigenvalues of Delta_plus from build_canonical_IST
    L_ops   : (n_IST, 8, 8) canonical Lindblad operators
    indices : sequence of ints selecting which operators to keep
    """
    L_ham_mat = (-1j * (qt.spre(H0) - qt.spost(H0))).full()
    I8        = np.eye(n_states)
    L_diss    = np.zeros((n_states**2, n_states**2), dtype=complex)

    for k in indices:
        Lk  = L_ops[k]
        LdL = Lk.conj().T @ Lk
        dk  = d_vals[k]
        L_diss += 2.0 * dk * (
            np.kron(Lk.conj(), Lk)
            - 0.5 * np.kron(I8, LdL)
            - 0.5 * np.kron(LdL.T, I8)
        )

    _mat      = L_ham_mat + L_diss
    _ev, _vec = np.linalg.eig(_mat)
    _vec_inv  = np.linalg.inv(_vec)
    _c0       = _vec_inv @ _rho0_vec_qobj.full().flatten()
    return _mat, _ev, _vec, _vec_inv, _c0


# Default (gamma_scale = 1.0) — used by the module-level rho_at()
L_mat, _L_evals, _L_evecs, _L_evecs_inv, _c0 = build_liouvillian(gamma_scale=1.0)


def rho_at(t):
    """Density matrix Qobj at time t [seconds], gamma_scale=1.0."""
    _v = _L_evecs @ (np.exp(_L_evals * t) * _c0)
    return qt.vector_to_operator(
        qt.Qobj(_v.reshape(_vec_shape), dims=_vec_dims)
    )


if __name__ == '__main__':
    print()
    print("=" * 55)
    print("Parameter summary")
    print("=" * 55)
    print(f"  Nuclei (spin order): {NUCLEI}")
    print(f"  B0                 = {B0} T  (ZULF)")
    print(f"  tau_c              = {tau_c:.1e} s")
    print()
    print("  Gyromagnetic ratios gamma / (2pi)  (MHz/T):")
    for name, g in GAMMA.items():
        print(f"    {name:>4s}:  {g/(2*np.pi)*1e-6:.4f}")
    print()
    print("  Larmor frequencies at B0 = 0 T: all zero (ZULF)")
    print()
    print("  Scalar couplings:")
    print(f"    J(19F-13C) = {J_FC:.2f} Hz")
    print(f"    J(13C-1H)  = {J_CH:.2f} Hz")
    print(f"    J(19F-1H)  = {J_FH:.2f} Hz")
    print()
    print("  Coordinates (Angstrom):")
    for name, xyz in coords_ang.items():
        print(f"    {name:>4s}:  {xyz}")

    # ------------------------------------------------------------------
    # H0 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("H0 eigenvalues")
    print("=" * 55)
    for n, e in enumerate(evals):
        print(f"  |{n}>  {e:+10.4f} rad/s   ({e/(2*np.pi):+10.4f} Hz)")
    print()
    print("  Transition frequencies |omega_mn| = |E_m - E_n|  (Hz):")
    for mi in range(len(evals)):
        for ni in range(mi + 1, len(evals)):
            dE = abs(evals[ni] - evals[mi]) / (2 * np.pi)
            print(f"    |{mi}>-|{ni}> :  {dE:.4f} Hz")

    # ------------------------------------------------------------------
    # Q_{k,m} Frobenius norm grid — reveals which (k,m) modes couple most
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("||Q_{k,m}||_F  [rad/s]   (Frobenius norm)")
    print("Rows k = -2..+2,  Cols m = -2..+2")
    print("=" * 55)
    norm_grid = np.array([
        [np.sqrt(abs((Q_ops[(k, m)].dag() * Q_ops[(k, m)]).tr()))
         for m in range(-2, 3)]
        for k in range(-2, 3)
    ])
    header = "       " + "  ".join(f"m={m:+d}" for m in range(-2, 3))
    print(header)
    for ki, k in enumerate(range(-2, 3)):
        row = "  ".join(f"{norm_grid[ki, mi]:.3e}" for mi in range(5))
        print(f"  k={k:+d} :  {row}")

    # ------------------------------------------------------------------
    # Gamma matrix statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Gamma matrix statistics")
    print("=" * 55)
    gp_max = np.abs(Gamma_plus).max()
    gm_max = np.abs(Gamma_minus).max()
    print(f"  max |Gamma_+| = {gp_max:.4e} rad/s")
    print(f"  max |Gamma_-| = {gm_max:.4e} rad/s")
    print(f"  |Gamma_-| / |Gamma_+| ratio = {gm_max / (gp_max + 1e-300):.4e}")
    print("  (At ZULF extreme-narrowing omega*tau_c << 1 => Gamma_- << Gamma_+)")

    # Identify the 5 largest entries in Gamma_plus (most active channels)
    flat_idx = np.argsort(np.abs(Gamma_plus).ravel())[::-1][:5]
    print()
    print("  Top-5 |Gamma_+| entries  (transition i -> transition j):")
    for rank, idx in enumerate(flat_idx):
        ii, jj = divmod(idx, n_trans)
        ni, mi = divmod(ii, n_states)
        nj, mj = divmod(jj, n_states)
        val = Gamma_plus[ii, jj]
        omega_i_hz = omega_trans[ii] / (2*np.pi)
        omega_j_hz = omega_trans[jj] / (2*np.pi)
        print(f"    [{rank+1}] i=({ni},{mi}) omega={omega_i_hz:+.1f}Hz  "
              f"j=({nj},{mj}) omega={omega_j_hz:+.1f}Hz  "
              f"|Gamma_+|={abs(val):.4e}")

    # ------------------------------------------------------------------
    # Fermi's Golden Rule rates — top 5
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Fermi's Golden Rule: top 5 population transfer rates")
    print("k_{i->f} = J(omega_fi) * sum_{k,m} |<f|Q_{km}|i>|^2")
    print("=" * 55)

    _top5_flat = np.argsort(fgr_rates.ravel())[::-1][:5]

    def _plain(lbl):
        """Strip LaTeX markup for terminal display."""
        return (lbl.replace('$', '')
                   .replace(r'\alpha', 'a').replace(r'\beta', 'b')
                   .replace(r'\rangle', '>').replace(r'\langle', '<')
                   .replace(r'\,', ','))

    print(f"  {'#':3}  {'Initial |i>':28}  {'Final |f>':28}  "
          f"{'omega_fi/2pi (Hz)':18}  {'k/(2pi) (Hz)':12}")
    print("  " + "-" * 98)
    fgr_top5 = []
    for rank, flat_i in enumerate(_top5_flat):
        f_idx, i_idx = divmod(flat_i, n_states)
        omega_fi_hz = _omega_fi_mat[f_idx, i_idx] / (2 * np.pi)
        rate_rads   = fgr_rates[f_idx, i_idx]
        rate_hz     = rate_rads / (2 * np.pi)
        li = state_labels[i_idx]
        lf = state_labels[f_idx]
        print(f"  [{rank+1}]  {_plain(li):28}  {_plain(lf):28}  "
              f"{omega_fi_hz:+14.2f}       {rate_hz:.4e}")
        fgr_top5.append((rank + 1, li, lf, omega_fi_hz, rate_rads, rate_hz))

    # ------------------------------------------------------------------
    # FGR rates from |3/2, +3/2> (= initial |↑↑↑⟩ state)
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("FGR rates: initial state |3/2, +3/2> only")
    print("=" * 55)

    _uuu_idx = next(n for n in range(n_states)
                    if abs(_S_qn[n] - 1.5) < 0.1 and abs(_M_qn[n] - 1.5) < 0.1)
    print(f"  |3/2,+3/2> = eigenstate index {_uuu_idx}  "
          f"(E = {evals[_uuu_idx]/(2*np.pi):+.4f} Hz)")

    _fgr_col = fgr_rates[:, _uuu_idx].copy()
    _fgr_col[_uuu_idx] = 0.0   # remove self-entry
    _top5_uuu = np.argsort(_fgr_col)[::-1][:5]

    print(f"  {'#':3}  {'Final |f>':28}  "
          f"{'omega_fi/2pi (Hz)':18}  {'k/(2pi) (Hz)':12}")
    print("  " + "-" * 70)
    fgr_top5_uuu = []
    for rank, f_idx in enumerate(_top5_uuu):
        omega_fi_hz = _omega_fi_mat[f_idx, _uuu_idx] / (2 * np.pi)
        rate_rads   = _fgr_col[f_idx]
        rate_hz     = rate_rads / (2 * np.pi)
        lf = state_labels[f_idx]
        print(f"  [{rank+1}]  {_plain(lf):28}  "
              f"{omega_fi_hz:+14.2f}       {rate_hz:.4e}")
        fgr_top5_uuu.append((rank + 1, lf, omega_fi_hz, rate_rads, rate_hz))

    # ------------------------------------------------------------------
    # Colorplots of |Gamma_+| and |Gamma_-|
    # ------------------------------------------------------------------
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Redfield rate matrices: $^{19}$F–$^{13}$C–$^{1}$H, ZULF, '
        + r'$\tau_c$=' + f'{tau_c:.0e} s',
        fontsize=12
    )

    mats   = [np.abs(Gamma_plus),  np.abs(Gamma_minus)]
    titles = [r'$|\Gamma_+|$ — Lindblad coefficient  (rad/s)',
              r'$|\Gamma_-|$ — Commutator coefficient  (rad/s)']

    for ax, mat, title in zip(axes, mats, titles):
        vmax = mat.max()
        if vmax > 0:
            vmin = max(vmax * 1e-8, 1e-30)
            im = ax.imshow(mat, cmap='inferno', interpolation='nearest',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(mat, cmap='inferno', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='rad/s', fraction=0.046, pad=0.04)
        # Mark block boundaries (every n_states columns/rows = same source eigenstate)
        for pos in range(n_states, n_trans, n_states):
            ax.axhline(pos - 0.5, color='cyan', lw=0.4, alpha=0.5)
            ax.axvline(pos - 0.5, color='cyan', lw=0.4, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'Transition $j$ — $(n_j, m_j)$ row-major', fontsize=9)
        ax.set_ylabel(r'Transition $i$ — $(n_i, m_i)$ row-major', fontsize=9)
        ax.set_xticks(range(0, n_trans, n_states))
        ax.set_yticks(range(0, n_trans, n_states))
        ax.set_xticklabels(state_labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(state_labels, fontsize=7)

    plt.tight_layout()
    fpath = os.path.join(DATA_DIR, f'gamma_matrices_tc{tau_c:.0e}_B0{B0:.2f}.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Colorplot saved -> {fpath}")

    # ------------------------------------------------------------------
    # Liouvillian summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Liouvillian  L  (64x64 superoperator)")
    print("=" * 55)
    L_evals_re = np.real(_L_evals)
    L_evals_im = np.imag(_L_evals)
    n_zero = np.sum(np.abs(L_evals_re) < gp_max * 1e-8)
    print(f"  Eigenvalue summary:")
    print(f"    Most-negative real part  : {L_evals_re.min():.4e} rad/s")
    print(f"    Most-positive real part  : {L_evals_re.max():.4e} rad/s")
    print(f"    Num near-zero real parts : {n_zero}")
    print(f"    Max |imag part|          : {np.abs(L_evals_im).max():.4e} rad/s")

    # ------------------------------------------------------------------
    # Time propagation: populations in H0 eigenbasis
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Time propagation  (eigenbasis populations)")
    print("=" * 55)

    t_end   = 100e-6    # s — 100 microseconds
    N_steps = 500
    tlist   = np.linspace(0.0, t_end, N_steps)

    print(f"  Propagating to t = {t_end*1e6:.0f} μs  ({N_steps} steps)")

    pops      = np.zeros((N_steps, n_states))   # P_n(t) = <E_n|rho(t)|E_n>
    min_eig   = np.zeros(N_steps)               # minimum eigenvalue of rho(t)

    for ti, t in enumerate(tlist):
        rho_t    = rho_at(t)
        rho_mat  = rho_t.full()
        for n_ in range(n_states):
            pops[ti, n_] = np.real(
                ekets_arr[n_].conj() @ rho_mat @ ekets_arr[n_]
            )
        min_eig[ti] = np.linalg.eigvalsh(rho_mat).min()

    print(f"  Final populations (t = {t_end:.2e} s):")
    for n_ in range(n_states):
        print(f"    |E_{n_}> : P = {pops[-1, n_]:.6f}")
    print(f"  Trace conservation: sum P_n at t=0 = {pops[0].sum():.8f}, "
          f"at t_end = {pops[-1].sum():.8f}")
    print(f"  Min eigenvalue of rho: initial = {min_eig[0]:.4e}, "
          f"final = {min_eig[-1]:.4e}")
    if min_eig.min() < -1e-10:
        print(f"  WARNING: positivity violation — min eig = {min_eig.min():.4e}")
    else:
        print("  Positivity maintained throughout (min eig >= 0)")

    # ------------------------------------------------------------------
    # Population dynamics plot
    # ------------------------------------------------------------------
    tlist_us = tlist * 1e6   # convert to microseconds for readability

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        r'Population dynamics: $^{19}$F–$^{13}$C–$^{1}$H ZULF, '
        + r'initial state $|\!\uparrow\uparrow\uparrow\rangle$, '
        + r'$\tau_c$=' + f'{tau_c:.0e} s',
        fontsize=11
    )

    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    for n_ in range(n_states):
        ax1.plot(tlist_us, pops[:, n_], color=colors[n_],
                 label=state_labels[n_],
                 lw=1.5)
    ax1.set_ylabel('Population', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.axhline(1.0/n_states, color='gray', ls='--', lw=0.8, alpha=0.6,
                label='equipartition')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Eigenbasis populations', fontsize=10)

    ax2.plot(tlist_us, min_eig, color='crimson', lw=1.5)
    ax2.axhline(0.0, color='k', ls='--', lw=0.8)
    ax2.set_ylabel(r'min eigenvalue of $\rho(t)$', fontsize=11)
    ax2.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Positivity check', fontsize=10)

    plt.tight_layout()
    pop_fpath = os.path.join(DATA_DIR,
                             f'populations_tc{tau_c:.0e}_B0{B0:.2f}.png')
    plt.savefig(pop_fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Population plot saved -> {pop_fpath}")

    # ------------------------------------------------------------------
    # Repeat with gamma_scale = 2.0  (Spinach-matched rates)
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Time propagation  (gamma_scale = 2.0, Spinach-matched)")
    print("=" * 55)

    _, _L_ev2, _L_evec2, _, _c0_2 = build_liouvillian(gamma_scale=2.0)

    def _rho_at_2x(t):
        _v = _L_evec2 @ (np.exp(_L_ev2 * t) * _c0_2)
        return qt.vector_to_operator(
            qt.Qobj(_v.reshape(_vec_shape), dims=_vec_dims)
        )

    print(f"  Propagating to t = {t_end*1e6:.0f} μs  ({N_steps} steps)")

    pops_2x    = np.zeros((N_steps, n_states))
    min_eig_2x = np.zeros(N_steps)

    for ti, t in enumerate(tlist):
        rho_t   = _rho_at_2x(t)
        rho_mat = rho_t.full()
        for n_ in range(n_states):
            pops_2x[ti, n_] = np.real(
                ekets_arr[n_].conj() @ rho_mat @ ekets_arr[n_]
            )
        min_eig_2x[ti] = np.linalg.eigvalsh(rho_mat).min()

    print(f"  Final populations (t = {t_end:.2e} s):")
    for n_ in range(n_states):
        print(f"    |E_{n_}> : P = {pops_2x[-1, n_]:.6f}")
    print(f"  Trace conservation: sum P_n at t=0 = {pops_2x[0].sum():.8f}, "
          f"at t_end = {pops_2x[-1].sum():.8f}")
    if min_eig_2x.min() < -1e-10:
        print(f"  WARNING: positivity violation — min eig = {min_eig_2x.min():.4e}")
    else:
        print("  Positivity maintained throughout (min eig >= 0)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        r'Population dynamics ($\Gamma\times 2$, Spinach-matched): '
        r'$^{19}$F–$^{13}$C–$^{1}$H ZULF, '
        + r'initial state $|\!\uparrow\uparrow\uparrow\rangle$, '
        + r'$\tau_c$=' + f'{tau_c:.0e} s',
        fontsize=11
    )

    for n_ in range(n_states):
        ax1.plot(tlist_us, pops_2x[:, n_], color=colors[n_],
                 label=state_labels[n_], lw=1.5)
    ax1.set_ylabel('Population', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.axhline(1.0/n_states, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r'Eigenbasis populations  ($\Gamma \times 2$)', fontsize=10)

    ax2.plot(tlist_us, min_eig_2x, color='crimson', lw=1.5)
    ax2.axhline(0.0, color='k', ls='--', lw=0.8)
    ax2.set_ylabel(r'min eigenvalue of $\rho(t)$', fontsize=11)
    ax2.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Positivity check', fontsize=10)

    plt.tight_layout()
    pop_fpath_2x = os.path.join(DATA_DIR,
                                f'populations_2x_tc{tau_c:.0e}_B0{B0:.2f}.png')
    plt.savefig(pop_fpath_2x, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Population plot (gamma_scale=2.0) saved -> {pop_fpath_2x}")

    # ------------------------------------------------------------------
    # IST-basis Liouvillian: projection onto two-spin rank-2 T^{2,k}_{ij}
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("IST-basis Liouvillian  (two-spin rank-2 T^{2,k}_{ij})")
    print("=" * 55)

    # Build and report IST operator norms / projection quality
    _IST_pairs = [(0, 1), (0, 2), (1, 2)]
    _IST_ks    = [-2, -1, 0, 1, 2]
    _P_raw = [T2_op(si, sj, k).full()
              for (si, sj) in _IST_pairs for k in _IST_ks]
    _P_arr = np.array(_P_raw, dtype=complex)
    _hs_norms = np.sqrt(np.array(
        [np.real(np.trace(p.conj().T @ p)) for p in _P_arr]
    ))
    _P_norm = _P_arr / _hs_norms[:, np.newaxis, np.newaxis]
    _n_IST  = len(_P_norm)

    _sigma_arr = np.array([
        np.outer(ekets_arr[n_], ekets_arr[m_].conj())
        for n_ in range(n_states) for m_ in range(n_states)
    ], dtype=complex)
    _c_mat = np.einsum('dab,jab->jd', _P_norm.conj(), _sigma_arr)

    # Fraction of ||Gamma_plus||_F captured by the IST projection
    _Gp = Gamma_plus
    _Delta_p = _c_mat.conj().T @ _Gp @ _c_mat
    _captured = np.linalg.norm(_Delta_p, 'fro') / (np.linalg.norm(_Gp, 'fro') + 1e-300)

    pair_labels = ['(F,C)', '(F,H)', '(C,H)']
    print(f"  Pool: {_n_IST} operators  "
          f"({len(_IST_pairs)} pairs × {len(_IST_ks)} components)")
    print(f"  H-S norms of raw T^{{2,k}}_{{ij}}:")
    for idx, ((si, sj), k) in enumerate(
            [(p, k) for p in _IST_pairs for k in _IST_ks]):
        print(f"    {pair_labels[_IST_pairs.index((si,sj))]} k={k:+d} : "
              f"||T||_HS = {_hs_norms[idx]:.4f}")
    print(f"  ||Delta_plus||_F / ||Gamma_plus||_F = {_captured:.4f}  "
          f"(IST projection completeness)")

    # Build IST Liouvillian and propagate
    _, _L_ev_IST, _L_evec_IST, _, _c0_IST = build_liouvillian_IST(gamma_scale=1.0)

    _L_evals_IST_re = np.real(_L_ev_IST)
    _n_zero_IST = np.sum(np.abs(_L_evals_IST_re) < gp_max * 1e-8)
    print(f"\n  IST Liouvillian eigenvalue summary:")
    print(f"    Most-negative real part  : {_L_evals_IST_re.min():.4e} rad/s")
    print(f"    Most-positive real part  : {_L_evals_IST_re.max():.4e} rad/s")
    print(f"    Num near-zero real parts : {_n_zero_IST}")
    print(f"    Max |imag part|          : {np.abs(np.imag(_L_ev_IST)).max():.4e} rad/s")

    print(f"\n  Propagating to t = {t_end*1e6:.0f} μs  ({N_steps} steps)")

    pops_IST    = np.zeros((N_steps, n_states))
    min_eig_IST = np.zeros(N_steps)

    for ti, t in enumerate(tlist):
        _v      = _L_evec_IST @ (np.exp(_L_ev_IST * t) * _c0_IST)
        rho_IST = qt.vector_to_operator(
            qt.Qobj(_v.reshape(_vec_shape), dims=_vec_dims)
        )
        rho_mat_IST = rho_IST.full()
        for n_ in range(n_states):
            pops_IST[ti, n_] = np.real(
                ekets_arr[n_].conj() @ rho_mat_IST @ ekets_arr[n_]
            )
        min_eig_IST[ti] = np.linalg.eigvalsh(rho_mat_IST).min()

    print(f"  Final populations (t = {t_end:.2e} s):")
    for n_ in range(n_states):
        print(f"    |E_{n_}> : P_eig = {pops[-1, n_]:.6f}   P_IST = {pops_IST[-1, n_]:.6f}")
    print(f"  Trace (eig): t=0 {pops[0].sum():.8f}  t_end {pops[-1].sum():.8f}")
    print(f"  Trace (IST): t=0 {pops_IST[0].sum():.8f}  t_end {pops_IST[-1].sum():.8f}")
    if min_eig_IST.min() < -1e-10:
        print(f"  WARNING: IST positivity violation — min eig = {min_eig_IST.min():.4e}")
    else:
        print("  IST positivity maintained throughout (min eig >= 0)")

    dpops = pops - pops_IST   # (N_steps, n_states)
    max_dev = np.abs(dpops).max()
    print(f"  Max |Delta P_n| over all n and t : {max_dev:.4e}")

    # ------------------------------------------------------------------
    # Comparison plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        r'Eigenstate-basis vs IST-basis Liouvillian: population comparison'
        + '\n' + r'$^{19}$F–$^{13}$C–$^{1}$H ZULF, '
        + r'initial $|\!\uparrow\uparrow\uparrow\rangle$, '
        + r'$\tau_c$=' + f'{tau_c:.0e} s',
        fontsize=11
    )

    for n_ in range(n_states):
        lbl = state_labels[n_]
        ax1.plot(tlist_us, pops[:, n_],     color=colors[n_], lw=2.0, ls='-',
                 label=lbl)
        ax1.plot(tlist_us, pops_IST[:, n_], color=colors[n_], lw=1.2, ls='--')
    ax1.set_ylabel('Population $P_n(t)$', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper right',
               title='solid = eig-basis,  dashed = IST-basis')
    ax1.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.6,
                label='equipartition')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Populations: eigenstate-basis (solid) vs IST-basis (dashed)',
                  fontsize=10)

    for n_ in range(n_states):
        ax2.plot(tlist_us, dpops[:, n_], color=colors[n_], lw=1.5,
                 label=state_labels[n_])
    ax2.axhline(0.0, color='k', ls='--', lw=0.8)
    ax2.set_ylabel(r'$\Delta P_n = P_n^{\rm eig} - P_n^{\rm IST}$', fontsize=11)
    ax2.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
    ax2.legend(fontsize=8, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'Population difference $\Delta P_n(t)$  '
                  + f'(max = {max_dev:.2e})', fontsize=10)

    plt.tight_layout()
    pop_IST_fpath = os.path.join(
        DATA_DIR, f'populations_IST_tc{tau_c:.0e}_B0{B0:.2f}.png'
    )
    plt.savefig(pop_IST_fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  IST comparison plot saved -> {pop_IST_fpath}")

    # ------------------------------------------------------------------
    # Canonical Lindblad form: diagonalise Delta_plus
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Canonical Lindblad form  (Delta_plus diagonalised)")
    print("=" * 55)

    d_vals, U_eigh, L_ops, Delta_p, Delta_m, L_can_mat, _ev_can, _evec_can, _, _c0_can = \
        build_canonical_IST(gamma_scale=1.0)

    # --- Hermitian check ---
    herm_err = np.linalg.norm(Delta_p - Delta_p.conj().T, 'fro')
    herm_rel = herm_err / (np.linalg.norm(Delta_p, 'fro') + 1e-300)
    print(f"  Hermitian check  ||Delta+ - Delta+^H||_F / ||Delta+||_F = {herm_rel:.2e}")

    # --- PSD check: all eigenvalues from eigh should be >= 0 ---
    neg_eigs = d_vals[d_vals < 0]
    print(f"  Eigenvalue range : [{d_vals.min():.4e},  {d_vals.max():.4e}]  rad/s")
    if len(neg_eigs) == 0 or d_vals.min() > -np.abs(d_vals).max() * 1e-10:
        print(f"  PSD check        : PASS  (most-negative = {d_vals.min():.2e})")
    else:
        print(f"  PSD check        : FAIL  (negative eigenvalues: {neg_eigs})")

    # --- Eigenvalue spectrum ---
    print(f"\n  Delta_plus eigenvalues d_k  (ascending, rad/s):")
    n_sig = np.sum(d_vals > np.abs(d_vals).max() * 1e-10)
    for k, dk in enumerate(d_vals):
        tag = f"  *** significant (#{k+1})" if dk > np.abs(d_vals).max() * 1e-10 else ""
        print(f"    k={k:2d}:  d_k = {dk:.6e}{tag}")
    print(f"  Number of significant eigenvalues (> 1e-10 * max): {n_sig}")

    # --- Canonical Lindblad operator norms ---
    print(f"\n  Canonical Lindblad operator norms  ||L_k||_HS = sqrt(Tr(L_k^dag L_k)):")
    for k in range(len(d_vals)):
        hs2 = np.real(np.trace(L_ops[k].conj().T @ L_ops[k]))
        print(f"    k={k:2d}:  ||L_k||_HS = {np.sqrt(max(hs2,0)):.6f}  "
              f"  d_k = {d_vals[k]:.4e}")

    # --- Verify canonical Liouvillian == IST Liouvillian ---
    _L_IST_full, *_ = build_liouvillian_IST(gamma_scale=1.0)
    can_vs_ist = np.linalg.norm(L_can_mat - _L_IST_full, 'fro')
    rel_err    = can_vs_ist / (np.linalg.norm(_L_IST_full, 'fro') + 1e-300)
    print(f"\n  Canonical vs IST Liouvillian:")
    print(f"    ||L_can - L_IST||_F         = {can_vs_ist:.4e} rad/s")
    print(f"    ||L_can - L_IST||_F / ||L_IST||_F = {rel_err:.4e}  (machine precision = OK)")

    # ------------------------------------------------------------------
    # Save canonical jump operators for off-line comparison
    # ------------------------------------------------------------------
    # Rebuild normalised IST basis (same convention as build_canonical_IST)
    _save_IST_pairs = [(0, 1), (0, 2), (1, 2)]
    _save_IST_ks    = [-2, -1, 0, 1, 2]
    _save_P_arr = np.array(
        [T2_op(si, sj, k).full()
         for (si, sj) in _save_IST_pairs for k in _save_IST_ks],
        dtype=complex,
    )
    _save_hs_norms = np.sqrt(np.array(
        [np.real(np.trace(p.conj().T @ p)) for p in _save_P_arr]
    ))
    _save_P_norm = _save_P_arr / _save_hs_norms[:, np.newaxis, np.newaxis]

    # "Diagonal" Q operators Q_{m,m}: IST component k == orientation index m.
    # Under the Universal Linbladian formalism these are the analytical jump
    # operators (up to the factor sqrt(tau_c / 5)).
    _save_Q_diag = np.array(
        [Q_ops[(m, m)].full() for m in range(-2, 3)], dtype=complex
    )

    _save_path = os.path.join(
        DATA_DIR,
        f'canonical_L_ops_tc{tau_c:.0e}_B0{B0:.2f}.npz',
    )
    np.savez(
        _save_path,
        # Canonical operators (from Delta_plus diagonalisation)
        d_vals=d_vals,          # (15,) eigenvalues of Delta_plus [rad/s]
        L_ops=L_ops,            # (15, 8, 8) canonical Lindblad operators
        U_eigh=U_eigh,          # (15, 15) eigenvector matrix of Delta_plus
        Delta_plus=Delta_p,     # (15, 15) Lindblad-rate matrix in IST basis
        Delta_minus=Delta_m,    # (15, 15) commutator-rate matrix in IST basis
        # Normalised IST basis {p_delta} used to define Delta_plus / L_ops
        P_norm=_save_P_norm,    # (15, 8, 8)  unit-HS-norm IST operators
        hs_norms=_save_hs_norms,  # (15,)    raw HS norms before normalisation
        # Analytical reference operators from Universal Linbladian formalism
        Q_diag=_save_Q_diag,    # (5, 8, 8)  Q_{m,m} for m = -2..+2 [rad/s]
        # H0 eigensystem
        ekets_arr=ekets_arr,    # (8, 8)  rows are eigenvectors
        evals=evals,            # (8,)    eigenvalues [rad/s]
        # Physical parameters
        tau_c=np.array([tau_c]),
        B0=np.array([B0]),
        b_FC=np.array([b_FC]),
        b_FH=np.array([b_FH]),
        b_CH=np.array([b_CH]),
        r_FC=r_FC,
        r_FH=r_FH,
        r_CH=r_CH,
        gamma_F=np.array([GAMMA['19F']]),
        gamma_C=np.array([GAMMA['13C']]),
        gamma_H=np.array([GAMMA['1H']]),
    )
    print(f"\n  Canonical jump operators saved -> {_save_path}")

    # ------------------------------------------------------------------
    # Dominant-5 approximation: top-rate operators only, Delta_minus = 0
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Dominant-5 approximation  (top 5 canonical L_k, no Delta_minus)")
    print("=" * 55)

    # eigh returns eigenvalues ascending; last 5 are the largest
    dom_indices = list(range(len(d_vals) - 5, len(d_vals)))   # [10, 11, 12, 13, 14]
    dom_rate_hz = 2.0 * d_vals[dom_indices[0]] / (2 * np.pi)

    print(f"  Selected indices : {dom_indices}")
    print(f"  d_k              : {d_vals[dom_indices[0]]:.4e} rad/s  "
          f"(rate 2d_k/(2pi) = {dom_rate_hz:.1f} Hz)")
    print(f"  Delta_minus      : set to zero")

    # --- Decompose each dominant L_k over the IST pool ---
    _IST_pairs_rep = [(0, 1), (0, 2), (1, 2)]
    _IST_ks_rep    = [-2, -1, 0, 1, 2]
    _spin_names    = NUCLEI   # ['19F', '13C', '1H']

    _ist_meta = []   # (pair_label, k_val) for each delta = 0..14
    for (si, sj) in _IST_pairs_rep:
        for k in _IST_ks_rep:
            _ist_meta.append((f"{_spin_names[si]}–{_spin_names[sj]}", k))

    print()
    print("  Canonical operators L_k = sum_delta  U*[delta,k]  p_hat_delta")
    print("  (coefficients with |c| >= 0.05 shown; p_hat = normalized T^(2,k)_ij)")
    print()
    for pos, k in enumerate(dom_indices):
        coeffs = U_eigh[:, k].conj()   # U*[delta, k] for all delta
        print(f"  L_{k} (dominant #{pos+1},  d_k = {d_vals[k]:.4e} rad/s):")
        for delta, c in enumerate(coeffs):
            if abs(c) >= 0.05:
                pair_lbl, k_val = _ist_meta[delta]
                print(f"      {c.real:+.4f}{c.imag:+.4f}j  *  "
                      f"T^(2,{k_val:+d})_({pair_lbl})")
        print()

    # --- Build approximate Liouvillian and propagate ---
    _, _ev_ap, _evec_ap, _, _c0_ap = build_approx_liouvillian(
        d_vals, L_ops, dom_indices
    )

    pops_ap    = np.zeros((N_steps, n_states))
    min_eig_ap = np.zeros(N_steps)

    for ti, t in enumerate(tlist):
        _v      = _evec_ap @ (np.exp(_ev_ap * t) * _c0_ap)
        rho_ap  = qt.vector_to_operator(
            qt.Qobj(_v.reshape(_vec_shape), dims=_vec_dims)
        )
        rho_mat_ap = rho_ap.full()
        for n_ in range(n_states):
            pops_ap[ti, n_] = np.real(
                ekets_arr[n_].conj() @ rho_mat_ap @ ekets_arr[n_]
            )
        min_eig_ap[ti] = np.linalg.eigvalsh(rho_mat_ap).min()

    dpops_ap = pops - pops_ap
    max_dev_ap = np.abs(dpops_ap).max()
    print(f"  Trace (approx): t=0 {pops_ap[0].sum():.8f}  "
          f"t_end {pops_ap[-1].sum():.8f}")
    if min_eig_ap.min() < -1e-10:
        print(f"  WARNING: positivity violation — min eig = {min_eig_ap.min():.4e}")
    else:
        print("  Positivity maintained throughout (min eig >= 0)")
    print(f"  Max |Delta P_n| vs full Liouvillian : {max_dev_ap:.4e}")

    # --- Comparison plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        r'Full canonical vs dominant-5 approximation ($\Delta^{-}=0$, top-5 $L_k$)'
        + '\n' + r'$^{19}$F–$^{13}$C–$^{1}$H ZULF, '
        + r'initial $|\!\uparrow\uparrow\uparrow\rangle$, '
        + r'$\tau_c$=' + f'{tau_c:.0e} s',
        fontsize=11
    )

    for n_ in range(n_states):
        lbl = state_labels[n_]
        ax1.plot(tlist_us, pops[:, n_],     color=colors[n_], lw=2.0, ls='-',
                 label=lbl)
        ax1.plot(tlist_us, pops_ap[:, n_],  color=colors[n_], lw=1.2, ls='--')
    ax1.set_ylabel('Population $P_n(t)$', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper right',
               title='solid = full,  dashed = dominant-5')
    ax1.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f'Full canonical (solid) vs dominant-5 approx. (dashed, '
        + f'rate≈{dom_rate_hz:.0f} Hz)', fontsize=10)

    for n_ in range(n_states):
        ax2.plot(tlist_us, dpops_ap[:, n_], color=colors[n_], lw=1.5,
                 label=state_labels[n_])
    ax2.axhline(0.0, color='k', ls='--', lw=0.8)
    ax2.set_ylabel(r'$\Delta P_n = P_n^{\rm full} - P_n^{\rm dom5}$', fontsize=11)
    ax2.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
    ax2.legend(fontsize=8, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'Population difference $\Delta P_n(t)$  '
                  + f'(max = {max_dev_ap:.2e})', fontsize=10)

    plt.tight_layout()
    pop_ap_fpath = os.path.join(
        DATA_DIR, f'populations_dom5_tc{tau_c:.0e}_B0{B0:.2f}.png'
    )
    plt.savefig(pop_ap_fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Dominant-5 comparison plot saved -> {pop_ap_fpath}")
