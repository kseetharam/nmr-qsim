"""
Generalized Redfield simulation of the 19F-13C-1H three-spin system with
Zeeman anisotropy (CSA) dephasing, implementing eq:Gen_Red_Gamma from
notes/liouville_hilbert_basis.tex.

Extends zulf_lindblad.py by adding:
  - Isotropic Zeeman shift in H_iso  (eq:iso_ham)
  - Rank-1 CSA system operators  Q^{(Z),1}_{k,m}  (9 operators)
  - Rank-2 CSA system operators  Q^{(Z),2}_{k,m}  (25 operators)
  - Combined  A^{(2)} = Q^{(Z),2} + Q_hat  (dipolar + CSA rank-2)
  - Rank-dependent spectral density  J^{(l)}(omega) = tau_c / ((2l+1)(1+(omega*tau_c)^2))
  - Generalized Gamma-bar rate matrices built from both l=1 and l=2 modes

The external field B = (Bx, By, Bz) is a tunable parameter (default: B along z).
At B0=0 all Q^{(Z),l} vanish and the result reduces to zulf_lindblad.py.
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
# =============================================================================

GAMMA = {
    '19F':  251.81520e6,
    '13C':   67.28284e6,
    '1H':   267.52218e6,
}

NUCLEI = ['19F', '13C', '1H']
gamma  = np.array([GAMMA[n] for n in NUCLEI])   # shape (3,)

# =============================================================================
# SECTION 3: NUCLEAR COORDINATES  (Angstroms, DFT optimized geometry)
# =============================================================================

ANGSTROM = 1e-10

coords_ang = {
    '19F': np.array([-3.9805 , -0.5583 ,  -0.6136]),
    '13C': np.array([-2.7608 ,  -0.2372 ,  -0.1625]),
    '1H':  np.array([-0.8183  ,  2.4463  ,  0.3685]),
}
coords_m = {k: v * ANGSTROM for k, v in coords_ang.items()}

r_FC = coords_m['13C'] - coords_m['19F']
r_FH = coords_m['1H']  - coords_m['19F']
r_CH = coords_m['1H']  - coords_m['13C']

def dipolar_coupling(gamma_i, gamma_j, r_vec):
    r = np.linalg.norm(r_vec)
    return -(mu0 / (4 * np.pi)) * gamma_i * gamma_j * hbar / r**3

b_FC = dipolar_coupling(GAMMA['19F'], GAMMA['13C'], r_FC)
b_FH = dipolar_coupling(GAMMA['19F'], GAMMA['1H'],  r_FH)
b_CH = dipolar_coupling(GAMMA['13C'], GAMMA['1H'],  r_CH)

# =============================================================================
# SECTION 4: J COUPLINGS (Hz)
# =============================================================================

J_FC = 243.5
J_CH =  10.71
J_FH =   0.0

J_FC_rad = 2 * np.pi * J_FC
J_CH_rad = 2 * np.pi * J_CH
J_FH_rad = 2 * np.pi * J_FH

# =============================================================================
# SECTION 5: ZEEMAN SHIELDING TENSORS (ppm, from DFT)
#
# Source: Table 1 and shielding tensor list in liouville_hilbert_basis.tex.
# sigma_ppm[X] is the raw DFT shielding tensor sigma(X) in ppm.
# The physical tensor is sigma_tilde(X) = 1 + 1e-6 * sigma(X).
# =============================================================================

sigma_ppm = {
    '19F': np.array([
        [-113.0955, -13.4634,  23.3880],
        [ -13.4634, -32.0922, -17.6116],
        [  23.3880, -17.6116,-153.6685],
    ]),
    '13C': np.array([
        [240.8318, 25.3402, 53.6419],
        [ 25.3402,162.9404,  5.1940],
        [ 53.6419,  5.1940,127.8480],
    ]),
    '1H': np.array([
        [ 5.0782,  2.5669, -1.6492],
        [ 2.5669, 10.7485,  0.0000],
        [-1.6492,  0.0000, 11.9705],
    ]),
}

# Physical shielding tensor sigma_tilde = 1 + 1e-6 * sigma_ppm (dimensionless)
sigma_tilde = {X: np.eye(3) + 1e-6 * sigma_ppm[X] for X in NUCLEI}

# =============================================================================
# SECTION 6: EXTERNAL FIELD AND SIMULATION PARAMETERS
# =============================================================================

B_vec = np.array([0.0, 0.0, 1e-3])  # T, field along z
B0    = np.linalg.norm(B_vec)        # scalar magnitude

tau_c = 1e-5    # s   rotational correlation time

# Isotropic shielding constants  sigma_iso(X)  (dimensionless)
sigma_iso = {X: np.trace(sigma_tilde[X]) / 3.0 for X in NUCLEI}

# =============================================================================
# SECTION 7: QUTIP SPIN OPERATORS
# =============================================================================

def spin_op(op_char, idx, n=3):
    ops = [qt.qeye(2)] * n
    ops[idx] = qt.jmat(0.5, op_char)
    return qt.tensor(ops)

Ix = [spin_op('x', k) for k in range(3)]
Iy = [spin_op('y', k) for k in range(3)]
Iz = [spin_op('z', k) for k in range(3)]
Ip = [Ix[k] + 1j * Iy[k] for k in range(3)]
Im = [Ix[k] - 1j * Iy[k] for k in range(3)]

# Spherical spin-1/2 components: (S)_0 = Sz, (S)_{+1} = -S+/sqrt(2), (S)_{-1} = S-/sqrt(2)
def S_sph(idx):
    """Spherical components of spin idx as a dict {q: Qobj}."""
    return {
         0:  Iz[idx],
        +1: -Ip[idx] / np.sqrt(2),
        -1:  Im[idx] / np.sqrt(2),
    }

def heisenberg_dot(i, j):
    return Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j]

# =============================================================================
# SECTION 8: H_ISO = HEISENBERG J-COUPLINGS + ISOTROPIC ZEEMAN SHIFT
#             (eq:iso_ham)
#
# H_iso = sum_ij J_ij S_i.S_j  -  sum_X gamma_X sigma_iso(X) S_X.B
# With B = B_vec = (Bx, By, Bz):
#   S_X.B = Bx*Ix + By*Iy + Bz*Iz
# =============================================================================

H_J = (J_FC_rad * heisenberg_dot(0, 1)
     + J_CH_rad * heisenberg_dot(1, 2)
     + J_FH_rad * heisenberg_dot(0, 2))

H_Z_iso = sum(
    -GAMMA[X] * sigma_iso[X] * (
        B_vec[0] * Ix[idx] + B_vec[1] * Iy[idx] + B_vec[2] * Iz[idx]
    )
    for idx, X in enumerate(NUCLEI)
)

H0 = H_J + H_Z_iso

evals, ekets = H0.eigenstates()

_M_tot_op = Iz[0] + Iz[1] + Iz[2]
_S2_tot_op = ((Ix[0]+Ix[1]+Ix[2])**2
             + (Iy[0]+Iy[1]+Iy[2])**2
             + (Iz[0]+Iz[1]+Iz[2])**2)
_M_qn  = np.round(2 * np.real([qt.expect(_M_tot_op,  ek) for ek in ekets])) / 2
_S2_ev = np.real([qt.expect(_S2_tot_op, ek) for ek in ekets])
_S_qn  = np.round(2 * (np.sqrt(np.clip(4*_S2_ev + 1, 0, None)) - 1) / 2) / 2

_E_Hz_arr = evals / (2 * np.pi)

# At high field there may be more than 2 distinct S=1/2 energies;
# label doublet states by index within their M sector instead.
_s12_mask = np.abs(_S_qn - 0.5) < 0.1
_s12_Es   = np.unique(np.round(_E_Hz_arr[_s12_mask], 1))

def _state_label(n):
    S, M = _S_qn[n], _M_qn[n]
    Mstr = {1.5:'+3/2', -1.5:'-3/2', 0.5:'+1/2', -0.5:'-1/2'}.get(round(M,1), f'{M:+.1f}')
    if abs(S - 1.5) < 0.1:
        return rf'$|3/2,{Mstr}\rangle$'
    # label doublets by energy rank (α=lowest, β=next, γ=...)
    E_rounded = round(_E_Hz_arr[n], 1)
    rank = np.searchsorted(_s12_Es, E_rounded)
    sub_chars = [r'\alpha', r'\beta', r'\gamma', r'\delta']
    sub = sub_chars[rank] if rank < len(sub_chars) else str(rank)
    return rf'$|1/2_{{{sub}}},{Mstr}\rangle$'

state_labels = [_state_label(n) for n in range(len(evals))]

n_states = len(evals)
n_trans  = n_states ** 2
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)  # (8, 8)

# =============================================================================
# SECTION 9: CG COEFFICIENTS FOR 1 ⊗ 1 → l
#
# cg1x1[l][m1, m2] = <1,m1;1,m2 | l, m1+m2>
# m indices stored offset by 1: m=-1→0, m=0→1, m=+1→2
# =============================================================================

# Hard-coded table for <1,m1;1,m2|l,M> with M = m1+m2
# Only nonzero entries listed; m1,m2 ∈ {-1,0,+1}
_rt2 = np.sqrt(2.0)
_rt3 = np.sqrt(3.0)
_rt6 = np.sqrt(6.0)

# cg[(l, m1, m2)] = CG coefficient
_CG = {
    # l = 0
    (0,  1, -1):  1.0/_rt3,
    (0,  0,  0): -1.0/_rt3,
    (0, -1,  1):  1.0/_rt3,
    # l = 1
    (1,  1,  0):  1.0/_rt2,
    (1,  0,  1): -1.0/_rt2,
    (1,  1, -1):  1.0/_rt2,
    (1,  0,  0):  0.0,
    (1, -1,  1): -1.0/_rt2,
    (1,  0, -1):  1.0/_rt2,
    (1, -1,  0): -1.0/_rt2,
    # l = 2
    (2,  1,  1):  1.0,
    (2,  1,  0):  1.0/_rt2,
    (2,  0,  1):  1.0/_rt2,
    (2,  1, -1):  1.0/_rt6,
    (2,  0,  0):  2.0/_rt6,
    (2, -1,  1):  1.0/_rt6,
    (2,  0, -1):  1.0/_rt2,
    (2, -1,  0):  1.0/_rt2,
    (2, -1, -1):  1.0,
}

def cg1x1(l, m1, m2):
    """<1,m1;1,m2|l,m1+m2>. Returns 0 if not in table."""
    return _CG.get((l, m1, m2), 0.0)

# =============================================================================
# SECTION 10: SPHERICAL FIELD COMPONENTS  B_q  (q = -1, 0, +1)
# =============================================================================

def B_spherical(B):
    """Spherical components of B = (Bx, By, Bz): dict {q: complex}."""
    Bx, By, Bz = B
    return {
         0:  Bz + 0j,
        +1: -(Bx + 1j*By) / np.sqrt(2),
        -1:  (Bx - 1j*By) / np.sqrt(2),
    }

B_sph = B_spherical(B_vec)

# =============================================================================
# SECTION 11: SINGLE-SPIN RANK-l TENSOR PRODUCTS  T^{(l)}_k(S_X, B)
#
# T^{(l)}_k = sum_{q1+q2=k} <1,q1;1,q2|l,k> (S_X)_{q1} B_{q2}
#
# Returns a QuTiP Qobj in the 8-dimensional 3-spin Hilbert space.
# =============================================================================

def T_lk_op(l, k, spin_idx, B=None):
    """
    Rank-l irreducible tensor product  [S_X ⊗ B]^{(l)}_k  for spin spin_idx.

    Parameters
    ----------
    l        : rank (0, 1, or 2)
    k        : component index (-l .. +l)
    spin_idx : 0=19F, 1=13C, 2=1H
    B        : spherical field components dict {q: complex}; uses B_sph if None
    """
    if B is None:
        B = B_sph
    Ss = S_sph(spin_idx)
    result = None
    for q1 in (-1, 0, 1):
        q2 = k - q1
        if q2 not in (-1, 0, 1):
            continue
        c = cg1x1(l, q1, q2)
        if abs(c) < 1e-15:
            continue
        term = c * B[q2] * Ss[q1]
        result = term if result is None else result + term
    if result is None:
        # zero operator
        return 0.0 * Iz[0]
    return result

# =============================================================================
# SECTION 12: SYMMETRY-ADAPTED ZEEMAN TENSOR COMPONENTS  sigma_{l,m}(X)
#
# Expressions from liouville_hilbert_basis.tex, "Zeeman anisotropy dephasing".
# Input: sigma_t = sigma_tilde(X)  (3x3 NumPy array, dimensionless).
# =============================================================================

def sigma_lm(sigma_t):
    """
    Returns dict {(l, m): complex} for l=0,1,2 and m=-l..+l.
    sigma_t is the physical shielding tensor sigma_tilde = 1 + 1e-6*sigma_ppm.
    """
    s = sigma_t   # short alias
    s_iso = (s[0,0] + s[1,1] + s[2,2]) / 3.0
    return {
        # l = 0
        (0,  0): -np.sqrt(3) * s_iso,
        # l = 1
        (1,  0): -1j / np.sqrt(2) * (s[0,1] - s[1,0]),
        (1, +1): -0.5 * ((s[2,0] - s[0,2]) + 1j*(s[2,1] - s[1,2])),
        (1, -1): -0.5 * ((s[2,0] - s[0,2]) - 1j*(s[2,1] - s[1,2])),
        # l = 2
        (2,  0):  np.sqrt(2.0/3.0) * (s[2,2] - s_iso),
        (2, +1): -0.5 * ((s[0,2] + s[2,0]) + 1j*(s[1,2] + s[2,1])),
        (2, -1): +0.5 * ((s[0,2] + s[2,0]) - 1j*(s[1,2] + s[2,1])),
        (2, +2):  0.5 * ((s[0,0] - s[1,1]) + 1j*(s[0,1] + s[1,0])),
        (2, -2):  0.5 * ((s[0,0] - s[1,1]) - 1j*(s[0,1] + s[1,0])),
    }

# Pre-compute sigma_{l,m}(X) for all three nuclei
sigma_lm_all = {X: sigma_lm(sigma_tilde[X]) for X in NUCLEI}

# =============================================================================
# SECTION 13: ZEEMAN CSA SYSTEM OPERATORS  Q^{(Z),l}_{k,m}
#
# Q^{(Z),l}_{k,m} = sum_X  gamma_X * (-1)^{m+1} * sigma_{l,m}(X) * T^{(l)}_{-k}(S_X, B)
#
# These are the system operators that enter the bath coupling for CSA relaxation.
# l=1: 9 operators (k,m ∈ {-1,0,+1})
# l=2: 25 operators (k,m ∈ {-2,-1,0,+1,+2})
# =============================================================================

def build_QZ_ops(l):
    """
    Build the (2l+1)^2 Zeeman CSA system operators Q^{(Z),l}_{k,m}
    for rank l (1 or 2).  Returns dict {(k,m): Qobj}.
    """
    ops = {}
    for k in range(-l, l+1):
        for m in range(-l, l+1):
            phase = (-1)**(m + 1)
            op = None
            for idx, X in enumerate(NUCLEI):
                slm = sigma_lm_all[X][(l, m)]
                T_neg_k = T_lk_op(l, -k, idx)
                contrib = GAMMA[X] * phase * slm * T_neg_k
                op = contrib if op is None else op + contrib
            ops[(k, m)] = op
    return ops

QZ1_ops = build_QZ_ops(l=1)   # 9 operators
QZ2_ops = build_QZ_ops(l=2)   # 25 operators

# =============================================================================
# SECTION 14: DIPOLAR IST OPERATORS  Q_hat_{k,m}  (as in zulf_lindblad.py)
# =============================================================================

def T2_op(i, j, k):
    """Rank-2 two-body IST T_{2,k}^{ij}."""
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
    rhat = r_vec / np.linalg.norm(r_vec)
    return 3.0 * np.outer(rhat, rhat) - np.eye(3)

def a2m_factors(r_vec):
    A = dipolar_tensor(r_vec)
    return {
         0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
    }

PAIRS = [
    (0, 1, b_FC, r_FC),
    (0, 2, b_FH, r_FH),
    (1, 2, b_CH, r_CH),
]

alm = {(i, j): a2m_factors(r_vec) for i, j, _, r_vec in PAIRS}

Qdip_ops = {
    (k, m): sum(b * alm[(i, j)][m] * T2_op(i, j, k)
                for i, j, b, _ in PAIRS)
    for k in range(-2, 3)
    for m in range(-2, 3)
}

# =============================================================================
# SECTION 15: COMBINED A^{(l)} OPERATORS
#
# A^{(1)}_{(k,m)} = Q^{(Z),1}_{k,m}                     (l=1 modes only)
# A^{(2)}_{(k,m)} = Q^{(Z),2}_{k,m} + Q_hat_{k,m}       (l=2 modes)
# =============================================================================

A1_ops = QZ1_ops   # {(k,m): Qobj}, k,m ∈ {-1,0,+1}

A2_ops = {
    (k, m): QZ2_ops[(k, m)] + Qdip_ops[(k, m)]
    for k in range(-2, 3)
    for m in range(-2, 3)
}

# =============================================================================
# SECTION 16: RANK-DEPENDENT SPECTRAL DENSITY
#
# J^{(l)}(omega) = (1/(2l+1)) * tau_c / (1 + omega^2 * tau_c^2)
# =============================================================================

def J_l(l, omega):
    return tau_c / ((2*l + 1) * (1.0 + (omega * tau_c)**2))

def J_spectral(omega):
    """Backward-compatible: l=2 spectral density (same as zulf_lindblad.py)."""
    return J_l(2, omega)

# =============================================================================
# SECTION 17: GENERALIZED GAMMA-BAR RATE MATRICES  (eq:Gen_Red_Gamma)
#
# Gamma_bar_ij(omega) = sum_{l=1,2} sum_{alpha_l} (A^{(l)}_{alpha_l})_i^*
#                       * 2*J^{(l)}(omega) * (A^{(l)}_{alpha_l})_j
#
# where (A^{(l)}_alpha)_i = Tr{A^{(l)}_alpha * sigma_i^dag}
#                          = <n_i|A^{(l)}_alpha|m_i>
#
# We build separate structure matrices C^{(l)}_{ij} = sum_alpha M^{(l)}_alpha_i^* M^{(l)}_alpha_j
# then Gamma_bar(omega) = 2 * sum_l C^{(l)} * J^{(l)}(omega)
#
# The factor of 2 comes from gamma_{alpha,beta}(omega) = 2*delta_{ab}*J^{(l)}(omega)
# (see eq in tex, same factor as in the dipolar-only case).
# =============================================================================

omega_trans = np.array([evals[m_] - evals[n_]
                        for n_ in range(n_states)
                        for m_ in range(n_states)])   # (64,)

def build_M_matrix(ops_dict, km_list):
    """
    Build M[alpha, i] = <n_i|A_alpha|m_i> for all transitions i.
    ops_dict: {(k,m): Qobj}
    km_list:  ordered list of (k,m) keys
    Returns M_3d (n_modes, n_states, n_states) and M_matrix (n_modes, n_trans).
    """
    n_modes = len(km_list)
    Q_stack = np.array([ops_dict[km].full() for km in km_list], dtype=complex)
    M_3d    = np.einsum('nk,akl,ml->anm', ekets_arr.conj(), Q_stack, ekets_arr)
    M_mat   = M_3d.reshape(n_modes, n_trans)
    return M_3d, M_mat

KM1_LIST = [(k, m) for k in range(-1, 2) for m in range(-1, 2)]   # 9 l=1 modes
KM2_LIST = [(k, m) for k in range(-2, 3) for m in range(-2, 3)]   # 25 l=2 modes

_, M1_matrix = build_M_matrix(A1_ops, KM1_LIST)   # (9, 64)
_, M2_matrix = build_M_matrix(A2_ops, KM2_LIST)   # (25, 64)

C1_matrix = M1_matrix.conj().T @ M1_matrix   # (64, 64)
C2_matrix = M2_matrix.conj().T @ M2_matrix   # (64, 64)

# Spectral densities evaluated at each transition frequency
J1_omegas = np.array([J_l(1, w) for w in omega_trans])   # (64,)
J2_omegas = np.array([J_l(2, w) for w in omega_trans])   # (64,)

# Gamma_bar(omega_j) = 2 * [C1 * J1(omega_j) + C2 * J2(omega_j)]
# Build as (64,64) matrices evaluated at omega_j (column index)
Gamma_bar_j = 2.0 * (C1_matrix * J1_omegas[np.newaxis, :]
                   + C2_matrix * J2_omegas[np.newaxis, :])   # (64,64)
Gamma_bar_i = 2.0 * (C1_matrix * J1_omegas[:, np.newaxis]
                   + C2_matrix * J2_omegas[:, np.newaxis])   # (64,64)

Gamma_minus = 0.5 * (Gamma_bar_j - Gamma_bar_i)
Gamma_plus  = 0.5 * (Gamma_bar_j + Gamma_bar_i)

# =============================================================================
# SECTION 18: LIOUVILLIAN AND PROPAGATOR
# =============================================================================

_up  = qt.basis(2, 0)
rho0 = qt.ket2dm(qt.tensor(_up, _up, _up))

_rho0_vec_qobj = qt.operator_to_vector(rho0)
_vec_dims  = _rho0_vec_qobj.dims
_vec_shape = _rho0_vec_qobj.shape


def build_liouvillian(gamma_scale=1.0):
    """
    Build Liouvillian from eq:Gen_Red_Gamma with generalized Gamma-bar.

    L[rho] = -i[H_iso, rho] + R[rho]
    R[rho] = -1/4 sum_{ij} (Gamma_bar_ij(omega_j) - Gamma_bar_ij(omega_i)) [sigma_i^dag sigma_j, rho]
             +1/4 sum_{ij} (Gamma_bar_ij(omega_j) + Gamma_bar_ij(omega_i))
                              (-{sigma_i^dag sigma_j, rho} + 2 sigma_j rho sigma_i^dag)

    Dissipator built in eigenstate Liouville basis then rotated to computational basis.
    """
    L_ham_mat = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

    Gamma_raw = (Gamma_bar_j * gamma_scale)   # (64,64): Gamma_bar evaluated at omega_j

    GR = Gamma_raw.reshape(n_states, n_states, n_states, n_states)
    T1 = np.einsum('amcm->ac', GR)
    T4 = np.einsum('nbnd->bd', GR)

    L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
            + np.einsum('acbd->abcd', GR)
            + np.einsum('dbca->abcd', GR)
            - np.einsum('ac,bd->abcd', np.eye(n_states), T4))

    L_diss_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_states**2, n_states**2)

    U = ekets_arr.T
    V = np.kron(U.conj(), U)
    L_diss_mat = V @ L_diss_eig @ V.conj().T

    _mat      = L_ham_mat + L_diss_mat
    _ev, _vec = np.linalg.eig(_mat)
    _vec_inv  = np.linalg.inv(_vec)
    _c0       = _vec_inv @ _rho0_vec_qobj.full().flatten()
    return _mat, _ev, _vec, _vec_inv, _c0


L_mat, _L_evals, _L_evecs, _L_evecs_inv, _c0 = build_liouvillian(gamma_scale=1.0)


def rho_at(t):
    _v = _L_evecs @ (np.exp(_L_evals * t) * _c0)
    return qt.vector_to_operator(
        qt.Qobj(_v.reshape(_vec_shape), dims=_vec_dims)
    )


# =============================================================================
# SECTION 19: FGR POPULATION TRANSFER RATES (generalized)
# =============================================================================

M1_3d, _ = build_M_matrix(A1_ops, KM1_LIST)
M2_3d, _ = build_M_matrix(A2_ops, KM2_LIST)

_Q2_fi_l1 = np.sum(np.abs(M1_3d)**2, axis=0)   # (8,8)
_Q2_fi_l2 = np.sum(np.abs(M2_3d)**2, axis=0)

_omega_fi_mat = evals[:, np.newaxis] - evals[np.newaxis, :]

fgr_rates = (np.vectorize(lambda w: J_l(1, w))(_omega_fi_mat) * _Q2_fi_l1
           + np.vectorize(lambda w: J_l(2, w))(_omega_fi_mat) * _Q2_fi_l2)
np.fill_diagonal(fgr_rates, 0.0)


# =============================================================================
# MAIN: diagnostic output and population dynamics plots
# =============================================================================

if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Generalized Redfield: 19F-13C-1H with CSA dephasing")
    print("=" * 60)
    print(f"  B_vec = {B_vec} T   (|B| = {B0:.3f} T)")
    print(f"  tau_c = {tau_c:.1e} s")
    print()
    print("  Isotropic shielding  sigma_iso(X)  (dimensionless):")
    for X in NUCLEI:
        print(f"    {X:>4s}:  {sigma_iso[X]:.8f}")
    print()

    # --- H_iso eigenvalues ---
    print("=" * 60)
    print("H_iso eigenvalues")
    print("=" * 60)
    for n, e in enumerate(evals):
        print(f"  |{n}>  {e:+12.4f} rad/s  ({e/(2*np.pi):+10.4f} Hz)  {state_labels[n]}")
    print()

    # --- Q^{(Z),l} Frobenius norms ---
    print("=" * 60)
    print("||Q^{(Z),1}_{k,m}||_F  (rad/s) — l=1 CSA operators")
    print("Rows k=-1..+1,  Cols m=-1..+1")
    print("=" * 60)
    norm1 = np.array([
        [np.sqrt(abs((QZ1_ops[(k,m)].dag()*QZ1_ops[(k,m)]).tr()))
         for m in range(-1, 2)]
        for k in range(-1, 2)
    ])
    for ki, k in enumerate(range(-1, 2)):
        row = "  ".join(f"{norm1[ki,mi]:.3e}" for mi in range(3))
        print(f"  k={k:+d}:  {row}")
    print()

    print("=" * 60)
    print("||Q^{(Z),2}_{k,m}||_F  (rad/s) — l=2 CSA operators")
    print("Rows k=-2..+2,  Cols m=-2..+2")
    print("=" * 60)
    norm2 = np.array([
        [np.sqrt(abs((QZ2_ops[(k,m)].dag()*QZ2_ops[(k,m)]).tr()))
         for m in range(-2, 3)]
        for k in range(-2, 3)
    ])
    for ki, k in enumerate(range(-2, 3)):
        row = "  ".join(f"{norm2[ki,mi]:.3e}" for mi in range(5))
        print(f"  k={k:+d}:  {row}")
    print()

    # --- Gamma statistics ---
    gp_max = np.abs(Gamma_plus).max()
    gm_max = np.abs(Gamma_minus).max()
    print("=" * 60)
    print("Generalized Gamma-bar matrix statistics")
    print("=" * 60)
    print(f"  max |Gamma_+| = {gp_max:.4e} rad/s")
    print(f"  max |Gamma_-| = {gm_max:.4e} rad/s")
    print(f"  |Gamma_-|/|Gamma_+| = {gm_max/(gp_max+1e-300):.4e}")
    print()

    # --- FGR top 5 ---
    print("=" * 60)
    print("Generalized FGR: top 5 population transfer rates")
    print("=" * 60)
    _top5 = np.argsort(fgr_rates.ravel())[::-1][:5]

    def _plain(lbl):
        return (lbl.replace('$','').replace(r'\alpha','a').replace(r'\beta','b')
                   .replace(r'\rangle','>').replace(r'\langle','<').replace(r'\,',','))

    for rank, flat_i in enumerate(_top5):
        f_idx, i_idx = divmod(flat_i, n_states)
        rate_hz = fgr_rates[f_idx, i_idx] / (2*np.pi)
        omega_hz = _omega_fi_mat[f_idx, i_idx] / (2*np.pi)
        print(f"  [{rank+1}]  {_plain(state_labels[i_idx]):28}  ->  "
              f"{_plain(state_labels[f_idx]):28}  "
              f"omega_fi={omega_hz:+10.3f} Hz  k/(2pi)={rate_hz:.4e} Hz")
    print()

    # --- Liouvillian eigenvalue summary ---
    L_evals_re = np.real(_L_evals)
    n_zero = np.sum(np.abs(L_evals_re) < gp_max * 1e-8)
    print("=" * 60)
    print("Liouvillian eigenvalue summary")
    print("=" * 60)
    print(f"  Most-negative real part  : {L_evals_re.min():.4e} rad/s")
    print(f"  Most-positive real part  : {L_evals_re.max():.4e} rad/s")
    print(f"  Num near-zero real parts : {n_zero}")
    print(f"  Max |imag part|          : {np.abs(np.imag(_L_evals)).max():.4e} rad/s")
    print()

    # --- Time propagation ---
    t_end   = 100e-6
    N_steps = 500
    tlist   = np.linspace(0.0, t_end, N_steps)
    tlist_us = tlist * 1e6

    print("=" * 60)
    print("Time propagation (eigenbasis populations)")
    print("=" * 60)
    print(f"  Propagating to t = {t_end*1e6:.0f} us  ({N_steps} steps)")

    pops    = np.zeros((N_steps, n_states))
    min_eig = np.zeros(N_steps)

    for ti, t in enumerate(tlist):
        rho_t   = rho_at(t)
        rho_mat = rho_t.full()
        for n_ in range(n_states):
            pops[ti, n_] = np.real(ekets_arr[n_].conj() @ rho_mat @ ekets_arr[n_])
        min_eig[ti] = np.linalg.eigvalsh(rho_mat).min()

    print(f"  Final populations (t = {t_end:.2e} s):")
    for n_ in range(n_states):
        print(f"    {_plain(state_labels[n_]):30}:  P = {pops[-1, n_]:.6f}")
    print(f"  Trace: t=0 {pops[0].sum():.8f}  t_end {pops[-1].sum():.8f}")
    if min_eig.min() < -1e-10:
        print(f"  WARNING: positivity violation — min eig = {min_eig.min():.4e}")
    else:
        print("  Positivity maintained (min eig >= 0)")

    # --- Population dynamics plot ---
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        r'Population dynamics with CSA dephasing: $^{19}$F–$^{13}$C–$^{1}$H'
        + '\n' + rf'$|B|={B0:.1f}$ T, $\tau_c={tau_c:.0e}$ s, '
        + r'initial $|\!\uparrow\uparrow\uparrow\rangle$',
        fontsize=11
    )

    for n_ in range(n_states):
        ax1.plot(tlist_us, pops[:, n_], color=colors[n_],
                 label=state_labels[n_], lw=1.5)
    ax1.set_ylabel('Population', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.axhline(1.0/n_states, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Eigenbasis populations (generalized Redfield + CSA)', fontsize=10)

    ax2.plot(tlist_us, min_eig, color='crimson', lw=1.5)
    ax2.axhline(0.0, color='k', ls='--', lw=0.8)
    ax2.set_ylabel(r'min eigenvalue of $\rho(t)$', fontsize=11)
    ax2.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Positivity check', fontsize=10)

    plt.tight_layout()
    B_tag = f"B{B0:.2e}T"
    pop_fpath = os.path.join(DATA_DIR,
                             f'populations_csa_tc{tau_c:.0e}_{B_tag}.png')
    plt.savefig(pop_fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Population plot saved -> {pop_fpath}")

    # --- Gamma colorplots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Generalized Redfield rate matrices (dipolar + CSA): '
        + rf'$^{{19}}$F–$^{{13}}$C–$^{{1}}$H, $|B|={B0:.1f}$ T',
        fontsize=11
    )
    mats   = [np.abs(Gamma_plus), np.abs(Gamma_minus)]
    titles = [r'$|\bar{\Gamma}_+|$  (rad/s)', r'$|\bar{\Gamma}_-|$  (rad/s)']
    for ax, mat, title in zip(axes, mats, titles):
        vmax = mat.max()
        if vmax > 0:
            vmin = max(vmax * 1e-8, 1e-30)
            im = ax.imshow(mat, cmap='inferno', interpolation='nearest',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(mat, cmap='inferno', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='rad/s', fraction=0.046, pad=0.04)
        for pos in range(n_states, n_trans, n_states):
            ax.axhline(pos - 0.5, color='cyan', lw=0.4, alpha=0.5)
            ax.axvline(pos - 0.5, color='cyan', lw=0.4, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'Transition $j$', fontsize=9)
        ax.set_ylabel(r'Transition $i$', fontsize=9)

    plt.tight_layout()
    gfpath = os.path.join(DATA_DIR, f'gamma_csa_tc{tau_c:.0e}_{B_tag}.png')
    plt.savefig(gfpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gamma colorplot saved -> {gfpath}")
