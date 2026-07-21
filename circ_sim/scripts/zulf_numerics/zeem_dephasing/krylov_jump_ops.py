"""
Construct Lindblad jump operators via the generalised Krylov expansion
(Eq. trunc_order_gen in notes/liouville_hilbert_basis.tex) for the
three-spin 19F–13C–1H model.

Physical prescription
---------------------
Collective system-bath coupling operators (tex, just above Eq. trunc_order_gen):

    A^{(1)}_{k,m} = Q^{(Z),1}_{k,m}
    A^{(2)}_{k,m} = Q^{(Z),2}_{k,m} + Q_hat_{k,m}

Dressed operators (Eq. trunc_order_gen):

    Abar^{(l)}_{k,m}(tau_c) = sum_{j=0}^{N} c_j * A^{(l),j}_{k,m}

    c_j = (1/j!) * 2^{j-1} * Gamma((j+1)/2)^2        (moments of K_0)
    A^{(l),j}_{k,m} = (i [tau_c H_0, .])^j  A^{(l)}_{k,m}    (Krylov chain)

Jump operators (Eq. TD_jumps_gen, strength gamma_{k,m}^{(l)} = 2*sqrt(2)/pi):

    L^{(l)}_{k,m} = sqrt(8 tau_c / (pi^2 (2l+1))) * Abar^{(l)}_{k,m}

Zeroth-order (j=0) limit:  L^{(l)}_{k,m} = sqrt(2 J^{(l)}(0)) * A^{(l)}_{k,m}

Diagnostics
-----------
For each (l, k, m) the script prints
  - ||L||_F at each Krylov order up to convergence
  - relative correction from first-order dressing
  - relative difference between zeroth-order and fully-dressed operator
"""

import os
import numpy as np
import qutip as qt
from scipy.special import gamma as gamma_func
from math import factorial

# =============================================================================
# PARAMETERS
# =============================================================================

hbar     = 1.054571817e-34
mu0      = 1.25663706212e-6
GAMMA    = {'19F': 251.81520e6, '13C': 67.28284e6, '1H': 267.52218e6}
NUCLEI   = ['19F', '13C', '1H']
ANGSTROM = 1e-10

tau_c = 1e-7   # s  — change here to explore dressing strength
B_vec = np.array([0.0, 0.0, 1e-3])   # T

coords_ang = {
    '19F': np.array([-3.9805, -0.5583, -0.6136]),
    '13C': np.array([-2.7608, -0.2372, -0.1625]),
    '1H':  np.array([-0.8183,  2.4463,  0.3685]),
}
coords_m = {k: v * ANGSTROM for k, v in coords_ang.items()}
r_FC = coords_m['13C'] - coords_m['19F']
r_FH = coords_m['1H']  - coords_m['19F']
r_CH = coords_m['1H']  - coords_m['13C']

J_FC_rad = 2 * np.pi * 243.5    # rad/s
J_CH_rad = 2 * np.pi * 10.71

sigma_ppm = {
    '19F': np.array([[-113.0955, -13.4634,  23.3880],
                     [ -13.4634, -32.0922, -17.6116],
                     [  23.3880, -17.6116,-153.6685]]),
    '13C': np.array([[240.8318, 25.3402, 53.6419],
                     [ 25.3402,162.9404,  5.1940],
                     [ 53.6419,  5.1940,127.8480]]),
    '1H':  np.array([[ 5.0782,  2.5669, -1.6492],
                     [ 2.5669, 10.7485,  0.0000],
                     [-1.6492,  0.0000, 11.9705]]),
}
sigma_tilde = {X: np.eye(3) + 1e-6 * sigma_ppm[X] for X in NUCLEI}
sigma_iso   = {X: np.trace(sigma_tilde[X]) / 3.0 for X in NUCLEI}

N_TRUNC = 8   # Krylov truncation order (convergence checked at runtime)

# =============================================================================
# SPIN OPERATORS  (QuTiP, then .full() for matrix arithmetic)
# =============================================================================

def spin_op(op_char, idx, n=3):
    ops = [qt.qeye(2)] * n
    ops[idx] = qt.jmat(0.5, op_char)
    return qt.tensor(ops)

Ix = [spin_op('x', k) for k in range(3)]
Iy = [spin_op('y', k) for k in range(3)]
Iz = [spin_op('z', k) for k in range(3)]
Ip = [Ix[k] + 1j*Iy[k] for k in range(3)]
Im = [Ix[k] - 1j*Iy[k] for k in range(3)]

def S_sph(idx):
    """Spherical components of spin-idx."""
    return {0: Iz[idx], +1: -Ip[idx]/np.sqrt(2), -1: Im[idx]/np.sqrt(2)}

def heis(i, j):
    return Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j]

# =============================================================================
# ISOTROPIC HAMILTONIAN  H_0 = H_iso  (Eq. iso_ham in tex)
# =============================================================================

H_J     = J_FC_rad * heis(0, 1) + J_CH_rad * heis(1, 2)
H_Z_iso = sum(-GAMMA[X] * sigma_iso[X] *
              (B_vec[0]*Ix[idx] + B_vec[1]*Iy[idx] + B_vec[2]*Iz[idx])
              for idx, X in enumerate(NUCLEI))
H0      = H_J + H_Z_iso
H0_mat  = H0.full()   # 8×8 numpy array for fast Krylov iteration

B0 = np.linalg.norm(B_vec)
print(f"tau_c = {tau_c:.1e} s,  |B| = {B0*1e3:.1f} mT")
print(f"J_FC * tau_c = {J_FC_rad * tau_c:.3e}  "
      f"  omega_F * tau_c = {GAMMA['19F']*B0*tau_c:.3e}")
print()

# =============================================================================
# SPHERICAL DECOMPOSITION OF ZEEMAN TENSORS
# sigma_{l,m}(X) — Eqs. in tex "Zeeman anisotropy dephasing" subsection
# =============================================================================

def sigma_lm(sigma_t):
    s    = sigma_t
    siso = (s[0,0] + s[1,1] + s[2,2]) / 3.0
    return {
        (0,  0): -np.sqrt(3) * siso,
        (1,  0): -1j/np.sqrt(2) * (s[0,1] - s[1,0]),
        (1, +1): -0.5 * ((s[2,0]-s[0,2]) + 1j*(s[2,1]-s[1,2])),
        (1, -1): -0.5 * ((s[2,0]-s[0,2]) - 1j*(s[2,1]-s[1,2])),
        (2,  0):  np.sqrt(2.0/3.0) * (s[2,2] - siso),
        (2, +1): -0.5 * ((s[0,2]+s[2,0]) + 1j*(s[1,2]+s[2,1])),
        (2, -1): +0.5 * ((s[0,2]+s[2,0]) - 1j*(s[1,2]+s[2,1])),
        (2, +2):  0.5 * ((s[0,0]-s[1,1]) + 1j*(s[0,1]+s[1,0])),
        (2, -2):  0.5 * ((s[0,0]-s[1,1]) - 1j*(s[0,1]+s[1,0])),
    }

sigma_lm_all = {X: sigma_lm(sigma_tilde[X]) for X in NUCLEI}

# =============================================================================
# T^{(l)}_{-k} OPERATOR  (rank-l IST from S⊗B, tex Eq. zeeman_tensor_product)
# =============================================================================

_rt2 = np.sqrt(2.0); _rt3 = np.sqrt(3.0); _rt6 = np.sqrt(6.0)
_CG = {
    (0,  1,-1):  1.0/_rt3, (0,  0, 0): -1.0/_rt3, (0, -1, 1):  1.0/_rt3,
    (1,  1, 0):  1.0/_rt2, (1,  0, 1): -1.0/_rt2,
    (1,  1,-1):  1.0/_rt2, (1,  0, 0):  0.0,
    (1, -1, 1): -1.0/_rt2, (1,  0,-1):  1.0/_rt2, (1, -1, 0): -1.0/_rt2,
    (2,  1, 1):  1.0,      (2,  1, 0):  1.0/_rt2, (2,  0, 1):  1.0/_rt2,
    (2,  1,-1):  1.0/_rt6, (2,  0, 0):  2.0/_rt6, (2, -1, 1):  1.0/_rt6,
    (2,  0,-1):  1.0/_rt2, (2, -1, 0):  1.0/_rt2, (2, -1,-1):  1.0,
}
def cg1x1(l, m1, m2):
    return _CG.get((l, m1, m2), 0.0)

def B_spherical(B):
    Bx, By, Bz = B
    return {0: Bz+0j, +1: -(Bx+1j*By)/np.sqrt(2), -1: (Bx-1j*By)/np.sqrt(2)}

B_sph = B_spherical(B_vec)

def T_lk_op(l, k, spin_idx):
    """T^{(l)}_{k} for spin spin_idx, contracted with B via CG coefficients."""
    Ss = S_sph(spin_idx)
    result = None
    for q1 in (-1, 0, 1):
        q2 = k - q1
        if q2 not in (-1, 0, 1):
            continue
        c = cg1x1(l, q1, q2)
        if abs(c) < 1e-15:
            continue
        term = c * B_sph[q2] * Ss[q1]
        result = term if result is None else result + term
    return 0.0 * Iz[0] if result is None else result

# =============================================================================
# Q^{(Z),l}_{k,m} OPERATORS
# Q^{(Z),l}_{k,m} = sum_X gamma_X (-1)^{m+1} sigma_{l,m}(X) T^{(l)}_{-k}
# =============================================================================

def build_QZ_ops(l):
    ops = {}
    for k in range(-l, l+1):
        for m in range(-l, l+1):
            phase = (-1)**(m + 1)
            op = None
            for idx, X in enumerate(NUCLEI):
                slm  = sigma_lm_all[X][(l, m)]
                T_nk = T_lk_op(l, -k, idx)
                contrib = GAMMA[X] * phase * slm * T_nk
                op = contrib if op is None else op + contrib
            ops[(k, m)] = op
    return ops

QZ1_ops = build_QZ_ops(l=1)   # (k,m) in {-1,0,1}^2
QZ2_ops = build_QZ_ops(l=2)   # (k,m) in {-2,...,2}^2

# =============================================================================
# DIPOLAR Q_hat_{k,m} OPERATORS  (collective IST over spin pairs)
# =============================================================================

def T2_pair(i, j, k):
    """Rank-2 IST for spin pair (i,j), component k."""
    if k ==  0: return (2*Iz[i]*Iz[j] - Ix[i]*Ix[j] - Iy[i]*Iy[j]) / np.sqrt(6)
    if k ==  1: return -(Ip[i]*Iz[j] + Iz[i]*Ip[j]) / 2
    if k == -1: return  (Im[i]*Iz[j] + Iz[i]*Im[j]) / 2
    if k ==  2: return  Ip[i]*Ip[j] / 2
    if k == -2: return  Im[i]*Im[j] / 2

def b_dip(gi, gj, r_vec):
    return -(mu0/(4*np.pi)) * gi * gj * hbar / np.linalg.norm(r_vec)**3

b_FC = b_dip(GAMMA['19F'], GAMMA['13C'], r_FC)
b_FH = b_dip(GAMMA['19F'], GAMMA['1H'],  r_FH)
b_CH = b_dip(GAMMA['13C'], GAMMA['1H'],  r_CH)

def a2m_factors(r_vec):
    """Orientational factors a^{(mu)}_{2,m}."""
    rhat = r_vec / np.linalg.norm(r_vec)
    A    = 3.0 * np.outer(rhat, rhat) - np.eye(3)
    return {
         0: (2*A[2,2]-A[0,0]-A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0]-A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0]-A[1,1] + 2j*A[0,1]) / 2,
    }

PAIRS = [(0, 1, b_FC, r_FC), (0, 2, b_FH, r_FH), (1, 2, b_CH, r_CH)]
alm   = {(i, j): a2m_factors(r) for i, j, _, r in PAIRS}

Qdip_ops = {
    (k, m): sum(b * alm[(i,j)][m] * T2_pair(i, j, k)
                for i, j, b, _ in PAIRS)
    for k in range(-2, 3) for m in range(-2, 3)
}

# =============================================================================
# COLLECTIVE OPERATORS  A^{(l)}_{k,m}
# A^{(1)}_{k,m} = Q^{(Z),1}_{k,m}
# A^{(2)}_{k,m} = Q^{(Z),2}_{k,m} + Q_hat_{k,m}
# =============================================================================

def A_op(l, k, m):
    """Return A^{(l)}_{k,m} as a numpy matrix."""
    if l == 1:
        return QZ1_ops[(k, m)].full()
    if l == 2:
        return (QZ2_ops[(k, m)] + Qdip_ops[(k, m)]).full()
    raise ValueError(f"l must be 1 or 2, got {l}")

def km_range(l):
    return [(k, m) for k in range(-l, l+1) for m in range(-l, l+1)]

# =============================================================================
# KRYLOV EXPANSION
#
# A^{(l),j}_{k,m} = (i [tau_c H_0, .])^j  A^{(l)}_{k,m}
# c_j              = (1/j!) * 2^{j-1} * Gamma((j+1)/2)^2
#
# Abar^{(l)}_{k,m} = sum_{j=0}^{N_TRUNC} c_j * A^{(l),j}_{k,m}
# =============================================================================

def krylov_chain(A0, H0_mat, tau_c, n_terms):
    """
    Yield A^j = (i[tau_c H0, .])^j A0 for j = 0, 1, ..., n_terms-1.
    Uses in-place iteration to avoid recomputing the full chain each time.
    """
    Aj = A0.copy()
    for _ in range(n_terms):
        yield Aj
        Aj = 1j * tau_c * (H0_mat @ Aj - Aj @ H0_mat)

def krylov_coeffs(n_terms):
    """c_j = (1/j!) * 2^{j-1} * Gamma((j+1)/2)^2  for j = 0 .. n_terms-1."""
    return np.array([
        (1.0 / factorial(j)) * 2.0**(j - 1) * gamma_func((j + 1) / 2.0)**2
        for j in range(n_terms)
    ])

def dressed_op(l, k, m, H0_mat, tau_c, n_terms=N_TRUNC):
    """
    Compute Abar^{(l)}_{k,m}(tau_c) via Eq. trunc_order_gen, truncated at
    j = n_terms - 1.  Returns (Abar, partial_norms) where partial_norms[j]
    is ||Abar truncated at j||_F for convergence monitoring.
    """
    coeffs    = krylov_coeffs(n_terms)
    A0        = A_op(l, k, m)
    Abar      = np.zeros_like(A0, dtype=complex)
    partial_norms = []
    for j, Aj in enumerate(krylov_chain(A0, H0_mat, tau_c, n_terms)):
        Abar += coeffs[j] * Aj
        partial_norms.append(np.linalg.norm(Abar, 'fro'))
    return Abar, np.array(partial_norms)

# Prefactor:  sqrt(8 tau_c / (pi^2 (2l+1)))  — from strength gamma = 2*sqrt(2)/pi
def jump_prefactor(l, tau_c):
    return np.sqrt(8.0 * tau_c / (np.pi**2 * (2*l + 1)))

# Zeroth-order limit:  L^{(l)}_{k,m} = sqrt(2 J^{(l)}(0)) * A^{(l)}_{k,m}
def J_l_0(l, tau_c):
    return tau_c / (2*l + 1)

# =============================================================================
# BUILD AND VERIFY JUMP OPERATORS
# =============================================================================

print("=" * 70)
print("Krylov jump operators  (Eq. trunc_order_gen)")
print(f"N_TRUNC = {N_TRUNC},  tau_c = {tau_c:.1e} s,  |B| = {B0*1e3:.1f} mT")
print("=" * 70)

_thresh = 1e-25   # ignore operators with Frobenius norm below this

all_jump_ops = []   # collect (l, k, m, L_mat) for downstream use

for l in (1, 2):
    pref = jump_prefactor(l, tau_c)
    J0   = J_l_0(l, tau_c)
    print(f"\n--- l = {l}  (prefactor = {pref:.4e},  J^({l})(0) = {J0:.4e} s) ---")
    print(f"  {'(k,m)':>8}  {'||A||_F':>12}  "
          f"{'||L_kry||_F':>14}  {'||L_ord0||_F':>14}  "
          f"{'rel_diff':>10}  {'converged@j':>12}")

    for k, m in km_range(l):
        A0   = A_op(l, k, m)
        norm_A = np.linalg.norm(A0, 'fro')
        if norm_A < _thresh:
            continue

        # Full Krylov dressed operator
        Abar, pnorms = dressed_op(l, k, m, H0_mat, tau_c, N_TRUNC)
        L_kry = pref * Abar

        # Zeroth-order reference
        L_ord0 = np.sqrt(2.0 * J0) * A0

        norm_kry  = np.linalg.norm(L_kry,  'fro')
        norm_ord0 = np.linalg.norm(L_ord0, 'fro')
        rel_diff  = np.linalg.norm(L_kry - L_ord0, 'fro') / norm_ord0

        # Find convergence order: first j where ||Abar_j - Abar_{j-1}|| / ||Abar_j|| < 1e-12
        conv_j = N_TRUNC - 1
        for j in range(1, N_TRUNC):
            if abs(pnorms[j] - pnorms[j-1]) / (pnorms[j] + 1e-30) < 1e-12:
                conv_j = j
                break

        print(f"  ({k:+d},{m:+d})  {norm_A:12.4e}  "
              f"{norm_kry:14.4e}  {norm_ord0:14.4e}  "
              f"{rel_diff:10.4e}  {conv_j:>12d}")

        all_jump_ops.append((l, k, m, L_kry))

# =============================================================================
# SUMMARY: relative Krylov correction aggregated over all operators
# =============================================================================

print()
print("=" * 70)
print("Summary: max relative Krylov correction per rank")
print("=" * 70)

for l in (1, 2):
    J0   = J_l_0(l, tau_c)
    pref = jump_prefactor(l, tau_c)
    max_rel = 0.0
    for ll, k, m, L_kry in all_jump_ops:
        if ll != l:
            continue
        A0     = A_op(l, k, m)
        L_ord0 = np.sqrt(2.0 * J0) * A0
        nA     = np.linalg.norm(L_ord0, 'fro')
        if nA > _thresh:
            max_rel = max(max_rel, np.linalg.norm(L_kry - L_ord0, 'fro') / nA)
    print(f"  l={l}:  max ||L_kry - L_ord0||_F / ||L_ord0||_F = {max_rel:.4e}")

print()
print(f"All {len(all_jump_ops)} non-trivial jump operators constructed.")
