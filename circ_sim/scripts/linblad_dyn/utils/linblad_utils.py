"""
linblad_utils.py — Utility functions for n spin-1/2 Lindblad NMR simulation.

Constructs the isotropic coherent Hamiltonian and all rank-1/rank-2 Lindblad
jump operators from molecular parameters.  No Redfield-specific code is included.

Shared interface convention
---------------------------
gammas        : array-like (n,)        gyromagnetic ratios [rad/s/T]
J_hz          : array-like (n,n)       J-coupling matrix [Hz], symmetric, zero diagonal
coords_ang    : array-like (n,3)       nuclear coordinates [Angstrom]
sigma_ppm_list: list of n (3,3) arrays Zeeman chemical shielding tensors [ppm]
B_vec         : array-like (3,)        external magnetic field [T]
tau_c         : float                  rotational correlation time [s]

All spin operators are QuTiP Qobj in the 2^n-dimensional Hilbert space.
Spin ordering follows the ordering of `gammas`.
"""

import numpy as np
import qutip as qt
from itertools import combinations
from math import factorial as _factorial
from scipy.special import gamma as _gamma_func

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR     = 1.054571817e-34    # J·s
MU0      = 1.25663706212e-6   # T·m/A
ANGSTROM = 1e-10              # m

# ---------------------------------------------------------------------------
# Clebsch-Gordan coefficients  <1, m1; 1, m2 | l, m1+m2>  for l = 0, 1, 2
# ---------------------------------------------------------------------------
_RT2 = np.sqrt(2.0)
_RT3 = np.sqrt(3.0)
_RT6 = np.sqrt(6.0)

_CG_TABLE = {
    # l = 0
    (0,  1, -1):  1.0/_RT3,
    (0,  0,  0): -1.0/_RT3,
    (0, -1,  1):  1.0/_RT3,
    # l = 1
    (1,  1,  0):  1.0/_RT2,  (1,  0,  1): -1.0/_RT2,
    (1,  1, -1):  1.0/_RT2,  (1,  0,  0):  0.0,
    (1, -1,  1): -1.0/_RT2,  (1,  0, -1):  1.0/_RT2,
    (1, -1,  0): -1.0/_RT2,
    # l = 2
    (2,  1,  1):  1.0,
    (2,  1,  0):  1.0/_RT2,  (2,  0,  1):  1.0/_RT2,
    (2,  1, -1):  1.0/_RT6,  (2,  0,  0):  2.0/_RT6,  (2, -1,  1):  1.0/_RT6,
    (2,  0, -1):  1.0/_RT2,  (2, -1,  0):  1.0/_RT2,
    (2, -1, -1):  1.0,
}

def _cg1x1(l, m1, m2):
    """<1,m1; 1,m2 | l, m1+m2>.  Returns 0 for entries not in the table."""
    return _CG_TABLE.get((l, m1, m2), 0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _zero_op(n):
    """Zero operator in the 2^n Hilbert space."""
    dim = [2] * n
    return qt.Qobj(np.zeros((2**n, 2**n), dtype=complex), dims=[dim, dim])


def _B_spherical(B_vec):
    """Spherical components of B = (Bx, By, Bz) as dict {q: complex}."""
    Bx, By, Bz = B_vec
    return {
         0:  Bz + 0j,
        +1: -(Bx + 1j*By) / np.sqrt(2),
        -1:  (Bx - 1j*By) / np.sqrt(2),
    }


def _sigma_lm(sigma_t):
    """
    Decompose a 3x3 shielding tensor into irreducible spherical components.

    Returns dict {(l, m): complex} for l = 0, 1, 2 and m = -l..+l.
    Input sigma_t is the physical tensor sigma_tilde = 1 + 1e-6 * sigma_ppm.
    """
    s = np.asarray(sigma_t, dtype=complex)
    s_iso = np.trace(s) / 3.0
    return {
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_spin_ops(n):
    """
    Build single-spin-1/2 operators embedded in the full 2^n Hilbert space.

    Parameters
    ----------
    n : int
        Number of spins.

    Returns
    -------
    dict with keys:
        'Ix', 'Iy', 'Iz' : list of n QuTiP Qobj   Cartesian spin operators
        'Ip', 'Im'        : list of n QuTiP Qobj   raising / lowering operators
        'S_sph'           : list of n dicts {q: Qobj}
                           spherical components  q=0: Sz,  q=±1: ∓S±/√2
    """
    def _embed(op_char, idx):
        ops_list = [qt.qeye(2)] * n
        ops_list[idx] = qt.jmat(0.5, op_char)
        return qt.tensor(ops_list)

    Ix = [_embed('x', k) for k in range(n)]
    Iy = [_embed('y', k) for k in range(n)]
    Iz = [_embed('z', k) for k in range(n)]
    Ip = [Ix[k] + 1j*Iy[k] for k in range(n)]
    Im = [Ix[k] - 1j*Iy[k] for k in range(n)]
    S_sph = [
        {0: Iz[k], +1: -Ip[k]/np.sqrt(2), -1: Im[k]/np.sqrt(2)}
        for k in range(n)
    ]
    return {'Ix': Ix, 'Iy': Iy, 'Iz': Iz, 'Ip': Ip, 'Im': Im, 'S_sph': S_sph}


def build_H_iso(gammas, J_hz, coords_ang, sigma_ppm_list, B_vec, ops=None):
    """
    Build the isotropic coherent Hamiltonian.

        H_iso = Σ_{i<j} 2π J_{ij} S_i·S_j  −  Σ_i γ_i σ_iso(i) S_i·B

    Parameters
    ----------
    gammas        : array-like (n,)
    J_hz          : array-like (n,n)   symmetric, diagonal entries ignored
    coords_ang    : array-like (n,3)   not used here, kept for interface consistency
    sigma_ppm_list: list of n (3,3) arrays
    B_vec         : array-like (3,)
    ops           : dict, optional     output of build_spin_ops; built if None

    Returns
    -------
    H_iso  : QuTiP Qobj     Hamiltonian [rad/s]
    ops    : dict            spin operators
    evals  : (2^n,) ndarray eigenvalues [rad/s], ascending
    ekets  : list of Qobj   corresponding eigenstates
    """
    gammas = np.asarray(gammas, dtype=float)
    J_hz   = np.asarray(J_hz,   dtype=float)
    B_vec  = np.asarray(B_vec,  dtype=float)
    n      = len(gammas)

    if ops is None:
        ops = build_spin_ops(n)
    Ix, Iy, Iz = ops['Ix'], ops['Iy'], ops['Iz']

    sigma_tilde_list = [np.eye(3) + 1e-6 * np.asarray(s) for s in sigma_ppm_list]
    sigma_iso = np.array([np.trace(st) / 3.0 for st in sigma_tilde_list])

    # Heisenberg J-coupling term
    H_J = _zero_op(n)
    for i, j in combinations(range(n), 2):
        if abs(J_hz[i, j]) > 0:
            H_J += 2*np.pi * J_hz[i, j] * (Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j])

    # Isotropic Zeeman shift
    H_Z = _zero_op(n)
    for i in range(n):
        H_Z += -gammas[i] * sigma_iso[i] * (
            B_vec[0]*Ix[i] + B_vec[1]*Iy[i] + B_vec[2]*Iz[i]
        )

    H_iso = H_J + H_Z
    evals, ekets = H_iso.eigenstates()
    return H_iso, ops, evals, ekets


def build_Qdip_ops(gammas, coords_ang, ops):
    """
    Build the 25 dipolar rank-2 system operators Q̂_{k,m}.

        Q̂_{k,m} = Σ_{i<j} b_{ij} a_{2,m}^{ij} T_{2,k}^{ij}

    b_{ij} = −(μ₀/4π) γ_i γ_j ℏ / r_{ij}³   [rad/s]
    a_{2,m}^{ij}: orientational factors from the traceless dipolar tensor
    T_{2,k}^{ij}: rank-2 two-spin IST

    Parameters
    ----------
    gammas     : array-like (n,)
    coords_ang : array-like (n,3)   [Angstrom]
    ops        : dict               output of build_spin_ops

    Returns
    -------
    dict {(k, m): QuTiP Qobj}   k, m ∈ {-2, -1, 0, +1, +2}
    """
    gammas   = np.asarray(gammas, dtype=float)
    coords_m = np.asarray(coords_ang, dtype=float) * ANGSTROM
    n        = len(gammas)
    Ix, Iy, Iz, Ip, Im = ops['Ix'], ops['Iy'], ops['Iz'], ops['Ip'], ops['Im']

    def _b_dip(i, j):
        r = np.linalg.norm(coords_m[j] - coords_m[i])
        return -(MU0 / (4*np.pi)) * gammas[i] * gammas[j] * HBAR / r**3

    def _a2m(r_vec):
        rhat = r_vec / np.linalg.norm(r_vec)
        A = 3.0*np.outer(rhat, rhat) - np.eye(3)
        return {
             0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
            +1: -(A[0,2] - 1j*A[1,2]),
            -1:  (A[0,2] + 1j*A[1,2]),
            +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
            -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
        }

    def _T2(i, j, k):
        if k ==  0: return (2*Iz[i]*Iz[j] - Ix[i]*Ix[j] - Iy[i]*Iy[j]) / np.sqrt(6)
        if k ==  1: return -(Ip[i]*Iz[j] + Iz[i]*Ip[j]) / 2
        if k == -1: return  (Im[i]*Iz[j] + Iz[i]*Im[j]) / 2
        if k ==  2: return   Ip[i]*Ip[j] / 2
        if k == -2: return   Im[i]*Im[j] / 2

    pairs = list(combinations(range(n), 2))
    b     = {(i, j): _b_dip(i, j) for i, j in pairs}
    a     = {(i, j): _a2m(coords_m[j] - coords_m[i]) for i, j in pairs}

    result = {}
    for k in range(-2, 3):
        for m in range(-2, 3):
            op = _zero_op(n)
            for i, j in pairs:
                op += b[(i,j)] * a[(i,j)][m] * _T2(i, j, k)
            result[(k, m)] = op
    return result


def build_QZ_ops(l, gammas, sigma_ppm_list, B_vec, ops):
    """
    Build the (2l+1)² Zeeman CSA system operators Q^{(Z),l}_{k,m}.

        Q^{(Z),l}_{k,m} = Σ_X γ_X (-1)^{m+1} σ_{l,m}(X) T^{(l)}_{-k}(S_X, B)

    T^{(l)}_k(S_X, B) = Σ_{q1+q2=k} <1,q1;1,q2|l,k> (S_X)_{q1} B_{q2}

    Parameters
    ----------
    l             : int (1 or 2)           rank
    gammas        : array-like (n,)
    sigma_ppm_list: list of n (3,3) arrays
    B_vec         : array-like (3,)
    ops           : dict                   output of build_spin_ops

    Returns
    -------
    dict {(k, m): QuTiP Qobj}   k, m ∈ {-l, ..., +l}
    """
    gammas = np.asarray(gammas, dtype=float)
    B_vec  = np.asarray(B_vec,  dtype=float)
    n      = len(gammas)
    B_sph  = _B_spherical(B_vec)
    S_sph  = ops['S_sph']
    Iz     = ops['Iz']

    sigma_tilde_list = [np.eye(3) + 1e-6*np.asarray(s) for s in sigma_ppm_list]
    slm_all = [_sigma_lm(st) for st in sigma_tilde_list]

    def _T_lk(l, k, spin_idx):
        """Rank-l tensor product [S_X ⊗ B]^{(l)}_k for spin spin_idx."""
        Ss = S_sph[spin_idx]
        result = None
        for q1 in (-1, 0, 1):
            q2 = k - q1
            if q2 not in (-1, 0, 1):
                continue
            c = _cg1x1(l, q1, q2)
            if abs(c) < 1e-15:
                continue
            term = c * B_sph[q2] * Ss[q1]
            result = term if result is None else result + term
        return 0.0 * Iz[0] if result is None else result

    result = {}
    for k in range(-l, l+1):
        for m in range(-l, l+1):
            phase = (-1)**(m + 1)
            op = _zero_op(n)
            for i in range(n):
                slm   = slm_all[i][(l, m)]
                T_neg = _T_lk(l, -k, i)
                op   += gammas[i] * phase * slm * T_neg
            result[(k, m)] = op
    return result


def build_jump_operators(gammas, J_hz, coords_ang, sigma_ppm_list, B_vec, tau_c):
    """
    Construct all Lindblad jump operators for an n spin-1/2 system.

    From subsubsection "Linbladian framework" in liouville_hilbert_basis.tex:

        L^{(1)}_{k,m} = √(2 J^{(1)}(0)) Q^{(Z),1}_{k,m}
        L^{(2)}_{k,m} = √(2 J^{(2)}(0)) (Q^{(Z),2}_{k,m} + Q̂_{k,m})

    with J^{(l)}(0) = τ_c / (2l+1)  (spectral density at zero frequency).

    Valid in the extreme-narrowing limit: ω τ_c ≪ 1 for all relevant transition
    frequencies ω.  Outside this regime use the full frequency-dependent Redfield
    rates (implemented in a separate module).

    Parameters
    ----------
    gammas        : array-like (n,)
    J_hz          : array-like (n,n)
    coords_ang    : array-like (n,3)   [Angstrom]
    sigma_ppm_list: list of n (3,3) arrays   [ppm]
    B_vec         : array-like (3,)   [T]
    tau_c         : float             [s]

    Returns
    -------
    H_iso  : QuTiP Qobj
        Isotropic Hamiltonian [rad/s].
    L1_ops : dict {(k, m): QuTiP Qobj}
        Rank-1 jump operators (9 entries, k,m ∈ {-1,0,+1}).
        All zero when all shielding tensors are symmetric (σ_{1,m}=0).
    L2_ops : dict {(k, m): QuTiP Qobj}
        Rank-2 jump operators (25 entries, k,m ∈ {-2,-1,0,+1,+2}).
    ops    : dict
        Spin operators from build_spin_ops.
    evals  : (2^n,) ndarray
        H_iso eigenvalues [rad/s], ascending.
    ekets  : list of QuTiP Qobj
        H_iso eigenstates.
    """
    gammas = np.asarray(gammas, dtype=float)

    ops = build_spin_ops(len(gammas))

    H_iso, ops, evals, ekets = build_H_iso(
        gammas, J_hz, coords_ang, sigma_ppm_list, B_vec, ops
    )

    QZ1  = build_QZ_ops(1, gammas, sigma_ppm_list, B_vec, ops)
    QZ2  = build_QZ_ops(2, gammas, sigma_ppm_list, B_vec, ops)
    Qdip = build_Qdip_ops(gammas, coords_ang, ops)

    scale1 = np.sqrt(2.0 * tau_c / 3.0)   # √(2 J^{(1)}(0))
    scale2 = np.sqrt(2.0 * tau_c / 5.0)   # √(2 J^{(2)}(0))

    L1_ops = {
        (k, m): scale1 * QZ1[(k, m)]
        for k in range(-1, 2) for m in range(-1, 2)
    }
    L2_ops = {
        (k, m): scale2 * (QZ2[(k, m)] + Qdip[(k, m)])
        for k in range(-2, 3) for m in range(-2, 3)
    }

    return H_iso, L1_ops, L2_ops, ops, evals, ekets


# ---------------------------------------------------------------------------
# Private helper: Krylov coefficient  c_j = (1/j!) 2^{j-1} Γ((j+1)/2)²
# These are the moments ∫₀^∞ u^j K₀(u) du / j!  (tex Eq. trunc_order_gen).
# ---------------------------------------------------------------------------

def _krylov_coeff(j):
    return (1.0 / _factorial(j)) * 2.0**(j - 1) * _gamma_func((j + 1) / 2.0)**2


def build_jump_operators_krylov(
        gammas, J_hz, coords_ang, sigma_ppm_list, B_vec, tau_c, max_order):
    """
    Construct Lindblad jump operators via the Krylov expansion of the dressed
    collective operators (Eq. trunc_order_gen, liouville_hilbert_basis.tex).

    The collective operators are:

        A^{(1)}_{k,m} = Q^{(Z),1}_{k,m}
        A^{(2)}_{k,m} = Q^{(Z),2}_{k,m} + Q̂_{k,m}

    Each is dressed by the isotropic Hamiltonian H_iso via the Krylov chain:

        Ā^{(l)}_{k,m} = Σ_{j ∈ orders} c_j · (i [τ_c H_iso, ·])^j  A^{(l)}_{k,m}

        c_j = (1/j!) · 2^{j-1} · Γ((j+1)/2)²

    and the jump operators are:

        L^{(l)}_{k,m} = sqrt(8 τ_c / (π² (2l+1))) · Ā^{(l)}_{k,m}

    Setting max_order=0 recovers the extreme-narrowing result of
    build_jump_operators (i.e. L = √(2J^{(l)}(0)) · A).

    Convergence note
    ----------------
    The series converges when ‖τ_c H_iso‖ ≲ 1.  Because odd- and even-order
    partial sums have alternating imaginary/real characters (factors of i^j),
    the residual error oscillates before decreasing; even-order-only subsets
    (e.g. max_order=[0, 2]) converge monotonically.

    Parameters
    ----------
    gammas        : array-like (n,)
    J_hz          : array-like (n,n)
    coords_ang    : array-like (n,3)   [Angstrom]
    sigma_ppm_list: list of n (3,3) arrays   [ppm]
    B_vec         : array-like (3,)   [T]
    tau_c         : float             [s]
    max_order     : int or iterable of int
        If int N, include all Krylov terms j = 0, 1, …, N.
        If an iterable of ints (e.g. [0, 2]), include only those specific
        terms — intermediate chain steps are still computed but not
        accumulated, allowing selective even- or odd-only subsets.

    Returns
    -------
    H_iso  : QuTiP Qobj
    L1_ops : dict {(k, m): QuTiP Qobj}   rank-1 jump operators
    L2_ops : dict {(k, m): QuTiP Qobj}   rank-2 jump operators
    ops    : dict                          spin operators
    evals  : (2^n,) ndarray
    ekets  : list of QuTiP Qobj
    """
    gammas = np.asarray(gammas, dtype=float)
    n      = len(gammas)

    # Resolve order_set and max_j from the max_order argument
    if isinstance(max_order, int):
        order_set = set(range(max_order + 1))
    else:
        order_set = set(int(j) for j in max_order)
    max_j = max(order_set)

    ops = build_spin_ops(n)
    H_iso, ops, evals, ekets = build_H_iso(
        gammas, J_hz, coords_ang, sigma_ppm_list, B_vec, ops
    )

    QZ1  = build_QZ_ops(1, gammas, sigma_ppm_list, B_vec, ops)
    QZ2  = build_QZ_ops(2, gammas, sigma_ppm_list, B_vec, ops)
    Qdip = build_Qdip_ops(gammas, coords_ang, ops)

    H0_mat = H_iso.full()
    D      = H0_mat.shape[0]
    dim    = [[2]*n, [2]*n]

    coeffs = {j: _krylov_coeff(j) for j in order_set}
    pref   = {l: np.sqrt(8.0 * tau_c / (np.pi**2 * (2*l + 1))) for l in (1, 2)}

    def _dress(A_qobj):
        """Return Ā accumulating only the terms whose index is in order_set."""
        A_mat = A_qobj.full()
        Abar  = np.zeros((D, D), dtype=complex)
        Aj    = A_mat.copy()
        for j in range(max_j + 1):
            if j in order_set:
                Abar += coeffs[j] * Aj
            if j < max_j:
                Aj = 1j * tau_c * (H0_mat @ Aj - Aj @ H0_mat)
        return Abar

    L1_ops = {
        (k, m): qt.Qobj(pref[1] * _dress(QZ1[(k, m)]), dims=dim)
        for k in range(-1, 2) for m in range(-1, 2)
    }
    L2_ops = {
        (k, m): qt.Qobj(pref[2] * _dress(QZ2[(k, m)] + Qdip[(k, m)]), dims=dim)
        for k in range(-2, 3) for m in range(-2, 3)
    }

    return H_iso, L1_ops, L2_ops, ops, evals, ekets


def active_jump_ops(L1_ops, L2_ops, thresh=1e-20):
    """
    Return a flat list of non-negligible jump operator matrices (numpy arrays).

    Filters out operators with Frobenius norm below `thresh`.  Useful for
    constructing the Lindblad dissipator efficiently.

    Parameters
    ----------
    L1_ops : dict {(k,m): QuTiP Qobj}   rank-1 jump operators
    L2_ops : dict {(k,m): QuTiP Qobj}   rank-2 jump operators
    thresh : float                        Frobenius norm threshold

    Returns
    -------
    list of (D, D) complex ndarrays   where D = 2^n
    """
    result = []
    for ops_dict in (L1_ops, L2_ops):
        for op in ops_dict.values():
            mat = op.full()
            if np.sqrt(np.real(np.trace(mat.conj().T @ mat))) > thresh:
                result.append(mat)
    return result


def build_lindblad_liouvillian(H_iso, L1_ops, L2_ops, thresh=1e-20):
    """
    Assemble the full Lindblad Liouvillian superoperator.

        L[ρ] = −i[H_iso, ρ] + Σ_α (L_α ρ L_α† − ½{L_α†L_α, ρ})

    Uses column-major (QuTiP) Liouville convention:
        L ρ L†  →  kron(L*, L)
        L†L ρ   →  kron(I, L†L)
        ρ L†L   →  kron((L†L)ᵀ, I)

    Parameters
    ----------
    H_iso  : QuTiP Qobj
    L1_ops : dict {(k,m): QuTiP Qobj}
    L2_ops : dict {(k,m): QuTiP Qobj}
    thresh : float   Frobenius norm threshold below which operators are skipped

    Returns
    -------
    L_mat : (D², D²) complex ndarray   full Liouvillian matrix, D = 2^n
    """
    L_ham = (-1j * (qt.spre(H_iso) - qt.spost(H_iso))).full()

    D = H_iso.shape[0]
    I_D = np.eye(D, dtype=complex)
    L_diss = np.zeros((D**2, D**2), dtype=complex)

    for Lmat in active_jump_ops(L1_ops, L2_ops, thresh=thresh):
        LdL = Lmat.conj().T @ Lmat
        L_diss += (np.kron(Lmat.conj(), Lmat)
                   - 0.5 * np.kron(I_D, LdL)
                   - 0.5 * np.kron(LdL.T, I_D))

    return L_ham + L_diss
