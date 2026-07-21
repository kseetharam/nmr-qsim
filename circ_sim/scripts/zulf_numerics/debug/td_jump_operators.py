"""
Time-domain construction of canonical Lindblad jump operators for the
19F-13C-1H ZULF system.

Reference: liouville_hilbert_basis.tex, equations labeled TD_jumps and
trunc_order (Universal Lindbladian formalism section).

For rank-2 dipolar interactions under isotropic tumbling the jump operators are

    L_{k,m} = (2/sqrt(pi)) * sqrt(lambda_m * tau_c / 5)
              * sum_mu phi_m(mu) * T_bar_{mu,k}              [TD_jumps]

Indices:
  k in {-2,-1,0,+1,+2}   spin IST component  (determines which T_{2,k}^pair)
  m in {-2,-1,0,+1,+2}   spatial orientation component
  mu                      spin-pair index (FC, FH, CH)

This gives 5 x 5 = 25 jump operators total.

where:
  lambda_m, phi_m(mu)   eigenvalue / eigenvector of the spatial noise covariance
                        N^(2,m)_{mu,nu} = b_mu a_{2,m}^mu * (b_nu a_{2,m}^nu)*
                        Since N^(2,m) is rank-1 (outer product), each m gives
                        exactly one non-zero eigenvalue lambda_m = ||v_m||^2
                        and eigenvector phi_m = v_m / ||v_m||,
                        with v_m(mu) = b_mu * a_{2,m}^mu.

  T_bar_{mu,k}          Krylov-dressed spin-IST operator for pair mu, component k
                        [trunc_order]:

    T_bar_{mu,k} = sum_{j=0}^{K} c_j * (i tau_c [H0, .])^j T_{2,k}^mu
    c_j          = (1/j!) * 2^(j-1) * Gamma^2((j+1)/2)   (from K_0 kernel integral)

  Note: at K=0, T_bar_{mu,k} = c_0 * T_{2,k}^mu = (pi/2) * T_{2,k}^mu.
  The (2*sqrt(2)/pi) prefactor and c_0 = pi/2 together yield sqrt(2*J(0)) * Q_hat
  at K=0, matching the flat-J Redfield dissipator exactly.  Higher K orders
  add corrections proportional to (tau_c * eigenvalue(H0))^j.

Run after (or together with) zulf_lindblad.py, which provides H0, T2_op,
a2m_factors, and the physical parameters.
"""

import sys
import os
import math
import numpy as np
from scipy.special import gamma as sp_gamma

# ---------------------------------------------------------------------------
# Setup: import zulf_lindblad for shared parameters / operators
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad (runs sections 1-10 on import) ...")
import zulf_lindblad as zl
print("Done.\n")


# ---------------------------------------------------------------------------
# 1.  System parameters (imported from zulf_lindblad)
# ---------------------------------------------------------------------------
tau_c = zl.tau_c   # rotational correlation time [s]

# Spin-pair ordering: 0=19F, 1=13C, 2=1H  (same as zulf_lindblad)
PAIR_SPINS = {'FC': (0, 1), 'FH': (0, 2), 'CH': (1, 2)}
PAIRS = [
    ('FC', float(zl.b_FC), zl.r_FC),
    ('FH', float(zl.b_FH), zl.r_FH),
    ('CH', float(zl.b_CH), zl.r_CH),
]
n_pairs = len(PAIRS)
k_vals  = [-2, -1, 0, +1, +2]   # spin IST components
m_vals  = [-2, -1, 0, +1, +2]   # spatial orientation components

# H0 as a plain numpy matrix (Heisenberg Hamiltonian, rad/s)
H0_np = zl.H0.full().astype(complex)   # (8, 8)


# ---------------------------------------------------------------------------
# 2.  Orientational factors  a_{2,m}^mu  for each pair and each m
# ---------------------------------------------------------------------------
alm = {}   # dict: (label, m) -> complex
for label, _, r_vec in PAIRS:
    for m, val in zl.a2m_factors(r_vec).items():
        alm[(label, m)] = val


# ---------------------------------------------------------------------------
# 3.  Noise covariance matrix  N^(2,m)  and its single non-zero eigenvector
#
#     N^(2,m)_{mu,nu} = b_mu a_{2,m}^mu * (b_nu a_{2,m}^nu)*
#     -> rank-1: N = v v^dagger,  lambda = ||v||^2,  phi = v / ||v||
# ---------------------------------------------------------------------------
def noise_cov(m):
    """
    Return (lambda, phi) for orientation component m.

    lambda : float   - the single non-zero eigenvalue of N^(2,m)
    phi    : (3,) complex ndarray - corresponding (normalised) eigenvector
    """
    v = np.array([b * alm[(lbl, m)] for lbl, b, _ in PAIRS], dtype=complex)
    lam = float(np.real(np.dot(v.conj(), v)))
    phi = v / np.sqrt(lam) if lam > 1e-60 else np.zeros(n_pairs, dtype=complex)
    return lam, phi


# ---------------------------------------------------------------------------
# 4.  Krylov dressing coefficients  c_k = (1/k!) * 2^(k-1) * Gamma^2((k+1)/2)
#
#     c_k = integral_0^inf K_0(u) u^k du / k!
#     Values: c_0 = pi/2,  c_1 = 1,  c_2 = pi/4,  c_3 = 2/3, ...
# ---------------------------------------------------------------------------
def krylov_coeffs(K_max):
    """
    Return array c[k] for k = 0, ..., K_max.

        c_k = (1/k!) * 2^(k-1) * Gamma^2((k+1)/2)
            = (1/k!) * integral_0^inf K_0(u) u^k du

    Values: c_0 = pi/2,  c_1 = 1,  c_2 = pi/4,  c_3 = 2/3, ...
    Not normalised: c_0 = pi/2 as derived analytically from the K_0 kernel.
    """
    c = np.zeros(K_max + 1, dtype=float)
    for k in range(K_max + 1):
        c[k] = 2.0**(k - 1) / math.factorial(k) * sp_gamma((k + 1) / 2.0)**2
    return c


# ---------------------------------------------------------------------------
# 5.  Krylov sequence  (i tau_c ad_{H0})^j T_0  for j = 0, ..., K_max
# ---------------------------------------------------------------------------
def krylov_sequence(T0_np, K_max):
    """
    Compute T_j = (i tau_c [H0, .])^j T_0  for j = 0, ..., K_max.

    Returns a list of K_max+1 (8,8) complex numpy arrays.
    """
    seq    = [T0_np.astype(complex)]
    T_prev = seq[0].copy()
    for _ in range(K_max):
        T_next = 1j * tau_c * (H0_np @ T_prev - T_prev @ H0_np)
        seq.append(T_next)
        T_prev = T_next
    return seq


# ---------------------------------------------------------------------------
# 6.  Dressed spin-IST operator  T_bar_{mu,k} = sum_j c_j (i tau_c [H0,.])^j T_{2,k}^mu
# ---------------------------------------------------------------------------
def dressed_op(T_bare_qobj, K_max):
    """
    Dress T_{2,k}^mu (bare spin-IST, QuTiP Qobj) with the Krylov expansion
    up to order K_max.  Returns an (8,8) complex numpy array.

    At K_max=0: T_bar = c_0 * T_{2,k}^mu = (pi/2) * T_{2,k}^mu.
    """
    T0   = T_bare_qobj.full().astype(complex)
    seq  = krylov_sequence(T0, K_max)
    coef = krylov_coeffs(K_max)
    return sum(c * T for c, T in zip(coef, seq))


# ---------------------------------------------------------------------------
# 7.  Build all 25 jump operators for a given truncation order
# ---------------------------------------------------------------------------
def build_td_jump_operators(trunc_order=0):
    """
    Construct the 25 Lindblad jump operators L_{k,m} [TD_jumps].

    For each spin-IST component k and spatial component m:

        L_{k,m} = (2/sqrt(pi)) * sqrt(lambda_m * tau_c / 5)
                  * sum_mu phi_m(mu) * T_bar_{mu,k}

    where:
      - lambda_m, phi_m  : eigenvalue/eigenvector of the spatial noise covariance
                           for orientation component m  (from noise_cov(m))
      - T_bar_{mu,k}     : Krylov-dressed T_{2,k}^{pair mu}  (from dressed_op)
      - The sum over mu runs over spin pairs (FC, FH, CH)

    Parameters
    ----------
    trunc_order : int
        Maximum Krylov order K.  K=0 = bare ISTs, K>0 = dressed by H0.

    Returns
    -------
    L_ops : dict  (k, m) -> (8, 8) complex ndarray
        25 jump operators.
    prefactors : dict  m -> float
        Amplitude prefactor (2/sqrt(pi)) * sqrt(lambda_m * tau_c / 5) per m.
    """
    K_max = trunc_order

    # Pre-compute Krylov-dressed T_{2,k}^{pair} for every (pair, spin component k)
    # T_bar_spin[(label, k)] = T_bar_{mu,k}  (8x8 array)
    T_bar_spin = {}
    for label, _, _ in PAIRS:
        si, sj = PAIR_SPINS[label]
        for k in k_vals:
            T_bar_spin[(label, k)] = dressed_op(zl.T2_op(si, sj, k), K_max)

    # Pre-compute spatial noise weights for every m
    noise = {m: noise_cov(m) for m in m_vals}   # m -> (lambda_m, phi_m)

    L_ops      = {}
    prefactors = {}

    for m in m_vals:
        lam, phi = noise[m]
        # (2*sqrt(2)/pi) * sqrt(lam*tau_c/5) ensures that at K=0, where c_0=pi/2,
        # the effective coefficient on Q_hat_{k,m} equals sqrt(2*J(0)) = sqrt(2*tau_c/5),
        # which reproduces the flat-J Redfield dissipator exactly.
        pref = (2.0 * np.sqrt(2.0) / np.pi) * np.sqrt(lam * tau_c / 5.0)
        prefactors[m] = pref

        for k in k_vals:
            # L_{k,m} = pref * sum_mu phi_m(mu) * T_bar_{mu,k}
            L = pref * sum(
                phi[mu] * T_bar_spin[(PAIRS[mu][0], k)]
                for mu in range(n_pairs)
            )
            L_ops[(k, m)] = L

    return L_ops, prefactors


# ---------------------------------------------------------------------------
# 8.  Hilbert-Schmidt utilities
# ---------------------------------------------------------------------------
def hs_norm(A):
    """Hilbert-Schmidt norm sqrt(Tr(A^dag A))."""
    return np.sqrt(max(0.0, np.real(np.trace(A.conj().T @ A))))

# ---------------------------------------------------------------------------
# 9.  Main: sweep trunc_order and report norms / convergence
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    print("=" * 65)
    print("Time-domain Lindblad jump operators  [TD_jumps + trunc_order]")
    print(f"System : 19F-13C-1H   tau_c = {tau_c:.1e} s   B0 = 0 T")
    print("=" * 65)

    # Krylov expansion parameter: how fast the series converges
    J_FC_rad = 2 * np.pi * zl.J_FC
    J_CH_rad = 2 * np.pi * zl.J_CH
    print(f"\nExpansion parameters  J*tau_c  (order of dressing corrections):")
    print(f"  J_FC * tau_c = {J_FC_rad * tau_c:.3e}")
    print(f"  J_CH * tau_c = {J_CH_rad * tau_c:.3e}")

    # -----------------------------------------------------------------------
    # Print Krylov coefficients
    # -----------------------------------------------------------------------
    K_show = 6
    print(f"\nKrylov dressing coefficients  c_k = (1/k!) 2^(k-1) Gamma^2((k+1)/2):")
    for k, c in enumerate(krylov_coeffs(K_show)):
        print(f"  k={k}: c_{k} = {c:.6f}")

    # -----------------------------------------------------------------------
    # Build operators for each truncation order and report HS norms
    # -----------------------------------------------------------------------
    print()
    for K in [0, 1, 2, 5]:
        L_ops, prefs = build_td_jump_operators(trunc_order=K)
        print(f"--- trunc_order = {K} ---  "
              f"(J_FC*tau_c ~ {J_FC_rad*tau_c:.1e}, corrections O((J*tau_c)^{K+1}))")
        print("  Rows = k (spin IST),  Cols = m (spatial orientation)  [||L||_HS]")
        header = "         " + "  ".join(f"  m={m:+d}" for m in m_vals)
        print(header)
        for k in k_vals:
            row = f"  k={k:+d}  "
            for m in m_vals:
                row += f" {hs_norm(L_ops[(k, m)]):7.4f}"
            print(row)
        print()

    # -----------------------------------------------------------------------
    # Convergence: ||L_{k,m}(K) - L_{k,m}(0)||_HS / ||L_{k,m}(0)||_HS
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("Convergence: ||L_{k,m}(K) - L_{k,m}(0)||_HS / ||L_{k,m}(0)||_HS")
    print("(max over all (k,m) pairs)")
    print("=" * 65)
    L_base, _ = build_td_jump_operators(trunc_order=0)
    for K in range(1, 6):
        L_curr, _ = build_td_jump_operators(trunc_order=K)
        rel_changes = [
            hs_norm(L_curr[(k, m)] - L_base[(k, m)]) / (hs_norm(L_base[(k, m)]) + 1e-60)
            for k in k_vals for m in m_vals
        ]
        print(f"  K={K}  max rel change = {max(rel_changes):.3e}  "
              f"mean = {np.mean(rel_changes):.3e}")

    print("\nDone.")
