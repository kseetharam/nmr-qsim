"""
Canonical Lindblad jump operators from the energy-domain formula.

Reference: liouville_hilbert_basis.tex, equation labeled `can_en_domain`
(just before the "Fermi's Golden Rule" subsection):

    L_{k,m} = sqrt(J(0))  *  sum_i  (Q_hat_{k,m})_i  *  sigma_i       [can_en_domain]

where:

  alpha = (k, m)
    Composite noise-channel index; k and m each run over {-2,-1,0,+1,+2},
    giving 25 independent channels from isotropic tumbling (distinct Wigner
    function components D_{k,m}^(2) are uncorrelated for different (k,m)).

  Q_hat_{k,m}  --  collective IST system-bath coupling operator
    [equation at line 504 of the notes; user refers to this as "col_IST"]

        Q_hat_{k,m} = sum_{pairs (i,j)}  b_{ij}  *  a_{2,m}^{ij}  *  T_{2,k}^{i,j}

    - b_{ij}  = -(mu_0/4pi) * gamma_i * gamma_j * hbar / r_{ij}^3
                dipolar coupling constant [rad/s]
    - a_{2,m}^{ij}  orientational factor for spatial component m
                    (from the dipolar tensor; eq. alm_start/end in notes)
    - T_{2,k}^{i,j}  rank-2 two-body IST operator for spin pair (i,j),
                      component k  [returned by zl.T2_op(i, j, k)]
    The index k labels the *spin* part of the IST; m labels the *spatial*
    part.  They are independent because H_DD = sum_{k,m} D_{k,m}^(2) Q_hat_{k,m}.

  sigma_i  --  transition operators (H_0 eigenstates)
    sigma_i = |n_i><m_i|  where |n_i>, |m_i> are H_0 eigenstates.
    The index i labels every off-diagonal pair (n, m) with n != m.
    For an 8-level system: 8*7 = 56 transition operators.

  (Q_hat_{k,m})_i  --  projection of Q_hat_{k,m} onto transition i
    (Q_hat_{k,m})_i = Tr{ sigma_i^dag * Q_hat_{k,m} }
                    = Tr{ |m_i><n_i| * Q_hat_{k,m} }
                    = <n_i| Q_hat_{k,m} |m_i>
    i.e., the (n_i, m_i) matrix element of Q_hat_{k,m} in the H_0 eigenbasis.

  J(0)  --  spectral density at zero frequency (Lorentzian peak value)

Reconstruction identity:
  Summing  (Q_hat_{k,m})_i * sigma_i  over ALL off-diagonal transitions i
  reconstructs the off-diagonal part of Q_hat_{k,m} in the H_0 eigenbasis:

      sum_i (Q_hat_{k,m})_i * sigma_i
        = sum_{n != m} <n|Q_hat_{k,m}|m>  |n><m|
        = off_diag( U^dag * Q_hat_{k,m} * U )   [in eigenbasis]

  Therefore the jump operator is simply:

      L_{k,m} = sqrt(J(0)) * off_diag( Q_hat_{k,m}^{(eig)} )

  where Q_hat_{k,m}^{(eig)} = U^dag * Q_hat_{k,m} * U  and off_diag()
  zeros the diagonal.  The diagonal elements of Q_hat_{k,m}^{(eig)} would
  contribute only a (Lamb-shift) commutator term, not dissipation.

System: 19F-13C-1H ZULF 3-spin model.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ZULF_DIR   = os.path.dirname(_SCRIPT_DIR)         # .../zulf_numerics/
if _ZULF_DIR not in sys.path:
    sys.path.insert(0, _ZULF_DIR)

import zulf_lindblad as zl

# ---------------------------------------------------------------------------
# System parameters (from zulf_lindblad)
# ---------------------------------------------------------------------------
n_states    = zl.n_states          # 8
H0_np       = zl.H0.full()         # (8, 8) Heisenberg Hamiltonian [rad/s]
J0          = zl.J_spectral(0.0)   # spectral density at omega=0  [s]

# Eigenbasis: ekets_arr[k] = k-th eigenstate as a row vector
# Columns of U are eigenvectors:  U = ekets_arr.T,  so U[:,k] = |E_k>
U           = zl.ekets_arr.T       # (8, 8), columns = eigenvectors

# Dipolar coupling constants b_pair [rad/s] and internuclear vectors r_pair
# Spin indices: 0=19F, 1=13C, 2=1H  (same as zulf_lindblad)
PAIRS = [
    ('FC', 0, 1, float(zl.b_FC), zl.r_FC),
    ('FH', 0, 2, float(zl.b_FH), zl.r_FH),
    ('CH', 1, 2, float(zl.b_CH), zl.r_CH),
]
n_pairs = len(PAIRS)

# IST component indices
k_vals = m_vals = [-2, -1, 0, +1, +2]

print("=" * 65)
print("Canonical Lindblad operators  [can_en_domain]")
print(f"System : 19F-13C-1H ZULF   B0 = 0 T")
print(f"J(0)   = {J0:.6e} s")
print(f"sqrt(J(0)) = {np.sqrt(J0):.6e} sqrt(s)")
print("=" * 65)

# ---------------------------------------------------------------------------
# Step 1 — Precompute orientational factors a_{2,m}^{pair} for each pair & m
#
# a_{2,m}^{pair} is a scalar complex number encoding the projection of the
# pair's internuclear direction onto the m-th spatial spherical harmonic.
# These are the same factors used in td_jump_operators.py.
# ---------------------------------------------------------------------------
a_factors = {}   # dict: (pair_label, m) -> complex scalar
for label, si, sj, b, r_vec in PAIRS:
    for m, val in zl.a2m_factors(r_vec).items():
        a_factors[(label, m)] = val

# ---------------------------------------------------------------------------
# Step 2 — Build Q_hat_{k,m} for every (k, m) pair
#
# Q_hat_{k,m} = sum_{pairs} b_{pair} * a_{2,m}^{pair} * T_{2,k}^{pair}
#
# Each T_{2,k}^{pair} is an 8x8 operator acting on the full 3-spin Hilbert
# space.  zl.T2_op(si, sj, k) returns the rank-2 IST with component index k
# for the spin pair (si, sj).
#
# Result: Q_hat[k_idx, m_idx] is an (8, 8) complex array.
# ---------------------------------------------------------------------------
Q_hat = {}   # dict: (k, m) -> (8, 8) complex array  [rad/s]

for k in k_vals:
    # Precompute T_{2,k}^{pair} for all pairs at this k (avoids recomputing)
    T2k = {label: zl.T2_op(si, sj, k).full().astype(complex)
           for label, si, sj, b, r_vec in PAIRS}
    for m in m_vals:
        Q = np.zeros((n_states, n_states), dtype=complex)
        for label, si, sj, b, r_vec in PAIRS:
            # Contribution of this pair to Q_hat_{k,m}
            # b [rad/s] * a_{2,m}^{pair} [dimensionless] * T_{2,k}^{pair} [dimensionless]
            Q += b * a_factors[(label, m)] * T2k[label]
        Q_hat[(k, m)] = Q

print(f"\nBuilt Q_hat_{{k,m}} for all {len(k_vals)*len(m_vals)} (k,m) combinations.\n")

# ---------------------------------------------------------------------------
# Step 3 — Express Q_hat_{k,m} in the H_0 eigenbasis
#
# Q_hat_{k,m}^{(eig)} = U^dag * Q_hat_{k,m} * U
#
# Element [a, b] = <E_a| Q_hat_{k,m} |E_b>  = (Q_hat_{k,m})_{sigma_i}
# for the transition sigma_i = |a><b|.
# ---------------------------------------------------------------------------
def to_eigenbasis(Q_np):
    """Transform 8x8 operator to H_0 eigenbasis: U^dag Q U."""
    return U.conj().T @ Q_np @ U

# ---------------------------------------------------------------------------
# Step 4 — Build jump operators L_{k,m}
#
# L_{k,m} = sqrt(J(0)) * off_diag( Q_hat_{k,m}^{(eig)} )
#
# The off-diagonal mask zeroes the diagonal, keeping only the
# transition-operator part that drives dissipation.
# ---------------------------------------------------------------------------
sqrt_J0 = np.sqrt(J0)

# L_{k,m} = sqrt(J(0)) * Q_hat_{k,m}^{(eig)}
# Transition operators sigma_i = |n_i><m_i| include diagonal n_i=m_i,
# so the full matrix (diagonal + off-diagonal) is used.
L_ops = {}   # dict: (k, m) -> (8, 8) complex jump operator  [rad^{1/2} s^{-1/2}]

for k in k_vals:
    for m in m_vals:
        L_ops[(k, m)] = sqrt_J0 * to_eigenbasis(Q_hat[(k, m)])

# ---------------------------------------------------------------------------
# Diagnostics — Hilbert-Schmidt norms and structure
# ---------------------------------------------------------------------------
def hs_norm(A):
    return np.sqrt(max(0.0, np.real(np.trace(A.conj().T @ A))))

print("Hilbert-Schmidt norms ||L_{k,m}||_HS  [sqrt(rad) / sqrt(s)]")
print("Rows = k (spin IST),  Cols = m (spatial orientation)")
print()
header = "       " + "  ".join(f"  m={m:+d}" for m in m_vals)
print(header)
print("-" * len(header))
for k in k_vals:
    row = f"  k={k:+d}  "
    for m in m_vals:
        n = hs_norm(L_ops[(k, m)])
        row += f" {n:7.4f}"
    print(row)

# ---------------------------------------------------------------------------
# Total dissipator norm per k  (sum over m)
# ---------------------------------------------------------------------------
print()
print("Sum ||L_{k,m}||_HS^2 over m  (total dissipation power per spin-IST component k):")
for k in k_vals:
    total = sum(hs_norm(L_ops[(k, m)])**2 for m in m_vals)
    print(f"  k={k:+d}:  {total:.6e}  rad/s")

# ---------------------------------------------------------------------------
# Dominant operators — top by norm
# ---------------------------------------------------------------------------
norms_flat = {(k, m): hs_norm(L_ops[(k, m)]) for k in k_vals for m in m_vals}
ranked = sorted(norms_flat.items(), key=lambda x: x[1], reverse=True)

print()
print("Top-5 channels by ||L_{k,m}||_HS:")
for (k, m), n in ranked[:5]:
    print(f"  (k={k:+d}, m={m:+d})  ||L||_HS = {n:.6e}  "
          f"rate (Hz) = {n**2 / (2*np.pi):.4e}")

print("\nDone.")
