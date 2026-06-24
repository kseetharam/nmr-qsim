"""
Compare canonical Lindblad operators from the numerical Redfield approach
(saved by zulf_lindblad.py) against the analytical jump operators from
the Universal Linbladian formalism (notes: liouville_hilbert_basis.tex,
section "Universal Linbladian formalism").

Analytical formula
------------------
For each orientation component m in {-2,-1,0,+1,+2} the noise covariance
matrix over spin pairs mu, nu is

    N^(2,m)_{mu,nu} = (b_mu * a^(mu)_{2,m}) * (b_nu * a^(nu)_{2,m})^*

where b_mu = -(mu0/4pi) * gamma_i * gamma_j * hbar / r^3 is the dipolar
coupling constant for pair mu, and a^(mu)_{2,m} is the orientational
factor.  N^(2,m) is rank-1 (outer product), so it has a single non-zero
eigenvalue

    lambda_m = sum_mu |b_mu a^(mu)_{2,m}|^2

and eigenvector  v_m(mu) = b_mu a^(mu)_{2,m} / sqrt(lambda_m).

The analytical jump operator is

    L^{an}(m) = sqrt(tau_c / 5) * sum_mu b_mu a^(mu)_{2,m} T^{2,m}_mu
              = sqrt(tau_c / 5) * Q_{m,m}

where Q_{m,m} are the "diagonal" system-bath coupling operators
(IST component k == orientation index m) already computed in zulf_lindblad.py
and saved as Q_diag.

Comparison strategy
-------------------
The two sets of operators live in the same 8x8 Hilbert space.  The canonical
operators L_ops from the numerical approach are 15 operators (one per IST
basis mode), while the analytical formula produces 5 operators (one per m).
We compare:

  1. Subspace overlap: col-space of analytical operators vs dominant canonical
     operators (largest d_vals).
  2. Direct reconstruction check: does sqrt(tau_c/5) * Q_{m,m} match any of
     the canonical L_k up to a global phase?
  3. Frobenius distance between the analytical operators and their best
     approximation in the span of the canonical L_ops.

Run after executing zulf_lindblad.py which creates the data file.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 1. Locate and load saved data
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

import glob
candidates = sorted(glob.glob(os.path.join(DATA_DIR, 'canonical_L_ops_tc*.npz')))
if not candidates:
    sys.exit(
        f"No canonical_L_ops_*.npz found in {DATA_DIR}.\n"
        "Run zulf_lindblad.py first."
    )

data_path = candidates[-1]   # use most recent if several exist
print(f"Loading {data_path}")
data = np.load(data_path)

d_vals    = data['d_vals']     # (15,)  eigenvalues of Delta_plus [rad/s]
L_ops     = data['L_ops']      # (15, 8, 8)  canonical Lindblad operators
P_norm    = data['P_norm']     # (15, 8, 8)  normalised IST basis operators
Q_diag    = data['Q_diag']     # (5,  8, 8)  Q_{m,m} for m = -2..+2 [rad/s]
tau_c     = float(data['tau_c'])
B0        = float(data['B0'])
b_FC      = float(data['b_FC'])
b_FH      = float(data['b_FH'])
b_CH      = float(data['b_CH'])
r_FC      = data['r_FC']
r_FH      = data['r_FH']
r_CH      = data['r_CH']
gamma_F   = float(data['gamma_F'])
gamma_C   = float(data['gamma_C'])
gamma_H   = float(data['gamma_H'])

n_IST  = len(d_vals)   # 15
n_can  = 5             # number of Q_{m,m} analytical operators

print(f"  tau_c = {tau_c:.1e} s,  B0 = {B0:.2f} T")
print(f"  {n_IST} canonical operators loaded, {n_can} analytical operators (Q_{{m,m}})")

# ---------------------------------------------------------------------------
# 2. Build analytical jump operators from Universal Linbladian formalism
# ---------------------------------------------------------------------------
# L^an(m) = sqrt(tau_c / 5) * Q_{m,m}
L_analytical = np.sqrt(tau_c / 5.0) * Q_diag  # (5, 8, 8)

# Hilbert-Schmidt norms of analytical operators
hs_norms_an = np.array([
    np.sqrt(np.real(np.trace(L.conj().T @ L)))
    for L in L_analytical
])

print()
print("=" * 60)
print("Analytical operators  L^an(m) = sqrt(tau_c/5) * Q_{m,m}")
print("=" * 60)
m_labels = list(range(-2, 3))
for idx, m in enumerate(m_labels):
    print(f"  m={m:+d}:  ||L^an||_HS = {hs_norms_an[idx]:.6f} [sqrt(rad/s)]")

# ---------------------------------------------------------------------------
# 3. Noise covariance matrix N^(2,m) and its diagonalisation
# ---------------------------------------------------------------------------
# Physical constants (consistent with zulf_lindblad.py)
hbar = 1.054571817e-34
mu0  = 1.25663706212e-6
ANGSTROM = 1e-10

PAIRS = [
    ('19F-13C', b_FC, r_FC, gamma_F, gamma_C),
    ('19F-1H',  b_FH, r_FH, gamma_F, gamma_H),
    ('13C-1H',  b_CH, r_CH, gamma_C, gamma_H),
]

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

alm = {label: a2m_factors(r) for label, _, r, _, _ in PAIRS}
b_vals = {label: b for label, b, *_ in PAIRS}
pair_labels = [p[0] for p in PAIRS]

print()
print("=" * 60)
print("Noise covariance matrix N^(2,m) [3x3 over pairs]")
print("=" * 60)

for m in m_labels:
    v = np.array([b_vals[lbl] * alm[lbl][m] for lbl in pair_labels])  # (3,) complex
    N = np.outer(v, v.conj())   # (3,3) rank-1 Hermitian PSD
    eigvals = np.linalg.eigvalsh(N)
    print(f"  m={m:+d}:  eigenvalues = {eigvals}  (rank-1 -> two zeros expected)")

# ---------------------------------------------------------------------------
# 4. Comparison: Frobenius distance between analytical and canonical
# ---------------------------------------------------------------------------
# Vectorise each operator as a flat complex vector (Hilbert-Schmidt inner product)
def hs_ip(A, B):
    """Hilbert-Schmidt inner product Tr(A^dag B)."""
    return np.trace(A.conj().T @ B)

def project_onto_span(A, basis_ops):
    """
    Project operator A onto the span of basis_ops using HS inner products.
    Returns (residual_norm, best_approx).
    """
    # Gram matrix
    n = len(basis_ops)
    G = np.array([[hs_ip(basis_ops[i], basis_ops[j]) for j in range(n)]
                  for i in range(n)], dtype=complex)
    b_vec = np.array([hs_ip(basis_ops[i], A) for i in range(n)], dtype=complex)
    # Least-squares projection (G may be rank-deficient)
    coeffs, _, _, _ = np.linalg.lstsq(G, b_vec, rcond=None)
    A_proj = sum(c * op for c, op in zip(coeffs, basis_ops))
    residual = A - A_proj
    res_norm = np.sqrt(max(0.0, np.real(hs_ip(residual, residual))))
    return res_norm, A_proj

print()
print("=" * 60)
print("Projection of L^an(m) onto span{L_ops} (all 15 canonical)")
print("=" * 60)
print(f"  {'m':>4}  {'||L^an||_HS':>13}  {'residual':>13}  {'rel_err':>10}")
print("  " + "-" * 50)

for idx, m in enumerate(m_labels):
    L_an = L_analytical[idx]
    an_norm = hs_norms_an[idx]
    if an_norm < 1e-30:
        print(f"  m={m:+d}   (zero operator, skipped)")
        continue
    res, _ = project_onto_span(L_an, [L_ops[k] for k in range(n_IST)])
    rel    = res / an_norm
    print(f"  m={m:+d}  {an_norm:13.4e}  {res:13.4e}  {rel:10.4e}")

# ---------------------------------------------------------------------------
# 5. Dominant canonical operators: overlap with analytical subspace
# ---------------------------------------------------------------------------
# Identify significant canonical operators (d_k > threshold)
thresh     = np.abs(d_vals).max() * 1e-10
sig_idx    = np.where(d_vals > thresh)[0]
dom5_idx   = np.argsort(d_vals)[::-1][:5]   # top-5 by eigenvalue

print()
print("=" * 60)
print("Projection of dominant canonical L_k onto span{L^an}")
print("=" * 60)
print(f"  Using {n_can} analytical operators as basis.")
print(f"  {'k':>4}  {'d_k [rad/s]':>14}  {'||L_k||_HS':>12}  "
      f"{'residual':>12}  {'rel_err':>10}")
print("  " + "-" * 60)

for k in dom5_idx:
    Lk      = L_ops[k]
    dk      = d_vals[k]
    lk_norm = np.sqrt(max(0.0, np.real(hs_ip(Lk, Lk))))
    if lk_norm < 1e-30:
        continue
    res, _ = project_onto_span(Lk, list(L_analytical))
    rel    = res / lk_norm
    print(f"  k={k:2d}  {dk:14.4e}  {lk_norm:12.4e}  {res:12.4e}  {rel:10.4e}")

# ---------------------------------------------------------------------------
# 6. Pairwise HS overlaps: |<L^an(m) | L_k>|^2 / (||L^an||^2 ||L_k||^2)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("HS overlaps |<L^an(m)|L_k>|^2 normalised")
print("Rows = m in {-2,-1,0,+1,+2},  Cols = top-5 canonical L_k")
print("=" * 60)

header_cols = "  ".join(f"L_{k:02d}" for k in dom5_idx)
print(f"  {'m':>4}  {header_cols}")
print("  " + "-" * (6 + 8 * len(dom5_idx)))

for idx, m in enumerate(m_labels):
    L_an   = L_analytical[idx]
    an_n2  = np.real(hs_ip(L_an, L_an))
    if an_n2 < 1e-60:
        print(f"  m={m:+d}  (zero)")
        continue
    overlaps = []
    for k in dom5_idx:
        Lk   = L_ops[k]
        lk_n2 = np.real(hs_ip(Lk, Lk))
        if lk_n2 < 1e-60:
            overlaps.append(0.0)
            continue
        ip2  = abs(hs_ip(L_an, Lk))**2
        overlaps.append(ip2 / (an_n2 * lk_n2))
    row = "  ".join(f"{v:6.3f}" for v in overlaps)
    print(f"  m={m:+d}  {row}")

print()
print("Done.")
