"""
verify_R_diagonal.py

Computes diagonal matrix elements of the Redfield relaxation superoperator
    R_kk = Tr{ o_k†  ·  R[o_k] }
for the three single-spin operators on spin 3 (1H):
    o = Sz_3 = E ⊗ E ⊗ Lz
    o = S+_3 = E ⊗ E ⊗ L+
    o = S-_3 = E ⊗ E ⊗ L-

each normalised to unit Hilbert–Schmidt norm: ô = o / sqrt(Tr(o†o)).

The relaxation channel is evaluated using the FIRST LINE of the double-
commutator Redfield equation (sec. 9.2 of the notes):

    R[ρ] = -∑_{i,j} Γ_{ij}(ω_j) [σ_i, [σ_j†, ρ]]

with:
    σ_i      = |n_i⟩⟨m_i|         (eigenstate transition operator)
    σ_j†     = |m_j⟩⟨n_j|
    Γ_{ij}(ω_j) = C[i,j] · J(ω_j)  (structure matrix × spectral density)
    C[i,j]   = ∑_α M[α,i]* M[α,j]  (already computed in zulf_lindblad.py)
    J(ω_j)   = 0.2 τ_c / (1 + ω_j² τ_c²)  (Lorentzian spectral density)

The double commutator expands as:
    [σ_i, [σ_j†, o]] = σ_i σ_j† o  -  σ_i o σ_j†
                      - σ_j† o σ_i  +  o σ_j† σ_i

so that:
    Tr{o† [σ_i,[σ_j†,o]]} = Tr{o† σ_i σ_j† o}   (T1)
                           - Tr{o† σ_i o σ_j†}    (T2)
                           - Tr{o† σ_j† o σ_i}    (T3)
                           + Tr{o† o σ_j† σ_i}    (T4)

All four terms are computed via vectorised einsum operations.
"""

import sys
import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad ...")
import zulf_lindblad as zl
print("Import complete.\n")

# =============================================================================
# Transition operators σ_j = |n_j⟩⟨m_j|  and  σ_j† = |m_j⟩⟨n_j|
# indexed as  j = n_j * n_states + m_j
# =============================================================================
n_s = zl.n_states   # 8
n_t = n_s ** 2      # 64  (transition operators)

sigma    = np.zeros((n_t, n_s, n_s), dtype=complex)
sigma_dg = np.zeros((n_t, n_s, n_s), dtype=complex)

for n_ in range(n_s):
    for m_ in range(n_s):
        j = n_ * n_s + m_
        kn = zl.ekets_arr[n_]
        km = zl.ekets_arr[m_]
        sigma[j]    = np.outer(kn, km.conj())   # |n><m|
        sigma_dg[j] = np.outer(km, kn.conj())   # |m><n|

# =============================================================================
# Γ_{ij}(ω_j) = C[i,j] · J(ω_j)     shape (64, 64)
# (first-line Redfield Gamma: spectral density evaluated at column freq ω_j)
# =============================================================================
Gamma_raw = zl.C_matrix * zl.J_omegas[np.newaxis, :]   # broadcast J over rows i

# =============================================================================
# Define probe operators on spin 3 (1H), normalize to unit H-S norm
# =============================================================================
ops_raw = {
    'Sz_3 = E⊗E⊗Lz': zl.Iz[2].full(),
    'S+_3 = E⊗E⊗L+': zl.Ip[2].full(),
    'S-_3 = E⊗E⊗L-': zl.Im[2].full(),
}

print("Hilbert–Schmidt norms  ||o||_HS = sqrt(Tr(o†o)):")
ops_norm = {}
hs_norms = {}
for name, o in ops_raw.items():
    hs2  = np.real(np.trace(o.conj().T @ o))
    norm = np.sqrt(hs2)
    hs_norms[name] = norm
    tag  = '→ normalised' if abs(norm - 1.0) > 1e-10 else '(unit norm)'
    print(f"  {name}:  ||o||_HS = {norm:.6f}   {tag}")
    ops_norm[name] = o / norm

print()

# =============================================================================
# Core computation: Tr{ ô† · R[ô] }
# =============================================================================
# Precompute eigenstate energies and labels for reporting
evals_hz = zl.evals / (2 * np.pi)

def _compute_R_diag(o):
    """Return Tr{ o†  R[o] } using the first-line Redfield formula."""
    o_dg = o.conj().T

    # Precompute intermediate 3-index tensors  (n_t, 8, 8):
    #   P[i] = o† σ_i
    #   Q[j] = σ_j† o
    P = np.einsum('ab,ibc->iac', o_dg, sigma)       # o† σ_i
    Q = np.einsum('iab,bc->iac', sigma_dg, o)       # σ_j† o

    # T1 = Tr{ o† σ_i  σ_j† o }  =  Tr{ P[i] Q[j] }
    T1 = np.einsum('iac,jca->ij', P, Q)

    # T2 = Tr{ o† σ_i  o  σ_j† }
    Po = np.einsum('iac,cb->iab', P, o)             # P[i] @ o = o† σ_i o
    T2 = np.einsum('iab,jba->ij', Po, sigma_dg)

    # T3 = Tr{ o† σ_j†  o  σ_i }
    P2  = np.einsum('ab,ibc->iac', o_dg, sigma_dg)  # o† σ_j†
    P2o = np.einsum('iac,cb->iab', P2, o)            # o† σ_j† o
    T3  = np.einsum('iab,jba->ij', P2o, sigma)

    # T4 = Tr{ o†o  σ_j† σ_i }
    oo  = o_dg @ o                                   # 8×8
    U   = np.einsum('ab,ibc->iac', oo, sigma_dg)    # o†o σ_j†
    T4  = np.einsum('iac,jca->ij', U, sigma)

    inner = T1 - T2 - T3 + T4   # Tr{ o† [σ_i, [σ_j†, o]] }  shape (64,64)
    return -np.sum(Gamma_raw * inner)


print("=" * 65)
print("Diagonal relaxation matrix elements  Tr{ ô† · R[ô] }")
print("R[ô] = -∑_{i,j} Γ_{ij}(ω_j) [σ_i, [σ_j†, ô]]")
print(f"(τ_c = {zl.tau_c:.1e} s,  B0 = {zl.B0:.2f} T)")
print("=" * 65)

results = {}
for name, o in ops_norm.items():
    val = _compute_R_diag(o)
    results[name] = val

    re_val = np.real(val)
    im_val = np.imag(val)
    re_hz  = re_val / (2 * np.pi)
    im_hz  = im_val / (2 * np.pi)

    print(f"\n  Operator : {name}")
    print(f"  Tr{{ô† R[ô]}}   (rad/s) : {re_val:+.8e}  +  {im_val:+.8e}j")
    print(f"  Re / (2π)  (Hz)      : {re_hz:+.8e}")
    if abs(im_val) > 1e-8 * abs(re_val) + 1e-15:
        print(f"  Im / (2π)  (Hz)      : {im_hz:+.8e}   *** non-negligible imaginary part ***")

# =============================================================================
# Cross-check: compare with diagonal entries of R_L saved by compute_R_lindblad.py
# Operators ExExLz, ExExL+, ExExL- correspond to k=1,2,3 (0-based) in R_L.
# R_L was computed with the full dissipator L_diss acting on the UNNORMALISED
# operators, so R_L[k,k] = Tr{ o_k†  L_diss[o_k] }  for the raw (unnorm.) o_k.
# To compare: R_L[k,k] / Tr(o_k† o_k)  should equal Tr{ ô† R[ô] } computed above.
# =============================================================================
rl_path = os.path.join(_DIR, 'data', 'R_L_lindblad_gamma1.npy')
if os.path.isfile(rl_path):
    R_L = np.load(rl_path)

    # Index mapping: k = a*16 + b*4 + c  (0-based), E=0,Lz=1,L+=2,L-=3
    # ExExLz: a=0,b=0,c=1 -> k=1
    # ExExL+: a=0,b=0,c=2 -> k=2
    # ExExL-: a=0,b=0,c=3 -> k=3
    rl_indices = {'Sz_3 = E⊗E⊗Lz': 1,
                  'S+_3 = E⊗E⊗L+': 2,
                  'S-_3 = E⊗E⊗L-': 3}

    print("\n" + "=" * 65)
    print("Cross-check against R_L from compute_R_lindblad.py")
    print("  R_L[k,k] / Tr(o†o)  should match  Tr{ ô† R[ô] }")
    print("=" * 65)
    print(f"  {'Operator':<20}  {'R_L[k,k]/norm (Hz)':>20}  "
          f"{'R[ô] here (Hz)':>18}  {'ratio':>8}")
    print("  " + "-" * 72)
    for name, k in rl_indices.items():
        rl_norm = np.real(R_L[k, k]) / (hs_norms[name] ** 2) / (2 * np.pi)
        r_here  = np.real(results[name]) / (2 * np.pi)
        ratio   = rl_norm / r_here if abs(r_here) > 1e-20 else float('nan')
        print(f"  {name:<20}  {rl_norm:>20.6e}  {r_here:>18.6e}  {ratio:>8.4f}")
else:
    print(f"\n  (R_L file not found at {rl_path}; run compute_R_lindblad.py first.)")

print()
