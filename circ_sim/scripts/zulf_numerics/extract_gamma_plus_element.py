"""
Extract the Gamma_plus[i, i] diagonal element corresponding to the
ΔM = +3 transition  sigma = |3/2,+3/2><3/2,-3/2|.

Index convention (from zulf_lindblad.py):
    transition index  i = n * n_states + m
    where sigma_i = |n><m|  (eigenstate n → eigenstate m)

For the ΔM = +3 transition:
    n  = eigenstate index of |3/2, +3/2>   (source / bra)
    m  = eigenstate index of |3/2, -3/2>   (sink  / ket)
    i  = n * 8 + m
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1. Make the zulf_numerics directory importable and bring in the module.
#    The module-level code (sections 1-10) runs on import; only the
#    if __name__ == '__main__' block is skipped.
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad (this runs sections 1-10 including Gamma_plus)...")
import zulf_lindblad as zl
print("Import complete.\n")

# ---------------------------------------------------------------------------
# 2. Identify eigenstate indices for |3/2, +3/2> and |3/2, -3/2>
# ---------------------------------------------------------------------------
n_states = zl.n_states   # should be 8

# Find eigenstate index where S = 3/2, M = +3/2
idx_p32 = next(
    n for n in range(n_states)
    if abs(zl._S_qn[n] - 1.5) < 0.1 and abs(zl._M_qn[n] - 1.5) < 0.1
)

# Find eigenstate index where S = 3/2, M = -3/2
idx_m32 = next(
    n for n in range(n_states)
    if abs(zl._S_qn[n] - 1.5) < 0.1 and abs(zl._M_qn[n] + 1.5) < 0.1
)

print(f"Eigenstate index of |3/2, +3/2>  :  idx_p32 = {idx_p32}")
print(f"  S = {zl._S_qn[idx_p32]},  M = {zl._M_qn[idx_p32]}")
print(f"  Energy = {zl.evals[idx_p32]/(2*np.pi):+.6f} Hz")
print()
print(f"Eigenstate index of |3/2, -3/2>  :  idx_m32 = {idx_m32}")
print(f"  S = {zl._S_qn[idx_m32]},  M = {zl._M_qn[idx_m32]}")
print(f"  Energy = {zl.evals[idx_m32]/(2*np.pi):+.6f} Hz")
print()

# ---------------------------------------------------------------------------
# 3. Build the flat transition index
#    sigma_i = |n><m|  →  i = n * n_states + m
#    Here n = idx_p32  (source / bra),  m = idx_m32  (sink / ket)
# ---------------------------------------------------------------------------
i = idx_p32 * n_states + idx_m32

print(f"Flat transition index  i = idx_p32 * {n_states} + idx_m32 = {idx_p32} * {n_states} + {idx_m32} = {i}")
print(f"  sigma_i = |{idx_p32}><{idx_m32}| = |3/2,+3/2><3/2,-3/2|  (ΔM = +3 transition)")
print()

# ---------------------------------------------------------------------------
# 4. Extract Gamma_plus[i, i]
# ---------------------------------------------------------------------------
gp_val = zl.Gamma_plus[i, i]
gp_re  = np.real(gp_val)
gp_im  = np.imag(gp_val)
gp_abs = abs(gp_val)

print("=" * 60)
print("Gamma_plus[i, i]  for the ΔM = +3 transition")
print("=" * 60)
print(f"  Real part         : {gp_re:.10e}  rad/s")
print(f"  Imaginary part    : {gp_im:.10e}  rad/s")
print(f"  |Gamma_plus[i,i]| : {gp_abs:.10e}  rad/s")
print(f"  Real part / 2pi   : {gp_re / (2*np.pi):.10e}  Hz")
print(f"  Imag part / 2pi   : {gp_im / (2*np.pi):.10e}  Hz")
print(f"  |value|   / 2pi   : {gp_abs / (2*np.pi):.10e}  Hz")
print()

# ---------------------------------------------------------------------------
# 5. Transition frequency for index i (verification)
#    omega_trans[i] = E_m - E_n = E(idx_m32) - E(idx_p32)
# ---------------------------------------------------------------------------
omega_i = zl.omega_trans[i]
print("=" * 60)
print("Transition frequency  omega_trans[i]  for index i")
print("=" * 60)
print(f"  omega_trans[i]        : {omega_i:.10e}  rad/s")
print(f"  omega_trans[i] / 2pi  : {omega_i / (2*np.pi):.10e}  Hz")
print(f"  (= E(idx_m32) - E(idx_p32) = E(|3/2,-3/2>) - E(|3/2,+3/2>))")
print()

# ---------------------------------------------------------------------------
# 6. Cross-check: print the full row/column of Gamma_plus at index i
#    to see off-diagonal structure
# ---------------------------------------------------------------------------
print("=" * 60)
print("Cross-check: diagonal of the n=idx_p32 block in Gamma_plus")
print(f"  Block rows/cols: {idx_p32*n_states} .. {idx_p32*n_states + n_states - 1}")
print("=" * 60)
block_start = idx_p32 * n_states
for k in range(n_states):
    row = block_start + k
    val = zl.Gamma_plus[row, row]
    print(f"  Gamma_plus[{row:2d},{row:2d}]  (n={idx_p32}, m={k})  = "
          f"{np.real(val):+.6e} + {np.imag(val):+.6e}j  rad/s  "
          f"  ({np.real(val)/(2*np.pi):+.6e} Hz)")

print()

# ===========================================================================
# PART 2: Additional Gamma_plus elements requested for FGR comparison
# ===========================================================================

# --- Identify remaining eigenstate indices -----------------------------------

# S=3/2, M=+1/2  (within the quartet)
idx_q12 = next(
    n for n in range(n_states)
    if abs(zl._S_qn[n] - 1.5) < 0.1 and abs(zl._M_qn[n] - 0.5) < 0.1
)

# Both S=1/2, M=+1/2 states (two doublets)
half_p12_idxs = [
    n for n in range(n_states)
    if abs(zl._S_qn[n] - 0.5) < 0.1 and abs(zl._M_qn[n] - 0.5) < 0.1
]

# Doublet state closer in energy to the quartet
E_quartet = zl.evals[idx_p32]
idx_d12_close = min(half_p12_idxs, key=lambda n: abs(zl.evals[n] - E_quartet))
idx_d12_far   = max(half_p12_idxs, key=lambda n: abs(zl.evals[n] - E_quartet))

print("=" * 65)
print("Eigenstate inventory for Cases 1 & 2")
print("=" * 65)
print(f"  |3/2, +1/2>  (quartet M=+1/2) : index {idx_q12}, "
      f"E = {zl.evals[idx_q12]/(2*np.pi):+.4f} Hz")
print(f"  |1/2, +1/2>  (doublet, close) : index {idx_d12_close}, "
      f"E = {zl.evals[idx_d12_close]/(2*np.pi):+.4f} Hz  "
      f"(dE = {abs(zl.evals[idx_d12_close]-E_quartet)/(2*np.pi):.4f} Hz from quartet)")
print(f"  |1/2, +1/2>  (doublet, far)   : index {idx_d12_far}, "
      f"E = {zl.evals[idx_d12_far]/(2*np.pi):+.4f} Hz  "
      f"(dE = {abs(zl.evals[idx_d12_far]-E_quartet)/(2*np.pi):.4f} Hz from quartet)")
print()

# ---------------------------------------------------------------------------
# CASE 1
#   sigma_j   = |3/2,+1/2><3/2,+3/2|  →  n_j = idx_q12,  m_j = idx_p32
#   sigma_i†  = |3/2,+3/2><3/2,+1/2|  =  (sigma_j)†  → DIAGONAL element
# ---------------------------------------------------------------------------
j_c1 = idx_q12 * n_states + idx_p32
i_c1 = j_c1   # diagonal

gp_c1 = zl.Gamma_plus[i_c1, j_c1]
C_c1  = zl.C_matrix[i_c1, j_c1]
o_c1  = zl.omega_trans[i_c1] / (2*np.pi)

print("=" * 65)
print("CASE 1")
print("  sigma_j   = |3/2,+1/2><3/2,+3/2|  (ΔM = -1, intra-quartet)")
print("  sigma_i†  = |3/2,+3/2><3/2,+1/2|  = sigma_j†  → DIAGONAL element")
print(f"  i = j = idx_q12*8 + idx_p32 = {idx_q12}*8 + {idx_p32} = {i_c1}")
print(f"  omega_trans[i] / 2pi = {o_c1:+.4f} Hz")
print(f"  C_matrix[i,i]        = {np.real(C_c1):.6e} + {np.imag(C_c1):.6e}j  (rad/s)²")
print(f"  Gamma_plus[i,i]      = {np.real(gp_c1):.6e} + {np.imag(gp_c1):.6e}j  rad/s")
print(f"  |Gamma_plus[i,i]|    = {abs(gp_c1):.6e} rad/s  "
      f"({abs(gp_c1)/(2*np.pi):.6e} Hz)")
print()

# FGR rate for the same transition: |3/2,+3/2> -> |3/2,+1/2>
fgr_c1 = zl.fgr_rates[idx_q12, idx_p32]
print(f"  FGR rate |3/2,+3/2>→|3/2,+1/2>: {fgr_c1/(2*np.pi):.6e} Hz")
print(f"  Ratio Gamma_plus[i,i] / fgr_rate = "
      f"{abs(gp_c1)/fgr_c1:.6f}  (expect 1.000)")
print()

# ---------------------------------------------------------------------------
# CASE 2
#   sigma_j   = |1/2,+1/2><3/2,+3/2|  →  n_j = idx_d12_close, m_j = idx_p32
#   sigma_i†  = |3/2,+3/2><1/2,+1/2|  →  sigma_i = sigma_j  → DIAGONAL
# ---------------------------------------------------------------------------
j_c2 = idx_d12_close * n_states + idx_p32
i_c2 = j_c2   # diagonal

gp_c2 = zl.Gamma_plus[i_c2, j_c2]
C_c2  = zl.C_matrix[i_c2, j_c2]
o_c2  = zl.omega_trans[i_c2] / (2*np.pi)

print("=" * 65)
print("CASE 2")
print("  sigma_j   = |1/2,+1/2><3/2,+3/2|  (inter-manifold, ΔM = -1)")
print("  sigma_i†  = |3/2,+3/2><1/2,+1/2|  = sigma_j†  → DIAGONAL element")
print(f"  i = j = idx_d12_close*8 + idx_p32 = {idx_d12_close}*8 + {idx_p32} = {i_c2}")
print(f"  omega_trans[i] / 2pi = {o_c2:+.4f} Hz")
print(f"  C_matrix[i,i]        = {np.real(C_c2):.6e} + {np.imag(C_c2):.6e}j  (rad/s)²")
print(f"  Gamma_plus[i,i]      = {np.real(gp_c2):.6e} + {np.imag(gp_c2):.6e}j  rad/s")
print(f"  |Gamma_plus[i,i]|    = {abs(gp_c2):.6e} rad/s  "
      f"({abs(gp_c2)/(2*np.pi):.6e} Hz)")
print()

# FGR rate for the same transition: |3/2,+3/2> -> |1/2,+1/2> (close doublet)
fgr_c2 = zl.fgr_rates[idx_d12_close, idx_p32]
print(f"  FGR rate |3/2,+3/2>→|1/2,+1/2(close)|: {fgr_c2/(2*np.pi):.6e} Hz")
print(f"  Ratio Gamma_plus[i,i] / fgr_rate = "
      f"{abs(gp_c2)/fgr_c2:.6f}  (expect 1.000)")
print()

print("Done.")
