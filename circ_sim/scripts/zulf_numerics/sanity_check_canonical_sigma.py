"""
Sanity check: canonical jump operators in the sigma_i basis.

Compares the canonical operators obtained by diagonalising Gamma_plus (64x64)
with the analytically expected operators from the Universal Linbladian formalism:

    L^ana_m[i] = sqrt(J(omega_i)) * M[(m,m), i]     (alpha = (m,m), m in {-2,...,+2})

where M[(k,m), i] = <E_{n_i}|Q_{k,m}|E_{m_i}> and sigma_i = |E_{n_i}><E_{m_i}|.

Three-level check
-----------------
(1) Block structure: verify Gamma_plus is exactly zero between transitions
    with different Delta_M = M_{n_i} - M_{m_i}.  This follows from
    Wigner-Eckart: Q_{k,m} connects states differing by Delta_M = k, so
    M[(k,m), i] != 0 only if Delta_M_i = k, making Gamma_plus block-diagonal.

(2) Eigenvector test: within each Delta_M = m block, check whether the
    analytical vector v^ana_m (restricted to that block) is an eigenvector
    of the block.  This holds if and only if the diagonal bath mode (k=m,
    m_alpha=m) dominates over off-diagonal orientation modes (k=m, m_alpha!=m).

(3) Overlap decomposition: for each analytical v^ana_m, decompose its squared
    norm over the canonical eigenvectors of the block.  An overlap of 1.0 on
    a single eigenvector means perfect agreement.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Import zulf_lindblad (runs sections 1-10, computes Gamma_plus etc.)
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad ...")
import zulf_lindblad as zl
print("Done.\n")

n_states = zl.n_states   # 8
n_trans  = zl.n_trans    # 64

# ---------------------------------------------------------------------------
# Delta_M for every transition i = n*n_states + m
# sigma_i = |E_n><E_m|  =>  Delta_M_i = M_tot(n) - M_tot(m)
# ---------------------------------------------------------------------------
delta_M_arr = np.array([
    zl._M_qn[n_] - zl._M_qn[m_]
    for n_ in range(n_states)
    for m_ in range(n_states)
])   # (64,), values in {-3,-2,-1,0,+1,+2,+3}

dm_vals_all = np.unique(np.round(delta_M_arr, 6))
print("Distinct Delta_M values in transition set:", dm_vals_all)
print()

# ---------------------------------------------------------------------------
# CHECK 1: Block-diagonal structure of Gamma_plus
# ---------------------------------------------------------------------------
print("=" * 65)
print("CHECK 1 — Block-diagonal structure of Gamma_plus")
print("=" * 65)

abs_Gp    = np.abs(zl.Gamma_plus)
max_Gp    = abs_Gp.max()
tol_block = max_Gp * 1e-12

n_violations = 0
max_offblock  = 0.0

for i in range(n_trans):
    for j in range(n_trans):
        if abs(delta_M_arr[i] - delta_M_arr[j]) > 0.5:   # different Delta_M
            v = abs_Gp[i, j]
            if v > tol_block:
                n_violations += 1
                if v > max_offblock:
                    max_offblock = v

print(f"  max |Gamma_plus|                        : {max_Gp:.4e} rad/s")
print(f"  tolerance (1e-12 * max)                 : {tol_block:.4e} rad/s")
print(f"  off-block violations                    : {n_violations}")
print(f"  max off-block |Gamma_plus[i,j]|         : {max_offblock:.4e} rad/s")
if n_violations == 0:
    print("  --> PASS: Gamma_plus is exactly block-diagonal in Delta_M")
else:
    print("  --> FAIL: unexpected off-block entries detected")
print()

# Delta_M = +-3 transitions: T^{2,k} has |k|<=2, so these should be zero
dm3_mask = np.abs(np.round(delta_M_arr)) == 3
dm3_max  = abs_Gp[np.ix_(dm3_mask, dm3_mask)].max() if dm3_mask.any() else 0.0
print(f"  Max |Gamma_plus| in Delta_M=+-3 block   : {dm3_max:.4e} rad/s")
print(f"  (rank-2 IST cannot drive |Delta_M|=3 transitions -> expected zero)")
print()

# ---------------------------------------------------------------------------
# Analytical operators: v^ana_m[i] = sqrt(J(omega_i)) * M[(m,m), i]
# Non-zero only on the Delta_M = m block by Wigner-Eckart.
# ---------------------------------------------------------------------------
sqrt_J = np.sqrt(zl.J_omegas)   # (64,)

L_ana = {}   # m -> (64,) complex vector
for m in range(-2, 3):
    alpha_idx = zl.KM_LIST.index((m, m))
    L_ana[m] = sqrt_J * zl.M_matrix[alpha_idx, :]

# ---------------------------------------------------------------------------
# CHECK 2 & 3: Per-block eigenvector test and overlap decomposition
# ---------------------------------------------------------------------------
print("=" * 65)
print("CHECK 2 & 3 — Block diagonalisation and eigenvector test")
print("=" * 65)
print()
print("For each Delta_M = m block:")
print("  * Diagonalise Gamma_plus|_block  (eigh -> real eigenvalues d_k)")
print("  * v^ana_m  restricted to block = sqrt(J_i) * M[(m,m), i]")
print("  * Test: is v^ana_m an eigenvector of the block?")
print("  * Overlap^2 = |<v_k | v^ana>|^2 / ||v^ana||^2  summed -> 1 if in eigenspace")
print()

for m in range(-2, 3):
    mask_m  = np.abs(np.round(delta_M_arr) - m) < 0.5
    blk_idx = np.where(mask_m)[0]
    n_blk   = len(blk_idx)

    Gp_blk = zl.Gamma_plus[np.ix_(blk_idx, blk_idx)]

    # Confirm the block is Hermitian (it should be)
    herm_err = np.linalg.norm(Gp_blk - Gp_blk.conj().T) / (np.linalg.norm(Gp_blk) + 1e-300)

    d_blk, V_blk = np.linalg.eigh(Gp_blk)   # ascending; V_blk columns = eigenvectors

    # Analytical vector restricted to this block
    v_ana_full = L_ana[m]
    v_ana      = v_ana_full[blk_idx]          # (n_blk,) complex
    v_norm     = np.linalg.norm(v_ana)

    print(f"--- Delta_M = {m:+d}  ({n_blk} transitions) ---")
    print(f"  Hermitian error of block         : {herm_err:.2e}")
    print(f"  Eigenvalues d_k  (rad/s) :")
    for k, dk in enumerate(d_blk):
        print(f"      k={k:2d}:  {dk:+.6e}  ({dk/(2*np.pi):+.4e} Hz)")

    if v_norm < 1e-30:
        print("  v^ana is zero (no Q_{m,m} matrix elements at Delta_M=m) -- skipped")
        print()
        continue

    # Eigenvector test: compute Gamma_plus @ v^ana and compare with lambda*v^ana
    Gp_v   = Gp_blk @ v_ana
    lam    = np.real(np.dot(v_ana.conj(), Gp_v)) / v_norm**2  # Rayleigh quotient
    resid  = Gp_v - lam * v_ana
    rel_res = np.linalg.norm(resid) / (abs(lam) * v_norm + 1e-300)

    print(f"  ||v^ana||                         : {v_norm:.6e}")
    print(f"  Rayleigh quotient lambda          : {lam:.6e} rad/s  ({lam/(2*np.pi):.4e} Hz)")
    print(f"  Residual ||Gp v - lam v|| / |lam| ||v|| : {rel_res:.4e}  (0 = eigenvec)")

    # Squared overlaps with canonical eigenvectors
    if v_norm > 1e-30:
        v_hat     = v_ana / v_norm
        overlaps  = np.abs(V_blk.conj().T @ v_hat)**2   # (n_blk,)
        total_ov  = overlaps.sum()
        dom_k     = np.argmax(overlaps)
        print(f"  Overlap^2 decomposition (sum={total_ov:.6f}):")
        for k, (dk, ov) in enumerate(zip(d_blk, overlaps)):
            bar = "#" * int(50 * ov)
            tag = " <- dominant" if k == dom_k else ""
            print(f"      k={k:2d}  d_k={dk:+.3e}  ov={ov:.4f}  {bar}{tag}")
    print()

# ---------------------------------------------------------------------------
# SUMMARY: contributions from off-diagonal orientation modes (m_alpha != m)
# ---------------------------------------------------------------------------
print("=" * 65)
print("SUMMARY — Off-diagonal orientation mode contributions")
print("=" * 65)
print()
print("For each Delta_M=m block, Gamma_plus[block] = 0.5*(J_i+J_j) * C[block]")
print("  C[i,j] = sum_{m_alpha} M[(m,m_alpha),i]* M[(m,m_alpha),j]")
print()
print("The analytical v^ana_m uses only the m_alpha=m term.")
print("Contribution of each orientation mode m_alpha to ||v^ana_m||^2:")
print()

for m in range(-2, 3):
    mask_m  = np.abs(np.round(delta_M_arr) - m) < 0.5
    blk_idx = np.where(mask_m)[0]
    if len(blk_idx) == 0:
        continue

    # For each orientation mode m_alpha, build the vector sqrt(J)*M[(m,m_alpha),:]
    # and report its norm relative to the full sum (which gives the row of C)
    norms_sq = {}
    for m_alpha in range(-2, 3):
        alpha_idx = zl.KM_LIST.index((m, m_alpha))
        v_alpha   = sqrt_J[blk_idx] * zl.M_matrix[alpha_idx, blk_idx]
        norms_sq[m_alpha] = np.real(np.dot(v_alpha.conj(), v_alpha))

    total_norm_sq = sum(norms_sq.values())
    if total_norm_sq < 1e-60:
        continue

    print(f"  Delta_M = {m:+d} block:")
    for m_alpha in range(-2, 3):
        frac = norms_sq[m_alpha] / total_norm_sq
        tag  = " <- diagonal (analytical)" if m_alpha == m else ""
        print(f"    m_alpha={m_alpha:+d}:  ||v_{m},{m_alpha}||^2 = {norms_sq[m_alpha]:.4e}  "
              f"fraction = {frac:.4f}{tag}")
    print()

print("Done.")

# ===========================================================================
# FLAT-J COMPARISON
# Replace J(omega_i) -> J(0) = tau_c/5 uniformly, rebuild Gamma_plus, and
# re-run the eigenvector test.  In this limit the analytical operator becomes
#
#     v^ana_flat_m[i] = sqrt(J0) * M[(m,m), i]
#
# and Gamma_plus_flat = J0 * C_matrix (a pure Gram matrix scaled by J0).
# If J inhomogeneity is the only source of mismatch, the residuals here
# should vanish (exact eigenvectors); if they remain, the mismatch is
# structural (orientation-mode mixing), independent of spectral flatness.
# ===========================================================================

print()
print("=" * 65)
print("FLAT-J COMPARISON  [J(omega) -> J(0) = tau_c/5 for all transitions]")
print("=" * 65)

J0 = zl.J_spectral(0.0)   # tau_c / 5  [seconds]
print(f"\n  J(0) = tau_c/5 = {J0:.6e} s")
print(f"  max |J(omega_i) - J(0)| / J(0) over all transitions: "
      f"{np.max(np.abs(zl.J_omegas - J0)) / J0:.2e}  (flatness check)\n")

# Gamma_plus_flat = J0 * C_matrix   [from sJ_flat[i,j] = 2*J0 -> 0.5*2J0*C = J0*C]
Gamma_plus_flat = J0 * zl.C_matrix   # (64, 64)

# Analytical vectors in the flat-J case
sqrt_J0 = np.sqrt(J0)
L_ana_flat = {}
for m in range(-2, 3):
    alpha_idx = zl.KM_LIST.index((m, m))
    L_ana_flat[m] = sqrt_J0 * zl.M_matrix[alpha_idx, :]   # (64,) complex

# -----------------------------------------------------------------------
# Per-block eigenvector test with flat Gamma_plus
# -----------------------------------------------------------------------
print("Per-block eigenvector test  (Gamma_plus_flat, v^ana_flat_m)")
print()
print(f"  {'DM':>4}  {'block':>6}  {'lambda_Rayleigh':>18}  "
      f"{'residual':>12}  {'ov_dominant':>13}  {'n_sig_eigs':>10}")
print("  " + "-" * 72)

results_flat   = {}
results_nflat  = {}

for m in range(-2, 3):
    mask_m  = np.abs(np.round(delta_M_arr) - m) < 0.5
    blk_idx = np.where(mask_m)[0]
    n_blk   = len(blk_idx)

    # --- flat-J block ---
    Gf_blk = Gamma_plus_flat[np.ix_(blk_idx, blk_idx)]
    df, Vf = np.linalg.eigh(Gf_blk)

    v_flat = L_ana_flat[m][blk_idx]
    v_norm = np.linalg.norm(v_flat)

    if v_norm < 1e-30:
        print(f"  m={m:+d}  (zero analytical vector, skip)")
        continue

    Gfv     = Gf_blk @ v_flat
    lam_f   = np.real(np.dot(v_flat.conj(), Gfv)) / v_norm**2
    resid_f = np.linalg.norm(Gfv - lam_f * v_flat) / (abs(lam_f) * v_norm + 1e-300)
    ov_f    = np.abs(Vf.conj().T @ (v_flat / v_norm))**2
    ov_dom  = ov_f.max()
    n_sig   = int(np.sum(df > np.abs(df).max() * 1e-10))

    results_flat[m] = dict(lam=lam_f, residual=resid_f, ov_dominant=ov_dom,
                           d_vals=df, overlaps=ov_f, V=Vf, blk_idx=blk_idx,
                           v_ana=v_flat)

    # --- original non-flat for comparison ---
    Gp_blk  = zl.Gamma_plus[np.ix_(blk_idx, blk_idx)]
    dn, Vn  = np.linalg.eigh(Gp_blk)
    v_nf    = L_ana[m][blk_idx]
    v_norm_nf = np.linalg.norm(v_nf)
    Gnv     = Gp_blk @ v_nf
    lam_n   = np.real(np.dot(v_nf.conj(), Gnv)) / v_norm_nf**2
    resid_n = np.linalg.norm(Gnv - lam_n * v_nf) / (abs(lam_n) * v_norm_nf + 1e-300)
    ov_n    = np.abs(Vn.conj().T @ (v_nf / v_norm_nf))**2
    results_nflat[m] = dict(lam=lam_n, residual=resid_n, ov_dominant=ov_n.max())

    print(f"  m={m:+d}  flat :  lam={lam_f:+.4e}  residual={resid_f:.4e}  "
          f"ov_dom={ov_dom:.6f}  n_sig={n_sig}")
    print(f"  m={m:+d}  nflat:  lam={lam_n:+.4e}  residual={resid_n:.4e}  "
          f"ov_dom={results_nflat[m]['ov_dominant']:.6f}")
    print()

# -----------------------------------------------------------------------
# Overlap decomposition for flat-J (same format as original check)
# -----------------------------------------------------------------------
print()
print("Overlap^2 decomposition  (flat-J canonical eigenvectors vs v^ana_flat)")
print()
for m in range(-2, 3):
    if m not in results_flat:
        continue
    r   = results_flat[m]
    d   = r['d_vals']
    ov  = r['overlaps']
    dom = np.argmax(ov)
    print(f"--- Delta_M = {m:+d} ---")
    for k, (dk, o) in enumerate(zip(d, ov)):
        tag = " <- dominant" if k == dom else ""
        bar = "#" * int(50 * o)
        print(f"    k={k:2d}  d_k={dk:+.3e}  ov={o:.4f}  {bar}{tag}")
    print()

# -----------------------------------------------------------------------
# Summary table: residuals flat vs non-flat
# -----------------------------------------------------------------------
print()
print("=" * 65)
print("Residual summary: flat-J vs non-flat-J")
print("=" * 65)
print(f"  {'Delta_M':>8}  {'residual (flat)':>18}  {'residual (non-flat)':>20}  {'ratio':>8}")
print("  " + "-" * 60)
for m in range(-2, 3):
    if m not in results_flat:
        continue
    rf = results_flat[m]['residual']
    rn = results_nflat[m]['residual']
    ratio = rf / (rn + 1e-300)
    print(f"  m={m:+d}      {rf:18.4e}  {rn:20.4e}  {ratio:8.4f}")
print()
print("Interpretation:")
print("  residual -> 0 in flat-J  =>  J inhomogeneity IS the source of mismatch")
print("  residual unchanged       =>  mismatch is structural (orientation modes)")
print("Done (flat-J section).")
