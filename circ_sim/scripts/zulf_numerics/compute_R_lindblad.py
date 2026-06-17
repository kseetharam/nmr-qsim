"""
compute_R_lindblad.py

Computes the 64×64 relaxation matrix R_L in the {E, Lz, L+, L-}^3
product-operator basis from the Lindblad dissipator in zulf_lindblad.py:

    R_L[i,j] = Tr{ o_i† · L_diss[o_j] }

where L_diss is the full dissipator (Gamma_plus + Gamma_minus terms,
gamma_scale=1.0), and {o_k} is the same product-operator basis used by
Spinach's sphten-liouv formalism with bas.approximation='none'.

Outputs (in data/):
  R_L_lindblad_gamma1.npy    -- 64×64 complex matrix (numpy)
  R_L_lindblad_gamma1.mat    -- same in MATLAB .mat format
  R_L_basis_labels.txt       -- row/column operator labels
  R_L_heatmap_gamma1.png     -- log-scale heatmap of |Re(R_L)|, |Im(R_L)|

Note: R_L is computed at gamma_scale=1.0 (as derived from Redfield theory).
If Spinach rates are 2x larger, expect R_Spinach ≈ 2 * R_L element-wise.
"""

import sys
import os
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import savemat

# ---------------------------------------------------------------------------
# Import zulf_lindblad (runs module-level setup at gamma_scale=1.0)
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad ...")
import zulf_lindblad as zl
print("Import complete.\n")

DATA_DIR = os.path.join(_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Isolate the dissipator: L_diss = L_total - (-i[H0, .])
# ---------------------------------------------------------------------------
L_ham_mat  = (-1j * (qt.spre(zl.H0) - qt.spost(zl.H0))).full()   # 64×64
L_diss_mat = zl.L_mat - L_ham_mat                                  # 64×64

# ---------------------------------------------------------------------------
# Build the {E, Lz, L+, L-}^3 product-operator basis
# (identical to the ops_h / ops_1d construction in fch_triple_redfield.m)
# ---------------------------------------------------------------------------
E2 = np.eye(2, dtype=complex)
Lz = np.diag([0.5, -0.5]).astype(complex)
Lp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
Lm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

_ops1 = [E2, Lz, Lp, Lm]
_lbl1 = ['E', 'Lz', 'L+', 'L-']

ops_basis = []
lbl_basis = []
for a in range(4):
    for b in range(4):
        for c in range(4):
            ops_basis.append(np.kron(np.kron(_ops1[a], _ops1[b]), _ops1[c]))
            lbl_basis.append(f'{_lbl1[a]}x{_lbl1[b]}x{_lbl1[c]}')

n_ops   = len(ops_basis)   # 64
_dims   = [[2, 2, 2], [2, 2, 2]]
_vshape = zl._rho0_vec_qobj.shape
_vdims  = zl._rho0_vec_qobj.dims

# ---------------------------------------------------------------------------
# Compute R_L[i,j] = Tr( o_i† · L_diss[o_j] )
#
# For each column j:
#   1. Vectorise o_j using QuTiP's column-stacking convention
#   2. Apply L_diss_mat to get vec( L_diss[o_j] )
#   3. Reshape back to an 8×8 operator
#   4. Contract with o_i† via Tr( o_i† · L_diss[o_j] ) for all i
# ---------------------------------------------------------------------------
print(f"Computing R_L ({n_ops}×{n_ops}) ...")
R_L = np.zeros((n_ops, n_ops), dtype=complex)

for j in range(n_ops):
    oj_qt  = qt.Qobj(ops_basis[j], dims=_dims)
    vec_oj = qt.operator_to_vector(oj_qt).full().flatten()
    vec_Lj = L_diss_mat @ vec_oj
    L_oj   = qt.vector_to_operator(
                 qt.Qobj(vec_Lj.reshape(_vshape), dims=_vdims)
             ).full()                                    # 8×8 matrix
    for i in range(n_ops):
        R_L[i, j] = np.trace(ops_basis[i].conj().T @ L_oj)

print("Done.\n")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
npy_path = os.path.join(DATA_DIR, 'R_L_lindblad_gamma1.npy')
mat_path = os.path.join(DATA_DIR, 'R_L_lindblad_gamma1.mat')
lbl_path = os.path.join(DATA_DIR, 'R_L_basis_labels.txt')

np.save(npy_path, R_L)
savemat(mat_path, {'R_L': R_L, 'labels': np.array(lbl_basis, dtype=object)})

with open(lbl_path, 'w') as f:
    f.write('# index  label  (row/column ordering of R_L)\n')
    for k, lbl in enumerate(lbl_basis):
        f.write(f'{k:3d}  {lbl}\n')

print(f"Saved: {npy_path}")
print(f"Saved: {mat_path}  (MATLAB-loadable: load('R_L_lindblad_gamma1.mat'))")
print(f"Saved: {lbl_path}\n")

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
re_RL     = np.real(R_L)
im_RL     = np.imag(R_L)
herm_err  = np.abs(R_L - R_L.conj().T).max()
evals_RL  = np.linalg.eigvals(R_L)
re_evals  = np.real(evals_RL)

print("=" * 60)
print("R_L diagnostics  (gamma_scale = 1.0)")
print("=" * 60)
print(f"  max |Re(R_L)|            : {np.abs(re_RL).max():.6e} rad/s "
      f"({np.abs(re_RL).max()/(2*np.pi):.6e} Hz)")
print(f"  max |Im(R_L)|            : {np.abs(im_RL).max():.6e} rad/s")
print(f"  Hermiticity check        : max|R_L - R_L†| = {herm_err:.4e} rad/s")
print(f"  Eigenvalue Re range      : [{re_evals.min():.4e}, {re_evals.max():.4e}] rad/s")
n_pos = (re_evals > 1e-6).sum()
if n_pos:
    print(f"  WARNING: {n_pos} eigenvalue(s) with positive real part")
else:
    print("  All eigenvalue real parts ≤ 0 (no positivity breakdown in R_L)")

# Gram matrix G[i,j] = Tr(o_i† o_j) — needed to relate R_L to rate equations
G = np.array([[np.trace(ops_basis[i].conj().T @ ops_basis[j])
               for j in range(n_ops)] for i in range(n_ops)])
print(f"\n  Gram matrix G[i,j] = Tr(o_i† o_j):")
print(f"    diagonal range   : [{np.real(np.diag(G)).min():.4f}, {np.real(np.diag(G)).max():.4f}]")
print(f"    off-diag max |G| : {np.abs(G - np.diag(np.diag(G))).max():.6e}")

# Top-10 largest off-diagonal |R_L| entries
flat_abs = np.abs(R_L).copy()
np.fill_diagonal(flat_abs, 0.0)
top10_idx = np.argsort(flat_abs.ravel())[::-1][:10]

print(f"\n  Top-10 off-diagonal |R_L[i,j]|  (rad/s):")
print(f"  {'i':>3}  {'j':>3}  {'label_i':<20}  {'label_j':<20}  {'|R_L|/2pi (Hz)':>16}")
print("  " + "-" * 72)
for idx in top10_idx:
    ii, jj = divmod(int(idx), n_ops)
    print(f"  {ii:3d}  {jj:3d}  {lbl_basis[ii]:<20}  {lbl_basis[jj]:<20}  "
          f"{abs(R_L[ii, jj]) / (2*np.pi):16.4e}")

# Diagonal entries (represent population decay / dephasing)
print(f"\n  Diagonal R_L[k,k]  (rad/s)  — decay/dephasing of each operator:")
print(f"  {'k':>3}  {'label':<20}  {'Re(R_L[k,k])/2pi (Hz)':>24}  {'Im(R_L[k,k])/2pi (Hz)':>24}")
print("  " + "-" * 76)
for k in range(n_ops):
    val = R_L[k, k]
    if abs(val) > 1e-6:   # only print non-trivial entries
        print(f"  {k:3d}  {lbl_basis[k]:<20}  "
              f"{np.real(val)/(2*np.pi):24.6e}  {np.imag(val)/(2*np.pi):24.6e}")

# ---------------------------------------------------------------------------
# Heatmap: |Re(R_L)| and |Im(R_L)|
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    r'$R_L$ (Lindblad, $\gamma$-scale = 1.0): '
    r'$^{19}$F–$^{13}$C–$^{1}$H ZULF,  $\tau_c$ = ' + f'{zl.tau_c:.0e} s',
    fontsize=12
)
mats   = [np.abs(re_RL), np.abs(im_RL)]
titles = [r'$|\mathrm{Re}(R_L)|$  (rad/s)', r'$|\mathrm{Im}(R_L)|$  (rad/s)']
for ax, mat, title in zip(axes, mats, titles):
    vmax = mat.max()
    if vmax > 0:
        im = ax.imshow(mat, cmap='inferno', interpolation='nearest',
                       norm=LogNorm(vmin=max(vmax * 1e-8, 1e-30), vmax=vmax))
    else:
        im = ax.imshow(mat, cmap='inferno', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='rad/s', fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('j  (operator index)', fontsize=9)
    ax.set_ylabel('i  (operator index)', fontsize=9)
    # Mark block boundaries every 4 columns/rows (same spin-1 block size)
    for pos in range(4, n_ops, 4):
        ax.axhline(pos - 0.5, color='cyan', lw=0.3, alpha=0.4)
        ax.axvline(pos - 0.5, color='cyan', lw=0.3, alpha=0.4)
plt.tight_layout()
fig_path = os.path.join(DATA_DIR, 'R_L_heatmap_gamma1.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Heatmap saved -> {fig_path}")
