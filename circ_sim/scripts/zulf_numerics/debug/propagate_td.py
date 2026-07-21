"""
Population dynamics: TD canonical jump operators vs flat-J Redfield reference.

Compares three Lindbladians on a single figure:
  - Flat-J Redfield  (solid)  : Gamma_plus built with omega_ij = 0 for all transitions
  - TD K=0           (dashed) : 5 bare IST jump operators (no Krylov dressing)
  - TD K=1           (dotted) : 5 first-order Krylov-dressed jump operators

Jump operators L_a^(2,m) are built by td_jump_operators.py (TD_jumps formula,
2/sqrt(pi) prefactor, Krylov coefficients c_k = (1/k!) 2^(k-1) Gamma^2((k+1)/2)).

Initial state : rho_0 = |up up up><up up up|
Evolution     : 100 microseconds
Output        : debug/figures/td_vs_flatJ_tc<tc>.png
"""

import sys
import os
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup: zulf_numerics/ for zulf_lindblad, debug/ for td_jump_operators
# ---------------------------------------------------------------------------
_DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
_ZULF_DIR  = os.path.dirname(_DEBUG_DIR)
if _ZULF_DIR  not in sys.path:
    sys.path.insert(0, _ZULF_DIR)
if _DEBUG_DIR not in sys.path:
    sys.path.insert(0, _DEBUG_DIR)

import zulf_lindblad as zl
import td_jump_operators as tdj

# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------
n_states     = zl.n_states
ekets_arr    = zl.ekets_arr        # (8,8), row k = k-th eigenstate row vector
state_labels = zl.state_labels
tau_c        = zl.tau_c
_vec_shape   = zl._vec_shape
_vec_dims    = zl._vec_dims

# H0 Liouville superoperator in column-major computational basis
L_ham_np = (-1j * (qt.spre(zl.H0) - qt.spost(zl.H0))).full()   # (64, 64)

J0 = zl.J_spectral(0.0)   # spectral density at omega=0 [s]

# ---------------------------------------------------------------------------
# Population extractor
# ---------------------------------------------------------------------------
def populations_from_vec(rho_liouv):
    rho_op  = qt.vector_to_operator(
        qt.Qobj(rho_liouv.reshape(_vec_shape), dims=_vec_dims)
    )
    rho_mat = rho_op.full()
    return np.array([
        np.real(ekets_arr[n].conj() @ rho_mat @ ekets_arr[n])
        for n in range(n_states)
    ])

# ---------------------------------------------------------------------------
# Flat-J Redfield Liouvillian  (J(omega_ij) -> J(0) for all transitions)
# ---------------------------------------------------------------------------
def build_flat_J_liouvillian():
    Gamma_raw = zl.C_matrix * J0
    GR = Gamma_raw.reshape(n_states, n_states, n_states, n_states)
    T1 = np.einsum('amcm->ac', GR)
    T4 = np.einsum('nbnd->bd', GR)
    L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
            + np.einsum('acbd->abcd', GR)
            + np.einsum('dbca->abcd', GR)
            - np.einsum('ac,bd->abcd', np.eye(n_states), T4))
    L_diss_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_states**2, n_states**2)
    U           = zl.ekets_arr.T
    V           = np.kron(U.conj(), U)
    L_diss_mat  = V @ L_diss_eig @ V.conj().T
    L_mat       = L_ham_np + L_diss_mat
    ev, vec     = np.linalg.eig(L_mat)
    c0          = np.linalg.inv(vec) @ zl._rho0_vec_qobj.full().flatten()
    return ev, vec, c0

# ---------------------------------------------------------------------------
# TD Lindblad Liouvillian  (from td_jump_operators.build_td_jump_operators)
# ---------------------------------------------------------------------------
def build_td_liouvillian(trunc_order):
    L_ops, _ = tdj.build_td_jump_operators(trunc_order=trunc_order)
    I8 = np.eye(n_states)
    D  = np.zeros((n_states**2, n_states**2), dtype=complex)
    for L in L_ops.values():   # L_ops is a dict (k, m) -> (8,8) array
        LdL = L.conj().T @ L
        D  += (np.kron(L.conj(), L)
               - 0.5 * np.kron(I8, LdL)
               - 0.5 * np.kron(LdL.T, I8))
    L_mat   = L_ham_np + D
    ev, vec = np.linalg.eig(L_mat)
    c0      = np.linalg.inv(vec) @ zl._rho0_vec_qobj.full().flatten()
    return ev, vec, c0

# ---------------------------------------------------------------------------
# Build Liouvillians
# ---------------------------------------------------------------------------
print("Building flat-J Redfield Liouvillian ...")
ev_fJ, vec_fJ, c0_fJ = build_flat_J_liouvillian()

print("Building TD Lindblad K=0 (bare IST) ...")
ev_td0, vec_td0, c0_td0 = build_td_liouvillian(trunc_order=0)

print("Building TD Lindblad K=1 (1st-order dressed) ...")
ev_td1, vec_td1, c0_td1 = build_td_liouvillian(trunc_order=1)

# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------
t_end    = 100e-6
N_steps  = 500
tlist    = np.linspace(0.0, t_end, N_steps)
tlist_us = tlist * 1e6

print(f"Propagating {N_steps} steps to t = {t_end*1e6:.0f} us ...")
pops_fJ  = np.zeros((N_steps, n_states))
pops_td0 = np.zeros((N_steps, n_states))
pops_td1 = np.zeros((N_steps, n_states))

for ti, t in enumerate(tlist):
    pops_fJ[ti]  = populations_from_vec(vec_fJ  @ (np.exp(ev_fJ  * t) * c0_fJ))
    pops_td0[ti] = populations_from_vec(vec_td0 @ (np.exp(ev_td0 * t) * c0_td0))
    pops_td1[ti] = populations_from_vec(vec_td1 @ (np.exp(ev_td1 * t) * c0_td1))

print("Done.\n")

# ---------------------------------------------------------------------------
# Report final populations
# ---------------------------------------------------------------------------
print(f"{'State':<30}  {'Flat-J Redf.':>13}  {'TD K=0':>10}  {'TD K=1':>10}")
print("-" * 70)
for n in range(n_states):
    lbl = state_labels[n].replace('$','').replace(r'\alpha','a').replace(r'\beta','b')
    print(f"  {lbl:<28}  {pops_fJ[-1,n]:>13.4f}  "
          f"{pops_td0[-1,n]:>10.4f}  {pops_td1[-1,n]:>10.4f}")
print()
for label, pops in [("Flat-J Redfield", pops_fJ),
                    ("TD K=0",          pops_td0),
                    ("TD K=1",          pops_td1)]:
    print(f"  {label}: trace = {pops[-1].sum():.6f}  (min pop = {pops[-1].min():.4e})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, n_states))

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    r'Population dynamics: $^{19}$F–$^{13}$C–$^{1}$H ZULF,  '
    r'$\rho_0=|\!\uparrow\uparrow\uparrow\rangle\langle\uparrow\uparrow\uparrow|$,  '
    + r'$\tau_c$=' + f'{tau_c:.0e} s',
    fontsize=11
)

for n in range(n_states):
    ax.plot(tlist_us, pops_fJ[: , n], color=colors[n], lw=1.8, ls='-')
    ax.plot(tlist_us, pops_td0[:, n], color=colors[n], lw=1.2, ls='--', alpha=0.85)
    ax.plot(tlist_us, pops_td1[:, n], color=colors[n], lw=1.0, ls=':',  alpha=0.85,
            label=state_labels[n])

ax.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
ax.set_ylabel('Population', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# State legend (right outside)
handles, labels = ax.get_legend_handles_labels()
leg1 = ax.legend(handles=handles, labels=labels, fontsize=7,
                 loc='upper right', bbox_to_anchor=(1.38, 1.0), ncol=1)
ax.add_artist(leg1)

# Line-style guide
style_handles = [
    Line2D([0], [0], color='k', lw=1.8, ls='-',  label=r'Flat-$J$ Redfield'),
    Line2D([0], [0], color='k', lw=1.2, ls='--', label=r'TD  $K{=}0$  (bare IST)'),
    Line2D([0], [0], color='k', lw=1.0, ls=':',  label=r'TD  $K{=}1$  (1st-order dressed)'),
]
ax.legend(handles=style_handles, fontsize=8, loc='upper right',
          bbox_to_anchor=(1.38, 0.35))

plt.tight_layout()
figures_dir = os.path.join(_DEBUG_DIR, 'figures')
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, f'td_vs_flatJ_tc{tau_c:.0e}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved -> {out_path}")
