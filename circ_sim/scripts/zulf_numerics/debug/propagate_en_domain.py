"""
Lindbladian population dynamics using the energy-domain canonical jump operators
(eq. can_en_domain, liouville_hilbert_basis.tex):

    L_{k,m} = sqrt(J(0)) * Q_hat_{k,m}

where Q_hat_{k,m} = sum_{pairs} b_pair * a_{2,m}^pair * T_{2,k}^pair.

Comparison: flat-J Redfield (Gamma_plus built with omega_ij = 0 for all
transitions) vs energy-domain Lindbladian.  Both approximations assume a
flat spectral density J(omega) = J(0), so exact agreement is expected if
the jump-operator construction is correct.

Both curves are shown on the same plot.  Solid = flat-J Redfield,
dashed = energy-domain Lindblad.

Initial state: rho_0 = |up up up><up up up|
Evolution:     100 microseconds
Output:        debug/figures/en_domain_vs_flatJ_tc<tc>.png
"""

import sys
import os
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
_ZULF_DIR  = os.path.dirname(_DEBUG_DIR)
if _ZULF_DIR not in sys.path:
    sys.path.insert(0, _ZULF_DIR)

import zulf_lindblad as zl

# ---------------------------------------------------------------------------
# Shared system objects
# ---------------------------------------------------------------------------
n_states     = zl.n_states
ekets_arr    = zl.ekets_arr         # (8,8), row k = k-th eigenstate row vector
state_labels = zl.state_labels
tau_c        = zl.tau_c
_vec_shape   = zl._vec_shape
_vec_dims    = zl._vec_dims

# H0 Liouville superoperator in column-major computational basis
L_ham_np = (-1j * (qt.spre(zl.H0) - qt.spost(zl.H0))).full()   # (64, 64)

U       = ekets_arr.T               # (8,8), columns = eigenvectors
J0      = zl.J_spectral(0.0)        # spectral density at omega=0  [s]
sqrt_J0 = np.sqrt(2.0 * J0)         # factor of sqrt(2): D_Redf = 2 sum D[L_alpha]

# ---------------------------------------------------------------------------
# Flat-J Redfield Liouvillian  (Gamma_plus built with omega_ij = 0)
#
# Replicates build_liouvillian() from zulf_lindblad.py but replaces
# J(omega_ij) -> J(0) for every transition, giving Gamma_raw = C * J(0).
# ---------------------------------------------------------------------------
def build_flat_J_liouvillian():
    Gamma_raw = zl.C_matrix * J0                            # (64, 64)
    GR = Gamma_raw.reshape(n_states, n_states, n_states, n_states)
    T1 = np.einsum('amcm->ac', GR)
    T4 = np.einsum('nbnd->bd', GR)
    L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
            + np.einsum('acbd->abcd', GR)
            + np.einsum('dbca->abcd', GR)
            - np.einsum('ac,bd->abcd', np.eye(n_states), T4))
    L_diss_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_states**2, n_states**2)
    V           = np.kron(U.conj(), U)
    L_diss_mat  = V @ L_diss_eig @ V.conj().T
    L_mat       = L_ham_np + L_diss_mat
    ev, vec     = np.linalg.eig(L_mat)
    c0          = np.linalg.inv(vec) @ zl._rho0_vec_qobj.full().flatten()
    return ev, vec, c0

# ---------------------------------------------------------------------------
# Energy-domain Lindbladian from jump operators L_{k,m}
#
# L_{k,m} = sqrt(J(0)) * Q_hat_{k,m}
#
# Q_hat_{k,m} = sum_{pairs} b_pair * a_{2,m}^pair * T_{2,k}^pair
#
# The expansion sum_i (Q_hat)_i sigma_i over ALL transition operators
# sigma_i = |n_i><m_i| (including diagonal n_i=m_i) reconstructs Q_hat
# itself via the identity resolution.  The operator is kept in the
# computational basis to be consistent with L_ham_np.
# ---------------------------------------------------------------------------
PAIRS = [
    ('FC', 0, 1, float(zl.b_FC), zl.r_FC),
    ('FH', 0, 2, float(zl.b_FH), zl.r_FH),
    ('CH', 1, 2, float(zl.b_CH), zl.r_CH),
]

a_factors = {}
for label, si, sj, b, r_vec in PAIRS:
    for m, val in zl.a2m_factors(r_vec).items():
        a_factors[(label, m)] = val

k_vals = m_vals = [-2, -1, 0, +1, +2]

def build_en_domain_liouvillian():
    I8 = np.eye(n_states)
    D  = np.zeros((n_states**2, n_states**2), dtype=complex)
    for k in k_vals:
        T2k = {label: zl.T2_op(si, sj, k).full().astype(complex)
               for label, si, sj, *_ in PAIRS}
        for m in m_vals:
            Q = np.zeros((n_states, n_states), dtype=complex)
            for label, si, sj, b, _ in PAIRS:
                Q += b * a_factors[(label, m)] * T2k[label]
            L   = sqrt_J0 * Q
            LdL = L.conj().T @ L
            D  += (np.kron(L.conj(), L)
                   - 0.5 * np.kron(I8, LdL)
                   - 0.5 * np.kron(LdL.T, I8))
    L_mat = L_ham_np + D
    ev, vec = np.linalg.eig(L_mat)
    c0      = np.linalg.inv(vec) @ zl._rho0_vec_qobj.full().flatten()
    return ev, vec, c0

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
# Build Liouvillians
# ---------------------------------------------------------------------------
print("Building flat-J Redfield Liouvillian ...")
ev_fJ, vec_fJ, c0_fJ = build_flat_J_liouvillian()

print("Building energy-domain Lindbladian (25 jump operators) ...")
ev_en, vec_en, c0_en = build_en_domain_liouvillian()

# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------
t_end    = 100e-6
N_steps  = 500
tlist    = np.linspace(0.0, t_end, N_steps)
tlist_us = tlist * 1e6

print(f"Propagating {N_steps} steps to t = {t_end*1e6:.0f} us ...")
pops_fJ = np.zeros((N_steps, n_states))
pops_en = np.zeros((N_steps, n_states))

for ti, t in enumerate(tlist):
    pops_fJ[ti] = populations_from_vec(vec_fJ @ (np.exp(ev_fJ * t) * c0_fJ))
    pops_en[ti] = populations_from_vec(vec_en @ (np.exp(ev_en * t) * c0_en))

print("Done.\n")

# ---------------------------------------------------------------------------
# Report final populations
# ---------------------------------------------------------------------------
print(f"{'State':<30}  {'Flat-J Redf.':>13}  {'En-domain':>12}")
print("-" * 62)
for n in range(n_states):
    lbl = state_labels[n].replace('$','').replace(r'\alpha','a').replace(r'\beta','b')
    print(f"  {lbl:<28}  {pops_fJ[-1,n]:>13.4f}  {pops_en[-1,n]:>12.4f}")
print()
for label, pops in [("Flat-J Redfield", pops_fJ), ("En-domain", pops_en)]:
    print(f"  {label}: trace = {pops[-1].sum():.6f}  (min pop = {pops[-1].min():.4e})")

# ---------------------------------------------------------------------------
# Plot — single panel, solid = flat-J Redfield, dashed = en-domain Lindblad
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
    ax.plot(tlist_us, pops_fJ[:, n], color=colors[n], lw=1.8, ls='-',
            label=state_labels[n])
    ax.plot(tlist_us, pops_en[:, n], color=colors[n], lw=1.2, ls='--',
            alpha=0.85)

ax.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel(r'Time  ($\mu$s)', fontsize=11)
ax.set_ylabel('Population', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Custom legend: states + linestyle guide
handles, labels = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color='k', lw=1.8, ls='-',  label=r'Flat-$J$ Redfield  ($\omega_{ij}=0$)'),
    Line2D([0], [0], color='k', lw=1.2, ls='--', label=r'En-domain Lindblad  ($L_\alpha$)'),
]
leg1 = ax.legend(handles=handles, labels=labels, fontsize=7,
                 loc='upper right', bbox_to_anchor=(1.38, 1.0), ncol=1)
ax.add_artist(leg1)
ax.legend(handles=style_handles, fontsize=8, loc='upper right',
          bbox_to_anchor=(1.38, 0.35))

plt.tight_layout()
figures_dir = os.path.join(_DEBUG_DIR, 'figures')
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, f'en_domain_vs_flatJ_tc{tau_c:.0e}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved -> {out_path}")
