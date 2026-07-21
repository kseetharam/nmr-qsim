"""
Verify TD jump operators by propagating rho_0 = |uuu><uuu| for 100 us.

Compares eigenbasis populations under four Lindbladians:
  - Redfield       : full Redfield from zulf_lindblad.py  (rho_at)
  - Redfield J=J0  : Redfield with J(omega) -> J(0) for all transitions
  - TD K=0         : 5 bare IST jump operators (no J-coupling dressing)
  - TD K=1         : 5 first-order Krylov-dressed jump operators

Equation of motion for TD approach:
    d rho/dt = -i [H0, rho]  +  sum_m D[L_m^TD]
    D[L](rho) = L rho L^dag - 1/2 {L^dag L, rho}

Requires zulf_lindblad.py and td_jump_operators.py in the same directory.
"""

import sys
import os
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup imports
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

print("Importing zulf_lindblad ...")
import zulf_lindblad as zl
print("Importing td_jump_operators ...")
import td_jump_operators as tdj
print("Done.\n")


# ---------------------------------------------------------------------------
# Shared objects from zulf_lindblad
# ---------------------------------------------------------------------------
n_states    = zl.n_states          # 8
ekets_arr   = zl.ekets_arr         # (8, 8)  row k = k-th eigenstate as row vec
state_labels = zl.state_labels     # list of 8 LaTeX strings
tau_c       = zl.tau_c
_vec_shape  = zl._vec_shape
_vec_dims   = zl._vec_dims

# H0 Liouville superoperator in column-major convention
L_ham_np = (-1j * (qt.spre(zl.H0) - qt.spost(zl.H0))).full()   # (64, 64)


# ---------------------------------------------------------------------------
# Population extractor  P_n(t) = <E_n|rho(t)|E_n>
# ---------------------------------------------------------------------------
def populations_from_vec(rho_vec):
    """Extract eigenbasis populations from a 64-element Liouville vector."""
    rho_op = qt.vector_to_operator(
        qt.Qobj(rho_vec.reshape(_vec_shape), dims=_vec_dims)
    )
    rho_mat = rho_op.full()
    return np.array([
        np.real(ekets_arr[n].conj() @ rho_mat @ ekets_arr[n])
        for n in range(n_states)
    ])


# ---------------------------------------------------------------------------
# Build flat-J Redfield Liouvillian  (J(omega) -> J(0) for all transitions)
# Replicates build_liouvillian() from zulf_lindblad.py with uniform J.
# ---------------------------------------------------------------------------
def build_flat_J_liouvillian():
    """
    Redfield Liouvillian with the spectral density evaluated at zero frequency
    for every transition: Gamma_raw[i,j] = C[i,j] * J(0).

    All other construction steps are identical to zulf_lindblad.build_liouvillian.
    """
    J0         = zl.J_spectral(0.0)                # scalar, rad/s
    Gamma_raw  = zl.C_matrix * J0                  # (64, 64), uniform J

    GR = Gamma_raw.reshape(n_states, n_states, n_states, n_states)
    T1 = np.einsum('amcm->ac', GR)
    T4 = np.einsum('nbnd->bd', GR)
    L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
            + np.einsum('acbd->abcd', GR)
            + np.einsum('dbca->abcd', GR)
            - np.einsum('ac,bd->abcd', np.eye(n_states), T4))

    L_diss_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_states**2, n_states**2)

    U = zl.ekets_arr.T
    V = np.kron(U.conj(), U)
    L_diss_mat = V @ L_diss_eig @ V.conj().T

    L_mat   = L_ham_np + L_diss_mat
    ev, vec = np.linalg.eig(L_mat)
    vec_inv = np.linalg.inv(vec)
    c0      = vec_inv @ zl._rho0_vec_qobj.full().flatten()
    return ev, vec, c0


# ---------------------------------------------------------------------------
# Build TD Liouvillian for a given truncation order
# ---------------------------------------------------------------------------
def build_td_liouvillian(trunc_order):
    """
    Return (L_mat, evals, evecs, evecs_inv, c0) for the TD Lindbladian:
        L = -i[H0, .] + sum_m D[L_m^TD]
    """
    L_ops_td, _ = tdj.build_td_jump_operators(trunc_order=trunc_order)

    I8 = np.eye(n_states)
    D  = np.zeros((n_states**2, n_states**2), dtype=complex)
    for L in L_ops_td:
        LdL  = L.conj().T @ L
        D   += (np.kron(L.conj(), L)
                - 0.5 * np.kron(I8, LdL)
                - 0.5 * np.kron(LdL.T, I8))

    L_mat    = L_ham_np + D
    ev, vec  = np.linalg.eig(L_mat)
    vec_inv  = np.linalg.inv(vec)
    rho0_vec = zl._rho0_vec_qobj.full().flatten()
    c0       = vec_inv @ rho0_vec
    return L_mat, ev, vec, c0


# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------
t_end   = 100e-6          # 100 microseconds
N_steps = 500
tlist   = np.linspace(0.0, t_end, N_steps)
tlist_us = tlist * 1e6

print("Building flat-J and TD Liouvillians ...")
ev_fJ, vec_fJ, c0_fJ = build_flat_J_liouvillian()
_, ev0, vec0, c0_K0  = build_td_liouvillian(trunc_order=0)
_, ev1, vec1, c0_K1  = build_td_liouvillian(trunc_order=1)
print(f"Propagating {N_steps} steps to t = {t_end*1e6:.0f} us ...")

pops_redf = np.zeros((N_steps, n_states))
pops_fJ   = np.zeros((N_steps, n_states))
pops_td0  = np.zeros((N_steps, n_states))
pops_td1  = np.zeros((N_steps, n_states))

for ti, t in enumerate(tlist):
    # Redfield (full, frequency-dependent J)
    rho_r   = zl.rho_at(t).full()
    for n in range(n_states):
        pops_redf[ti, n] = np.real(ekets_arr[n].conj() @ rho_r @ ekets_arr[n])

    # Redfield with flat J = J(0)
    pops_fJ[ti]  = populations_from_vec(vec_fJ @ (np.exp(ev_fJ * t) * c0_fJ))

    # TD K=0
    pops_td0[ti] = populations_from_vec(vec0  @ (np.exp(ev0  * t) * c0_K0))

    # TD K=1
    pops_td1[ti] = populations_from_vec(vec1  @ (np.exp(ev1  * t) * c0_K1))

print("Done.\n")

# ---------------------------------------------------------------------------
# Print final populations for verification
# ---------------------------------------------------------------------------
print(f"{'State':<30}  {'Redfield':>10}  {'Redf. J=J0':>12}  {'TD K=0':>10}  {'TD K=1':>10}")
print("-" * 78)
for n in range(n_states):
    lbl = state_labels[n].replace('$', '').replace(r'\alpha', 'a').replace(r'\beta', 'b')
    print(f"  {lbl:<28}  {pops_redf[-1, n]:>10.4f}  "
          f"{pops_fJ[-1, n]:>12.4f}  "
          f"{pops_td0[-1, n]:>10.4f}  {pops_td1[-1, n]:>10.4f}")
print()

for label, pops in [("Redfield",   pops_redf),
                    ("Redf. J=J0", pops_fJ),
                    ("TD K=0",     pops_td0),
                    ("TD K=1",     pops_td1)]:
    print(f"  {label}: trace = {pops[-1].sum():.6f}  "
          f"(min pop = {pops[-1].min():.4e})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, n_states))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(
    r'Population dynamics: $^{19}$F–$^{13}$C–$^{1}$H ZULF,  '
    r'$\rho_0=|\!\uparrow\uparrow\uparrow\rangle\langle\uparrow\uparrow\uparrow|$,  '
    + r'$\tau_c$=' + f'{tau_c:.0e} s',
    fontsize=11
)

# --- Panel 0: Redfield (solid) + flat-J Redfield (dashed) ---
ax = axes[0]
for n in range(n_states):
    ax.plot(tlist_us, pops_redf[:, n], color=colors[n],
            lw=1.5, ls='-',  label=state_labels[n])
    ax.plot(tlist_us, pops_fJ[:, n],   color=colors[n],
            lw=1.5, ls='--', alpha=0.7)
ax.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_title(r'Redfield  (solid) vs $J(\omega){=}J(0)$  (dashed)', fontsize=9)
ax.set_xlabel(r'Time  ($\mu$s)', fontsize=10)
ax.set_ylabel('Population', fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# --- Panels 1 & 2: TD ---
td_data  = [pops_td0, pops_td1]
td_titles = ['TD  K=0  (bare IST)', 'TD  K=1  (1st-order dressed)']
for ax, pops, title in zip(axes[1:], td_data, td_titles):
    for n in range(n_states):
        ax.plot(tlist_us, pops[:, n], color=colors[n],
                lw=1.5, label=state_labels[n])
    ax.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'Time  ($\mu$s)', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

axes[2].legend(fontsize=7, ncol=1, loc='upper right',
               bbox_to_anchor=(1.38, 1.0))

plt.tight_layout()
out_path = os.path.join(_DIR, 'data',
                        f'td_vs_redfield_tc{tau_c:.0e}.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved -> {out_path}")
