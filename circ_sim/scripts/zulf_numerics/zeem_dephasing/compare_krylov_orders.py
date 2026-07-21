"""
Compare Lindblad population dynamics at four Krylov expansion cases
(Eq. trunc_order_gen, notes/liouville_hilbert_basis.tex) against Spinach
(full Redfield) reference for the three-spin 19F–13C–1H model.

Cases compared:
  {0}       — extreme-narrowing (zeroth-order only)
  {0,1}     — adds first H_iso-dressing correction
  {0,1,2}   — adds second H_iso-dressing correction
  {0,2}     — even-orders only (skips odd imaginary term)

At tau_c = 2e-6 s (omega_F * tau_c ≈ 0.50) the series is in the
intermediate regime: non-trivial corrections are visible but the truncation
converges within 2–3 terms.  The expected pattern is:

    |Δ({0,1,2})| < |Δ({0})|        (even partial sums improve)
    |Δ({0,1})| may exceed |Δ({0})|  (odd-order overshoot)
    |Δ({0,2})| ≈ |Δ({0,1,2})|      (even-only ≈ full even partial sum)

Operator and Liouvillian construction is delegated entirely to
circ_sim/scripts/linblad_dyn/utils/linblad_utils.py.
"""

import os, re, sys
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Resolve linblad_utils from the repo tree
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.join(_HERE, '..', '..', 'linblad_dyn', 'utils')
sys.path.insert(0, os.path.normpath(_UTILS))
from linblad_utils import build_jump_operators_krylov, build_lindblad_liouvillian

DATA_DIR     = os.path.join(_HERE, 'data')
SPINACH_FILE = os.path.join(DATA_DIR, 'spinach_pops_tc2e-06_B1.00e-03.mat')

# =============================================================================
# SPIN-SYSTEM PARAMETERS  (19F–13C–1H, DFT geometry)
# =============================================================================

GAMMA  = {'19F': 251.81520e6, '13C': 67.28284e6, '1H': 267.52218e6}
NUCLEI = ['19F', '13C', '1H']

gammas = [GAMMA[X] for X in NUCLEI]

J_hz        = np.zeros((3, 3))
J_hz[0, 1]  = J_hz[1, 0] = 243.5
J_hz[1, 2]  = J_hz[2, 1] = 10.71

coords_ang = [
    [-3.9805, -0.5583, -0.6136],
    [-2.7608, -0.2372, -0.1625],
    [-0.8183,  2.4463,  0.3685],
]

sigma_ppm = [
    [[-113.0955, -13.4634,  23.3880],
     [ -13.4634, -32.0922, -17.6116],
     [  23.3880, -17.6116,-153.6685]],
    [[ 240.8318,  25.3402,  53.6419],
     [  25.3402, 162.9404,   5.1940],
     [  53.6419,   5.1940, 127.8480]],
    [[   5.0782,   2.5669,  -1.6492],
     [   2.5669,  10.7485,   0.0000],
     [  -1.6492,   0.0000,  11.9705]],
]

B_vec = np.array([0.0, 0.0, 1e-3])   # T
B0    = np.linalg.norm(B_vec)
tau_c = 2e-6   # s

print(f"tau_c = {tau_c:.1e} s,  |B| = {B0*1e3:.1f} mT")
print(f"omega_F * tau_c = {GAMMA['19F'] * B0 * tau_c:.3f}  "
      f"(extreme narrowing requires << 1)")
print()

# =============================================================================
# FOUR CASES TO COMPARE
#   key  : human-readable label
#   value: max_order argument (int or list) for build_jump_operators_krylov
# =============================================================================

CASES = [
    ("Order 0",       0),
    ("Order 0+1",     1),
    ("Order 0+1+2",   2),
    ("Order 0+2",     [0, 2]),
]
CASE_KEYS  = [c[0] for c in CASES]
CASE_ARGS  = {c[0]: c[1] for c in CASES}
LINESTYLES = {
    "Order 0":     ('-',   2.0),
    "Order 0+1":   ('--',  1.5),
    "Order 0+1+2": (':',   1.5),
    "Order 0+2":   ('-.', 1.5),
}

# =============================================================================
# BUILD JUMP OPERATORS AND LIOUVILLIANS FOR EACH CASE
# =============================================================================

lindbladians = {}
eigsystems   = {}

for label in CASE_KEYS:
    print(f"Building {label} jump operators ...")
    H_iso, L1, L2, ops, evals, ekets = build_jump_operators_krylov(
        gammas, J_hz, coords_ang, sigma_ppm, B_vec, tau_c,
        max_order=CASE_ARGS[label]
    )
    lindbladians[label] = build_lindblad_liouvillian(H_iso, L1, L2)
    eigsystems[label]   = (evals, ekets, H_iso)

print()

# =============================================================================
# PROPAGATION
# =============================================================================

# All cases share the same H_iso → same eigenstates
_, ekets_any, H_iso_any = eigsystems[CASE_KEYS[0]]
n_states  = len(ekets_any)
ekets_arr = np.array([ek.full().flatten() for ek in ekets_any], dtype=complex)

_M_op = sum(
    qt.tensor([qt.qeye(2) if i != k else qt.jmat(0.5, 'z') for i in range(3)])
    for k in range(3)
)
M_qn  = np.round(2 * np.real([qt.expect(_M_op, ek) for ek in ekets_any])) / 2
E_Hz  = eigsystems[CASE_KEYS[0]][0] / (2 * np.pi)

_up      = qt.basis(2, 0)
rho0     = qt.ket2dm(qt.tensor(_up, _up, _up))
rho0_vec = qt.operator_to_vector(rho0)
vec_dims  = rho0_vec.dims
vec_shape = rho0_vec.shape

t_end   = 2e-2   # s — match Spinach time axis
N_steps = 500
tlist   = np.linspace(0.0, t_end, N_steps)


def propagate(L_total, label):
    ev, evec = np.linalg.eig(L_total)
    evec_inv = np.linalg.inv(evec)
    c0       = evec_inv @ rho0_vec.full().flatten()
    pops     = np.zeros((N_steps, n_states))
    print(f"  Propagating {label} ...")
    for ti, t in enumerate(tlist):
        v       = evec @ (np.exp(ev * t) * c0)
        rho_mat = qt.vector_to_operator(
            qt.Qobj(v.reshape(vec_shape), dims=vec_dims)).full()
        for n in range(n_states):
            pops[ti, n] = np.real(ekets_arr[n].conj() @ rho_mat @ ekets_arr[n])
    tr0, trT = pops[0].sum(), pops[-1].sum()
    print(f"    trace: t=0 → {tr0:.6f},  t_end → {trT:.6f}")
    return pops


pops = {label: propagate(lindbladians[label], label) for label in CASE_KEYS}
print()

# =============================================================================
# LOAD SPINACH REFERENCE
# =============================================================================

if not os.path.exists(SPINACH_FILE):
    raise FileNotFoundError(
        f"Spinach reference file not found:\n  {SPINACH_FILE}\n"
        "Run fch_triple_redfield(2e-6, 1e-3, 'none') in MATLAB first."
    )

d_sp     = scipy.io.loadmat(SPINACH_FILE)
t_sp     = d_sp['t_axis'].ravel()
pops_sp  = d_sp['populations']
sp_lbls  = [str(d_sp['state_labels'][0, n][0]) for n in range(8)]
evals_sp = d_sp['evals_hilb'].ravel()

# =============================================================================
# STATE MATCHING
# =============================================================================

def parse_spinach_M(lbl):
    m = re.search(r'M=([+-])(\d+)/(\d+)', lbl)
    if m:
        sign = 1 if m.group(1) == '+' else -1
        return sign * int(m.group(2)) / int(m.group(3))
    m2 = re.search(r'M=([+-]?\d+\.?\d*)', lbl)
    return float(m2.group(1)) if m2 else None

sp2py = {}
for si, lbl in enumerate(sp_lbls):
    M_sp    = parse_spinach_M(lbl)
    E_sp_hz = evals_sp[si] / (2 * np.pi)
    cands   = [ni for ni in range(n_states)
               if (M_sp is None or abs(M_qn[ni] - M_sp) < 0.1)
               and ni not in sp2py.values()]
    if cands:
        sp2py[si] = min(cands, key=lambda ni: abs(E_Hz[ni] - E_sp_hz))

sp_interp = {
    si: interp1d(t_sp, pops_sp[si, :], kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    for si in sp2py
}

# =============================================================================
# PLOT
# Two-panel figure:
#   Top:    populations — four Krylov cases (lines) + Spinach (markers)
#   Bottom: residuals for all four cases
# =============================================================================

colors   = plt.cm.tab10(np.linspace(0, 1, n_states))
markers  = ['o', 's', '^', 'D', 'v', '<', '>', 'P']
mk_every = max(1, len(t_sp) // 20)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
fig.suptitle(
    r'$B_z = 1\,\mathrm{mT}$: Krylov jump operators vs Spinach (full Redfield)'
    '\n' + r'$^{19}$F–$^{13}$C–$^{1}$H,  '
    + rf'$\tau_c = {tau_c:.0e}$ s,  '
    + r'initial $|\!\uparrow\uparrow\uparrow\rangle$'
    + rf'  ($\omega_F\tau_c \approx {GAMMA["19F"]*B0*tau_c:.2f}$)',
    fontsize=11
)

# --- top panel ---
for si in sp2py:
    ni  = sp2py[si]
    clr = colors[ni]
    mk  = markers[si % len(markers)]
    for label in CASE_KEYS:
        ls, lw = LINESTYLES[label]
        ax1.plot(tlist * 1e3, pops[label][:, ni], color=clr, lw=lw, ls=ls)
    ax1.plot(t_sp * 1e3, pops_sp[si, :],
             color=clr, ls='none', marker=mk,
             markersize=5, markevery=mk_every, alpha=0.85)

ax1.set_ylabel('Population', fontsize=11)
ax1.set_ylim(-0.06, 1.06)
ax1.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(handles=[
    Line2D([0],[0], color='k', lw=2.0, ls='-',   label='Order 0'),
    Line2D([0],[0], color='k', lw=1.5, ls='--',  label='Order 0+1'),
    Line2D([0],[0], color='k', lw=1.5, ls=':',   label='Order 0+1+2'),
    Line2D([0],[0], color='k', lw=1.5, ls='-.',  label='Order 0+2'),
    Line2D([0],[0], color='k', lw=0,   marker='o', markersize=5,
           label='Spinach (Redfield)', alpha=0.85),
], fontsize=9, loc='upper right')
ax1.set_title('Eigenbasis populations', fontsize=10)

# --- bottom panel ---
max_res = {}
for label in CASE_KEYS:
    max_res[label] = 0.0
    ls, lw = LINESTYLES[label]
    for si in sp2py:
        ni    = sp2py[si]
        delta = pops[label][:, ni] - sp_interp[si](tlist)
        ax2.plot(tlist * 1e3, delta, color=colors[ni], lw=lw, ls=ls)
        max_res[label] = max(max_res[label], np.abs(delta).max())

ax2.axhline(0.0, color='k', ls='--', lw=0.8)
ax2.set_ylabel(r'$\Delta P_n$  (Krylov $-$ Spinach)', fontsize=11)
ax2.set_xlabel(r'Time (ms)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(handles=[
    Line2D([0],[0], color='k', lw=2.0, ls='-',
           label=f'Order 0      (max|Δ| = {max_res["Order 0"]:.4f})'),
    Line2D([0],[0], color='k', lw=1.5, ls='--',
           label=f'Order 0+1    (max|Δ| = {max_res["Order 0+1"]:.4f})'),
    Line2D([0],[0], color='k', lw=1.5, ls=':',
           label=f'Order 0+1+2  (max|Δ| = {max_res["Order 0+1+2"]:.4f})'),
    Line2D([0],[0], color='k', lw=1.5, ls='-.',
           label=f'Order 0+2    (max|Δ| = {max_res["Order 0+2"]:.4f})'),
], fontsize=9, loc='upper right')
ax2.set_title('Residuals vs Spinach (full Redfield)', fontsize=10)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, f'compare_krylov_B{B0:.2e}T_tc{tau_c:.0e}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Max |residual|  Order 0:      {max_res['Order 0']:.5f}")
print(f"Max |residual|  Order 0+1:    {max_res['Order 0+1']:.5f}")
print(f"Max |residual|  Order 0+1+2:  {max_res['Order 0+1+2']:.5f}")
print(f"Max |residual|  Order 0+2:    {max_res['Order 0+2']:.5f}")
print(f"Plot saved -> {out_path}")
