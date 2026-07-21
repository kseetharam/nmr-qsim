"""
fch_lindblad_example.py — Verification example for linblad_utils.

Computes Lindblad population dynamics for the ¹⁹F–¹³C–¹H three-spin system
using build_jump_operators / build_lindblad_liouvillian from utils/linblad_utils.py,
and compares against Spinach reference data.

Parameters
----------
B_vec  = (0, 0, 1e-3) T          ω τ_c ≈ 0.03 for ¹⁹F  →  extreme-narrowing limit
τ_c    = 1e-7 s
t_end  = 10000 τ_c = 1 ms

Spinach reference: zeem_dephasing/data/spinach_pops_tc1e-07_B1.00e-03.mat
Figure output:     linblad_dyn/figures/fch_lindblad_tc1e-07_B1mT.png
"""

import os, re, sys
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR   = os.path.join(SCRIPT_DIR, 'utils')
FIG_DIR     = os.path.join(SCRIPT_DIR, 'figures')
SPINACH_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),          # circ_sim/scripts/
    'zulf_numerics', 'zeem_dephasing', 'data'
)

sys.path.insert(0, SCRIPT_DIR)
from utils.linblad_utils import build_jump_operators, build_lindblad_liouvillian

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
GAMMA = {'19F': 251.81520e6, '13C': 67.28284e6, '1H': 267.52218e6}

gammas = [GAMMA['19F'], GAMMA['13C'], GAMMA['1H']]   # spin order: F, C, H

J_hz = np.zeros((3, 3))
J_hz[0, 1] = J_hz[1, 0] = 243.5    # J(¹⁹F–¹³C)  Hz
J_hz[1, 2] = J_hz[2, 1] =  10.71   # J(¹³C–¹H)   Hz

coords_ang = np.array([
    [-3.9805, -0.5583, -0.6136],    # ¹⁹F
    [-2.7608, -0.2372, -0.1625],    # ¹³C
    [-0.8183,  2.4463,  0.3685],    # ¹H
])

sigma_ppm = [
    np.array([[-113.0955, -13.4634,  23.3880],   # ¹⁹F
              [ -13.4634, -32.0922, -17.6116],
              [  23.3880, -17.6116,-153.6685]]),
    np.array([[240.8318, 25.3402, 53.6419],       # ¹³C
              [ 25.3402,162.9404,  5.1940],
              [ 53.6419,  5.1940,127.8480]]),
    np.array([[ 5.0782,  2.5669, -1.6492],        # ¹H
              [ 2.5669, 10.7485,  0.0000],
              [-1.6492,  0.0000, 11.9705]]),
]

B_vec = np.array([0.0, 0.0, 1e-3])   # T
tau_c = 1e-7                           # s
t_end = 10000 * tau_c                  # 1 ms

print("=" * 60)
print("¹⁹F–¹³C–¹H Lindblad example  (via linblad_utils)")
print("=" * 60)
print(f"  B_vec  = {B_vec} T")
print(f"  tau_c  = {tau_c:.1e} s   (ω_F τ_c ≈ {GAMMA['19F']*np.linalg.norm(B_vec)*tau_c:.3f})")
print(f"  t_end  = {t_end*1e3:.1f} ms  (= 10000 τ_c)")
print()

# ---------------------------------------------------------------------------
# Build Hamiltonian and jump operators
# ---------------------------------------------------------------------------
H_iso, L1_ops, L2_ops, ops, evals, ekets = build_jump_operators(
    gammas, J_hz, coords_ang, sigma_ppm, B_vec, tau_c
)

n_states  = len(evals)
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)

_M_op = ops['Iz'][0] + ops['Iz'][1] + ops['Iz'][2]
M_qn  = np.round(2 * np.real([qt.expect(_M_op, ek) for ek in ekets])) / 2
E_Hz  = evals / (2 * np.pi)

print("H_iso eigenstates:")
for n in range(n_states):
    print(f"  |{n}>  M={M_qn[n]:+.1f}  E={E_Hz[n]:+.2f} Hz")

n_L1_active = sum(
    1 for op in L1_ops.values()
    if np.linalg.norm(op.full()) > 1e-20
)
n_L2_active = sum(
    1 for op in L2_ops.values()
    if np.linalg.norm(op.full()) > 1e-20
)
print(f"\nActive jump operators: L^(1) {n_L1_active}/9,  L^(2) {n_L2_active}/25")

# ---------------------------------------------------------------------------
# Assemble Liouvillian and propagate
# ---------------------------------------------------------------------------
L_mat    = build_lindblad_liouvillian(H_iso, L1_ops, L2_ops)
ev, evec = np.linalg.eig(L_mat)
evec_inv = np.linalg.inv(evec)

_up  = qt.basis(2, 0)
rho0 = qt.ket2dm(qt.tensor(_up, _up, _up))
rho0_vec  = qt.operator_to_vector(rho0)
vec_dims  = rho0_vec.dims
vec_shape = rho0_vec.shape
c0 = evec_inv @ rho0_vec.full().flatten()

N_steps = 500
tlist   = np.linspace(0.0, t_end, N_steps)

pops_py = np.zeros((N_steps, n_states))
min_eig = np.zeros(N_steps)

print("\nPropagating ...")
for ti, t in enumerate(tlist):
    v       = evec @ (np.exp(ev * t) * c0)
    rho_mat = qt.vector_to_operator(
        qt.Qobj(v.reshape(vec_shape), dims=vec_dims)).full()
    for n in range(n_states):
        pops_py[ti, n] = np.real(ekets_arr[n].conj() @ rho_mat @ ekets_arr[n])
    min_eig[ti] = np.linalg.eigvalsh(rho_mat).min()

print(f"  Trace:  t=0 → {pops_py[0].sum():.8f},  t_end → {pops_py[-1].sum():.8f}")
print(f"  min eigenvalue: [{min_eig.min():.2e}, {min_eig.max():.2e}]  "
      + ("(CPTP ✓)" if min_eig.min() >= -1e-10 else "(positivity violated)"))

# ---------------------------------------------------------------------------
# Load Spinach reference
# ---------------------------------------------------------------------------
spinach_file = os.path.join(SPINACH_DIR, 'spinach_pops_tc1e-07_B1.00e-03.mat')
if not os.path.exists(spinach_file):
    raise FileNotFoundError(f"Spinach reference not found:\n  {spinach_file}")

d_sp     = scipy.io.loadmat(spinach_file)
t_sp     = d_sp['t_axis'].ravel()
pops_sp  = d_sp['populations']
sp_lbls  = [str(d_sp['state_labels'][0, n][0]) for n in range(n_states)]
evals_sp = d_sp['evals_hilb'].ravel()

print("\nSpinach eigenstates:")
for i, lbl in enumerate(sp_lbls):
    print(f"  [{i}]  {lbl:35s}  E={evals_sp[i]/(2*np.pi):+.2f} Hz  "
          f"P(t=0)={pops_sp[i,0]:.4f}")

# ---------------------------------------------------------------------------
# State matching: M_total then energy proximity
# ---------------------------------------------------------------------------
def _parse_M(lbl):
    m = re.search(r'M=([+-])(\d+)/(\d+)', lbl)
    if m:
        return (1 if m.group(1)=='+' else -1) * int(m.group(2)) / int(m.group(3))
    m2 = re.search(r'M=([+-]?\d+\.?\d*)', lbl)
    return float(m2.group(1)) if m2 else None

sp2py = {}
for si, lbl in enumerate(sp_lbls):
    M_sp    = _parse_M(lbl)
    E_sp_hz = evals_sp[si] / (2 * np.pi)
    candidates = [ni for ni in range(n_states)
                  if (M_sp is None or abs(M_qn[ni] - M_sp) < 0.1)
                  and ni not in sp2py.values()]
    if candidates:
        sp2py[si] = min(candidates, key=lambda ni: abs(E_Hz[ni] - E_sp_hz))

print("\nState matching (Spinach → Python):")
for si, ni in sp2py.items():
    print(f"  [{si}] {sp_lbls[si]:35s} → |{ni}>  M={M_qn[ni]:+.1f}  E={E_Hz[ni]:+.2f} Hz")

# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------
colors   = plt.cm.tab10(np.linspace(0, 1, n_states))
markers  = ['o', 's', '^', 'D', 'v', '<', '>', 'P']
mk_every = max(1, len(t_sp) // 20)

def _clean(lbl):
    return lbl.replace('(1)', 'α').replace('(2)', 'β').replace('(3)', 'γ')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    r'$^{19}$F–$^{13}$C–$^{1}$H  Lindblad vs Spinach'
    '\n'
    r'$B_z = 1\,\mathrm{mT}$, '
    rf'$\tau_c = {tau_c:.0e}$ s '
    rf'($\omega_F\tau_c \approx 0.03$), '
    r'$t_\mathrm{end} = 10^4\,\tau_c$, '
    r'initial $|\!\uparrow\uparrow\uparrow\rangle$',
    fontsize=11
)

line_handles = []
for si in range(n_states):
    ni = sp2py.get(si)
    if ni is None:
        continue
    clr = colors[ni]
    lbl = _clean(sp_lbls[si])
    h, = ax1.plot(tlist * 1e3, pops_py[:, ni], color=clr, lw=2.0, label=lbl)
    ax1.plot(t_sp * 1e3, pops_sp[si, :], color=clr, ls='none',
             marker=markers[si % len(markers)], markersize=5,
             markevery=mk_every, alpha=0.85)
    line_handles.append(h)

ax1.set_ylabel('Population', fontsize=11)
ax1.set_ylim(-0.06, 1.06)
ax1.axhline(1.0/n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(handles=line_handles, fontsize=8, ncol=2, loc='upper right',
           title='solid = Lindblad  ($L\\propto A^{(l)}$),   markers = Spinach')
ax1.set_title('H$_\\mathrm{iso}$ eigenbasis populations', fontsize=10)

for si in range(n_states):
    ni = sp2py.get(si)
    if ni is None:
        continue
    interp = interp1d(t_sp, pops_sp[si, :], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
    ax2.plot(tlist * 1e3, pops_py[:, ni] - interp(tlist),
             color=colors[ni], lw=1.5, label=_clean(sp_lbls[si]))

ax2.axhline(0.0, color='k', ls='--', lw=0.8)
ax2.set_ylabel(r'$\Delta P_n$ (Lindblad $-$ Spinach)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7, ncol=2, loc='upper right')
ax2.set_title(
    r'Residuals  ($\omega_F\tau_c \approx 0.03 \ll 1$: extreme-narrowing limit holds)',
    fontsize=10
)

plt.tight_layout()
out = os.path.join(FIG_DIR, f'fch_lindblad_tc{tau_c:.0e}_B{np.linalg.norm(B_vec)*1e3:.0f}mT.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nFigure saved → {out}')
