"""
Compare ZULF (B0 = 0) population dynamics:
  - Our Redfield description (zeem_lindblad physics at B=0)   → continuous lines
  - Spinach reference calculation                              → markers

At B=0 the Q^{(Z),l} operators vanish (they scale as B), so zeem_lindblad
reduces to the dipolar-only Redfield, which is what Spinach computes.
"""

import os, re
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
SPINACH_FILE = os.path.join(DATA_DIR, 'spinach_pops_tc1e-05_B0.00e+00.mat')

# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

hbar  = 1.054571817e-34
mu0   = 1.25663706212e-6
GAMMA = {'19F': 251.81520e6, '13C': 67.28284e6, '1H': 267.52218e6}
NUCLEI  = ['19F', '13C', '1H']
ANGSTROM = 1e-10
tau_c = 1e-5   # s

coords_ang = {
    '19F': np.array([-3.9805, -0.5583, -0.6136]),
    '13C': np.array([-2.7608, -0.2372, -0.1625]),
    '1H':  np.array([-0.8183,  2.4463,  0.3685]),
}
coords_m = {k: v * ANGSTROM for k, v in coords_ang.items()}
r_FC = coords_m['13C'] - coords_m['19F']
r_FH = coords_m['1H']  - coords_m['19F']
r_CH = coords_m['1H']  - coords_m['13C']

J_FC_rad = 2 * np.pi * 243.5
J_CH_rad = 2 * np.pi * 10.71

# =============================================================================
# SPIN OPERATORS (8-dim Hilbert space, B=0 → pure Heisenberg H0)
# =============================================================================

def spin_op(op_char, idx, n=3):
    ops = [qt.qeye(2)] * n
    ops[idx] = qt.jmat(0.5, op_char)
    return qt.tensor(ops)

Ix = [spin_op('x', k) for k in range(3)]
Iy = [spin_op('y', k) for k in range(3)]
Iz = [spin_op('z', k) for k in range(3)]
Ip = [Ix[k] + 1j*Iy[k] for k in range(3)]
Im = [Ix[k] - 1j*Iy[k] for k in range(3)]

def heis(i, j):
    return Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j]

H0 = J_FC_rad * heis(0, 1) + J_CH_rad * heis(1, 2)
evals, ekets = H0.eigenstates()
n_states  = len(evals)
n_trans   = n_states ** 2
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)

# Quantum numbers
_M_op  = Iz[0] + Iz[1] + Iz[2]
_S2_op = ((Ix[0]+Ix[1]+Ix[2])**2
         + (Iy[0]+Iy[1]+Iy[2])**2
         + (Iz[0]+Iz[1]+Iz[2])**2)
S_qn = (np.round(2*(np.sqrt(np.clip(4*np.real(
            [qt.expect(_S2_op, ek) for ek in ekets])+1, 0, None))-1)/2)/2)
M_qn = np.round(2 * np.real([qt.expect(_M_op, ek) for ek in ekets])) / 2
E_Hz = evals / (2 * np.pi)

# Doublet rank: 0 = lower energy S=1/2 doublet, 1 = higher energy S=1/2 doublet
_s12_Es = np.unique(np.round(E_Hz[np.abs(S_qn - 0.5) < 0.1], 1))

def doublet_rank(n):
    if abs(S_qn[n] - 0.5) > 0.1:
        return None
    return int(np.searchsorted(_s12_Es, round(E_Hz[n], 1)))

print("Python eigenstates (ascending energy):")
for n in range(n_states):
    dr = doublet_rank(n)
    dr_str = f', doublet_rank={dr}' if dr is not None else ''
    print(f"  |{n}>  S={S_qn[n]:.1f}  M={M_qn[n]:+.1f}  E={E_Hz[n]:+.2f} Hz{dr_str}")

# =============================================================================
# DIPOLAR SYSTEM OPERATORS Q_hat_{k,m}
# =============================================================================

def b_dip(gi, gj, r_vec):
    return -(mu0/(4*np.pi)) * gi * gj * hbar / np.linalg.norm(r_vec)**3

b_FC = b_dip(GAMMA['19F'], GAMMA['13C'], r_FC)
b_FH = b_dip(GAMMA['19F'], GAMMA['1H'],  r_FH)
b_CH = b_dip(GAMMA['13C'], GAMMA['1H'],  r_CH)

def T2_op(i, j, k):
    if k ==  0: return (2*Iz[i]*Iz[j] - Ix[i]*Ix[j] - Iy[i]*Iy[j]) / np.sqrt(6)
    if k ==  1: return -(Ip[i]*Iz[j] + Iz[i]*Ip[j]) / 2
    if k == -1: return  (Im[i]*Iz[j] + Iz[i]*Im[j]) / 2
    if k ==  2: return  Ip[i]*Ip[j] / 2
    if k == -2: return  Im[i]*Im[j] / 2

def a2m_factors(r_vec):
    rhat = r_vec / np.linalg.norm(r_vec)
    A = 3.0 * np.outer(rhat, rhat) - np.eye(3)
    return {
         0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
    }

PAIRS = [(0, 1, b_FC, r_FC), (0, 2, b_FH, r_FH), (1, 2, b_CH, r_CH)]
alm = {(i, j): a2m_factors(r) for i, j, _, r in PAIRS}

Q_ops = {
    (k, m): sum(b * alm[(i,j)][m] * T2_op(i, j, k) for i, j, b, _ in PAIRS)
    for k in range(-2, 3) for m in range(-2, 3)
}

# =============================================================================
# STRUCTURE MATRIX AND GAMMA-BAR RATE MATRICES
#
# At B=0: A^{(2)} = Q_hat (dipolar only), A^{(1)} = 0
# Gamma_bar_ij(omega_j) = 2 * C_ij * J^{(2)}(omega_j) * gamma_scale
# where J^{(2)}(omega) = tau_c / (5 * (1 + (omega*tau_c)^2))
#
# Gamma_bar[i,j] = gamma_scale * C[i,j] * J(omega_j)
# Matches zulf_lindblad.py build_liouvillian(gamma_scale=2.0), which is the
# Spinach-matched rate.  The single factor gamma_scale=2.0 absorbs the
# Spinach normalisation convention; no additional 2 is needed here.
# =============================================================================

gamma_scale = 1.0

omega_trans = np.array([evals[m_] - evals[n_]
                        for n_ in range(n_states)
                        for m_ in range(n_states)])

KM_LIST = [(k, m) for k in range(-2, 3) for m in range(-2, 3)]
Q_stack = np.array([Q_ops[km].full() for km in KM_LIST], dtype=complex)
M_3d    = np.einsum('nk,akl,ml->anm', ekets_arr.conj(), Q_stack, ekets_arr)
M_mat   = M_3d.reshape(25, n_trans)
C_mat   = M_mat.conj().T @ M_mat

def J_l2(omega):
    return tau_c / (5.0 * (1.0 + (omega * tau_c)**2))

J_omegas = np.vectorize(J_l2)(omega_trans)

Gamma_bar_j = gamma_scale * C_mat * J_omegas[np.newaxis, :]
Gamma_bar_i = gamma_scale * C_mat * J_omegas[:, np.newaxis]

# =============================================================================
# LIOUVILLIAN (eigenstate basis, then rotate to computational basis)
# =============================================================================

L_ham = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

GR  = Gamma_bar_j.reshape(n_states, n_states, n_states, n_states)
T1  = np.einsum('amcm->ac', GR)
T4  = np.einsum('nbnd->bd', GR)
L_4d = (- np.einsum('ac,bd->abcd', T1, np.eye(n_states))
        + np.einsum('acbd->abcd', GR)
        + np.einsum('dbca->abcd', GR)
        - np.einsum('ac,bd->abcd', np.eye(n_states), T4))
L_eig = L_4d.transpose(1, 0, 3, 2).reshape(n_trans, n_trans)

U = ekets_arr.T
V = np.kron(U.conj(), U)
L_mat = L_ham + V @ L_eig @ V.conj().T

ev, evec    = np.linalg.eig(L_mat)
evec_inv    = np.linalg.inv(evec)

# =============================================================================
# INITIAL STATE AND TIME PROPAGATION
# =============================================================================

_up  = qt.basis(2, 0)
rho0 = qt.ket2dm(qt.tensor(_up, _up, _up))
rho0_vec  = qt.operator_to_vector(rho0)
vec_dims  = rho0_vec.dims
vec_shape = rho0_vec.shape
c0 = evec_inv @ rho0_vec.full().flatten()

t_end   = 1e-4
N_steps = 500
tlist   = np.linspace(0.0, t_end, N_steps)

pops_py  = np.zeros((N_steps, n_states))
min_eig  = np.zeros(N_steps)

print("\nPropagating Python Redfield at B=0 ...")
for ti, t in enumerate(tlist):
    v       = evec @ (np.exp(ev * t) * c0)
    rho_mat = qt.vector_to_operator(
        qt.Qobj(v.reshape(vec_shape), dims=vec_dims)).full()
    for n in range(n_states):
        pops_py[ti, n] = np.real(ekets_arr[n].conj() @ rho_mat @ ekets_arr[n])
    min_eig[ti] = np.linalg.eigvalsh(rho_mat).min()

print(f"  Trace at t=0: {pops_py[0].sum():.8f}  at t_end: {pops_py[-1].sum():.8f}")
print(f"  min eigenvalue range: [{min_eig.min():.4e}, {min_eig.max():.4e}]")

# =============================================================================
# LOAD SPINACH DATA
# =============================================================================

if not os.path.exists(SPINACH_FILE):
    raise FileNotFoundError(
        f"Spinach reference file not found:\n  {SPINACH_FILE}\n"
        "Run fch_triple_redfield(1e-5, 0.0, 'none') in MATLAB first."
    )

d_sp     = scipy.io.loadmat(SPINACH_FILE)
t_sp     = d_sp['t_axis'].ravel()
pops_sp  = d_sp['populations']                            # (8, n_steps_sp)
sp_lbls  = [str(d_sp['state_labels'][0, n][0]) for n in range(8)]
evals_sp = d_sp['evals_hilb'].ravel()

print("\nSpinach eigenstates:")
for i, lbl in enumerate(sp_lbls):
    print(f"  [{i}]  {lbl}  E={evals_sp[i]/(2*np.pi):+.2f} Hz  P(t=0)={pops_sp[i,0]:.4f}")

# =============================================================================
# STATE MATCHING: Spinach index → Python index
#
# Parse Spinach label 'S=3/2, M=+3/2'  → (S=1.5, M=+1.5, rank=None)
#       'S=1/2(1), M=-1/2'             → (S=0.5, M=-0.5, rank=0)  [0-indexed]
# Then match Python state by (S_qn, M_qn, doublet_rank).
# =============================================================================

def parse_spinach_label(lbl):
    """Return (S, M, doublet_rank_0indexed_or_None)."""
    m_frac = re.search(r'M=([+-])(\d+)/2', lbl)
    sign   = 1 if m_frac.group(1) == '+' else -1
    M      = sign * int(m_frac.group(2)) / 2.0
    if 'S=3/2' in lbl:
        return 1.5, M, None
    mi = re.search(r'S=1/2\((\d+)\)', lbl)
    return 0.5, M, int(mi.group(1)) - 1   # convert to 0-indexed

sp2py = {}   # spinach_idx → python_idx
unmatched_sp = []

for si, lbl in enumerate(sp_lbls):
    S_sp, M_sp, rank_sp = parse_spinach_label(lbl)
    found = False
    for ni in range(n_states):
        if abs(S_qn[ni] - S_sp) > 0.1:
            continue
        if abs(M_qn[ni] - M_sp) > 0.1:
            continue
        if S_sp < 1.0 and doublet_rank(ni) != rank_sp:
            continue
        sp2py[si] = ni
        found = True
        break
    if not found:
        unmatched_sp.append((si, lbl))

print("\nState matching (Spinach → Python):")
for si, ni in sp2py.items():
    print(f"  Spinach[{si}] {sp_lbls[si]:30s} → Python |{ni}>"
          f"  S={S_qn[ni]:.1f} M={M_qn[ni]:+.1f} E={E_Hz[ni]:+.2f} Hz")
if unmatched_sp:
    print("  UNMATCHED Spinach states:")
    for si, lbl in unmatched_sp:
        print(f"    [{si}] {lbl}")

# =============================================================================
# COMPARISON PLOT
# =============================================================================

colors = plt.cm.tab10(np.linspace(0, 1, n_states))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P']
mk_every = max(1, len(t_sp) // 20)

# readable Spinach label (replace (1)→α, (2)→β)
def clean_lbl(lbl):
    return (lbl.replace('S=3/2', 'S=3/2')
               .replace('(1)', 'α').replace('(2)', 'β').replace('(3)', 'γ'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    r'ZULF ($B_0 = 0$): Redfield vs Spinach — $^{19}$F–$^{13}$C–$^{1}$H'
    + f'\n$\\tau_c = {tau_c:.0e}$ s,  initial state '
    + r'$|\!\uparrow\uparrow\uparrow\rangle$',
    fontsize=12
)

line_handles = []
for si in range(8):
    ni = sp2py.get(si)
    if ni is None:
        continue
    clr = colors[ni]
    mk  = markers[si % len(markers)]
    lbl = clean_lbl(sp_lbls[si])

    # Python: continuous line
    h_py, = ax1.plot(tlist * 1e6, pops_py[:, ni],
                     color=clr, lw=2.0, label=lbl)
    # Spinach: same color, markers only
    ax1.plot(t_sp * 1e6, pops_sp[si, :],
             color=clr, ls='none',
             marker=mk, markersize=5, markevery=mk_every, alpha=0.85)
    line_handles.append(h_py)

ax1.set_ylabel('Population', fontsize=11)
ax1.set_ylim(-0.06, 1.06)
ax1.axhline(1.0 / n_states, color='gray', ls=':', lw=0.8, alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(handles=line_handles, fontsize=8, ncol=2, loc='upper right',
           title='solid = Redfield,  markers = Spinach')
ax1.set_title('Eigenbasis populations', fontsize=10)

# Residuals: interpolate Spinach onto Python time grid
for si in range(8):
    ni = sp2py.get(si)
    if ni is None:
        continue
    sp_interp = interp1d(t_sp, pops_sp[si, :], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    delta = pops_py[:, ni] - sp_interp(tlist)
    ax2.plot(tlist * 1e6, delta,
             color=colors[ni], lw=1.5, label=clean_lbl(sp_lbls[si]))

ax2.axhline(0.0, color='k', ls='--', lw=0.8)
ax2.set_ylabel(r'$\Delta P_n$ (Redfield $-$ Spinach)', fontsize=11)
ax2.set_xlabel(r'Time ($\mu$s)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7, ncol=2, loc='upper right')
ax2.set_title('Residuals', fontsize=10)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, f'compare_zulf_B0_tc{tau_c:.0e}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nComparison plot saved -> {out_path}')
