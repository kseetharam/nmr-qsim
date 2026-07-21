"""
Compare population dynamics at B = (0, 0, 1e-3) T:
  - Generalized Redfield (dipolar + CSA dephasing)  → continuous lines
  - Spinach reference at the same field              → markers

Analogous to compare_zulf.py (B=0 case), but with:
  - H0 including isotropic Zeeman shift
  - A^{(1)} = Q^{(Z),1}  and  A^{(2)} = Q^{(Z),2} + Q_hat operators
  - Rank-dependent spectral density J^{(l)}(omega) for l=1,2
  - gamma_scale = 1.0  (confirmed correct by B=0 comparison)

At 1 mT the Zeeman splitting (~40 kHz for 19F) dominates the J-couplings
(~243 Hz), so the eigenstates are close to Zeeman product states and S_total
is no longer a good quantum number.  State matching uses M_total + energy
proximity within each M sector.
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
SPINACH_FILE = os.path.join(DATA_DIR, 'spinach_pops_tc1e-05_B1.00e-03.mat')

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

# Zeeman shielding tensors (ppm, DFT)
sigma_ppm = {
    '19F': np.array([[-113.0955, -13.4634,  23.3880],
                     [ -13.4634, -32.0922, -17.6116],
                     [  23.3880, -17.6116,-153.6685]]),
    '13C': np.array([[240.8318, 25.3402, 53.6419],
                     [ 25.3402,162.9404,  5.1940],
                     [ 53.6419,  5.1940,127.8480]]),
    '1H':  np.array([[ 5.0782,  2.5669, -1.6492],
                     [ 2.5669, 10.7485,  0.0000],
                     [-1.6492,  0.0000, 11.9705]]),
}
sigma_tilde = {X: np.eye(3) + 1e-6 * sigma_ppm[X] for X in NUCLEI}
sigma_iso   = {X: np.trace(sigma_tilde[X]) / 3.0 for X in NUCLEI}

# External field
B_vec = np.array([0.0, 0.0, 1e-3])   # T
B0    = np.linalg.norm(B_vec)

# =============================================================================
# SPIN OPERATORS (8-dim Hilbert space)
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

def S_sph(idx):
    return {0: Iz[idx], +1: -Ip[idx]/np.sqrt(2), -1: Im[idx]/np.sqrt(2)}

def heis(i, j):
    return Ix[i]*Ix[j] + Iy[i]*Iy[j] + Iz[i]*Iz[j]

# H0 = J-couplings + isotropic Zeeman shift
H_J    = J_FC_rad * heis(0, 1) + J_CH_rad * heis(1, 2)
H_Z_iso = sum(-GAMMA[X] * sigma_iso[X] *
              (B_vec[0]*Ix[idx] + B_vec[1]*Iy[idx] + B_vec[2]*Iz[idx])
              for idx, X in enumerate(NUCLEI))
H0 = H_J + H_Z_iso

evals, ekets = H0.eigenstates()
n_states  = len(evals)
n_trans   = n_states ** 2
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)

# Quantum numbers (approximate at finite field)
_M_op  = Iz[0] + Iz[1] + Iz[2]
_S2_op = ((Ix[0]+Ix[1]+Ix[2])**2
         + (Iy[0]+Iy[1]+Iy[2])**2
         + (Iz[0]+Iz[1]+Iz[2])**2)
M_qn = np.round(2 * np.real([qt.expect(_M_op, ek) for ek in ekets])) / 2
S_qn = (np.round(2*(np.sqrt(np.clip(
            4*np.real([qt.expect(_S2_op, ek) for ek in ekets])+1, 0, None))-1)/2))/2
E_Hz = evals / (2 * np.pi)

print(f"B_vec = {B_vec} T   (|B| = {B0:.3f} T)")
print(f"Larmor frequencies at 1 mT (approx, rad/s): "
      f"19F={GAMMA['19F']*B0:.0f}, 13C={GAMMA['13C']*B0:.0f}, 1H={GAMMA['1H']*B0:.0f}")
print()
print("Python eigenstates (ascending energy):")
for n in range(n_states):
    print(f"  |{n}>  S≈{S_qn[n]:.1f}  M={M_qn[n]:+.1f}  E={E_Hz[n]:+.2f} Hz")

# =============================================================================
# CG COEFFICIENTS FOR 1 ⊗ 1 → l
# =============================================================================

_rt2 = np.sqrt(2.0); _rt3 = np.sqrt(3.0); _rt6 = np.sqrt(6.0)
_CG = {
    (0,  1,-1):  1.0/_rt3, (0,  0, 0): -1.0/_rt3, (0, -1, 1):  1.0/_rt3,
    (1,  1, 0):  1.0/_rt2, (1,  0, 1): -1.0/_rt2,
    (1,  1,-1):  1.0/_rt2, (1,  0, 0):  0.0,
    (1, -1, 1): -1.0/_rt2, (1,  0,-1):  1.0/_rt2, (1, -1, 0): -1.0/_rt2,
    (2,  1, 1):  1.0,      (2,  1, 0):  1.0/_rt2, (2,  0, 1):  1.0/_rt2,
    (2,  1,-1):  1.0/_rt6, (2,  0, 0):  2.0/_rt6, (2, -1, 1):  1.0/_rt6,
    (2,  0,-1):  1.0/_rt2, (2, -1, 0):  1.0/_rt2, (2, -1,-1):  1.0,
}

def cg1x1(l, m1, m2):
    return _CG.get((l, m1, m2), 0.0)

def B_spherical(B):
    Bx, By, Bz = B
    return {0: Bz+0j, +1: -(Bx+1j*By)/np.sqrt(2), -1: (Bx-1j*By)/np.sqrt(2)}

B_sph = B_spherical(B_vec)

def T_lk_op(l, k, spin_idx, B=None):
    if B is None: B = B_sph
    Ss = S_sph(spin_idx)
    result = None
    for q1 in (-1, 0, 1):
        q2 = k - q1
        if q2 not in (-1, 0, 1): continue
        c = cg1x1(l, q1, q2)
        if abs(c) < 1e-15: continue
        term = c * B[q2] * Ss[q1]
        result = term if result is None else result + term
    return 0.0 * Iz[0] if result is None else result

# =============================================================================
# ZEEMAN SHIELDING DECOMPOSITION  sigma_{l,m}(X)
# =============================================================================

def sigma_lm(sigma_t):
    s = sigma_t
    s_iso = (s[0,0] + s[1,1] + s[2,2]) / 3.0
    return {
        (0,  0): -np.sqrt(3) * s_iso,
        (1,  0): -1j/np.sqrt(2) * (s[0,1] - s[1,0]),
        (1, +1): -0.5 * ((s[2,0]-s[0,2]) + 1j*(s[2,1]-s[1,2])),
        (1, -1): -0.5 * ((s[2,0]-s[0,2]) - 1j*(s[2,1]-s[1,2])),
        (2,  0):  np.sqrt(2.0/3.0) * (s[2,2] - s_iso),
        (2, +1): -0.5 * ((s[0,2]+s[2,0]) + 1j*(s[1,2]+s[2,1])),
        (2, -1): +0.5 * ((s[0,2]+s[2,0]) - 1j*(s[1,2]+s[2,1])),
        (2, +2):  0.5 * ((s[0,0]-s[1,1]) + 1j*(s[0,1]+s[1,0])),
        (2, -2):  0.5 * ((s[0,0]-s[1,1]) - 1j*(s[0,1]+s[1,0])),
    }

sigma_lm_all = {X: sigma_lm(sigma_tilde[X]) for X in NUCLEI}

# =============================================================================
# ZEEMAN CSA OPERATORS  Q^{(Z),l}_{k,m}
# =============================================================================

def build_QZ_ops(l):
    ops = {}
    for k in range(-l, l+1):
        for m in range(-l, l+1):
            phase = (-1)**(m + 1)
            op = None
            for idx, X in enumerate(NUCLEI):
                slm  = sigma_lm_all[X][(l, m)]
                T_nk = T_lk_op(l, -k, idx)
                contrib = GAMMA[X] * phase * slm * T_nk
                op = contrib if op is None else op + contrib
            ops[(k, m)] = op
    return ops

QZ1_ops = build_QZ_ops(l=1)
QZ2_ops = build_QZ_ops(l=2)

# =============================================================================
# DIPOLAR OPERATORS  Q_hat_{k,m}
# =============================================================================

def T2_op(i, j, k):
    if k ==  0: return (2*Iz[i]*Iz[j] - Ix[i]*Ix[j] - Iy[i]*Iy[j]) / np.sqrt(6)
    if k ==  1: return -(Ip[i]*Iz[j] + Iz[i]*Ip[j]) / 2
    if k == -1: return  (Im[i]*Iz[j] + Iz[i]*Im[j]) / 2
    if k ==  2: return  Ip[i]*Ip[j] / 2
    if k == -2: return  Im[i]*Im[j] / 2

def b_dip(gi, gj, r_vec):
    return -(mu0/(4*np.pi)) * gi * gj * hbar / np.linalg.norm(r_vec)**3

b_FC = b_dip(GAMMA['19F'], GAMMA['13C'], r_FC)
b_FH = b_dip(GAMMA['19F'], GAMMA['1H'],  r_FH)
b_CH = b_dip(GAMMA['13C'], GAMMA['1H'],  r_CH)

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
alm   = {(i, j): a2m_factors(r) for i, j, _, r in PAIRS}

Qdip_ops = {
    (k, m): sum(b * alm[(i,j)][m] * T2_op(i, j, k) for i, j, b, _ in PAIRS)
    for k in range(-2, 3) for m in range(-2, 3)
}

# =============================================================================
# COMBINED A^{(l)} OPERATORS
# A^{(1)} = Q^{(Z),1}          (l=1, 9 modes)
# A^{(2)} = Q^{(Z),2} + Q_hat  (l=2, 25 modes)
# =============================================================================

A1_ops = QZ1_ops
A2_ops = {(k, m): QZ2_ops[(k,m)] + Qdip_ops[(k,m)]
          for k in range(-2, 3) for m in range(-2, 3)}

# =============================================================================
# STRUCTURE MATRICES AND GAMMA-BAR RATE MATRIX
#
# Gamma_bar[i,j](omega_j) = C1[i,j]*J1(omega_j) + C2[i,j]*J2(omega_j)
# gamma_scale = 1.0  (confirmed correct by B=0 comparison with Spinach)
# =============================================================================

gamma_scale = 1.0

omega_trans = np.array([evals[m_] - evals[n_]
                        for n_ in range(n_states)
                        for m_ in range(n_states)])

def build_M_matrix(ops_dict, km_list):
    Q_stack = np.array([ops_dict[km].full() for km in km_list], dtype=complex)
    M_3d    = np.einsum('nk,akl,ml->anm', ekets_arr.conj(), Q_stack, ekets_arr)
    return M_3d.reshape(len(km_list), n_trans)

KM1_LIST = [(k, m) for k in range(-1, 2) for m in range(-1, 2)]
KM2_LIST = [(k, m) for k in range(-2, 3) for m in range(-2, 3)]

M1_mat = build_M_matrix(A1_ops, KM1_LIST)   # (9, 64)
M2_mat = build_M_matrix(A2_ops, KM2_LIST)   # (25, 64)

C1_matrix = M1_mat.conj().T @ M1_mat   # (64, 64)
C2_matrix = M2_mat.conj().T @ M2_mat   # (64, 64)

def J_l(l, omega):
    return tau_c / ((2*l + 1) * (1.0 + (omega * tau_c)**2))

J1_omegas = np.vectorize(lambda w: J_l(1, w))(omega_trans)
J2_omegas = np.vectorize(lambda w: J_l(2, w))(omega_trans)

Gamma_bar_j = gamma_scale * (C1_matrix * J1_omegas[np.newaxis, :]
                            + C2_matrix * J2_omegas[np.newaxis, :])

# =============================================================================
# LIOUVILLIAN
# =============================================================================

L_ham = (-1j * (qt.spre(H0) - qt.spost(H0))).full()

GR   = Gamma_bar_j.reshape(n_states, n_states, n_states, n_states)
T1   = np.einsum('amcm->ac', GR)
T4   = np.einsum('nbnd->bd', GR)
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
# INITIAL STATE AND PROPAGATION
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

pops_py = np.zeros((N_steps, n_states))
min_eig = np.zeros(N_steps)

print("\nPropagating Python Redfield at B = 1 mT ...")
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
        "Run fch_triple_redfield(1e-5, 1e-3, 'none') in MATLAB first."
    )

d_sp     = scipy.io.loadmat(SPINACH_FILE)
t_sp     = d_sp['t_axis'].ravel()
pops_sp  = d_sp['populations']                     # (8, n_steps_sp)
sp_lbls  = [str(d_sp['state_labels'][0, n][0]) for n in range(8)]
evals_sp = d_sp['evals_hilb'].ravel()

print("\nSpinach eigenstates:")
for i, lbl in enumerate(sp_lbls):
    print(f"  [{i}]  {lbl:35s}  E={evals_sp[i]/(2*np.pi):+.2f} Hz  P(t=0)={pops_sp[i,0]:.4f}")

# =============================================================================
# STATE MATCHING
#
# Strategy:
#  1. Parse M_total from Spinach label (works for both B=0 and B≠0 labels).
#  2. Within states that share the same M_total, match by closest energy.
#
# Label formats Spinach may use:
#   "S=3/2, M=+3/2"      → M = +3/2
#   "S=1/2(1), M=+1/2"   → M = +1/2
#   "M=+1/2"             → M = +1/2  (high-field product-state label)
# =============================================================================

def parse_spinach_M(lbl):
    """Extract total M quantum number from Spinach label."""
    m = re.search(r'M=([+-])(\d+)/(\d+)', lbl)
    if m:
        sign = 1 if m.group(1) == '+' else -1
        return sign * int(m.group(2)) / int(m.group(3))
    # fallback: half-integer or integer M without fraction
    m2 = re.search(r'M=([+-]?\d+\.?\d*)', lbl)
    if m2:
        return float(m2.group(1))
    return None

sp2py = {}
unmatched_sp = []

for si, lbl in enumerate(sp_lbls):
    M_sp = parse_spinach_M(lbl)
    E_sp_hz = evals_sp[si] / (2 * np.pi)

    # Candidates: Python states with matching M_total
    if M_sp is not None:
        candidates = [ni for ni in range(n_states)
                      if abs(M_qn[ni] - M_sp) < 0.1
                      and ni not in sp2py.values()]
    else:
        candidates = [ni for ni in range(n_states)
                      if ni not in sp2py.values()]

    if not candidates:
        unmatched_sp.append((si, lbl))
        continue

    # Among candidates, pick closest energy
    best = min(candidates, key=lambda ni: abs(E_Hz[ni] - E_sp_hz))
    sp2py[si] = best

print("\nState matching (Spinach → Python):")
for si, ni in sp2py.items():
    print(f"  Spinach[{si}] {sp_lbls[si]:35s} → Python |{ni}>"
          f"  M={M_qn[ni]:+.1f}  E={E_Hz[ni]:+.2f} Hz")
if unmatched_sp:
    print("  UNMATCHED Spinach states:")
    for si, lbl in unmatched_sp:
        print(f"    [{si}] {lbl}")

# =============================================================================
# COMPARISON PLOT
# =============================================================================

colors   = plt.cm.tab10(np.linspace(0, 1, n_states))
markers  = ['o', 's', '^', 'D', 'v', '<', '>', 'P']
mk_every = max(1, len(t_sp) // 20)

def clean_lbl(lbl):
    return (lbl.replace('S=3/2', 'S=3/2')
               .replace('(1)', 'α').replace('(2)', 'β').replace('(3)', 'γ'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    r'$B_z = 1\,\mathrm{mT}$: Redfield (dipolar+CSA) vs Spinach — '
    r'$^{19}$F–$^{13}$C–$^{1}$H'
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

    h_py, = ax1.plot(tlist * 1e6, pops_py[:, ni],
                     color=clr, lw=2.0, label=lbl)
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
out_path = os.path.join(DATA_DIR, f'compare_zeeman_B{B0:.2e}T_tc{tau_c:.0e}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nComparison plot saved -> {out_path}')
