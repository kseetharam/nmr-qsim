"""
Compute Krylov-dressed Lindblad jump operators for the 5-spin gemcitabine
subsystem (F0, F1, H2, H3, C0) at truncation orders 0 through 6.

For each collective operator A^{(l)}_{k,m}, the Krylov chain is computed ONCE
up to order 6, and all partial sums (orders 0,1,...,6) are derived from that
single chain to avoid redundant commutator evaluations.

Spin mapping from the full 10-spin ordering [F0,F1,H0,H1,H2,H3,H4,H5,H6,C0]:
    F0 → 0,  F1 → 1,  H2 → 4,  H3 → 5,  C0 → 9

Physical parameters: B = (0, 0, 5e-7) T  (ZULF),  tau_c = 1e-10 s
    omega_F * tau_c ≈ 1.26e-8  (deep extreme-narrowing)

Output: data/gemcitabine_5spin_krylov_ops.pkl
    {
      'metadata': {...},
      'H_iso':    (D, D) complex ndarray       [rad/s]
      'evals':    (D,) float ndarray            [rad/s]
      'ekets':    (D, D) complex ndarray        rows = eigenstates
      'L1_by_order': {order: {(k,m): (D,D) ndarray}}   rank-1
      'L2_by_order': {order: {(k,m): (D,D) ndarray}}   rank-2
    }
"""

import os, sys, pickle
import numpy as np
from math import factorial
from scipy.special import gamma as _gamma_func

_HERE  = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(_HERE, '..', 'linblad_dyn', 'utils'))
sys.path.insert(0, _UTILS)
from linblad_utils import build_spin_ops, build_H_iso, build_QZ_ops, build_Qdip_ops

# =============================================================================
# Full 10-spin Gemcitabine parameters  (gemcitabine_jump_operators.py)
# =============================================================================

GAMMA_RAD = {'19F': 251.81520e6, '1H': 267.52218e6, '13C': 67.28284e6}  # rad/s/T

_gammas_full = np.array([
    GAMMA_RAD['19F'], GAMMA_RAD['19F'],
    GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],
    GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],
    GAMMA_RAD['13C'],
])

_N = 10
_J_full = np.zeros((_N, _N))
for _i, _j, _v in [
    (0,1,175.91),(0,2, 0.00),(0,3, 0.11),(0,4, 0.05),(0,5, 6.66),
    (0,6,-0.32), (0,7, 0.05),(0,8,11.59),(0,9,-226.85),
    (1,2, 0.09), (1,3,-0.20),(1,4, 0.28),(1,5, 5.91),(1,6, 0.08),
    (1,7,-0.09), (1,8, 0.48),(1,9,-203.41),
    (2,3,-3.66), (2,4, 0.73),(2,5,-0.14),(2,6,-0.10),(2,7,-0.11),
    (2,8, 0.18), (2,9,-0.13),
    (3,4, 2.88), (3,5,-0.12),(3,6,-0.10),(3,7,-0.10),(3,8, 0.38),(3,9,-0.10),
    (4,5, 0.22), (4,6, 0.02),(4,7,-0.09),(4,8, 2.12),(4,9, 0.31),
    (5,6, 0.38), (5,7,-0.07),(5,8,-0.37),(5,9,-0.65),
    (6,7, 2.67), (6,8,-0.15),(6,9, 0.02),
    (7,8,-0.14), (7,9,-0.08),
    (8,9,-2.68),
]:
    _J_full[_i, _j] = _J_full[_j, _i] = _v

_coords_full = np.array([
    [ 0.1666, -1.2783, -1.3875],   # F0
    [ 0.9661, -2.5748,  0.3352],   # F1
    [ 4.2344,  1.2314, -0.5006],   # H0
    [ 2.7307,  1.8998, -1.1779],   # H1
    [ 2.8928,  0.2529,  1.3947],   # H2
    [ 0.4474, -0.6659,  1.8321],   # H3
    [-1.7462, -0.8907,  2.3837],   # H4
    [-4.1117, -0.4940,  1.8959],   # H5
    [ 2.4649, -0.4748, -1.5086],   # H6
    [ 0.9133, -1.2851, -0.2046],   # C0
])

_sigma_full = [
    np.array([[-130.0601, -14.9649, -63.1405],
              [-14.9649, -215.3146,  25.9348],
              [-63.1405,  25.9348, -160.2068]]),   # F0
    np.array([[-157.2094, -12.7347, -67.3755],
              [-12.7347, -235.8144,   5.7085],
              [-67.3755,   5.7085, -228.8689]]),   # F1
    np.array([[2.2651, -0.6354,  2.4859],
              [-0.6354,  2.7073, -2.7546],
              [2.4859, -2.7546,  6.9013]]),         # H0
    np.array([[7.0348, -2.4557, -1.2496],
              [-2.4557,  3.6480, -0.3257],
              [-1.2496, -0.3257,  1.9779]]),         # H1
    np.array([[1.5527,  0.6046, -2.3124],
              [0.6046,  5.3322,  0.1728],
              [-2.3124,  0.1728,  4.7300]]),         # H2
    np.array([[4.3528, -1.0174,  1.5080],
              [-1.0174,  5.8782,  1.7827],
              [1.5080,  1.7827,  3.5154]]),          # H3
    np.array([[3.1215,  0.1121,  1.7281],
              [0.1121,  9.5089,  0.5658],
              [1.7281,  0.5658,  7.4617]]),          # H4
    np.array([[2.5152,  0.8210, -0.8704],
              [0.8210,  6.6786,  1.2727],
              [-0.8704,  1.2727,  4.7920]]),         # H5
    np.array([[4.4701,  0.6055,  0.2616],
              [0.6055,  4.4715,  1.0493],
              [0.2616,  1.0493,  2.7238]]),          # H6
    np.array([[241.3681, -1.4291, -6.1424],
              [-1.4291, 245.7345,  3.0563],
              [-6.1424,  3.0563, 250.0215]]),        # C0
]

# =============================================================================
# Extract 5-spin subsystem
# =============================================================================

IDX   = [0, 1, 4, 5, 9]          # F0, F1, H2, H3, C0 in full ordering
LBLS  = ['F0', 'F1', 'H2', 'H3', 'C0']

gammas  = _gammas_full[IDX]
J_hz    = _J_full[np.ix_(IDX, IDX)]
coords  = [_coords_full[i].tolist() for i in IDX]
sigma   = [_sigma_full[i] for i in IDX]

B_vec = np.array([0.0, 0.0, 5e-7])   # T
tau_c = 1e-10                          # s

n = len(gammas)
D = 2 ** n

print(f"5-spin subsystem: {LBLS}")
print(f"tau_c = {tau_c:.1e} s,  B0 = {B_vec[2]*1e6:.2f} µT")
print(f"omega_F * tau_c = {GAMMA_RAD['19F'] * abs(B_vec[2]) * tau_c:.2e}  "
      f"(deep extreme-narrowing)")
print(f"Hilbert-space dimension D = {D}")
print()

# =============================================================================
# Build base operators (done once)
# =============================================================================

print("Building H_iso and collective operators ...")
ops  = build_spin_ops(n)
H_iso, ops, evals, ekets = build_H_iso(gammas, J_hz, coords, sigma, B_vec, ops)
QZ1  = build_QZ_ops(1, gammas, sigma, B_vec, ops)
QZ2  = build_QZ_ops(2, gammas, sigma, B_vec, ops)
Qdip = build_Qdip_ops(gammas, coords, ops)
print("  done")
print()

H0_mat = H_iso.full()
pref   = {l: np.sqrt(8.0 * tau_c / (np.pi**2 * (2*l + 1))) for l in (1, 2)}

def _krylov_coeff(j):
    return (1.0 / factorial(j)) * 2.0**(j - 1) * _gamma_func((j + 1) / 2.0)**2

# =============================================================================
# Single-pass Krylov chain: compute partial sums for orders 0 … MAX_ORDER
# =============================================================================

MAX_ORDER = 6
ORDERS    = list(range(MAX_ORDER + 1))


def krylov_partial_sums(A_mat, H0, tc, max_ord):
    """
    Given bare operator A_mat (D×D) and H0, compute
        Abar_n = sum_{j=0}^{n} c_j * (i*tc*[H0, .])^j A
    for n = 0, 1, ..., max_ord  in a single forward pass.

    Returns dict {n: Abar_n (D×D complex ndarray)}.
    """
    Aj   = A_mat.astype(complex, copy=True)
    Abar = np.zeros((D, D), dtype=complex)
    sums = {}
    for j in range(max_ord + 1):
        Abar = Abar + _krylov_coeff(j) * Aj
        sums[j] = Abar.copy()
        if j < max_ord:
            Aj = 1j * tc * (H0 @ Aj - Aj @ H0)
    return sums


print("Computing Krylov partial sums for L1 operators (rank-1) ...")
L1_chains = {}   # {(k,m): {order: ndarray}}
for k in range(-1, 2):
    for m in range(-1, 2):
        ps = krylov_partial_sums(QZ1[(k, m)].full(), H0_mat, tau_c, MAX_ORDER)
        L1_chains[(k, m)] = {order: pref[1] * ps[order] for order in ORDERS}
print(f"  {len(L1_chains)} operators × {len(ORDERS)} orders")

print("Computing Krylov partial sums for L2 operators (rank-2) ...")
L2_chains = {}   # {(k,m): {order: ndarray}}
for k in range(-2, 3):
    for m in range(-2, 3):
        A_km = (QZ2[(k, m)] + Qdip[(k, m)]).full()
        ps   = krylov_partial_sums(A_km, H0_mat, tau_c, MAX_ORDER)
        L2_chains[(k, m)] = {order: pref[2] * ps[order] for order in ORDERS}
print(f"  {len(L2_chains)} operators × {len(ORDERS)} orders")
print()

# Rearrange to order-first indexing for easy downstream use
L1_by_order = {order: {km: L1_chains[km][order] for km in L1_chains}
               for order in ORDERS}
L2_by_order = {order: {km: L2_chains[km][order] for km in L2_chains}
               for order in ORDERS}

# =============================================================================
# Correction norms (relative to order 0)
# =============================================================================

print("Relative L2 corrections vs order 0:")
for order in ORDERS[1:]:
    deltas = []
    for km in L2_chains:
        F0  = np.linalg.norm(L2_chains[km][0], 'fro')
        dF  = np.linalg.norm(L2_chains[km][order] - L2_chains[km][0], 'fro')
        if F0 > 1e-30:
            deltas.append(dF / F0)
    print(f"  order {order}:  max = {max(deltas):.3e},  mean = {np.mean(deltas):.3e}")

print()

# =============================================================================
# Save
# =============================================================================

result = {
    'metadata': {
        'system':            f'gemcitabine 5-spin ({", ".join(LBLS)})',
        'spin_labels':       LBLS,
        'full_system_indices': IDX,
        'B_vec_T':           B_vec.tolist(),
        'tau_c_s':           tau_c,
        'omega_F_tau_c':     float(GAMMA_RAD['19F'] * abs(B_vec[2]) * tau_c),
        'krylov_orders':     ORDERS,
        'D':                 D,
        'L1_keys':           list(L1_chains.keys()),
        'L2_keys':           list(L2_chains.keys()),
        'units':             {'H_iso': 'rad/s', 'L': 'sqrt(rad/s)'},
        'note':              'L{n}_by_order[n][(k,m)] = rank-n jump op at Krylov order n',
    },
    'H_iso':       H0_mat,
    'evals':       evals,
    'ekets':       np.array([ek.full().flatten() for ek in ekets], dtype=complex),
    'L1_by_order': L1_by_order,
    'L2_by_order': L2_by_order,
}

DATA_DIR = os.path.join(_HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
out_path = os.path.join(DATA_DIR, 'gemcitabine_5spin_krylov_ops.pkl')

with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

file_mb = os.path.getsize(out_path) / 1e6
print(f"Saved → {out_path}")
print(f"  File size: {file_mb:.1f} MB")
print(f"  D = {D},  L1 ops: {len(L1_chains)},  L2 ops: {len(L2_chains)},  "
      f"orders: 0–{MAX_ORDER}")
