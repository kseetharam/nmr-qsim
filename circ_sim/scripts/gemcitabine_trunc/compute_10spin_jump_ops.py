"""
Compute Krylov-dressed Lindblad jump operators for the full 10-spin gemcitabine
system (F0, F1, H0–H6, C0) at truncation orders 0 through 6.

For each collective operator A^{(l)}_{k,m}, the Krylov chain is computed ONCE
up to order 6, and all partial sums (orders 0,1,...,6) are derived from that
single chain.

Physical parameters: B = (0, 0, 5e-7) T  (ZULF),  tau_c = 1e-10 s
    omega_F * tau_c ≈ 1.26e-8  (deep extreme-narrowing)

Hilbert-space dimension D = 1024.

Output files: data/gemcitabine_10spin_krylov_ops_order{n}.pkl  for n in 0..6
  Each file:
    {
      'metadata':  {...},
      'H_iso':     (1024, 1024) complex ndarray   [rad/s]   (order-0 only)
      'evals':     (1024,) float ndarray           [rad/s]   (order-0 only)
      'ekets':     (1024, 1024) complex ndarray              (order-0 only)
      'L1':        {(k,m): (1024,1024) ndarray}   rank-1 ops at this order
      'L2':        {(k,m): (1024,1024) ndarray}   rank-2 ops at this order
    }
  (H_iso/evals/ekets are identical for all orders; included only in order-0 file
   and set to None in higher-order files to avoid duplication.)

Estimated file size: ~560 MB per order  (34 ops × 16 MB each).
Estimated runtime:   3–8 minutes on a modern laptop (34 × 6 BLAS matmuls of
                     1024×1024 complex128, then 34 × 7 partial sums).
"""

import os, sys, pickle, time
import numpy as np
from math import factorial
from scipy.special import gamma as _gamma_func

_HERE  = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(_HERE, '..', 'linblad_dyn', 'utils'))
sys.path.insert(0, _UTILS)
from linblad_utils import build_spin_ops, build_H_iso, build_QZ_ops, build_Qdip_ops

# =============================================================================
# Full 10-spin Gemcitabine parameters
# =============================================================================

GAMMA_RAD  = {'19F': 251.81520e6, '1H': 267.52218e6, '13C': 67.28284e6}  # rad/s/T
SPIN_ORDER = ['F0', 'F1', 'H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'C0']

gammas = np.array([
    GAMMA_RAD['19F'], GAMMA_RAD['19F'],
    GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],
    GAMMA_RAD['1H'],  GAMMA_RAD['1H'],  GAMMA_RAD['1H'],
    GAMMA_RAD['13C'],
])

N = 10
J_hz = np.zeros((N, N))
for i, j, v in [
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
    J_hz[i, j] = J_hz[j, i] = v

coords_ang = np.array([
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

sigma_ppm = [
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

B_vec = np.array([0.0, 0.0, 5e-7])   # T
tau_c = 1e-10                          # s

D = 2 ** N   # 1024

print(f"10-spin Gemcitabine: {SPIN_ORDER}")
print(f"tau_c = {tau_c:.1e} s,  B0 = {B_vec[2]*1e6:.2f} µT")
print(f"omega_F * tau_c = {GAMMA_RAD['19F'] * abs(B_vec[2]) * tau_c:.2e}  "
      f"(deep extreme-narrowing)")
print(f"Hilbert-space dimension D = {D}")
print()

# =============================================================================
# Build base operators (done once)
# =============================================================================

t0 = time.time()
print("Building H_iso and collective operators ...")
ops  = build_spin_ops(N)
H_iso, ops, evals, ekets = build_H_iso(gammas, J_hz, coords_ang.tolist(),
                                        sigma_ppm, B_vec, ops)
QZ1  = build_QZ_ops(1, gammas, sigma_ppm, B_vec, ops)
QZ2  = build_QZ_ops(2, gammas, sigma_ppm, B_vec, ops)
Qdip = build_Qdip_ops(gammas, coords_ang.tolist(), ops)
print(f"  done  ({time.time()-t0:.1f} s)")
print()

H0_mat   = H_iso.full()
ekets_arr = np.array([ek.full().flatten() for ek in ekets], dtype=complex)
pref      = {l: np.sqrt(8.0 * tau_c / (np.pi**2 * (2*l + 1))) for l in (1, 2)}

def _krylov_coeff(j):
    return (1.0 / factorial(j)) * 2.0**(j - 1) * _gamma_func((j + 1) / 2.0)**2

# =============================================================================
# Single-pass Krylov chain over all operators → partial sums for orders 0…6
# =============================================================================

MAX_ORDER = 6
ORDERS    = list(range(MAX_ORDER + 1))

# Store partial sums: chains[l][(k,m)][order] = ndarray
chains = {1: {}, 2: {}}

print(f"Computing Krylov chains (single pass, orders 0–{MAX_ORDER}) ...")
print("  Rank-1 operators (9 total) ...")
t1 = time.time()
for k in range(-1, 2):
    for m in range(-1, 2):
        Aj   = QZ1[(k, m)].full().astype(complex, copy=True)
        Abar = np.zeros((D, D), dtype=complex)
        sums = {}
        for j in range(MAX_ORDER + 1):
            Abar += _krylov_coeff(j) * Aj
            sums[j] = pref[1] * Abar
            if j < MAX_ORDER:
                Aj = 1j * tau_c * (H0_mat @ Aj - Aj @ H0_mat)
        chains[1][(k, m)] = sums
print(f"    done  ({time.time()-t1:.1f} s)")

print("  Rank-2 operators (25 total) ...")
t2 = time.time()
for k in range(-2, 3):
    for m in range(-2, 3):
        Aj   = (QZ2[(k, m)] + Qdip[(k, m)]).full().astype(complex, copy=True)
        Abar = np.zeros((D, D), dtype=complex)
        sums = {}
        for j in range(MAX_ORDER + 1):
            Abar += _krylov_coeff(j) * Aj
            sums[j] = pref[2] * Abar
            if j < MAX_ORDER:
                Aj = 1j * tau_c * (H0_mat @ Aj - Aj @ H0_mat)
        chains[2][(k, m)] = sums
print(f"    done  ({time.time()-t2:.1f} s)")
print()

# Relative corrections (rank-2, order n vs order 0)
print("Relative L2 corrections vs order 0:")
for order in ORDERS[1:]:
    deltas = []
    for km in chains[2]:
        F0 = np.linalg.norm(chains[2][km][0], 'fro')
        dF = np.linalg.norm(chains[2][km][order] - chains[2][km][0], 'fro')
        if F0 > 1e-30:
            deltas.append(dF / F0)
    print(f"  order {order}:  max = {max(deltas):.3e},  mean = {np.mean(deltas):.3e}")
print()

# =============================================================================
# Save — one file per order to keep individual file sizes manageable
# =============================================================================

DATA_DIR = os.path.join(_HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

shared_meta = {
    'system':              'gemcitabine 10-spin (F0,F1,H0–H6,C0)',
    'spin_labels':         SPIN_ORDER,
    'B_vec_T':             B_vec.tolist(),
    'tau_c_s':             tau_c,
    'omega_F_tau_c':       float(GAMMA_RAD['19F'] * abs(B_vec[2]) * tau_c),
    'krylov_orders':       ORDERS,
    'D':                   D,
    'L1_keys':             list(chains[1].keys()),
    'L2_keys':             list(chains[2].keys()),
    'units':               {'H_iso': 'rad/s', 'L': 'sqrt(rad/s)'},
    'note':                'H_iso/evals/ekets stored only in order-0 file',
}

print("Saving per-order files ...")
for order in ORDERS:
    payload = {
        'metadata': {**shared_meta, 'this_order': order},
        'H_iso':    H0_mat   if order == 0 else None,
        'evals':    evals    if order == 0 else None,
        'ekets':    ekets_arr if order == 0 else None,
        'L1':       {km: chains[1][km][order] for km in chains[1]},
        'L2':       {km: chains[2][km][order] for km in chains[2]},
    }
    out_path = os.path.join(DATA_DIR, f'gemcitabine_10spin_krylov_ops_order{order}.pkl')
    with open(out_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  order {order}: {out_path}  ({mb:.0f} MB)")

print(f"\nTotal wall time: {time.time()-t0:.1f} s")
