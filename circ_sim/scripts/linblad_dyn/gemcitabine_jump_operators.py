"""
gemcitabine_jump_operators.py

Build H_iso and all rank-2 Lindblad jump operators L^{(2)}_{k,m} for the
gemcitabine 10-spin system as Pauli-string dictionaries without any matrix
intermediates.

Spin order: (F0, F1, H0, H1, H2, H3, H4, H5, H6, C0)
            2x19F, 7x1H, 1x13C

Physics
-------
    L^{(2)}_{k,m} = sqrt(2 tau_c / 5) * (Q^{(Z),2}_{k,m} + Q_dip_{k,m})

    Q^{(Z),2}_{k,m} = sum_X gamma_X (-1)^{m+1} sigma_{2,m}(X) T^{(2)}_{-k}(S_X, B)
    Q_dip_{k,m}     = sum_{i<j} b_ij a_{2,m}^{ij} T^{(2)}_k(i,j)

Valid in the extreme-narrowing limit: omega * tau_c << 1.

Parameters
----------
B_vec = (0, 0, 5e-7) T,  tau_c = 1e-10 s
omega_F * tau_c ~ 252e6 * 5e-7 * 1e-10 = 1.26e-8  (extreme narrowing holds)

Output: linblad_dyn/data/gemcitabine_operators.pkl
"""

import os, pickle
import numpy as np
from itertools import combinations

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34   # J s
MU0  = 1.25663706212e-6  # T m / A
ANG  = 1e-10             # m per Angstrom

# ---------------------------------------------------------------------------
# System parameters (from .tex "Gemcitabine parameters" section)
# ---------------------------------------------------------------------------
N = 10
SPIN_ORDER = ['19F', '19F', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '13C']

GAMMA = {'19F': 251.81520e6, '1H': 267.52218e6, '13C': 67.28284e6}  # rad/s/T
gammas = np.array([
    GAMMA['19F'], GAMMA['19F'],
    GAMMA['1H'], GAMMA['1H'], GAMMA['1H'], GAMMA['1H'],
    GAMMA['1H'], GAMMA['1H'], GAMMA['1H'],
    GAMMA['13C'],
])

# Upper-triangular J-coupling matrix (Hz); symmetric
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

# Nuclear coordinates (Angstrom): F0, F1, H0-H6, C0
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

# Chemical shielding tensors (ppm); symmetric, from DFT/B3LYP
sigma_ppm = [
    np.array([[-130.0601, -14.9649, -63.1405],   # F0
              [-14.9649, -215.3146,  25.9348],
              [-63.1405,  25.9348, -160.2068]]),
    np.array([[-157.2094, -12.7347, -67.3755],   # F1
              [-12.7347, -235.8144,   5.7085],
              [-67.3755,   5.7085, -228.8689]]),
    np.array([[2.2651, -0.6354,  2.4859],         # H0
              [-0.6354,  2.7073, -2.7546],
              [2.4859, -2.7546,  6.9013]]),
    np.array([[7.0348, -2.4557, -1.2496],         # H1
              [-2.4557,  3.6480, -0.3257],
              [-1.2496, -0.3257,  1.9779]]),
    np.array([[1.5527,  0.6046, -2.3124],         # H2
              [0.6046,  5.3322,  0.1728],
              [-2.3124,  0.1728,  4.7300]]),
    np.array([[4.3528, -1.0174,  1.5080],         # H3
              [-1.0174,  5.8782,  1.7827],
              [1.5080,  1.7827,  3.5154]]),
    np.array([[3.1215,  0.1121,  1.7281],         # H4
              [0.1121,  9.5089,  0.5658],
              [1.7281,  0.5658,  7.4617]]),
    np.array([[2.5152,  0.8210, -0.8704],         # H5
              [0.8210,  6.6786,  1.2727],
              [-0.8704,  1.2727,  4.7920]]),
    np.array([[4.4701,  0.6055,  0.2616],         # H6
              [0.6055,  4.4715,  1.0493],
              [0.2616,  1.0493,  2.7238]]),
    np.array([[241.3681, -1.4291, -6.1424],       # C0
              [-1.4291, 245.7345,  3.0563],
              [-6.1424,  3.0563, 250.0215]]),
]

B_vec = np.array([0.0, 0.0, 5e-7])   # T (along z)
tau_c = 1e-10                          # s
Bz    = B_vec[2]

# ---------------------------------------------------------------------------
# Pauli string arithmetic (pure Python, no matrices)
# ---------------------------------------------------------------------------
# Single-qubit Pauli multiplication table: (P1, P2) -> (result_char, phase)
_PM = {
    ('I','I'):('I', 1+0j), ('I','X'):('X', 1+0j), ('I','Y'):('Y', 1+0j), ('I','Z'):('Z', 1+0j),
    ('X','I'):('X', 1+0j), ('X','X'):('I', 1+0j), ('X','Y'):('Z',  1j),  ('X','Z'):('Y', -1j),
    ('Y','I'):('Y', 1+0j), ('Y','X'):('Z', -1j),  ('Y','Y'):('I', 1+0j), ('Y','Z'):('X',  1j),
    ('Z','I'):('Z', 1+0j), ('Z','X'):('Y',  1j),  ('Z','Y'):('X', -1j),  ('Z','Z'):('I', 1+0j),
}

_I10 = 'I' * N   # all-identity string for N=10


def _add(*dicts):
    """Sum of Pauli-string operator dicts."""
    r = {}
    for d in dicts:
        for k, v in d.items():
            r[k] = r.get(k, 0j) + v
    return {k: v for k, v in r.items() if abs(v) > 1e-30}


def _scale(d, c):
    """Multiply all coefficients by scalar c."""
    if abs(c) < 1e-30:
        return {}
    return {k: v * c for k, v in d.items()}


def _product(d1, d2):
    """Operator product of two Pauli-string dicts (handles same-site terms)."""
    r = {}
    for s1, c1 in d1.items():
        for s2, c2 in d2.items():
            chars, ph = [], 1+0j
            for ch1, ch2 in zip(s1, s2):
                ch, p = _PM[(ch1, ch2)]
                chars.append(ch)
                ph *= p
            key = ''.join(chars)
            r[key] = r.get(key, 0j) + c1 * c2 * ph
    return {k: v for k, v in r.items() if abs(v) > 1e-30}


def _clean(d, thr=1e-20):
    return {k: v for k, v in d.items() if abs(v) > thr}


# ---------------------------------------------------------------------------
# Single-spin operator Pauli dicts (spin-1/2: Sx=X/2, Sy=Y/2, Sz=Z/2)
# ---------------------------------------------------------------------------
def _ix(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j}
def _iy(k): return {_I10[:k]+'Y'+_I10[k+1:]: 0.5+0j}
def _iz(k): return {_I10[:k]+'Z'+_I10[k+1:]: 0.5+0j}
def _ip(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j, _I10[:k]+'Y'+_I10[k+1:]:  0.5j}  # S+ = Sx + i Sy
def _im(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j, _I10[:k]+'Y'+_I10[k+1:]: -0.5j}  # S- = Sx - i Sy

_RT2 = np.sqrt(2.0)

def _s_sph(k, q):
    """Spherical spin-1/2: q=0 → Sz,  q=+1 → -S+/√2,  q=-1 → +S-/√2"""
    if q ==  0: return _iz(k)
    if q == +1: return _scale(_ip(k), -1/_RT2)
    if q == -1: return _scale(_im(k), +1/_RT2)


# ---------------------------------------------------------------------------
# H_iso = Σ_{i<j} 2π J_ij S_i·S_j  −  Σ_i γ_i σ_iso(i) Bz Iz(i)
# ---------------------------------------------------------------------------
print("Building H0 ...")

sigma_tilde = [np.eye(3) + 1e-6 * s for s in sigma_ppm]
sigma_iso   = np.array([np.trace(st) / 3 for st in sigma_tilde])

H0 = {}
for i, j in combinations(range(N), 2):
    if abs(J_hz[i, j]) < 1e-12:
        continue
    c = 2 * np.pi * J_hz[i, j]
    H0 = _add(H0,
              _scale(_product(_ix(i), _ix(j)), c),
              _scale(_product(_iy(i), _iy(j)), c),
              _scale(_product(_iz(i), _iz(j)), c))

for i in range(N):
    H0 = _add(H0, _scale(_iz(i), -gammas[i] * sigma_iso[i] * Bz))

H0 = _clean(H0)
print(f"  {len(H0)} non-zero Pauli terms")


# ---------------------------------------------------------------------------
# Dipolar coupling constants b_ij and orientational factors a_{2,m}^{ij}
# ---------------------------------------------------------------------------
coords_m = coords_ang * ANG   # convert to meters


def _b_dip(i, j):
    r = np.linalg.norm(coords_m[j] - coords_m[i])
    return -(MU0 / (4 * np.pi)) * gammas[i] * gammas[j] * HBAR / r**3


def _a2m_dict(r_vec):
    """Rank-2 orientational factors {m: complex} for unit-vector direction r_vec."""
    rhat = r_vec / np.linalg.norm(r_vec)
    A = 3 * np.outer(rhat, rhat) - np.eye(3)
    return {
         0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
    }


pairs  = list(combinations(range(N), 2))
b_dip  = {(i, j): _b_dip(i, j)                                for i, j in pairs}
a_geo  = {(i, j): _a2m_dict(coords_m[j] - coords_m[i])        for i, j in pairs}


# ---------------------------------------------------------------------------
# T^{(2)}_k(i, j): rank-2 two-spin ISTs as Pauli dicts (i ≠ j)
# ---------------------------------------------------------------------------
_RT6 = np.sqrt(6.0)


def _T2ij(i, j, k):
    if k ==  0:
        return _scale(_add(_scale(_product(_iz(i), _iz(j)), 2),
                           _scale(_product(_ix(i), _ix(j)), -1),
                           _scale(_product(_iy(i), _iy(j)), -1)), 1/_RT6)
    if k == +1: return _scale(_add(_product(_ip(i), _iz(j)), _product(_iz(i), _ip(j))), -0.5)
    if k == -1: return _scale(_add(_product(_im(i), _iz(j)), _product(_iz(i), _im(j))),  0.5)
    if k == +2: return _scale(_product(_ip(i), _ip(j)), 0.5)
    if k == -2: return _scale(_product(_im(i), _im(j)), 0.5)


# ---------------------------------------------------------------------------
# Q_dip_{k,m} = Σ_{i<j} b_ij a_{2,m}^{ij} T^{(2)}_k(i,j)
# ---------------------------------------------------------------------------
print("Building Q_dip operators (45 pairs × 25 (k,m)) ...")

Q_dip = {}
for k in range(-2, 3):
    for m in range(-2, 3):
        op = {}
        for i, j in pairs:
            c = b_dip[(i, j)] * a_geo[(i, j)][m]
            if abs(c) < 1e-30:
                continue
            op = _add(op, _scale(_T2ij(i, j, k), c))
        Q_dip[(k, m)] = _clean(op)

print("  done")


# ---------------------------------------------------------------------------
# sigma_{2,m}(X): rank-2 spherical components of shielding tensor
# ---------------------------------------------------------------------------
def _sigma2m(sigma_t):
    """Rank-2 components of sigma_tilde = 1 + 1e-6 * sigma_ppm."""
    s = np.asarray(sigma_t, dtype=complex)
    s_iso = np.trace(s) / 3
    return {
         0:  np.sqrt(2/3) * (s[2,2] - s_iso),
        +1: -0.5 * ((s[0,2]+s[2,0]) + 1j*(s[1,2]+s[2,1])),
        -1: +0.5 * ((s[0,2]+s[2,0]) - 1j*(s[1,2]+s[2,1])),
        +2:  0.5 * ((s[0,0]-s[1,1]) + 1j*(s[0,1]+s[1,0])),
        -2:  0.5 * ((s[0,0]-s[1,1]) - 1j*(s[0,1]+s[1,0])),
    }


slm_all = [_sigma2m(st) for st in sigma_tilde]

# CG coefficients <1,q1; 1,q2 | 2, q1+q2>  (only rank-2, keyed by (q1,q2))
_CG2 = {
    ( 1, 1): 1.0,
    ( 1, 0): 1/_RT2,  (0,  1): 1/_RT2,
    ( 1,-1): 1/_RT6,  (0,  0): 2/_RT6,  (-1, 1): 1/_RT6,
    ( 0,-1): 1/_RT2,  (-1, 0): 1/_RT2,
    (-1,-1): 1.0,
}


def _T2_CSA(spin_idx, neg_k):
    """
    T^{(2)}_{neg_k}(S_X, B) = Σ_{q1+q2=neg_k} CG(2,q1,q2) B_{q2} S_sph[q1]

    With B=(0,0,Bz): only q2=0 term survives.
    - neg_k=-2,+2: zero (q1=±2 out of range for spin-1/2)
    - neg_k=-1: (1/√2)*Bz * S_sph[-1][X]   → +Bz*Im/2 / √2 ... = Bz*Im/2
       actually: cg(2,-1,0)=1/√2, Bz * (1/√2) * S_sph[-1] = Bz/√2 * Im/√2 = Bz*Im/2
    - neg_k= 0: cg(2,0,0)=2/√6, term = (2/√6)*Bz*Iz  = √(2/3)*Bz*Iz
    - neg_k=+1: cg(2,+1,0)=1/√2, term = (1/√2)*Bz*S_sph[+1] = (1/√2)*Bz*(-Ip/√2) = -Bz*Ip/2
    """
    op = {}
    for q1 in (-1, 0, 1):
        q2 = neg_k - q1
        if q2 not in (-1, 0, 1):
            continue
        Bq2 = Bz if q2 == 0 else 0.0   # B_sph: only q2=0 is nonzero
        if abs(Bq2) < 1e-30:
            continue
        cg = _CG2.get((q1, q2), 0.0)
        if abs(cg) < 1e-15:
            continue
        op = _add(op, _scale(_s_sph(spin_idx, q1), cg * Bq2))
    return op


# ---------------------------------------------------------------------------
# Q^{(Z),2}_{k,m} = Σ_X γ_X (-1)^{m+1} σ_{2,m}(X) T^{(2)}_{-k}(S_X, B)
# ---------------------------------------------------------------------------
print("Building Q_CSA operators ...")

Q_CSA = {}
for k in range(-2, 3):
    for m in range(-2, 3):
        phase = (-1) ** (m + 1)
        op = {}
        for i in range(N):
            slm = slm_all[i][m]
            if abs(slm) < 1e-30:
                continue
            T = _T2_CSA(i, -k)   # T^{(2)}_{-k}(S_i, B)
            if not T:
                continue
            op = _add(op, _scale(T, gammas[i] * phase * slm))
        Q_CSA[(k, m)] = _clean(op)

print("  done")


# ---------------------------------------------------------------------------
# L^{(2)}_{k,m} = scale2 * (Q^{(Z),2}_{k,m} + Q_dip_{k,m})
# ---------------------------------------------------------------------------
scale2 = np.sqrt(2 * tau_c / 5)

print("Assembling L^{(2)} operators ...")
L_ops = {}
for k in range(-2, 3):
    for m in range(-2, 3):
        op = _add(Q_CSA[(k, m)], Q_dip[(k, m)])
        L_ops[(k, m)] = _clean(_scale(op, scale2))

n_active = sum(1 for v in L_ops.values() if v)
n_terms  = {km: len(v) for km, v in L_ops.items()}
print(f"  {n_active}/25 non-zero  |  Pauli terms per op: "
      f"min={min(n_terms.values())}  max={max(n_terms.values())}  "
      f"mean={np.mean(list(n_terms.values())):.1f}")


# ---------------------------------------------------------------------------
# NMR protocol operators  (weights = gamma_i / gamma_1H, matching Spinach)
# ---------------------------------------------------------------------------
# Sz_weighted : pre-pulse magnetization   rho_sud = sum_i w_i Sz_i
# Sy_op       : Y-pulse generator         Sy = sum_i w_i Sy_i
#               rho0 = exp(-i pi/2 Sy_op) rho_sud exp(i pi/2 Sy_op)
# coil        : detection observable      coil = sum_i w_i S+_i
#               FID(t) = Tr(coil . rho(t))
# All three are dimensionless operator dictionaries.

Sz_weighted = {}
Sy_op       = {}
coil        = {}
for i in range(N):
    w = gammas[i] / GAMMA['1H']
    Sz_weighted = _add(Sz_weighted, _scale(_iz(i), w))
    Sy_op       = _add(Sy_op,       _scale(_iy(i), w))
    coil        = _add(coil,        _scale(_ip(i), w))

Sz_weighted = _clean(Sz_weighted)
Sy_op       = _clean(Sy_op)
coil        = _clean(coil)

# ---------------------------------------------------------------------------
# Assemble output dictionary (matches zulf_3spin_operators.pkl format)
# ---------------------------------------------------------------------------
output = {
    "metadata": {
        "system":           "gemcitabine 10-spin (2x19F, 7x1H, 1x13C)",
        "spin_order":       SPIN_ORDER,
        "B_vec_T":          B_vec.tolist(),
        "tau_c_s":          tau_c,
        "pauli_convention": (
            f"{N}-char strings from {{I,X,Y,Z}}^{N}, ordered as spin_order; "
            "M = sum_P c_P (P[0] otimes ... otimes P[9]); "
            f"c_P = Tr(M P) / 2^{N}"
        ),
        "H0_units":         "rad/s",
        "L_units":          "sqrt(rad/s)",
        "L_description": (
            "Rank-2 Lindblad jump operators L^{(2)}_{k,m} = sqrt(2 tau_c/5) "
            "* (Q^{(Z),2}_{k,m} + Q_dip_{k,m}); "
            "25 operators indexed by (k,m) in {-2,-1,0,+1,+2}^2; "
            "extreme-narrowing limit (omega*tau_c << 1); "
            "L^{(1)} = 0 (symmetric shielding tensors imply sigma_{1,m}=0)"
        ),
        "nmr_protocol": (
            "Sudden-transfer + hard-pulse ZULF experiment. "
            "weights w_i = gamma_i / gamma_1H. "
            "Sz_weighted = sum_i w_i Sz_i (pre-pulse state rho_sud); "
            "Sy_op = sum_i w_i Sy_i (pi/2 Y-pulse generator); "
            "rho0 = exp(-i pi/2 Sy_op) rho_sud exp(i pi/2 Sy_op); "
            "coil = sum_i w_i S+_i (quadrature detection observable, dimensionless); "
            "FID(t) = Tr(coil . rho(t))"
        ),
    },
    "H0":          H0,
    "Sz_weighted": Sz_weighted,
    "Sy_op":       Sy_op,
    "coil":        coil,
}
for k in range(-2, 3):
    for m in range(-2, 3):
        output[f"L_({k},{m})"] = L_ops[(k, m)]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
out_path = os.path.join(DATA_DIR, 'gemcitabine_operators.pkl')

with open(out_path, 'wb') as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved → {out_path}")
total_L = sum(len(v) for v in L_ops.values())
print(f"Keys: 'metadata', 'H0' ({len(H0)} terms), "
      f"'Sz_weighted' ({len(Sz_weighted)} terms), "
      f"'Sy_op' ({len(Sy_op)} terms), "
      f"'coil' ({len(coil)} terms), "
      f"25 x 'L_(k,m)' ({total_L} total Pauli terms)")
