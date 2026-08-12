"""
gemcitabine_jump_operators_exact.py

Same as gemcitabine_jump_operators.py but loads all spin parameters (J-couplings,
coordinates, CSA tensors, gyromagnetic ratios) from the Spinach-exported file
  zulf_numerics/data/gemcitabine_10spin_params_exact.mat

Output: linblad_dyn/data/gemcitabine_operators_exact.pkl
"""

import os, pickle
import numpy as np
import scipy.io
from itertools import combinations

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34   # J s
MU0  = 1.25663706212e-6  # T m / A
ANG  = 1e-10             # m per Angstrom

# ---------------------------------------------------------------------------
# Load exact parameters from Spinach/MATLAB export
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_EXACT_PATH = os.path.normpath(
    os.path.join(_HERE, '..', 'zulf_numerics', 'data',
                 'gemcitabine_10spin_params_exact.mat'))

_d = scipy.io.loadmat(_EXACT_PATH)

N          = 10
SPIN_ORDER = ['19F', '19F', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '13C']
GAMMA_1H   = 267.52218e6   # rad/s/T  (reference for weights)

gammas     = _d['gammas_vec'].flatten()              # (10,) rad/s/T
J_hz       = _d['J_matrix'].astype(float)            # (10,10) Hz, symmetric
coords_ang = _d['coords_ang_mat'].astype(float)      # (10,3) Angstrom
sigma_ppm  = [_d['csa_flat'][i].reshape(3, 3)        # list of 10 (3,3) arrays
              for i in range(N)]

B_vec = np.array([0.0, 0.0, 5e-7])   # T (along z)
tau_c = 1e-10                          # s
Bz    = B_vec[2]

print(f"Loaded exact parameters from:\n  {_EXACT_PATH}")
print(f"gammas (rad/s/T): {gammas}")
print(f"coords_ang[0]:    {coords_ang[0]}")

# ---------------------------------------------------------------------------
# Pauli string arithmetic (pure Python, no matrices)
# ---------------------------------------------------------------------------
_PM = {
    ('I','I'):('I', 1+0j), ('I','X'):('X', 1+0j), ('I','Y'):('Y', 1+0j), ('I','Z'):('Z', 1+0j),
    ('X','I'):('X', 1+0j), ('X','X'):('I', 1+0j), ('X','Y'):('Z',  1j),  ('X','Z'):('Y', -1j),
    ('Y','I'):('Y', 1+0j), ('Y','X'):('Z', -1j),  ('Y','Y'):('I', 1+0j), ('Y','Z'):('X',  1j),
    ('Z','I'):('Z', 1+0j), ('Z','X'):('Y',  1j),  ('Z','Y'):('X', -1j),  ('Z','Z'):('I', 1+0j),
}

_I10 = 'I' * N


def _add(*dicts):
    r = {}
    for d in dicts:
        for k, v in d.items():
            r[k] = r.get(k, 0j) + v
    return {k: v for k, v in r.items() if abs(v) > 1e-30}


def _scale(d, c):
    if abs(c) < 1e-30:
        return {}
    return {k: v * c for k, v in d.items()}


def _product(d1, d2):
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
# Single-spin operator Pauli dicts (spin-1/2)
# ---------------------------------------------------------------------------
def _ix(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j}
def _iy(k): return {_I10[:k]+'Y'+_I10[k+1:]: 0.5+0j}
def _iz(k): return {_I10[:k]+'Z'+_I10[k+1:]: 0.5+0j}
def _ip(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j, _I10[:k]+'Y'+_I10[k+1:]:  0.5j}
def _im(k): return {_I10[:k]+'X'+_I10[k+1:]: 0.5+0j, _I10[:k]+'Y'+_I10[k+1:]: -0.5j}

_RT2 = np.sqrt(2.0)

def _s_sph(k, q):
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
coords_m = coords_ang * ANG


def _b_dip(i, j):
    r = np.linalg.norm(coords_m[j] - coords_m[i])
    return -(MU0 / (4 * np.pi)) * gammas[i] * gammas[j] * HBAR / r**3


def _a2m_dict(r_vec):
    rhat = r_vec / np.linalg.norm(r_vec)
    A = 3 * np.outer(rhat, rhat) - np.eye(3)
    return {
         0: (2*A[2,2] - A[0,0] - A[1,1]) / np.sqrt(6),
        +1: -(A[0,2] - 1j*A[1,2]),
        -1:  (A[0,2] + 1j*A[1,2]),
        +2:  (A[0,0] - A[1,1] - 2j*A[0,1]) / 2,
        -2:  (A[0,0] - A[1,1] + 2j*A[0,1]) / 2,
    }


pairs = list(combinations(range(N), 2))
b_dip = {(i, j): _b_dip(i, j)                         for i, j in pairs}
a_geo = {(i, j): _a2m_dict(coords_m[j] - coords_m[i]) for i, j in pairs}


# ---------------------------------------------------------------------------
# T^{(2)}_k(i, j): rank-2 two-spin ISTs as Pauli dicts
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

_CG2 = {
    ( 1, 1): 1.0,
    ( 1, 0): 1/_RT2,  (0,  1): 1/_RT2,
    ( 1,-1): 1/_RT6,  (0,  0): 2/_RT6,  (-1, 1): 1/_RT6,
    ( 0,-1): 1/_RT2,  (-1, 0): 1/_RT2,
    (-1,-1): 1.0,
}


def _T2_CSA(spin_idx, neg_k):
    op = {}
    for q1 in (-1, 0, 1):
        q2 = neg_k - q1
        if q2 not in (-1, 0, 1):
            continue
        Bq2 = Bz if q2 == 0 else 0.0
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
            T = _T2_CSA(i, -k)
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
# NMR protocol operators
# ---------------------------------------------------------------------------
Sz_weighted = {}
Sy_op       = {}
coil        = {}
for i in range(N):
    w = gammas[i] / GAMMA_1H
    Sz_weighted = _add(Sz_weighted, _scale(_iz(i), w))
    Sy_op       = _add(Sy_op,       _scale(_iy(i), w))
    coil        = _add(coil,        _scale(_ip(i), w))

Sz_weighted = _clean(Sz_weighted)
Sy_op       = _clean(Sy_op)
coil        = _clean(coil)


# ---------------------------------------------------------------------------
# Assemble and save
# ---------------------------------------------------------------------------
output = {
    "metadata": {
        "system":           "gemcitabine 10-spin (2x19F, 7x1H, 1x13C)",
        "spin_order":       SPIN_ORDER,
        "B_vec_T":          B_vec.tolist(),
        "tau_c_s":          tau_c,
        "params_source":    _EXACT_PATH,
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

DATA_DIR = os.path.join(_HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
out_path = os.path.join(DATA_DIR, 'gemcitabine_operators_exact.pkl')

with open(out_path, 'wb') as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved → {out_path}")
total_L = sum(len(v) for v in L_ops.values())
print(f"Keys: 'metadata', 'H0' ({len(H0)} terms), "
      f"'Sz_weighted' ({len(Sz_weighted)} terms), "
      f"'Sy_op' ({len(Sy_op)} terms), "
      f"'coil' ({len(coil)} terms), "
      f"25 x 'L_(k,m)' ({total_L} total Pauli terms)")
