"""
Ad-hoc numerical check (not a permanent pipeline script): does the canonical
jump operator L_{k,m} = pref*(Q^{(Z),2}_{k,m} + Qhat_dip_{k,m}) -- which mixes
a weight-1 (CSA) and a weight-2 (dipolar) Pauli-decomposition at NONZERO field
-- still produce a C_{j,00} = sum_{n<n'}(c_n c_n'^* P_n' P_n - h.c.) with zero
weight-0/1 Pauli content, once CSA is amplified to be comparable to dipolar?

Uses the Gemcitabine 5-spin geometry/shielding tensors (gemcitabine_trunc/
compute_5spin_jump_ops.py), but with B_z artificially raised from the ZULF
value (5e-7 T) to 1 T so CSA is not negligible.
"""
import os, sys
from itertools import product

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(_HERE, '..', 'linblad_dyn', 'utils'))
sys.path.insert(0, _UTILS)
from linblad_utils import build_spin_ops, build_QZ_ops, build_Qdip_ops

GAMMA_RAD = {'19F': 251.81520e6, '1H': 267.52218e6, '13C': 67.28284e6}
_gammas_full = np.array([
    GAMMA_RAD['19F'], GAMMA_RAD['19F'],
    GAMMA_RAD['1H'], GAMMA_RAD['1H'], GAMMA_RAD['1H'], GAMMA_RAD['1H'],
    GAMMA_RAD['1H'], GAMMA_RAD['1H'], GAMMA_RAD['1H'],
    GAMMA_RAD['13C'],
])
_coords_full = np.array([
    [0.1666, -1.2783, -1.3875], [0.9661, -2.5748, 0.3352],
    [4.2344, 1.2314, -0.5006], [2.7307, 1.8998, -1.1779],
    [2.8928, 0.2529, 1.3947], [0.4474, -0.6659, 1.8321],
    [-1.7462, -0.8907, 2.3837], [-4.1117, -0.4940, 1.8959],
    [2.4649, -0.4748, -1.5086], [0.9133, -1.2851, -0.2046],
])
_sigma_full = [
    np.array([[-130.0601, -14.9649, -63.1405], [-14.9649, -215.3146, 25.9348], [-63.1405, 25.9348, -160.2068]]),
    np.array([[-157.2094, -12.7347, -67.3755], [-12.7347, -235.8144, 5.7085], [-67.3755, 5.7085, -228.8689]]),
    np.array([[2.2651, -0.6354, 2.4859], [-0.6354, 2.7073, -2.7546], [2.4859, -2.7546, 6.9013]]),
    np.array([[7.0348, -2.4557, -1.2496], [-2.4557, 3.6480, -0.3257], [-1.2496, -0.3257, 1.9779]]),
    np.array([[1.5527, 0.6046, -2.3124], [0.6046, 5.3322, 0.1728], [-2.3124, 0.1728, 4.7300]]),
    np.array([[4.3528, -1.0174, 1.5080], [-1.0174, 5.8782, 1.7827], [1.5080, 1.7827, 3.5154]]),
    np.array([[3.1215, 0.1121, 1.7281], [0.1121, 9.5089, 0.5658], [1.7281, 0.5658, 7.4617]]),
    np.array([[2.5152, 0.8210, -0.8704], [0.8210, 6.6786, 1.2727], [-0.8704, 1.2727, 4.7920]]),
    np.array([[4.4701, 0.6055, 0.2616], [0.6055, 4.4715, 1.0493], [0.2616, 1.0493, 2.7238]]),
    np.array([[241.3681, -1.4291, -6.1424], [-1.4291, 245.7345, 3.0563], [-6.1424, 3.0563, 250.0215]]),
]
IDX = [0, 1, 4, 5, 9]
LBLS = ['F0', 'F1', 'H2', 'H3', 'C0']
gammas = _gammas_full[IDX]
coords = [_coords_full[i].tolist() for i in IDX]
sigma = [_sigma_full[i] for i in IDX]
n = len(gammas)
D = 2 ** n

ops = build_spin_ops(n)

B_ZULF = 5e-7
B_AMPLIFIED = 1.0  # T -- makes CSA comparable to / larger than dipolar

_PAULI1 = {'I': np.eye(2, dtype=complex),
           'X': np.array([[0, 1], [1, 0]], dtype=complex),
           'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
           'Z': np.array([[1, 0], [0, -1]], dtype=complex)}


def _pauli_string_matrix(chars):
    M = _PAULI1[chars[0]]
    for c in chars[1:]:
        M = np.kron(M, _PAULI1[c])
    return M


def pauli_decompose(A, thresh=1e-9):
    terms = {}
    for chars in product('IXYZ', repeat=n):
        P = _pauli_string_matrix(chars)
        c = np.trace(A @ P) / D
        if abs(c) > thresh:
            terms[''.join(chars)] = c
    return terms


def weight(pstr):
    return sum(1 for ch in pstr if ch != 'I')


def C00(pauli_terms):
    """sum_{n<n'} (c_n c_n'^* P_n' P_n - h.c.), pauli_terms = list of (pstr, c)."""
    C = np.zeros((D, D), dtype=complex)
    N = len(pauli_terms)
    for a in range(N):
        pstr_n, c_n = pauli_terms[a]
        P_n = _pauli_string_matrix(pstr_n)
        for b in range(a + 1, N):
            pstr_np, c_np = pauli_terms[b]
            P_np = _pauli_string_matrix(pstr_np)
            term = c_n * np.conj(c_np) * (P_np @ P_n)
            C += term - term.conj().T
    return C


def report(label, op_mat):
    terms = pauli_decompose(op_mat)
    ordered = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
    weights_present = sorted(set(weight(p) for p in terms))
    print(f"\n--- {label} ---")
    print(f"  # Pauli terms: {len(terms)}, weights present in L_j itself: {weights_present}")
    C = C00(ordered)
    Cterms = pauli_decompose(C, thresh=1e-9 * max(1.0, np.linalg.norm(C, 'fro')))
    if not Cterms:
        print("  C_{j,00} is exactly zero (no Pauli content above threshold).")
        return
    cweights = sorted(set(weight(p) for p in Cterms))
    print(f"  C_j,00: # terms = {len(Cterms)}, weights present = {cweights}")
    leak = {p: c for p, c in Cterms.items() if weight(p) <= 1}
    if leak:
        print(f"  *** WEIGHT-0/1 LEAKAGE DETECTED: {leak}")
    else:
        print("  No weight-0/1 leakage (safe).")


for B_label, B_val in [("ZULF (B_z=5e-7 T)", B_ZULF), ("AMPLIFIED (B_z=1 T)", B_AMPLIFIED)]:
    print("=" * 70)
    print(B_label)
    print("=" * 70)
    B_vec = np.array([0.0, 0.0, B_val])
    QZ1 = build_QZ_ops(1, gammas, sigma, B_vec, ops)
    QZ2 = build_QZ_ops(2, gammas, sigma, B_vec, ops)
    Qdip = build_Qdip_ops(gammas, coords, ops)

    # rank-1: pure CSA (no dipolar counterpart) -- internal safety check
    report(f"L1_(k=1,m=0)  [pure CSA, {B_label}]", QZ1[(1, 0)].full())

    # rank-2, (k,m)=(0,0): CSA + dipolar, both real-coefficient-dominated (m=0)
    L2_00 = (QZ2[(0, 0)] + Qdip[(0, 0)]).full()
    report(f"L2_(k=0,m=0) = QZ2+Qdip  [{B_label}]", L2_00)

    # rank-2, (k,m)=(1,1): CSA + dipolar with complex (m!=0) coefficients
    L2_11 = (QZ2[(1, 1)] + Qdip[(1, 1)]).full()
    report(f"L2_(k=1,m=1) = QZ2+Qdip  [{B_label}]", L2_11)

    # rank-2, (k,m)=(1,-1)
    L2_1m1 = (QZ2[(1, -1)] + Qdip[(1, -1)]).full()
    report(f"L2_(k=1,m=-1) = QZ2+Qdip  [{B_label}]", L2_1m1)

    # rank-2, (k,m)=(2,2)
    L2_22 = (QZ2[(2, 2)] + Qdip[(2, 2)]).full()
    report(f"L2_(k=2,m=2) = QZ2+Qdip  [{B_label}]", L2_22)
