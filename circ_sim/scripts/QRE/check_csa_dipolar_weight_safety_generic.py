"""
Generic (non-Gemcitabine-specific) stress test: random 5-spin geometry,
random (non-symmetric, so rank-1 CSA != 0) shielding tensors, and a random
(tilted, not B||z) B-field direction. Checks, for EVERY (k,m) of BOTH rank-1
(CSA-only) and rank-2 (CSA+dipolar) canonical jump operators, whether
C_{j,00} = sum_{n<n'}(c_n c_n'^* P_n' P_n - h.c.) ever produces weight-0/1
Pauli content. If the phase-alignment property survives here too, it is a
structural consequence of the ladder-operator (spherical-tensor) construction,
not a coincidence of Gemcitabine's specific (near-symmetric, B||z) input data.
"""
import os, sys
from itertools import product

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(_HERE, '..', 'linblad_dyn', 'utils'))
sys.path.insert(0, _UTILS)
from linblad_utils import build_spin_ops, build_QZ_ops, build_Qdip_ops

rng = np.random.default_rng(0)
n = 4
D = 2 ** n

gammas = rng.uniform(0.5, 1.5, size=n) * 1e8
coords = rng.uniform(-3, 3, size=(n, 3))
sigma = [rng.uniform(-5, 5, size=(3, 3)) for _ in range(n)]  # generic, NOT symmetrized
B_vec = rng.normal(size=3)
B_vec = B_vec / np.linalg.norm(B_vec) * 0.3  # tilted, arbitrary direction, 0.3 T

ops = build_spin_ops(n)
QZ1 = build_QZ_ops(1, gammas, sigma, B_vec, ops)
QZ2 = build_QZ_ops(2, gammas, sigma, B_vec, ops)
Qdip = build_Qdip_ops(gammas, coords, ops)

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


def check(label, op_mat):
    terms = pauli_decompose(op_mat)
    if not terms:
        return None
    ordered = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
    C = C00(ordered)
    fro = np.linalg.norm(C, 'fro')
    if fro < 1e-12:
        return (label, [], True)
    Cterms = pauli_decompose(C, thresh=1e-9 * max(1.0, fro))
    cweights = sorted(set(weight(p) for p in Cterms))
    leak = {p: c for p, c in Cterms.items() if weight(p) <= 1}
    return (label, cweights, len(leak) == 0)


results = []
for k in range(-1, 2):
    for m in range(-1, 2):
        r = check(f"L1_({k},{m})", QZ1[(k, m)].full())
        if r:
            results.append(r)
for k in range(-2, 3):
    for m in range(-2, 3):
        A = (QZ2[(k, m)] + Qdip[(k, m)]).full()
        r = check(f"L2_({k},{m})=QZ2+Qdip", A)
        if r:
            results.append(r)

n_leak = sum(1 for _, _, safe in results if not safe)
print(f"Tested {len(results)} nonzero canonical jump operators (random geometry, "
      f"non-symmetric sigma => rank-1 CSA != 0, tilted B).")
for label, cweights, safe in results:
    flag = "SAFE" if safe else "*** LEAKAGE ***"
    print(f"  {label:28s} C_j,00 weights={cweights}  {flag}")
print(f"\nTotal leaking cases: {n_leak} / {len(results)}")
