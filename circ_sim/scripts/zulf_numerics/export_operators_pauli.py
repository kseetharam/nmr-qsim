"""
Export the 19F-13C-1H ZULF 3-spin Hamiltonian and canonical Lindblad jump
operators as Pauli-product dictionaries and save to zulf_numerics/data/.

Output file: zulf_numerics/data/zulf_3spin_operators.pkl

Dictionary layout
-----------------
"metadata"      : dict with system parameters and conventions
"H0"            : Pauli decomposition of the Heisenberg Hamiltonian [rad/s]
"L_(k,m)"       : Pauli decomposition of the K=0 jump operator for each
                  (k, m) with k, m in {-2, -1, 0, +1, +2}  (25 entries)

Each Pauli-decomposition value is itself a dict:
    { "IXZ": complex_coeff, "ZYI": complex_coeff, ... }

Pauli strings are 3-character sequences from {I, X, Y, Z}, ordered as
[19F, 13C, 1H].  The decomposition satisfies:

    M = sum_P  c_P * (P_a ⊗ P_b ⊗ P_c)

with  c_P = (1/8) * Tr(M * P_a ⊗ P_b ⊗ P_c)

where P_{I,X,Y,Z} are the standard 2x2 Pauli matrices (not normalised to
spin-1/2 convention; the identity P_I = I_2).

Units
-----
H0            : rad / s
L_(k,m)       : sqrt(rad / s)   [so that D[L] has units 1/s in the Lindblad eq.]

Jump operators are the K=0 (bare IST) time-domain operators from td_jump_operators.py
with prefactor 2*sqrt(2)/pi, calibrated so that the K=0 dissipator reproduces
the flat-spectral-density (J(omega)=J(0)) Redfield master equation exactly.
"""

import sys
import os
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEBUG_DIR  = os.path.join(_SCRIPT_DIR, 'debug')
for p in [_SCRIPT_DIR, _DEBUG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import zulf_lindblad as zl
import td_jump_operators as tdj

# ---------------------------------------------------------------------------
# Pauli basis  (single-qubit)
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {'I': _I, 'X': _X, 'Y': _Y, 'Z': _Z}
PAULI_NAMES = ['I', 'X', 'Y', 'Z']
N_QUBITS = 3
DIM = 2 ** N_QUBITS   # 8

# Pre-build all 64 three-qubit Pauli tensor products
_PAULI_BASIS = {}   # label -> (8, 8) array
for a in PAULI_NAMES:
    for b in PAULI_NAMES:
        for c in PAULI_NAMES:
            label = a + b + c
            _PAULI_BASIS[label] = np.kron(np.kron(PAULIS[a], PAULIS[b]), PAULIS[c])


def pauli_decompose(M, threshold=1e-12):
    """
    Decompose an (8, 8) complex matrix M into the 3-qubit Pauli basis.

    Returns a dict { pauli_string: complex_coefficient } containing only
    terms whose absolute coefficient exceeds `threshold`.

    Reconstruction: M = sum_P coeffs[P] * P_a ⊗ P_b ⊗ P_c
    """
    coeffs = {}
    for label, P in _PAULI_BASIS.items():
        c = np.trace(M @ P) / DIM
        if abs(c) > threshold:
            coeffs[label] = complex(c)
    return coeffs


# ---------------------------------------------------------------------------
# H0 — Heisenberg Hamiltonian
# ---------------------------------------------------------------------------
print("Decomposing H0 ...")
H0_np = zl.H0.full().astype(complex)
H0_pauli = pauli_decompose(H0_np)
print(f"  {len(H0_pauli)} non-zero Pauli terms")

# ---------------------------------------------------------------------------
# Jump operators  L_(k,m)  at K=0
# ---------------------------------------------------------------------------
print("Building K=0 jump operators and decomposing ...")
L_ops, _ = tdj.build_td_jump_operators(trunc_order=0)   # dict (k,m) -> (8,8)

k_vals = m_vals = [-2, -1, 0, +1, +2]

L_pauli = {}
for k in k_vals:
    for m in m_vals:
        key = f"L_({k},{m})"
        L_pauli[key] = pauli_decompose(L_ops[(k, m)])

n_terms = {k: len(v) for k, v in L_pauli.items()}
print(f"  {len(L_pauli)} jump operators decomposed")
print(f"  Non-zero Pauli terms per operator: "
      f"min={min(n_terms.values())}  max={max(n_terms.values())}  "
      f"mean={np.mean(list(n_terms.values())):.1f}")

# ---------------------------------------------------------------------------
# Assemble dictionary
# ---------------------------------------------------------------------------
output = {
    "metadata": {
        "system":       "19F-13C-1H ZULF 3-spin",
        "B0":           "0 T",
        "tau_c":        float(zl.tau_c),
        "J_FC_Hz":      float(zl.J_FC),
        "J_CH_Hz":      float(zl.J_CH),
        "b_FC_rad_s":   float(zl.b_FC),
        "b_FH_rad_s":   float(zl.b_FH),
        "b_CH_rad_s":   float(zl.b_CH),
        "spin_order":   ["19F", "13C", "1H"],
        "pauli_convention": (
            "3-char string from {I,X,Y,Z}^3; "
            "M = sum_P c_P * (P[0] otimes P[1] otimes P[2]); "
            "c_P = Tr(M P) / 8"
        ),
        "H0_units":     "rad/s",
        "L_units":      "sqrt(rad/s)",
        "L_description": (
            "K=0 time-domain canonical jump operators; "
            "25 operators indexed by (k,m) with k=spin-IST component, "
            "m=spatial orientation, both in {-2,-1,0,+1,+2}; "
            "prefactor 2*sqrt(2)/pi calibrated to reproduce flat-J Redfield at K=0"
        ),
    },
    "H0": H0_pauli,
}
output.update(L_pauli)   # adds "L_(k,m)" keys for all 25 operators

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
data_dir = os.path.join(_SCRIPT_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, 'zulf_3spin_operators.pkl')

with open(out_path, 'wb') as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved -> {out_path}")
print(f"Keys: 'metadata', 'H0', " +
      ", ".join(f"'L_({k},{m})'" for k in [-2, 0, 2] for m in [-2, 0, 2]) +
      ", ...")

# ---------------------------------------------------------------------------
# Quick verification: reconstruct H0 and check against original
# ---------------------------------------------------------------------------
print("\nVerification — reconstructing H0 from Pauli coefficients:")
H0_recon = sum(c * _PAULI_BASIS[p] for p, c in H0_pauli.items())
err = np.max(np.abs(H0_recon - H0_np))
print(f"  max |H0_recon - H0| = {err:.2e} rad/s  "
      f"({'PASS' if err < 1e-8 else 'FAIL'})")

print("\nTop-5 Pauli terms in H0 by |coefficient|:")
for p, c in sorted(H0_pauli.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {p}  {c.real:+.4f}  (|c| = {abs(c):.4f} rad/s)")
