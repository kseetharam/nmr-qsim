"""
Reusable coil-observable readout gadget (pipeline_plan.md open question 5,
"coil read-out resolved 2026-09-18"): coil = Sx_coll + i*Sy_coll is
non-Hermitian, so <coil> is reconstructed classically from TWO
INDEPENDENT circuit runs -- one measuring Sx_coll (X basis), one measuring
Sy_coll (Y basis) -- combined as <coil> = <Sx_coll> + i*<Sy_coll>. This is
a genuine structural fact (two distinct circuit variants), not a shot-count
question -- how many repeated shots of each variant are needed to converge
<Sx_coll>/<Sy_coll> to a target precision remains out of scope, per
pipeline_plan.md's Decision log ("resources needed for a single circuit
shot").

Both Sx_coll and Sy_coll (molecule_operators.py's Sx_coll_dict/
Sy_coll_dict) are sums of single-qubit Pauli terms on DISTINCT qubits
(weight 1 each: {'XIIII': w0, 'IXIII': w1, ...}), not one multi-qubit
Pauli string -- so, unlike apply_pauli_exponential's CNOT-ladder-based
parity collection, no entangling gates are needed at all: each qubit is
independently basis-rotated and measured, and the weighted sum over
per-qubit outcomes is pure classical post-processing (left to the
caller/estimator, not built into the circuit). Basis change matches
apply_pauli_exponential's own convention (X: Hadamard; Y: S^dagger then
Hadamard), reusing qualtran.bloqs.basic_gates.MeasureX (which already
folds a Hadamard into its own definition) rather than building H+MeasureZ
by hand: X-basis readout is bare MeasureX; Y-basis readout is
S^dagger -> MeasureX, equivalent to S^dagger -> H -> MeasureZ since
S^dagger X S = Y exactly (verified numerically in this module's
__main__), i.e. measuring X after conjugating by S^dagger realizes a
native Y-basis measurement.

Cost is therefore, per basis variant, n_sys single-qubit Clifford gates
(zero T-gates) plus n_sys measurements -- the cheapest gadget in this
pipeline by construction: there is no entangling structure to even be
Clifford-costly about.

The dilation ancilla register (used during evolution for jump-operator
dispatch) plays no role here -- coil is defined purely on system qubits,
so the ancilla register is simply left unmeasured/untouched at readout
time, equivalent to tracing it out (no extra circuitry needed).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran.bloqs.basic_gates import MeasureX, SGate


def build_coil_readout(bb, qubits, weight_dict, tol=1e-9):
    """Measure every qubit that has a term in weight_dict (Sx_coll_dict or
    Sy_coll_dict convention: one weight-1 'X'/'Y' Pauli string per
    covered qubit, real-valued coefficient) in the basis its term
    specifies. Qubits with no term (weight 0) are left untouched in the
    returned qubits list. Returns (bb, qubits, readouts), where qubits[i]
    is None for measured qubits and the original Soquet otherwise, and
    readouts is a list of (qubit_index, weight, classical_bit_soquet) --
    the classical combination <S_coll> ~= sum(w*(1-2*bit) for ...) over
    shots is left to the caller."""
    readouts = []
    for pstr, coeff in weight_dict.items():
        if abs(coeff) < tol:
            continue
        support = [i for i, ch in enumerate(pstr) if ch != 'I']
        if not support:
            continue
        (i,) = support  # weight-1 by construction (Sx_coll/Sy_coll)
        ch = pstr[i]
        q = qubits[i]
        if ch == 'Y':
            q = bb.add(SGate().adjoint(), q=q)
        elif ch != 'X':
            raise ValueError(f"coil readout expects X or Y terms only, got {ch!r}")
        c = bb.add(MeasureX(), q=q)
        qubits[i] = None
        readouts.append((i, coeff.real, c))
    return bb, qubits, readouts


if __name__ == '__main__':
    import numpy as np
    from qualtran import BloqBuilder, QBit, Register, Side
    from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

    import molecule_operators as mo

    # ---- correctness check: Sdg->MeasureX realizes a Y-basis measurement,
    # i.e. S^dagger X S = Y exactly (the one new piece of physics here).
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    # Want V such that V^dagger X V = Y (so "apply V, then measure X" == "measure Y"
    # directly): V=Sdg solves this, since S X S^dagger = Y exactly.
    err = np.abs(S @ X @ S.conj().T - Y).max()
    print(f'S-conjugation check: max|S X Sdg - Y| = {err:.3e} (expect 0; '
          f'justifies "apply Sdg, then measure X" == measure Y)')
    assert err < 1e-12

    # ---- resource count on Gemcitabine's real coil weights ----
    ops = mo.load_molecule_operators('gemcitabine5')
    n_sys = ops['n']

    for label, weight_dict in [('Sx_coll (X-basis readout)', ops['Sx_coll_dict']),
                                ('Sy_coll (Y-basis readout)', ops['Sy_coll_dict'])]:
        supports = [i for pstr in weight_dict for i, ch in enumerate(pstr) if ch != 'I']
        assert len(supports) == len(set(supports)), 'expected weight-1 terms on distinct qubits'

        bb = BloqBuilder()
        sys_reg = bb.add_register(Register('sys', QBit(), shape=(n_sys,), side=Side.LEFT), None)
        qubits = list(sys_reg)
        bb, qubits, readouts = build_coil_readout(bb, qubits, weight_dict)
        assert all(q is None for q in qubits), 'Gemcitabine has no zero-weight spin'
        circuit = bb.finalize(**{f'c{i}': c for i, _, c in readouts})

        cost = get_cost_value(circuit, QECGatesCost())
        qcount = get_cost_value(circuit, QubitCount())
        print(f'\n{label}: {cost}')
        print(f'  qubits={qcount}, measured={len(readouts)}/{n_sys}, '
              f'T-count={cost.total_t_count()} (expect 0: Clifford + measurement only)')
        print(f'  weights: {[round(w, 4) for _, w, _ in sorted(readouts)]}')
