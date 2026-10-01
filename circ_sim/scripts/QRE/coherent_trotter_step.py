"""
First actual Qualtran circuit for this pipeline (everything before this was
either physics/error analysis in Pauli-algebra/numpy, or generic Qualtran
utilities not yet wired to any molecule): builds one Trotter step of the
COHERENT part only (Zeeman layer + edge-colored Heisenberg groups) for a
molecule loaded via molecule_operators.py, as a Qualtran CompositeBloq, and
reports its resource cost.

Does NOT yet include the jump-operator (dissipative) gadget or ancilla
dispatch register -- see pipeline_plan.md for what's still missing before a
COMPLETE circuit exists. This is deliberately the simplest piece first (no
ancilla register needed).

Each Pauli-string exponential e^{-i*angle*P} compiles to the standard
basis-change + CNOT-ladder + single rotation + undo circuit, with the
rotation applied via a RotationSynthesisStrategy (rotation_synthesis_strategy.py)
so the direct-vs-phase-gradient choice stays swappable, per that module's
decision log entry.

Rz(theta) = e^{-i*theta*Z/2} (qualtran convention, verified against direct
matrix exponentiation) -- so a physical rotation e^{-i*phi*P} needs
angle = 2*phi passed to the rotation strategy.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate
from qualtran.resource_counting import get_cost_value, QECGatesCost

from rotation_synthesis_strategy import DirectSynthesisStrategy, RotationSynthesisStrategy


def apply_pauli_exponential(bb, qubits, pauli_string, angle, eps, strategy):
    """Apply e^{-i*angle*P} for Hermitian Pauli string P (angle absorbs any
    coefficient/Dt factor already), on the given list of qubit Soquets, via
    the given RotationSynthesisStrategy. Returns (bb, qubits)."""
    support = [i for i, ch in enumerate(pauli_string) if ch != 'I']
    if not support:
        return bb, qubits  # identity term: no gate (a global phase, dropped)

    # basis change: X -> H; Y -> Sdg then H; Z -> nothing
    for i in support:
        ch = pauli_string[i]
        if ch == 'X':
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        elif ch == 'Y':
            qubits[i] = bb.add(SGate().adjoint(), q=qubits[i])
            qubits[i] = bb.add(Hadamard(), q=qubits[i])

    # CNOT ladder collecting parity onto the last support qubit
    target = support[-1]
    for i in support[:-1]:
        qubits[i], qubits[target] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[target])

    # the rotation itself, via the swappable strategy (Rz(theta)=e^{-i theta Z/2})
    bb, ctx = strategy.prepare(bb, eps)
    bb, ctx, qubits[target] = strategy.apply(bb, ctx, qubits[target], 2 * angle, eps)
    bb = strategy.finish(bb, ctx)

    # undo CNOT ladder
    for i in reversed(support[:-1]):
        qubits[i], qubits[target] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[target])

    # undo basis change
    for i in support:
        ch = pauli_string[i]
        if ch == 'X':
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        elif ch == 'Y':
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
            qubits[i] = bb.add(SGate(), q=qubits[i])

    return bb, qubits


def build_coherent_trotter_step(bb, qubits, coherent_dicts, Dt, eps_gate,
                                 strategy: RotationSynthesisStrategy):
    """One first-order Trotter step of the coherent part: for each fragment
    (Zeeman layer, then edge-colored Heisenberg groups, matching
    molecule_operators.py's ordering), apply e^{-i*Dt*h_p} as a product over
    h_p's own Pauli terms (exact within a fragment for the Zeeman layer and
    for each single edge; the XX/YY/ZZ terms of a given edge commute
    exactly, so no Trotter error is introduced within a fragment either)."""
    for frag in coherent_dicts:
        for pstr, coeff in frag.items():
            angle = Dt * coeff.real  # coherent-fragment coefficients are real
            bb, qubits = apply_pauli_exponential(bb, qubits, pstr, angle, eps_gate, strategy)
    return bb, qubits


if __name__ == '__main__':
    import molecule_operators as mo
    from gate_synthesis_budget import count_rotations_per_step, full_budget
    import error_budget_solver as ebs

    ops = mo.load_molecule_operators('gemcitabine5')
    n = ops['n']
    R_coherent, R_nested = count_rotations_per_step(ops)

    K_prf, K_mixed = ebs.get_coefficients()
    EPS_TOTAL, T_ACQ = 0.1, 1.0
    budget = full_budget(EPS_TOTAL, T_ACQ, K_prf, K_mixed, R_per_step=(R_coherent, R_nested))
    Dt = budget['dt_solution'].Dt
    eps_gate = budget['eps_gate']
    print(f"Dt* = {Dt:.3e} s, eps_gate = {eps_gate:.3e}")

    bb = BloqBuilder()
    qreg = bb.add_register('q', n)
    qubits = list(bb.split(qreg))
    strategy = DirectSynthesisStrategy()
    bb, qubits = build_coherent_trotter_step(bb, qubits, ops['coherent_dicts'], Dt, eps_gate, strategy)
    qreg = bb.join(qubits)
    circuit = bb.finalize(q=qreg)

    cost = get_cost_value(circuit, QECGatesCost())
    # NOTE: total_t_count()'s default ts_per_rotation=11 is a generic
    # fallback constant ("Mixed fallback protocol, error budget 1e-3"),
    # NOT derived from the eps actually passed to each Rz -- must pass our
    # own eps_gate-derived T-per-rotation explicitly, or every rotation
    # silently gets costed at a precision we didn't choose.
    t_per_gate = budget['t_per_gate']
    t_count = cost.total_t_count(ts_per_rotation=t_per_gate)
    print(f'One coherent Trotter step: {cost}')
    print(f'  T count (at eps_gate={eps_gate:.3e}, {t_per_gate} T/rotation) = {t_count}')
    print(f'  cross-check vs hand count ({R_coherent} rotations x {t_per_gate}) = '
          f'{R_coherent * t_per_gate}  (match: {t_count == R_coherent * t_per_gate})')
    print(f'  default total_t_count() would silently report: {cost.total_t_count()} '
          f'(ts_per_rotation=11 fallback -- WRONG for our eps_gate)')
