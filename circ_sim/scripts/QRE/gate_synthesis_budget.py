"""
Gate-synthesis error budget (gap identified 2026-09-20, addressed
2026-09-21): error_budget_solver.py only allocates the target observable
error epsilon across Trotter/purification error (K_prf, K_mixed, K_nested).
It says nothing about the error from imperfect ROTATION SYNTHESIS -- every
Rz in the actual circuit is only synthesized to some finite precision
eps_gate, and that is what determines T-count via the direct-synthesis
formula (matches qualtran.cirq_interop.t_complexity_protocol.py exactly).

Framework: split the total budget eps = eps_trotter + eps_synth. eps_trotter
feeds error_budget_solver's Dt-solver as before. eps_synth is allocated
across every rotation gate in the FULL M-step circuit via the same
hybrid/telescoping argument as qre.tex Eq. fullEvo_error (unitary
differences add linearly along a gate sequence): if every one of R_total
rotation gates has synthesis error <= eps_gate, the total deviation from
the ideal (exactly-Trotterized) circuit is <= R_total * eps_gate.

Baseline choice (2026-09-21): split eps 50/50 between Trotter and
synthesis error, and allocate the synthesis budget EQUALLY across all
R_total rotation gates (not adaptively weighted by each gate's own
sensitivity) -- simplest defensible choice for a first baseline, not
claimed optimal.

R_total counts EVERY rotation gate across the WHOLE circuit (coherent
fragments AND nested-jump-operator Pauli terms, ungrouped -- grouping
reduces which pairs contribute to the Trotter *error* formulas, not the
raw gate count, since no gate-fusion circuit has been designed for grouped
terms; see nested_trotter_grouping.py), times M Trotter steps -- so this
module needs the full rotation count even though (as of 2026-09-21) only
the coherent-fragment bloq has actually been built.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def hop_angle_phase(coeff, tol=1e-9):
    """(mag, phi) with coeff = mag*exp(i*phi), or None if mag < tol
    (negligible term, no gate). The exact hop-term generator for Pauli
    term (P, c=c_{j,n}) is Xi_j(phi)@P with Xi_j(phi)=cos(phi)X-sin(phi)Y
    (qre.tex Sec.~sec:Vj_block_encoding); mag*Xi_j(phi) = Re(c)X - Im(c)Y.
    Used both for circuit-building (nested_trotter_step.py) and rotation
    counting (count_hop_term_rotations) from the same decomposition, so the
    two stay in sync by construction."""
    mag = abs(coeff)
    if mag < tol:
        return None
    return mag, math.atan2(coeff.imag, coeff.real)


def count_hop_term_rotations(coeff, tol=1e-9):
    """Elementary rotation count for one hop term's coefficient c_{j,n}.
    Xi_j(phi)@P is a SINGLE Hermitian-unitary generator, not literally
    decomposable into independent Re(c)-X and Im(c)-Y rotations (X@P and
    Y@P anticommute, so a naive 2-piece product picks up an uncontrolled
    O(Dt) discrepancy per term whenever both are nonzero -- caught
    2026-09-22, see nested_trotter_step.py's module docstring). The exact
    realization instead conjugates a single X@P rotation by two ancilla
    Rz(+-phi) gates (Rz's free when phi is a multiple of pi/2, i.e. when
    the term is purely real or purely imaginary): 0 rotations if
    negligible, 1 if purely real or purely imaginary (Im(c) or Re(c) below
    tol), 3 (2 conjugation Rz + 1 payload rotation) otherwise."""
    ap = hop_angle_phase(coeff, tol)
    if ap is None:
        return 0
    mag, phi = ap
    if abs(coeff.imag) < tol or abs(coeff.real) < tol:
        return 1
    return 3


def count_hop_rotations(jump_pauli_terms, tol=1e-9):
    """Actual elementary rotation count for the nested/jump-operator gadget
    -- NOT the raw Pauli-term count ($\\sum_jP_j=1075$ for Gemcitabine) and
    NOT the earlier (flawed, superseded 2026-09-22) 2-piece-per-term count
    of 1800 either -- see count_hop_term_rotations's docstring."""
    return sum(count_hop_term_rotations(c, tol) for terms in jump_pauli_terms for _, c in terms)


def count_rotations_per_step(ops):
    """(R_coherent, R_nested) rotation-gate counts for one Trotter step,
    from a molecule_operators.load_molecule_operators() bundle. R_nested is
    the actual elementary hop-rotation count (count_hop_rotations), not the
    raw Pauli-term count -- see that function's docstring."""
    R_coherent = sum(len(frag) for frag in ops['coherent_dicts'])
    R_nested = count_hop_rotations(ops['jump_pauli_terms'])
    return R_coherent, R_nested


def gate_synthesis_precision(eps_synth_total, R_total):
    """Per-gate synthesis precision, equal-split across R_total rotation
    gates, s.t. R_total * eps_gate <= eps_synth_total."""
    return eps_synth_total / R_total


def t_count_direct(eps_gate):
    """Direct-synthesis T-count per rotation (matches
    qualtran.cirq_interop.t_complexity_protocol.py's num_t_gates_from_eps)."""
    return math.ceil(1.149 * math.log2(1.0 / eps_gate) + 9.2)


def full_budget(eps_total, t, K_prf, K_mixed, K_nested=0.0, eps_split=0.5,
                 R_per_step=None, model=None):
    """End-to-end: split eps_total, solve Dt*, compute R_total and eps_gate.
    R_per_step: (R_coherent, R_nested) for one Trotter step; if given,
    R_total = sum(R_per_step) * M is also returned.

    Returns a dict: eps_trotter, eps_synth, dt_solution, R_total (or None),
    eps_gate (or None), t_per_gate (or None).
    """
    import error_budget_solver as ebs

    model = model or ebs.LinearStateIndependentModel()
    eps_trotter = eps_total * eps_split
    eps_synth = eps_total * (1 - eps_split)

    coeffs = ebs.ErrorCoefficients(K_prf=K_prf, K_mixed=K_mixed, K_nested=K_nested)
    dt_solution = model.solve_dt(eps_trotter, t, coeffs)

    result = dict(eps_trotter=eps_trotter, eps_synth=eps_synth, dt_solution=dt_solution,
                  R_total=None, eps_gate=None, t_per_gate=None)
    if R_per_step is not None and dt_solution.achievable:
        R_total = sum(R_per_step) * dt_solution.M
        eps_gate = gate_synthesis_precision(eps_synth, R_total)
        result.update(R_total=R_total, eps_gate=eps_gate, t_per_gate=t_count_direct(eps_gate))
    return result


if __name__ == '__main__':
    import molecule_operators as mo
    import error_budget_solver as ebs

    ops = mo.load_molecule_operators('gemcitabine5')
    R_coherent, R_nested = count_rotations_per_step(ops)
    print(f'R_coherent (per step) = {R_coherent}')
    print(f'R_nested   (per step) = {R_nested}')

    K_prf, K_mixed = ebs.get_coefficients()
    EPS_TOTAL = 0.1   # absolute error in <coil>; |<coil(0)>| ~ 30.3, so ~0.3% relative
    T_ACQ = 1.0        # s; ~1 Hz frequency resolution, a representative ZULF FID length

    result = full_budget(EPS_TOTAL, T_ACQ, K_prf, K_mixed,
                          R_per_step=(R_coherent, R_nested))
    print(f'\neps_total={EPS_TOTAL}, t={T_ACQ}s, 50/50 Trotter/synthesis split:')
    print(f'  eps_trotter = {result["eps_trotter"]}')
    print(f'  Dt solution = {result["dt_solution"]}')
    print(f'  R_total (all M steps, coherent+nested) = {result["R_total"]:,}')
    print(f'  eps_gate (per-rotation)                = {result["eps_gate"]:.3e}')
    print(f'  T per gate (direct synthesis)           = {result["t_per_gate"]}')
    print(f'  Total T (coherent+nested, all M steps)  = {result["t_per_gate"]*result["R_total"]:,}')
    M = result['dt_solution'].M
    print(f'  Total T (coherent part only, all M steps) = {result["t_per_gate"]*R_coherent*M:,}')
