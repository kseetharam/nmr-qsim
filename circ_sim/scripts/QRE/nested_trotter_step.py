"""
Reusable nested/jump-operator Trotter-step gadget, mirroring
coherent_trotter_step.build_coherent_trotter_step: builds the dissipative
part of one outer Trotter step (Sec.~sec:nested_trot_leading of qre.tex),
using the non-local (log-register + unary-iteration) ancilla dispatch
scheme -- the one found to dominate local (one-hot) dispatch in
jump_dispatch_comparison.py (16 fewer qubits at a 0.15%-of-payload T-gate
dispatch overhead).

Promoted out of jump_dispatch_comparison.py's build_nonlocal_dispatch
(2026-09-21) with one correctness fix (2026-09-22, caught while doing this
promotion): that version compiled each hop term's coefficient c_{j,n} into
UP TO TWO separate, sequential Pauli-string exponentials -- one on X_a@P
(angle ~ Re(c)), one on Y_a@P (angle ~ Im(c)) -- whenever both were
nonzero. This is not what qre.tex Sec.~sec:Vj_block_encoding derives: the
hop-term generator Xi_j(phi)@P (phi = arg(c_{j,n})) is a SINGLE
Hermitian-unitary operator, meant to be applied as one atomic exponential
at angle theta = |c_{j,n}|*sqrt(Dt). Splitting it into X_a@P and Y_a@P
pieces is not innocuous: those two operators ANTICOMMUTE ({X@P, Y@P} =
(XY+YX)@P^2 = 0), so their product-formula composition disagrees with the
true exponential already at O(theta^2) = O(Dt) whenever both pieces are
present (a short BCH expansion: exp(-i*theta_x*X@P)*exp(-i*theta_y*Y@P) =
exp(-i*theta_x*X@P - i*theta_y*Y@P - theta_x*theta_y*[X@P,Y@P]/2 + ...),
and [X@P,Y@P] = 2i*Z_a (nonzero) picks up exactly the theta_x*theta_y
cross term the atomic exponential does not have). That is a NEW,
previously uncharacterized error source at the same order as K_nested --
whose vanishing at ZULF field is the entire basis for this pipeline's
clean O(Dt^2) scaling result -- so it was not safe to just carry forward
uncosted.

Exact fix used here instead: Xi_j(phi) = cos(phi)*X - sin(phi)*Y is a
Z-conjugated X operator, Xi_j(phi) = Rz(-phi) @ X @ Rz(-phi)^dagger (a
standard single-qubit identity: e^{-i*alpha*Z/2} X e^{i*alpha*Z/2} =
cos(alpha)*X + sin(alpha)*Y, at alpha=-phi). So the exact hop-term
exponential is realized as Rz(phi) -> [X@P rotation, angle
theta=|c_{j,n}|*sqrt(Dt)] -> Rz(-phi), with NO Trotter error at all (an
exact identity, not an approximation) -- at the cost of up to 2 extra
rotations per term relative to the old (incorrect) scheme. When the term
is purely real (Im(c)~=0) or purely imaginary (Re(c)~=0), phi is a
multiple of pi/2 and Xi_j(phi) is already a bare Pauli operator (+-X or
+-Y) -- the conjugation Rz's are skipped entirely (0 extra cost), matching
the old scheme's cost in that case. Verified end-to-end in this module's
__main__ against a direct numpy matrix exponential of the same generator
Xi_j(phi)@P at a representative angle, to check the identity itself (not
just gate-count it).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran import BloqBuilder
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from coherent_trotter_step import apply_pauli_exponential
from gate_synthesis_budget import count_hop_rotations, hop_angle_phase, t_count_direct
from rotation_synthesis_strategy import DirectSynthesisStrategy, RotationSynthesisStrategy
from unary_iteration import build_unary_tree


def apply_hop_term(bb, anc_qubit, sys_qubits, pstr, coeff, Dt, eps_gate, strategy, tol=1e-9):
    """Exact e^{-i*sqrt(Dt)*Xi_j(phi)@P} for one hop term (pstr, coeff),
    where Xi_j(phi)=cos(phi)X-sin(phi)Y, phi=arg(coeff). anc_qubit plays
    the "ancilla-role" qubit (a dedicated one-hot qubit for local dispatch,
    or the unary-iteration walk's own flag for non-local -- identical
    compilation either way, per jump_dispatch_comparison.py's original
    module docstring). Returns (bb, anc_qubit, sys_qubits, n_rotations)."""
    ap = hop_angle_phase(coeff, tol)
    if ap is None:
        return bb, anc_qubit, sys_qubits, 0
    mag, phi = ap
    theta = mag * math.sqrt(Dt)
    n_rot = 0

    pure_real = abs(coeff.imag) < tol
    pure_imag = abs(coeff.real) < tol
    if pure_real or pure_imag:
        # Xi_j(phi) is already a bare +-X or +-Y: no conjugation needed.
        anc_char = 'X' if pure_real else 'Y'
        signed_angle = coeff.real if pure_real else -coeff.imag
        signed_angle = math.copysign(theta, signed_angle)
        combined = [anc_qubit] + list(sys_qubits)
        bb, combined = apply_pauli_exponential(
            bb, combined, anc_char + pstr, signed_angle, eps_gate, strategy)
        anc_qubit, sys_qubits = combined[0], combined[1:]
        return bb, anc_qubit, sys_qubits, 1

    # General case: Rz(phi) -> X@P rotation(theta) -> Rz(-phi), exact.
    bb, ctx = strategy.prepare(bb, eps_gate)
    bb, ctx, anc_qubit = strategy.apply(bb, ctx, anc_qubit, phi, eps_gate)
    bb = strategy.finish(bb, ctx)
    n_rot += 1

    combined = [anc_qubit] + list(sys_qubits)
    bb, combined = apply_pauli_exponential(bb, combined, 'X' + pstr, theta, eps_gate, strategy)
    anc_qubit, sys_qubits = combined[0], combined[1:]
    n_rot += 1

    bb, ctx = strategy.prepare(bb, eps_gate)
    bb, ctx, anc_qubit = strategy.apply(bb, ctx, anc_qubit, -phi, eps_gate)
    bb = strategy.finish(bb, ctx)
    n_rot += 1

    return bb, anc_qubit, sys_qubits, n_rot


def build_nested_trotter_step(bb, sel_qubits, sys_qubits, jump_pauli_terms, Dt, eps_gate,
                               strategy: RotationSynthesisStrategy):
    """One Trotter step of the dissipative part: dispatches to each jump
    operator's dilation-ancilla level via unary iteration over sel_qubits
    (non-local scheme, jump_dispatch_comparison.py's winning choice), then
    applies every one of that jump operator's hop terms exactly
    (apply_hop_term) using the walk's own flag qubit as the ancilla-role
    qubit for the payload. sel_qubits must hold
    ceil(log2(len(jump_pauli_terms))) qubits. Returns
    (bb, sel_qubits, sys_qubits, n_rotations)."""
    S_k = len(jump_pauli_terms)
    state = {'sys': list(sys_qubits)}
    n_rot = [0]

    def payload(bb, flag, leaf_idx):
        for pstr, coeff in jump_pauli_terms[leaf_idx]:
            bb, flag, state['sys'], r = apply_hop_term(
                bb, flag, state['sys'], pstr, coeff, Dt, eps_gate, strategy)
            n_rot[0] += r
        return bb, flag

    bb, sel_qubits = build_unary_tree(bb, sel_qubits, payload, 0, S_k)
    return bb, sel_qubits, state['sys'], n_rot[0]


if __name__ == '__main__':
    import numpy as np

    import molecule_operators as mo

    # ---- correctness check: the Rz-conjugation identity itself, against
    # a direct numpy matrix exponential of Xi_j(phi)@P at a representative
    # (non-special) angle, before trusting the Qualtran circuit built on it.
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def expm_herm(H):
        w, v = np.linalg.eigh(H)
        return (v * np.exp(-1j * w)) @ v.conj().T

    phi_test = 0.37
    theta_test = 0.83
    Xi = math.cos(phi_test) * X - math.sin(phi_test) * Y
    P_test = Z  # stand-in single-qubit "system" Pauli, since pstr='I' here
    exact = expm_herm(theta_test * np.kron(Xi, P_test))

    Rz = lambda a: expm_herm(a * np.kron(Z, I2) / 2)  # Rz(a)=e^{-i a Z/2} on the ancilla factor
    Xrot = expm_herm(theta_test * np.kron(X, P_test))
    reconstructed = Rz(-phi_test) @ Xrot @ Rz(phi_test)
    err = np.abs(exact - reconstructed).max()
    print(f'Rz-conjugation identity check: max|exact - reconstructed| = {err:.3e} '
          f'(expect ~0, confirms Xi_j(phi)=Rz(-phi) X Rz(phi) exactly)')
    assert err < 1e-12

    # ---- resource count on Gemcitabine's real jump operators ----
    ops = mo.load_molecule_operators('gemcitabine5')
    n_sys = ops['n']
    jump_terms = ops['jump_pauli_terms']
    S_k = len(jump_terms)
    N_SEL = math.ceil(math.log2(S_k))

    EPS_GATE = 1e-9
    DT = 9.33e-5  # representative Delta-t*, see optimal_error_budget_split.py
    strategy = DirectSynthesisStrategy()
    T_PER_GATE = t_count_direct(EPS_GATE)

    bb = BloqBuilder()
    sel_reg = bb.add_register('sel', N_SEL)
    sys_reg = bb.add_register('sys', n_sys)
    sel_qs = list(bb.split(sel_reg))
    sys_qs = list(bb.split(sys_reg))
    bb, sel_qs, sys_qs, n_rot = build_nested_trotter_step(
        bb, sel_qs, sys_qs, jump_terms, DT, EPS_GATE, strategy)
    circuit = bb.finalize(sel=bb.join(sel_qs), sys=bb.join(sys_qs))

    cost = get_cost_value(circuit, QECGatesCost())
    qubits = get_cost_value(circuit, QubitCount())
    t_count = cost.total_t_count(ts_per_rotation=T_PER_GATE)
    r_expected = count_hop_rotations(jump_terms)
    print(f'\nOne nested Trotter step (non-local dispatch): {cost}')
    print(f'  qubits={qubits}, rotations={n_rot} (cross-check vs '
          f'gate_synthesis_budget.count_hop_rotations: {r_expected}, match: {n_rot == r_expected})')
    print(f'  T count (at eps_gate={EPS_GATE:.3e}, {T_PER_GATE} T/rotation) = {t_count:,}')
