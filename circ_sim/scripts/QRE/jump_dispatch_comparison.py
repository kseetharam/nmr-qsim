"""
Concrete Toffoli/qubit comparison: local (one-hot ancilla-per-jump-operator)
vs non-local (log-register + unary iteration) dispatch, on Gemcitabine's
actual 25 jump operators -- the follow-up promised in qre.tex
sec:unary_iteration's "What this resolves, and what it does not" paragraph.

Both schemes compile each hop term via apply_hop_term (nested_trotter_step.py):
what differs is only WHICH physical qubit plays the "ancilla-role" part for a
given branch -- a dedicated one-hot qubit (local), or the unary-iteration
walk's own transient flag (non-local, unary_iteration.build_unary_tree) -- so
this script isolates exactly the dispatch overhead, not the payload. The
non-local branch below IS build_nested_trotter_step (promoted out of this
script 2026-09-21, reused here rather than duplicated 2026-09-22); only the
local branch has its own build_local_dispatch, for the comparison.

Vacuum needs no dedicated qubit in EITHER scheme: a one-hot register over
S_k=25 "excitation" qubits with at most one excited already represents
S_k+1=26 states (25 "jump" states + the all-zero vacuum state) -- so local
dispatch costs S_k=25 physical qubits, not S_k+1. (This also means the
SELECT iteration range needed is L=S_k=25, not 26: H_0's own Trotter step
is entirely separate and ancilla-independent, already handled by
coherent_trotter_step.py -- nothing needs to "dispatch to vacuum". qre.tex
sec:unary_iteration's L=4 worked example and verified-counts table used
L=S_k+1 for generality/pedagogy; the concrete gadget below only needs
L=S_k.)

Non-power-of-two note: unary_iteration.build_unary_tree only builds full
2**n-leaf trees; S_k=25 is padded to L=32 (n=5) with 7 unused leaves,
costing 30 Ands instead of the 23 a tight non-power-of-two segment tree
would need (7 Ands / 28 T-gates more than strictly necessary) -- small
next to the payload, not optimized away here.

History of the rotation-count correction (now resolved, kept for context):
gate_synthesis_budget.count_rotations_per_step's original R_nested=1075
(= sum of P_j, the raw Pauli-term count) undercounted the actual rotation
count. A first fix (2026-09-22) gave 1800 by splitting each hop term into
up to 2 separate X_a@P/Y_a@P rotations -- but that 2-piece split is itself
inexact (X@P and Y@P anticommute, so their product-formula composition
diverges from the true single exponential at O(Dt) per term whenever both
pieces are present). The exact fix (nested_trotter_step.apply_hop_term,
same day) conjugates a single X@P rotation by ancilla Rz(+-phi) gates
instead, giving the true count of 2600 (800 terms need the full 3-rotation
conjugated form, 200 need only 1, 75 are negligible) -- see
nested_trotter_step.py's module docstring for the full derivation.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran import BloqBuilder
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from gate_synthesis_budget import count_hop_rotations, t_count_direct
from nested_trotter_step import apply_hop_term, build_nested_trotter_step
from rotation_synthesis_strategy import DirectSynthesisStrategy


def build_local_dispatch(bb, anc_qubits, sys_qubits, jump_pauli_terms, Dt, eps_gate, strategy):
    """One physical ancilla qubit per jump operator (vacuum = all-zero, no
    dedicated qubit); each hop term compiles directly via apply_hop_term on
    anc_qubits[j] + sys_qubits -- no dispatch machinery at all. Returns
    (bb, anc_qubits, sys_qubits, n_rotations)."""
    n_rot = 0
    for j, terms in enumerate(jump_pauli_terms):
        for pstr, coeff in terms:
            bb, anc_qubits[j], sys_qubits, r = apply_hop_term(
                bb, anc_qubits[j], sys_qubits, pstr, coeff, Dt, eps_gate, strategy)
            n_rot += r
    return bb, anc_qubits, sys_qubits, n_rot


if __name__ == '__main__':
    import molecule_operators as mo

    ops = mo.load_molecule_operators('gemcitabine5')
    n_sys = ops['n']
    jump_terms = ops['jump_pauli_terms']
    S_k = len(jump_terms)
    N_SEL = math.ceil(math.log2(S_k))  # 5

    EPS_GATE = 1e-9  # illustrative, matching state_prep.py's convention
    DT = 9.33e-5      # representative Delta-t*, see optimal_error_budget_split.py
    strategy = DirectSynthesisStrategy()
    T_PER_GATE = t_count_direct(EPS_GATE)

    print(f'S_k={S_k} jump operators, n_sys={n_sys}, N_SEL={N_SEL} (L padded to {2**N_SEL})')
    print(f'raw Pauli-term count (gate_synthesis_budget convention): '
          f'{sum(len(t) for t in jump_terms)}')
    print(f'actual elementary hop rotations (exact conjugation scheme): '
          f'{count_hop_rotations(jump_terms)}')

    # ---- local (one-hot) ----
    bb = BloqBuilder()
    anc_reg = bb.add_register('anc', S_k)
    sys_reg = bb.add_register('sys', n_sys)
    anc_qs = list(bb.split(anc_reg))
    sys_qs = list(bb.split(sys_reg))
    bb, anc_qs, sys_qs, n_rot_local = build_local_dispatch(
        bb, anc_qs, sys_qs, jump_terms, DT, EPS_GATE, strategy)
    local_circuit = bb.finalize(anc=bb.join(anc_qs), sys=bb.join(sys_qs))
    local_cost = get_cost_value(local_circuit, QECGatesCost())
    local_qubits = get_cost_value(local_circuit, QubitCount())
    t_local = local_cost.total_t_count(ts_per_rotation=T_PER_GATE)
    print(f'\nLOCAL:    {local_cost}')
    print(f'  qubits={local_qubits} (= S_k={S_k} + n_sys={n_sys}), rotations={n_rot_local}, '
          f'T={t_local}')

    # ---- non-local (log-register + unary iteration) ----
    bb = BloqBuilder()
    sel_reg = bb.add_register('sel', N_SEL)
    sys_reg = bb.add_register('sys', n_sys)
    sel_qs = list(bb.split(sel_reg))
    sys_qs = list(bb.split(sys_reg))
    bb, sel_qs, sys_qs, n_rot_nonlocal = build_nested_trotter_step(
        bb, sel_qs, sys_qs, jump_terms, DT, EPS_GATE, strategy)
    nonlocal_circuit = bb.finalize(sel=bb.join(sel_qs), sys=bb.join(sys_qs))
    nonlocal_cost = get_cost_value(nonlocal_circuit, QECGatesCost())
    nonlocal_qubits = get_cost_value(nonlocal_circuit, QubitCount())
    t_nonlocal = nonlocal_cost.total_t_count(ts_per_rotation=T_PER_GATE)
    print(f'\nNON-LOCAL: {nonlocal_cost}')
    print(f'  qubits={nonlocal_qubits} (= N_SEL={N_SEL} + n_sys={n_sys} + '
          f'flag_ancilla={N_SEL - 1}), rotations={n_rot_nonlocal}, T={t_nonlocal}')

    print(f'\nSame rotation count both schemes: {n_rot_local == n_rot_nonlocal}')
    print(f'Dispatch T-gate overhead (non-local - local): {t_nonlocal - t_local} '
          f'(expect 4*(32-2)=120)')
    print(f'Qubit savings (local - non-local): {local_qubits - nonlocal_qubits}')
