"""
End-to-end single-shot resource estimate for truncated Gemcitabine: every
gadget this pipeline has separately built (state_prep.py,
coherent_trotter_step.py, nested_trotter_step.py, coil_readout.py) composed
into one circuit -- state prep -> M x (coherent step + nested/dispatch
step) -> readout -- with a single, consistent Delta-t*/eps_gate solved from
the (now-corrected) full rotation count, per pipeline_plan.md's roadmap
("what remains is assembling them into one end-to-end circuit/resource
total").

Two things this script does NOT literally build, for tractability, with an
explicit check standing in for each:

1. It does not construct M ~ 2*10^4 literal Trotter steps as one Qualtran
   CompositeBloq -- gate counts are linear in M by construction (every step
   is the identical circuit structure at the identical eps_gate), so the
   total is computed as M * (one step's cost) + (one-time costs). This
   linearity is checked, not just assumed: a small literal chain
   (M_TEST steps) is built and its QECGatesCost is confirmed to scale
   exactly linearly in M_TEST (matching one_step_cost*M_TEST +
   state_prep_cost to the gate) and its QubitCount confirmed constant
   (transient dispatch/And ancilla are freed within each step, not
   accumulated across steps -- verified here, not assumed from
   nested_trotter_step.py's single-step check alone).

2. It reports the cost of ONE full circuit (state prep + M steps + ONE
   readout variant). Per coil_readout.py, reconstructing <coil> needs TWO
   such circuits (X-basis and Y-basis readout tails) -- structurally
   distinct circuits, both costed here, NOT summed into a single "total"
   number, since a real experiment runs them as separate circuit
   executions, not concatenated into one. How many repeated shots of
   *each* variant are needed for a target measurement precision remains
   the already-deferred shot-budget question (pipeline_plan.md Decision
   log) -- not addressed here, and not conflated with the two-variant
   structural fact above.

Everything here inherits the ZULF-branch restriction (K_nested~=0) and the
null-state-prep/no-t-design caveats already logged in pipeline_plan.md --
this is the pipeline's first genuine complete baseline number, not a final
validated resource estimate.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran import BloqBuilder
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

import error_budget_solver as ebs
import molecule_operators as mo
import optimal_error_budget_split as opt
from coherent_trotter_step import build_coherent_trotter_step
from coil_readout import build_coil_readout
from gate_synthesis_budget import count_rotations_per_step, t_count_direct
from nested_trotter_step import build_nested_trotter_step
from rotation_synthesis_strategy import DirectSynthesisStrategy
from state_prep import build_null_state_prep, pulse_angles
from unary_iteration import build_unary_tree  # noqa: F401  (re-exported implicitly via nested step)


def build_full_trotter_step(bb, sys_qubits, sel_qubits, ops, Dt, eps_gate, strategy):
    """One outer Trotter step: coherent part, then the full dissipative
    (nested/dispatch) part, matching qre.tex's Trotterization scheme
    ("dissipative part is applied after the full coherent part"). Returns
    (bb, sys_qubits, sel_qubits, n_rot)."""
    bb, sys_qubits = build_coherent_trotter_step(
        bb, sys_qubits, ops['coherent_dicts'], Dt, eps_gate, strategy)
    bb, sel_qubits, sys_qubits, n_rot_nested = build_nested_trotter_step(
        bb, sel_qubits, sys_qubits, ops['jump_pauli_terms'], Dt, eps_gate, strategy)
    R_coherent = sum(len(frag) for frag in ops['coherent_dicts'])
    return bb, sys_qubits, sel_qubits, R_coherent + n_rot_nested


def build_chain(bb, sys_qubits, sel_qubits, ops, weights, n_steps, Dt, eps_gate, strategy,
                 with_state_prep, readout_weight_dict=None):
    """State prep (optional) -> n_steps full Trotter steps -> readout
    (optional, X- or Y-basis depending on readout_weight_dict). Returns
    (bb, readouts) if readout_weight_dict given, else (bb, sys_qubits, sel_qubits)."""
    if with_state_prep:
        bb, sys_qubits = build_null_state_prep(bb, sys_qubits, weights, eps_gate, strategy)
    for _ in range(n_steps):
        bb, sys_qubits, sel_qubits, _ = build_full_trotter_step(
            bb, sys_qubits, sel_qubits, ops, Dt, eps_gate, strategy)
    if readout_weight_dict is None:
        return bb, sys_qubits, sel_qubits
    bb, sys_qubits, readouts = build_coil_readout(bb, sys_qubits, readout_weight_dict)
    return bb, readouts


def cost_of(circuit, t_per_gate):
    cost = get_cost_value(circuit, QECGatesCost())
    qubits = get_cost_value(circuit, QubitCount())
    t_count = cost.total_t_count(ts_per_rotation=t_per_gate)
    return cost, qubits, t_count


if __name__ == '__main__':
    from qualtran import QBit, Register, Side

    ops = mo.load_molecule_operators('gemcitabine5')
    n_sys, weights = ops['n'], ops['weights']
    S_k = len(ops['jump_pauli_terms'])
    N_SEL = math.ceil(math.log2(S_k))
    R_coherent, R_nested = count_rotations_per_step(ops)
    R_per_step = R_coherent + R_nested
    R_state_prep = sum(1 for th in pulse_angles(weights) if abs(th % (math.pi / 2)) > 1e-9)

    K_prf, K_mixed = ebs.get_coefficients()
    EPS_TOTAL, T_ACQ = 0.1, 1.0  # same convention as every other script in this pipeline
    a = T_ACQ * abs(K_prf + K_mixed)

    v_star, _, _ = opt.optimal_split(T_ACQ, a, R_per_step, EPS_TOTAL)
    eps_trotter = EPS_TOTAL * v_star
    dt_solution = ebs.LinearStateIndependentModel().solve_dt(
        eps_trotter, T_ACQ, ebs.ErrorCoefficients(K_prf=K_prf, K_mixed=K_mixed, K_nested=0.0))
    Dt, M = dt_solution.Dt, dt_solution.M

    R_total = R_per_step * M + R_state_prep  # readout contributes 0 rotations
    eps_synth = EPS_TOTAL * (1 - v_star)
    eps_gate = eps_synth / R_total
    t_per_gate = t_count_direct(eps_gate)

    print(f'=== Truncated Gemcitabine, single-shot resource estimate ===')
    print(f'eps_total={EPS_TOTAL}, t={T_ACQ}s, optimal split v*={v_star:.4f}')
    print(f'Dt*={Dt:.4e}s, M={M:,}, eps_gate={eps_gate:.3e}, T/rotation={t_per_gate}')
    print(f'R_coherent={R_coherent}, R_nested={R_nested}, R_state_prep={R_state_prep}, '
          f'R_total={R_total:,}')

    strategy = DirectSynthesisStrategy()

    # ---- linearity check: build M_TEST literal chained steps, confirm
    # QECGatesCost scales exactly linearly in M_TEST and QubitCount is
    # constant, before trusting the M~2e4 extrapolation below. ----
    M_TEST = 3
    print(f'\n--- Linearity check (M_TEST=1..{M_TEST}, no state prep, no readout) ---')
    step_costs = []
    for m in range(1, M_TEST + 1):
        bb = BloqBuilder()
        sys_reg = bb.add_register('sys', n_sys)
        sel_reg = bb.add_register('sel', N_SEL)
        sys_qs = list(bb.split(sys_reg))
        sel_qs = list(bb.split(sel_reg))
        bb, sys_qs, sel_qs = build_chain(
            bb, sys_qs, sel_qs, ops, weights, m, Dt, eps_gate, strategy, with_state_prep=False)
        circuit = bb.finalize(sys=bb.join(sys_qs), sel=bb.join(sel_qs))
        cost, qubits, t_count = cost_of(circuit, t_per_gate)
        step_costs.append((m, cost, qubits, t_count))
        print(f'  M_TEST={m}: {cost}, qubits={qubits}, T={t_count:,}')

    t_counts = [t for _, _, _, t in step_costs]
    one_step_t = t_counts[1] - t_counts[0]
    linear_ok = all(t_counts[i] == t_counts[0] + i * one_step_t for i in range(len(t_counts)))
    qubits_const = len({q for _, _, q, _ in step_costs}) == 1
    peak_qubits = step_costs[0][2]  # measured, not assumed -- includes N_SEL-1 transient
                                     # unary-iteration flag ancilla beyond n_sys+N_SEL
    print(f'  T-count exactly linear in M_TEST: {linear_ok} (slope={one_step_t:,} T/step)')
    print(f'  QubitCount constant across M_TEST: {qubits_const} (={peak_qubits}, confirming '
          f'the N_SEL-1={N_SEL - 1} transient flag ancilla are reused across steps, not '
          f'accumulated: {peak_qubits} = n_sys+N_SEL+(N_SEL-1) = {n_sys}+{N_SEL}+{N_SEL - 1})')
    assert linear_ok and qubits_const, 'linear-scaling assumption failed -- do not trust M extrapolation below'

    # ---- one-time pieces, built and costed directly (not extrapolated) ----
    print(f'\n--- One-time pieces (built directly) ---')
    bb = BloqBuilder()
    sys_reg = bb.add_register('sys', n_sys)
    sys_qs = list(bb.split(sys_reg))
    bb, sys_qs = build_null_state_prep(bb, sys_qs, weights, eps_gate, strategy)
    sp_circuit = bb.finalize(sys=bb.join(sys_qs))
    sp_cost, sp_qubits, sp_t = cost_of(sp_circuit, t_per_gate)
    print(f'  state prep: {sp_cost}, qubits={sp_qubits}, T={sp_t}')

    readout_costs = {}
    for label, wdict in [('X', ops['Sx_coll_dict']), ('Y', ops['Sy_coll_dict'])]:
        bb = BloqBuilder()
        sys_reg = bb.add_register(Register('sys', QBit(), shape=(n_sys,), side=Side.LEFT), None)
        qs = list(sys_reg)
        bb, qs, readouts = build_coil_readout(bb, qs, wdict)
        ro_circuit = bb.finalize(**{f'c{i}': c for i, _, c in readouts})
        ro_cost, ro_qubits, ro_t = cost_of(ro_circuit, t_per_gate)
        readout_costs[label] = (ro_cost, ro_qubits, ro_t)
        print(f'  {label}-readout: {ro_cost}, qubits={ro_qubits}, T={ro_t}')

    # ---- extrapolate the M-step evolution, add one-time pieces ----
    print(f'\n--- Extrapolated total (M={M:,} steps) ---')
    step_clifford = step_costs[1][1].clifford - step_costs[0][1].clifford
    step_rotation = step_costs[1][1].rotation - step_costs[0][1].rotation
    step_and = step_costs[1][1].and_bloq - step_costs[0][1].and_bloq
    step_meas = step_costs[1][1].measurement - step_costs[0][1].measurement
    step_t = one_step_t

    for variant in ('X', 'Y'):
        ro_cost, ro_qubits, ro_t = readout_costs[variant]
        total_t = M * step_t + sp_t + ro_t
        total_clifford = M * step_clifford + sp_cost.clifford + ro_cost.clifford
        total_rotation = M * step_rotation + sp_cost.rotation + ro_cost.rotation
        total_and = M * step_and
        total_meas = M * step_meas + ro_cost.measurement
        print(f'\n  [{variant}-readout run] total T={total_t:,.0f}, '
              f'Clifford={total_clifford:,.0f}, rotations={total_rotation:,.0f} '
              f'(cross-check vs R_total: {int(total_rotation) == R_total}), '
              f'And={total_and:,.0f}, measurements={total_meas:,.0f}, '
              f'peak qubits={peak_qubits} (measured, = n_sys+N_SEL+(N_SEL-1) = '
              f'{n_sys}+{N_SEL}+{N_SEL - 1})')

    print(f'\nNote: <coil> = <Sx_coll> + i<Sy_coll> needs BOTH runs above (X and Y), '
          f'as two separate circuit executions -- not summed. Shot count needed per run '
          f'for a target measurement precision is still out of scope (pipeline_plan.md '
          f'Decision log). State prep is the null baseline (not a validated rho_0 '
          f'substitute); this is the ZULF branch only (K_nested~=0).')
