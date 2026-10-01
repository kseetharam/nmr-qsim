"""
End-to-end trajectory-based resource estimate for truncated Gemcitabine,
mirroring ../full_pipeline_estimate.py's role for the ancilla pipeline:
state prep -> M x (coherent + anisotropic Trotter step) -> readout, composed
from every gadget this sub-pipeline has separately built (hermitian_generators.py,
anisotropic_trotter_step.py, trotter_scheme.py, plus ../state_prep.py and
../coil_readout.py reused unmodified).

Per traj_based/README.md's Decision log, this phase assumes the
hyperparameters (N trajectories, M Trotter steps, longest evolution time T
hence Dt=T/M, eps_gate, Trotter order) are externally given -- the
error-budget derivation that would produce them is explicitly out of scope
here (unlike ../full_pipeline_estimate.py, which DOES call
error_budget_solver/optimal_error_budget_split; this script deliberately
does not). The example values below (Dt=(1/J_max)/2, T=2s -> M=906) are the
already-validated operating point from
../../trajectory_based_sim/testing/plot6_exact_fid_spectrum_refined.py /
plot7_group_trotter_fid_convergence.py, not re-derived here.

Three things this script checks that ../full_pipeline_estimate.py either
didn't need to (no randomness there) or handles differently (single fixed
time, not an FID):

1. Linearity in M -- same discipline as the ancilla pipeline (a literal
   M_TEST=1,2,3-step chain, QECGatesCost confirmed to scale exactly
   linearly, QubitCount confirmed constant) -- but here EVERY step also
   draws fresh classical randomness (dW), so this also implicitly re-checks
   (at the assembled-chain level, not just one isolated step) that gate
   count doesn't depend on which numbers were drawn.

2. Trajectory-independence, explicitly -- not present in the ancilla
   pipeline at all, since it has no per-shot randomness. The same M_TEST
   chain is rebuilt with an independent random seed; QECGatesCost/QubitCount
   must match exactly. This is the traj_based/README.md "What's actually new
   here" point-4 resolution, checked here at the assembled-chain level
   (anisotropic_trotter_step.py's own __main__ already checked it for one
   isolated step).

3. FID-depth aggregation (README.md's Pipeline architecture item 4): an FID
   needs the observable at M different depths, each requiring a SEPARATE
   circuit execution (state prep -> m steps -> readout, for m=1..M) since a
   real circuit cannot non-destructively checkpoint mid-execution. Total
   cost per trajectory is therefore sum_{m=1}^{M} (state_prep + m*step +
   readout) = M*(state_prep+readout) + step*M*(M+1)/2, not just M*step +
   state_prep + readout (the ../full_pipeline_estimate.py formula, correct
   there because it targets one fixed final time, not a full FID). The grand
   total additionally multiplies by N trajectories and is reported
   separately for the X- and Y-readout variants (not summed -- same
   structural fact as ../full_pipeline_estimate.py: <coil> needs both as
   separate circuit executions).

Also demonstrated directly, not just asserted: swapping LieScheme for
StrangScheme is a one-line change (trotter_scheme.py's whole reason for
existing) -- both are run through the same linearity/trajectory-independence
checks and reported side by side.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.dirname(HERE)
for _p in (HERE, QRE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qualtran import BloqBuilder, QBit, Register, Side
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

import hermitian_generators as hg  # noqa: E402
import molecule_operators as mo  # noqa: E402
from coil_readout import build_coil_readout  # noqa: E402
from gate_synthesis_budget import t_count_direct  # noqa: E402
from rotation_synthesis_strategy import DirectSynthesisStrategy  # noqa: E402
from state_prep import build_null_state_prep  # noqa: E402
from trotter_scheme import LieScheme, StrangScheme  # noqa: E402


def build_chain(bb, qubits, coherent_dicts, anisotropic_data, scheme, rng, n_steps, Dt,
                 eps_gate, strategy, with_state_prep, weights=None, readout_weight_dict=None):
    """State prep (optional) -> n_steps Trotter steps, EACH with its own
    freshly-drawn dW -> readout (optional, X- or Y-basis depending on
    readout_weight_dict). Returns (bb, readouts) if readout_weight_dict
    given, else (bb, qubits)."""
    if with_state_prep:
        bb, qubits = build_null_state_prep(bb, qubits, weights, eps_gate, strategy)
    n_gen = len(anisotropic_data['V_dicts'])
    for _ in range(n_steps):
        dW = rng.normal(scale=np.sqrt(Dt), size=n_gen)
        bb, qubits, _ = scheme.apply_step(
            bb, qubits, coherent_dicts, anisotropic_data, dW, Dt, eps_gate, strategy)
    if readout_weight_dict is None:
        return bb, qubits
    bb, qubits, readouts = build_coil_readout(bb, qubits, readout_weight_dict)
    return bb, readouts


def cost_of(circuit, t_per_gate):
    cost = get_cost_value(circuit, QECGatesCost())
    qubits = get_cost_value(circuit, QubitCount())
    t_count = cost.total_t_count(ts_per_rotation=t_per_gate)
    return cost, qubits, t_count


if __name__ == '__main__':
    mol_ops = mo.load_molecule_operators('gemcitabine5')
    n_sys, weights = mol_ops['n'], mol_ops['weights']
    data = hg.build_anisotropic_data(mol_ops)
    coherent_dicts = mol_ops['coherent_dicts']

    # ---- hyperparameters assumed externally given (Decision log) ----
    J_MAX_HZ = 226.85
    DT = (1.0 / J_MAX_HZ) / 2.0
    T_TOTAL = 2.0
    N_T = round(T_TOTAL / DT)  # total time points including t=0, matching plot6/plot7's convention
    M = N_T - 1  # propagation steps: t=0 needs no step
    EPS_GATE = 1e-9
    N_EXAMPLES = [100, 200, 400]  # illustrative; matches plot7's own N sweep
    t_per_gate = t_count_direct(EPS_GATE)

    print(f'=== Trajectory-based Gemcitabine resource estimate ===')
    print(f'Dt={DT:.4e}s, T={T_TOTAL}s -> M={M}, eps_gate={EPS_GATE:.1e}, {t_per_gate} T/rotation')
    print('(hyperparameters assumed given for this phase, not derived from an error budget)')

    strategy = DirectSynthesisStrategy()

    for scheme_name, scheme in [('Lie', LieScheme()), ('Strang', StrangScheme())]:
        print(f'\n{"=" * 20} {scheme_name}Scheme {"=" * 20}')

        # ---- linearity check: M_TEST literal chained steps, fresh dW every
        # step, seed fixed per M_TEST for reproducibility ----
        M_TEST = 3
        print(f'--- Linearity check (M_TEST=1..{M_TEST}, seed=0, no state prep/readout) ---')
        step_costs = []
        for m in range(1, M_TEST + 1):
            bb = BloqBuilder()
            qreg = bb.add_register('q', n_sys)
            qubits = list(bb.split(qreg))
            rng = np.random.default_rng(0)
            bb, qubits = build_chain(bb, qubits, coherent_dicts, data, scheme, rng, m, DT,
                                      EPS_GATE, strategy, with_state_prep=False)
            circuit = bb.finalize(q=bb.join(qubits))
            cost, qubits_n, t_count = cost_of(circuit, t_per_gate)
            step_costs.append((m, cost, qubits_n, t_count))
            print(f'  M_TEST={m}: {cost}, qubits={qubits_n}, T={t_count:,}')

        t_counts = [t for _, _, _, t in step_costs]
        one_step_t = t_counts[1] - t_counts[0]
        linear_ok = all(t_counts[i] == t_counts[0] + i * one_step_t for i in range(len(t_counts)))
        qubits_const = len({q for _, _, q, _ in step_costs}) == 1
        peak_qubits = step_costs[0][2]
        print(f'  T-count exactly linear in M_TEST: {linear_ok} (slope={one_step_t:,} T/step)')
        print(f'  QubitCount constant across M_TEST: {qubits_const} (={peak_qubits}, no ancilla '
              f'at all -- unlike the dispatch pipeline, there is no transient register to free)')
        assert linear_ok and qubits_const, 'linear-scaling assumption failed'
        assert peak_qubits == n_sys, f'expected exactly {n_sys} qubits (no ancilla), got {peak_qubits}'

        # ---- trajectory-independence check: same M_TEST, different seed ----
        print(f'--- Trajectory-independence check (M_TEST={M_TEST}, seed=1 vs. seed=0) ---')
        bb = BloqBuilder()
        qreg = bb.add_register('q', n_sys)
        qubits = list(bb.split(qreg))
        rng = np.random.default_rng(1)
        bb, qubits = build_chain(bb, qubits, coherent_dicts, data, scheme, rng, M_TEST, DT,
                                  EPS_GATE, strategy, with_state_prep=False)
        circuit_seed1 = bb.finalize(q=bb.join(qubits))
        cost_seed1, qubits_seed1, t_seed1 = cost_of(circuit_seed1, t_per_gate)
        cost_seed0, qubits_seed0, t_seed0 = step_costs[M_TEST - 1][1:4]
        traj_independent = (cost_seed1 == cost_seed0 and qubits_seed1 == qubits_seed0)
        print(f'  seed=0: {cost_seed0}, qubits={qubits_seed0}, T={t_seed0:,}')
        print(f'  seed=1: {cost_seed1}, qubits={qubits_seed1}, T={t_seed1:,}')
        print(f'  identical (gate count independent of which trajectory was drawn): {traj_independent}')
        assert traj_independent, 'gate count should not depend on the trajectory drawn'

        # ---- one-time pieces ----
        print(f'--- One-time pieces ---')
        bb = BloqBuilder()
        qreg = bb.add_register('q', n_sys)
        qubits = list(bb.split(qreg))
        bb, qubits = build_null_state_prep(bb, qubits, weights, EPS_GATE, strategy)
        sp_circuit = bb.finalize(q=bb.join(qubits))
        sp_cost, sp_qubits, sp_t = cost_of(sp_circuit, t_per_gate)
        print(f'  state prep: {sp_cost}, qubits={sp_qubits}, T={sp_t}')

        readout_costs = {}
        for label, wdict in [('X', mol_ops['Sx_coll_dict']), ('Y', mol_ops['Sy_coll_dict'])]:
            bb = BloqBuilder()
            sys_reg = bb.add_register(Register('q', QBit(), shape=(n_sys,), side=Side.LEFT), None)
            qs = list(sys_reg)
            bb, qs, readouts = build_coil_readout(bb, qs, wdict)
            ro_circuit = bb.finalize(**{f'c{i}': c for i, _, c in readouts})
            ro_cost, ro_qubits, ro_t = cost_of(ro_circuit, t_per_gate)
            readout_costs[label] = (ro_cost, ro_qubits, ro_t)
            print(f'  {label}-readout: {ro_cost}, qubits={ro_qubits}, T={ro_t}')

        # ---- FID-depth aggregation: sum_{m=1}^{M} (state_prep + m*step + readout) ----
        print(f'--- FID totals (M={M} depths, per trajectory, then x N) ---')
        step_t = one_step_t
        for variant in ('X', 'Y'):
            ro_cost, ro_qubits, ro_t = readout_costs[variant]
            one_traj_fid_t = M * (sp_t + ro_t) + step_t * M * (M + 1) // 2
            print(f'\n  [{variant}-readout] one trajectory, full FID ({M} depths): '
                  f'T={one_traj_fid_t:,}')
            for N in N_EXAMPLES:
                print(f'    N={N:>4} trajectories: total T = {N * one_traj_fid_t:,.0f}')

    print(f'\nNote: <coil> = <Sx_coll> + i<Sy_coll> needs BOTH X- and Y-readout runs above, as '
          f'separate circuit executions -- not summed. Shot count per (trajectory, depth, '
          f'readout-variant) circuit needed for a target measurement precision is still out of '
          f'scope (same deferred question as ../pipeline_plan.md\'s Decision log). State prep is '
          f'the null baseline (not a validated rho_0 substitute). Hyperparameters above are '
          f'assumed given, not derived from an error budget (see Decision log).')
