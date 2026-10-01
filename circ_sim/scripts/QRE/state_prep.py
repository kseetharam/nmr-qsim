"""
State-preparation circuit: baseline decision (2026-09-21, pipeline_plan.md
open question 5) is to treat the ZULF protocol's real physical state-prep
step (sudden transfer to a high-temperature deviation state
rho_sud = sum_n w_n S_{n,z}, then a pi/2-about-y pulse -- see qre.tex's
"Reference channels" paragraph and
molecule_operators.build_zulf_protocol_operators) as NULL for baseline
resource-estimation purposes: skip modeling rho_sud altogether and apply the
SAME per-spin y-pulse directly to the all-zero computational basis state
|0...0>, which Qualtran registers already start in. This gives a concrete,
buildable, costed circuit now, in place of the harder (not yet designed)
trajectory/ensemble-sampling machinery needed to reproduce rho_sud's actual
mixed-state statistics -- deferred to a future refinement via approximate
state t-designs (per pipeline_plan.md's state-preparation decision log entry).

Not claimed physically equivalent to rho_0: the resulting product state's
implicit higher-Hamming-weight Pauli content differs from rho_0's purely
weight-1 structure (rho_0 is a SUM of single-spin deviation terms; a pure
product state's density matrix is a PRODUCT, so it also carries weight-2+
cross terms rho_0 does not have). Closing that gap is exactly what a future
t-design step is for; treat this module's output as a placeholder baseline,
not a validated substitute for the rho_0-anchored K_prf/K_mixed coefficients.

Rotation-angle convention: the per-spin pulse angle theta_i=(pi/2)*w_i (same
weights as molecule_operators.build_zulf_protocol_operators,
w_i=gamma_i/gamma_H) implements the physical rotation exp(-i*theta_i*S_y(i))
with S_y=Y/2 (spin-1/2 convention, matching rho0_dict's own derivation:
exp(-i theta S_y) S_z exp(i theta S_y) = cos(theta) S_z + sin(theta) S_x).
Via apply_pauli_exponential's exp(-i*angle*FullPauli) convention this needs
angle=theta_i/2 (checked: exp(-i*(theta/2)*Y)|0> = cos(theta/2)|0> +
sin(theta/2)|1>, the standard Ry(theta) gate convention -- verified
numerically below against a direct kron-product statevector, not just
asserted).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qualtran import BloqBuilder
from qualtran.resource_counting import get_cost_value, QECGatesCost

from coherent_trotter_step import apply_pauli_exponential
from rotation_synthesis_strategy import DirectSynthesisStrategy, RotationSynthesisStrategy


def pulse_angles(weights):
    """Per-spin physical pulse angle theta_i=(pi/2)*w_i, matching
    molecule_operators.build_zulf_protocol_operators exactly."""
    return [(np.pi / 2) * w for w in weights]


def build_null_state_prep(bb, qubits, weights, eps_gate, strategy: RotationSynthesisStrategy):
    """Baseline state prep: qubits already start at |0...0> (nothing to do
    for the "sudden transfer" step, treated as null); apply the per-spin
    Sy pulse exp(-i*theta_i*S_y(i)) directly to that. Returns (bb, qubits)."""
    n = len(qubits)
    thetas = pulse_angles(weights)
    for i in range(n):
        pauli_string = 'I' * i + 'Y' + 'I' * (n - i - 1)
        bb, qubits = apply_pauli_exponential(bb, qubits, pauli_string, thetas[i] / 2,
                                              eps_gate, strategy)
    return bb, qubits


if __name__ == '__main__':
    import molecule_operators as mo
    from gate_synthesis_budget import t_count_direct

    ops = mo.load_molecule_operators('gemcitabine5')
    n, weights = ops['n'], ops['weights']

    EPS_GATE = 1e-9  # arbitrary illustrative precision; not yet folded into
                      # gate_synthesis_budget's R_total (one-time cost, not
                      # per-Trotter-step -- negligible next to R_total ~ 1e7)
    strategy = DirectSynthesisStrategy()

    bb = BloqBuilder()
    qreg = bb.add_register('q', n)
    qubits = list(bb.split(qreg))
    bb, qubits = build_null_state_prep(bb, qubits, weights, EPS_GATE, strategy)
    qreg = bb.join(qubits)
    circuit = bb.finalize(q=qreg)

    cost = get_cost_value(circuit, QECGatesCost())
    t_per_gate = t_count_direct(EPS_GATE)
    t_count = cost.total_t_count(ts_per_rotation=t_per_gate)
    n_generic = sum(1 for theta in pulse_angles(weights) if abs(theta % (np.pi / 2)) > 1e-9)
    print(f'State-prep pulse circuit ({n} qubits, {n} rotations requested): {cost}')
    print(f'  T count (eps_gate={EPS_GATE:.1e}, {t_per_gate} T/rotation) = {t_count}')
    print(f'  {n - n_generic} of {n} pulses hit an exact Clifford angle '
          f'(theta_i a multiple of pi/2 -- happens whenever w_i=1, i.e. a proton, '
          f'since theta_i=(pi/2)*w_i) and cost 0 T; the remaining {n_generic} are '
          f'genuinely synthesized.')
    print(f'  cross-check vs hand count ({n_generic} x t_per_gate) = {n_generic * t_per_gate} '
          f'(match: {t_count == n_generic * t_per_gate})')

    # Correctness check: compare the state prepared from |0...0> (i.e. this
    # circuit's unitary matrix applied to the all-zero basis vector) against
    # explicit Ry(theta_i) products on |0...0>, built independently with
    # plain numpy (not reusing apply_pauli_exponential's own logic).
    U_circuit = circuit.tensor_contract()
    psi_circuit = U_circuit[:, 0]
    psi_expected = np.array([1.0 + 0j])
    for theta in pulse_angles(weights):
        ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                       [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
        psi_expected = np.kron(psi_expected, ry @ np.array([1, 0], dtype=complex))
    idx = int(np.argmax(np.abs(psi_expected)))
    phase = psi_circuit[idx] / psi_expected[idx]
    diff = float(np.max(np.abs(psi_circuit - phase * psi_expected)))
    print(f'  max amplitude diff vs direct Ry(theta_i) product state = {diff:.3e} '
          f'(phase factor {phase:.6f}, should have |phase|=1)')
