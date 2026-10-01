"""
Anisotropic-part ("noise-driven") Trotter-step gadget for the trajectory-
based QRE pipeline (README.md's "Pipeline architecture" item 2) -- the
direct counterpart to ../nested_trotter_step.py, but with no ancilla
dilation/dispatch register at all: realizes one trajectory's already-drawn,
already-classical Delta W_j values as a circuit acting purely on the system
register.

Structure: aggregate each of the 105 pooled unique Pauli strings'
coefficient across all 25 generators (hermitian_generators.py's
coef_matrix @ dW), then apply every one of the resulting rotations
sequentially via ../coherent_trotter_step.py's apply_pauli_exponential --
the same primitive that gadget uses for H0's own terms -- walking the 16
mutually-commuting groups in the fixed order hermitian_generators.py
returns. Terms within a group commute exactly (regardless of qubit-support
overlap: e^{i*sum_k a_k*P_k} = prod_k e^{i*a_k*P_k} whenever the P_k
commute), so their relative order carries zero error; the across-group
order is where the accumulated first-order Trotter error (already
characterized numerically in
../../trajectory_based_sim/testing/plot4_group_trotter_convergence.py and
plot7_group_trotter_fid_convergence.py) comes from.

Deliberately 105 individual rotation gadgets, not 16 group-fused ones (see
hermitian_generators.py's module docstring and traj_based/README.md's
Decision log): a joint-diagonalization circuit for each commuting clique
would only reduce Clifford/CNOT count, never T-count, since T-count here is
driven purely by rotation count and every pooled term carries its own
independent, generically-distinct numeric coefficient (Delta W_j are iid
Gaussian) -- so there's no fewer-than-105-rotations version of this circuit
to build.

Differs from ../nested_trotter_step.py in exactly two ways: (1) no ancilla
register of any kind -- every rotation acts only on the system qubits, and
(2) no hop-term Xi_j(phi)=cos(phi)X-sin(phi)Y ancilla-conjugation trick --
that existed there only to encode a COMPLEX non-Hermitian-jump-operator
coefficient via an ancilla rotation; every coefficient here is already real
(a Hermitian generator decomposed in the real Pauli basis), so each term is
just a bare apply_pauli_exponential call at angle = its aggregate real
coefficient.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.dirname(HERE)
for _p in (HERE, QRE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coherent_trotter_step import apply_pauli_exponential  # noqa: E402
import hermitian_generators as hg  # noqa: E402


def build_anisotropic_trotter_step(bb, qubits, anisotropic_data, dW, eps_gate, strategy):
    """One first-order Trotter step of the anisotropic part, for a single
    trajectory's already-drawn dW (length-n_gen real array of Delta W_j,
    SAME order as anisotropic_data['V_dicts']'s columns in coef_matrix).

    Returns (bb, qubits, n_rotations)."""
    coefs = anisotropic_data['coef_matrix'] @ dW  # (n_terms,) real: aggregate per pooled term
    unique_strings = anisotropic_data['unique_strings']
    n_rot = 0
    for idxs in anisotropic_data['group_term_indices']:
        for idx in idxs:
            bb, qubits = apply_pauli_exponential(
                bb, qubits, unique_strings[idx], coefs[idx], eps_gate, strategy)
            n_rot += 1
    return bb, qubits, n_rot


if __name__ == '__main__':
    import numpy as np
    from scipy.linalg import expm

    from qualtran import BloqBuilder
    from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

    import molecule_operators as mo
    from rotation_synthesis_strategy import DirectSynthesisStrategy
    from gate_synthesis_budget import t_count_direct

    mol_ops = mo.load_molecule_operators('gemcitabine5')
    data = hg.build_anisotropic_data(mol_ops)
    n = data['n']
    n_gen = len(data['V_dicts'])
    n_terms = len(data['unique_strings'])

    # one concrete trajectory's draw, at this pipeline's validated (delta_t,
    # T) operating point (traj_based/README.md's Numerical validation section)
    J_MAX_HZ = 226.85
    DT = (1.0 / J_MAX_HZ) / 2.0
    rng = np.random.default_rng(0)
    dW = rng.normal(scale=np.sqrt(DT), size=n_gen)

    EPS_GATE = 1e-9
    strategy = DirectSynthesisStrategy()

    bb = BloqBuilder()
    qreg = bb.add_register('q', n)
    qubits = list(bb.split(qreg))
    bb, qubits, n_rot = build_anisotropic_trotter_step(bb, qubits, data, dW, EPS_GATE, strategy)
    qreg = bb.join(qubits)
    circuit = bb.finalize(q=qreg)

    cost = get_cost_value(circuit, QECGatesCost())
    n_qubits = get_cost_value(circuit, QubitCount())
    t_per_gate = t_count_direct(EPS_GATE)
    t_count = cost.total_t_count(ts_per_rotation=t_per_gate)

    print(f'n_rotations = {n_rot} (expect {n_terms} pooled terms, not {len(data["groups"])} groups)')
    print(f'qubits = {n_qubits} (expect {n}: no ancilla of any kind)')
    print(f'{cost}')
    print(f'T count (eps_gate={EPS_GATE:.1e}, {t_per_gate} T/rotation) = {t_count}')
    print(f'  cross-check vs hand count ({n_terms} x {t_per_gate}) = {n_terms * t_per_gate} '
          f'(match: {t_count == n_terms * t_per_gate})')

    # Correctness check: this circuit's tensor-contracted unitary vs. a dense
    # reference built via the IDENTICAL sequential order (same groups, same
    # term order within each group) -- not the fully-exact channel, since
    # the circuit is only ever meant to implement this specific first-order
    # approximation (already validated against the exact channel, densely,
    # in plot4/plot7). Mirrors state_prep.py's tensor_contract()-based check.
    U_circuit = circuit.tensor_contract()

    D = 2 ** n
    U_ref = np.eye(D, dtype=complex)
    coefs = data['coef_matrix'] @ dW
    for idxs in data['group_term_indices']:
        for idx in idxs:
            P_dense = hg._pauli_mat(data['unique_strings'][idx])
            U_ref = expm(-1j * coefs[idx] * P_dense) @ U_ref

    flat_ref = U_ref.flatten()
    i = int(np.argmax(np.abs(flat_ref)))
    phase = U_circuit.flatten()[i] / flat_ref[i]
    diff = float(np.max(np.abs(U_circuit - phase * U_ref)))
    print(f'\nmax|U_circuit - phase*U_ref| = {diff:.3e} (phase={phase:.6f}, |phase| should be 1)')
    assert diff < 1e-9, 'circuit unitary disagrees with the dense sequential-group reference'
    print('Matches the dense sequential-group construction (plot4/plot7\'s scheme) exactly.')
