"""
Swappable Trotter-order abstraction for the trajectory-based QRE pipeline
(README.md's "Pipeline architecture" item 3 / Decision log: "first-order Lie
splitting is the working baseline, behind an abstraction so Strang is a
config swap, not a rewrite").

Mirrors ../rotation_synthesis_strategy.py's pattern (an ABC with swappable
concrete implementations, so downstream code is written against the
interface and switching implementations later is a config choice), but the
thing being made swappable here is different: RotationSynthesisStrategy
swaps how a single Rz gets synthesized; TrotterScheme swaps how one outer
step's coherent (H0) and anisotropic (noise) PARTS are combined --
../coherent_trotter_step.build_coherent_trotter_step and
../traj_based/anisotropic_trotter_step.build_anisotropic_trotter_step are
both reused unmodified by every scheme; only the order/time-slicing in which
a TrotterScheme calls them differs. There is no shared quantum resource to
prepare/finish here (unlike RotationSynthesisStrategy's phase-gradient
register), so the interface is a single apply_step method rather than
prepare/apply/finish.

LieScheme (white_noise_trotter_1.pdf Eq. 3): U = exp(-i*H0*Dt) @
exp(-i*sum_j dW_j*V_j), i.e. the anisotropic part acts on the state FIRST,
the full coherent step SECOND -- matches this pipeline's own numerical
validation exactly (plot3_lie_eq8_verification.py /
plot7_group_trotter_fid_convergence.py's U = U_H @ U_V convention), not an
arbitrary re-derivation of the ordering here.

StrangScheme (Eq. 4): U = exp(-i*H0*Dt/2) @ exp(-i*sum_j dW_j*V_j) @
exp(-i*H0*Dt/2) -- H0 at half-step, the full anisotropic step, H0 at
half-step again. Doubles the coherent-step rotation count per outer step
relative to Lie (two half-strength applications instead of one full-strength
one); the anisotropic-step rotation count (105 pooled terms for Gemcitabine)
is unchanged and applied once per outer step at the FULL dW -- only H0's
exposure is split, not the noise increment's own variance (still drawn for
the full Dt).

Caught by the __main__ self-test's T-count cross-check, not assumed: since
DT=(1/J_max)/2 is tied to J_max by construction, the F0-C0 edge's own three
coherent-step terms (its XX/YY/ZZ) land EXACTLY on a special Rz angle, not
approximately. At full DT (LieScheme), their Rz argument is exactly pi/2 --
a Clifford (S-gate) angle, 0 T. At half DT (StrangScheme, applied twice),
the SAME three terms' Rz argument is exactly pi/4 -- the T-gate's own native
angle, costing exactly 1 T each rather than the generic
eps-dependent-synthesis cost (44 T at eps_gate=1e-9). A naive
n_rotations * t_per_gate hand count would be off by 3*44=132 T for Lie and
6*43=258 T for Strang; the self-test's expected_t_cost accounts for this
exactly rather than reporting a T-count that silently doesn't match its own
rotation count.
"""
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.dirname(HERE)
for _p in (HERE, QRE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coherent_trotter_step import build_coherent_trotter_step  # noqa: E402
from anisotropic_trotter_step import build_anisotropic_trotter_step  # noqa: E402


def _count_coherent_terms(coherent_dicts):
    """Rotation count for one application of the coherent step -- raw Pauli-
    term count across all fragments, matching build_coherent_trotter_step's
    own one-rotation-per-term loop (no group fusion there either)."""
    return sum(len(frag) for frag in coherent_dicts)


class TrotterScheme(ABC):
    """Interface: combine one outer Trotter step's coherent and anisotropic
    parts into a single per-step gate sequence, for a step of duration Dt
    and one trajectory's already-drawn dW."""

    @abstractmethod
    def apply_step(self, bb, qubits, coherent_dicts, anisotropic_data, dW, Dt, eps_gate, strategy):
        """Returns (bb, qubits, n_rotations)."""


@dataclass(frozen=True)
class LieScheme(TrotterScheme):
    """First-order (Eq. 3): anisotropic part first, then the full coherent step."""

    def apply_step(self, bb, qubits, coherent_dicts, anisotropic_data, dW, Dt, eps_gate, strategy):
        bb, qubits, n_aniso = build_anisotropic_trotter_step(
            bb, qubits, anisotropic_data, dW, eps_gate, strategy)
        bb, qubits = build_coherent_trotter_step(
            bb, qubits, coherent_dicts, Dt, eps_gate, strategy)
        return bb, qubits, n_aniso + _count_coherent_terms(coherent_dicts)


@dataclass(frozen=True)
class StrangScheme(TrotterScheme):
    """Second-order (Eq. 4): H0 half-step, full anisotropic step, H0 half-step."""

    def apply_step(self, bb, qubits, coherent_dicts, anisotropic_data, dW, Dt, eps_gate, strategy):
        bb, qubits = build_coherent_trotter_step(
            bb, qubits, coherent_dicts, Dt / 2, eps_gate, strategy)
        bb, qubits, n_aniso = build_anisotropic_trotter_step(
            bb, qubits, anisotropic_data, dW, eps_gate, strategy)
        bb, qubits = build_coherent_trotter_step(
            bb, qubits, coherent_dicts, Dt / 2, eps_gate, strategy)
        return bb, qubits, n_aniso + 2 * _count_coherent_terms(coherent_dicts)


if __name__ == '__main__':
    import numpy as np
    from scipy.linalg import expm

    from qualtran import BloqBuilder
    from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

    import hermitian_generators as hg
    import molecule_operators as mo
    from rotation_synthesis_strategy import DirectSynthesisStrategy
    from gate_synthesis_budget import t_count_direct

    mol_ops = mo.load_molecule_operators('gemcitabine5')
    data = hg.build_anisotropic_data(mol_ops)
    n = data['n']
    n_gen = len(data['V_dicts'])
    coherent_dicts = mol_ops['coherent_dicts']

    J_MAX_HZ = 226.85
    DT = (1.0 / J_MAX_HZ) / 2.0
    rng = np.random.default_rng(0)
    dW = rng.normal(scale=np.sqrt(DT), size=n_gen)

    EPS_GATE = 1e-9
    strategy = DirectSynthesisStrategy()

    # ---- dense references, built from the SAME per-term sequential order
    # each scheme's circuit uses (not the fully-exact one-shot exponentials
    # -- each piece's own approximation is already validated separately in
    # anisotropic_trotter_step.py / the ancilla pipeline's coherent-step
    # checks; what's new here is whether apply_step's ASSEMBLY/ordering is
    # correct) ----
    D = 2 ** n

    def dense_coherent(coherent_dicts, dt):
        U = np.eye(D, dtype=complex)
        for frag in coherent_dicts:
            for pstr, coeff in frag.items():
                U = expm(-1j * dt * coeff.real * hg._pauli_mat(pstr)) @ U
        return U

    def dense_anisotropic(data, dW):
        coefs = data['coef_matrix'] @ dW
        U = np.eye(D, dtype=complex)
        for idxs in data['group_term_indices']:
            for idx in idxs:
                U = expm(-1j * coefs[idx] * hg._pauli_mat(data['unique_strings'][idx])) @ U
        return U

    U_aniso = dense_anisotropic(data, dW)
    U_coherent_full = dense_coherent(coherent_dicts, DT)
    U_coherent_half = dense_coherent(coherent_dicts, DT / 2)

    expected = {
        'Lie': U_coherent_full @ U_aniso,
        'Strang': U_coherent_half @ U_aniso @ U_coherent_half,
    }

    def classify_angle(rz_arg, tol=1e-9):
        """Rz(rz_arg)'s exact-angle cost class: 0 T at a Clifford angle
        (multiple of pi/2), 1 T at the T-gate's own native angle (pi/4 mod
        pi/2), else the generic eps-dependent synthesis cost. Checked here
        rather than assumed because DT=(1/J_max)/2 is tied to J_max by
        construction, and the strongest coupling's own rotation angle turns
        out to land exactly on one of these special values -- see below."""
        rem = rz_arg % (np.pi / 2)
        rem = min(rem, np.pi / 2 - rem)
        if rem < tol:
            return 0
        if abs(rem - np.pi / 4) < tol:
            return 1
        return None  # generic: costs t_per_gate

    t_per_gate = t_count_direct(EPS_GATE)

    def expected_t_cost(dt_applications):
        """dt_applications: list of dt values, one per coherent-step
        application in this scheme (e.g. [DT] for Lie, [DT/2, DT/2] for
        Strang). Returns (expected_T, n_special) summed over every
        coherent-fragment term at each application, plus the anisotropic
        part's 105 pooled terms (all generic: random Gaussian coefficients
        essentially never land on a special angle)."""
        total, n_special = 0, 0
        for dt in dt_applications:
            for frag in coherent_dicts:
                for pstr, coeff in frag.items():
                    special = classify_angle(2 * dt * coeff.real)
                    if special is None:
                        total += t_per_gate
                    else:
                        total += special
                        n_special += 1
        total += len(data['unique_strings']) * t_per_gate  # anisotropic part, always generic
        return total, n_special

    dt_applications = {'Lie': [DT], 'Strang': [DT / 2, DT / 2]}

    for name, scheme in [('Lie', LieScheme()), ('Strang', StrangScheme())]:
        bb = BloqBuilder()
        qreg = bb.add_register('q', n)
        qubits = list(bb.split(qreg))
        bb, qubits, n_rot = scheme.apply_step(
            bb, qubits, coherent_dicts, data, dW, DT, EPS_GATE, strategy)
        qreg = bb.join(qubits)
        circuit = bb.finalize(q=qreg)

        cost = get_cost_value(circuit, QECGatesCost())
        n_qubits = get_cost_value(circuit, QubitCount())
        t_count = cost.total_t_count(ts_per_rotation=t_per_gate)
        t_expected, n_special = expected_t_cost(dt_applications[name])

        U_circuit = circuit.tensor_contract()
        U_ref = expected[name]
        flat_ref = U_ref.flatten()
        i = int(np.argmax(np.abs(flat_ref)))
        phase = U_circuit.flatten()[i] / flat_ref[i]
        diff = float(np.max(np.abs(U_circuit - phase * U_ref)))

        print(f'{name}Scheme: n_rotations(hand-called)={n_rot}, qubits={n_qubits}, '
              f'{n_special} of the coherent-step calls land on a special (non-generic) '
              f'angle -- a direct consequence of DT being tied to J_max, not a fluke.')
        print(f'  T={t_count} (Qualtran) vs. {t_expected} (hand count including the '
              f'{n_special} special-angle rotations at their exact cost, not '
              f'{t_per_gate}T each) -- match: {t_count == t_expected}')
        print(f'  max|U_circuit - phase*U_ref| = {diff:.3e} (phase={phase:.6f})')
        assert diff < 1e-9, f'{name}Scheme circuit disagrees with its expected dense assembly'
        assert n_qubits == n, f'{name}Scheme should need exactly {n} qubits (no ancilla), got {n_qubits}'
        assert t_count == t_expected, f'{name}Scheme T-count does not match the special-angle-aware hand count'
    print('\nBoth schemes match their expected dense assembly exactly; no ancilla in either; '
          'T-counts match a hand count that accounts for the special-angle rotations exactly.')
