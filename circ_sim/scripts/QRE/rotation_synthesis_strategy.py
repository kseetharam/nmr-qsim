"""
Swappable rotation-synthesis strategy for the QRE pipeline's Qualtran bloqs
(pipeline_plan.md open question 7, "Gate-synthesis cost model").

Every Pauli-string exponential e^{-i theta P} in this pipeline (coherent
Heisenberg/Zeeman fragments, nested-Trotter jump-operator Pauli terms)
compiles to a CNOT ladder (Clifford, free) wrapping a single Rz(theta) --
so ALL of the pipeline's precision-dependent gate cost lives in how that one
Rz per Pauli term gets synthesized. This module makes that choice a
swappable strategy object rather than a hardcoded bloq, mirroring the
ErrorAccumulationModel pattern in error_budget_solver.py: build the Trotter
circuit against RotationSynthesisStrategy's interface, and switching
approaches (or comparing them) later is a matter of passing a different
strategy instance, not rewriting the circuit-building code.

Decision (2026-09-19/20): DirectSynthesisStrategy is the baseline for
resource-estimation purposes. PhaseGradientStrategy is implemented and
usable, but NOT recommended yet: benchmarked here (compare_strategies),
phase-gradient synthesis via Qualtran's current bloqs (PhaseGradientState +
ZPowConstViaPhaseGradient) costs *more* T gates than direct synthesis at
representative rotation counts/precisions -- e.g. N=20 rotations at
eps=1e-8: 220 T (direct) vs 2538 T (phase-gradient). This is because each
phase-gradient addition costs O(bitwidth) Toffolis (a ripple-carry-style
controlled adder), which at Toffoli~4T is already more expensive per
rotation than a single direct-synthesis Rz's ~11 T at this precision --
the usual "amortize a shared one-time cost over many rotations" argument
does not save this, since the *marginal* per-rotation cost is what's
compared, not just the shared setup. This may not hold at extreme
precisions/rotation counts or with a different adder, but there's no
reason to assume it flips without checking, so it remains an open,
not a resolved, question -- exactly what this swappable abstraction is
for.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from qualtran import Bloq, BloqBuilder, SoquetT
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.rotations.zpow_via_phase_gradient import ZPowConstViaPhaseGradient
from qualtran.resource_counting import get_cost_value, QECGatesCost


class RotationSynthesisStrategy(ABC):
    """Interface: prepare any shared state once, apply many rotations
    against it, then free the shared state. DirectSynthesisStrategy's
    context is trivial (no shared state); PhaseGradientStrategy's is the
    shared phase-gradient ancilla register."""

    @abstractmethod
    def prepare(self, bb: BloqBuilder, eps: float) -> Tuple[BloqBuilder, Any]:
        """Allocate any shared state needed for a batch of rotations at
        precision eps. Returns (bb, context)."""

    @abstractmethod
    def apply(self, bb: BloqBuilder, context: Any, q: SoquetT, angle: float,
              eps: float) -> Tuple[BloqBuilder, Any, SoquetT]:
        """Apply Rz(angle) (up to global phase) to qubit q. Returns
        (bb, context, q)."""

    @abstractmethod
    def finish(self, bb: BloqBuilder, context: Any) -> BloqBuilder:
        """Free/uncompute any shared state from prepare(). Returns bb."""

    def apply_rotations(self, bb: BloqBuilder, q: SoquetT, angles: List[float],
                         eps: float) -> Tuple[BloqBuilder, SoquetT]:
        """Convenience: prepare, apply every angle in sequence to q, finish."""
        bb, context = self.prepare(bb, eps)
        for angle in angles:
            bb, context, q = self.apply(bb, context, q, angle, eps)
        bb = self.finish(bb, context)
        return bb, q


@dataclass(frozen=True)
class DirectSynthesisStrategy(RotationSynthesisStrategy):
    """Baseline (decision 2026-09-19/20): each Rz directly T-synthesized,
    qualtran.bloqs.basic_gates.Rz's default cost model
    (ceil(1.149*log2(1/eps)+9.2) T per rotation)."""

    def prepare(self, bb, eps):
        return bb, None

    def apply(self, bb, context, q, angle, eps):
        q = bb.add(Rz(angle=angle, eps=eps), q=q)
        return bb, context, q

    def finish(self, bb, context):
        return bb


@dataclass(frozen=True)
class PhaseGradientStrategy(RotationSynthesisStrategy):
    """Shared phase-gradient ancilla register (allocated once per
    apply_rotations batch), each rotation a controlled-add into it via
    ZPowConstViaPhaseGradient. Implemented and usable, but NOT the
    recommended default yet -- see module docstring; benchmarked worse
    than direct synthesis at the precisions/counts checked so far."""

    def prepare(self, bb, eps):
        b = ZPowConstViaPhaseGradient.from_precision(1.0, eps=eps).phase_grad_bitsize
        phase_grad = bb.add(PhaseGradientState(bitsize=b))
        return bb, (phase_grad, b)

    def apply(self, bb, context, q, angle, eps):
        phase_grad, b = context
        # Z^t == Rz(pi*t) up to a global phase, irrelevant for expectation-value estimation.
        zpow = ZPowConstViaPhaseGradient(exponent=angle / 3.141592653589793, phase_grad_bitsize=b)
        q, phase_grad = bb.add(zpow, q=q, phase_grad=phase_grad)
        return bb, (phase_grad, b), q

    def finish(self, bb, context):
        phase_grad, _b = context
        bb.free(phase_grad)
        return bb


def compare_strategies(angles: List[float], eps: float):
    """Build the same batch of rotations under both strategies and report
    QECGatesCost side by side."""
    results = {}
    for name, strategy in [('direct', DirectSynthesisStrategy()),
                            ('phase_gradient', PhaseGradientStrategy())]:
        bb = BloqBuilder()
        q = bb.add_register('q', 1)
        bb, q = strategy.apply_rotations(bb, q, angles, eps)
        circuit = bb.finalize(q=q)
        cost = get_cost_value(circuit, QECGatesCost())
        results[name] = cost
    return results


if __name__ == '__main__':
    N = 20
    EPS = 1e-8
    angles = [0.1 * (i + 1) for i in range(N)]

    results = compare_strategies(angles, EPS)
    print(f'N={N} rotations, eps={EPS:.0e}:')
    for name, cost in results.items():
        print(f'  {name:15s}  {cost}  total T = {cost.total_t_count()}')
