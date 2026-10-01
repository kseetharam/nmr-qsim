"""
Minimal end-to-end smoke test for the Qualtran environment (pipeline_plan.md
open question 1): confirms BloqBuilder composite-bloq construction and
resource counting (QECGatesCost, including rotation-synthesis T-cost) both
work, before building anything real (edge-colored Heisenberg/Zeeman
fragments, nested-Trotter jump-operator gadgets).

Environment: qualtran 0.7.0, installed into
/Users/luismartinezmartinez/pyenvs/qiskit_backends (Python 3.11.5).
"""
from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Rz
from qualtran.resource_counting import QECGatesCost, get_cost_value


def build_toy_circuit():
    bb = BloqBuilder()
    q0 = bb.add_register('q0', 1)
    q1 = bb.add_register('q1', 1)
    q0 = bb.add(Hadamard(), q=q0)
    q0 = bb.add(Rz(angle=0.31415), q=q0)
    q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)
    return bb.finalize(q0=q0, q1=q1)


if __name__ == '__main__':
    cbloq = build_toy_circuit()
    print('CompositeBloq built OK:', cbloq)

    gate_cost = get_cost_value(cbloq, QECGatesCost())
    print('QECGatesCost:', gate_cost)
    print('T count:', gate_cost.total_t_count())
