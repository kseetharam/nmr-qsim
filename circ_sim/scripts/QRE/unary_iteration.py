"""
Custom unary-iteration SELECT dispatch (qre.tex sec:unary_iteration), built
directly from qualtran.bloqs.mcmt.and_bloq.And rather than the higher-level
qualtran.bloqs.multiplexers.ApplyLthBloq.

Why not just use ApplyLthBloq: it dispatches by wrapping each leaf's WHOLE
bloq in a generic controlled-bloq (controlled on its own internal walk
flag, which is never exposed to the caller). For a leaf built from
Pauli-string exponentials (basis-change Cliffords + a CNOT ladder + one
Rz, per coherent_trotter_step.apply_pauli_exponential), generic controlling
turns every CNOT into a Toffoli and every basis-change H/S into a
controlled-H/controlled-S (T-costly, unlike the Clifford originals) --
empirically, ~7000 spurious Toffolis and ~14x too many "rotation"-costed
bloqs for the Gemcitabine jump-operator payload (see
jump_dispatch_comparison.py). This module instead exposes the walk's own
flag qubit directly, so a leaf's payload can use it AS its own
ancilla-role qubit inside apply_pauli_exponential-style calls -- zero
extra controlling, only the dispatch tree's own And/And-dagger gates cost
anything beyond the payload itself.

Construction (matches qre.tex sec:unary_iteration exactly, generalized
from the worked L=4 example to arbitrary depth): a full binary tree over
2**len(sel_qubits) leaves. The root split is free (uses sel_qubits[0]'s
two polarities directly, no gate). Every other internal node spends
exactly one And to descend into its "b=1" child, recurses, CNOT-flips the
same flag to represent the "b=0" child (Eq. unary_children:
c_right = c XOR c_left), recurses again, then uncomputes with And-dagger.

IMPORTANT, verified-the-hard-way subtlety: the uncomputing And must use
cv2=(1 - compute cv2) for its second control, not the same cv as the
compute And. The CNOT changes what boolean combination the flag equals
(from "ctrl AND b" to "ctrl AND NOT b"), so uncomputing with the compute
step's original cv is a genuine bug -- it asks And-dagger to uncompute a
target value inconsistent with its own controls, and produces a
non-unitary (all-zero, in tensor_contract) circuit. Caught only by
tensor-contracting a trivial (identity-payload) tree and finding it
wasn't the identity; cross-checked against qualtran's own ApplyLthBloq
decomposition, whose And/And-dagger pair uses exactly this cv1=1,cv2=1
(compute) / cv1=1,cv2=0 (uncompute) pattern.

Verified: build_unary_tree with an identity payload is the identity on
the selection register for n=2,3,5; with a target-flipping payload at
every possible leaf for n=3 (L=8), the right (and only the right) leaf
fires for all 8 choices, checked via tensor_contract against the exact
expected unitary -- not just gate counts.
"""
import numpy as np

from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt.and_bloq import And


def _go(bb, l, r, depth, ctrl, ctrl_polarity, sel_qubits, payload_fn, l_iter, r_iter):
    negated = False
    if not ctrl_polarity:
        ctrl = bb.add(XGate(), q=ctrl)
        negated = True
    if r - l == 1:
        if l_iter <= l < r_iter:
            bb, ctrl = payload_fn(bb, ctrl, l)
        if negated:
            ctrl = bb.add(XGate(), q=ctrl)
        return bb, ctrl
    mid = (l + r) // 2
    b = sel_qubits[depth]
    (ctrl, b), f = bb.add(And(cv1=1, cv2=1), ctrl=np.array([ctrl, b]))
    bb, f = _go(bb, mid, r, depth + 1, f, True, sel_qubits, payload_fn, l_iter, r_iter)  # b=1 half
    ctrl, f = bb.add(CNOT(), ctrl=ctrl, target=f)
    bb, f = _go(bb, l, mid, depth + 1, f, True, sel_qubits, payload_fn, l_iter, r_iter)  # b=0 half
    ctrl, b = bb.add(And(cv1=1, cv2=0, uncompute=True), ctrl=np.array([ctrl, b]), target=f)
    sel_qubits[depth] = b
    if negated:
        ctrl = bb.add(XGate(), q=ctrl)
    return bb, ctrl


def build_unary_tree(bb, sel_qubits, payload_fn, l_iter, r_iter):
    """Unary-iteration dispatch over a full 2**n-leaf tree (n=len(sel_qubits)).

    payload_fn(bb, flag, leaf_idx) -> (bb, flag) is called, for every
    leaf_idx in [l_iter, r_iter), with flag=1 iff the selection register
    equals leaf_idx; it must return the (possibly gate-modified) flag
    Soquet. Leaves outside [l_iter, r_iter) cost dispatch gates (the tree
    is still a full 2**n-leaf tree) but get no payload -- pad l_iter/r_iter
    to a power of two and skip the unused leaves if L isn't already one.

    Returns (bb, sel_qubits) with the selection register's qubits restored
    (same logical content, threaded through as fresh Soquets).
    """
    n = len(sel_qubits)
    L = 2 ** n
    mid = L // 2
    b0 = sel_qubits[0]
    bb, b0 = _go(bb, mid, L, 1, b0, True, sel_qubits, payload_fn, l_iter, r_iter)
    bb, b0 = _go(bb, 0, mid, 1, b0, False, sel_qubits, payload_fn, l_iter, r_iter)
    sel_qubits[0] = b0
    return bb, sel_qubits
