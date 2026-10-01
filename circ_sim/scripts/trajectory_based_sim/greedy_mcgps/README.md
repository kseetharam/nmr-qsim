# greedy_mcgps: Greedy Mutually-Commuting Groups of Pauli Strings

A small, standalone module that partitions a list of Pauli products
(strings over `{I,X,Y,Z}`, each with a scalar coefficient) into groups of
mutually commuting terms, using a greedy algorithm that deliberately favors
grouping the largest-coefficient terms together. No dependency on anything else in this
repository, and only `numpy`/`scipy` for the (optional) example/correctness
check -- `pauli_grouping.py` itself has no dependencies beyond the Python
standard library.


## Why group by commutation at all

For any set of mutually commuting Hermitian Pauli strings
$\{P_1,\dots,P_K\}$ — regardless of whether they act on disjoint or
overlapping qubits, and regardless of whether their coefficients are real or
complex — the joint exponential of their weighted sum factors **exactly**
into a product of individual exponentials, in any order:

$$\exp\!\Big(i\sum_k c_k P_k\Big) = \exp(ic_1P_1)\,\exp(ic_2P_2)\cdots\exp(ic_KP_K).$$

This is an exact operator-algebra identity, not an approximation. So
partitioning a Hamiltonian/generator's Pauli-term decomposition into
mutually-commuting groups identifies exactly which terms can be fused with
**zero** Trotter error. Only terms that end up in *different* groups incur
error when sequenced relative to each other — and that error (the usual
Trotter-splitting bound) scales with the coefficient-weighted overlap left
between groups, roughly $\sum_{\text{cross-group pairs}}|c_n||c_{n'}|$.

## Why this specific greedy algorithm, not plain graph coloring

An ordinary greedy graph-coloring heuristic (e.g. largest-conflict-degree
first, assign the lowest available color) produces *a* valid partition into
mutually-commuting groups, but it is blind to coefficient magnitude — which
specific terms land together is just an accident of processing order. That
matters here because the goal isn't "minimize the number of groups"; it's
"minimize the coefficient-weighted overlap left *between* groups" — i.e.
deliberately co-locate the largest-magnitude commuting terms, so that only
small-coefficient terms are left to spill across group boundaries (where
they contribute the least to the Trotter-error bound above).

## Algorithm

1. Sort all terms by $|\text{coefficient}|$, descending.
2. Repeat until no terms remain:
   - Start a new, empty group.
   - Scan the remaining terms **in that fixed sorted order**. Add a term to
     the current group if it commutes with *every* term already placed in
     the group; otherwise leave it for a later pass.
   - Remove every term absorbed in this pass. The current group is
     complete; start the next one.

The single largest-coefficient remaining term always seeds each new group
(it trivially "commutes with everything" in an empty group), and every
other term then gets priority — in decreasing coefficient order — for
joining that group. Unlike graph coloring, there is **no** attempt to
minimize the total number of groups produced; that's an intentional
trade-off for this use case, not an oversight.

## Two notions of commutation

Both are supported (`commutativity='fc'` or `'qwc'`), matching
`meas_utils.py` exactly:

- **`'fc'` (full commutation, the default).** Two Pauli strings commute iff
  they disagree, at positions where both are non-identity, an **even**
  number of times. This is the condition that makes the exact-fusion
  identity above valid, and is the one relevant for Trotter-circuit
  compilation.
- **`'qwc'` (qubit-wise commutation).** Stricter: two Pauli strings commute
  iff, at *every single qubit*, they agree or at least one is identity.
  `qwc` implies `fc` but not the reverse — e.g. `'XX'` and `'YY'` commute
  under `fc` but not `qwc`. Some measurement schemes specifically want
  `qwc` groups (diagonalizable by single-qubit rotations alone, no
  entangling gates); pass `commutativity='qwc'` if that's your use case.

## Input / output format

A Pauli term is `(pauli_string, coefficient)`:
- `pauli_string`: a `str` of length $n$ over `{'I','X','Y','Z'}`, one
  character per qubit. All terms in one call must have the same length.
- `coefficient`: any scalar, real or complex. Its magnitude sets processing
  order; its value (sign, phase) is never inspected otherwise and is
  carried through to the output completely unchanged.

```python
from pauli_grouping import greedy_group_commuting, verify_grouping

terms = [
    ('XXII', 10.0),
    ('YYII', -9.5),
    ('ZZII', 8.7),
    ('XIII', 0.3),
]

groups, group_index = greedy_group_commuting(terms)   # commutativity='fc' by default
verify_grouping(groups)   # raises AssertionError if anything is wrong

# groups: list of list of (pauli_string, coefficient) -- every pair within
#         a group commutes.
# group_index: list of int, parallel to `terms` (original input order):
#         group_index[i] gives which group terms[i] landed in.
```

## Running the example

```
python example.py
```

Builds a small 4-qubit test case, prints the resulting grouping (showing the
large-coefficient terms landing together), then directly verifies the
mathematical property the whole scheme relies on: for each group, the joint
matrix exponential of its combined generator matches the sequential product
of its individual term exponentials **in a randomly shuffled order**, to
numerical precision — confirming both exactness and order-independence
within a group.

