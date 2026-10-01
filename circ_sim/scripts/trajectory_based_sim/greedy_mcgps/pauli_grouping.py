"""
Greedy partitioning of a list of Pauli products into groups of mutually
commuting terms ("mcgps" = Mutually Commuting Groups of Pauli Strings).

Replicates the greedy-grouping algorithm in
Q4Bio/PhaseNott/heuristic_noise_phasecraft/utils/meas_utils.py's
get_greedy_grouping, ported here from OpenFermion QubitOperators to this
project's own (pauli_string, coefficient) tuple convention, with NO
OpenFermion/Cirq dependency -- a collaborator's workflow expects to pipe
inputs/outputs in that plain tuple form.

Why group by commutation, mathematically: for ANY set of mutually commuting
Hermitian Pauli strings {P_1,...,P_K} -- regardless of qubit-support
overlap, and regardless of whether their scalar coefficients are real or
complex -- the joint exponential of their weighted sum factors EXACTLY into
a product of individual exponentials, in any order:

    exp(i * sum_k c_k P_k) = exp(i*c_1*P_1) * exp(i*c_2*P_2) * ... * exp(i*c_K*P_K).

So partitioning a Hamiltonian/generator's Pauli-term decomposition into
mutually-commuting groups identifies exactly which terms can be fused with
ZERO Trotter error; only terms split across DIFFERENT groups contribute
error when sequenced relative to each other. That error (the usual
Trotter-splitting bound) scales with the coefficient-weighted overlap left
between terms in different groups -- concretely, with
sum_{cross-group pairs} |c_n||c_n'|. That is why this grouping scheme
deliberately favors co-locating large-coefficient terms, rather than simply
minimizing the number of groups (an ordinary graph-coloring objective, which
is coefficient-blind and would not target this at all).

Algorithm ("largest terms first", exactly mirroring get_greedy_grouping):
    1. Sort all terms by |coefficient|, descending.
    2. Repeat until no terms remain:
       a. Start a new, empty group.
       b. Scan the remaining terms IN THAT SORTED ORDER. Add each one to the
          current group if it commutes with EVERY term already placed in
          the group; otherwise leave it for a later pass.
       c. Remove every term absorbed in this pass; the current group is
          complete.
    The single largest-coefficient remaining term always starts each new
    group (it trivially "commutes with everything" in an empty group), and
    every other term then gets priority for joining THAT group in
    decreasing coefficient order. Unlike plain graph coloring, there is no
    attempt to minimize the total number of groups produced -- that is an
    intentional trade-off, not an oversight.

Two notions of commutation are supported, matching meas_utils.py exactly:
  - 'fc' (full commutation, the default, and the one that makes exact
    Trotter-exponential fusion valid as derived above): two Pauli strings
    commute iff they disagree, at positions where both are non-identity, an
    even number of times.
  - 'qwc' (qubit-wise commutation, a STRICTER condition some measurement
    schemes require): two Pauli strings commute iff, at every single qubit,
    they agree or at least one is identity. qwc implies fc but not the
    reverse -- e.g. 'XX' and 'YY' commute under fc but not qwc.

Grouping depends on coefficient MAGNITUDE only insofar as it sets processing
order; coefficient VALUES (phase, sign) are never inspected and are carried
through the input/output completely unchanged.

Input/output convention matches the rest of this project: a Pauli term is
(pauli_string, coefficient), where pauli_string is a string of length n over
{'I','X','Y','Z'} (one character per qubit) and coefficient is any scalar
(real or complex).
"""
from typing import List, Sequence, Tuple

PauliTerm = Tuple[str, complex]


def _check_same_length_and_alphabet(strings):
    length = len(strings[0])
    for s in strings:
        if len(s) != length:
            raise ValueError('All Pauli strings must have the same length')
        bad = set(s) - set('IXYZ')
        if bad:
            raise ValueError(f"Pauli string {s!r} contains character(s) {sorted(bad)} "
                              f"outside 'IXYZ'")


def pauli_commute(p1: str, p2: str) -> bool:
    """True iff the two equal-length Pauli strings commute under FULL
    (general operator) commutation.

    Two Pauli strings anticommute iff they disagree (both non-identity, and
    different from each other) at an ODD number of positions; they commute
    iff that count is even -- including zero.
    """
    if len(p1) != len(p2):
        raise ValueError(f'Pauli strings must have the same length: {len(p1)} vs {len(p2)}')
    disagree = sum(1 for a, b in zip(p1, p2) if a != 'I' and b != 'I' and a != b)
    return disagree % 2 == 0


def qubit_wise_commute(p1: str, p2: str) -> bool:
    """True iff the two equal-length Pauli strings commute QUBIT-WISE: at
    every position, either at least one is 'I', or they're equal. Stricter
    than pauli_commute (qwc implies fc, not conversely) -- e.g. 'XX' and
    'YY' commute under pauli_commute but not under this function."""
    if len(p1) != len(p2):
        raise ValueError(f'Pauli strings must have the same length: {len(p1)} vs {len(p2)}')
    return all(a == 'I' or b == 'I' or a == b for a, b in zip(p1, p2))


def _commutes(p1: str, p2: str, commutativity: str) -> bool:
    if commutativity == 'fc':
        return pauli_commute(p1, p2)
    if commutativity == 'qwc':
        return qubit_wise_commute(p1, p2)
    raise ValueError(f"commutativity must be 'fc' or 'qwc', got {commutativity!r}")


def greedy_group_commuting(terms: Sequence[PauliTerm], commutativity: str = 'fc'
                            ) -> Tuple[List[List[PauliTerm]], List[int]]:
    """Partition `terms` into groups of pairwise mutually-commuting Pauli
    products, via the "largest coefficient first" greedy algorithm described
    in this module's docstring (replicating
    heuristic_noise_phasecraft/utils/meas_utils.py's get_greedy_grouping).

    Parameters
    ----------
    terms : sequence of (pauli_string, coefficient)
        pauli_string: str of length n over {'I','X','Y','Z'}, same n for
        every term. coefficient: any scalar (real or complex) -- its
        MAGNITUDE sets processing order; its value is otherwise unused and
        carried through unchanged.
    commutativity : 'fc' (default) or 'qwc'
        'fc' = full/general Pauli commutation (the condition that makes
        exact Trotter-exponential fusion valid -- see module docstring).
        'qwc' = the stricter qubit-wise commutation some measurement
        schemes require.

    Returns
    -------
    groups : list of list of PauliTerm
        groups[g] is the list of terms assigned to group g, in the order
        they were absorbed (largest-coefficient-first within the group).
        Every pair of terms within a group commutes (per `commutativity`).
    group_index : list of int
        group_index[i] is the group terms[i] was assigned to, where i
        indexes the ORIGINAL input order (not the internal sorted-by-size
        order) -- i.e. terms[i] is in groups[group_index[i]]. Parallel to
        `terms`.
    """
    n = len(terms)
    if n == 0:
        return [], []
    _check_same_length_and_alphabet([t[0] for t in terms])

    # Sort by |coefficient| descending, tracking original input indices so
    # group_index can be reported against the input order.
    order = sorted(range(n), key=lambda i: abs(terms[i][1]), reverse=True)
    remaining = list(order)

    groups: List[List[PauliTerm]] = []
    group_of_index = [-1] * n
    g = 0
    while remaining:
        current: List[int] = []
        leftover: List[int] = []
        for idx in remaining:
            if all(_commutes(terms[idx][0], terms[m][0], commutativity) for m in current):
                current.append(idx)
            else:
                leftover.append(idx)
        groups.append([terms[idx] for idx in current])
        for idx in current:
            group_of_index[idx] = g
        g += 1
        remaining = leftover

    return groups, group_of_index


def verify_grouping(groups: Sequence[Sequence[PauliTerm]], commutativity: str = 'fc') -> None:
    """Raise AssertionError if any two terms within the same group fail to
    commute (under `commutativity`). Always true for a grouping produced by
    greedy_group_commuting itself; useful to re-check after merging/editing
    groups by hand, or when auditing a grouping that came from elsewhere."""
    for g_idx, group in enumerate(groups):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                p1, p2 = group[a][0], group[b][0]
                if not _commutes(p1, p2, commutativity):
                    raise AssertionError(
                        f'Group {g_idx} contains a non-{commutativity}-commuting pair: '
                        f'{p1!r}, {p2!r}')
