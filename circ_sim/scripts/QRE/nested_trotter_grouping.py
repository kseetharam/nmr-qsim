"""
Nested-Trotter "grouping" mitigation (qre.tex sec:nested_trot_leading,
Mitigation (i)), decided as the baseline mitigation for resource estimates
(pipeline_plan.md open question 7): partition a jump operator L_j's own
Pauli-term decomposition into groups of pairwise-compatible terms --
commuting Pauli strings AND matching Xi_j phase -- so that terms within a
group can be fused into one exact exponential (zero Trotter error between
them), and the nested-Trotter product formula only needs to Trotterize
ACROSS groups, not across every individual Pauli term.

Correction to qre.tex's stated condition: the text says Xi_j(phi_{j,n}) and
Xi_j(phi_{j,n'}) commute "only if phi_{j,n}=phi_{j,n'}", citing
[Xi_j(phi),Xi_j(phi')] = 2i*sin(phi-phi')*(|j><j|-|0><0|). But sin(phi-phi')=0
whenever phi-phi' is any integer multiple of pi, not just 0 -- i.e. the
correct commuting condition is phi_{j,n} == phi_{j,n'} (mod pi), not exact
equality. At phi-phi'=pi, Xi_j(phi') = -Xi_j(phi) exactly (verified
numerically: commutator norm ~1e-16, and Xi_j(phi+pi) == -Xi_j(phi) exactly)
-- a real (if modest) additional grouping opportunity, implemented here.

Applies equally to the nested-Trotter error itself (Eq. nested_trot_final)
and to the "mixed" coherent x same-jump-operator term (Eq. Mpj_pair and its
companion, sec:combined_trot_nested): both are built from the SAME
sequential-Trotter-product structure over a jump operator's own Pauli
terms, so any pair fused into one group contributes to NEITHER the leading
nested term NOR the mixed term's cross-term sums.
"""
import cmath
from typing import List, Tuple

Complex = complex
PauliTerm = Tuple[str, Complex]  # (pauli_string, coefficient)


def pauli_commute(p1: str, p2: str) -> bool:
    """Two Pauli strings commute iff they disagree (both non-identity, and
    different) at an even number of sites."""
    disagree = sum(1 for a, b in zip(p1, p2) if a != 'I' and b != 'I' and a != b)
    return disagree % 2 == 0


def same_phase_mod_pi(c1: Complex, c2: Complex, tol: float = 1e-9) -> bool:
    """True iff arg(c1) == arg(c2) (mod pi) within tol -- the actual
    commuting condition for Xi_j(phi), not just exact phase equality."""
    diff = (cmath.phase(c1) - cmath.phase(c2)) % cmath.pi
    return min(diff, cmath.pi - diff) < tol


def compatible(term1: PauliTerm, term2: PauliTerm, phase_tol: float = 1e-9) -> bool:
    (p1, c1), (p2, c2) = term1, term2
    return pauli_commute(p1, p2) and same_phase_mod_pi(c1, c2, phase_tol)


def group_jump_operator_terms(pauli_terms: List[PauliTerm], phase_tol: float = 1e-9
                               ) -> Tuple[List[List[PauliTerm]], List[int]]:
    """Partition pauli_terms into groups of pairwise-compatible (commuting,
    matching-phase-mod-pi) terms, via greedy coloring of the conflict graph
    (edge = incompatible pair; each color class is then a clique in the
    compatibility graph, i.e. a valid group). Largest-conflict-degree-first
    ordering, matching the style of greedy_edge_coloring elsewhere in this
    project.

    Returns (groups, group_index): groups[g] is the list of (pauli_string,
    coeff) in group g; group_index[n] is which group pauli_terms[n] landed
    in.
    """
    N = len(pauli_terms)
    conflict = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if not compatible(pauli_terms[i], pauli_terms[j], phase_tol):
                conflict[i].add(j)
                conflict[j].add(i)

    order = sorted(range(N), key=lambda i: -len(conflict[i]))
    color_of = {}
    for i in order:
        used = {color_of[j] for j in conflict[i] if j in color_of}
        c = 0
        while c in used:
            c += 1
        color_of[i] = c

    n_groups = (max(color_of.values()) + 1) if color_of else 0
    groups: List[List[PauliTerm]] = [[] for _ in range(n_groups)]
    group_index = [0] * N
    for i in range(N):
        groups[color_of[i]].append(pauli_terms[i])
        group_index[i] = color_of[i]
    return groups, group_index


def cross_group_l1_sum(pauli_terms: List[PauliTerm], group_index: List[int]) -> float:
    """sum_{n<n', different groups} |c_n||c_n'| -- the reduced effective
    cross-term prefactor for the nested-Trotter error (Eq. nested_trot_final)
    and the mixed term (Eq. Mpj_pair's companion), once intra-group
    (exactly-fused, zero-Trotter-error) pairs are excluded from the sum."""
    N = len(pauli_terms)
    total = 0.0
    for n in range(N):
        for np_ in range(n + 1, N):
            if group_index[n] != group_index[np_]:
                total += abs(pauli_terms[n][1]) * abs(pauli_terms[np_][1])
    return total


def full_l1_sum(pauli_terms: List[PauliTerm]) -> float:
    """sum_{n<n'} |c_n||c_n'| with no grouping -- the ungrouped baseline,
    for comparison."""
    N = len(pauli_terms)
    total = 0.0
    for n in range(N):
        for np_ in range(n + 1, N):
            total += abs(pauli_terms[n][1]) * abs(pauli_terms[np_][1])
    return total


if __name__ == '__main__':
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import molecule_operators as mo

    ops = mo.load_molecule_operators('gemcitabine5')
    print(f"{'jump op':>8}  {'N terms':>8}  {'N groups':>9}  "
          f"{'full sum':>12}  {'grouped sum':>12}  {'reduction':>10}")
    total_full, total_grouped = 0.0, 0.0
    for j, pauli_terms in enumerate(ops['jump_pauli_terms']):
        groups, group_index = group_jump_operator_terms(pauli_terms)
        full = full_l1_sum(pauli_terms)
        grouped = cross_group_l1_sum(pauli_terms, group_index)
        total_full += full
        total_grouped += grouped
        reduction = 1 - grouped / full if full > 0 else 0.0
        print(f"{j:>8}  {len(pauli_terms):>8}  {len(groups):>9}  "
              f"{full:>12.4f}  {grouped:>12.4f}  {reduction:>9.1%}")

    print(f"\nTotal (summed over all {len(ops['jump_pauli_terms'])} active jump operators):")
    print(f"  ungrouped sum_{{n<n'}}|c_n||c_n'|  = {total_full:.4f}")
    print(f"  grouped (cross-group only)         = {total_grouped:.4f}")
    print(f"  reduction                          = {1 - total_grouped/total_full:.1%}")
