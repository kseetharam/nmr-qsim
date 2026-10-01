"""
Hermitian-generator + cross-generator pooling/grouping module for the
trajectory-based QRE pipeline (README.md's "Pipeline architecture" item 1).

Promotes the ad hoc construction validated in
../../trajectory_based_sim/testing/plot4_group_trotter_convergence.py into a
reusable module every other trajectory-based gadget (the anisotropic-step
gadget, the TrotterScheme implementations) builds on -- mirroring
../molecule_operators.py's and ../nested_trotter_grouping.py's role for the
ancilla pipeline.

What this module produces, given a molecule's canonical (generally
non-Hermitian) jump operators L_j (../molecule_operators.py's L_dicts):

  1. The 25 independent, real Hermitian noise generators V_j
     (build_hermitian_generators_pauli), as Pauli-term dicts -- the Pauli-
     dict analogue of ../../trajectory_based_sim/testing/trajectory_convergence.py's
     dense-matrix build_hermitian_generators, needed here because grouping
     (step 3) operates on Pauli strings, which dense matrices don't expose.
  2. The pooled set of unique Pauli strings across ALL 25 generators
     (pool_unique_terms), plus the coefficient matrix C[p, j] letting a
     trajectory's 25 drawn scalars dW_j become one aggregate coefficient per
     unique string via a single matrix-vector product.
  3. A mutually-commuting partition of that pooled term set
     (group_unique_terms), reusing ../nested_trotter_grouping.py's existing
     greedy conflict-graph coloring UNCHANGED.

Why reusing nested_trotter_grouping.py unchanged is correct here, not just
convenient: its compatibility criterion is "commuting AND matching phase mod
pi", needed in the ancilla pipeline because non-Hermitian jump operators are
compiled via a complex-coefficient X/Y ancilla-rotation trick whose two
pieces only fuse exactly when phases align mod pi. Every V_j here is
Hermitian, decomposed in the real {I,X,Y,Z}^n basis, so every coefficient is
real -- the phase-mod-pi condition is then automatically satisfied for any
pair (arg(real number) in {0, pi}, always mod-pi-equal), so the grouping
degenerates to pure Pauli commutation, which is coefficient-independent and
therefore identical for every trajectory (see README.md's "What's actually
new here" point 4 for the full argument). This is enforced here, not just
asserted, by grouping with identical dummy placeholder coefficients (see
group_unique_terms) -- if the phase check were ever doing real work for this
real-coefficient case, using a placeholder would silently change the
grouping, which validate_against_dense's caller-facing check would not by
itself catch, but the placeholder choice makes that failure mode structural
rather than possible.

Validated (validate_against_dense, also run in __main__ for gemcitabine5):
reconstructing the 25 generators from Pauli dicts and converting back to
dense matrices reproduces trajectory_convergence.py's already-trusted dense
construction to machine precision.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.dirname(HERE)
TRAJ_TESTING_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'trajectory_based_sim', 'testing'))
for _p in (HERE, QRE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nested_trotter_grouping as ntg  # noqa: E402

_PAULI1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_mat(s):
    M = _PAULI1[s[0]]
    for c in s[1:]:
        M = np.kron(M, _PAULI1[c])
    return M


def _dict_to_dense(d, n):
    D = 2 ** n
    M = np.zeros((D, D), dtype=complex)
    for p, c in d.items():
        M += c * _pauli_mat(p)
    return M


# ---------------------------------------------------------------------------
# 1. Hermitian generators, as Pauli dicts.
# ---------------------------------------------------------------------------

def build_hermitian_generators_pauli(jump_ops_dense, L_dicts, tol=1e-9):
    """Pauli-dict form of trajectory_convergence.build_hermitian_generators:
    same conjugate-pairing detection (done on the dense jump operators, a
    Frobenius-norm comparison Pauli dicts don't simplify), but each
    generator's own Pauli decomposition is read off directly from L_dicts
    via per-term Re/Im extraction -- for any operator L, (L+L^dagger)/2 and
    (L-L^dagger)/(2i) are, term by term, exactly Re(c_P) and Im(c_P) of L's
    own Pauli dict, since every Pauli string is its own Hermitian conjugate.
    No dense matrices needed for this step; dense matrices are only used
    above to detect which jump operators are self-conjugate or paired.

    .real/.imag casts below are not no-ops: hermitian_idx is only
    "Hermitian up to tol", so its coefficients are Python complex with an
    exactly-zero (but still complex-typed) imaginary part -- assigning those
    into a float array elsewhere (pool_unique_terms's coef_matrix) raises
    TypeError on current numpy without this explicit cast.
    """
    n = len(jump_ops_dense)
    hermitian_idx = [j for j in range(n)
                      if np.linalg.norm(jump_ops_dense[j] - jump_ops_dense[j].conj().T, 'fro')
                      < tol * np.linalg.norm(jump_ops_dense[j], 'fro')]
    remaining = [j for j in range(n) if j not in hermitian_idx]
    consumed, pairs = set(), []
    for j in remaining:
        if j in consumed:
            continue
        for k in remaining:
            if k in consumed or k == j:
                continue
            for s in (1.0, -1.0):
                if (np.linalg.norm(jump_ops_dense[j].conj().T - s * jump_ops_dense[k], 'fro')
                        < 1e-6 * np.linalg.norm(jump_ops_dense[j], 'fro')):
                    pairs.append((j, k))
                    consumed.add(j)
                    consumed.add(k)
                    break
            if j in consumed:
                break
    if len(hermitian_idx) + 2 * len(pairs) != n:
        raise ValueError(f'Failed to account for all {n} jump operators: '
                          f'{len(hermitian_idx)} self-conjugate + {len(pairs)} pairs')

    V_dicts = [{P: c.real for P, c in L_dicts[j].items()} for j in hermitian_idx]
    for j, _k in pairs:
        V_dicts.append({P: np.sqrt(2) * c.real for P, c in L_dicts[j].items()})
        V_dicts.append({P: np.sqrt(2) * c.imag for P, c in L_dicts[j].items()})
    return V_dicts


# ---------------------------------------------------------------------------
# 2. Cross-generator pooling.
# ---------------------------------------------------------------------------

def pool_unique_terms(V_dicts):
    """Unique Pauli strings across ALL V_dicts, and the coefficient matrix
    C[p, j] = coefficient of unique string p in generator j (0 if absent).
    Returns (unique_strings, str_index, coef_matrix)."""
    unique_strings = sorted({p for d in V_dicts for p in d})
    str_index = {s: i for i, s in enumerate(unique_strings)}
    n_terms, n_gen = len(unique_strings), len(V_dicts)
    coef_matrix = np.zeros((n_terms, n_gen))
    for j, d in enumerate(V_dicts):
        for p, c in d.items():
            coef_matrix[str_index[p], j] = c
    return unique_strings, str_index, coef_matrix


# ---------------------------------------------------------------------------
# 3. Mutually-commuting grouping of the pooled term set.
# ---------------------------------------------------------------------------

def group_unique_terms(unique_strings):
    """Partition unique_strings into mutually-commuting groups via
    ../nested_trotter_grouping.py's existing greedy conflict-graph coloring,
    reused unmodified. Coefficients passed to it are IDENTICAL placeholders
    (1.0+0j for every term) on purpose: commutation is coefficient-
    independent, and using identical placeholders makes the phase-mod-pi
    check trivially satisfied for every pair by construction, so this
    reduces to pure-commutation grouping -- the physically correct
    criterion for this real-coefficient setting (see module docstring).
    Returns (groups, group_term_indices): groups[g] is the list of Pauli
    strings in group g; group_term_indices[g] is the same, as indices into
    unique_strings."""
    placeholder_terms = [(p, 1.0 + 0j) for p in unique_strings]
    groups_terms, _ = ntg.group_jump_operator_terms(placeholder_terms)
    groups = [[p for p, _ in g] for g in groups_terms]
    str_index = {s: i for i, s in enumerate(unique_strings)}
    group_term_indices = [[str_index[p] for p in g] for g in groups]
    return groups, group_term_indices


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------

def build_anisotropic_data(mol_ops):
    """Given ../molecule_operators.py's load_molecule_operators(...) output,
    returns the full data bundle every trajectory-based circuit gadget needs
    for the dissipative/anisotropic part:
        V_dicts             -- Hermitian generators, real-coefficient Pauli dicts
        unique_strings      -- pooled unique Pauli strings across all V_j
        str_index           -- {pauli_string: index into unique_strings}
        coef_matrix         -- (n_terms, n_gen) real array, coef_matrix[p, j]
        groups              -- mutually-commuting partition of unique_strings
        group_term_indices  -- same partition, as indices into unique_strings
        n                   -- number of system qubits (for convenience)
    """
    n = mol_ops['n']
    jump_ops_dense = [_dict_to_dense(d, n) for d in mol_ops['L_dicts']]
    V_dicts = build_hermitian_generators_pauli(jump_ops_dense, mol_ops['L_dicts'])
    unique_strings, str_index, coef_matrix = pool_unique_terms(V_dicts)
    groups, group_term_indices = group_unique_terms(unique_strings)
    return dict(V_dicts=V_dicts, unique_strings=unique_strings, str_index=str_index,
                coef_matrix=coef_matrix, groups=groups,
                group_term_indices=group_term_indices, n=n)


# ---------------------------------------------------------------------------
# Validation against the already-trusted dense construction.
# ---------------------------------------------------------------------------

def validate_against_dense(V_dicts, mol_ops, tol=1e-10):
    """Checks V_dicts (Pauli-dict form) against
    trajectory_convergence.build_hermitian_generators's trusted dense
    construction, built independently here from mol_ops['L_dicts'] (not
    imported from a cached value) so this is a real cross-check, not a
    tautology. Raises AssertionError if they disagree beyond tol. Returns
    the max elementwise Frobenius-norm difference (for logging). Imports
    trajectory_based_sim/testing lazily, only here, so normal use of this
    module (build_anisotropic_data) has no dependency on that tree."""
    if TRAJ_TESTING_DIR not in sys.path:
        sys.path.insert(0, TRAJ_TESTING_DIR)
    from trajectory_convergence import build_hermitian_generators  # noqa: E402

    n = mol_ops['n']
    jump_ops_dense = [_dict_to_dense(d, n) for d in mol_ops['L_dicts']]
    V_dense_trusted = build_hermitian_generators(jump_ops_dense)
    V_dense_new = [_dict_to_dense(d, n) for d in V_dicts]
    if len(V_dense_new) != len(V_dense_trusted):
        raise AssertionError(f'generator count mismatch: {len(V_dense_new)} vs '
                              f'{len(V_dense_trusted)}')
    max_diff = max(np.linalg.norm(a - b) for a, b in zip(V_dense_new, V_dense_trusted))
    assert max_diff < tol, f'Pauli-dict generators disagree with trusted dense construction: {max_diff:.3e}'
    return max_diff


if __name__ == '__main__':
    import molecule_operators as mo

    mol_ops = mo.load_molecule_operators('gemcitabine5')
    data = build_anisotropic_data(mol_ops)

    max_diff = validate_against_dense(data['V_dicts'], mol_ops)
    print(f'Validated against dense construction: max diff = {max_diff:.3e}')

    print(f"n_gen (V_j) = {len(data['V_dicts'])}")
    print(f"n_unique_terms (pooled) = {len(data['unique_strings'])}")
    print(f"n_groups = {len(data['groups'])}")
    print(f"group sizes = {sorted((len(g) for g in data['groups']), reverse=True)}")
    print(f"coef_matrix shape = {data['coef_matrix'].shape}")
