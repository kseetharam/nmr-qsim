"""
Minimal usage example for pauli_grouping.py:
  1. basic usage, chosen to make visible that LARGE-coefficient terms are
     prioritized into a shared group (the actual point of this algorithm,
     not simply producing *a* valid partition);
  2. a direct numerical check of the property the grouping exists to
     guarantee: terms within one returned group can be exponentiated in any
     order with zero error, because exp(i*sum c_k P_k) = prod exp(i*c_k*P_k)
     exactly whenever the P_k commute.
"""
import numpy as np
from scipy.linalg import expm

from pauli_grouping import greedy_group_commuting, verify_grouping

_PAULI1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(pauli_string):
    M = _PAULI1[pauli_string[0]]
    for ch in pauli_string[1:]:
        M = np.kron(M, _PAULI1[ch])
    return M


if __name__ == '__main__':
    # 4 qubits. XXII, YYII, ZZII are the standard mutually-commuting triple
    # (same qubit-pair support) and are given the largest coefficients.
    # XIII commutes with XXII but NOT with YYII or ZZII, so it cannot join
    # that group no matter how the scan proceeds. IIXY has disjoint support
    # from all four others, so it commutes with everything and is a "free
    # rider" that joins the big group too, despite its tiny coefficient --
    # there's no reason to exclude it.
    terms = [
        ('XXII', 10.0 + 0j),
        ('YYII', -9.5 + 0j),
        ('ZZII', 8.7 + 0j),
        ('XIII', 0.3 + 0j),
        ('IIXY', 0.1 + 0.05j),
    ]

    groups, group_index = greedy_group_commuting(terms, commutativity='fc')
    verify_grouping(groups, commutativity='fc')

    print(f'{len(terms)} terms -> {len(groups)} mutually-commuting groups (fc)')
    for g, group in enumerate(groups):
        labels = [f'{p}({c:.2g})' for p, c in group]
        print(f'  group {g}: {labels}')
    print(f'  group_index (parallel to input order): {group_index}')
    print('  -> the three largest-coefficient terms land together in group 0, along with '
          'the tiny IIXY (which happens to commute with all of them); only XIII, which '
          'conflicts with YYII and ZZII, is pushed into its own group.')

    # Correctness check: for each group, exp(i*sum_k c_k*P_k) computed as
    # ONE joint matrix exponential must equal the SEQUENTIAL PRODUCT of the
    # group's own individual term exponentials, in ANY order -- the exact
    # property this partition exists to guarantee.
    n = len(terms[0][0])
    D = 2 ** n
    rng = np.random.default_rng(0)
    print()
    for g, group in enumerate(groups):
        mats = [pauli_matrix(p) for p, _ in group]
        coeffs = [c for _, c in group]

        joint_generator = sum(c * P for c, P in zip(coeffs, mats))
        U_joint = expm(1j * joint_generator)

        order = list(range(len(group)))
        rng.shuffle(order)  # order should not matter for commuting terms
        U_sequential = np.eye(D, dtype=complex)
        for idx in order:
            U_sequential = expm(1j * coeffs[idx] * mats[idx]) @ U_sequential

        diff = np.max(np.abs(U_joint - U_sequential))
        print(f'  group {g}: max|U_joint - U_sequential (shuffled order)| = {diff:.3e}')
        assert diff < 1e-10, f'group {g} terms do not actually commute as claimed'

    print('\nAll groups verified: joint exponential matches the sequential product in a'
          ' shuffled order, confirming zero-error fusion within each group.')
