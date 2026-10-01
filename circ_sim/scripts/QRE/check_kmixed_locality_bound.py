"""
Verifies the locality-weighted bound on K_mixed (circ_sim/scripts/QRE/notes/
qre.tex, Sec. combined_trot_nested, "Locality-weighted bound", Eq.
Kmixed_final) against the exact value on the truncated Gemcitabine system.

The bound is built entirely from ||h_p||_inf (each coherent fragment's own
operator norm -- a small, local quantity) and the jump operators' Pauli
coefficients/supports, via submultiplicativity of Pauli-string operator
norms (||P||_inf=1) -- no dense D_sys x D_sys products beyond ||h_p||_inf
itself, so (unlike the exact K_mixed computation before
pauli_error_coefficients.py existed) this was always polynomial-effort.

    |delta<O(Dt)>_mixed| <= Dt^2 sum_i |w_i| sum_{p,j} ||h_p||_inf *
        [ (2/3) sum_n            |c_{j,n}|^2                (locality-pruned)
        + (10/3) sum_{n<n'}      |c_{j,n}||c_{j,n'}|         (locality-pruned) ]
        + O(Dt^{5/2})

Checks RHS >= |Dt^2 * K_mixed| at a small Dt (K_mixed from the cached exact
value, error_budget_solver.get_coefficients()).
"""
import os
import sys
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pstr_support(pstr):
    return {k for k, ch in enumerate(pstr) if ch != 'I'}


def h_p_support(p_idx, tpg):
    if p_idx == 0:
        return set(range(tpg.n))  # Zeeman layer: touches every site
    color = tpg.ordered_colors[p_idx - 1]
    s = set()
    for (i, j) in tpg.groups[color]:
        s.add(i)
        s.add(j)
    return s


def locality_bound(tpg, Dt):
    """Eq. Kmixed_final's right-hand side, at the given Dt."""
    weights = tpg.weights
    h_p_norms = [np.linalg.norm(hp, ord=2) for hp in tpg.coherent_pieces]
    h_p_supports = [h_p_support(p, tpg) for p in range(len(tpg.coherent_pieces))]

    total = 0.0
    for h_norm, h_supp in zip(h_p_norms, h_p_supports):
        for pauli_terms in tpg.jump_pauli_terms:
            N = len(pauli_terms)
            supports = [pstr_support(pstr) for pstr, _ in pauli_terms]
            cs = [c for _, c in pauli_terms]
            for i in range(tpg.n):
                pair_sum = sum(abs(cs[nn]) ** 2 for nn in range(N)
                                if i in h_supp or i in supports[nn])
                triple_sum = sum(
                    abs(cs[a]) * abs(cs[b])
                    for a in range(N) for b in range(a + 1, N)
                    if i in h_supp or i in supports[a] or i in supports[b])
                total += weights[i] * h_norm * (2 / 3 * pair_sum + 10 / 3 * triple_sum)
    return (Dt ** 2) * total


if __name__ == '__main__':
    tpg = _load_module('tpg', 'trotter_prf_vs_trot_gemcitabine5.py')
    tpg.n = 5  # not otherwise exposed as a module attribute

    ebs = _load_module('ebs', 'error_budget_solver.py')
    _, K_mixed = ebs.get_coefficients()

    Dt_test = tpg.delta_t / 2000
    rhs = locality_bound(tpg, Dt_test)
    lhs = abs(Dt_test ** 2 * K_mixed)

    print(f'Dt = {Dt_test:.3e} s (delta_t/2000)')
    print(f'LHS |Dt^2 K_mixed|          = {lhs:.6e}')
    print(f'RHS (locality-weighted bound) = {rhs:.6e}')
    print(f'bound holds (RHS >= LHS)?   {rhs >= lhs}')
    print(f'looseness ratio RHS/LHS     = {rhs / lhs:.2f}')
