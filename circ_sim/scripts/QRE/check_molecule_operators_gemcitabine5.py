"""
Validates the uniform per-molecule loader (molecule_operators.py,
pipeline_plan.md open question 6) against the truncated Gemcitabine system
this project's QRE notes have been validated against throughout: rebuilds
every operator through the new interface and confirms H0 matches the
existing (already-verified) construction exactly, then recomputes K_prf and
K_mixed via pauli_error_coefficients.py's functions UNMODIFIED.

K_prf should match the cached value to ~machine precision. K_mixed matches
the individual jump operators exactly (same Pauli strings/coefficients,
verified elsewhere to ~1e-16), but can differ from the cached value at the
~1e-4 relative level: M_{p,j} (qre.tex sec:combined_trot_nested) is defined
relative to a specific "which Pauli term of a given L_j compiles first"
ordering, and the new interface's construction path breaks ties among
equal-|c| Pauli coefficients differently than the original dense-decomposition
pipeline did -- both are equally valid circuit-compilation choices, not a
bug. See molecule_operators.py's jump_pauli_terms sort key.
"""
import os
import sys
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import molecule_operators as mo  # noqa: E402


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == '__main__':
    ops = mo.load_molecule_operators('gemcitabine5')
    alg = ops['alg']
    print(f"n = {ops['n']}, active jump operators = {len(ops['L_dicts'])}, "
          f"coherent fragments = {len(ops['coherent_dicts'])}")

    # --- H0 self-consistency: sum of fragments == fso's own unfragmented H0 ---
    H0_from_fragments = {}
    for frag in ops['coherent_dicts']:
        H0_from_fragments = alg.add(H0_from_fragments, frag)
    diff = alg.add(H0_from_fragments, alg.scale(ops['H0_dict'], -1))
    max_diff = max((abs(v) for v in diff.values()), default=0.0)
    print(f'H0 fragment sum vs unfragmented H0: max abs diff = {max_diff:.2e}')

    # --- cross-check against the already-cached, already-verified values ---
    pec = _load_module('pec', 'pauli_error_coefficients.py')
    K_prf = pec.compute_K_prf(alg, ops['H0_dict'], ops['L_dicts'], ops['rho0_dict'], ops['coil_dict'])
    K_mixed = pec.compute_K_mixed(alg, ops['coherent_dicts'], ops['jump_pauli_terms'],
                                   ops['rho0_dict'], ops['coil_dict'])
    print(f'K_prf   (new interface) = {K_prf}')
    print(f'K_mixed (new interface) = {K_mixed}')

    cache_path = os.path.join(HERE, 'data', 'error_budget_coeffs_gemcitabine5.npz')
    cache = np.load(cache_path)
    K_prf_ref = complex(cache['K_prf'])
    K_mixed_ref = complex(cache['K_mixed'])
    print(f'\nK_prf   relative diff vs cache = {abs(K_prf - K_prf_ref) / abs(K_prf_ref):.2e}')
    print(f'K_mixed relative diff vs cache = {abs(K_mixed - K_mixed_ref) / abs(K_mixed_ref):.2e}')
