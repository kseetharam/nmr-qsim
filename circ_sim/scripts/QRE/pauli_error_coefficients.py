"""
Polynomial-effort (in the number of spins) computation of K_prf and K_mixed
via Pauli-string algebra (circ_sim/scripts/linblad_dyn/utils/pauli_algebra.py)
instead of dense 2^n x 2^n matrices.

Motivation (pipeline_plan.md open question 4 / qre.tex sec:dt_solver): the
existing computation of K_mixed (check_combined_trotter_scaling.py) and
K_prf (error_budget_solver.py) builds dense D_sys x D_sys matrices and
multiplies them -- O(D_sys^3) = O(8^n) per multiplication. H_0, every L_j,
rho0, and hat{O} are all sums of a polynomial number of bounded-weight Pauli
strings, and M_{p,j}'s formula (Eq. Mpj_pair + the three-distinct-fragment
piece, qre.tex sec:combined_trot_nested) only ever multiplies together a
handful of such operators per term -- so the whole computation is
polynomial in n if done via Pauli-string algebra (each product/trace is
O(1)-ish work, not O(8^n)). Verified on the truncated Gemcitabine system:
both K_prf and K_mixed match the dense-matrix reference to ~1e-11/1e-14
relative precision, in a fraction of the dense computation's wall-clock
time even though n=5 is far too small for the exponential-vs-polynomial gap
to matter much yet.

The core functions (compute_K_prf, compute_K_mixed, Mpj_pauli, ...) take
Pauli dicts as input and are molecule-agnostic; gemcitabine5_pauli_operators()
is the concrete loader used for validation here, until a general
per-molecule Pauli-dict loader exists (pipeline_plan.md open question 6).

Pauli-dict convention (matches pauli_algebra.py and every dense array in
trotter_prf_vs_trot_gemcitabine5.py): n-character strings over {I,X,Y,Z},
spin-1/2 single-spin operators Sx=X/2 etc., S+ = Sx + i*Sy.
"""
import os
import sys
import time
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(HERE, '..', 'linblad_dyn', 'utils'))
sys.path.insert(0, _UTILS)
from pauli_algebra import PauliAlgebra  # noqa: E402


# ---------------------------------------------------------------------------
# Core, molecule-agnostic computations (Pauli dicts in, Pauli dicts/scalars out)
# ---------------------------------------------------------------------------

def build_Gamma(alg, L_dicts):
    Gamma = {}
    for Lj in L_dicts:
        Gamma = alg.add(Gamma, alg.product(alg.dagger(Lj), Lj))
    return Gamma


def build_k2_k4_kappas(alg, H0_dict, L_dicts, Gamma_dict):
    """k2, k4, {kappa_j} -- Eqs. R_rho/Cprf (Kraus expansion, qre.tex sec:prf_leading)."""
    k2 = alg.add(alg.scale(H0_dict, -1j), alg.scale(Gamma_dict, -0.5))

    sum_LdHL = {}
    for Lj in L_dicts:
        sum_LdHL = alg.add(sum_LdHL, alg.product(alg.product(alg.dagger(Lj), H0_dict), Lj))
    k4 = alg.add(
        alg.scale(alg.product(H0_dict, H0_dict), -0.5),
        alg.scale(alg.add(alg.product(H0_dict, Gamma_dict), sum_LdHL,
                           alg.product(Gamma_dict, H0_dict)), 1j / 6),
        alg.scale(alg.product(Gamma_dict, Gamma_dict), 1 / 24),
    )

    kappas = []
    for Lj in L_dicts:
        kap = alg.add(
            alg.scale(alg.add(alg.product(H0_dict, Lj), alg.product(Lj, H0_dict)), -0.5),
            alg.scale(alg.product(Lj, Gamma_dict), 1j / 6))
        kappas.append(kap)
    return k2, k4, kappas


def lindbladian(alg, H0_dict, L_dicts, rho):
    out = alg.add(alg.scale(alg.product(H0_dict, rho), -1j),
                   alg.scale(alg.product(rho, H0_dict), 1j))
    for Lj in L_dicts:
        LdL = alg.product(alg.dagger(Lj), Lj)
        out = alg.add(out,
                       alg.product(alg.product(Lj, rho), alg.dagger(Lj)),
                       alg.scale(alg.product(LdL, rho), -0.5),
                       alg.scale(alg.product(rho, LdL), -0.5))
    return out


def compute_K_prf(alg, H0_dict, L_dicts, rho0_dict, O_dict):
    """K_prf = Tr{O C_prf[rho0]}, Eq. prf_leading_obs."""
    Gamma = build_Gamma(alg, L_dicts)
    k2, k4, kappas = build_k2_k4_kappas(alg, H0_dict, L_dicts, Gamma)

    def R(rho):
        out = alg.add(alg.product(k4, rho), alg.product(rho, alg.dagger(k4)),
                       alg.product(alg.product(k2, rho), alg.dagger(k2)))
        for Lj, kap in zip(L_dicts, kappas):
            out = alg.add(out,
                           alg.scale(alg.product(alg.product(Lj, rho), alg.dagger(kap)), -1j),
                           alg.scale(alg.product(alg.product(kap, rho), alg.dagger(Lj)), 1j))
        return out

    L2rho = lindbladian(alg, H0_dict, L_dicts, lindbladian(alg, H0_dict, L_dicts, rho0_dict))
    C_prf = alg.add(R(rho0_dict), alg.scale(L2rho, -0.5))
    return (2 ** alg.n) * alg.trace_product(O_dict, C_prf)


def Mpj_pauli(alg, h_p_dict, pauli_terms):
    """M_{p,j}, Eq. Mpj_pair + three-distinct-fragment piece (sec:combined_trot_nested).
    pauli_terms: list of (pauli_string, coefficient) for one jump operator's
    own Pauli decomposition."""
    N = len(pauli_terms)
    M = {}
    for pstr, c in pauli_terms:
        Pn = {pstr: 1.0 + 0j}
        term = alg.add(h_p_dict, alg.scale(alg.product(alg.product(Pn, h_p_dict), Pn), -1))
        M = alg.add(M, alg.scale(term, -1j / 6 * abs(c) ** 2))
    for a in range(N):
        pn, cn = pauli_terms[a]
        Pn = {pn: 1.0 + 0j}
        for b in range(a + 1, N):
            pnp, cnp = pauli_terms[b]
            Pnp = {pnp: 1.0 + 0j}
            t1 = alg.add(alg.product(alg.product(Pn, h_p_dict), Pnp),
                         alg.product(alg.product(Pn, Pnp), h_p_dict),
                         alg.scale(alg.product(h_p_dict, alg.product(Pn, Pnp)), -5))
            t2 = alg.add(alg.product(h_p_dict, alg.product(Pnp, Pn)),
                         alg.product(Pnp, alg.product(h_p_dict, Pn)),
                         alg.product(Pnp, alg.product(Pn, h_p_dict)))
            contrib = alg.add(alg.scale(t1, np.conj(cn) * cnp), alg.scale(t2, cn * np.conj(cnp)))
            M = alg.add(M, alg.scale(contrib, 1j / 6))
    return M


def compute_K_mixed(alg, coherent_dicts, jump_pauli_terms, rho0_dict, O_dict):
    """K_mixed = Tr{[O, M_total] rho0}, Eq. Kmixed."""
    M_total = {}
    for h_p_dict in coherent_dicts:
        for pauli_terms in jump_pauli_terms:
            M_total = alg.add(M_total, Mpj_pauli(alg, h_p_dict, pauli_terms))
    comm = alg.commutator(O_dict, M_total)
    return (2 ** alg.n) * alg.trace_product(comm, rho0_dict)


# ---------------------------------------------------------------------------
# Gemcitabine-5-spin loader (validation harness; reuses the physical
# parameters from trotter_prf_vs_trot_gemcitabine5.py, but builds every
# operator directly as a Pauli dict -- no dense matrices involved except in
# the (separately run) cross-checks against the cached dense values).
# ---------------------------------------------------------------------------

def gemcitabine5_pauli_operators():
    spec = importlib.util.spec_from_file_location(
        'tpg', os.path.join(HERE, 'trotter_prf_vs_trot_gemcitabine5.py'))
    tpg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpg)

    n = 5
    alg = PauliAlgebra(n)

    h_zeeman = {}
    for i in range(n):
        h_zeeman = alg.add(h_zeeman, alg.scale(alg.iz(i), -tpg.gammas[i] * tpg.sigma_iso[i] * tpg.B_VEC[2]))
    h_edges = {}
    for (i, j) in tpg.edges:
        term = alg.add(alg.product(alg.ix(i), alg.ix(j)), alg.product(alg.iy(i), alg.iy(j)),
                        alg.product(alg.iz(i), alg.iz(j)))
        h_edges[(i, j)] = alg.scale(term, 2 * np.pi * tpg.J_HZ_5[i, j])
    coherent_dicts = [h_zeeman]
    for c in tpg.ordered_colors:
        gd = {}
        for e in tpg.groups[c]:
            gd = alg.add(gd, h_edges[e])
        coherent_dicts.append(gd)

    H0_dict = alg.add(*coherent_dicts)
    L_dicts = [dict(terms) for terms in tpg.jump_pauli_terms]

    coil_dict = {}
    rho0_dict = {}
    for i in range(n):
        coil_dict = alg.add(coil_dict, alg.scale(alg.ip(i), tpg.weights[i]))
        theta = (np.pi / 2) * tpg.weights[i]
        term = alg.add(alg.scale(alg.iz(i), np.cos(theta)), alg.scale(alg.ix(i), np.sin(theta)))
        rho0_dict = alg.add(rho0_dict, alg.scale(term, tpg.weights[i]))

    return dict(alg=alg, n=n, H0_dict=H0_dict, coherent_dicts=coherent_dicts,
                L_dicts=L_dicts, jump_pauli_terms=tpg.jump_pauli_terms,
                rho0_dict=rho0_dict, coil_dict=coil_dict)


if __name__ == '__main__':
    ops = gemcitabine5_pauli_operators()
    alg = ops['alg']

    t0 = time.time()
    K_prf = compute_K_prf(alg, ops['H0_dict'], ops['L_dicts'], ops['rho0_dict'], ops['coil_dict'])
    print(f'K_prf   = {K_prf}   ({time.time()-t0:.1f}s)')

    t0 = time.time()
    K_mixed = compute_K_mixed(alg, ops['coherent_dicts'], ops['jump_pauli_terms'],
                               ops['rho0_dict'], ops['coil_dict'])
    print(f'K_mixed = {K_mixed}   ({time.time()-t0:.1f}s)')

    cache_path = os.path.join(HERE, 'data', 'error_budget_coeffs_gemcitabine5.npz')
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        K_prf_dense = complex(cache['K_prf'])
        K_mixed_dense = complex(cache['K_mixed'])
        print(f'\ncross-check vs dense-matrix cache:')
        print(f'  K_prf   relative diff = {abs(K_prf - K_prf_dense) / abs(K_prf_dense):.2e}')
        print(f'  K_mixed relative diff = {abs(K_mixed - K_mixed_dense) / abs(K_mixed_dense):.2e}')
