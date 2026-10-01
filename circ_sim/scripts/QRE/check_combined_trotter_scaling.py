"""
Numerical check of the corrected combined-Trotter-error scaling derived in
circ_sim/scripts/QRE/notes/qre.tex, Sec. combined_trot_nested: once the
coherent fragments h_p AND every jump operator's own Pauli-term decomposition
are ALL first-order Trotterized together (not analyzed as two separate,
additively-combined error sources), the leading observable error is O(Dt^2),
sourced by a mixed coherent x same-jump-operator-Pauli-pair cross term
K_mixed -- not O(Dt^3) (the pure-coherent-only result) or O(Dt^3/2) (which
does not exist, per the ancilla-hopping-parity argument in
sec:nested_trot_leading).

This script:
  1. Builds M_{p,j} (Eq. Mpj_pair + three-distinct-fragment piece) and
     K_mixed = sum_{p,j} Tr{[O, M_{p,j}] rho0} for the truncated Gemcitabine
     5-spin system (reusing trotter_prf_vs_trot_gemcitabine5.py's machinery).
  2. Runs a fine Dt sweep (well below the delta_t/50 floor used in the
     original small-Dt sweep) of the FULLY Trotterized (coherent-fragmented +
     nested-jump-Trotterized) single-step observable error, and checks that
     its local log-log slope converges to 2.00 and its magnitude converges
     to the closed-form prediction Dt^2 * K_mixed.

Zero field (this system's B_z = 5e-7 T, ZULF): the CSA/ladder-phase-lock
mechanism protecting the O(Dt) and O(Dt^3/2) terms is exact here (Sec.
nested_trot_numerics), so this is the cleanest regime to see the O(Dt^2)
mixed term's asymptotics unobscured.
"""
import os
import sys
import time
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'tpg', os.path.join(HERE, 'trotter_prf_vs_trot_gemcitabine5.py'))
tpg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpg)

D = tpg.D_sys


def _vtilde_local(pstr, c):
    """Tilde-V_{j,n} embedded in the local 2*D_sys ancilla-pair block {0,j}."""
    P = tpg._pauli_string_matrix(pstr)
    block = np.zeros((2 * D, 2 * D), dtype=complex)
    block[D:2 * D, 0:D] = c * P
    block[0:D, D:2 * D] = np.conj(c) * P
    return block


def _hemb(h):
    return np.kron(np.eye(2), h)


def _sym3(A, B, C):
    return A @ B @ C + A @ C @ B + B @ A @ C + B @ C @ A + C @ A @ B + C @ B @ A


def build_M_total():
    """M_total = sum_{p,j} M_{p,j}, Eq. Mpj_pair + three-distinct-fragment piece."""
    M_total = np.zeros((D, D), dtype=complex)
    t0 = time.time()
    for p_idx, h_p in enumerate(tpg.coherent_pieces):
        Hp = _hemb(h_p)
        for pauli_terms in tpg.jump_pauli_terms:
            N = len(pauli_terms)
            cs = [c for (_, c) in pauli_terms]
            Ps = [tpg._pauli_string_matrix(pstr) for (pstr, _) in pauli_terms]
            Vs = [_vtilde_local(pstr, c) for (pstr, c) in pauli_terms]

            # pairwise piece (closed form, Eq. Mpj_pair)
            for n in range(N):
                M_total += -(1j / 6) * (abs(cs[n]) ** 2) * (h_p - Ps[n] @ h_p @ Ps[n])

            # three-distinct-fragment piece (no compact closed form; brute force)
            for n in range(N):
                for np_ in range(n + 1, N):
                    a, b, c = Hp, Vs[n], Vs[np_]
                    full = 1j * (_sym3(a, b, c) / 6 - a @ b @ c)
                    M_total += full[0:D, 0:D]
        print(f'  coherent fragment {p_idx + 1}/{len(tpg.coherent_pieces)} done, '
              f'elapsed {time.time() - t0:.1f}s')
    return M_total


def main():
    print('Building M_total = sum_{p,j} M_{p,j} (this takes a few minutes)...')
    M_total = build_M_total()

    def comm(A, B):
        return A @ B - B @ A

    K_mixed = np.trace(comm(tpg.coil, M_total) @ tpg.rho0)
    print(f'\nK_mixed = {K_mixed:.6f}   |K_mixed| = {abs(K_mixed):.4f}')
    print(f'||M_total||_F = {np.linalg.norm(M_total):.4f}')

    delta_t = tpg.delta_t
    fine_Dts = [delta_t / f for f in (100, 200, 500, 1000, 2000, 5000)]

    print(f'\ndelta_t = {delta_t * 1e3:.4f} ms\n')
    print('Fully-Trotterized (coherent + nested) single-step error vs. Dt^2*K_mixed:')
    results = []
    for Dt in fine_Dts:
        dil_step = tpg.make_dilation_step(Dt)
        trot_step = tpg.make_trotter_step(Dt, tpg.order_A)
        O_dil = tpg.expect(tpg.coil, dil_step(tpg.rho0))
        O_trot = tpg.expect(tpg.coil, trot_step(tpg.rho0))
        trot_err = O_dil - O_trot
        predicted = Dt ** 2 * K_mixed
        results.append((Dt, trot_err, predicted))
        print(f'  Dt=delta_t/{delta_t / Dt:.0f}: trot={trot_err:.6e}  '
              f'predicted={predicted:.6e}  |ratio|={abs(trot_err / predicted):.4f}')

    print('\nLocal log-log slopes of |trot| vs Dt (expect -> 2.00):')
    for i in range(len(results) - 1):
        Dt1, e1, _ = results[i]
        Dt2, e2, _ = results[i + 1]
        slope = np.log(abs(e1) / abs(e2)) / np.log(Dt1 / Dt2)
        print(f'  delta_t/{delta_t / Dt1:.0f} -> delta_t/{delta_t / Dt2:.0f}: slope = {slope:.4f}')


if __name__ == '__main__':
    main()
