"""
Error-budget-to-Dt solver (circ_sim/scripts/QRE/notes/qre.tex, Sec. dt_solver
"Total error budget over M Trotter steps: the Dt-solver").

Computes K_prf and K_mixed (never given explicit numeric values before this
script) for the truncated Gemcitabine system, caches them, and implements
the two-branch Dt solver:

  - K_nested ~ 0 (zero field, or axial field restricted to protected
    channels, per sec:nested_trot_numerics): per-step error is O(Dt^2), and
        Dt* = eps / (t * |K_prf + K_mixed|).
  - K_nested significantly nonzero (general/tilted field, or the (k=0,m!=0)
    axial-field channel): per-step error has a genuine O(Dt) piece whose
    M-step accumulation is Dt-INDEPENDENT (= t*K_nested). If the target
    eps <= t*|K_nested|, no Dt achieves it with the current (unmitigated,
    first-order) nested-Trotter gadget -- this is flagged, not solved.
    Otherwise the remaining budget is solved the same way as above.

K_nested is an explicit input (default 0, i.e. ZULF/protected-axial-field
assumption) -- to model off-ZULF operation, supply its value computed via
the same machinery as check_csa_dipolar_weight_safety.py.

The accumulation model itself (per-step coefficients -> Dt*(eps, t)) sits
behind the ErrorAccumulationModel interface below. LinearStateIndependentModel
is the only implementation for now (state-independent coefficients, linear-
in-t accumulation, per Eq. Mstep_estimate -- an assumption, not a proven
bound; see sec:dt_solver's "Two paths from per-step to M-step error"). This
exists so a future, refined model (e.g. one honoring the rigorous operator-
norm bound once K_mixed's locality-weighted bound is derived, or one that
re-evaluates coefficients at intermediate states) can be substituted without
touching any caller of solve_dt().
"""
import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'error_budget_coeffs_gemcitabine5.npz')


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_K_prf_dense(tpg):
    """K_prf = Tr{O C_prf[rho0]}, Eq. R_rho/Cprf/prf_leading_obs. Dense
    D_sys x D_sys matrices -- O(8^n), kept only as a cross-check reference
    for pauli_error_coefficients.compute_K_prf; not the default path."""
    H0 = tpg.H_iso
    Ls = tpg.jump_ops
    rho0 = tpg.rho0
    O = tpg.coil

    Gamma = sum(L.conj().T @ L for L in Ls)
    k2 = -1j * H0 - 0.5 * Gamma
    k4 = (-0.5 * H0 @ H0
          + (1j / 6) * (H0 @ Gamma + sum(L.conj().T @ H0 @ L for L in Ls) + Gamma @ H0)
          + (1 / 24) * Gamma @ Gamma)
    kappas = [(-0.5 * (H0 @ L + L @ H0) + (1j / 6) * (L @ Gamma)) for L in Ls]

    def lindbladian(rho):
        out = -1j * (H0 @ rho - rho @ H0)
        for L in Ls:
            out += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return out

    def R(rho):
        out = k4 @ rho + rho @ k4.conj().T + k2 @ rho @ k2.conj().T
        for L, kap in zip(Ls, kappas):
            out += -1j * L @ rho @ kap.conj().T + 1j * kap @ rho @ L.conj().T
        return out

    L2rho = lindbladian(lindbladian(rho0))
    C_prf = R(rho0) - 0.5 * L2rho
    return np.trace(O @ C_prf)


def compute_K_mixed_dense(tpg, ccts):
    """K_mixed = Tr{[O, M_total] rho0}, Eq. Kmixed (sec:combined_trot_nested).
    Dense D_sys x D_sys matrices -- O(8^n), kept only as a cross-check
    reference for pauli_error_coefficients.compute_K_mixed; not the default
    path (this is the one that took several minutes on a 5-spin system)."""
    M_total = ccts.build_M_total()

    def comm(A, B):
        return A @ B - B @ A

    return np.trace(comm(tpg.coil, M_total) @ tpg.rho0), M_total


def get_coefficients(force_recompute=False, method='pauli'):
    """K_prf, K_mixed for the truncated Gemcitabine system, cached to disk.

    method='pauli' (default): polynomial-in-n Pauli-string algebra
    (pauli_error_coefficients.py) -- scales to real molecule sizes, unlike
    the dense-matrix path.
    method='dense': the original O(8^n) dense-matrix computation, kept for
    cross-validation (K_mixed's build_M_total() takes several minutes even
    on this 5-spin test system).
    """
    if not force_recompute and os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return complex(data['K_prf']), complex(data['K_mixed'])

    if method == 'pauli':
        pec = _load_module('pec', 'pauli_error_coefficients.py')
        ops = pec.gemcitabine5_pauli_operators()
        alg = ops['alg']
        print('Computing K_prf (Pauli algebra) ...')
        K_prf = pec.compute_K_prf(alg, ops['H0_dict'], ops['L_dicts'], ops['rho0_dict'], ops['coil_dict'])
        print(f'  K_prf = {K_prf}')
        print('Computing K_mixed (Pauli algebra) ...')
        K_mixed = pec.compute_K_mixed(alg, ops['coherent_dicts'], ops['jump_pauli_terms'],
                                       ops['rho0_dict'], ops['coil_dict'])
        print(f'  K_mixed = {K_mixed}')
    elif method == 'dense':
        tpg = _load_module('tpg', 'trotter_prf_vs_trot_gemcitabine5.py')
        ccts = _load_module('ccts', 'check_combined_trotter_scaling.py')
        print('Computing K_prf (dense) ...')
        K_prf = compute_K_prf_dense(tpg)
        print(f'  K_prf = {K_prf}')
        print('Computing K_mixed (dense, this takes a few minutes) ...')
        K_mixed, _ = compute_K_mixed_dense(tpg, ccts)
        print(f'  K_mixed = {K_mixed}')
    else:
        raise ValueError(f"method must be 'pauli' or 'dense', got {method!r}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(CACHE_PATH, K_prf=K_prf, K_mixed=K_mixed)
    return K_prf, K_mixed


@dataclass
class ErrorCoefficients:
    """Per-step leading-order error coefficients (qre.tex sec:dt_solver)."""
    K_prf: complex
    K_mixed: complex
    K_nested: complex = 0.0


@dataclass
class DtSolution:
    achievable: bool
    Dt: Optional[float]
    M: Optional[int]
    floor: float
    note: str


class ErrorAccumulationModel(ABC):
    """Interface for M-step error accumulation models. Swap implementations
    to refine how per-step coefficients turn into Dt*(eps, t) without
    touching any caller."""

    @abstractmethod
    def solve_dt(self, eps: float, t: float, coeffs: ErrorCoefficients) -> DtSolution:
        """Given target absolute error eps and total evolution time t,
        return the largest Dt achieving it (or achievable=False if no Dt
        can, e.g. a Dt-independent error floor exceeds eps)."""
        raise NotImplementedError


class LinearStateIndependentModel(ErrorAccumulationModel):
    """Current (2026-09-18) model: two-branch solver from Eqs.
    Mstep_estimate/Dt_solver_zulf/Dt_solver_offzulf. Assumes K_prf, K_mixed,
    K_nested (evaluated at rho0) remain approximately valid at every one of
    the M steps -- a plausible, numerically-corroborated estimate, not a
    proven bound (sec:dt_solver, "Two paths from per-step to M-step error").
    """

    def solve_dt(self, eps: float, t: float, coeffs: ErrorCoefficients) -> DtSolution:
        K_dt2 = abs(coeffs.K_prf + coeffs.K_mixed)
        K_nested_mag = abs(coeffs.K_nested)
        floor = t * K_nested_mag

        negligible = K_nested_mag < 1e-12 * K_dt2 if K_dt2 > 0 else K_nested_mag < 1e-12
        if negligible:
            # K_nested negligible: Eq. Dt_solver_zulf
            Dt = eps / (t * K_dt2)
            return DtSolution(
                achievable=True, Dt=Dt, M=int(np.ceil(t / Dt)), floor=0.0,
                note='K_nested ~ 0 (zero field / protected axial-field channels): '
                     'simple O(Dt^2) solve.')

        if eps <= floor:
            return DtSolution(
                achievable=False, Dt=None, M=None, floor=floor,
                note=f'NOT ACHIEVABLE with the current (unmitigated) nested-Trotter '
                     f'gadget: target eps={eps:.3e} <= Dt-independent floor '
                     f't*|K_nested|={floor:.3e}. Implement a mitigation from '
                     f'sec:nested_trot_leading (grouping or Strang inner splitting) '
                     f'before choosing Dt.')

        Dt = (eps - floor) / (t * K_dt2)
        return DtSolution(
            achievable=True, Dt=Dt, M=int(np.ceil(t / Dt)), floor=floor,
            note=f'K_nested nonzero (away from ZULF/axial field): solved remaining '
                 f'budget after the Dt-independent floor t*|K_nested|={floor:.3e}.')


DEFAULT_MODEL = LinearStateIndependentModel()


def solve_dt(eps, t, K_prf, K_mixed, K_nested=0.0, model: ErrorAccumulationModel = None):
    """Convenience wrapper: solve_dt(...) with the default accumulation
    model. Equivalent to model.solve_dt(eps, t, ErrorCoefficients(...))."""
    model = model or DEFAULT_MODEL
    coeffs = ErrorCoefficients(K_prf=K_prf, K_mixed=K_mixed, K_nested=K_nested)
    return model.solve_dt(eps, t, coeffs)


if __name__ == '__main__':
    K_prf, K_mixed = get_coefficients()
    print(f'\nK_prf   = {K_prf}')
    print(f'K_mixed = {K_mixed}')
    print(f'|K_prf + K_mixed| = {abs(K_prf + K_mixed):.4f}')

    t_example = 50e-3  # 50 ms total simulated time, illustrative
    eps_example = 1e-3  # target absolute error in <coil>, illustrative

    print(f'\n--- Example: t={t_example*1e3:.1f} ms, target eps={eps_example:.1e} ---')

    print('\nZULF / axial-field, protected channels (K_nested = 0):')
    result = solve_dt(eps_example, t_example, K_prf, K_mixed, K_nested=0.0)
    print(f'  {result}')

    print('\nIllustrative off-ZULF case (K_nested = 50 + 0j, i.e. a non-negligible,'
          ' unsuppressed nested-Trotter coefficient), same target (eps=1e-3):')
    result = solve_dt(eps_example, t_example, K_prf, K_mixed, K_nested=50.0 + 0j)
    print(f'  {result}')

    print('\nSame off-ZULF case, but with a looser target (eps=10) above the'
          ' Dt-independent floor -- demonstrates the "solve remaining budget" branch:')
    result = solve_dt(10.0, t_example, K_prf, K_mixed, K_nested=50.0 + 0j)
    print(f'  {result}')
