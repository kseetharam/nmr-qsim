"""
Plot 2: scaling of the trajectory-based (U_unsplit) observable error with
delta_t, at FIXED trajectory count N=3000 (the value Plot 1 showed already
sits well past the statistical/systematic crossover at delta_t=1/J_max).

epsilon(delta_t) = |<coil>_exact(delta_t) - <coil>_traj(N=3000; delta_t)|,
delta_t swept over {0.1, 0.3, 1, 3, 10} x (1/J_max), J_max=226.85 Hz
(truncated Gemcitabine's strongest coupling, F0-C0). Still U_unsplit (Eq. 5
of white_noise_trotter_1.pdf) throughout -- no H0/noise Trotter splitting,
per the same convention as Plot 1. The only delta_t-dependence being probed
here is (a) how U_unsplit's own systematic (piecewise-constant-noise) bias
scales with delta_t, and (b) how the statistical (sampling) error at fixed
N changes with delta_t through sigma_O(delta_t) -- NOT a splitting error.

Variance reduction (common random numbers): a single set of 3000x25
standard-normal draws Z is generated ONCE; at each delta_t, the noise
kicks are dW = Z*sqrt(delta_t). This reuses the identical underlying
randomness at every delta_t (same distribution, N(0,delta_t), just drawn
via a shared seed rescaled appropriately) so the resulting curve reflects
the true delta_t-trend rather than being additionally scrambled by
independent statistical noise at each point -- the delta_t-sweep analogue
of Plot 1's single cumulative trajectory stream.
"""
import os
import pickle
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
sys.path.insert(0, HERE)

from trajectory_convergence import build_hermitian_generators, sstt  # noqa: E402

J_MAX_HZ = 226.85
DT_UNIT = 1.0 / J_MAX_HZ
DT_MULTIPLIERS = [0.1, 0.3, 1.0, 3.0, 10.0]
N_FIXED = 3000
SEED = 0


def trajectory_batch_crn(H0, V_list, rho0, coil, dt, Z):
    """Z: (N, n_generators) standard-normal draws, shared across delta_t
    values (common random numbers). dW = Z*sqrt(dt). Returns array of
    per-trajectory <coil> estimates, length N."""
    N, n = Z.shape
    dW_all = Z * np.sqrt(dt)
    vals = np.empty(N, dtype=complex)
    for i in range(N):
        generator = H0 * dt
        for w, V in zip(dW_all[i], V_list):
            generator = generator + w * V
        U = expm(-1j * generator)
        rho_traj = U @ rho0 @ U.conj().T
        vals[i] = np.trace(coil @ rho_traj)
    return vals


if __name__ == '__main__':
    H0, jump_ops, rho0, coil = sstt.H0, sstt.jump_ops, sstt.rho0, sstt.coil
    V_list = build_hermitian_generators(jump_ops)
    n_gen = len(V_list)
    print(f'{n_gen} Hermitian generators; N={N_FIXED} trajectories/point; '
          f'1/J_max = {DT_UNIT:.6e} s')

    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal(size=(N_FIXED, n_gen))  # shared across the whole sweep

    results = {'J_max_Hz': J_MAX_HZ, 'DT_unit': DT_UNIT, 'N': N_FIXED, 'seed': SEED,
               'dt_multipliers': DT_MULTIPLIERS, 'data': []}
    print(f'\n{"dt/((1/Jmax))":>14}  {"delta_t (s)":>14}  {"eps(delta_t)":>14}  '
          f'{"sigma_O":>12}  {"sigma_O/sqrt(N)":>16}')
    for mult in DT_MULTIPLIERS:
        dt = mult * DT_UNIT
        O_exact = sstt.expect(coil, sstt.exact_rho(dt))
        vals = trajectory_batch_crn(H0, V_list, rho0, coil, dt, Z)
        O_traj_mean = vals.mean()
        sigma_O = vals.std(ddof=1)
        eps = abs(O_exact - O_traj_mean)
        sem = sigma_O / np.sqrt(N_FIXED)
        print(f'{mult:14.3g}  {dt:14.6e}  {eps:14.6e}  {sigma_O:12.4e}  {sem:16.4e}')
        results['data'].append(dict(mult=mult, dt=dt, O_exact=O_exact, O_traj=O_traj_mean,
                                     eps=eps, sigma_O=sigma_O, sem=sem))

    out_pkl = os.path.join(DATA_DIR, 'plot2_trajectory_vs_dt.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dts = np.array([d['dt'] for d in results['data']])
    eps_vals = np.array([d['eps'] for d in results['data']])
    sem_vals = np.array([d['sem'] for d in results['data']])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(dts, eps_vals, 'o-', color='C0',
            label=r'$\varepsilon(\Delta t)=|\langle O\rangle_{\rm exact}-\langle O\rangle_{\rm traj}(N=3000)|$')
    ax.plot(dts, sem_vals, 's--', color='C1', label=r'$\sigma_O(\Delta t)/\sqrt{3000}$ (statistical only)')
    ax.axvline(DT_UNIT, color='gray', linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t$ (s)')
    ax.set_ylabel('error')
    ax.set_title(r'Trajectory ($U_{\rm unsplit}$) error vs. $\Delta t$, $N=3000$ fixed'
                  '\n(no H0/noise splitting)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot2_trajectory_vs_dt.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
