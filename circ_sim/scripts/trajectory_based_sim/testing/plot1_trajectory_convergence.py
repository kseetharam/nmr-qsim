"""
Plot 1: convergence of the trajectory-based (U_unsplit) estimate of <coil>
with the number of trajectories N, at fixed delta_t = 1/J_max (the largest
J-coupling of truncated Gemcitabine, |J_F0-C0| = 226.85 Hz).

epsilon(N) = |<coil>_exact(delta_t) - <coil>_traj(N)|, where <coil>_exact is
the density-matrix-level exact Lindbladian propagation (single step,
molecule_operators.py's rho0 -> exact_rho(delta_t)), and <coil>_traj(N) is
the sample mean over N independent U_unsplit trajectories (Eq. 5 of
white_noise_trotter_1.pdf; trajectory_convergence.py's construction).

Deliberately no H0/noise Trotter SPLITTING here (per the user's request):
U_unsplit combines H0 and every noise kick into one exponential, so the
only two error sources present are (a) finite-N sampling error, expected to
shrink as sigma_O/sqrt(N), and (b) U_unsplit's own N-independent systematic
bias from representing continuous white noise as one piecewise-constant
kick per step (white_noise_trotter_1.pdf Sec. 3.3) -- NOT a splitting
error, but a real floor neither more trajectories nor different random
seeds can remove.

N range and methodology, per the pilot run: sigma_O ~ 0.052, and the
systematic floor ~ 1.6e-2 was already resolved unambiguously (54 sigma
above the N=30000 s.e.m.), giving a statistical/systematic crossover at
N_x = (sigma_O/floor)^2 ~ 10. Swept N = {10, 30, 100, 300, 1000, 3000}
(half-decade log-spacing) straddles this crossover and extends well past
it to confirm the plateau. To get a single smooth, reproducible curve
(not one independently-redrawn, noisy realization per N) rather than
costing 6 separate batches, ONE stream of N_max=3000 trajectories is drawn
once, and each swept N takes the cumulative mean of the first N draws from
that same stream.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
sys.path.insert(0, HERE)

from trajectory_convergence import build_hermitian_generators, draw_unsplit_unitary, sstt  # noqa: E402

J_MAX_HZ = 226.85
DT = 1.0 / J_MAX_HZ
N_SWEEP = [10, 30, 100, 300, 1000, 3000]
SEED = 0


def cumulative_means(vals):
    """Running mean after each of len(vals) draws."""
    return np.cumsum(vals) / np.arange(1, len(vals) + 1)


if __name__ == '__main__':
    H0, jump_ops, rho0, coil = sstt.H0, sstt.jump_ops, sstt.rho0, sstt.coil
    V_list = build_hermitian_generators(jump_ops)
    print(f'delta_t = 1/J_max = 1/{J_MAX_HZ} Hz = {DT:.6e} s')

    O_exact = sstt.expect(coil, sstt.exact_rho(DT))
    print(f'<coil>_exact(delta_t) = {O_exact}')

    N_max = max(N_SWEEP)
    rng = np.random.default_rng(SEED)
    print(f'\nDrawing one stream of {N_max} trajectories (seed={SEED})...')
    vals = np.empty(N_max, dtype=complex)
    for i in range(N_max):
        U = draw_unsplit_unitary(H0, V_list, DT, rng)
        rho_traj = U @ rho0 @ U.conj().T
        vals[i] = np.trace(coil @ rho_traj)

    running = cumulative_means(vals)
    sigma_O = vals.std(ddof=1)

    print(f'\n{"N":>6}  {"<coil>_traj(N)":>28}  {"eps(N)=|exact-traj|":>20}  '
          f'{"sigma_O/sqrt(N)":>16}')
    results = {'Dt': DT, 'J_max_Hz': J_MAX_HZ, 'O_exact': O_exact, 'seed': SEED,
               'sigma_O_full_stream': sigma_O, 'N_sweep': N_SWEEP, 'data': []}
    for N in N_SWEEP:
        O_traj_N = running[N - 1]
        eps_N = abs(O_exact - O_traj_N)
        sem_N = sigma_O / np.sqrt(N)
        print(f'{N:6d}  {str(np.round(O_traj_N, 6)):>28}  {eps_N:20.6e}  {sem_N:16.6e}')
        results['data'].append(dict(N=N, O_traj=O_traj_N, eps=eps_N, sem=sem_N))

    out_pkl = os.path.join(DATA_DIR, 'plot1_trajectory_convergence.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Ns = np.array(N_SWEEP)
    eps_vals = np.array([d['eps'] for d in results['data']])
    sem_vals = np.array([d['sem'] for d in results['data']])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(Ns, eps_vals, 'o-', color='C0', label=r'$\varepsilon(N)=|\langle O\rangle_{\rm exact}-\langle O\rangle_{\rm traj}(N)|$')
    ax.plot(Ns, sem_vals, 's--', color='C1', label=r'$\sigma_O/\sqrt{N}$ (statistical only)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of trajectories $N$')
    ax.set_ylabel('error')
    ax.set_title(r'Trajectory convergence, $U_{\rm unsplit}$, $\Delta t=1/J_{\max}$'
                  f' = {DT:.3e} s\n(no H0/noise splitting)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot1_trajectory_convergence.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
