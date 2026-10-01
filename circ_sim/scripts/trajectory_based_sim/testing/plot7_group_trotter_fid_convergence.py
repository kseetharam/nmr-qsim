"""
Group-Trotterized, trajectory-sampled FID vs. the fully exact FID/spectrum
(plot6_exact_fid_spectrum_refined.py), at the validated parameters
delta_t=(1/J_max)/2, T=2 s (N_t=907 points, M=906 propagation steps) --
the actual "how bad/good is this approach" comparison this project's
trajectory-based exploration has been building up to.

Per-trajectory unitary, per step: identical to
plot4_group_trotter_convergence.py's scheme (the more refined of the two
built so far, not plot3's single-joint-exponential "Lie" scheme) --
exp(-i*H0*dt) exactly, followed by a first-order sequential Trotter product
over the 16 mutually-commuting Pauli-term groups (pooled across all 25
Hermitian noise generators, precomputed once, dt/trajectory-independent;
see that module's docstring for why this grouping is exact and
trajectory-independent here). Reused directly from plot4
(H0, rho0, coil, D_sys, N_GEN, COEF_MATRIX, GROUP_TERM_INDICES, GROUP_MATS),
not re-derived.

Trajectory semantics (resolved explicitly, since "N trajectories per time
point" is ambiguous and the two readings differ in cost by ~M/2 ~ 450x):
each of the N trajectories is ONE independent, continuously-evolved path,
propagated once through all M steps with fresh noise draws at every step,
recording <coil> at every intermediate step along the way -- not M^2-scaling
independent re-draws per time point. This is both the only computationally
tractable reading (~100*M step-applications instead of ~100*M^2/2) and the
standard meaning of "trajectory" in quantum-trajectory/stochastic
unravelling methods generally.

Convergence in N: one stream of N_MAX=400 trajectories is drawn once (fresh
noise every step, but the SAME stream reused across the N-subsets below --
not three independent re-draws), and N=100/200/400 are obtained as
cumulative-mean subsets of that one stream, mirroring
plot1_trajectory_convergence.py's established convention.

Reference: plot6_exact_fid_spectrum_refined.py's already-computed and
already-inspected exact FID/spectrum, loaded from its saved pickle rather
than recomputed, so this comparison is against literally what was already
reviewed, not a fresh (potentially subtly different) recomputation.

Cost: ~3.3 ms/trajectory/step (16 sequential group exponentials, batched
across trajectories per step -- see plot4's docstring for the einsum-vs-
matmul lesson this relies on), calibrated directly before committing to the
full run: 10 steps at N=400 took ~13.4s, so the full M=906 steps is
expected to take ~20 minutes.
"""
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

from plot4_group_trotter_convergence import (  # noqa: E402
    H0, rho0, coil, D_sys, N_GEN, COEF_MATRIX, GROUP_TERM_INDICES, GROUP_MATS,
)


def batched_group_trotter_fid(dt, M, N_traj, rng, print_every=100):
    """<coil> at t_m = m*dt, m=0..M, for N_traj independent, continuously-
    evolved group-Trotter trajectories. Returns (N_traj, M+1) complex array."""
    evalsH, evecsH = np.linalg.eigh(H0 * dt)
    U_H = (evecsH * np.exp(-1j * evalsH)) @ evecsH.conj().T

    rho = np.broadcast_to(rho0, (N_traj, D_sys, D_sys)).copy()
    vals = np.empty((N_traj, M + 1), dtype=complex)
    vals[:, 0] = np.einsum('ab,nba->n', coil, rho)

    t0 = time.time()
    for m in range(1, M + 1):
        dW = rng.normal(scale=np.sqrt(dt), size=(N_traj, N_GEN))
        coefs = dW @ COEF_MATRIX.T
        U_V = np.broadcast_to(np.eye(D_sys, dtype=complex), (N_traj, D_sys, D_sys)).copy()
        for idxs, gm in zip(GROUP_TERM_INDICES, GROUP_MATS):
            w = coefs[:, idxs]
            G = np.einsum('mi,iab->mab', w, gm)
            evals, evecs = np.linalg.eigh(G)
            U_g = (evecs * np.exp(-1j * evals)[:, None, :]) @ np.conj(np.transpose(evecs, (0, 2, 1)))
            U_V = U_g @ U_V
        U = U_H[None, :, :] @ U_V
        Udag = np.conj(np.transpose(U, (0, 2, 1)))
        rho = (U @ rho) @ Udag
        vals[:, m] = np.einsum('ab,nba->n', coil, rho)
        if m % print_every == 0 or m == M:
            elapsed = time.time() - t0
            print(f'  step {m}/{M}  ({elapsed:.1f}s elapsed, '
                  f'{elapsed / m * (M - m):.1f}s remaining est.)', flush=True)
    return vals


if __name__ == '__main__':
    ref_pkl = os.path.join(DATA_DIR, 'plot6_exact_fid_spectrum_refined.pkl')
    with open(ref_pkl, 'rb') as fh:
        ref = pickle.load(fh)
    DT, N_T = ref['DT'], ref['N_T']
    fid_exact, freqs, spectrum_exact = ref['fid'], ref['freqs'], ref['spectrum']
    M = N_T - 1
    print(f'Loaded exact reference: DT={DT:.6e} s, N_t={N_T}, M={M} steps, '
          f'T_total={ref["T_total"]:.4f} s')

    N_MAX = 400
    N_CHECK = [100, 200, 400]
    SEED = 0
    rng = np.random.default_rng(SEED)

    print(f'\nDrawing {N_MAX} group-Trotter trajectories, {M} steps each...')
    vals = batched_group_trotter_fid(DT, M, N_MAX, rng)

    results = {'DT': DT, 'N_T': N_T, 'M': M, 'seed': SEED, 'N_check': N_CHECK,
               't_grid': ref['t_grid'], 'freqs': freqs,
               'fid_exact': fid_exact, 'spectrum_exact': spectrum_exact, 'series': []}

    print(f'\n{"N":>6}  {"FID RMS err":>14}  {"spectrum RMS err":>18}')
    for N in N_CHECK:
        fid_traj = vals[:N].mean(axis=0)
        spectrum_traj = np.fft.fftshift(np.fft.fft(fid_traj))
        fid_rms = np.sqrt(np.mean(np.abs(fid_traj - fid_exact) ** 2))
        spec_rms = np.sqrt(np.mean(np.abs(spectrum_traj - spectrum_exact) ** 2))
        print(f'{N:6d}  {fid_rms:14.4e}  {spec_rms:18.4e}')
        results['series'].append(dict(N=N, fid_traj=fid_traj, spectrum_traj=spectrum_traj,
                                       fid_rms=fid_rms, spec_rms=spec_rms))

    out_pkl = os.path.join(DATA_DIR, 'plot7_group_trotter_fid_convergence.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_grid = ref['t_grid']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    ax1.plot(t_grid, np.abs(fid_exact), '-', color='k', linewidth=1.5, label='exact')
    colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(N_CHECK)))
    for series, color in zip(results['series'], colors):
        ax1.plot(t_grid, np.abs(series['fid_traj']), '-', color=color, linewidth=1,
                  label=f"N={series['N']}")
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel(r'$|\langle\mathrm{coil}(t)\rangle|$')
    ax1.set_title(rf'Group-Trotterized trajectory FID vs. exact, $\Delta t={DT:.4e}$ s')
    ax1.legend(fontsize=9)

    ax2.plot(freqs, np.abs(spectrum_exact), '-', color='k', linewidth=1.5, label='exact')
    for series, color in zip(results['series'], colors):
        ax2.plot(freqs, np.abs(series['spectrum_traj']), '-', color=color, linewidth=1,
                  label=f"N={series['N']}")
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel(r'$|\mathrm{FFT}[\langle\mathrm{coil}\rangle](f)|$')
    ax2.set_title('Spectrum: group-Trotterized trajectory average vs. exact')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot7_group_trotter_fid_convergence.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
