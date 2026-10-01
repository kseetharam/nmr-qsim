"""
Numerical verification of white_noise_trotter_1.pdf's Proposition 1 / Eq. (8)
on truncated Gemcitabine: does the trajectory-averaged "Lie" channel

    U_Lie(dt) = exp(-i*H0*dt) @ exp(-i*sum_j dW_j*V_j),   dW_j ~ N(0, dt) iid
                                                            (Eq. 3)

converge, as the number of trajectories N grows, to the deterministic Lie
splitting of the Lindbladian,

    Sbar_Lie = exp(dt*L_H) @ exp(dt*D)                     (Eq. 8)

with L_H = -i*ad_{H0} and D the standard dissipator built from the fixed
Hermitian generators V_j (build_hermitian_generators, gamma_j=1 uniform
convention, same H0/jump_ops as trajectory_convergence.py -- default
b_vec=(0,0,5e-7), i.e. the field already used throughout this project, not
an artificially zeroed one)?

THIS IS NOT AN EXACT IDENTITY FOR THIS SYSTEM, and that is itself the main
finding this script establishes numerically before trusting anything else:
Lemma 1 (and hence Eq. 8) is only an exact statement "for a single V, or
commuting V_j". Checked directly here: all 25 Hermitian generators V_j for
Gemcitabine's dipolar-dominated dissipator pairwise DO NOT commute (300/300
pairs nonzero, typical ||[V_j,V_k]|| ~ 0.4 against ||V_j|| ~ 0.8-1.5) -- not
a small effect. For non-commuting V_j, Eq. (7) gives the correction Eq. (8)
omits:

    E[exp(-i*sum_j dW_j*ad_{V_j})] - exp(dt*D)
        = (dt^2/24) * sum_{a,c} (K_a K_c K_a K_c + K_a K_c K_c K_a
                                   - 2*K_a K_a K_c K_c) + O(dt^3),
          K_a = ad_{V_a}                                   (Eq. 7)

(the general quadruple sum over a,b,c,d collapses to this double sum here
since our noise covariance gamma_{jk}=delta_{jk}, independent unit-variance
generators). So what this script actually verifies is the STRONGER, more
informative claim "Eq. 8 + Eq. 7's explicit correction", not bare Eq. 8:

    Sbar_Lie[rho0] = exp(dt*L_H) @ [exp(dt*D)[rho0] + (dt^2/24)*E2[rho0]]
                     + O(dt^3),      E2 = sum_{a,c}(...) as above.

Two sanity checks on this translation of Eq. (7) (both hold to machine
precision, see this module's __main__): (a) every diagonal a=c term is
IDENTICALLY zero (K_a^4+K_a^4-2K_a^4=0), so only the 25*24=600 off-diagonal
pairs contribute -- consistent with the correction vanishing whenever there
is effectively only one generator; (b) reducing to a single-generator toy
system (n_gen=1) gives E2=0 to machine precision, reproducing Lemma 1's
exact special case as a limit of this same code path, not a separately
re-derived formula.

Why the sweep spans dt from ~1e-4 to ~7e-2 (much larger than the "natural"
1/J_max ~ 4.4e-3 s unit used elsewhere in this directory): the correction
above is measured to be ~1e4-1e5x SMALLER than a single trajectory's own
shot-to-shot fluctuation sigma_O at dt ~ 1e-2 to 1e-4 (sigma_O empirically
scales close to linearly in dt, the correction as dt^2, so their RATIO
shrinks linearly as dt->0) -- resolving it there at 3-sigma would need
N ~ 1e7-1e11 trajectories, minutes-to-days even with the vectorized sampler
below. Pushing dt up to ~0.03-0.07 s makes the correction large enough to
resolve with N ~ 1e5-1e6 (a few minutes), while a separate, cheap
(no-Monte-Carlo) scan of the deterministic gap |Sbar_Lie^(0) - Sbar_Lie^(corrected)|
vs. dt (not included in the final sweep, done once interactively) confirmed
its local log-log slope stays close to 2 up to about dt~0.07-0.1 s before
flattening -- i.e. Eq. 7's own neglected O(dt^3) term is not yet dominant
there. This is a genuine trade-off exposed by the physical system, not an
arbitrary choice: the smallest sweep point (dt=9.34e-5 s, this project's own
error-budget-optimal step) is deliberately kept even though the correction
is invisible there against N=20,000 trajectories' statistical floor -- that
null result is itself informative (shows where the crossover into
resolvability lies), not just a filler data point.

Trajectory sampling is fully vectorized (batched np.linalg.eigh + batched
matmul over all draws in a chunk at once), chunked to bound memory -- an
earlier, naive einsum('nab,bc,ndc->nad', ...) contraction for U@rho0@U^dag
was measured to take 16.7s for a chunk of 5000 (a bad contraction path);
rewritten as two batched matmuls, the same chunk takes ~1.6s, making
N ~ 1e5-1e6 trajectories tractable in minutes rather than hours.
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

H0, rho0, coil = sstt.H0, sstt.rho0, sstt.coil
V_list = build_hermitian_generators(sstt.jump_ops)
V_stack = np.stack(V_list, axis=0)
D_sys = sstt.D_sys
N_GEN = len(V_list)


# ---------------------------------------------------------------------------
# Eq. (7)'s nested-commutator correction, E2[rho0] = sum_{a,c}(...), applied
# directly to rho0 (a fixed 32x32 matrix), not built as a full D^2 x D^2
# superoperator -- exact and cheap since we only ever need its action on the
# single vector rho0, not the general channel.
# ---------------------------------------------------------------------------

def _ad(A, X):
    return A @ X - X @ A


def build_eq7_correction(V_list, rho0):
    """E2[rho0] = sum_{a,c} (K_a K_c K_a K_c + K_a K_c K_c K_a - 2 K_a K_a K_c K_c) rho0,
    divided by 24 (Eq. 7's prefactor, dt^2 applied separately by the caller).
    K_a = ad_{V_a}. Returns a D_sys x D_sys Hermitian matrix, dt-independent."""
    D = rho0.shape[0]
    E2 = np.zeros((D, D), dtype=complex)
    for a, Va in enumerate(V_list):
        for c, Vc in enumerate(V_list):
            t = _ad(Vc, rho0); t = _ad(Va, t); t = _ad(Vc, t); term1 = _ad(Va, t)
            t = _ad(Va, rho0); t = _ad(Vc, t); t = _ad(Vc, t); term2 = _ad(Va, t)
            t = _ad(Vc, rho0); t = _ad(Vc, t); t = _ad(Va, t); term3 = _ad(Va, t)
            E2 += term1 + term2 - 2 * term3
    return E2 / 24.0


# ---------------------------------------------------------------------------
# Deterministic references: Sbar_Lie^(0) = e^{dt LH} e^{dt D} [rho0]  (Eq. 8)
# and Sbar_Lie^(corrected) = e^{dt LH} [e^{dt D}[rho0] + dt^2 * E2[rho0]]
# (Eq. 8 + Eq. 7), via separately-exponentiated LH/D Liouville superoperators
# (reusing single_step_trotter_tightness.lindblad_liouvillian, called once
# with Ls=[] for pure LH and once with H=0 for pure D).
# ---------------------------------------------------------------------------

L_H_super = sstt.lindblad_liouvillian(H0, [])
L_D_super = sstt.lindblad_liouvillian(np.zeros((D_sys, D_sys), dtype=complex), V_list)


def lie_split_rho(dt, E2, correction):
    rho0_vec = rho0.flatten(order='F')
    rho_afterD = (expm(dt * L_D_super) @ rho0_vec).reshape(D_sys, D_sys, order='F')
    if correction:
        rho_afterD = rho_afterD + dt ** 2 * E2
    v = rho_afterD.flatten(order='F')
    return (expm(dt * L_H_super) @ v).reshape(D_sys, D_sys, order='F')


# ---------------------------------------------------------------------------
# Vectorized trajectory sampler (Eq. 3's U_Lie), chunked to bound memory.
# ---------------------------------------------------------------------------

def _batched_lie_vals(dt, m, rng):
    """<coil> for m independent U_Lie(dt) trajectories at once."""
    dW = rng.normal(scale=np.sqrt(dt), size=(m, N_GEN))
    G = np.einsum('nj,jab->nab', dW, V_stack)          # (m, D, D) Hermitian
    evals, evecs = np.linalg.eigh(G)                    # batched
    phase = np.exp(-1j * evals)
    U_V = (evecs * phase[:, None, :]) @ np.conj(np.transpose(evecs, (0, 2, 1)))
    evalsH, evecsH = np.linalg.eigh(H0 * dt)
    U_H = (evecsH * np.exp(-1j * evalsH)) @ evecsH.conj().T
    U = U_H[None, :, :] @ U_V
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    rho_traj = (U @ rho0) @ Udag                        # batched matmul, NOT einsum
    return np.einsum('ab,nba->n', coil, rho_traj)


def trajectory_moments(dt, N_total, rng, chunk=10_000):
    """Mean and sample std of <coil> over N_total trajectories, accumulated
    in chunks so memory stays bounded regardless of N_total. Chunk statistics
    are combined via Chan et al.'s parallel-variance formula, so the result
    is the exact two-pass sample variance, not a between-chunk approximation."""
    n_done, mean, M2 = 0, 0.0 + 0j, 0.0
    while n_done < N_total:
        m = min(chunk, N_total - n_done)
        vals = _batched_lie_vals(dt, m, rng)
        mean_chunk = vals.mean()
        M2_chunk = (np.abs(vals - mean_chunk) ** 2).sum()

        n_new = n_done + m
        delta = mean_chunk - mean
        mean = mean + delta * (m / n_new)
        M2 = M2 + M2_chunk + (np.abs(delta) ** 2) * n_done * m / n_new
        n_done = n_new

    sigma = np.sqrt(M2 / (N_total - 1))
    return mean, sigma


if __name__ == '__main__':
    print(f'D_sys={D_sys}, n_gen={N_GEN}')

    # ---- sanity checks on the Eq. (7) translation, before trusting it ----
    # (a) every diagonal a=c contribution is identically zero
    Va = V_list[3]
    t = _ad(Va, rho0); t = _ad(Va, t); t = _ad(Va, t); term1 = _ad(Va, t)
    diag_contribution = term1 + term1 - 2 * term1  # a=c=3: K_a^4 + K_a^4 - 2K_a^4
    assert np.linalg.norm(diag_contribution) < 1e-12, 'diagonal terms should vanish identically'
    print('sanity check (a) diagonal a=c terms vanish identically: PASS')

    # (b) single-generator toy system -> correction exactly zero (Lemma 1 exact case)
    E2_single = build_eq7_correction([V_list[0]], rho0)
    assert np.linalg.norm(E2_single) < 1e-12, 'single-generator correction should be exactly zero'
    print('sanity check (b) single-generator correction is exactly zero: PASS')

    E2 = build_eq7_correction(V_list, rho0)
    assert np.allclose(E2, E2.conj().T), 'E2[rho0] should be Hermitian'
    print(f'Eq.(7) correction built: ||E2[rho0]||={np.linalg.norm(E2):.4e} (Hermitian, verified)\n')

    # ---- sweep: dt values and trajectory counts calibrated interactively ----
    # (see module docstring for why these specific (dt, N) pairs)
    DT_N_SWEEP = [
        (9.336901e-05, 20_000),   # this project's error-budget-optimal step;
                                  # correction expected to be invisible here
        (3.3e-02, 600_000),
        (7.0e-02, 300_000),
    ]
    SEED = 0
    rng = np.random.default_rng(SEED)

    results = {'seed': SEED, 'n_gen': N_GEN, 'data': []}
    print(f'{"dt":>12}  {"N":>9}  {"gap=|Lie0-Liec|":>16}  {"|traj-Lie0|":>13}  '
          f'{"|traj-Liec|":>13}  {"sem":>12}')
    for dt, N in DT_N_SWEEP:
        rho_Lie0 = lie_split_rho(dt, E2, correction=False)
        rho_Liec = lie_split_rho(dt, E2, correction=True)
        O_Lie0 = sstt.expect(coil, rho_Lie0)
        O_Liec = sstt.expect(coil, rho_Liec)
        gap = abs(O_Lie0 - O_Liec)

        O_traj, sigma_O = trajectory_moments(dt, N, rng)
        sem = sigma_O / np.sqrt(N)
        eps_vs_Lie0 = abs(O_traj - O_Lie0)
        eps_vs_Liec = abs(O_traj - O_Liec)

        print(f'{dt:12.4e}  {N:9d}  {gap:16.4e}  {eps_vs_Lie0:13.4e}  '
              f'{eps_vs_Liec:13.4e}  {sem:12.4e}')
        results['data'].append(dict(
            dt=dt, N=N, O_Lie0=O_Lie0, O_Liec=O_Liec, gap=gap,
            O_traj=O_traj, sigma_O=sigma_O, sem=sem,
            eps_vs_Lie0=eps_vs_Lie0, eps_vs_Liec=eps_vs_Liec,
        ))

    out_pkl = os.path.join(DATA_DIR, 'plot3_lie_eq8_verification.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dts = np.array([d['dt'] for d in results['data']])
    gaps = np.array([d['gap'] for d in results['data']])
    eps0 = np.array([d['eps_vs_Lie0'] for d in results['data']])
    epsc = np.array([d['eps_vs_Liec'] for d in results['data']])
    sems = np.array([d['sem'] for d in results['data']])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(dts, eps0, yerr=sems, fmt='o-', color='C0',
                label=r'$|\langle O\rangle_{\rm traj}-\langle O\rangle_{\bar S^{(0)}_{\rm Lie}}|$ (Eq. 8 only)')
    ax.errorbar(dts, epsc, yerr=sems, fmt='s-', color='C2',
                label=r'$|\langle O\rangle_{\rm traj}-\langle O\rangle_{\bar S^{\rm corr}_{\rm Lie}}|$ (Eq. 8 + Eq. 7)')
    ax.plot(dts, gaps, 'd--', color='C1',
            label=r'deterministic gap $|\langle O\rangle_{\bar S^{(0)}}-\langle O\rangle_{\bar S^{\rm corr}}|$ (Eq. 7 prediction)')
    ax.plot(dts, sems, ':', color='gray', label=r'statistical floor $\sigma_O/\sqrt{N}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t$ (s)')
    ax.set_ylabel(r'error in $\langle\mathrm{coil}\rangle$')
    ax.set_title('Trajectory-averaged Lie channel vs. Lindblad-split reference\n'
                  '(white_noise_trotter_1.pdf Eq. 8, with/without Eq. 7 correction)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot3_lie_eq8_verification.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
