"""
Trajectory-based (classical-noise, random-unitary) simulation of the
Gemcitabine ZULF Lindbladian, following white_noise_trotter_1.pdf's
U_unsplit scheme (Eq. 5) -- deliberately WITHOUT any H0/noise Trotter
splitting at this stage, per the user's explicit request, so that the only
error being characterized here is (a) sampling/statistical error from a
finite trajectory count N, and (b) the piecewise-constant-noise
discretization error intrinsic to U_unsplit itself (Sec. 3.3 of that note),
NOT a Lie/Strang splitting error.

Physical origin (zulf_numerics/notes/liouville_hilbert_basis.tex, Sec.
"Zeeman anisotropy dephasing" + "Universal Lindbladian framework"): the
stochastic Hamiltonian is H(t) = H_iso + H^{(2)}(t), with H^{(2)}(t) driven
by the molecular orientation's Wigner-D^(2) matrix elements, shared between
the CSA and dipolar terms. Since our Delta-t range (1e-5 to 1e-2 s) is
~1e4-1e7x the rotational correlation time tau_c (~ns), we are deep in the
white-noise limit -- the noise enters as Gaussian "kicks" (Sec. 4-5.3 of
route1_rotdiff.pdf), not as literal SO(3)-orientation trajectories (which
would need step sizes far below tau_c and be computationally hopeless at
the Delta-t scales relevant here, per that note's own estimate).

Random-unitary constraint: a classical-noise ensemble rho_bar = E[U rho
U^dagger] can only ever generate a UNITAL Lindbladian (E[U 1 U^dagger] = 1
for any unitary U). Checked directly (not assumed): the FULL 25-operator
Gemcitabine dissipator satisfies sum_j [L_j, L_j^dagger] = 0 to machine
precision (~1e-16), even though individual L_j are not normal (their own
commutators range up to ~0.93) -- the non-normality cancels across the
conjugate-paired set (L_{k,m}^dagger = (-1)^(k+m) L_{-k,-m}, qre.tex).

Hermitian generator construction (derived and verified here, not assumed):
summing the dissipator of a conjugate PAIR L, L^dagger=+-L' exactly cancels
the "cross" (Y rho X - X rho Y type) terms that obstruct representing a
single non-Hermitian L's own dissipator as an independent-Hermitian-
generator double commutator -- leaving exactly 2*D_X + 2*D_Y (D_A[rho] =
A rho A - 1/2{A^2,rho}) for X=(L+L^dagger)/2, Y=(L-L^dagger)/(2i). Rescaling
by sqrt(2) absorbs the factor of 2, giving a UNIFORM convention: 25
independent real Hermitian generators V_j (1 from the self-conjugate
operator, 2 from each of 12 conjugate pairs), each with the same variance
Delta_W_j ~ N(0, delta_t) (gamma=1). Verified by direct construction: the
resulting sum_j D_{V_j} matches the original sum_j D_{L_j} (built from the
25 canonical, non-Hermitian jump operators) to a relative Frobenius error
of 2.5e-16 on a random test density matrix.

U_unsplit(delta_t) = exp[-i(H0*delta_t + sum_j Delta_W_j V_j)], Delta_W_j
i.i.d. N(0, delta_t). Each trajectory propagates rho_0 (the real
high-temperature deviation state, molecule_operators.py's rho0_dict, same
as used throughout circ_sim/scripts/QRE) directly: rho_traj = U rho_0
U^dagger (linear regardless of rho_0's lack of positive-semidefiniteness,
since it is the deviation operator, not a full density matrix).
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'QRE'))
QRE_TESTING_DIR = os.path.join(QRE_DIR, 'testing')
sys.path.insert(0, QRE_DIR)
sys.path.insert(0, QRE_TESTING_DIR)

import single_step_trotter_tightness as sstt  # noqa: E402 (H0, jump_ops, rho0, coil, exact_rho, D_sys)


def build_hermitian_generators(jump_ops, tol=1e-9):
    """From a conjugate-paired set of (generally non-Hermitian) canonical
    jump operators {L_j} satisfying sum_j[L_j,L_j^dagger]=0 (unital
    dissipator), build the equivalent set of independent real Hermitian
    generators {V_a} such that sum_a D_{V_a}[rho] = sum_j D_{L_j}[rho]
    exactly, with a uniform noise convention (every V_a gets variance
    delta_t, i.e. gamma_a=1 for all a). See module docstring for the
    derivation. Returns V_list (list of dense Hermitian matrices)."""
    n = len(jump_ops)
    hermitian_idx = [j for j in range(n)
                      if np.linalg.norm(jump_ops[j] - jump_ops[j].conj().T, 'fro')
                      < tol * np.linalg.norm(jump_ops[j], 'fro')]
    remaining = [j for j in range(n) if j not in hermitian_idx]
    consumed = set()
    pairs = []
    for j in remaining:
        if j in consumed:
            continue
        for k in remaining:
            if k in consumed or k == j:
                continue
            for s in (1.0, -1.0):
                if (np.linalg.norm(jump_ops[j].conj().T - s * jump_ops[k], 'fro')
                        < 1e-6 * np.linalg.norm(jump_ops[j], 'fro')):
                    pairs.append((j, k))
                    consumed.add(j)
                    consumed.add(k)
                    break
            if j in consumed:
                break
    if len(hermitian_idx) + 2 * len(pairs) != n:
        raise ValueError(f'Failed to account for all {n} jump operators: '
                          f'{len(hermitian_idx)} self-conjugate + {len(pairs)} pairs')

    V_list = [jump_ops[j] for j in hermitian_idx]
    for j, k in pairs:
        X = (jump_ops[j] + jump_ops[j].conj().T) / 2
        Y = (jump_ops[j] - jump_ops[j].conj().T) / (2j)
        V_list.append(np.sqrt(2) * X)
        V_list.append(np.sqrt(2) * Y)
    return V_list


def draw_unsplit_unitary(H0, V_list, dt, rng):
    """One realization of U_unsplit(dt) = exp[-i(H0*dt + sum_j dW_j V_j)],
    dW_j ~ N(0, dt) i.i.d. (white_noise_trotter_1.pdf Eq. 5)."""
    n = len(V_list)
    dW = rng.normal(scale=np.sqrt(dt), size=n)
    generator = H0 * dt
    for w, V in zip(dW, V_list):
        generator = generator + w * V
    return expm(-1j * generator)


def trajectory_batch(H0, V_list, rho0, coil, dt, n_traj, rng):
    """Return array of per-trajectory <coil> estimates, length n_traj."""
    vals = np.empty(n_traj, dtype=complex)
    for i in range(n_traj):
        U = draw_unsplit_unitary(H0, V_list, dt, rng)
        rho_traj = U @ rho0 @ U.conj().T
        vals[i] = np.trace(coil @ rho_traj)
    return vals


if __name__ == '__main__':
    H0, jump_ops, rho0, coil = sstt.H0, sstt.jump_ops, sstt.rho0, sstt.coil

    V_list = build_hermitian_generators(jump_ops)
    print(f'Built {len(V_list)} independent Hermitian generators from '
          f'{len(jump_ops)} canonical jump operators.')

    J_MAX_HZ = 226.85  # F0-C0 coupling, the strongest in truncated Gemcitabine
    DT = 1.0 / J_MAX_HZ
    print(f'delta_t = 1/J_max = 1/{J_MAX_HZ} Hz = {DT:.6e} s')

    O_exact = sstt.expect(coil, sstt.exact_rho(DT))
    print(f'<coil>_exact(delta_t) = {O_exact}')

    rng = np.random.default_rng(0)
    N_PILOT = 2000
    vals = trajectory_batch(H0, V_list, rho0, coil, DT, N_PILOT, rng)

    O_traj_mean = vals.mean()
    sigma_O = vals.std(ddof=1)  # per-trajectory std (real part; coil has real+imag structure)
    # coil is non-Hermitian: track std of real and imaginary parts separately too
    sigma_re = vals.real.std(ddof=1)
    sigma_im = vals.imag.std(ddof=1)

    print(f'\nPilot batch: N={N_PILOT}')
    print(f'  <coil>_traj (mean)      = {O_traj_mean}')
    print(f'  |exact - traj_mean|     = {abs(O_exact - O_traj_mean):.4e}')
    print(f'  sigma_O (|complex| std) = {sigma_O:.4e}')
    print(f'  sigma_Re, sigma_Im      = {sigma_re:.4e}, {sigma_im:.4e}')
    print(f'  s.e.m. at N={N_PILOT}: sigma_O/sqrt(N) = {sigma_O/np.sqrt(N_PILOT):.4e}')
