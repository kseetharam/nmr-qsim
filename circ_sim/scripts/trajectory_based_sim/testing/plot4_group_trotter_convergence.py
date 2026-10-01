"""
A finer-grained Trotterization of the per-trajectory unitary, as a follow-up
to plot3_lie_eq8_verification.py's check of white_noise_trotter_1.pdf's
Eq. (3)/(8). There, one trajectory's unitary split the isotropic (H0) and
anisotropic (noise, sum_j dW_j*V_j) parts, but still applied the anisotropic
part as ONE exact joint exponential exp(-i*sum_j dW_j*V_j). Here, that joint
exponential is itself Trotterized: its Pauli-term decomposition (pooled
across ALL 25 Hermitian generators V_j, not kept separate per generator) is
partitioned into groups of MUTUALLY COMMUTING Pauli strings (each group
fused into one exact exponential, zero error within a group), and the
resulting groups are applied SEQUENTIALLY (first-order Lie-Trotter product
across groups -- the new error source this script characterizes). One
trajectory's unitary is therefore

    U(dt) = exp(-i*H0*dt) @ exp(-i*G_{16}) @ ... @ exp(-i*G_2) @ exp(-i*G_1),

where G_g = sum_{P in group g} c_P^{(traj)} * P, c_P^{(traj)} = sum_j dW_j*c_{j,P}
is the trajectory's random aggregate coefficient of Pauli string P (summed
over every generator that has a term on P), and group membership is fixed
--precomputed once, reused for every trajectory and every dt.

Why cross-generator grouping is safe here (resolves an open question flagged
in QRE/traj_based/README.md's point 4): commutation between two Pauli
strings never depends on their coefficients, only on which qubits they act
on nontrivially and how. Since every V_j's own Pauli-term coefficients are
real (Hermitian generator decomposed in the real {I,X,Y,Z}^n basis), there
is no analogue of the ancilla-based Lindbladian pipeline's "hop term" phase
condition (nested_trotter_grouping.py's same_phase_mod_pi, needed there only
because non-Hermitian jump operators get encoded via a complex-coefficient
X/Y ancilla-rotation trick) -- so pure Pauli commutation is already the
exact fusion criterion, and the resulting grouping is identical for every
trajectory. Checked directly: 105 unique Pauli strings across the 25
generators collapse into 16 mutually-commuting groups (sizes 4-12) via
nested_trotter_grouping.group_jump_operator_terms's existing greedy
conflict-graph coloring, reused unchanged (its phase-mod-pi check is simply
always satisfied for real coefficients, so it behaves as pure-commutation
grouping here without modification).

The 25 Hermitian generators' Pauli-DICT decomposition (needed for grouping;
trajectory_convergence.build_hermitian_generators only builds them as dense
matrices) is derived independently here via a per-term Re/Im extraction
(for any operator L, (L+L^dagger)/2 and (L-L^dagger)/(2i) are, term by term,
exactly Re(c_P) and Im(c_P) of L's own Pauli-dict -- no dense matrices
needed), reusing the SAME conjugate-pairing detection logic as
build_hermitian_generators (still done on the dense jump_ops, since that
part is a Frobenius-norm comparison, not something Pauli-dicts simplify).
Verified before use: converting these Pauli dicts back to dense matrices
reproduces trajectory_convergence.py's already-trusted dense V_list exactly
(max elementwise difference ~2e-16, same generator order).

Reference: the FULLY exact channel (sstt.exact_rho(dt), dense Liouvillian
exponentiation -- no Lie/anisotropic splitting, no group-Trotter splitting,
no sampling at all), not the Lie-split reference from plot3. So the error
tracked here is the total of three stacked approximations: (i) the
isotropic/anisotropic Lie splitting itself (already characterized in
plot3), (ii) this script's new group-sequential Trotter error, and (iii)
finite-N statistical sampling -- as a function of N, for a sweep of
dt in [0.2, 10] x (1/J_max), 5 log-spaced points (J_max = 226.85 Hz,
truncated Gemcitabine's strongest coupling, |J_F0-C0|).

Vectorization: batched over trajectories within each group (batched
np.linalg.eigh + batched matmul, per plot3_lie_eq8_verification.py's
established pattern -- including its einsum-vs-matmul lesson: U@rho0@U^dag
is built via two batched matmuls, not a raw einsum contraction, which
earlier was found to pick a very slow contraction path). The 16 sequential
group exponentials make this ~16x the per-trajectory cost of plot3's single
joint exponential (~3.4 ms/trajectory here vs. ~0.2 ms there, measured
directly) -- still fast enough that N up to several thousand per dt runs in
seconds, no chunking needed at the trajectory counts used below.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
QRE_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'QRE'))
sys.path.insert(0, HERE)
sys.path.insert(0, QRE_DIR)

from trajectory_convergence import build_hermitian_generators, sstt  # noqa: E402
import molecule_operators as mo  # noqa: E402
import nested_trotter_grouping as ntg  # noqa: E402

H0, rho0, coil = sstt.H0, sstt.rho0, sstt.coil
D_sys = sstt.D_sys
J_MAX_HZ = 226.85
DT_UNIT = 1.0 / J_MAX_HZ


# ---------------------------------------------------------------------------
# Pauli-dict Hermitian generators (mirrors build_hermitian_generators' pairing
# logic exactly, but builds each V_j as a Pauli dict via per-term Re/Im
# extraction instead of a dense-matrix computation) + validation against the
# already-trusted dense V_list.
# ---------------------------------------------------------------------------

def build_hermitian_generators_pauli(jump_ops_dense, L_dicts, tol=1e-9):
    n = len(jump_ops_dense)
    hermitian_idx = [j for j in range(n)
                      if np.linalg.norm(jump_ops_dense[j] - jump_ops_dense[j].conj().T, 'fro')
                      < tol * np.linalg.norm(jump_ops_dense[j], 'fro')]
    remaining = [j for j in range(n) if j not in hermitian_idx]
    consumed, pairs = set(), []
    for j in remaining:
        if j in consumed:
            continue
        for k in remaining:
            if k in consumed or k == j:
                continue
            for s in (1.0, -1.0):
                if (np.linalg.norm(jump_ops_dense[j].conj().T - s * jump_ops_dense[k], 'fro')
                        < 1e-6 * np.linalg.norm(jump_ops_dense[j], 'fro')):
                    pairs.append((j, k))
                    consumed.add(j)
                    consumed.add(k)
                    break
            if j in consumed:
                break
    if len(hermitian_idx) + 2 * len(pairs) != n:
        raise ValueError('pairing failed to account for all jump operators')

    # .real is an explicit cast, not a no-op: hermitian_idx is only "Hermitian
    # up to tol", so coefficients are Python complex with an exactly-zero (but
    # still complex-typed) imaginary part -- assigning those into a float
    # array later raises TypeError on current numpy (no implicit narrowing).
    V_dicts = [{P: c.real for P, c in L_dicts[j].items()} for j in hermitian_idx]
    for j, _k in pairs:
        V_dicts.append({P: np.sqrt(2) * c.real for P, c in L_dicts[j].items()})
        V_dicts.append({P: np.sqrt(2) * c.imag for P, c in L_dicts[j].items()})
    return V_dicts


_PAULI1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_mat(s):
    M = _PAULI1[s[0]]
    for c in s[1:]:
        M = np.kron(M, _PAULI1[c])
    return M


def _dict_to_dense(d, n):
    D = 2 ** n
    M = np.zeros((D, D), dtype=complex)
    for p, c in d.items():
        M += c * _pauli_mat(p)
    return M


ops = mo.load_molecule_operators('gemcitabine5')
V_dicts = build_hermitian_generators_pauli(sstt.jump_ops, ops['L_dicts'])
N_GEN = len(V_dicts)

# ---- validate against trajectory_convergence's dense-matrix construction ----
_V_list_dense_trusted = build_hermitian_generators(sstt.jump_ops)
_V_dicts_dense = [_dict_to_dense(d, sstt.n) for d in V_dicts]
_max_diff = max(np.linalg.norm(a - b) for a, b in zip(_V_dicts_dense, _V_list_dense_trusted))
assert _max_diff < 1e-10, f'Pauli-dict generators disagree with trusted dense V_list: {_max_diff:.3e}'


# ---------------------------------------------------------------------------
# Mutually-commuting grouping of the pooled Pauli-term set (trajectory- and
# dt-independent: commutation never depends on coefficient values).
# ---------------------------------------------------------------------------

ALL_STRINGS = sorted({p for d in V_dicts for p in d})
_STR_INDEX = {s: i for i, s in enumerate(ALL_STRINGS)}
N_TERMS = len(ALL_STRINGS)

# C[p, j] = coefficient of unique Pauli string p in generator V_j (0 if absent)
COEF_MATRIX = np.zeros((N_TERMS, N_GEN))
for j, d in enumerate(V_dicts):
    for p, c in d.items():
        COEF_MATRIX[_STR_INDEX[p], j] = c

_terms_for_grouping = [(p, 1.0 + 0j) for p in ALL_STRINGS]  # coefficient irrelevant: pure commutation
GROUPS, _ = ntg.group_jump_operator_terms(_terms_for_grouping)
GROUP_TERM_INDICES = [[_STR_INDEX[p] for p, _ in g] for g in GROUPS]
_TERM_MATS = np.stack([_pauli_mat(s) for s in ALL_STRINGS], axis=0)  # (N_TERMS, D, D)
GROUP_MATS = [_TERM_MATS[idxs] for idxs in GROUP_TERM_INDICES]       # list of (|g|, D, D)


# ---------------------------------------------------------------------------
# Vectorized per-trajectory sampler: exp(-i*H0*dt) followed by a sequential,
# first-order Trotter product over the 16 mutually-commuting groups.
# ---------------------------------------------------------------------------

def batched_group_trotter_vals(dt, m, rng):
    dW = rng.normal(scale=np.sqrt(dt), size=(m, N_GEN))
    coefs = dW @ COEF_MATRIX.T  # (m, N_TERMS): aggregate coefficient per unique Pauli string

    evalsH, evecsH = np.linalg.eigh(H0 * dt)
    U_H = (evecsH * np.exp(-1j * evalsH)) @ evecsH.conj().T

    U_V = np.broadcast_to(np.eye(D_sys, dtype=complex), (m, D_sys, D_sys)).copy()
    for idxs, gm in zip(GROUP_TERM_INDICES, GROUP_MATS):
        w = coefs[:, idxs]                             # (m, |g|)
        G = np.einsum('mi,iab->mab', w, gm)             # (m, D, D) Hermitian
        evals, evecs = np.linalg.eigh(G)
        U_g = (evecs * np.exp(-1j * evals)[:, None, :]) @ np.conj(np.transpose(evecs, (0, 2, 1)))
        U_V = U_g @ U_V

    U = U_H[None, :, :] @ U_V
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    rho_traj = (U @ rho0) @ Udag                        # batched matmul, not einsum (see docstring)
    return np.einsum('ab,nba->n', coil, rho_traj)


def cumulative_means(vals):
    return np.cumsum(vals) / np.arange(1, len(vals) + 1)


if __name__ == '__main__':
    print(f'D_sys={D_sys}, n_gen={N_GEN}, n_unique_terms={N_TERMS}, n_groups={len(GROUPS)}')
    print(f'group sizes: {sorted((len(g) for g in GROUPS), reverse=True)}')
    print(f'Pauli-dict generators validated against dense V_list: max diff = {_max_diff:.3e}\n')

    DT_MULTIPLIERS = np.geomspace(0.2, 10.0, 5)
    N_SWEEP = [10, 30, 100, 300, 1000, 3000, 5000]
    N_MAX = max(N_SWEEP)
    SEED = 0
    rng = np.random.default_rng(SEED)

    results = {'seed': SEED, 'J_max_Hz': J_MAX_HZ, 'DT_unit': DT_UNIT,
               'dt_multipliers': DT_MULTIPLIERS.tolist(), 'N_sweep': N_SWEEP, 'series': []}

    for mult in DT_MULTIPLIERS:
        dt = mult * DT_UNIT
        O_exact = sstt.expect(coil, sstt.exact_rho(dt))
        vals = batched_group_trotter_vals(dt, N_MAX, rng)
        running = cumulative_means(vals)
        sigma_O = vals.std(ddof=1)

        print(f'dt = {mult:.3g} x (1/J_max) = {dt:.4e} s   O_exact = {O_exact:.6f}   sigma_O = {sigma_O:.4e}')
        print(f'  {"N":>6}  {"O_traj(N)":>26}  {"eps(N)=|exact-traj|":>20}  {"sigma_O/sqrt(N)":>16}')
        series = {'mult': mult, 'dt': dt, 'O_exact': O_exact, 'sigma_O': sigma_O, 'data': []}
        for N in N_SWEEP:
            O_traj_N = running[N - 1]
            eps_N = abs(O_exact - O_traj_N)
            sem_N = sigma_O / np.sqrt(N)
            print(f'  {N:6d}  {str(np.round(O_traj_N, 6)):>26}  {eps_N:20.6e}  {sem_N:16.6e}')
            series['data'].append(dict(N=N, O_traj=O_traj_N, eps=eps_N, sem=sem_N))
        results['series'].append(series)
        print()

    out_pkl = os.path.join(DATA_DIR, 'plot4_group_trotter_convergence.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results['series'])))
    Ns = np.array(N_SWEEP)
    for series, color in zip(results['series'], colors):
        eps_vals = np.array([d['eps'] for d in series['data']])
        sem_vals = np.array([d['sem'] for d in series['data']])
        label = rf"$\Delta t={series['mult']:.2g}/J_{{\max}}$"
        ax.plot(Ns, eps_vals, 'o-', color=color, label=label)
        ax.plot(Ns, sem_vals, ':', color=color, linewidth=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of trajectories $N$')
    ax.set_ylabel(r'$|\langle O\rangle_{\rm exact}-\langle O\rangle_{\rm group\,Trotter}(N)|$')
    ax.set_title('Group-Trotterized anisotropic part vs. fully exact channel\n'
                  '(solid: error vs. exact; dotted: $\\sigma_O/\\sqrt{N}$ statistical floor, same color)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot4_group_trotter_convergence.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
