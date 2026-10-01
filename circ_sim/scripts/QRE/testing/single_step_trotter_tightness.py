"""
Single-Trotter-step numerical tightness check (qre.tex
Sec.~sec:trotter_tightness_check): compares the EXACT, combined (prf+trot+
nested+mixed, undifferentiated) single-step observable error

    epsilon(Dt) = |Tr{ coil * (V_Dt - Vtilde_trot,Dt)[rho_0] }|

against the Dt-solver's leading-order estimate Dt^2*|K_prf+K_mixed|
(qre.tex Eq. Mstep_estimate at M=1), at 5 Dt values log-spaced from
Dt* ~ 9.34e-5 (the baseline resource estimate's actual operating point,
qre.tex Sec. full_pipeline) to 1e-2 (~107x larger) -- to see whether the
leading-order estimate the Dt-solver relies on is actually tight at the
step size the baseline uses, or how much slack exists.

Both channels are built as DENSE matrices sourced directly from
molecule_operators.py's canonical Pauli-dict representation of H0, {L_j},
rho0, coil, coherent_dicts, jump_pauli_terms -- the SAME source of truth
coherent_trotter_step.py / nested_trotter_step.py / error_budget_solver.py
already use -- so this tests the actual circuit's mathematical channel,
not an independently-reconstructed stand-in. The dilation/nested-Trotter
embedding machinery (embed_coherent, embed_jump_exp_nested,
_embed_local_block, the Liouville-space exact reference) is adapted
directly from ../trotter_prf_vs_trot_gemcitabine5.py, with three changes:
(a) data sourced from molecule_operators.py instead of the raw
gemcitabine_trunc pipeline, (b) a single, FIXED jump-operator order
matching jump_pauli_terms' own list order (= nested_trotter_step.py's
unary-iteration leaf order 0..S_k-1) -- not the two random A/B orderings
used there to test ordering-insensitivity, since here we specifically want
the order our actual circuit uses, not a robustness check, and (c) a
single Trotter step (M=1) at each Dt, evaluated independently from rho_0
every time, rather than an M=1..10 accumulated sweep at fixed small Dt.

The dilation ancilla uses its natural (S_k+1)-level qudit dimension (26
for Gemcitabine), not the qubit encoding padded to 32=2^5 that
unary_iteration.py uses for the actual circuit -- already proven to
compile an identical payload (qre.tex Sec. unary_iteration), so this is a
difference of numerical convenience only. Likewise, each hop term's exact
exponential is built here via a direct matrix exponential of its own
local 2*D_sys block (embed_jump_exp_nested), not via the Rz-conjugation
gate decomposition nested_trotter_step.py uses -- already proven to be an
exact realization of the same operator (qre.tex Sec. unary_iteration's
bug-fix paragraph), so the two routes are guaranteed to agree; this script
does not re-verify that identity, only relies on it.
"""
import os
import pickle
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
QRE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, 'data')
sys.path.insert(0, QRE_DIR)

import error_budget_solver as ebs  # noqa: E402
import molecule_operators as mo  # noqa: E402

# ---------------------------------------------------------------------------
# Pauli-dict -> dense matrix. Standard (unscaled) Pauli matrices; spin-operator
# normalization is already baked into each dict's own coefficients, matching
# pauli_algebra.PauliAlgebra's convention (e.g. ix(k) = {'...X...': 0.5}).
# ---------------------------------------------------------------------------
_PAULI1 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(chars):
    M = _PAULI1[chars[0]]
    for c in chars[1:]:
        M = np.kron(M, _PAULI1[c])
    return M


def dict_to_dense(pauli_dict, n):
    D = 2 ** n
    M = np.zeros((D, D), dtype=complex)
    for pstr, coeff in pauli_dict.items():
        M += coeff * _pauli_string_matrix(pstr)
    return M


# ---------------------------------------------------------------------------
# Load canonical operators (same source of truth as the rest of the pipeline)
# ---------------------------------------------------------------------------
ops = mo.load_molecule_operators('gemcitabine5')
n = ops['n']
D_sys = 2 ** n

H0 = dict_to_dense(ops['H0_dict'], n)
jump_ops = [dict_to_dense(Ld, n) for Ld in ops['L_dicts']]
rho0 = dict_to_dense(ops['rho0_dict'], n)
coil = dict_to_dense(ops['coil_dict'], n)
coherent_pieces = [dict_to_dense(frag, n) for frag in ops['coherent_dicts']]
jump_pauli_terms = ops['jump_pauli_terms']  # list of list[(pauli_string, coeff)], fixed circuit order

n_jumps = len(jump_ops)
D_anc = n_jumps + 1  # natural qudit dilation (vacuum + one level per jump operator)
D_full = D_anc * D_sys
print(f'n={n}, D_sys={D_sys}, active jump operators={n_jumps}, D_anc={D_anc}, D_full={D_full}')

# ---------------------------------------------------------------------------
# Dilation/nested-Trotter embedding machinery (adapted from
# ../trotter_prf_vs_trot_gemcitabine5.py)
# ---------------------------------------------------------------------------
I_anc = np.eye(D_anc, dtype=complex)


def embed_coherent(h_piece, dt):
    """exp(-i*dt*(1_anc (x) h_piece)) = 1_anc (x) exp(-i*dt*h_piece), exact
    (each coherent_dicts fragment is a sum of mutually-commuting Pauli
    terms by construction, so exponentiating the fragment's own dense sum
    directly is identical to the circuit's sequential per-term product)."""
    return np.kron(I_anc, expm(-1j * dt * h_piece))


def _embed_local_block(local_U, j_index):
    """Embed a 2*D_sys-dim local-ancilla-pair unitary (basis {0, j}) into
    the full D_full-dim space, identity elsewhere."""
    U = np.eye(D_full, dtype=complex)
    idx0 = np.arange(0, D_sys)
    idxj = np.arange(j_index * D_sys, (j_index + 1) * D_sys)
    full_idx = np.concatenate([idx0, idxj])
    U[np.ix_(full_idx, full_idx)] = local_U
    return U


def embed_jump_exp_nested(pauli_terms, j_index, dt):
    """Nested-Trotter approximation to exp(-i*sqrt(dt)*V_j) (qre.tex Sec.
    nested_trot_leading): a first-order product over V_j's own Pauli terms
    c_{j,n} P_{j,n}, IN LIST ORDER (matching nested_trotter_step.py's
    apply_hop_term loop exactly), built within the local 2*D_sys-dim
    ancilla-pair block {0, j} and embedded into the full space once."""
    theta = np.sqrt(dt)
    small_U = np.eye(2 * D_sys, dtype=complex)
    for pstr, c_n in pauli_terms:
        P_n = _pauli_string_matrix(pstr)
        block = np.zeros((2 * D_sys, 2 * D_sys), dtype=complex)
        block[D_sys:2 * D_sys, 0:D_sys] = c_n * P_n
        block[0:D_sys, D_sys:2 * D_sys] = np.conj(c_n) * P_n
        Un = expm(-1j * theta * block)
        small_U = Un @ small_U
    return _embed_local_block(small_U, j_index)


def ptrace_anc(rho_full):
    out = np.zeros((D_sys, D_sys), dtype=complex)
    for a in range(D_anc):
        out += rho_full[a * D_sys:(a + 1) * D_sys, a * D_sys:(a + 1) * D_sys]
    return out


def make_trotter_step(dt):
    """One outer Trotter step matching the actual circuit's temporal
    order exactly (coherent_trotter_step.py then nested_trotter_step.py):
    coherent fragments (Zeeman, then edge-color groups, in
    coherent_dicts's own order) applied first, then every jump operator's
    own nested-Trotter gadget, in jump_pauli_terms's own (= circuit's
    unary-iteration leaf) order 0..n_jumps-1."""
    U = np.eye(D_full, dtype=complex)
    for h in coherent_pieces:
        U = embed_coherent(h, dt) @ U
    for jidx in range(n_jumps):
        U = embed_jump_exp_nested(jump_pauli_terms[jidx], jidx + 1, dt) @ U
    return U


def trotter_rho(dt):
    U = make_trotter_step(dt)
    rho_full = np.zeros((D_full, D_full), dtype=complex)
    rho_full[0:D_sys, 0:D_sys] = rho0
    return ptrace_anc(U @ rho_full @ U.conj().T)


# ---------------------------------------------------------------------------
# Exact Lindbladian reference (Liouville-space eigendecomposition, one-time
# cost; exact_rho(dt) is then cheap for every Dt in the sweep)
# ---------------------------------------------------------------------------
def lindblad_liouvillian(H, Ls):
    Dd = H.shape[0]
    Id = np.eye(Dd, dtype=complex)
    L_ham = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    L_diss = np.zeros((Dd ** 2, Dd ** 2), dtype=complex)
    for Lop in Ls:
        LdL = Lop.conj().T @ Lop
        L_diss += (np.kron(Lop.conj(), Lop)
                   - 0.5 * np.kron(Id, LdL)
                   - 0.5 * np.kron(LdL.T, Id))
    return L_ham + L_diss


L_liou = lindblad_liouvillian(H0, jump_ops)
evals, evecs = np.linalg.eig(L_liou)
evecs_inv = np.linalg.inv(evecs)
rho0_vec = rho0.flatten(order='F')
c0 = evecs_inv @ rho0_vec


def exact_rho(dt):
    vec = evecs @ (np.exp(evals * dt) * c0)
    return vec.reshape(D_sys, D_sys, order='F')


def expect(O, rho):
    return np.trace(O @ rho)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    K_prf, K_mixed = ebs.get_coefficients()
    K_sum = abs(K_prf + K_mixed)
    print(f'K_prf={K_prf}, K_mixed={K_mixed}, |K_prf+K_mixed|={K_sum:.4e}')

    DT_MIN, DT_MAX, N_POINTS = 9.336901e-05, 1e-2, 5  # DT_MIN = Dt* from optimal_error_budget_split.py
    Dt_values = np.geomspace(DT_MIN, DT_MAX, N_POINTS)
    print(f'\nDt sweep (s): {[f"{dt:.4e}" for dt in Dt_values]}')

    O_ex0 = expect(coil, rho0)
    print(f'|<coil>(0)| = {abs(O_ex0):.4f}\n')

    results = {'Dt_values': Dt_values, 'K_prf': K_prf, 'K_mixed': K_mixed, 'data': []}
    print(f'{"Dt (s)":>12}  {"eps_exact":>12}  {"eps_leading":>12}  {"ratio(lead/exact)":>18}')
    for Dt in Dt_values:
        rho_ex = exact_rho(Dt)
        rho_tr = trotter_rho(Dt)

        O_ex = expect(coil, rho_ex)
        O_tr = expect(coil, rho_tr)
        eps_exact = abs(O_ex - O_tr)
        eps_leading = Dt ** 2 * K_sum
        ratio = eps_leading / eps_exact if eps_exact > 0 else np.nan

        print(f'{Dt:12.4e}  {eps_exact:12.4e}  {eps_leading:12.4e}  {ratio:18.4f}')
        results['data'].append(dict(Dt=Dt, O_exact=O_ex, O_trot=O_tr,
                                     eps_exact=eps_exact, eps_leading=eps_leading, ratio=ratio))

    out_pkl = os.path.join(DATA_DIR, 'single_step_trotter_tightness_gemcitabine5.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eps_exact_arr = np.array([d['eps_exact'] for d in results['data']])
    eps_leading_arr = np.array([d['eps_leading'] for d in results['data']])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(Dt_values, eps_exact_arr, 'o-', color='C0', label=r'exact $\varepsilon(\Delta t)$')
    ax.plot(Dt_values, eps_leading_arr, 's--', color='C1',
            label=r'leading-order $\Delta t^2|K_{\rm prf}+K_{\rm mixed}|$')
    ax.axvline(DT_MIN, color='gray', linestyle=':', linewidth=1)
    ax.text(DT_MIN, ax.get_ylim()[0] if False else eps_exact_arr[0], r'  $\Delta t^*$',
            va='bottom', ha='left', fontsize=9, color='gray')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t$ (s)')
    ax.set_ylabel(r'single-step observable error')
    ax.set_title('Truncated Gemcitabine: exact vs. leading-order single-step\n'
                  'Trotter+prf error, $M=1$')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'single_step_trotter_tightness_gemcitabine5.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
