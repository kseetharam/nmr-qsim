"""
Numerically-exact comparison of the "prf" (purification/dilation) and "trot"
(Trotter) contributions to the observable error delta<O(t)>, on the 5-spin
truncated Gemcitabine ZULF model, as derived in circ_sim/scripts/QRE/notes/qre.tex.

For each time step Dt in {1,3,5,7,9}*delta_t/50, delta_t = 1/(2*|J_F0-C0|)
(restricted to the small-Dt end of the original sweep, where the asymptotic
scaling of Sec. nested_trot_leading should be visible), and for two fixed
(random) orderings of the jump operators, this builds three channels acting
on the SAME system-only rho:

    exact Lindbladian   V(t)            = expm(t * L_liouville) applied once
    exact dilation      Vtilde_Dt^M     = M-fold composition of the CPTP map
                                          rho -> Tr_a{ e^{-i Dt Htilde} (|0><0| ox rho) e^{+i Dt Htilde} }
    Trotterized dilation Vtilde_trot,Dt^M = M-fold composition of the same map
                                          with e^{-i Dt Htilde} replaced by a
                                          first-order Lie-Trotter product

and reports, at each t = M*Dt (M = 1..10):
    delta<O(t)>_prf  = Tr{ O (V(t) - Vtilde_Dt^M) rho0 }
    delta<O(t)>_trot = Tr{ O (Vtilde_Dt^M - Vtilde_trot,Dt^M) rho0 }

O = coil = sum_n w_n S+_n (non-Hermitian ZULF quadrature-detection operator),
rho0 = the sudden-transfer + pi/2-pulse ZULF initial state.

Coherent-part Trotter ordering: a single fused Zeeman layer (all single-spin
terms commute exactly) is applied first, followed by Heisenberg-coupling
edge-color groups (greedy edge coloring of the J-coupling graph; edges within
a group are vertex-disjoint and so commute/fuse exactly), ordered by
decreasing summed |J| (rad/s) within each group.

Jump-operator part: the 25 active (nonzero-norm) rank-2 canonical jump
operators (Krylov order 0 -- rank-1 vanishes identically for this system) are
applied after the coherent part, in one of two fixed random orderings. Unlike
the earlier version of this script, each jump-operator gadget is now ITSELF a
nested-Trotter product over that V_j's own Pauli-string decomposition
(exp(-i sqrt(Dt) V_j) is no longer applied as a single exact exponential),
per qre.tex Sec. nested_trot_leading -- so "trot" here measures the error of
the fully Trotterized circuit (coherent splitting AND nested jump-operator
splitting combined), not just the coherent part. The inner Pauli-term order
within each L_j is fixed (sorted by decreasing |c_{j,n}|), independent of the
outer jump-operator ordering A/B.
"""
import os
import pickle

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
GEMC_DATA = os.path.normpath(os.path.join(HERE, '..', 'gemcitabine_trunc', 'data'))

GAMMA_RAD = {'19F': 251.81520e6, '1H': 267.52218e6, '13C': 67.28284e6}
SPIN_TYPES = ['19F', '19F', '1H', '1H', '13C']
SPIN_LABELS = ['F0', 'F1', 'H2', 'H3', 'C0']

# Raw parameters needed to split H_iso into Zeeman + per-edge Heisenberg pieces
# (reproduced from gemcitabine_trunc/compute_5spin_jump_ops.py's 5-spin subset)
J_HZ_5 = np.zeros((5, 5))
_j_entries = {(0, 1): 175.91, (0, 2): 0.05, (0, 3): 6.66, (0, 4): -226.85,
              (1, 2): 0.28, (1, 3): 5.91, (1, 4): -203.41,
              (2, 3): 0.22, (2, 4): 0.31, (3, 4): -0.65}
for (i, j), v in _j_entries.items():
    J_HZ_5[i, j] = J_HZ_5[j, i] = v

SIGMA_5 = [
    np.array([[-130.0601, -14.9649, -63.1405], [-14.9649, -215.3146, 25.9348], [-63.1405, 25.9348, -160.2068]]),
    np.array([[-157.2094, -12.7347, -67.3755], [-12.7347, -235.8144, 5.7085], [-67.3755, 5.7085, -228.8689]]),
    np.array([[1.5527, 0.6046, -2.3124], [0.6046, 5.3322, 0.1728], [-2.3124, 0.1728, 4.7300]]),
    np.array([[4.3528, -1.0174, 1.5080], [-1.0174, 5.8782, 1.7827], [1.5080, 1.7827, 3.5154]]),
    np.array([[241.3681, -1.4291, -6.1424], [-1.4291, 245.7345, 3.0563], [-6.1424, 3.0563, 250.0215]]),
]
B_VEC = np.array([0.0, 0.0, 5e-7])
J_F0C0_HZ = 226.85  # |J|, sets delta_t

n = 5
D_sys = 2 ** n


# ---------------------------------------------------------------------------
# Single-spin operators (no qutip dependency; plain dense numpy)
# ---------------------------------------------------------------------------
def _embed(op2, k):
    mats = [np.eye(2)] * n
    mats[k] = op2
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


_sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
_sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
_sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

Ix = [_embed(_sx, k) for k in range(n)]
Iy = [_embed(_sy, k) for k in range(n)]
Iz = [_embed(_sz, k) for k in range(n)]
Ip = [Ix[k] + 1j * Iy[k] for k in range(n)]

gammas = np.array([GAMMA_RAD[s] for s in SPIN_TYPES])
weights = gammas / GAMMA_RAD['1H']
sigma_tilde = [np.eye(3) + 1e-6 * s for s in SIGMA_5]
sigma_iso = np.array([np.trace(st) / 3.0 for st in sigma_tilde])

# --- coherent-Hamiltonian pieces ---
h_zeeman = np.zeros((D_sys, D_sys), dtype=complex)
for i in range(n):
    h_zeeman += -gammas[i] * sigma_iso[i] * B_VEC[2] * Iz[i]

edges = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J_HZ_5[i, j]) > 0]
h_edge = {}
for (i, j) in edges:
    h_edge[(i, j)] = 2 * np.pi * J_HZ_5[i, j] * (Ix[i] @ Ix[j] + Iy[i] @ Iy[j] + Iz[i] @ Iz[j])

# --- greedy edge coloring of the J-coupling graph ---
def greedy_edge_coloring(edge_list):
    color_of = {}
    adj_colors = {v: set() for v in range(n)}
    for (i, j) in edge_list:
        used = adj_colors[i] | adj_colors[j]
        c = 0
        while c in used:
            c += 1
        color_of[(i, j)] = c
        adj_colors[i].add(c)
        adj_colors[j].add(c)
    return color_of


color_of = greedy_edge_coloring(edges)
n_colors = max(color_of.values()) + 1
groups = {c: [] for c in range(n_colors)}
for e, c in color_of.items():
    groups[c].append(e)

# order color groups by descending summed |J| (rad/s)
group_strength = {c: sum(abs(2 * np.pi * J_HZ_5[i, j]) for (i, j) in es) for c, es in groups.items()}
ordered_colors = sorted(groups.keys(), key=lambda c: -group_strength[c])

coherent_pieces = [h_zeeman] + [
    sum((h_edge[e] for e in groups[c]), np.zeros((D_sys, D_sys), dtype=complex))
    for c in ordered_colors
]
print("Coherent Trotter ordering (edge coloring, strongest-J-first):")
print(f"  1 Zeeman layer, then {n_colors} Heisenberg color groups")
for c in ordered_colors:
    print(f"    group (sum|J|={group_strength[c]/2/np.pi:.2f} Hz): {groups[c]}")

# --- ZULF protocol operators ---
rho_sud = sum(weights[k] * Iz[k] for k in range(n))
Sy_op = sum(weights[k] * Iy[k] for k in range(n))
coil = sum(weights[k] * Ip[k] for k in range(n))
U_pulse = expm(-1j * np.pi / 2 * Sy_op)
rho0 = U_pulse @ rho_sud @ U_pulse.conj().T

# ---------------------------------------------------------------------------
# Load jump operators (Krylov order 0) and keep only the active (nonzero) ones
# ---------------------------------------------------------------------------
with open(os.path.join(GEMC_DATA, 'gemcitabine_5spin_krylov_ops.pkl'), 'rb') as fh:
    ops_data = pickle.load(fh)

H_iso = np.asarray(ops_data['H_iso'], dtype=complex)
L2_0 = ops_data['L2_by_order'][0]
jump_ops = []
jump_labels = []
for km, mat in L2_0.items():
    mat = np.asarray(mat, dtype=complex)
    if np.linalg.norm(mat, 'fro') > 1e-8:
        jump_ops.append(mat)
        jump_labels.append(('L2', km))
n_jumps = len(jump_ops)
D_anc = n_jumps + 1
D_full = D_anc * D_sys
print(f"\nActive jump operators: {n_jumps}  ->  D_anc={D_anc}, D_sys={D_sys}, D_full={D_full}")

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(2024)
order_A = rng1.permutation(n_jumps)
order_B = rng2.permutation(n_jumps)
print("jump-operator ordering A:", order_A.tolist())
print("jump-operator ordering B:", order_B.tolist())

# ---------------------------------------------------------------------------
# Pauli decomposition of each jump operator: L_j = sum_n c_{j,n} P_{j,n}
# (needed for the nested-Trotter jump-operator gadget, qre.tex Sec. nested_trot_leading)
# ---------------------------------------------------------------------------
_PAULI1 = {'I': np.eye(2, dtype=complex), 'X': 2 * _sx, 'Y': 2 * _sy, 'Z': 2 * _sz}


def _pauli_string_matrix(chars):
    M = _PAULI1[chars[0]]
    for c in chars[1:]:
        M = np.kron(M, _PAULI1[c])
    return M


def pauli_decompose(A, thresh=1e-10):
    """{pauli_string: c_n} for A = sum_n c_n P_n, c_n = Tr(A P_n)/2^n."""
    from itertools import product
    terms = {}
    for chars in product('IXYZ', repeat=n):
        P = _pauli_string_matrix(chars)
        c = np.trace(A @ P) / D_sys
        if abs(c) > thresh:
            terms[''.join(chars)] = c
    return terms


print("\nDecomposing jump operators into Pauli strings (nested-Trotter gadget)...")
jump_pauli_terms = []   # list of dict {pauli_string: c_n}, one per jump_ops[jidx]
for Lj in jump_ops:
    terms = pauli_decompose(Lj)
    # deterministic, reproducible inner order: strongest terms first
    ordered = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
    jump_pauli_terms.append(ordered)
n_terms = [len(t) for t in jump_pauli_terms]
print(f"  Pauli terms per jump operator: min={min(n_terms)}, max={max(n_terms)}, "
      f"total={sum(n_terms)}")

# ---------------------------------------------------------------------------
# Embedding helpers (ancilla-outer convention: index = anc*D_sys + sys)
# ---------------------------------------------------------------------------
I_anc = np.eye(D_anc, dtype=complex)
I_sys = np.eye(D_sys, dtype=complex)


def embed_coherent(h_piece, dt):
    """exp(-i*dt*(1_anc (x) h_piece)) = 1_anc (x) exp(-i*dt*h_piece), exact."""
    return np.kron(I_anc, expm(-1j * dt * h_piece))


def embed_jump_exp(Lj, j_index, dt):
    """exp(-i*sqrt(dt)*V_j), V_j = |j><0| ox Lj + |0><j| ox Lj^dagger.
    Nontrivial only on ancilla indices {0, j}; identity elsewhere."""
    theta = np.sqrt(dt)
    block = np.zeros((2 * D_sys, 2 * D_sys), dtype=complex)
    # local ancilla basis: index 0 -> "0", index 1 -> "j"
    block[D_sys:2 * D_sys, 0:D_sys] = Lj              # |j><0| ox Lj
    block[0:D_sys, D_sys:2 * D_sys] = Lj.conj().T      # |0><j| ox Lj^dagger
    small_U = expm(-1j * theta * block)
    return _embed_local_block(small_U, j_index)


def _embed_local_block(local_U, j_index):
    """Embed a 2*D_sys-dim local-ancilla-pair unitary (basis {0, j}) into the
    full D_full-dim space, identity elsewhere."""
    U = np.eye(D_full, dtype=complex)
    idx0 = np.arange(0, D_sys)
    idxj = np.arange(j_index * D_sys, (j_index + 1) * D_sys)
    full_idx = np.concatenate([idx0, idxj])
    U[np.ix_(full_idx, full_idx)] = local_U
    return U


def embed_jump_exp_nested(pauli_terms, j_index, dt):
    """Nested-Trotter approximation to exp(-i*sqrt(dt)*V_j) (qre.tex
    Sec. nested_trot_leading): a first-order product over V_j's own Pauli
    terms, c_{j,n} P_{j,n}, built entirely within the local 2*D_sys-dim
    ancilla-pair block {0, j} and embedded into the full space only once."""
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


def build_Htilde_full(dt):
    Htilde = np.kron(I_anc, H_iso).astype(complex)
    for jidx, Lj in enumerate(jump_ops, start=1):
        e0 = np.zeros((D_anc, D_anc), dtype=complex); e0[jidx, 0] = 1.0
        ej0 = np.zeros((D_anc, D_anc), dtype=complex); ej0[0, jidx] = 1.0
        Htilde += (dt ** -0.5) * (np.kron(e0, Lj) + np.kron(ej0, Lj.conj().T))
    return Htilde


def ptrace_anc(rho_full):
    out = np.zeros((D_sys, D_sys), dtype=complex)
    for a in range(D_anc):
        out += rho_full[a * D_sys:(a + 1) * D_sys, a * D_sys:(a + 1) * D_sys]
    return out


def make_dilation_step(dt):
    """Exact dilation channel: rho -> Tr_a{ e^{-i dt Htilde} (|0><0| ox rho) e^{+i dt Htilde} }."""
    Htilde = build_Htilde_full(dt)
    U = expm(-1j * dt * Htilde)
    Ud = U.conj().T

    def step(rho_sys):
        rho_full = np.zeros((D_full, D_full), dtype=complex)
        rho_full[0:D_sys, 0:D_sys] = rho_sys
        return ptrace_anc(U @ rho_full @ Ud)
    return step


def make_trotter_step(dt, jump_order):
    """First-order Lie-Trotter: coherent pieces (Zeeman, then color groups,
    strongest-J-first), then jump-operator gadgets in the given order. Each
    jump-operator gadget is itself the nested-Trotter product over that V_j's
    own Pauli terms (qre.tex Sec. nested_trot_leading), not an exact atomic
    exponential of V_j."""
    factors = [embed_coherent(h, dt) for h in coherent_pieces]
    for jidx in jump_order:
        factors.append(embed_jump_exp_nested(jump_pauli_terms[jidx], jidx + 1, dt))
    U = np.eye(D_full, dtype=complex)
    for F in factors:
        U = F @ U
    Ud = U.conj().T

    def step(rho_sys):
        rho_full = np.zeros((D_full, D_full), dtype=complex)
        rho_full[0:D_sys, 0:D_sys] = rho_sys
        return ptrace_anc(U @ rho_full @ Ud)
    return step


# ---------------------------------------------------------------------------
# Exact Lindbladian reference (Liouville-space eigendecomposition)
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


L_liou = lindblad_liouvillian(H_iso, jump_ops)
evals, evecs = np.linalg.eig(L_liou)
evecs_inv = np.linalg.inv(evecs)
rho0_vec = rho0.flatten(order='F')  # column-major, matching lindblad_liouvillian's convention
c0 = evecs_inv @ rho0_vec


def exact_rho(t):
    vec = evecs @ (np.exp(evals * t) * c0)
    return vec.reshape(D_sys, D_sys, order='F')


def expect(O, rho):
    return np.trace(O @ rho)


delta_t = 1.0 / (2 * J_F0C0_HZ)
SMALL_FACTOR = 50
Dt_values = [k * delta_t / SMALL_FACTOR for k in (1, 3, 5, 7, 9)]   # up to 9*delta_t/50 only
M_steps = 10


def run_sweep():
    print(f"\ndelta_t = 1/(2*J_F0C0) = {delta_t*1e3:.4f} ms")
    print("Dt values (ms):", [f"{dt*1e3:.5f}" for dt in Dt_values])

    results = {'Dt_values': Dt_values, 'delta_t': delta_t, 'M_steps': M_steps,
               'small_factor': SMALL_FACTOR,
               'orders': {'A': order_A.tolist(), 'B': order_B.tolist()}, 'data': {}}

    for Dt in Dt_values:
        print(f"\n=== Dt = {Dt*1e3:.4f} ms ===")
        dil_step = make_dilation_step(Dt)
        trot_step_A = make_trotter_step(Dt, order_A)
        trot_step_B = make_trotter_step(Dt, order_B)

        rho_dil = rho0.copy()
        rho_trot_A = rho0.copy()
        rho_trot_B = rho0.copy()

        ts, prf_vals, trot_A_vals, trot_B_vals, signal_vals = [], [], [], [], []
        for M in range(1, M_steps + 1):
            rho_dil = dil_step(rho_dil)
            rho_trot_A = trot_step_A(rho_trot_A)
            rho_trot_B = trot_step_B(rho_trot_B)

            t = M * Dt
            rho_ex = exact_rho(t)

            O_ex = expect(coil, rho_ex)
            O_dil = expect(coil, rho_dil)
            O_trot_A = expect(coil, rho_trot_A)
            O_trot_B = expect(coil, rho_trot_B)

            prf = O_ex - O_dil
            trot_A = O_dil - O_trot_A
            trot_B = O_dil - O_trot_B

            ts.append(t)
            prf_vals.append(prf)
            trot_A_vals.append(trot_A)
            trot_B_vals.append(trot_B)
            signal_vals.append(O_ex)
            print(f"  M={M:2d}  t={t*1e3:9.5f} ms  |prf|/|O|={abs(prf)/abs(O_ex):.4e}  "
                  f"|trot_A|/|O|={abs(trot_A)/abs(O_ex):.4e}  "
                  f"|trot_B|/|O|={abs(trot_B)/abs(O_ex):.4e}  |signal|={abs(O_ex):.4e}")

        results['data'][Dt] = {
            't': np.array(ts), 'prf': np.array(prf_vals),
            'trot_A': np.array(trot_A_vals), 'trot_B': np.array(trot_B_vals),
            'signal': np.array(signal_vals),
        }
    return results


def make_plot(results):
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=min(Dt_values) / delta_t, vmax=max(Dt_values) / delta_t)

    for Dt in Dt_values:
        col = cmap(norm(Dt / delta_t))
        r = results['data'][Dt]
        t_ms = r['t'] * 1e3
        rel_prf = np.abs(r['prf']) / np.abs(r['signal'])
        rel_trot_A = np.abs(r['trot_A']) / np.abs(r['signal'])
        rel_trot_B = np.abs(r['trot_B']) / np.abs(r['signal'])
        label_dt = f"{Dt/delta_t:.3g}$\\delta_t$"
        ax.plot(t_ms, rel_prf, color=col, linestyle='-', marker='o', ms=4,
                 label=f"prf, $\\Delta t$={label_dt}")
        ax.plot(t_ms, rel_trot_A, color=col, linestyle='--', marker='^', ms=4,
                 label=f"trot (A), $\\Delta t$={label_dt}")
        ax.plot(t_ms, rel_trot_B, color=col, linestyle=':', marker='s', ms=4,
                 label=f"trot (B), $\\Delta t$={label_dt}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\Delta t / \delta_t$')

    ax.set_xlabel('t (ms)')
    ax.set_ylabel(r'$|\delta\langle \hat{O}(t)\rangle| / |\langle \hat{O}(t)\rangle|$  (relative error)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Gemcitabine 5-spin: relative prf vs trot observable error\n'
                 '(nested-Trotterized jump operators)')

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='gray', linestyle='-', marker='o', ms=4, label='prf'),
        Line2D([0], [0], color='gray', linestyle='--', marker='^', ms=4, label='trot (order A)'),
        Line2D([0], [0], color='gray', linestyle=':', marker='s', ms=4, label='trot (order B)'),
    ]
    ax.legend(handles=style_handles, fontsize=9, loc='upper left')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    results = run_sweep()
    with open(os.path.join(DATA_DIR, 'trotter_prf_vs_trot_gemcitabine5_nested.pkl'), 'wb') as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    fig = make_plot(results)
    out_png = os.path.join(DATA_DIR, 'trotter_prf_vs_trot_gemcitabine5_nested.png')
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot -> {out_png}")
