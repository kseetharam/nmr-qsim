"""
Uniform per-molecule loader for the QRE pipeline (pipeline_plan.md open
question 6): turns "molecule X, atom subset" into the full Pauli-dict
operator bundle every downstream script (error_budget_solver.py,
pauli_error_coefficients.py, check_kmixed_locality_bound.py,
check_combined_trotter_scaling.py) needs, built entirely via
circ_sim/scripts/linblad_dyn/utils/pauli_algebra.py -- polynomial effort in
the number of spins, no dense matrices.

Two things this module adds on top of the already-existing, already-scalable
circ_sim/scripts/linblad_dyn/fluo_suite_operators.py (H0, coil, L1/L2 jump
operators, all symbolic):
  1. build_coherent_fragments(): splits H0 into the Zeeman-layer +
     edge-colored-Heisenberg-group fragments h_p the trot/mixed error
     formulas need (fluo_suite_operators.py only builds the unfragmented H0).
  2. build_zulf_protocol_operators(): rho0 and the two Hermitian quadrature
     pieces S_x^coll, S_y^coll (qre.tex sec:trot_leading, "Why coil is
     non-Hermitian..."), via the exact algebraic pi/2-y-pulse identity
     rho0 = sum_i w_i(cos(theta_i) I_{i,z} + sin(theta_i) I_{i,x}),
     theta_i = (pi/2) w_i -- no dense exponentials needed.

load_molecule_operators(mol_id, atom_indices) dispatches on mol_id:
'gemcitabine5' uses the hardcoded 10-atom truncation this project's QRE
notes have been validated against throughout (NOT the same atom selection as
fluo_suite's own mol11 entry for gemcitabine, which is the full 22-atom
NMR-active set -- reconciling those is a separate, not-yet-done exercise,
noted below); any other mol_id defers to
zulf_numerics/fluo_suite/moment_convergence.load_molecule +
fluo_suite_operators.load_shielding_tensors, the existing fluo_suite loader.

Validated: rebuilding gemcitabine5 through this interface and recomputing
K_prf, K_mixed (pauli_error_coefficients.py's functions, unmodified)
reproduces the already-cached, already-verified values to the same
precision as before (see check_molecule_operators_gemcitabine5.py).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LINBLAD_DYN = os.path.normpath(os.path.join(HERE, '..', 'linblad_dyn'))
_UTILS = os.path.join(_LINBLAD_DYN, 'utils')
_FLUO_SUITE = os.path.normpath(os.path.join(HERE, '..', 'zulf_numerics', 'fluo_suite'))
for _p in (_LINBLAD_DYN, _UTILS, _FLUO_SUITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pauli_algebra import PauliAlgebra  # noqa: E402
import fluo_suite_operators as fso  # noqa: E402

GAMMA = fso.GAMMA  # {'F':..., 'H':..., 'C':...}, rad/(s.T)


# ---------------------------------------------------------------------------
# Molecule raw-data loaders: (atoms, edges, coords_ang, sigma_full) in the
# exact shape fso.build_all_operators expects.
#   atoms      : list of (idx, element in {'H','C','F'})
#   edges      : dict {(i, j): J_hz}, i < j
#   coords_ang : dict {idx: (3,) ndarray}, Angstrom
#   sigma_full : dict {idx: (3,3) ndarray}, ppm
# ---------------------------------------------------------------------------

def _gemcitabine5_raw():
    """The 10-atom (2F, 7H, 1C) Gemcitabine truncation hardcoded throughout
    circ_sim/scripts/gemcitabine_trunc/ and circ_sim/scripts/QRE/ -- NOT the
    same atom set as fluo_suite's mol11 (its full 22-atom NMR-active set)."""
    elements = ['F', 'F', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C']
    atoms = list(enumerate(elements))

    j_entries = [
        (0, 1, 175.91), (0, 2, 0.00), (0, 3, 0.11), (0, 4, 0.05), (0, 5, 6.66),
        (0, 6, -0.32), (0, 7, 0.05), (0, 8, 11.59), (0, 9, -226.85),
        (1, 2, 0.09), (1, 3, -0.20), (1, 4, 0.28), (1, 5, 5.91), (1, 6, 0.08),
        (1, 7, -0.09), (1, 8, 0.48), (1, 9, -203.41),
        (2, 3, -3.66), (2, 4, 0.73), (2, 5, -0.14), (2, 6, -0.10), (2, 7, -0.11),
        (2, 8, 0.18), (2, 9, -0.13),
        (3, 4, 2.88), (3, 5, -0.12), (3, 6, -0.10), (3, 7, -0.10), (3, 8, 0.38), (3, 9, -0.10),
        (4, 5, 0.22), (4, 6, 0.02), (4, 7, -0.09), (4, 8, 2.12), (4, 9, 0.31),
        (5, 6, 0.38), (5, 7, -0.07), (5, 8, -0.37), (5, 9, -0.65),
        (6, 7, 2.67), (6, 8, -0.15), (6, 9, 0.02),
        (7, 8, -0.14), (7, 9, -0.08),
        (8, 9, -2.68),
    ]
    edges = {(i, j): v for i, j, v in j_entries}

    coords_list = [
        [0.1666, -1.2783, -1.3875], [0.9661, -2.5748, 0.3352],
        [4.2344, 1.2314, -0.5006], [2.7307, 1.8998, -1.1779],
        [2.8928, 0.2529, 1.3947], [0.4474, -0.6659, 1.8321],
        [-1.7462, -0.8907, 2.3837], [-4.1117, -0.4940, 1.8959],
        [2.4649, -0.4748, -1.5086], [0.9133, -1.2851, -0.2046],
    ]
    coords_ang = {i: np.array(c) for i, c in enumerate(coords_list)}

    sigma_list = [
        [[-130.0601, -14.9649, -63.1405], [-14.9649, -215.3146, 25.9348], [-63.1405, 25.9348, -160.2068]],
        [[-157.2094, -12.7347, -67.3755], [-12.7347, -235.8144, 5.7085], [-67.3755, 5.7085, -228.8689]],
        [[2.2651, -0.6354, 2.4859], [-0.6354, 2.7073, -2.7546], [2.4859, -2.7546, 6.9013]],
        [[7.0348, -2.4557, -1.2496], [-2.4557, 3.6480, -0.3257], [-1.2496, -0.3257, 1.9779]],
        [[1.5527, 0.6046, -2.3124], [0.6046, 5.3322, 0.1728], [-2.3124, 0.1728, 4.7300]],
        [[4.3528, -1.0174, 1.5080], [-1.0174, 5.8782, 1.7827], [1.5080, 1.7827, 3.5154]],
        [[3.1215, 0.1121, 1.7281], [0.1121, 9.5089, 0.5658], [1.7281, 0.5658, 7.4617]],
        [[2.5152, 0.8210, -0.8704], [0.8210, 6.6786, 1.2727], [-0.8704, 1.2727, 4.7920]],
        [[4.4701, 0.6055, 0.2616], [0.6055, 4.4715, 1.0493], [0.2616, 1.0493, 2.7238]],
        [[241.3681, -1.4291, -6.1424], [-1.4291, 245.7345, 3.0563], [-6.1424, 3.0563, 250.0215]],
    ]
    sigma_full = {i: np.array(s) for i, s in enumerate(sigma_list)}

    return atoms, edges, coords_ang, sigma_full


_RAW_LOADERS = {
    'gemcitabine5': _gemcitabine5_raw,
}

# Default atom_indices per mol_id, used by load_molecule_operators when the
# caller doesn't specify a subset explicitly. 'gemcitabine5' names the
# 5-spin (F0, F1, H2, H3, C0) truncation this project's QRE notes have been
# validated against throughout -- NOT the raw loader's full 10-atom set.
_DEFAULT_ATOM_INDICES = {
    'gemcitabine5': [0, 1, 4, 5, 9],
}


def load_molecule_raw(mol_id):
    """(atoms, edges, coords_ang, sigma_full) for mol_id, dispatching to the
    custom gemcitabine5 loader or the general fluo_suite workbook loader."""
    if mol_id in _RAW_LOADERS:
        return _RAW_LOADERS[mol_id]()
    from moment_convergence import load_molecule  # noqa: E402
    atoms, edges, _bond_order, coords_ang = load_molecule(mol_id)
    sigma_full = fso.load_shielding_tensors(mol_id)
    return atoms, edges, coords_ang, sigma_full


# ---------------------------------------------------------------------------
# Coherent-fragment splitting (Zeeman layer + edge-colored Heisenberg
# groups), generalizing trotter_prf_vs_trot_gemcitabine5.py's
# greedy_edge_coloring/coherent_pieces construction to arbitrary molecules.
# ---------------------------------------------------------------------------

def greedy_edge_coloring(edge_list, n):
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


def build_coherent_fragments(alg, edges, atom_indices):
    """[h_group_1, ..., h_group_K] (Heisenberg edge-color groups only -- the
    Zeeman layer is built separately by _zeeman_fragment, which needs sigma
    tensors this function doesn't take) as Pauli dicts (positions 0..n-1 in
    atom_indices order), plus per-fragment supports, ordered by decreasing
    summed |J| within each edge-color group (matching
    trotter_prf_vs_trot_gemcitabine5.py's convention)."""
    n = len(atom_indices)
    atom_pos = {idx: i for i, idx in enumerate(atom_indices)}
    subset = set(atom_indices)

    active_edges = [(i, j) for (i, j) in edges if i in subset and j in subset and abs(edges[(i, j)]) > 0]
    local_edges = [(atom_pos[i], atom_pos[j]) for (i, j) in active_edges]
    j_by_local_edge = {(atom_pos[i], atom_pos[j]): edges[(i, j)] for (i, j) in active_edges}

    color_of = greedy_edge_coloring(local_edges, n)
    n_colors = (max(color_of.values()) + 1) if color_of else 0
    groups = {c: [] for c in range(n_colors)}
    for e, c in color_of.items():
        groups[c].append(e)
    group_strength = {c: sum(abs(j_by_local_edge[e]) for e in es) for c, es in groups.items()}
    ordered_colors = sorted(groups.keys(), key=lambda c: -group_strength[c])

    fragments = []
    supports = []
    for c in ordered_colors:
        gd = {}
        supp = set()
        for (pi, pj) in groups[c]:
            j_hz = j_by_local_edge[(pi, pj)]
            coeff = 2 * np.pi * j_hz
            gd = alg.add(gd,
                         alg.scale(alg.product(alg.ix(pi), alg.ix(pj)), coeff),
                         alg.scale(alg.product(alg.iy(pi), alg.iy(pj)), coeff),
                         alg.scale(alg.product(alg.iz(pi), alg.iz(pj)), coeff))
            supp.add(pi)
            supp.add(pj)
        fragments.append(gd)
        supports.append(supp)
    return fragments, supports


def _zeeman_fragment(alg, atoms, atom_indices, sigma_full, b_vec):
    elements = dict(atoms)
    n = len(atom_indices)
    gammas = [GAMMA[elements[idx]] for idx in atom_indices]
    h_zeeman = {}
    for i, idx in enumerate(atom_indices):
        sigma_tilde = np.eye(3) + 1e-6 * sigma_full[idx]
        sigma_iso = np.trace(sigma_tilde) / 3.0
        c = -gammas[i] * sigma_iso
        h_zeeman = alg.add(h_zeeman, alg.scale(alg.iz(i), c * b_vec[2]))
        if abs(b_vec[0]) > 0 or abs(b_vec[1]) > 0:
            h_zeeman = alg.add(h_zeeman, alg.scale(alg.ix(i), c * b_vec[0]),
                               alg.scale(alg.iy(i), c * b_vec[1]))
    return h_zeeman, set(range(n))


# ---------------------------------------------------------------------------
# ZULF protocol operators (rho0, quadrature-split coil), via the exact
# algebraic pi/2-y-pulse identity (verified against dense rho0 to
# np.allclose precision when this was first derived).
# ---------------------------------------------------------------------------

def build_zulf_protocol_operators(alg, atoms, atom_indices):
    elements = dict(atoms)
    n = len(atom_indices)
    weights = [GAMMA[elements[idx]] / GAMMA['H'] for idx in atom_indices]

    rho0_dict, Sx_coll, Sy_coll, coil_dict = {}, {}, {}, {}
    for i, w in enumerate(weights):
        theta = (np.pi / 2) * w
        term = alg.add(alg.scale(alg.iz(i), np.cos(theta)), alg.scale(alg.ix(i), np.sin(theta)))
        rho0_dict = alg.add(rho0_dict, alg.scale(term, w))
        Sx_coll = alg.add(Sx_coll, alg.scale(alg.ix(i), w))
        Sy_coll = alg.add(Sy_coll, alg.scale(alg.iy(i), w))
        coil_dict = alg.add(coil_dict, alg.scale(alg.ip(i), w))
    return rho0_dict, Sx_coll, Sy_coll, coil_dict, weights


# ---------------------------------------------------------------------------
# Top-level uniform loader
# ---------------------------------------------------------------------------

def load_molecule_operators(mol_id, atom_indices=None, tau_c=1e-10, b_vec=(0.0, 0.0, 5e-7)):
    """Full Pauli-dict operator bundle for one molecule/atom-subset.

    Returns a dict: alg, n, atom_indices, H0_dict, coherent_dicts,
    coherent_supports, L_dicts (list, active rank-2 jump ops only),
    jump_pauli_terms (list of (pauli_string, coeff) lists, same), rho0_dict,
    coil_dict, Sx_coll_dict, Sy_coll_dict, weights.
    """
    atoms, edges, coords_ang, sigma_full = load_molecule_raw(mol_id)
    if atom_indices is None:
        atom_indices = _DEFAULT_ATOM_INDICES.get(mol_id, [idx for idx, _ in atoms])
    n = len(atom_indices)
    alg = PauliAlgebra(n)

    h0, sz_weighted, sy_op, coil, l1_ops, l2_ops, j_coverage = fso.build_all_operators(
        atoms, edges, coords_ang, sigma_full, atom_indices, tau_c=tau_c, b_vec=b_vec)

    heisenberg_fragments, heisenberg_supports = build_coherent_fragments(alg, edges, atom_indices)
    h_zeeman, zeeman_support = _zeeman_fragment(alg, atoms, atom_indices, sigma_full, b_vec)
    coherent_dicts = [h_zeeman] + heisenberg_fragments
    coherent_supports = [zeeman_support] + heisenberg_supports

    rho0_dict, Sx_coll, Sy_coll, coil_dict2, weights = build_zulf_protocol_operators(alg, atoms, atom_indices)

    L_dicts = [op for op in l2_ops.values() if op]
    # Sort key includes the Pauli string itself as a tie-break: M_{p,j}
    # (qre.tex sec:combined_trot_nested) is defined relative to a specific
    # "which Pauli term of this L_j compiles first" convention, so exact
    # K_mixed values are (benignly) sensitive to how equal-|c| ties are
    # broken -- any deterministic tie-break is an equally valid, if
    # numerically slightly different, circuit-ordering choice.
    jump_pauli_terms = [sorted(op.items(), key=lambda kv: (-abs(kv[1]), kv[0])) for op in L_dicts]

    return dict(
        alg=alg, n=n, atom_indices=atom_indices,
        H0_dict=h0, coherent_dicts=coherent_dicts, coherent_supports=coherent_supports,
        L_dicts=L_dicts, jump_pauli_terms=jump_pauli_terms,
        rho0_dict=rho0_dict, coil_dict=coil, Sx_coll_dict=Sx_coll, Sy_coll_dict=Sy_coll,
        weights=weights, j_coverage=j_coverage,
    )
