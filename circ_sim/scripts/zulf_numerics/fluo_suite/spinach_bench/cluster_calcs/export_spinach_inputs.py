"""
export_spinach_inputs.py

Exports raw molecular parameters (isotopes, coordinates, J-couplings,
full shielding tensors) for the 7 Spinach cross-verification systems --
mol5a, mol5b, mol5c, mol6, mol13, mol18a, mol18b, each at its "second
cutoff" connectivity from Table 2 of nuclei_selection_survey.tex -- as
MATLAB-loadable .mat files under ./inputs/, one per system.

Atom subsets are NOT re-derived here: TARGET_LABELS is matched against
fluo_suite_operators.list_systems(), the exact same manifest that already
produced circ_sim/scripts/linblad_dyn/data/fluo_suite/mol5a_3.0A_operators.pkl
etc., so the atoms going into the Spinach run are guaranteed identical to
the atoms behind the existing Lindblad-operator reference files.

Unlike fluo_suite_operators.py, this script does NOT build Lindblad jump
operators -- it exports the raw per-atom quantities (coordinates, full
3x3 GIAO shielding tensors in ppm, 1-bond/2-bond J-couplings in Hz) so
that run_spinach_zulf.m can independently build its own Hamiltonian and
Bloch-Redfield relaxation superoperator: a genuinely different simulation
method (Spinach/Redfield vs. this codebase's explicit extreme-narrowing
Lindblad jump operators), for cross-verification rather than a re-check
of the same derivation.

Physical parameters matched to fluo_suite_operators.py's fixed generation
pass: B_vec_T = (0, 0, 5e-7) T, tau_c = 1e-10 s.

Usage
-----
    python export_spinach_inputs.py
"""
import os
import sys

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
LINBLAD_DYN_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'linblad_dyn'))
sys.path.insert(0, LINBLAD_DYN_DIR)
import fluo_suite_operators as fso   # noqa: E402  (also wires up moment_convergence's own sys.path)

OUT_DIR = os.path.join(HERE, 'inputs')

# The 7 requested systems, each at its Table-2 "second cutoff" -- identical
# labels to the corresponding files already in
# circ_sim/scripts/linblad_dyn/data/fluo_suite/.
TARGET_LABELS = [
    'mol5a_3.0A', 'mol5b_3.0A', 'mol5c_3.0A',
    'mol6_3.0A', 'mol13_3.0A',
    'mol18a_2.5A', 'mol18b_2.5A',
]


def export_system(sys_rec):
    mol_id = sys_rec['mol_id']
    atoms, edges, bond_order, coords_ang = fso.load_molecule(mol_id)
    sigma_full = fso.load_shielding_tensors(mol_id)
    mw = fso.load_molecular_weight(mol_id)
    elements = dict(atoms)
    atom_indices = sys_rec['atom_indices']
    n = len(atom_indices)
    atom_pos = {idx: i for i, idx in enumerate(atom_indices)}
    subset = set(atom_indices)

    isotopes = [f"1{elements[idx]}" if elements[idx] == 'H' else
                (f"19{elements[idx]}" if elements[idx] == 'F' else f"13{elements[idx]}")
                for idx in atom_indices]

    coords = np.array([coords_ang[idx] for idx in atom_indices], dtype=float)

    sigma_ppm_3d = np.zeros((n, 3, 3))
    for i, idx in enumerate(atom_indices):
        sigma_ppm_3d[i] = sigma_full[idx]

    # Only 1-bond/2-bond pairs are populated (same limitation as H0 in the
    # Lindblad reference pkls -- see their 'j_coupling_coverage' metadata).
    # Only the upper triangle is written: Spinach's own convention (see
    # e.g. build_noesy_methanol.m in this repo) is to specify one entry
    # per unordered pair and let create() symmetrize, NOT to populate both
    # {i,j} and {j,i} with the full value (that would double-count).
    J_hz_upper = np.zeros((n, n))
    for (i, j), j_hz in edges.items():
        if i in subset and j in subset:
            pi, pj = atom_pos[i], atom_pos[j]
            lo, hi = min(pi, pj), max(pi, pj)
            J_hz_upper[lo, hi] = j_hz

    label = fso.system_label(sys_rec)
    out = {
        'mol_id': mol_id,
        'drug_name': sys_rec['name'],
        'molecular_weight_g_per_mol': mw if mw is not None else np.nan,
        'anchor_carbon': sys_rec['anchor_c'],
        'anchor_bonded_F': np.array(sys_rec['anchor_f'], dtype=int),
        'cutoff_ang': sys_rec['cutoff_ang'],
        'n_spins': n,
        'isotopes': np.array(isotopes, dtype=object),
        'atom_indices_source_workbook': np.array(atom_indices, dtype=int),
        'coords_ang': coords,
        'sigma_ppm_3d': sigma_ppm_3d,
        'J_hz_upper': J_hz_upper,
        'B_vec_T': np.array(fso.B_VEC_T, dtype=float),
        'tau_c_s': float(fso.TAU_C_S),
        'j_coupling_note': (
            "J_hz_upper populated ONLY from 1-bond/2-bond ORCA/PySCF data "
            "(same limitation as H0 in the Lindblad reference pkls); all "
            "other pairs are exactly 0."),
        'generated_by': 'circ_sim/scripts/zulf_numerics/fluo_suite/spinach_bench/cluster_calcs/export_spinach_inputs.py',
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{label}_spinach_input.mat')
    scipy.io.savemat(out_path, out, oned_as='row')
    return out_path, n


def main():
    systems = fso.list_systems()
    by_label = {fso.system_label(s): s for s in systems}
    missing = [lbl for lbl in TARGET_LABELS if lbl not in by_label]
    if missing:
        raise SystemExit(f"Target labels not found in manifest: {missing}")

    for label in TARGET_LABELS:
        sys_rec = by_label[label]
        path, n = export_system(sys_rec)
        print(f"{label:16s} ({sys_rec['name']:24s}) -> {path}  ({n} spins)")


if __name__ == '__main__':
    main()
