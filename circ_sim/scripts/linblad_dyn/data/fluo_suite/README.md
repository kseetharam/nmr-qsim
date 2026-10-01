# fluo_suite ZULF Lindbladian operator sets

62 `.pkl` files, one per (molecule, fluorinated anchor, dipolar cutoff)
system in Table 2 ("Filtered cutoff selection") of the nuclei-selection
survey. Each file contains everything needed to build and run a Lindblad
master-equation simulation of a ZULF (Zero/Ultra-Low Field) FID for that
system: the coherent Hamiltonian, 34 dissipative jump operators, and the
NMR-protocol operators (initial state generator, detection operator).

See **`nuclei_selection_survey.pdf`** (copied into this directory) for the
full atom-selection methodology — how nuclei are chosen per molecule, what
the anchor/cutoff choices mean physically, and why 62 systems (not 21, the
number of molecules) come out of it. Table 2 in that document is the exact
manifest these files implement.

## How these were generated

- Script: `circ_sim/scripts/linblad_dyn/fluo_suite_operators.py`
- Atom selection (which nuclei belong to each system): imported directly
  from `circ_sim/scripts/zulf_numerics/fluo_suite/moment_convergence.py`,
  so the systems here are guaranteed identical to the survey's Table 2 —
  not a re-derivation.
- Source data: `circ_sim/data/fluo_suite/ORCA_NMR_summary_merged.xlsx`
  (ORCA/PySCF-computed J-couplings, shielding tensors, atomic coordinates).
- Run on a cluster via `circ_sim/scripts/linblad_dyn/submit_fluo_suite_array.sh`,
  a SLURM array job (`--array=0-61`), one task per system. `logs/` in that
  same directory holds the per-task stdout/stderr from that run.
- Fixed for every system in this generation pass: `B_vec = (0, 0, 5e-7)` T,
  `tau_c = 1e-10` s (extreme-narrowing, dipolar + CSA relaxation).

## File naming

```
<mol_id><anchor letter>_<cutoff>A_operators.pkl
```

- `mol_id`: e.g. `mol11` (Gemcitabine). Numbering matches the survey.
- anchor letter (`a`, `b`, `c`, ...): present only for molecules with more
  than one chemically distinct fluorinated anchor (natural-abundance
  ¹³C means only one carbon is isotopically active at a time, so each
  anchor is a separate, mutually-exclusive scenario — see the survey PDF,
  §"Multiple fluorinated anchors"). Absent when a molecule has exactly one
  anchor.
- `cutoff`: dipolar-graph cutoff in Å used to decide which non-exchangeable
  protons were reachable from the seed (always 5.0 Å, plus a second,
  tighter cutoff where Table 2 identifies one).

## Full manifest (62 systems)

| File | Drug | Anchor C (bonded F) | Cutoff (Å) | Spins total (F/H/C) | 1J/2J coverage |
|---|---|---|---|---|---|
| `mol1_5.0A_operators.pkl` | Lorlatinib | C6 (F8) | 5.0 | 19 (1/17/1) | 7.6% |
| `mol1_3.0A_operators.pkl` | Lorlatinib | C6 (F8) | 3.0 | 15 (1/13/1) | 9.5% |
| `mol2a_5.0A_operators.pkl` | Voxilaprevir | C40 (F54,55) | 5.0 | 54 (4/49/1) | 2.5% |
| `mol2a_3.0A_operators.pkl` | Voxilaprevir | C40 (F54,55) | 3.0 | 48 (4/43/1) | 2.9% |
| `mol2b_5.0A_operators.pkl` | Voxilaprevir | C13 (F14,15) | 5.0 | 54 (4/49/1) | 2.5% |
| `mol2b_3.0A_operators.pkl` | Voxilaprevir | C13 (F14,15) | 3.0 | 48 (4/43/1) | 2.9% |
| `mol3a_5.0A_operators.pkl` | Glecaprevir | C18 (F19,20) | 5.0 | 48 (4/43/1) | 2.5% |
| `mol3a_3.0A_operators.pkl` | Glecaprevir | C18 (F19,20) | 3.0 | 44 (4/39/1) | 3.0% |
| `mol3b_5.0A_operators.pkl` | Glecaprevir | C37 (F38,39) | 5.0 | 48 (4/43/1) | 2.6% |
| `mol3b_3.0A_operators.pkl` | Glecaprevir | C37 (F38,39) | 3.0 | 44 (4/39/1) | 3.1% |
| `mol5a_5.0A_operators.pkl` | Belzutifan | C16 (F22) | 5.0 | 15 (3/11/1) | 7.6% |
| `mol5a_3.0A_operators.pkl` | Belzutifan | C16 (F22) | 3.0 | 9 (3/5/1) | 13.9% |
| `mol5b_5.0A_operators.pkl` | Belzutifan | C8 (F23) | 5.0 | 15 (3/11/1) | 8.6% |
| `mol5b_3.0A_operators.pkl` | Belzutifan | C8 (F23) | 3.0 | 9 (3/5/1) | 16.7% |
| `mol5c_5.0A_operators.pkl` | Belzutifan | C7 (F24) | 5.0 | 15 (3/11/1) | 9.5% |
| `mol5c_3.0A_operators.pkl` | Belzutifan | C7 (F24) | 3.0 | 9 (3/5/1) | 19.4% |
| `mol6_5.0A_operators.pkl` | Pexidartinib | C25 (F26,27,28) | 5.0 | 17 (3/13/1) | 5.9% |
| `mol6_3.0A_operators.pkl` | Pexidartinib | C25 (F26,27,28) | 3.0 | 9 (3/5/1) | 19.4% |
| `mol8a_5.0A_operators.pkl` | Ivosidenib | C29 (F31,32) | 5.0 | 25 (3/21/1) | 3.7% |
| `mol8a_3.0A_operators.pkl` | Ivosidenib | C29 (F31,32) | 3.0 | 23 (3/19/1) | 4.3% |
| `mol8b_5.0A_operators.pkl` | Ivosidenib | C11 (F15) | 5.0 | 25 (3/21/1) | 2.7% |
| `mol8b_3.0A_operators.pkl` | Ivosidenib | C11 (F15) | 3.0 | 23 (3/19/1) | 3.2% |
| `mol9a_5.0A_operators.pkl` | Tezacaftor | C33 (F35,36) | 5.0 | 27 (3/23/1) | 4.0% |
| `mol9a_3.0A_operators.pkl` | Tezacaftor | C33 (F35,36) | 3.0 | 19 (3/15/1) | 7.0% |
| `mol9b_5.0A_operators.pkl` | Tezacaftor | C10 (F19) | 5.0 | 27 (3/23/1) | 4.0% |
| `mol9b_3.0A_operators.pkl` | Tezacaftor | C10 (F19) | 3.0 | 19 (3/15/1) | 7.0% |
| `mol10_5.0A_operators.pkl` | Emtricitabine | C9 (F15) | 5.0 | 9 (1/7/1) | 11.1% |
| `mol11_5.0A_operators.pkl` | Gemcitabine | C9 (F16,17) | 5.0 | 10 (2/7/1) | 13.3% |
| `mol12_5.0A_operators.pkl` | Sofosbuvir | C15 (F27) | 5.0 | 28 (1/26/1) | 5.0% |
| `mol12_3.0A_operators.pkl` | Sofosbuvir | C15 (F27) | 3.0 | 21 (1/19/1) | 6.2% |
| `mol13_5.0A_operators.pkl` | Tedizolid phosphate | C13 (F30) | 5.0 | 13 (1/11/1) | 5.1% |
| `mol13_3.0A_operators.pkl` | Tedizolid phosphate | C13 (F30) | 3.0 | 9 (1/7/1) | 11.1% |
| `mol14a_5.0A_operators.pkl` | Cabotegravir | C23 (F26) | 5.0 | 18 (2/15/1) | 5.9% |
| `mol14b_5.0A_operators.pkl` | Cabotegravir | C21 (F27) | 5.0 | 18 (2/15/1) | 5.2% |
| `mol15_5.0A_operators.pkl` | Linezolid | C13 (F23) | 5.0 | 21 (1/19/1) | 5.2% |
| `mol15_3.0A_operators.pkl` | Linezolid | C13 (F23) | 3.0 | 18 (1/16/1) | 5.2% |
| `mol16_5.0A_operators.pkl` | Eravacycline | C9 (F39) | 5.0 | 26 (1/24/1) | 4.6% |
| `mol16_3.0A_operators.pkl` | Eravacycline | C9 (F39) | 3.0 | 16 (1/14/1) | 8.3% |
| `mol17a_5.0A_operators.pkl` | Atogepant | C9 (F10,11,12) | 5.0 | 28 (6/21/1) | 4.0% |
| `mol17a_3.0A_operators.pkl` | Atogepant | C9 (F10,11,12) | 3.0 | 19 (6/12/1) | 7.6% |
| `mol17b_5.0A_operators.pkl` | Atogepant | C38 (F41) | 5.0 | 28 (6/21/1) | 3.4% |
| `mol17b_3.0A_operators.pkl` | Atogepant | C38 (F41) | 3.0 | 19 (6/12/1) | 6.4% |
| `mol17c_5.0A_operators.pkl` | Atogepant | C39 (F40) | 5.0 | 28 (6/21/1) | 3.2% |
| `mol17c_3.0A_operators.pkl` | Atogepant | C39 (F40) | 3.0 | 19 (6/12/1) | 5.8% |
| `mol17d_5.0A_operators.pkl` | Atogepant | C35 (F42) | 5.0 | 28 (6/21/1) | 3.2% |
| `mol17d_3.0A_operators.pkl` | Atogepant | C35 (F42) | 3.0 | 19 (6/12/1) | 5.8% |
| `mol18a_5.0A_operators.pkl` | Letermovir | C8 (F9,10,11) | 5.0 | 32 (4/27/1) | 3.4% |
| `mol18a_2.5A_operators.pkl` | Letermovir | C8 (F9,10,11) | 2.5 | 10 (4/5/1) | 20.0% |
| `mol18b_5.0A_operators.pkl` | Letermovir | C16 (F20) | 5.0 | 32 (4/27/1) | 3.2% |
| `mol18b_2.5A_operators.pkl` | Letermovir | C16 (F20) | 2.5 | 11 (4/6/1) | 14.5% |
| `mol19_5.0A_operators.pkl` | Siponimod | C24 (F25,26,27) | 5.0 | 38 (3/34/1) | 3.1% |
| `mol19_3.0A_operators.pkl` | Siponimod | C24 (F25,26,27) | 3.0 | 20 (3/16/1) | 6.3% |
| `mol20a_5.0A_operators.pkl` | Fluticasone furoate | C22 (F23) | 5.0 | 32 (3/28/1) | 3.8% |
| `mol20a_2.5A_operators.pkl` | Fluticasone furoate | C22 (F23) | 2.5 | 28 (3/24/1) | 5.0% |
| `mol20b_5.0A_operators.pkl` | Fluticasone furoate | C14 (F34) | 5.0 | 32 (3/28/1) | 3.8% |
| `mol20b_2.5A_operators.pkl` | Fluticasone furoate | C14 (F34) | 2.5 | 28 (3/24/1) | 5.0% |
| `mol20c_5.0A_operators.pkl` | Fluticasone furoate | C6 (F36) | 5.0 | 32 (3/28/1) | 4.0% |
| `mol20c_2.5A_operators.pkl` | Fluticasone furoate | C6 (F36) | 2.5 | 28 (3/24/1) | 5.3% |
| `mol21a_5.0A_operators.pkl` | Fluocinolone acetonide | C4 (F30) | 5.0 | 31 (2/28/1) | 4.3% |
| `mol21b_5.0A_operators.pkl` | Fluocinolone acetonide | C20 (F29) | 5.0 | 31 (2/28/1) | 4.5% |
| `mol22a_5.0A_operators.pkl` | Dutasteride | C19 (F20,21,22) | 5.0 | 35 (6/28/1) | 3.5% |
| `mol22b_5.0A_operators.pkl` | Dutasteride | C23 (F24,25,26) | 5.0 | 35 (6/28/1) | 3.5% |

Spin-count column is total (¹⁹F/¹H/¹³C); every system has exactly one ¹³C
(the natural-abundance anchor for that row).

## What's inside each `.pkl`

A `dict` with these keys (`n` = number of spins in that system, from the
table above):

| Key | Meaning |
|---|---|
| `metadata` | dict — see below |
| `H0` | coherent Hamiltonian: isotropic J-coupling + isotropic chemical shift, rad/s |
| `Sz_weighted` | γ-weighted total Sz; the pre-pulse ("sudden-transfer") density matrix ρ_sud |
| `Sy_op` | γ-weighted total Sy; generator of the π/2 hard pulse |
| `coil` | γ-weighted total S+; the quadrature detection observable |
| `L1_(k,m)` | 9 operators, `k,m ∈ {-1,0,1}` — rank-1 CSA (antisymmetric shielding tensor) jump operators |
| `L_(k,m)` | 25 operators, `k,m ∈ {-2,...,2}` — rank-2 jump operators (symmetric CSA + dipolar) |

Every operator (`H0`, `Sz_weighted`, `Sy_op`, `coil`, and all 34 `L`/`L1`)
is a **Pauli-string dictionary**, not a dense matrix:

```
{ "IXZI...": coefficient, ... }
```

- Each key is an `n`-character string over `{I, X, Y, Z}`, one character
  per spin, ordered as `metadata['spin_order']`.
- The operator is `M = Σ_P c_P · (P[0] ⊗ P[1] ⊗ ... ⊗ P[n-1])`, with
  `c_P = Tr(M·P) / 2^n`.
- This scales with how sparse the Hamiltonian/dissipator actually is,
  rather than with `2^n` — necessary for the 54-spin systems, where a
  dense matrix would be `2^54 × 2^54`.
- To turn one into a dense matrix (only feasible up to ~12-14 spins) or to
  build sums/products of these dicts directly, use
  `circ_sim/scripts/linblad_dyn/utils/pauli_algebra.py`
  (`PauliAlgebra` class — same convention, has `product`/`add`/`scale`/
  `dagger`/`commutator` for the symbolic form, or convert each single-site
  character to its 2×2 matrix and Kronecker them together for the dense
  form).

### `metadata` fields

| Field | Meaning |
|---|---|
| `system` | human-readable description |
| `source_molecule_id`, `source_molecule_name` | e.g. `mol11`, `Gemcitabine` |
| `source_molecular_weight_g_per_mol` | for provenance only (not used — `tau_c` is fixed, see above) |
| `anchor_carbon`, `anchor_bonded_F` | which ¹³C is the natural-abundance anchor, and which F it's 1-bond coupled to |
| `n_fluorinated_anchors_in_molecule` | how many mutually-exclusive anchor versions this molecule has (this file is one of them) |
| `cutoff_ang` | dipolar-graph cutoff (Å) used to pick the non-exchangeable-proton subset |
| `atom_indices_source_workbook` | original atom indices in `ORCA_NMR_summary_merged.xlsx`, in `spin_order` order — use this to trace an operator's spin back to a specific atom in the source workbook |
| `spin_order` | list like `['19F','19F','1H',...,'13C']`, defines the Pauli-string character positions |
| `B_vec_T`, `tau_c_s` | fixed physical parameters for this generation pass (see above) |
| `pauli_convention` | the Pauli-string formula, spelled out in prose |
| `H0_units`, `L_units` | `rad/s` and `sqrt(rad/s)` respectively |
| `L_description` | which physical channel each `L1_`/`L_` block corresponds to |
| `nmr_protocol` | the ZULF pulse sequence equations (see below) |
| `j_coupling_coverage` | dict with `populated_pairs`/`total_pairs`/`fraction` — **known limitation**, see below |
| `generated_by` | path to the generating script |

## Physics: master equation and NMR protocol

Given `H0` and the 34 jump operators `{L_a} = {L1_(k,m)} ∪ {L_(k,m)}`,
the Lindblad master equation is:

```
dρ/dt = -i[H0, ρ] + Σ_a ( L_a ρ L_a† - ½{L_a† L_a, ρ} )
```

(all 9 + 25 = 34 operators contribute additively — L1 and L2 are not
alternatives, both are physically active whenever their tensor is
nonzero).

The ZULF sudden-transfer + hard-pulse protocol (from `metadata['nmr_protocol']`):

```
ρ_sud = Sz_weighted
ρ0    = exp(-i π/2 · Sy_op) · ρ_sud · exp(+i π/2 · Sy_op)     # π/2 pulse
FID(t) = Tr[ coil · ρ(t) ],    ρ(t) solves the master equation above with ρ(0) = ρ0
```

`Sz_weighted`, `Sy_op`, and `coil` all use γ-weights
`w_i = γ_i / γ_(1H)`, i.e. lower-γ nuclei (¹³C, then ¹H, then ¹⁹F is
actually the highest-γ of the three here) contribute proportionally less
per spin — already baked into the saved operators, no extra rescaling
needed downstream.

## Known limitation: J-coupling coverage

`H0`'s J-coupling term only includes pairs with a measured 1-bond/2-bond
J in the source workbook (`metadata['j_coupling_coverage']`). Every other
pair in a given subset defaults to `J = 0` — an unverified approximation,
not a computed/confirmed negligibility. `j_coupling_coverage['fraction']`
in each file's metadata reports how many of that system's `n(n-1)/2` pairs
this actually covers (2.5%–20% across the manifest — see table above); the
dipolar and CSA channels (`L_`/`L1_`) are unaffected by this limitation,
since they're computed from atomic coordinates and shielding tensors for
every pair/atom, not from the sparse J table.
