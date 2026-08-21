# Spinach ZULF cross-verification benchmark

Independent cross-check of the ZULF FID predictions in
`circ_sim/scripts/linblad_dyn/data/fluo_suite/` using Spinach's built-in
Bloch-Redfield relaxation theory instead of this codebase's explicit
extreme-narrowing Lindblad jump operators -- same molecules, same atom
subsets, same `tau_c`/field, genuinely different simulation method.

Scope: 7 systems (the smallest in the fluo_suite manifest, so exact/dense
methods stay tractable) x 3 IK-0 basis-truncation levels each = 21 runs.

| System | Drug | Cutoff | Spins |
|---|---|---|---|
| `mol5a_3.0A` | Belzutifan | 3.0 Å | 9 |
| `mol5b_3.0A` | Belzutifan | 3.0 Å | 9 |
| `mol5c_3.0A` | Belzutifan | 3.0 Å | 9 |
| `mol6_3.0A` | Pexidartinib | 3.0 Å | 9 |
| `mol13_3.0A` | Tedizolid phosphate | 3.0 Å | 9 |
| `mol18a_2.5A` | Letermovir | 2.5 Å | 10 |
| `mol18b_2.5A` | Letermovir | 2.5 Å | 11 |

These are each molecule's Table-2 ("Filtered cutoff selection") *second*
cutoff, not the default 5.0 Å -- the smaller, tighter-cutoff atom subset.
`bas.level` (Spinach's IK-0 restricted-Liouville-space truncation order)
is swept over 5, 6, 7 for each.

## Pipeline

```
export_spinach_inputs.py  -->  inputs/<label>_spinach_input.mat  -->  run_spinach_zulf.m  -->  ../data/<label>_bas<level>_fid.mat
```

1. **`export_spinach_inputs.py`** (Python, run once -- already done in this
   checkout, re-run only if the source workbook or manifest changes).
   Pulls the exact atom subset for each of the 7 systems from
   `fluo_suite_operators.list_systems()` (the same manifest that produced
   the existing `*_operators.pkl` reference files, so the atoms are
   identical, not re-derived), then re-reads raw coordinates, full 3x3
   GIAO shielding tensors (ppm), and 1-bond/2-bond J-couplings (Hz) for
   just those atoms from `ORCA_NMR_summary_merged.xlsx`, and writes one
   `.mat` per system to `inputs/`.

2. **`run_spinach_zulf.m`** (MATLAB + Spinach, one call per system x level).
   Loads one `inputs/*.mat`, builds `sys`/`inter` (isotopes, coordinates,
   Zeeman shielding tensors, scalar couplings), sets
   `bas.approximation='IK-0'`, `bas.level=<level>`,
   `inter.relaxation={'redfield'}`, `inter.tau_c={1e-10}`,
   `sys.magnet=5e-7` T -- matching the fixed physical parameters in
   `fluo_suite_operators.py` -- runs the same sudden-transfer + hard-pulse
   ZULF single-pulse protocol as the two Spinach scripts already in this
   repo (`circ_sim/data/big_fluo_mols/gemcitabine_5spin_ZULF_13C.m`,
   `fluticasone_ZULF_13C_single.m`), and saves FID + spectrum + a
   `metadata` struct (which molecule, which cutoff, which basis level,
   `tau_c`, field, isotopes, acquisition settings) to `../data/`.

3. **`submit_spinach_bench_array.sh`** (SLURM array, 21 tasks: 7 systems x
   3 levels). Calls `run_spinach_zulf.m` once per task.

## Before submitting

- `inputs/*.mat` must already exist (step 1) -- they do in this checkout.
- Set `SPINACH_PATH` (Spinach toolbox root: the directory containing
  `interfaces/`, `kernel/`, `experiments/`, `etc/`) and `MATLAB_MODULE`
  (the `module load` name on this cluster) as env vars when submitting:

  ```bash
  SPINACH_PATH=/path/to/spinach MATLAB_MODULE=matlab/R2022b sbatch submit_spinach_bench_array.sh
  ```

  Run `sbatch` from *inside* this directory (`cluster_calcs/`) -- the
  `#SBATCH --output`/`--error` paths are relative to wherever `sbatch` is
  invoked, since SLURM resolves them before the script body's own `cd`
  runs.

- **`run_spinach_zulf.m` has not been run yet** -- it closely follows the
  two Spinach ZULF scripts already validated in this repo, but there was
  no MATLAB/Spinach available to test it while writing it. Worth a single
  interactive/foreground run of one task before submitting the full array,
  e.g. (run from inside `cluster_calcs/`, and note the explicit `addpath`
  for this directory itself -- MATLAB's "current folder is implicitly on
  the path" behavior isn't reliable under every cluster's `matlab`
  wrapper/module, so don't drop it):

  ```bash
  matlab -batch "addpath(pwd); addpath(genpath('/path/to/spinach')); run_spinach_zulf('mol13_3.0A', 5)"
  ```

- The `--time`/`--mem`/`--cpus-per-task` in `submit_spinach_bench_array.sh`
  are conservative guesses (9-11 spins is small, but IK-0 level 6-7 basis
  construction cost on this cluster is unbenchmarked) -- adjust after the
  first run if needed.

## Output

Each `../data/<label>_bas<level>_fid.mat` contains `fid_raw_real`,
`fid_raw_imag`, `spec_real`, `spec_imag`, `freq`, and a `metadata` struct
identifying the molecule, cutoff, basis level, and all physical/
acquisition parameters used -- load with `scipy.io.loadmat` on the Python
side to compare against the corresponding Lindblad FID.
