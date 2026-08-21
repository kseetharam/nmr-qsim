#!/bin/bash
# submit_spinach_bench_array.sh
#
# SLURM array job for the Spinach ZULF cross-verification benchmark:
# 7 fluo_suite systems (mol5a, mol5b, mol5c, mol6, mol13, mol18a, mol18b,
# each at its Table-2 "second cutoff") x 3 IK-0 basis-truncation levels
# (5, 6, 7) = 21 tasks, via run_spinach_zulf.m.
#
# Before submitting:
#   1. Run export_spinach_inputs.py once (locally or on a login node --
#      it only needs numpy/openpyxl/scipy, not MATLAB) to populate
#      ./inputs/*.mat. This has already been done in-repo; re-run only if
#      the source workbook or manifest changes.
#   2. Set SPINACH_PATH below (env var override) to this cluster's Spinach
#      toolbox root (the directory containing interfaces/, kernel/, etc.)
#      and MATLAB_MODULE to the module name for `module load`.
#
# Usage:
#   SPINACH_PATH=/path/to/spinach MATLAB_MODULE=matlab/R2022b sbatch submit_spinach_bench_array.sh
#
# Each array task builds one (molecule, basis-level) system (9-11 spins;
# expect this to be quick, but IK-0 level 6-7 restricted-space
# construction cost has not been benchmarked on this cluster -- the
# requested walltime below is a conservative guess, adjust after the
# first run) and writes one .mat file to
# circ_sim/scripts/zulf_numerics/fluo_suite/spinach_bench/data/.

#SBATCH --job-name=spinach_zulf_bench
#SBATCH --array=0-20
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/spinach_bench_%A_%a.out
#SBATCH --error=logs/spinach_bench_%A_%a.err

set -euo pipefail
mkdir -p logs
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- cluster-specific config: fill these in for your environment ---
SPINACH_PATH="${SPINACH_PATH:-/path/to/spinach}"
MATLAB_MODULE="${MATLAB_MODULE:-matlab}"

module load "$MATLAB_MODULE"

# --- 21-task manifest: 7 systems x 3 basis-truncation levels ---
MOL_LABELS=(mol5a_3.0A mol5b_3.0A mol5c_3.0A mol6_3.0A mol13_3.0A mol18a_2.5A mol18b_2.5A)
BAS_LEVELS=(5 6 7)

mol_idx=$(( SLURM_ARRAY_TASK_ID / ${#BAS_LEVELS[@]} ))
lvl_idx=$(( SLURM_ARRAY_TASK_ID % ${#BAS_LEVELS[@]} ))
MOL_LABEL="${MOL_LABELS[$mol_idx]}"
BAS_LEVEL="${BAS_LEVELS[$lvl_idx]}"

echo "Task $SLURM_ARRAY_TASK_ID -> mol_label=$MOL_LABEL, bas_level=$BAS_LEVEL"

matlab -batch "addpath(genpath('$SPINACH_PATH')); run_spinach_zulf('$MOL_LABEL', $BAS_LEVEL)"
