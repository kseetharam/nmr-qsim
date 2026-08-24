#!/bin/bash
# submit_spinach_bench_array_lvl456.sh
#
# Second-pass SLURM array job for the Spinach ZULF cross-verification
# benchmark: re-targets IK-0 basis-truncation levels 4, 5, 6 (instead of
# the original 5, 6, 7 -- see submit_spinach_bench_array.sh), and OMITS
# the (system, level) combinations that already completed in the first
# run: level 5 for the five 9-spin systems (mol5a/b/c_3.0A, mol6_3.0A,
# mol13_3.0A). Those succeeded; levels 6-7 for those systems, and every
# level for the two Letermovir systems (mol18a/b_2.5A, 10-11 spins), hit
# memory limits under the first script's --mem=16G.
#
# 16 tasks total:
#   mol5a_3.0A, mol5b_3.0A, mol5c_3.0A, mol6_3.0A, mol13_3.0A : levels 4, 6  (2 each = 10)
#   mol18a_2.5A, mol18b_2.5A                                  : levels 4, 5, 6 (3 each = 6)
#
# Before submitting:
#   1. inputs/*.mat already exist (produced once by export_spinach_inputs.py;
#      basis level doesn't affect the exported molecular parameters, so no
#      re-export needed).
#   2. Set SPINACH_PATH and MATLAB_MODULE as before.
#
# Usage (run sbatch FROM this directory -- see submit_spinach_bench_array.sh
# for why):
#   cd circ_sim/scripts/zulf_numerics/fluo_suite/spinach_bench/cluster_calcs
#   SPINACH_PATH=/path/to/spinach MATLAB_MODULE=matlab/R2022b sbatch submit_spinach_bench_array_lvl456.sh
#
# Memory/time below are bumped from the first script's (16G / 2h), which
# was not enough for level 6-7 even on the 9-spin systems -- still an
# UNBENCHMARKED guess, not a measured requirement. Override per-submission
# without editing this file via SLURM's own CLI flags, e.g.:
#   sbatch --mem=128G --time=08:00:00 submit_spinach_bench_array_lvl456.sh

#SBATCH --job-name=spinach_zulf_bench_l456
#SBATCH --array=0-15
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/spinach_bench_l456_%A_%a.out
#SBATCH --error=logs/spinach_bench_l456_%A_%a.err

set -euo pipefail

# Resolve to an ABSOLUTE path: don't rely on MATLAB's "current folder is
# implicitly on the path" behavior (see submit_spinach_bench_array.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# --- cluster-specific config: fill these in for your environment ---
SPINACH_PATH="${SPINACH_PATH:-/path/to/spinach}"
MATLAB_MODULE="${MATLAB_MODULE:-matlab}"

module load "$MATLAB_MODULE"

# --- 16-task manifest: explicit (label, level) pairs, NOT a full
#     7 x 3 cartesian product -- the omissions above are baked in here
#     rather than computed, so the manifest is self-documenting. ---
TASKS=(
    "mol5a_3.0A:4"  "mol5a_3.0A:6"
    "mol5b_3.0A:4"  "mol5b_3.0A:6"
    "mol5c_3.0A:4"  "mol5c_3.0A:6"
    "mol6_3.0A:4"   "mol6_3.0A:6"
    "mol13_3.0A:4"  "mol13_3.0A:6"
    "mol18a_2.5A:4" "mol18a_2.5A:5" "mol18a_2.5A:6"
    "mol18b_2.5A:4" "mol18b_2.5A:5" "mol18b_2.5A:6"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
MOL_LABEL="${TASK%%:*}"
BAS_LEVEL="${TASK##*:}"

echo "Task $SLURM_ARRAY_TASK_ID -> mol_label=$MOL_LABEL, bas_level=$BAS_LEVEL"

matlab -batch "addpath('$SCRIPT_DIR'); addpath(genpath('$SPINACH_PATH')); run_spinach_zulf('$MOL_LABEL', $BAS_LEVEL)"
