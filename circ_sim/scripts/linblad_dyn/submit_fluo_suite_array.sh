#!/bin/bash
# submit_fluo_suite_array.sh
#
# SLURM array job template for generating ZULF jump operators for every
# system in nuclei_selection_survey.tex (Table 2), via fluo_suite_operators.py.
#
# The array bound (0-61) matches the current 62-system manifest. If the
# survey table or its underlying data change, re-run
#     python fluo_suite_operators.py --list | tail -1
# to get the new system count and update --array below accordingly.
#
# Usage:
#   sbatch submit_fluo_suite_array.sh
#
# Each array task builds exactly one system (~1 minute for the largest,
# 54-atom system; smaller systems are faster) and writes one .pkl file
# to circ_sim/scripts/linblad_dyn/data/fluo_suite/.

#SBATCH --job-name=fluo_jumpops
#SBATCH --array=0-61
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/fluo_jumpops_%A_%a.out
#SBATCH --error=logs/fluo_jumpops_%A_%a.err

set -euo pipefail
mkdir -p logs

# Point this at whichever Python environment has numpy/qutip/openpyxl
# installed (see the project's feedback_python_env memory for this repo's
# default interpreter).
PYTHON="${FLUO_SUITE_PYTHON:-python3}"

cd "$(dirname "${BASH_SOURCE[0]}")"

"$PYTHON" fluo_suite_operators.py --index "$SLURM_ARRAY_TASK_ID"
