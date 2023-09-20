#!/bin/bash
#SBATCH -J mat_char
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64000
#SBATCH -t 3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/holyscratch01/jaffe_lab/Everyone/kis/std/mat_char_%A_%a.out
#SBATCH -e /n/holyscratch01/jaffe_lab/Everyone/kis/std/mat_char_%A_%a.err

module load matlab/R2022b-fasrc01
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -r "mat_characterization_cluster"