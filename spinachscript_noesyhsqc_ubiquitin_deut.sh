#!/bin/bash
#SBATCH -J spinach_checks
#SBATCH -p ultramem
#SBATCH -n 64
#SBATCH --mem=2000000
#SBATCH -t 3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.out
#SBATCH -e /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.err

module load matlab/R2022b-fasrc01
matlab -batch "spinach_checks_cluster"