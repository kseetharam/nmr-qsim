#!/bin/bash
#SBATCH -J spinach_checks
#SBATCH -p shared
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=64000
#SBATCH -t 0-00:10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.out
#SBATCH -e /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.err

module load matlab/R2022a-fasrc01
matlab -batch "spinach_checks_cluster"