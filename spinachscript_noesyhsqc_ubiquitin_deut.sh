#!/bin/bash
#SBATCH -J spinach_checks
#SBATCH -p gpu
#SBATCH -n 32
#SBATCH --gres=gpu:2
#SBATCH --gpu-freq=high
#SBATCH --mem=498000
#SBATCH -t 6-23:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.out
#SBATCH -e /n/holyscratch01/jaffe_lab/Everyone/kis/std/spinach_checks_%A_%a.err

module load matlab/R2022a-fasrc01
matlab -batch "spinach_checks_cluster"