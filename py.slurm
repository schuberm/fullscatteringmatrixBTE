#!/bin/bash
#SBATCH --job-name=py
#SBATCH --output=py.out 
#SBATCH --error=py.err 
#SBATCH --partition=qsingle_fat
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=end
#SBATCH --mail-user=you@example.com

python readAf.py
