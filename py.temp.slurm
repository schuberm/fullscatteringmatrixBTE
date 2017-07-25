#!/bin/bash
#SBATCH --job-name=py
#SBATCH --output=py.P.QX.out 
#SBATCH --error=py.P.QX.err 
#SBATCH --partition=qsingle_fat,qinfband,qsingle_reg
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mail-type=end
#SBATCH --mail-user=you@example.com

python sstg.P.QX.py
