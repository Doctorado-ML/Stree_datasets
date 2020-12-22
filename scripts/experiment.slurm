#!/bin/bash

#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4096M   # memory per CPU core
#SBATCH -J "<experiment>-<data>-<model>-<kernel>"   # job name
#SBATCH --mail-user=ricardo.montanana@alu.uclm.es   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd <folder>
python experiment.py -H galgo -e <experiment> -m <model> -d <data> -S tanveer -k <kernel> -n 1