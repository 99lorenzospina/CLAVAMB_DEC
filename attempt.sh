#!/bin/bash
#SBATCH -J test
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 00:03:00
#SBATCH -n 1
#SBATCH -p allgroups

cd /nfsd/bcb/bcbg/spina/
source tesi_env/bin/activate
cd CLAVAMB/test

srun singularity exec --nv ../../my-python-app.sif python parsebam.py 